// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/hip/Exceptions.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>

#include "pytorch_extension_utils_rocm.h"
#include "sgl_kernel_ops.h"

#if SGL_W8A8_GFX928_MINIMAX_M3_MOE_M1_BUILD

namespace {

constexpr int kExperts = 129;
constexpr int kRoutes = 5;
constexpr int kHidden = 6144;
constexpr int kIntermediate = 384;
constexpr int kGateUp = 2 * kIntermediate;
constexpr int kThreads = 256;
constexpr int kWaveSize = 64;
constexpr int kOutputsPerBlock = kThreads / kWaveSize;

__device__ __forceinline__ int sdot4(int a, int b, int acc) {
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(a, b, acc, false);
#else
  const auto* a8 = reinterpret_cast<const int8_t*>(&a);
  const auto* b8 = reinterpret_cast<const int8_t*>(&b);
  return acc + static_cast<int>(a8[0]) * static_cast<int>(b8[0]) + static_cast<int>(a8[1]) * static_cast<int>(b8[1]) +
         static_cast<int>(a8[2]) * static_cast<int>(b8[2]) + static_cast<int>(a8[3]) * static_cast<int>(b8[3]);
#endif
}

__device__ __forceinline__ uint16_t float_to_bf16_rne(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
  return static_cast<uint16_t>(bits);
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  __builtin_memcpy(&result, &bits, sizeof(result));
  return result;
}

__device__ __forceinline__ float fp16_bits_to_float(uint16_t value) {
  return __half2float(*reinterpret_cast<const __half*>(&value));
}

__device__ __forceinline__ uint16_t float_to_fp16_rne(float value) {
  const __half result = __float2half_rn(value);
  uint16_t bits;
  __builtin_memcpy(&bits, &result, sizeof(bits));
  return bits;
}

template <bool kIsBf16>
__device__ __forceinline__ float value_to_float(uint16_t value) {
  return kIsBf16 ? bf16_bits_to_float(value) : fp16_bits_to_float(value);
}

template <bool kIsBf16>
__device__ __forceinline__ uint16_t value_from_float(float value) {
  return kIsBf16 ? float_to_bf16_rne(value) : float_to_fp16_rne(value);
}

__device__ __forceinline__ int quantize_one(float value, float reciprocal) {
  const float rounded = rintf(value * reciprocal);
  return static_cast<int>(fminf(127.0f, fmaxf(-127.0f, rounded)));
}

__device__ __forceinline__ uint32_t pack4_i8(int a, int b, int c, int d) {
  return static_cast<uint32_t>(static_cast<uint8_t>(a)) | (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

template <bool kIsBf16, int kSize>
__global__ void quantize_kernel(
    const uint16_t* __restrict__ input, int8_t* __restrict__ output, float* __restrict__ scale) {
  const int route = blockIdx.x;
  const uint16_t* row = input + route * kSize;
  int8_t* output_row = output + route * kSize;
  float maximum = 0.0f;
  for (int index = threadIdx.x; index < kSize; index += blockDim.x) {
    maximum = fmaxf(maximum, fabsf(value_to_float<kIsBf16>(row[index])));
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_down(maximum, offset));
  __shared__ float warp_max[kThreads / 64];
  if ((threadIdx.x & 63) == 0) warp_max[threadIdx.x / 64] = maximum;
  __syncthreads();
  if (threadIdx.x == 0) {
    maximum = warp_max[0];
#pragma unroll
    for (int warp = 1; warp < kThreads / 64; ++warp) maximum = fmaxf(maximum, warp_max[warp]);
    warp_max[0] = maximum;
  }
  __syncthreads();
  const float absmax = fmaxf(warp_max[0], 1e-10f);
  const float reciprocal = 127.0f / absmax;
  for (int index = threadIdx.x; index < kSize; index += blockDim.x) {
    output_row[index] = static_cast<int8_t>(quantize_one(value_to_float<kIsBf16>(row[index]), reciprocal));
  }
  if (threadIdx.x == 0) scale[route] = absmax / 127.0f;
}

__global__ void w13_kernel(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    const int32_t* __restrict__ topk_ids,
    uint16_t* __restrict__ gate_up,
    bool is_bf16) {
  const int lane = threadIdx.x & (kWaveSize - 1);
  const int output_column = blockIdx.x * kOutputsPerBlock + threadIdx.x / kWaveSize;
  const int route = blockIdx.y;
  const int expert = topk_ids[route];
  int accumulator = 0;
  if (output_column < kGateUp && expert >= 0 && expert < kExperts) {
    const int8_t* weight_row = weight + (static_cast<int64_t>(expert) * kGateUp + output_column) * kHidden;
    for (int index = lane * 4; index < kHidden; index += kWaveSize * 4) {
      int input_word;
      int weight_word;
      __builtin_memcpy(&input_word, input + index, sizeof(input_word));
      __builtin_memcpy(&weight_word, weight_row + index, sizeof(weight_word));
      accumulator = sdot4(input_word, weight_word, accumulator);
    }
  }
#pragma unroll
  for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) accumulator += __shfl_down(accumulator, offset);
  if (lane == 0 && output_column < kGateUp) {
    float result = 0.0f;
    if (expert >= 0 && expert < kExperts) {
      result = static_cast<float>(accumulator) * input_scale[0] *
               weight_scale[static_cast<int64_t>(expert) * kGateUp + output_column];
    }
    gate_up[route * kGateUp + output_column] =
        is_bf16 ? float_to_bf16_rne(result) : float_to_fp16_rne(result);
  }
}

template <bool kIsBf16>
__global__ void activate_kernel(
    const uint16_t* __restrict__ gate_up, uint16_t* __restrict__ activated, float alpha, float limit) {
  const int index = blockIdx.x * kThreads + threadIdx.x;
  const int route = blockIdx.y;
  if (index >= kIntermediate) return;
  float gate = value_to_float<kIsBf16>(gate_up[route * kGateUp + index]);
  float up = value_to_float<kIsBf16>(gate_up[route * kGateUp + kIntermediate + index]);
  gate = fminf(gate, limit);
  up = fminf(limit, fmaxf(-limit, up));
  const float result = gate * (1.0f / (1.0f + expf(-gate * alpha))) * (up + 1.0f);
  activated[route * kIntermediate + index] = value_from_float<kIsBf16>(result);
}

__global__ void w2_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ input_scale,
    const int8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    uint16_t* __restrict__ route_output,
    bool is_bf16) {
  const int lane = threadIdx.x & (kWaveSize - 1);
  const int output_column = blockIdx.x * kOutputsPerBlock + threadIdx.x / kWaveSize;
  const int route = blockIdx.y;
  const int expert = topk_ids[route];
  int accumulator = 0;
  if (output_column < kHidden && expert >= 0 && expert < kExperts) {
    const int8_t* input_row = input + route * kIntermediate;
    const int8_t* weight_row = weight + (static_cast<int64_t>(expert) * kHidden + output_column) * kIntermediate;
    for (int index = lane * 4; index < kIntermediate; index += kWaveSize * 4) {
      int input_word;
      int weight_word;
      __builtin_memcpy(&input_word, input_row + index, sizeof(input_word));
      __builtin_memcpy(&weight_word, weight_row + index, sizeof(weight_word));
      accumulator = sdot4(input_word, weight_word, accumulator);
    }
  }
#pragma unroll
  for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) accumulator += __shfl_down(accumulator, offset);
  if (lane == 0 && output_column < kHidden) {
    float result = 0.0f;
    if (expert >= 0 && expert < kExperts) {
      result = static_cast<float>(accumulator) * input_scale[route] *
               weight_scale[static_cast<int64_t>(expert) * kHidden + output_column] * topk_weights[route];
    }
    route_output[route * kHidden + output_column] =
        is_bf16 ? float_to_bf16_rne(result) : float_to_fp16_rne(result);
  }
}

template <bool kIsBf16>
__global__ void reduce_routes_kernel(const uint16_t* __restrict__ route_output, uint16_t* __restrict__ output) {
  const int index = blockIdx.x * kThreads + threadIdx.x;
  if (index >= kHidden) return;
  float accumulator = 0.0f;
#pragma unroll
  for (int route = 0; route < kRoutes; ++route) {
    accumulator += value_to_float<kIsBf16>(route_output[route * kHidden + index]);
  }
  output[index] = value_from_float<kIsBf16>(accumulator);
}

void check_tensor(const at::Tensor& tensor, at::ScalarType dtype, at::IntArrayRef shape, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.sizes() == shape, name, " has an invalid shape");
}

}  // namespace

at::Tensor w8a8_gfx928_minimax_m3_moe_m1(
    at::Tensor& x,
    const at::Tensor& w13_weight,
    const at::Tensor& w2_weight,
    const at::Tensor& w13_scale,
    const at::Tensor& w2_scale,
    const at::Tensor& topk_ids,
    const at::Tensor& topk_weights,
    at::Tensor& x_q,
    at::Tensor& x_scale,
    at::Tensor& gate_up,
    at::Tensor& activated,
    at::Tensor& activated_q,
    at::Tensor& activated_scale,
    at::Tensor& route_output,
    double alpha,
    double limit) {
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be float16 or bfloat16");
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
  check_tensor(x, x.scalar_type(), {1, kHidden}, "x");
  check_tensor(w13_weight, at::kChar, {kExperts, kGateUp, kHidden}, "w13_weight");
  check_tensor(w2_weight, at::kChar, {kExperts, kHidden, kIntermediate}, "w2_weight");
  check_tensor(w13_scale, at::kFloat, {kExperts, kGateUp, 1}, "w13_scale");
  check_tensor(w2_scale, at::kFloat, {kExperts, kHidden, 1}, "w2_scale");
  check_tensor(topk_ids, at::kInt, {1, kRoutes}, "topk_ids");
  check_tensor(topk_weights, at::kFloat, {1, kRoutes}, "topk_weights");
  check_tensor(x_q, at::kChar, {kHidden}, "x_q");
  check_tensor(x_scale, at::kFloat, {1}, "x_scale");
  check_tensor(gate_up, x.scalar_type(), {kRoutes, kGateUp}, "gate_up");
  check_tensor(activated, x.scalar_type(), {kRoutes, kIntermediate}, "activated");
  check_tensor(activated_q, at::kChar, {kRoutes, kIntermediate}, "activated_q");
  check_tensor(activated_scale, at::kFloat, {kRoutes}, "activated_scale");
  check_tensor(route_output, x.scalar_type(), {kRoutes, kHidden}, "route_output");
  TORCH_CHECK(
      x.device() == w13_weight.device() && x.device() == w2_weight.device() && x.device() == w13_scale.device() &&
          x.device() == w2_scale.device() && x.device() == topk_ids.device() && x.device() == topk_weights.device() &&
          x.device() == x_q.device() && x.device() == x_scale.device() && x.device() == gate_up.device() &&
          x.device() == activated.device() && x.device() == activated_q.device() && x.device() == activated_scale.device() &&
          x.device() == route_output.device(),
      "all tensors must be on the same device");

  const hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  const bool is_bf16 = x.scalar_type() == at::kBFloat16;
  if (is_bf16) {
    hipLaunchKernelGGL(
        (quantize_kernel<true, kHidden>), dim3(1), dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(x.data_ptr()), x_q.data_ptr<int8_t>(), x_scale.data_ptr<float>());
  } else {
    hipLaunchKernelGGL(
        (quantize_kernel<false, kHidden>), dim3(1), dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(x.data_ptr()), x_q.data_ptr<int8_t>(), x_scale.data_ptr<float>());
  }
  hipLaunchKernelGGL(
      w13_kernel, dim3(kGateUp / kOutputsPerBlock, kRoutes), dim3(kThreads), 0, stream, x_q.data_ptr<int8_t>(),
      w13_weight.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w13_scale.data_ptr<float>(),
      topk_ids.data_ptr<int32_t>(), reinterpret_cast<uint16_t*>(gate_up.data_ptr()), is_bf16);
  if (is_bf16) {
    hipLaunchKernelGGL(
        (activate_kernel<true>), dim3(2, kRoutes), dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(gate_up.data_ptr()), reinterpret_cast<uint16_t*>(activated.data_ptr()),
        static_cast<float>(alpha), static_cast<float>(limit));
    hipLaunchKernelGGL(
        (quantize_kernel<true, kIntermediate>), dim3(kRoutes), dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(activated.data_ptr()), activated_q.data_ptr<int8_t>(),
        activated_scale.data_ptr<float>());
  } else {
    hipLaunchKernelGGL(
        (activate_kernel<false>), dim3(2, kRoutes), dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(gate_up.data_ptr()), reinterpret_cast<uint16_t*>(activated.data_ptr()),
        static_cast<float>(alpha), static_cast<float>(limit));
    hipLaunchKernelGGL(
        (quantize_kernel<false, kIntermediate>), dim3(kRoutes), dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(activated.data_ptr()), activated_q.data_ptr<int8_t>(),
        activated_scale.data_ptr<float>());
  }
  hipLaunchKernelGGL(
      w2_kernel, dim3(kHidden / kOutputsPerBlock, kRoutes), dim3(kThreads), 0, stream,
      activated_q.data_ptr<int8_t>(), activated_scale.data_ptr<float>(), w2_weight.data_ptr<int8_t>(),
      w2_scale.data_ptr<float>(), topk_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(route_output.data_ptr()), is_bf16);
  if (is_bf16) {
    hipLaunchKernelGGL(
        (reduce_routes_kernel<true>), dim3(24), dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(route_output.data_ptr()), reinterpret_cast<uint16_t*>(x.data_ptr()));
  } else {
    hipLaunchKernelGGL(
        (reduce_routes_kernel<false>), dim3(24), dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(route_output.data_ptr()), reinterpret_cast<uint16_t*>(x.data_ptr()));
  }
  AT_CUDA_CHECK(hipGetLastError());
  return x;
}

#else

at::Tensor w8a8_gfx928_minimax_m3_moe_m1(
    at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    double,
    double) {
  TORCH_CHECK(false, "w8a8_gfx928_minimax_m3_moe_m1 is only available on gfx928");
}

#endif
