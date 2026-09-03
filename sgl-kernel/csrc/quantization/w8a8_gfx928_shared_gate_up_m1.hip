// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/MemoryOverlap.h>
#include <ATen/hip/Exceptions.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <optional>

#include "pytorch_extension_utils_rocm.h"
#include "sgl_kernel_ops.h"

#if SGL_W8A8_GFX928_SHARED_GATE_UP_M1_BUILD

namespace {

constexpr int kK = 6144;
constexpr int kN = 768;
constexpr int kK8 = kK / 8;
constexpr int kSplit = 64;
constexpr int kThreads = 256;
constexpr int kGridX = kN / kThreads;
constexpr int kK4PerSegment = (kK / 4) / kSplit;
constexpr int kUnroll = 12;
constexpr int kP1Win = 16;
constexpr int kQuantizeReductionSlots = kThreads / 32;
static_assert(kGridX == 3);
static_assert(kK4PerSegment == 24);
static_assert(static_cast<int64_t>(kK) * 127 * 128 < INT32_MAX);

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
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  const uint32_t exponent = (value >> 10) & 0x1fu;
  const uint32_t mantissa = value & 0x3ffu;
  uint32_t bits;
  if (exponent == 0u) {
    if (mantissa == 0u) {
      bits = sign;
    } else {
      uint32_t normalized = mantissa;
      int adjustment = -1;
      do {
        normalized <<= 1;
        --adjustment;
      } while ((normalized & 0x400u) == 0u);
      bits = sign | (static_cast<uint32_t>(adjustment + 114) << 23) | ((normalized & 0x3ffu) << 13);
    }
  } else if (exponent == 31u) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float result;
  __builtin_memcpy(&result, &bits, sizeof(result));
  return result;
}

__device__ __forceinline__ uint16_t float_to_fp16_rne(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
  const uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 128) {
    return static_cast<uint16_t>(sign | 0x7c00u | ((mantissa >> 13) & 0x3ffu) | ((mantissa != 0u) ? 0x200u : 0u));
  }
  if (exponent > 15) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exponent >= -14) {
    const uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    uint32_t half = sign | (static_cast<uint32_t>(exponent + 15) << 10) | half_mantissa;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
      ++half;
    }
    return static_cast<uint16_t>(half);
  }
  if (exponent < -25) return static_cast<uint16_t>(sign);
  const uint32_t shift = static_cast<uint32_t>(-(exponent + 1));
  const uint32_t full_mantissa = (1u << 23) | mantissa;
  uint32_t half_mantissa = full_mantissa >> shift;
  const uint32_t remainder = full_mantissa & ((1u << shift) - 1u);
  const uint32_t halfway = 1u << (shift - 1u);
  if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) {
    ++half_mantissa;
  }
  if (half_mantissa > 1023u) {
    return static_cast<uint16_t>(sign | 0x400u);
  }
  return static_cast<uint16_t>(sign | half_mantissa);
}

template <bool kIsBf16>
__device__ __forceinline__ float input_to_float(uint16_t value) {
  return kIsBf16 ? bf16_bits_to_float(value) : fp16_bits_to_float(value);
}

template <bool kIsBf16>
__device__ __forceinline__ uint16_t output_from_float(float value) {
  return kIsBf16 ? float_to_bf16_rne(value) : float_to_fp16_rne(value);
}

template <bool kIsBf16>
__device__ __forceinline__ uint16_t add_output_dtype(uint16_t lhs, uint16_t rhs) {
  return output_from_float<kIsBf16>(input_to_float<kIsBf16>(lhs) + input_to_float<kIsBf16>(rhs));
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

__device__ __forceinline__ uint32_t word_absmax4(uint64_t word) {
  const uint32_t low = static_cast<uint32_t>(word) & 0x7fff7fffu;
  const uint32_t high = static_cast<uint32_t>(word >> 32) & 0x7fff7fffu;
  uint32_t a = low & 0xffffu;
  uint32_t b = low >> 16;
  uint32_t c = high & 0xffffu;
  uint32_t d = high >> 16;
  a = (a > b) ? a : b;
  c = (c > d) ? c : d;
  return (a > c) ? a : c;
}

template <bool kIsBf16>
__device__ __forceinline__ float absmax_bits_to_float(uint32_t value) {
  if (kIsBf16) {
    const uint32_t bits = value << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
  }
  return fp16_bits_to_float(static_cast<uint16_t>(value));
}

template <bool kIsBf16>
__device__ __forceinline__ float full_input_max(const uint16_t* __restrict__ x, int tid) {
  const auto* x64 = reinterpret_cast<const uint64_t*>(x);
  constexpr int kWords = kK / 4;
  (void)kP1Win;
  uint32_t maxima[4] = {0, 0, 0, 0};
  int index = tid;
  for (; index + 3 * kThreads < kWords; index += 4 * kThreads) {
    uint64_t words[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      words[i] = x64[index + i * kThreads];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t value = word_absmax4(words[i]);
      maxima[i] = (value > maxima[i]) ? value : maxima[i];
    }
  }
  for (; index < kWords; index += kThreads) {
    const uint32_t value = word_absmax4(x64[index]);
    maxima[0] = (value > maxima[0]) ? value : maxima[0];
  }
  uint32_t maximum = (maxima[0] > maxima[1]) ? maxima[0] : maxima[1];
  maximum = (maximum > maxima[2]) ? maximum : maxima[2];
  maximum = (maximum > maxima[3]) ? maximum : maxima[3];
  __shared__ uint32_t warp_max[kQuantizeReductionSlots];
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint32_t other = __shfl_xor(maximum, offset);
    maximum = (maximum > other) ? maximum : other;
  }
  if ((tid & 31) == 0) warp_max[tid >> 5] = maximum;
  __syncthreads();
  if (tid < kQuantizeReductionSlots) {
    uint32_t value = warp_max[tid];
#pragma unroll
    for (int offset = kQuantizeReductionSlots >> 1; offset > 0; offset >>= 1) {
      const uint32_t other = __shfl_xor(value, offset);
      value = (value > other) ? value : other;
    }
    if (tid == 0) warp_max[0] = value;
  }
  __syncthreads();
  return absmax_bits_to_float<kIsBf16>(warp_max[0]);
}

template <bool kIsBf16, bool kHasBias>
__global__ void __launch_bounds__(kThreads) w8a8_gfx928_shared_gate_up_m1_kernel(
    const uint16_t* __restrict__ x,
    const uint64_t* __restrict__ weight_k8n,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    uint16_t* __restrict__ output,
    int32_t* __restrict__ accumulation,
    uint32_t* __restrict__ counter) {
  const int tid = threadIdx.x;
  const int segment = blockIdx.y;
  const int k4_begin = segment * kK4PerSegment;
  __shared__ uint32_t quantized[kK4PerSegment];

  const float maximum = full_input_max<kIsBf16>(x, tid);
  const float absmax = fmaxf(maximum, 1e-10f);
  const float scale = absmax / 127.0f;
  const float reciprocal = 127.0f / absmax;

  const auto* x64 = reinterpret_cast<const uint64_t*>(x);
  if (tid < kK4PerSegment) {
    const uint64_t word = x64[k4_begin + tid];
    const uint32_t low = static_cast<uint32_t>(word);
    const uint32_t high = static_cast<uint32_t>(word >> 32);
    quantized[tid] = pack4_i8(
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(low)), reciprocal),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(low >> 16)), reciprocal),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(high)), reciprocal),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(high >> 16)), reciprocal));
  }
  __syncthreads();

  const int column = blockIdx.x * kThreads + tid;
  int accumulators[2] = {0, 0};
  const int k8_begin = k4_begin / 2;
#pragma unroll
  for (int word = 0; word < kUnroll; ++word) {
    const uint64_t weights = weight_k8n[(k8_begin + word) * kN + column];
    accumulators[word & 1] = sdot4(
        static_cast<int>(quantized[2 * word]),
        static_cast<int>(static_cast<uint32_t>(weights)),
        accumulators[word & 1]);
    accumulators[word & 1] = sdot4(
        static_cast<int>(quantized[2 * word + 1]),
        static_cast<int>(static_cast<uint32_t>(weights >> 32)),
        accumulators[word & 1]);
  }
  atomicAdd(&accumulation[column], accumulators[0] + accumulators[1]);
  __threadfence();
  __syncthreads();

  __shared__ int is_last_block;
  if (tid == 0) {
    constexpr unsigned kBlocks = kGridX * kSplit;
    const unsigned old = atomicAdd(counter, 1u);
    is_last_block = (old == kBlocks - 1) ? 1 : 0;
  }
  __syncthreads();
  if (is_last_block) {
    __threadfence();
    for (int output_column = tid; output_column < kN; output_column += kThreads) {
      const float value = static_cast<float>(accumulation[output_column]) * scale * weight_scale[output_column];
      if (kHasBias) {
        output[output_column] = add_output_dtype<kIsBf16>(
            output_from_float<kIsBf16>(value), output_from_float<kIsBf16>(bias[output_column]));
      } else {
        output[output_column] = output_from_float<kIsBf16>(value);
      }
      accumulation[output_column] = 0;
    }
    __threadfence();
    if (tid == 0) counter[0] = 0;
  }
}

void check_tensor(const at::Tensor& tensor, at::ScalarType dtype, at::IntArrayRef shape, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.sizes() == shape, name, " has an invalid shape");
}

void check_no_overlap(
    const at::Tensor& mutable_tensor, const at::Tensor& input, const char* mutable_name, const char* input_name) {
  TORCH_CHECK(
      at::get_overlap_status(mutable_tensor, input) == at::MemOverlapStatus::No,
      mutable_name,
      " must not overlap ",
      input_name);
}

void check_inputs(
    const at::Tensor& x,
    const at::Tensor& weight_k8n,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& accumulation,
    const at::Tensor& counter) {
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be float16 or bfloat16");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.sizes() == at::IntArrayRef({1, kK}), "x must have shape [1, 6144]");
  check_tensor(weight_k8n, at::kLong, {kK8, kN}, "weight_k8n");
  check_tensor(weight_scale, at::kFloat, {kN}, "weight_scale");
  check_tensor(accumulation, at::kInt, {kN}, "acc");
  check_tensor(counter, at::kInt, {1}, "counter");
  if (bias.has_value()) check_tensor(*bias, at::kFloat, {kN}, "bias");

  TORCH_CHECK(
      x.device() == weight_k8n.device() && x.device() == weight_scale.device() && x.device() == accumulation.device() &&
          x.device() == counter.device() && (!bias.has_value() || x.device() == bias->device()),
      "all tensors must be on the same device");

  check_no_overlap(accumulation, x, "acc", "x");
  check_no_overlap(accumulation, weight_k8n, "acc", "weight_k8n");
  check_no_overlap(accumulation, weight_scale, "acc", "weight_scale");
  check_no_overlap(counter, x, "counter", "x");
  check_no_overlap(counter, weight_k8n, "counter", "weight_k8n");
  check_no_overlap(counter, weight_scale, "counter", "weight_scale");
  check_no_overlap(accumulation, counter, "acc", "counter");
  if (bias.has_value()) {
    check_no_overlap(accumulation, *bias, "acc", "bias");
    check_no_overlap(counter, *bias, "counter", "bias");
  }
}

}  // namespace

at::Tensor w8a8_gfx928_shared_gate_up_m1(
    const at::Tensor& x,
    const at::Tensor& weight_k8n,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias,
    at::Tensor& accumulation,
    at::Tensor& counter) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
  check_inputs(x, weight_k8n, weight_scale, bias, accumulation, counter);
  const hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  at::Tensor output = at::empty({1, kN}, x.options());
  const float* bias_pointer = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  const dim3 grid(kGridX, kSplit);

#define LAUNCH_SHARED_GATE_UP(K_IS_BF16, K_HAS_BIAS)                     \
  hipLaunchKernelGGL(                                                    \
      (w8a8_gfx928_shared_gate_up_m1_kernel<K_IS_BF16, K_HAS_BIAS>),     \
      grid,                                                              \
      dim3(kThreads),                                                    \
      0,                                                                 \
      stream,                                                            \
      reinterpret_cast<const uint16_t*>(x.data_ptr()),                   \
      reinterpret_cast<const uint64_t*>(weight_k8n.data_ptr<int64_t>()), \
      weight_scale.data_ptr<float>(),                                    \
      bias_pointer,                                                      \
      reinterpret_cast<uint16_t*>(output.data_ptr()),                    \
      accumulation.data_ptr<int32_t>(),                                  \
      reinterpret_cast<uint32_t*>(counter.data_ptr<int32_t>()))

  if (x.scalar_type() == at::kBFloat16) {
    if (bias.has_value()) {
      LAUNCH_SHARED_GATE_UP(true, true);
    } else {
      LAUNCH_SHARED_GATE_UP(true, false);
    }
  } else if (bias.has_value()) {
    LAUNCH_SHARED_GATE_UP(false, true);
  } else {
    LAUNCH_SHARED_GATE_UP(false, false);
  }
#undef LAUNCH_SHARED_GATE_UP

  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

#else

at::Tensor w8a8_gfx928_shared_gate_up_m1(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::optional<at::Tensor>&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "w8a8_gfx928_shared_gate_up_m1 is only available on gfx928");
}

#endif  // SGL_W8A8_GFX928_SHARED_GATE_UP_M1_BUILD
