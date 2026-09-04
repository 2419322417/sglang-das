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

#if SGL_W8A8_GFX928_O_PROJ_M1_BUILD

namespace {

// MiniMax-M3 per-rank self_attn.o_proj, TP8 decode M=1: K=1024, N=6144.
// Port of the quick unified-template case o_proj_tp8_m1 (mode 1 TILE):
// fused_tile_kernel<SEGS=32, UNROLL=8>, 1024 threads, grid kN/32 = 192,
// tile uint64 layout-3 weights. No global acc/counter: the dot product is
// reduced inside the block (smem red_s) and warp 0 runs the epilogue.
constexpr int kK = 1024;
constexpr int kN = 6144;
constexpr int kK8 = kK / 8;
constexpr int kSegs = 32;
constexpr int kUnroll = 8;
constexpr int kThreads = 32 * kSegs;
constexpr int kGridX = kN / 32;
constexpr int kTileWords = kK8 * 32;
static_assert(kGridX == 192);
static_assert(kThreads == 1024);
static_assert(kTileWords == 4096);
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

// Round-10 full-x-row magnitude-bit absmax scan. For o_proj K=1024 the data
// words (256 uint64) are read exclusively by warp 0; the NSLOTS = kThreads/32
// = 32 slot reduction covers all 32 warps (round-9 promoted fix).
template <bool kIsBf16, int kP1Win, int kNSLOTS>
__device__ __forceinline__ float full_x_max_v2(const uint16_t* __restrict__ x, int tid, int nthreads) {
  constexpr int kK4Words = kK / 4;
  const auto* x64 = reinterpret_cast<const uint64_t*>(x);
  uint32_t maxima[4] = {0, 0, 0, 0};
  int index = tid;
  for (; index + 3 * nthreads < kK4Words; index += 4 * nthreads) {
    uint64_t words[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) words[i] = x64[index + i * nthreads];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t value = word_absmax4(words[i]);
      maxima[i] = (value > maxima[i]) ? value : maxima[i];
    }
  }
  for (; index < kK4Words; index += nthreads) {
    const uint32_t value = word_absmax4(x64[index]);
    maxima[0] = (value > maxima[0]) ? value : maxima[0];
  }
  uint32_t maximum = (maxima[0] > maxima[1]) ? maxima[0] : maxima[1];
  maximum = (maximum > maxima[2]) ? maximum : maxima[2];
  maximum = (maximum > maxima[3]) ? maximum : maxima[3];
  (void)kP1Win;
  __shared__ uint32_t warp_max[kNSLOTS];
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint32_t other = __shfl_xor(maximum, offset);
    maximum = (maximum > other) ? maximum : other;
  }
  if ((tid & 31) == 0) warp_max[tid >> 5] = maximum;
  __syncthreads();
  if (tid < kNSLOTS) {
    uint32_t value = warp_max[tid];
#pragma unroll
    for (int offset = kNSLOTS >> 1; offset > 0; offset >>= 1) {
      const uint32_t other = __shfl_xor(value, offset);
      value = (value > other) ? value : other;
    }
    if (tid == 0) warp_max[0] = value;
  }
  __syncthreads();
  return absmax_bits_to_float<kIsBf16>(warp_max[0]);
}

__device__ __forceinline__ void scale_and_recip(float vmax, float& scale, float& recip) {
  const float absmax = fmaxf(vmax, 1e-10f);
  scale = absmax / 127.0f;
  recip = 127.0f / absmax;
}

// Single-launch tile-major GEMV: per-block redundant quantize into smem
// (qx_s[kK] bytes, K=1024 is the exact smem upper bound), linear uint64 tile
// reads, cross-warp smem reduce, warp-0 epilogue. No cross-block state.
template <bool kIsBf16, bool kHasBias>
__global__ void __launch_bounds__(kThreads) w8a8_gfx928_o_proj_m1_kernel(
    const uint16_t* __restrict__ x,
    const uint64_t* __restrict__ weight_tile,  // [kN/32][kK8*32]
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    uint16_t* __restrict__ output) {
  const int tid = threadIdx.x;
  const int seg = tid >> 5;   // warp == k segment (kSegs warps)
  const int lane = tid & 31;  // column within the 32-column tile
  const int col = blockIdx.x * 32 + lane;
  constexpr int kNWarps = kThreads / 32;
  __shared__ __attribute__((aligned(16))) uint8_t qx_s[kK];  // kK = 1024

  // ---- per-block redundant quantization (reciprocal multiply) ----
  const float vmax = full_x_max_v2<kIsBf16, 4, kNWarps>(x, tid, kThreads);
  float scale, recip;
  scale_and_recip(vmax, scale, recip);
  for (int i = tid; i < kK; i += kThreads) {
    const float v = input_to_float<kIsBf16>(x[i]) * recip;
    const float rounded = rintf(v);
    int qi = static_cast<int>(rounded);
    if (qi < -127) qi = -127;
    if (qi > 127) qi = 127;
    qx_s[i] = static_cast<uint8_t>(qi);
  }
  __syncthreads();

  // ---- GEMV: linear read of this block's tile ----
  const uint32_t* qw = reinterpret_cast<const uint32_t*>(qx_s);
  int32_t acc[kUnroll];
#pragma unroll
  for (int u = 0; u < kUnroll; ++u) acc[u] = 0;
  const uint64_t* tile = weight_tile + static_cast<int64_t>(blockIdx.x) * kTileWords;
  int w = tid;
  const int step = kThreads;
  for (; w + step * kUnroll <= kTileWords; w += step * kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int ww = w + u * step;
      const uint64_t value = tile[ww];
      const int k8w = ww >> 5;
      acc[u] = sdot4(static_cast<int>(qw[2 * k8w]), static_cast<int>(static_cast<uint32_t>(value)), acc[u]);
      acc[u] = sdot4(
          static_cast<int>(qw[2 * k8w + 1]), static_cast<int>(static_cast<uint32_t>(value >> 32)), acc[u]);
    }
  }
  for (; w < kTileWords; w += step) {
    const uint64_t value = tile[w];
    const int k8w = w >> 5;
    acc[0] = sdot4(static_cast<int>(qw[2 * k8w]), static_cast<int>(static_cast<uint32_t>(value)), acc[0]);
    acc[0] = sdot4(static_cast<int>(qw[2 * k8w + 1]),
                   static_cast<int>(static_cast<uint32_t>(value >> 32)), acc[0]);
  }
  int32_t sum = 0;
#pragma unroll
  for (int u = 0; u < kUnroll; ++u) sum += acc[u];

  // ---- cross-warp reduce + fused epilogue (warp 0) ----
  __shared__ int32_t red_s[kSegs][32];
  red_s[seg][lane] = sum;
  __syncthreads();
  if (seg == 0) {
    int32_t total = 0;
#pragma unroll
    for (int s = 0; s < kSegs; ++s) total += red_s[s][lane];
    const float value = static_cast<float>(total) * scale * weight_scale[col];
    if (kHasBias) {
      output[col] = add_output_dtype<kIsBf16>(output_from_float<kIsBf16>(value), output_from_float<kIsBf16>(bias[col]));
    } else {
      output[col] = output_from_float<kIsBf16>(value);
    }
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
    const at::Tensor& weight_tile,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be float16 or bfloat16");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.sizes() == at::IntArrayRef({1, kK}), "x must have shape [1, 1024]");
  check_tensor(weight_tile, at::kLong, {kGridX, kTileWords}, "weight_tile");
  check_tensor(weight_scale, at::kFloat, {kN}, "weight_scale");
  if (bias.has_value()) check_tensor(*bias, at::kFloat, {kN}, "bias");

  TORCH_CHECK(
      x.device() == weight_tile.device() && x.device() == weight_scale.device() &&
          (!bias.has_value() || x.device() == bias->device()),
      "all tensors must be on the same device");

  check_no_overlap(weight_tile, x, "weight_tile", "x");
  check_no_overlap(weight_scale, x, "weight_scale", "x");
  if (bias.has_value()) check_no_overlap(*bias, x, "bias", "x");
}

}  // namespace

at::Tensor w8a8_gfx928_o_proj_m1(
    const at::Tensor& x,
    const at::Tensor& weight_tile,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
  check_inputs(x, weight_tile, weight_scale, bias);
  const hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  at::Tensor output = at::empty({1, kN}, x.options());
  const float* bias_pointer = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  const dim3 grid(kGridX);

#define LAUNCH_O_PROJ(K_IS_BF16, K_HAS_BIAS)                                   \
  hipLaunchKernelGGL(                                                          \
      (w8a8_gfx928_o_proj_m1_kernel<K_IS_BF16, K_HAS_BIAS>),                   \
      grid,                                                                    \
      dim3(kThreads),                                                          \
      0,                                                                       \
      stream,                                                                  \
      reinterpret_cast<const uint16_t*>(x.data_ptr()),                         \
      reinterpret_cast<const uint64_t*>(weight_tile.data_ptr<int64_t>()),      \
      weight_scale.data_ptr<float>(),                                          \
      bias_pointer,                                                            \
      reinterpret_cast<uint16_t*>(output.data_ptr()))

  if (x.scalar_type() == at::kBFloat16) {
    if (bias.has_value()) {
      LAUNCH_O_PROJ(true, true);
    } else {
      LAUNCH_O_PROJ(true, false);
    }
  } else if (bias.has_value()) {
    LAUNCH_O_PROJ(false, true);
  } else {
    LAUNCH_O_PROJ(false, false);
  }
#undef LAUNCH_O_PROJ

  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

#else

at::Tensor w8a8_gfx928_o_proj_m1(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::optional<at::Tensor>&) {
  TORCH_CHECK(false, "w8a8_gfx928_o_proj_m1 is only available on gfx928");
}

#endif  // SGL_W8A8_GFX928_O_PROJ_M1_BUILD
