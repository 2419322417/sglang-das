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

#if SGL_W8A8_GFX928_QKV_PROJ_M1_BUILD

namespace {

// MiniMax-M3 per-rank self_attn.qkv_proj, TP8 decode M=1: K=6144, N=1280.
// Port of the quick unified-template case qkv_proj_tp8_m1 (mode 0 ATOMIC,
// variant 10): fused_atomic_kernel<LAYOUT=0, SPLIT=32, THREADS=256, CPT=1,
// UNROLL=16, P1WIN=4>, grid dim3(5, 32), k4n uint32 weight words, global
// int32 acc + counter/last-block epilogue (acc/counter self-reset).
constexpr int kK = 6144;
constexpr int kN = 1280;
constexpr int kK4 = kK / 4;
constexpr int kSplit = 32;
constexpr int kThreads = 256;
constexpr int kGridX = kN / kThreads;
constexpr int kK4PerSegment = kK4 / kSplit;
constexpr int kUnroll = 16;
constexpr int kP1Win = 4;
constexpr int kQuantizeReductionSlots = kThreads / 32;
static_assert(kGridX == 5);
static_assert(kK4PerSegment == 48);
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

// q = clamp(rintf(x * recip), -127, 127) with recip = 127/absmax (RNE,
// round-half-even, matches torch.round). Round-10 reciprocal-multiply.
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

// Round-10 full-x-row magnitude-bit absmax scan (round-9 fix: NSLOTS =
// nthreads/32 slot reduction covers THREADS > 256; for THREADS = 256 it is
// bit-identical to the original 8-slot scan). Nonfinite activations must
// never reach here (the selector rejects them upstream).
template <bool kIsBf16, int kP1Win, int kNSLOTS>
__device__ __forceinline__ float full_x_max_v2(const uint16_t* __restrict__ x, int tid, int nthreads) {
  constexpr int kK4Words = kK4;  // uint64 words = K/4
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

// Round-10 quantization constants: absmax = max(vmax, 1e-10), scale =
// absmax/127 (output scale), recip = 127/absmax (quantize multiplier).
__device__ __forceinline__ void scale_and_recip(float vmax, float& scale, float& recip) {
  const float absmax = fmaxf(vmax, 1e-10f);
  scale = absmax / 127.0f;
  recip = 127.0f / absmax;
}

// Single-launch k4n GEMV: per-block redundant quantize of the full row,
// smem-slice sdot4, atomicAdd into the global int32 acc, last block (counter)
// dequantizes (fp32 acc*scale*w_scale, single RNE cast to the output dtype,
// cast-before-bias) and resets acc/counter for the next invocation.
template <bool kIsBf16, bool kHasBias>
__global__ void __launch_bounds__(kThreads) w8a8_gfx928_qkv_proj_m1_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight_k4n,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    uint16_t* __restrict__ output,
    int32_t* __restrict__ accumulation,
    uint32_t* __restrict__ counter) {
  const int tid = threadIdx.x;
  const int segment = blockIdx.y;
  const int k4_begin = segment * kK4PerSegment;
  __shared__ uint32_t quantized[kK4PerSegment];

  // ---- phase A: full-x row absmax (redundant per block) ----
  const float vmax = full_x_max_v2<kIsBf16, kP1Win, kQuantizeReductionSlots>(x, tid, kThreads);
  float scale, recip;
  scale_and_recip(vmax, scale, recip);

  // ---- phase B: quantize own slice into smem (reciprocal multiply) ----
  const auto* x64 = reinterpret_cast<const uint64_t*>(x);
  for (int w = tid; w < kK4PerSegment; w += kThreads) {
    const uint64_t word = x64[k4_begin + w];
    const uint32_t low = static_cast<uint32_t>(word);
    const uint32_t high = static_cast<uint32_t>(word >> 32);
    quantized[w] = pack4_i8(
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(low)), recip),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(low >> 16)), recip),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(high)), recip),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(high >> 16)), recip));
  }
  __syncthreads();

  // ---- phase C: GEMV own k4 slice, k4n uint32 word loads, 8 chains ----
  const int column = blockIdx.x * kThreads + tid;
  if (column < kN) {
    int accv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t* wcol = weight_k4n + column;
    int w = 0;
    for (; w + kUnroll - 1 < kK4PerSegment; w += kUnroll) {
      uint32_t q[kUnroll], ww[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        q[u] = quantized[w + u];
        ww[u] = wcol[(k4_begin + w + u) * kN];
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u)
        accv[u & 7] = sdot4(static_cast<int>(q[u]), static_cast<int>(ww[u]), accv[u & 7]);
    }
    for (; w < kK4PerSegment; ++w)
      accv[0] = sdot4(static_cast<int>(quantized[w]),
                      static_cast<int>(wcol[(k4_begin + w) * kN]), accv[0]);
    int total = 0;
#pragma unroll
    for (int u = 0; u < 8; ++u) total += accv[u];
    atomicAdd(&accumulation[column], total);
  }
  __threadfence();
  __syncthreads();

  // ---- phase D: last block dequantizes and resets acc/counter ----
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
    const at::Tensor& weight_k4n,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& accumulation,
    const at::Tensor& counter) {
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be float16 or bfloat16");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.sizes() == at::IntArrayRef({1, kK}), "x must have shape [1, 6144]");
  check_tensor(weight_k4n, at::kInt, {kK4, kN}, "weight_k4n");
  check_tensor(weight_scale, at::kFloat, {kN}, "weight_scale");
  check_tensor(accumulation, at::kInt, {kN}, "acc");
  check_tensor(counter, at::kInt, {1}, "counter");
  if (bias.has_value()) check_tensor(*bias, at::kFloat, {kN}, "bias");

  TORCH_CHECK(
      x.device() == weight_k4n.device() && x.device() == weight_scale.device() && x.device() == accumulation.device() &&
          x.device() == counter.device() && (!bias.has_value() || x.device() == bias->device()),
      "all tensors must be on the same device");

  check_no_overlap(accumulation, x, "acc", "x");
  check_no_overlap(accumulation, weight_k4n, "acc", "weight_k4n");
  check_no_overlap(accumulation, weight_scale, "acc", "weight_scale");
  check_no_overlap(counter, x, "counter", "x");
  check_no_overlap(counter, weight_k4n, "counter", "weight_k4n");
  check_no_overlap(counter, weight_scale, "counter", "weight_scale");
  check_no_overlap(accumulation, counter, "acc", "counter");
  if (bias.has_value()) {
    check_no_overlap(accumulation, *bias, "acc", "bias");
    check_no_overlap(counter, *bias, "counter", "bias");
  }
}

}  // namespace

at::Tensor w8a8_gfx928_qkv_proj_m1(
    const at::Tensor& x,
    const at::Tensor& weight_k4n,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias,
    at::Tensor& acc,
    at::Tensor& counter) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
  check_inputs(x, weight_k4n, weight_scale, bias, acc, counter);
  const hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  at::Tensor output = at::empty({1, kN}, x.options());
  const float* bias_pointer = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  const dim3 grid(kGridX, kSplit);

#define LAUNCH_QKV_PROJ(K_IS_BF16, K_HAS_BIAS)                      \
  hipLaunchKernelGGL(                                               \
      (w8a8_gfx928_qkv_proj_m1_kernel<K_IS_BF16, K_HAS_BIAS>),      \
      grid,                                                         \
      dim3(kThreads),                                               \
      0,                                                            \
      stream,                                                       \
      reinterpret_cast<const uint16_t*>(x.data_ptr()),              \
      reinterpret_cast<const uint32_t*>(weight_k4n.data_ptr<int32_t>()), \
      weight_scale.data_ptr<float>(),                               \
      bias_pointer,                                                 \
      reinterpret_cast<uint16_t*>(output.data_ptr()),               \
      acc.data_ptr<int32_t>(),                                      \
      reinterpret_cast<uint32_t*>(counter.data_ptr<int32_t>()))

  if (x.scalar_type() == at::kBFloat16) {
    if (bias.has_value()) {
      LAUNCH_QKV_PROJ(true, true);
    } else {
      LAUNCH_QKV_PROJ(true, false);
    }
  } else if (bias.has_value()) {
    LAUNCH_QKV_PROJ(false, true);
  } else {
    LAUNCH_QKV_PROJ(false, false);
  }
#undef LAUNCH_QKV_PROJ

  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

#else

at::Tensor w8a8_gfx928_qkv_proj_m1(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::optional<at::Tensor>&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "w8a8_gfx928_qkv_proj_m1 is only available on gfx928");
}

#endif  // SGL_W8A8_GFX928_QKV_PROJ_M1_BUILD
