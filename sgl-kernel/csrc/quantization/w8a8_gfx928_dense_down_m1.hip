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

#if SGL_W8A8_GFX928_DENSE_DOWN_M1_BUILD

namespace {

// MiniMax-M3 dense-layer (0-2) mlp.down_proj, TP8 decode M=1: K=1536,
// N=6144. Port of the quick unified-template case dense_down_proj_tp8_m1
// (mode 4 PARTIAL fused, variant 10): 2 launches per invocation on the
// caller stream -- quantize_global_kernel (grid 2, qx[kK4] + act_scale[1])
// then partial_pq_fused_kernel<DEPTH=6, THREADS=256, ATOM64=false> (grid
// dim3(6, 16), seg_k4 = 24, smem xw[kK4], software-pipelined uint4 GEMV +
// 4x int32 atomicAdd into the global acc, arrival-counter rendezvous (no
// spin) + last-block full-kN epilogue). Layer-owned workspace
// qx/act_scale/acc/counter; acc zeroed and counter reset by the last block.
constexpr int kK = 1536;
constexpr int kN = 6144;
constexpr int kK4 = kK / 4;
constexpr int kThreads = 256;
constexpr int kSegs = 16;  // split (kK4/16 = 24 k4 rows per segment)
constexpr int kGridX = kN / (4 * kThreads);  // 1024 columns per block
constexpr int kSegK4 = kK4 / kSegs;
constexpr int kDepth = 6;
constexpr int kBlocks = kGridX * kSegs;
constexpr int kQuantizeBlocks = (kK4 + kThreads - 1) / kThreads;
static_assert(kK4 == 384);
static_assert(kSegs == 16);
static_assert(kGridX == 6);
static_assert(kSegK4 == 24);
static_assert(kDepth == 6);
static_assert(kBlocks == 96);
static_assert(kQuantizeBlocks == 2);
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

__device__ __forceinline__ void scale_and_recip(float vmax, float& scale, float& recip) {
  const float absmax = fmaxf(vmax, 1e-10f);
  scale = absmax / 127.0f;
  recip = 127.0f / absmax;
}

// Launch 1: per-invocation full-row quantization (no cross-invocation
// activation-quantize cache, round-10 hygiene). Grid kQuantizeBlocks = 2,
// each block redundantly scans the full row and quantizes its own k4 slice
// into qx[kK4]; every block writes act_scale[0] (= absmax/127, same value).
template <bool kIsBf16>
__global__ void __launch_bounds__(kThreads) quantize_global_kernel(
    const uint16_t* __restrict__ x, uint32_t* __restrict__ qx, float* __restrict__ scale_out) {
  const int tid = threadIdx.x;
  constexpr int kWordsPerBlock = (kK4 + kQuantizeBlocks - 1) / kQuantizeBlocks;
  const int w0 = blockIdx.x * kWordsPerBlock;
  const int w1 = min(w0 + kWordsPerBlock, kK4);
  const auto* x64 = reinterpret_cast<const uint64_t*>(x);

  // redundant full-x max (ILP-4 batched)
  uint32_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
  int idx = tid;
  for (; idx + 3 * kThreads < kK4; idx += 4 * kThreads) {
    uint64_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = x64[idx + j * kThreads];
    m0 = (word_absmax4(w[0]) > m0) ? word_absmax4(w[0]) : m0;
    m1 = (word_absmax4(w[1]) > m1) ? word_absmax4(w[1]) : m1;
    m2 = (word_absmax4(w[2]) > m2) ? word_absmax4(w[2]) : m2;
    m3 = (word_absmax4(w[3]) > m3) ? word_absmax4(w[3]) : m3;
  }
  for (; idx < kK4; idx += kThreads) {
    const uint32_t a = word_absmax4(x64[idx]);
    m0 = (a > m0) ? a : m0;
  }
  uint32_t vmax_bits = (m0 > m1) ? m0 : m1;
  vmax_bits = (vmax_bits > m2) ? vmax_bits : m2;
  vmax_bits = (vmax_bits > m3) ? vmax_bits : m3;
  __shared__ uint32_t s_warp_max[8];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint32_t o = __shfl_xor(vmax_bits, off);
    vmax_bits = (vmax_bits > o) ? vmax_bits : o;
  }
  if ((tid & 31) == 0) s_warp_max[tid >> 5] = vmax_bits;
  __syncthreads();
  if (tid < 8) {
    uint32_t v = s_warp_max[tid];
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      const uint32_t o = __shfl_xor(v, off);
      v = (v > o) ? v : o;
    }
    if (tid == 0) s_warp_max[0] = v;
  }
  __syncthreads();
  const float vmax = absmax_bits_to_float<kIsBf16>(s_warp_max[0]);
  float scale, recip;
  scale_and_recip(vmax, scale, recip);
  if (tid == 0) scale_out[0] = scale;

  // quantize own slice (FP32 reciprocal multiply)
  for (int w = w0 + tid; w < w1; w += kThreads) {
    const uint64_t xw = x64[w];
    const uint32_t a = static_cast<uint32_t>(xw);
    const uint32_t b = static_cast<uint32_t>(xw >> 32);
    qx[w] = pack4_i8(
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(a)), recip),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(a >> 16)), recip),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(b)), recip),
        quantize_one(input_to_float<kIsBf16>(static_cast<uint16_t>(b >> 16)), recip));
  }
}

// Launch 2: partial-pq fused GEMV (round-7 promoted default, variant 10).
// Split = 16 segments of kSegK4 = 24 k4 rows; grid dim3(kGridX=6, kSegs=16)
// = 96 co-resident blocks; each block stages the whole qx row into smem
// xw[kK4], runs a DEPTH-6 software-pipelined uint4 (16B, 4 columns x 4 k4
// rows per word row) sdot4 GEMV over its own 1024-column slice, adds its 4
// int32 partials into the global acc with 4x atomicAdd, then an
// arrival-counter rendezvous (no spin): exactly the last block reads back
// all kN columns from acc (L2-hot 24KB), scales with act_scale[0]*w_scale,
// RNE-casts (cast-before-bias), zeroes acc for the next invocation and
// resets counter; every other block exits right after its counter add.
template <bool kIsBf16, bool kHasBias>
__global__ void __launch_bounds__(kThreads) partial_pq_fused_kernel(
    const uint32_t* __restrict__ qx,        // [kK4]
    const uint32_t* __restrict__ weight_k4n,  // [kK4][kN] uint32
    int32_t* __restrict__ acc,              // [kN], zeroed at launch, self-reset
    uint32_t* __restrict__ counter,         // [1], zero at launch, self-reset
    const float* __restrict__ act_scale,    // [1]
    const float* __restrict__ weight_scale, // [kN]
    const float* __restrict__ bias,         // [kN] or nullptr
    uint16_t* __restrict__ output) {        // [kN]
  const int tid = threadIdx.x;
  const int seg = blockIdx.y;
  const int col0 = blockIdx.x * (4 * kThreads) + tid * 4;

  __shared__ uint32_t xw[kK4];
  for (int w = tid; w < kK4; w += kThreads) xw[w] = qx[w];
  __syncthreads();

  const int k4_begin = seg * kSegK4;
  const int k4_end = min(k4_begin + kSegK4, kK4);
  int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  int c = k4_begin;
  if (k4_end - c >= kDepth) {
    uint4 w[kDepth];
#pragma unroll
    for (int i = 0; i < kDepth; ++i)
      w[i] = *reinterpret_cast<const uint4*>(&weight_k4n[(size_t)(c + i) * kN + col0]);
    // steady state: prefetch the next kDepth vectors (guarded so all loads
    // are in range), then consume w[].  kDepth*4 loads in flight.
    for (; c + 2 * kDepth - 1 < k4_end; c += kDepth) {
      uint4 nxt[kDepth];
#pragma unroll
      for (int i = 0; i < kDepth; ++i)
        nxt[i] = *reinterpret_cast<const uint4*>(&weight_k4n[(size_t)(c + kDepth + i) * kN + col0]);
#pragma unroll
      for (int i = 0; i < kDepth; ++i) {
        const uint32_t a = xw[c + i];
        a0 = sdot4(static_cast<int>(a), static_cast<int>(w[i].x), a0);
        a1 = sdot4(static_cast<int>(a), static_cast<int>(w[i].y), a1);
        a2 = sdot4(static_cast<int>(a), static_cast<int>(w[i].z), a2);
        a3 = sdot4(static_cast<int>(a), static_cast<int>(w[i].w), a3);
      }
#pragma unroll
      for (int i = 0; i < kDepth; ++i) w[i] = nxt[i];
    }
    // exit state: kDepth <= k4_end - c < 2*kDepth, and w[] holds vectors for
    // indices c..c+kDepth-1 (all in range) -> consume all, then scalar tail.
#pragma unroll
    for (int i = 0; i < kDepth; ++i) {
      const uint32_t a = xw[c + i];
      a0 = sdot4(static_cast<int>(a), static_cast<int>(w[i].x), a0);
      a1 = sdot4(static_cast<int>(a), static_cast<int>(w[i].y), a1);
      a2 = sdot4(static_cast<int>(a), static_cast<int>(w[i].z), a2);
      a3 = sdot4(static_cast<int>(a), static_cast<int>(w[i].w), a3);
    }
    c += kDepth;
  } else {
    // fewer than kDepth vectors: 4-chunked loads then scalar (same shape as
    // the pre-template body for the short-segment path)
    for (; c + 3 < k4_end; c += 4) {
      const uint4 w0 = *reinterpret_cast<const uint4*>(&weight_k4n[(size_t)(c + 0) * kN + col0]);
      const uint4 w1 = *reinterpret_cast<const uint4*>(&weight_k4n[(size_t)(c + 1) * kN + col0]);
      const uint4 w2 = *reinterpret_cast<const uint4*>(&weight_k4n[(size_t)(c + 2) * kN + col0]);
      const uint4 w3 = *reinterpret_cast<const uint4*>(&weight_k4n[(size_t)(c + 3) * kN + col0]);
      const uint32_t a = xw[c + 0], a_ = xw[c + 1], a__ = xw[c + 2], a___ = xw[c + 3];
      a0 = sdot4(static_cast<int>(a), static_cast<int>(w0.x), a0);
      a1 = sdot4(static_cast<int>(a), static_cast<int>(w0.y), a1);
      a2 = sdot4(static_cast<int>(a), static_cast<int>(w0.z), a2);
      a3 = sdot4(static_cast<int>(a), static_cast<int>(w0.w), a3);
      a0 = sdot4(static_cast<int>(a_), static_cast<int>(w1.x), a0);
      a1 = sdot4(static_cast<int>(a_), static_cast<int>(w1.y), a1);
      a2 = sdot4(static_cast<int>(a_), static_cast<int>(w1.z), a2);
      a3 = sdot4(static_cast<int>(a_), static_cast<int>(w1.w), a3);
      a0 = sdot4(static_cast<int>(a__), static_cast<int>(w2.x), a0);
      a1 = sdot4(static_cast<int>(a__), static_cast<int>(w2.y), a1);
      a2 = sdot4(static_cast<int>(a__), static_cast<int>(w2.z), a2);
      a3 = sdot4(static_cast<int>(a__), static_cast<int>(w2.w), a3);
      a0 = sdot4(static_cast<int>(a___), static_cast<int>(w3.x), a0);
      a1 = sdot4(static_cast<int>(a___), static_cast<int>(w3.y), a1);
      a2 = sdot4(static_cast<int>(a___), static_cast<int>(w3.z), a2);
      a3 = sdot4(static_cast<int>(a___), static_cast<int>(w3.w), a3);
    }
  }
  for (; c < k4_end; ++c) {
    const uint4 w = *reinterpret_cast<const uint4*>(&weight_k4n[(size_t)c * kN + col0]);
    const uint32_t xwv = xw[c];
    a0 = sdot4(static_cast<int>(xwv), static_cast<int>(w.x), a0);
    a1 = sdot4(static_cast<int>(xwv), static_cast<int>(w.y), a1);
    a2 = sdot4(static_cast<int>(xwv), static_cast<int>(w.z), a2);
    a3 = sdot4(static_cast<int>(xwv), static_cast<int>(w.w), a3);
  }
  // ATOM64=false: four plain 32-bit atomic adds (order-independent; result
  // stays bit-exact vs the CPU int64 oracle).
  atomicAdd(&acc[col0 + 0], a0);
  atomicAdd(&acc[col0 + 1], a1);
  atomicAdd(&acc[col0 + 2], a2);
  atomicAdd(&acc[col0 + 3], a3);
  __threadfence();
  __syncthreads();

  // Arrival-counter rendezvous without spin.  tid0's atomicAdd return value
  // kBlocks-1 proves every block's acc atomicAdds (released device-wide by
  // the per-thread __threadfence above, ordered before tid0's counter add by
  // __syncthreads) are globally visible, so exactly the last block reads a
  // complete acc; every other block exits immediately.  Nothing ever waits
  // on a value that another block resets, so no poller can be stranded --
  // the previous spin-on-counter==kBlocks rendezvous had a reset window that
  // could race a slow poller and hang the whole grid forever.
  __shared__ int is_last_block;
  if (tid == 0) {
    is_last_block = (atomicAdd(counter, 1) == kBlocks - 1);
  }
  __syncthreads();
  if (is_last_block) {
    __threadfence();
    // Single read-back + scale + store + zero pass over all kN columns
    // (6 int4 per thread); no cross-block read/clear overlap exists because
    // only this block touches acc in the epilogue.
    const float as = act_scale[0];
#pragma unroll
    for (int g = 0; g < kGridX; ++g) {
      const int c0 = g * (4 * kThreads) + tid * 4;
      const int4 s = *reinterpret_cast<const int4*>(acc + c0);
      const float ws0 = weight_scale[c0 + 0], ws1 = weight_scale[c0 + 1];
      const float ws2 = weight_scale[c0 + 2], ws3 = weight_scale[c0 + 3];
      if (kHasBias) {
        // Round-10 cast-before-bias (bias added in the output dtype)
        output[c0 + 0] = add_output_dtype<kIsBf16>(
            output_from_float<kIsBf16>(static_cast<float>(s.x) * as * ws0), output_from_float<kIsBf16>(bias[c0 + 0]));
        output[c0 + 1] = add_output_dtype<kIsBf16>(
            output_from_float<kIsBf16>(static_cast<float>(s.y) * as * ws1), output_from_float<kIsBf16>(bias[c0 + 1]));
        output[c0 + 2] = add_output_dtype<kIsBf16>(
            output_from_float<kIsBf16>(static_cast<float>(s.z) * as * ws2), output_from_float<kIsBf16>(bias[c0 + 2]));
        output[c0 + 3] = add_output_dtype<kIsBf16>(
            output_from_float<kIsBf16>(static_cast<float>(s.w) * as * ws3), output_from_float<kIsBf16>(bias[c0 + 3]));
      } else {
        output[c0 + 0] = output_from_float<kIsBf16>(static_cast<float>(s.x) * as * ws0);
        output[c0 + 1] = output_from_float<kIsBf16>(static_cast<float>(s.y) * as * ws1);
        output[c0 + 2] = output_from_float<kIsBf16>(static_cast<float>(s.z) * as * ws2);
        output[c0 + 3] = output_from_float<kIsBf16>(static_cast<float>(s.w) * as * ws3);
      }
      *reinterpret_cast<int4*>(acc + c0) = make_int4(0, 0, 0, 0);
    }
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
    const at::Tensor& qx,
    const at::Tensor& act_scale,
    const at::Tensor& acc,
    const at::Tensor& counter) {
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be float16 or bfloat16");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.sizes() == at::IntArrayRef({1, kK}), "x must have shape [1, 1536]");
  check_tensor(weight_k4n, at::kInt, {kK4, kN}, "weight_k4n");
  check_tensor(weight_scale, at::kFloat, {kN}, "weight_scale");
  check_tensor(qx, at::kInt, {kK4}, "qx");
  check_tensor(act_scale, at::kFloat, {1}, "act_scale");
  check_tensor(acc, at::kInt, {kN}, "acc");
  check_tensor(counter, at::kInt, {1}, "counter");
  if (bias.has_value()) check_tensor(*bias, at::kFloat, {kN}, "bias");

  TORCH_CHECK(
      x.device() == weight_k4n.device() && x.device() == weight_scale.device() && x.device() == qx.device() &&
          x.device() == act_scale.device() && x.device() == acc.device() && x.device() == counter.device() &&
          (!bias.has_value() || x.device() == bias->device()),
      "all tensors must be on the same device");

  check_no_overlap(qx, x, "qx", "x");
  check_no_overlap(qx, weight_k4n, "qx", "weight_k4n");
  check_no_overlap(act_scale, x, "act_scale", "x");
  check_no_overlap(act_scale, weight_k4n, "act_scale", "weight_k4n");
  check_no_overlap(acc, x, "acc", "x");
  check_no_overlap(acc, weight_k4n, "acc", "weight_k4n");
  check_no_overlap(counter, x, "counter", "x");
  check_no_overlap(counter, weight_k4n, "counter", "weight_k4n");
  check_no_overlap(qx, act_scale, "qx", "act_scale");
  check_no_overlap(qx, acc, "qx", "acc");
  check_no_overlap(qx, counter, "qx", "counter");
  check_no_overlap(act_scale, acc, "act_scale", "acc");
  check_no_overlap(act_scale, counter, "act_scale", "counter");
  check_no_overlap(acc, counter, "acc", "counter");
}

}  // namespace

at::Tensor w8a8_gfx928_dense_down_m1(
    const at::Tensor& x,
    const at::Tensor& weight_k4n,
    const at::Tensor& weight_scale,
    const std::optional<at::Tensor>& bias,
    at::Tensor& qx,
    at::Tensor& act_scale,
    at::Tensor& acc,
    at::Tensor& counter) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
  check_inputs(x, weight_k4n, weight_scale, bias, qx, act_scale, acc, counter);
  const hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  at::Tensor output = at::empty({1, kN}, x.options());
  const float* bias_pointer = bias.has_value() ? bias->data_ptr<float>() : nullptr;

  const dim3 quantize_grid(kQuantizeBlocks);
  const dim3 partial_grid(kGridX, kSegs);
  if (x.scalar_type() == at::kBFloat16) {
    hipLaunchKernelGGL(
        (quantize_global_kernel<true>), quantize_grid, dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(x.data_ptr()), reinterpret_cast<uint32_t*>(qx.data_ptr<int32_t>()),
        act_scale.data_ptr<float>());
  } else {
    hipLaunchKernelGGL(
        (quantize_global_kernel<false>), quantize_grid, dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(x.data_ptr()), reinterpret_cast<uint32_t*>(qx.data_ptr<int32_t>()),
        act_scale.data_ptr<float>());
  }
#define LAUNCH_DENSE_DOWN(K_IS_BF16, K_HAS_BIAS)                                              \
  hipLaunchKernelGGL(                                                                         \
      (partial_pq_fused_kernel<K_IS_BF16, K_HAS_BIAS>), partial_grid, dim3(kThreads), 0,      \
      stream,                                                                                 \
      reinterpret_cast<const uint32_t*>(qx.data_ptr<int32_t>()),                              \
      reinterpret_cast<const uint32_t*>(weight_k4n.data_ptr<int32_t>()),                      \
      acc.data_ptr<int32_t>(),                                                                \
      reinterpret_cast<uint32_t*>(counter.data_ptr<int32_t>()),                               \
      act_scale.data_ptr<float>(), weight_scale.data_ptr<float>(), bias_pointer,              \
      reinterpret_cast<uint16_t*>(output.data_ptr()))

  if (x.scalar_type() == at::kBFloat16) {
    if (bias.has_value()) {
      LAUNCH_DENSE_DOWN(true, true);
    } else {
      LAUNCH_DENSE_DOWN(true, false);
    }
  } else if (bias.has_value()) {
    LAUNCH_DENSE_DOWN(false, true);
  } else {
    LAUNCH_DENSE_DOWN(false, false);
  }
#undef LAUNCH_DENSE_DOWN

  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

#else

at::Tensor w8a8_gfx928_dense_down_m1(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::optional<at::Tensor>&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "w8a8_gfx928_dense_down_m1 is only available on gfx928");
}

#endif  // SGL_W8A8_GFX928_DENSE_DOWN_M1_BUILD
