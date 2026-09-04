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

#if SGL_W8A8_GFX928_DENSE_GATE_UP_M1_BUILD

namespace {

// MiniMax-M3 dense-layer (0-2) mlp.gate_up_proj, TP8 decode M=1: K=6144,
// N=3072. Port of the quick unified-template case dense_gate_up_proj_tp8_m1
// (mode 3 GRIDSYNC): 2 launches per invocation on the caller stream --
// quantize_global_kernel (grid 6, qx[kK4] + act_scale[1]) then
// fused_gridsync_kernel (grid 384 = UW_STRIDE*COLS4/256, wave-ordered
// dot products accumulated into the global acc[kN] via plain int32
// atomicAdd, then an arrival-counter rendezvous (no spin): exactly the last
// block reads back all kN columns, fp32-scales, RNE-casts (cast-before-bias)
// and stores the output, zeroes acc for the next invocation and resets
// counter; every other block exits right after its counter add. Host-side
// g_dir inversion per invocation; layer-owned workspace qx/act_scale/acc/
// counter, counter self-resets (last block).
constexpr int kK = 6144;
constexpr int kN = 3072;
constexpr int kK4 = kK / 4;
constexpr int kThreads = 256;
constexpr int kUWWaves = 12;
constexpr int kUWStride = 128;
constexpr int kCOLS4 = kN / 4;
constexpr int kBlocks = kUWStride * kCOLS4 / kThreads;
constexpr int kQuantizeBlocks = (kK4 + kThreads - 1) / kThreads;
constexpr int kEpilogueGroups = kN / (4 * kThreads);
static_assert(kK4 == 1536);
static_assert(kUWWaves * kUWStride == kK4);
static_assert(kCOLS4 == 768);
static_assert(kBlocks == 384);
static_assert(kQuantizeBlocks == 6);
static_assert(kEpilogueGroups == 3);
static_assert(static_cast<int64_t>(kK) * 127 * 128 < INT32_MAX);

static int g_dir = 1;  // mode-3 wave-order alternation (host-side, per call)

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
// activation-quantize cache, round-10 hygiene). Grid kQuantizeBlocks, each
// block redundantly scans the full row and quantizes its own k4 slice into
// qx[kK4]; every block writes act_scale[0] (= absmax/127, same value).
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

// Wave order alternation (deterministic per captured/replayed node).
__device__ __forceinline__ int uw_wave_of(int dir, int pos) {
  const int table[2][kUWWaves] = {
      {11, 0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5},
      {5, 2, 6, 9, 4, 1, 7, 10, 3, 0, 8, 11},
  };
  return table[(dir < 0) ? 1 : 0][pos];
}

// Launch 2: co-resident grid-sync GEMV over the pre-quantized qx. Each of
// the kBlocks blocks covers one (wave-row base, 4-column group) cell of the
// 12-wave x 128-base x 768-col4 space, sdot4 x 4 columns per wave word, then
// adds its 4 int32 partials into the global acc[4q..4q+3] with 4 plain
// atomicAdds (each output column accumulates the same 128 base addends;
// int32 addition is order-independent and overflow-free, so the result stays
// bit-exact in any arrival order). An arrival-counter rendezvous (no spin):
// exactly the last block reads back all kN columns from acc (L2-hot 12KB),
// scales with act_scale[0]*w_scale, RNE-casts (cast-before-bias), zeroes
// acc for the next invocation and resets counter; every other block exits
// right after its counter add.
template <bool kIsBf16>
__global__ void __launch_bounds__(kThreads) fused_gridsync_kernel(
    const uint32_t* __restrict__ weight_k4n,  // [kK4, kN] uint32
    const uint32_t* __restrict__ qx,          // [kK4]
    int32_t* __restrict__ acc,                // [kN], zeroed at launch, self-reset
    const float* __restrict__ act_scale,      // [1]
    const float* __restrict__ weight_scale,   // [kN]
    const float* __restrict__ bias,           // [kN] or nullptr
    uint16_t* __restrict__ output,            // [kN]
    uint32_t* __restrict__ counter,           // [1], zero at launch, self-reset
    int dir) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int q = tid % kCOLS4;
  const int base = tid / kCOLS4;
  const uint32_t off0 = (uint32_t)(base * kCOLS4 + q);
  int dot[4] = {0, 0, 0, 0};
#pragma unroll
  for (int chunk = 0; chunk < 2; ++chunk) {
    int aq[6];
    uint4 vv[6];
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      const int pos = chunk * 6 + j;
      const int w = uw_wave_of(dir, pos);
      aq[j] = static_cast<int>(qx[(uint32_t)w * kUWStride + base]);
      vv[j] = reinterpret_cast<const uint4*>(weight_k4n)[(uint32_t)w * (kUWStride * kCOLS4) + off0];
    }
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      const int a = aq[j];
      dot[0] = sdot4(a, static_cast<int>(vv[j].x), dot[0]);
      dot[1] = sdot4(a, static_cast<int>(vv[j].y), dot[1]);
      dot[2] = sdot4(a, static_cast<int>(vv[j].z), dot[2]);
      dot[3] = sdot4(a, static_cast<int>(vv[j].w), dot[3]);
    }
  }
  // The same 4 columns are accumulated by all 128 bases (tid stride kCOLS4),
  // so the per-column addends form a fixed multiset; plain int32 atomicAdd is
  // order-independent and overflow-free (see static_assert), hence the result
  // is bit-exact whatever the arrival order.
  atomicAdd(&acc[4 * q + 0], dot[0]);
  atomicAdd(&acc[4 * q + 1], dot[1]);
  atomicAdd(&acc[4 * q + 2], dot[2]);
  atomicAdd(&acc[4 * q + 3], dot[3]);
  __threadfence();
  __syncthreads();

  // Arrival-counter rendezvous without spin: tid0's atomicAdd return value
  // kBlocks-1 proves every block's acc atomicAdds (released device-wide by
  // the per-thread __threadfence above, ordered before tid0's counter add by
  // __syncthreads) are globally visible, so exactly the last block reads a
  // complete acc; every other block exits immediately. Nothing ever waits on
  // a value that another block resets, so no poller can be stranded -- the
  // previous spin-on-counter==kBlocks rendezvous had a reset window that
  // could race a slow poller and hang the whole grid forever.
  __shared__ int is_last_block;
  if (threadIdx.x == 0) {
    is_last_block = (atomicAdd(counter, 1) == kBlocks - 1);
  }
  __syncthreads();
  if (is_last_block) {
    __threadfence();
    // Single read-back + scale + store + zero pass over all kN columns
    // (kEpilogueGroups int4 per thread); no cross-block read/clear overlap
    // exists because only this block touches acc in the epilogue.
    const float as = act_scale[0];
#pragma unroll
    for (int g = 0; g < kEpilogueGroups; ++g) {
      const int c0 = g * (4 * kThreads) + threadIdx.x * 4;
      const int4 s = *reinterpret_cast<const int4*>(acc + c0);
      const float ws0 = weight_scale[c0 + 0], ws1 = weight_scale[c0 + 1];
      const float ws2 = weight_scale[c0 + 2], ws3 = weight_scale[c0 + 3];
      if (bias != nullptr) {
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
    if (threadIdx.x == 0) counter[0] = 0;
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
  TORCH_CHECK(x.sizes() == at::IntArrayRef({1, kK}), "x must have shape [1, 6144]");
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

at::Tensor w8a8_gfx928_dense_gate_up_m1(
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
  const int dir = g_dir;
  g_dir = -g_dir;

  const dim3 quantize_grid(kQuantizeBlocks);
  const dim3 gridsync_grid(kBlocks);
  if (x.scalar_type() == at::kBFloat16) {
    hipLaunchKernelGGL(
        (quantize_global_kernel<true>), quantize_grid, dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(x.data_ptr()), reinterpret_cast<uint32_t*>(qx.data_ptr<int32_t>()),
        act_scale.data_ptr<float>());
    hipLaunchKernelGGL(
        (fused_gridsync_kernel<true>), gridsync_grid, dim3(kThreads), 0, stream,
        reinterpret_cast<const uint32_t*>(weight_k4n.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(qx.data_ptr<int32_t>()), acc.data_ptr<int32_t>(),
        act_scale.data_ptr<float>(), weight_scale.data_ptr<float>(), bias_pointer,
        reinterpret_cast<uint16_t*>(output.data_ptr()),
        reinterpret_cast<uint32_t*>(counter.data_ptr<int32_t>()), dir);
  } else {
    hipLaunchKernelGGL(
        (quantize_global_kernel<false>), quantize_grid, dim3(kThreads), 0, stream,
        reinterpret_cast<const uint16_t*>(x.data_ptr()), reinterpret_cast<uint32_t*>(qx.data_ptr<int32_t>()),
        act_scale.data_ptr<float>());
    hipLaunchKernelGGL(
        (fused_gridsync_kernel<false>), gridsync_grid, dim3(kThreads), 0, stream,
        reinterpret_cast<const uint32_t*>(weight_k4n.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(qx.data_ptr<int32_t>()), acc.data_ptr<int32_t>(),
        act_scale.data_ptr<float>(), weight_scale.data_ptr<float>(), bias_pointer,
        reinterpret_cast<uint16_t*>(output.data_ptr()),
        reinterpret_cast<uint32_t*>(counter.data_ptr<int32_t>()), dir);
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

#else

at::Tensor w8a8_gfx928_dense_gate_up_m1(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::optional<at::Tensor>&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "w8a8_gfx928_dense_gate_up_m1 is only available on gfx928");
}

#endif  // SGL_W8A8_GFX928_DENSE_GATE_UP_M1_BUILD
