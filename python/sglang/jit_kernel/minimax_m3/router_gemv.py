# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


_M = 1
_K = 6144
_N = 128
_BLOCK_K = 512


@triton.jit
def _minimax_m3_router_gemv_kernel(
    hidden_states_ptr,
    weight_ptr,
    output_ptr,
    K_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for start in tl.static_range(0, K_SIZE, BLOCK_K):
        hidden_states = tl.load(hidden_states_ptr + start + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + expert * K_SIZE + start + offsets).to(
            tl.float32
        )
        accumulator += hidden_states * weight
    tl.store(output_ptr + expert, tl.sum(accumulator, axis=0))


def minimax_m3_router_gemv(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty((_M, _N), device=hidden_states.device, dtype=torch.float32)
    _minimax_m3_router_gemv_kernel[(_N,)](
        hidden_states,
        weight,
        output,
        K_SIZE=_K,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    return output
