"""MiniMax-H3 scale/shift kernels for the SGLang 0.5.15 kernel layout.

The 0.5.15 DCU tree already contains the DCU-oriented implementation under
``sglang.jit_kernel.diffusion.triton.scale_shift``.  MiniMax-H3's VAE imports
from the newer ``sglang.kernels.ops`` path and additionally needs
``try_fused_scaled_residual_add_exact``.

Keep the existing 0.5.15 implementations as the source for the older fused
scale/shift and layer-norm kernels, and add only the missing compatibility
implementation here.  This avoids replacing the existing DCU kernels with a
CUDA-only implementation.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.jit_kernel.diffusion.triton.scale_shift import (
    fuse_layernorm_scale_shift_gate_select01_kernel,
    fuse_residual_layernorm_scale_shift_gate_select01_kernel,
    fuse_scale_shift_kernel,
    fuse_scale_shift_kernel_blc_opt,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


@triton.jit
def _fp32_mul_add_rn(x, scale, residual):
    """Perform separate FP32 multiply and add operations with round-to-nearest."""
    return tl.inline_asm_elementwise(
        asm="""{
            .reg .f32 product;
            mul.rn.f32 product, $1, $2;
            add.rn.f32 $0, $3, product;
        }""",
        constraints="=f,f,f,f",
        args=(x, scale, residual),
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fused_scaled_residual_add_exact_kernel(
    output_ptr,
    residual_ptr,
    x_ptr,
    scale_ptr,
    numel: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute ``output = residual + x * scale`` element by element."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x_value = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    scale_value = tl.load(scale_ptr + offsets % width, mask=mask)
    residual_value = tl.load(residual_ptr + offsets, mask=mask)
    output = _fp32_mul_add_rn(x_value, scale_value, residual_value)
    tl.store(output_ptr + offsets, output, mask=mask)


def try_fused_scaled_residual_add_exact(
    residual: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor | None:
    """Try the exact fused residual add and return ``None`` when unsupported.

    The MiniMax-H3 VAE caller intentionally falls back to eager PyTorch when
    this function returns ``None``.  On DCU, ``current_platform.is_cuda()``
    normally rejects this CUDA inline-assembly kernel, so the existing eager
    path remains the safe default.
    """
    if (
        not current_platform.is_cuda()
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or residual.dtype != torch.float32
        or x.dtype not in (torch.float16, torch.bfloat16)
        or scale.dtype != torch.float32
        or not residual.is_cuda
        or residual.device != x.device
        or residual.device != scale.device
        or residual.shape != x.shape
        or scale.shape != (x.shape[-1],)
        or not residual.is_contiguous()
        or not x.is_contiguous()
        or not scale.is_contiguous()
        or x.numel() == 0
    ):
        return None

    output = torch.empty_like(residual)
    block_size = 1024
    _fused_scaled_residual_add_exact_kernel[
        (triton.cdiv(x.numel(), block_size),)
    ](
        output,
        residual,
        x,
        scale,
        numel=x.numel(),
        width=x.shape[-1],
        BLOCK_SIZE=block_size,
    )
    return output


__all__ = [
    "try_fused_scaled_residual_add_exact",
    "fuse_scale_shift_kernel_blc_opt",
    "fuse_scale_shift_kernel",
    "fuse_layernorm_scale_shift_gate_select01_kernel",
    "fuse_residual_layernorm_scale_shift_gate_select01_kernel",
]
