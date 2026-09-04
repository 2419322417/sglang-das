# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exact-shape gfx928 W8A8 self_attn.o_proj M=1 integration."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.patch_torch import register_fake_if_exists

_PREFIX_SUFFIX = "self_attn.o_proj"
_INPUT_SHAPE = (1, 1024)
_WEIGHT_SHAPE = (6144, 1024)
_PACKED_SHAPE = (192, 4096)
_OUTPUT_SHAPE = (1, 6144)
_PACKED_BUFFER = "w8a8_gfx928_o_proj_m1_tile"
_SCALE_BUFFER = "w8a8_gfx928_o_proj_m1_weight_scale"
_OP_NAME = "w8a8_gfx928_o_proj_m1"

if torch.version.hip is not None:
    import sgl_kernel  # noqa: F401 -- registers the custom op

    @register_fake_if_exists(f"sgl_kernel::{_OP_NAME}")
    def _w8a8_gfx928_o_proj_m1_fake(
        x,
        weight_tile,
        weight_scale,
        bias,
    ):
        return x.new_empty(_OUTPUT_SHAPE)


def pack_signed_int8_tile(weight_nk: torch.Tensor) -> torch.Tensor:
    """Pack signed INT8 ``[N,K]`` as the layout-3 uint64 tile ``[N/32][K/8*32]``.

    Word index ``(c/32)*(k8n*32) + k8*32 + (c%32)`` with ``k8n = K/8``; word
    ``(k8, c)`` holds rows ``8*k8..8*k8+7`` of column ``c`` as little-endian
    INT64.
    """
    if weight_nk.dtype != torch.int8 or tuple(weight_nk.shape) != _WEIGHT_SHAPE:
        raise ValueError("expected signed int8 checkpoint weight [6144, 1024]")

    k8n = _WEIGHT_SHAPE[1] // 8
    lanes = (
        weight_nk.t()
        .contiguous()
        .reshape(k8n, 8, _WEIGHT_SHAPE[0])
        .to(torch.int64)
    )
    k8n_packed = lanes[:, 0] & 0xFF
    for lane in range(1, 8):
        k8n_packed = k8n_packed | ((lanes[:, lane] & 0xFF) << (8 * lane))
    return (
        k8n_packed.reshape(k8n, _WEIGHT_SHAPE[0] // 32, 32)
        .permute(1, 0, 2)
        .reshape(_WEIGHT_SHAPE[0] // 32, k8n * 32)
        .contiguous()
    )


def _is_gfx928(device: torch.device) -> bool:
    if torch.version.hip is None or device.type != "cuda":
        return False
    arch = torch.cuda.get_device_properties(device).gcnArchName
    return arch.split(":", 1)[0] == "gfx928"


def initialize_o_proj_m1(layer: torch.nn.Module) -> None:
    layer.register_buffer(_PACKED_BUFFER, None, persistent=False)
    layer.register_buffer(_SCALE_BUFFER, None, persistent=False)


def install_o_proj_m1(
    layer: torch.nn.Module,
    prefix: str,
    *,
    compressed_dynamic_symmetric_channel: bool = True,
) -> bool:
    """Install layer-owned packed weight before weight transpose."""
    weight = layer.weight.data
    weight_scale = layer.weight_scale.data
    if (
        not envs.SGLANG_ROCM_GFX928_W8A8_O_PROJ_M1.get()
        or not prefix.endswith(_PREFIX_SUFFIX)
        or tuple(weight.shape) != _WEIGHT_SHAPE
        or weight.dtype != torch.int8
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.float32
        or weight_scale.numel() != 6144
        or not weight_scale.is_contiguous()
        or not compressed_dynamic_symmetric_channel
        or not _is_gfx928(weight.device)
    ):
        return False

    setattr(layer, _PACKED_BUFFER, pack_signed_int8_tile(weight))
    setattr(layer, _SCALE_BUFFER, weight_scale.view(-1))
    return True


def try_o_proj_m1(
    layer: torch.nn.Module,
    prefix: str,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return the specialized result, or ``None`` for the unchanged fallback.

    Single-launch kernel with no cross-block state and no mutable workspace,
    so it is safe under stream capture and concurrent layer use. Finite
    activation values are an operator precondition. The selector performs
    metadata-only checks so it introduces no data-dependent host
    synchronization.
    """
    packed = layer.w8a8_gfx928_o_proj_m1_tile
    weight_scale = layer.w8a8_gfx928_o_proj_m1_weight_scale
    if (
        not envs.SGLANG_ROCM_GFX928_W8A8_O_PROJ_M1.get()
        or not prefix.endswith(_PREFIX_SUFFIX)
        or tuple(x.shape) != _INPUT_SHAPE
        or x.device.type != "cuda"
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_contiguous()
        or packed is None
        or weight_scale is None
        or not hasattr(torch.ops.sgl_kernel, _OP_NAME)
    ):
        return None

    if (
        tuple(packed.shape) != _PACKED_SHAPE
        or packed.dtype != torch.int64
        or not packed.is_contiguous()
        or weight_scale.dtype != torch.float32
        or weight_scale.numel() != 6144
        or not weight_scale.is_contiguous()
        or packed.device != x.device
        or weight_scale.device != x.device
    ):
        return None
    if bias is not None and (
        bias.dtype != torch.float32
        or bias.numel() != 6144
        or not bias.is_contiguous()
        or bias.device != x.device
    ):
        return None

    return torch.ops.sgl_kernel.w8a8_gfx928_o_proj_m1(
        x,
        packed,
        weight_scale,
        bias,
    )
