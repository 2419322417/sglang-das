# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exact-shape gfx928 W8A8 shared-expert gate/up integration."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.patch_torch import register_fake_if_exists

_PREFIX_SUFFIX = "shared_experts.gate_up_proj"
_INPUT_SHAPE = (1, 6144)
_WEIGHT_SHAPE = (768, 6144)
_PACKED_SHAPE = (768, 768)
_OUTPUT_SHAPE = (1, 768)
_PACKED_BUFFER = "w8a8_gfx928_shared_gate_up_m1_k8n"
_SCALE_BUFFER = "w8a8_gfx928_shared_gate_up_m1_weight_scale"
_ACC_BUFFER = "w8a8_gfx928_shared_gate_up_m1_acc"
_COUNTER_BUFFER = "w8a8_gfx928_shared_gate_up_m1_counter"
_OP_NAME = "w8a8_gfx928_shared_gate_up_m1"

if torch.version.hip is not None:
    import sgl_kernel  # noqa: F401 -- registers the custom op

    @register_fake_if_exists(f"sgl_kernel::{_OP_NAME}")
    def _w8a8_gfx928_shared_gate_up_m1_fake(
        x,
        weight_k8n,
        weight_scale,
        bias,
        acc,
        counter,
    ):
        return x.new_empty(_OUTPUT_SHAPE)


def pack_signed_int8_k8n(weight_nk: torch.Tensor) -> torch.Tensor:
    """Pack signed INT8 ``[N,K]`` as little-endian INT64 ``[K/8,N]``."""
    if weight_nk.dtype != torch.int8 or tuple(weight_nk.shape) != _WEIGHT_SHAPE:
        raise ValueError("expected signed int8 checkpoint weight [768, 6144]")

    lanes = weight_nk.t().contiguous().reshape(768, 8, 768).to(torch.int64)
    packed = lanes[:, 0] & 0xFF
    for lane in range(1, 8):
        packed = packed | ((lanes[:, lane] & 0xFF) << (8 * lane))
    return packed.contiguous()


def _is_gfx928(device: torch.device) -> bool:
    if torch.version.hip is None or device.type != "cuda":
        return False
    arch = torch.cuda.get_device_properties(device).gcnArchName
    return arch.split(":", 1)[0] == "gfx928"


def initialize_shared_gate_up_m1(layer: torch.nn.Module) -> None:
    layer.register_buffer(_PACKED_BUFFER, None, persistent=False)
    layer.register_buffer(_SCALE_BUFFER, None, persistent=False)
    layer.register_buffer(_ACC_BUFFER, None, persistent=False)
    layer.register_buffer(_COUNTER_BUFFER, None, persistent=False)


def install_shared_gate_up_m1(
    layer: torch.nn.Module,
    prefix: str,
    *,
    compressed_dynamic_symmetric_channel: bool = True,
) -> bool:
    """Install layer-owned packed weight and workspace before weight transpose."""
    weight = layer.weight.data
    weight_scale = layer.weight_scale.data
    if (
        not envs.SGLANG_ROCM_GFX928_W8A8_SHARED_GATE_UP_M1.get()
        or not prefix.endswith(_PREFIX_SUFFIX)
        or tuple(weight.shape) != _WEIGHT_SHAPE
        or weight.dtype != torch.int8
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.float32
        or weight_scale.numel() != 768
        or not weight_scale.is_contiguous()
        or not compressed_dynamic_symmetric_channel
        or not _is_gfx928(weight.device)
    ):
        return False

    setattr(layer, _PACKED_BUFFER, pack_signed_int8_k8n(weight))
    setattr(layer, _SCALE_BUFFER, weight_scale.view(-1))
    setattr(
        layer,
        _ACC_BUFFER,
        torch.zeros(768, dtype=torch.int32, device=weight.device),
    )
    setattr(
        layer,
        _COUNTER_BUFFER,
        torch.zeros(1, dtype=torch.int32, device=weight.device),
    )
    return True


def try_shared_gate_up_m1(
    layer: torch.nn.Module,
    prefix: str,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return the specialized result, or ``None`` for the unchanged fallback.

    The native kernel self-resets ``acc`` and ``counter`` after each launch. They
    are layer-owned and therefore assume serialized use of a layer on one stream;
    concurrent forwards on different streams require external serialization.
    Finite activation values are an operator precondition. The selector performs
    metadata-only checks so it introduces no data-dependent host synchronization.
    """
    packed = layer.w8a8_gfx928_shared_gate_up_m1_k8n
    weight_scale = layer.w8a8_gfx928_shared_gate_up_m1_weight_scale
    acc = layer.w8a8_gfx928_shared_gate_up_m1_acc
    counter = layer.w8a8_gfx928_shared_gate_up_m1_counter
    if (
        not envs.SGLANG_ROCM_GFX928_W8A8_SHARED_GATE_UP_M1.get()
        or not prefix.endswith(_PREFIX_SUFFIX)
        or tuple(x.shape) != _INPUT_SHAPE
        or x.device.type != "cuda"
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_contiguous()
        or packed is None
        or weight_scale is None
        or acc is None
        or counter is None
        or not hasattr(torch.ops.sgl_kernel, _OP_NAME)
    ):
        return None

    if (
        tuple(packed.shape) != _PACKED_SHAPE
        or packed.dtype != torch.int64
        or not packed.is_contiguous()
        or weight_scale.dtype != torch.float32
        or weight_scale.numel() != 768
        or not weight_scale.is_contiguous()
        or tuple(acc.shape) != (768,)
        or acc.dtype != torch.int32
        or not acc.is_contiguous()
        or tuple(counter.shape) != (1,)
        or counter.dtype != torch.int32
        or not counter.is_contiguous()
        or packed.device != x.device
        or weight_scale.device != x.device
        or acc.device != x.device
        or counter.device != x.device
    ):
        return None
    if bias is not None and (
        bias.dtype != torch.float32
        or bias.numel() != 768
        or not bias.is_contiguous()
        or bias.device != x.device
    ):
        return None

    return torch.ops.sgl_kernel.w8a8_gfx928_shared_gate_up_m1(
        x,
        packed,
        weight_scale,
        bias,
        acc,
        counter,
    )
