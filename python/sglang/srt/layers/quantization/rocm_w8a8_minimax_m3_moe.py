# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-shape gfx928 MiniMax M3 W8A8 MoE decode integration."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils.patch_torch import register_fake_if_exists

_PREFIX_SUFFIX = ".experts"
_OP_NAME = "w8a8_gfx928_minimax_m3_moe_m1"
_EXPERTS = 129
_ROUTES = 5
_HIDDEN = 6144
_INTERMEDIATE = 384
_GATE_UP = 768
_WORKSPACE_NAMES = (
    "w8a8_gfx928_moe_m1_x_q",
    "w8a8_gfx928_moe_m1_x_scale",
    "w8a8_gfx928_moe_m1_gate_up",
    "w8a8_gfx928_moe_m1_activated",
    "w8a8_gfx928_moe_m1_activated_q",
    "w8a8_gfx928_moe_m1_activated_scale",
    "w8a8_gfx928_moe_m1_route_output",
)

if torch.version.hip is not None:
    import sgl_kernel  # noqa: F401 -- registers the custom op

    @register_fake_if_exists(f"sgl_kernel::{_OP_NAME}")
    def _w8a8_gfx928_minimax_m3_moe_m1_fake(
        x,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        x_q,
        x_scale,
        gate_up,
        activated,
        activated_q,
        activated_scale,
        route_output,
        alpha,
        limit,
    ):
        return x

try:
    _MOE_OP = torch.ops.sgl_kernel.w8a8_gfx928_minimax_m3_moe_m1
except AttributeError:
    _MOE_OP = None


def _is_gfx928(device: torch.device) -> bool:
    if torch.version.hip is None or device.type != "cuda":
        return False
    return torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0] == "gfx928"


def initialize_minimax_m3_moe_m1(layer: torch.nn.Module) -> None:
    for name in _WORKSPACE_NAMES:
        layer.register_buffer(name, None, persistent=False)


def install_minimax_m3_moe_m1(
    layer: torch.nn.Module,
    prefix: str,
    *,
    runner_backend: MoeRunnerBackend,
    activation: str,
    is_gated: bool,
    apply_router_weight_on_input: bool,
    inplace: bool,
    no_combine: bool,
    routed_scaling_factor: Optional[float],
    gemm1_alpha: Optional[float],
    gemm1_clamp_limit: Optional[float],
    gate_up_interleaved: bool,
) -> bool:
    weight = layer.w13_weight
    device = weight.device
    eligible = (
        envs.SGLANG_ROCM_GFX928_W8A8_SHARED_GATE_UP_M1.get()
        and prefix.endswith(_PREFIX_SUFFIX)
        and runner_backend.is_triton()
        and layer.moe_tp_size == 8
        and layer.moe_ep_size == 1
        and layer.num_local_experts == _EXPERTS
        and layer.num_fused_shared_experts == 1
        and layer.hidden_size == _HIDDEN
        and layer.intermediate_size_per_partition == _INTERMEDIATE
        and tuple(layer.w13_weight.shape) == (_EXPERTS, _GATE_UP, _HIDDEN)
        and layer.w13_weight.dtype == torch.int8
        and layer.w13_weight.is_contiguous()
        and tuple(layer.w2_weight.shape) == (_EXPERTS, _HIDDEN, _INTERMEDIATE)
        and layer.w2_weight.dtype == torch.int8
        and layer.w2_weight.is_contiguous()
        and tuple(layer.w13_weight_scale.shape) == (_EXPERTS, _GATE_UP, 1)
        and layer.w13_weight_scale.dtype == torch.float32
        and layer.w13_weight_scale.is_contiguous()
        and tuple(layer.w2_weight_scale.shape) == (_EXPERTS, _HIDDEN, 1)
        and layer.w2_weight_scale.dtype == torch.float32
        and layer.w2_weight_scale.is_contiguous()
        and activation == "silu"
        and is_gated
        and not apply_router_weight_on_input
        and inplace
        and not no_combine
        and routed_scaling_factor is None
        and gemm1_alpha == 1.702
        and gemm1_clamp_limit == 7.0
        and not gate_up_interleaved
        and _MOE_OP is not None
        and _is_gfx928(device)
    )
    if not eligible:
        return False

    layer.w8a8_gfx928_moe_m1_x_q = torch.empty(_HIDDEN, dtype=torch.int8, device=device)
    layer.w8a8_gfx928_moe_m1_x_scale = torch.empty(1, dtype=torch.float32, device=device)
    layer.w8a8_gfx928_moe_m1_gate_up = torch.empty(
        (_ROUTES, _GATE_UP), dtype=layer.moe_runner_config.params_dtype, device=device
    )
    layer.w8a8_gfx928_moe_m1_activated = torch.empty(
        (_ROUTES, _INTERMEDIATE), dtype=layer.moe_runner_config.params_dtype, device=device
    )
    layer.w8a8_gfx928_moe_m1_activated_q = torch.empty(
        (_ROUTES, _INTERMEDIATE), dtype=torch.int8, device=device
    )
    layer.w8a8_gfx928_moe_m1_activated_scale = torch.empty(
        _ROUTES, dtype=torch.float32, device=device
    )
    layer.w8a8_gfx928_moe_m1_route_output = torch.empty(
        (_ROUTES, _HIDDEN), dtype=layer.moe_runner_config.params_dtype, device=device
    )
    return True


def try_minimax_m3_moe_m1(
    layer: torch.nn.Module,
    prefix: str,
    x: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    x_q = layer.w8a8_gfx928_moe_m1_x_q
    x_scale = layer.w8a8_gfx928_moe_m1_x_scale
    gate_up = layer.w8a8_gfx928_moe_m1_gate_up
    activated = layer.w8a8_gfx928_moe_m1_activated
    activated_q = layer.w8a8_gfx928_moe_m1_activated_q
    activated_scale = layer.w8a8_gfx928_moe_m1_activated_scale
    route_output = layer.w8a8_gfx928_moe_m1_route_output
    if (
        not envs.SGLANG_ROCM_GFX928_W8A8_SHARED_GATE_UP_M1.get()
        or not prefix.endswith(_PREFIX_SUFFIX)
        or hidden_states_scale is not None
        or bias is not None
        or tuple(x.shape) != (1, _HIDDEN)
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_contiguous()
        or tuple(topk_ids.shape) != (1, _ROUTES)
        or topk_ids.dtype != torch.int32
        or not topk_ids.is_contiguous()
        or tuple(topk_weights.shape) != (1, _ROUTES)
        or topk_weights.dtype != torch.float32
        or not topk_weights.is_contiguous()
        or x_q is None
        or x_scale is None
        or gate_up is None
        or activated is None
        or activated_q is None
        or activated_scale is None
        or route_output is None
        or tuple(x_q.shape) != (_HIDDEN,)
        or x_q.dtype != torch.int8
        or not x_q.is_contiguous()
        or tuple(x_scale.shape) != (1,)
        or x_scale.dtype != torch.float32
        or not x_scale.is_contiguous()
        or tuple(gate_up.shape) != (_ROUTES, _GATE_UP)
        or not gate_up.is_contiguous()
        or tuple(activated.shape) != (_ROUTES, _INTERMEDIATE)
        or not activated.is_contiguous()
        or tuple(activated_q.shape) != (_ROUTES, _INTERMEDIATE)
        or activated_q.dtype != torch.int8
        or not activated_q.is_contiguous()
        or tuple(activated_scale.shape) != (_ROUTES,)
        or activated_scale.dtype != torch.float32
        or not activated_scale.is_contiguous()
        or tuple(route_output.shape) != (_ROUTES, _HIDDEN)
        or not route_output.is_contiguous()
        or _MOE_OP is None
    ):
        return None

    if (
        x.dtype != gate_up.dtype
        or x.dtype != activated.dtype
        or x.dtype != route_output.dtype
        or x.device != layer.w13_weight.device
        or x.device != layer.w2_weight.device
        or x.device != layer.w13_weight_scale.device
        or x.device != layer.w2_weight_scale.device
        or x.device != topk_ids.device
        or x.device != topk_weights.device
        or x.device != x_q.device
        or x.device != x_scale.device
        or x.device != gate_up.device
        or x.device != activated.device
        or x.device != activated_q.device
        or x.device != activated_scale.device
        or x.device != route_output.device
    ):
        return None

    return _MOE_OP(
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
        topk_ids,
        topk_weights,
        x_q,
        x_scale,
        gate_up,
        activated,
        activated_q,
        activated_scale,
        route_output,
        1.702,
        7.0,
    )
