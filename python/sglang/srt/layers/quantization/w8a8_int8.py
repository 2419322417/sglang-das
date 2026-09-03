# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import os
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, cast

import torch
from torch.nn.parameter import Parameter

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.rocm_w8a8_minimax_m3_moe import (
    initialize_minimax_m3_moe_m1,
    install_minimax_m3_moe_m1,
    try_minimax_m3_moe_m1,
)
from sglang.srt.layers.quantization.rocm_w8a8_shared_gate_up import (
    initialize_shared_gate_up_m1,
    install_shared_gate_up_m1,
    try_shared_gate_up_m1,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hcu,
    is_host_cpu_arm64,
    set_weight_attrs,
    use_intel_amx_backend,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

from lightop import gemm_ops as quant_ops

_is_cuda = is_cuda()
_is_hcu = is_hcu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_cpu_arm64 = is_host_cpu_arm64()

if _is_hcu:
    from lightop.quant import per_token_quant_int8

if _is_cuda:
    from sgl_kernel import int8_scaled_mm  # noqa: F401 -- registers the custom op

    @register_fake_if_exists("sgl_kernel::int8_scaled_mm")
    def _int8_scaled_mm_abstract(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias=None,
    ):
        M = mat_a.shape[-2]
        N = mat_b.shape[-1]
        return mat_a.new_empty((M, N), dtype=out_dtype)


logger = logging.getLogger(__name__)


class W8A8Int8Config(QuantizationConfig):
    """Config class for W8A8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        self.quant_description = quant_config
        self.is_dynamic = quant_config.get("is_dynamic", False)
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "w8a8_int8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = []
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Int8Config:
        return cls(config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.fp8 import Fp8KVCacheMethod
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedEmbeddingMethod()
            return W8A8Int8LinearMethod(self, prefix)
        elif isinstance(layer, FusedMoE):
            if should_ignore_layer(
                prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedFusedMoEMethod(
                    layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
                )
            return W8A8Int8MoEMethod(self, prefix)
        elif isinstance(layer, RadixAttention):
            if should_ignore_layer(
                prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
            ):
                return None
            else:
                return Fp8KVCacheMethod(self)
        return None

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = (
                    self.quant_description[shard_prefix + ".weight"] == "FLOAT"
                )

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision."
                    )
        else:
            is_skipped = self.quant_description[prefix + ".weight"] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


class W8A8Int8LinearMethod(LinearMethodBase):
    def __init__(self, quantization_config: W8A8Int8Config, prefix: str):
        self.quantization_config = quantization_config
        self.prefix = prefix

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu:
            if _is_cpu_amx_available:
                _amx_process_weight_after_loading(layer, ["weight"])
            elif _is_cpu_arm64:
                layer.weight = Parameter(layer.weight.data, requires_grad=False)
            else:
                assert False, "W8A8Int8LinearMethod on CPU only works on AMX or Arm64"
        else:
            install_shared_gate_up_m1(layer, self.prefix)
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)
        # PORT_CUSTOM_DEQUANT (sglang_full alignment): fold the per-output-
        # channel scale into the weight as bf16 -- dequant = int8.to(bf16) *
        # scale.to(bf16) (per-output-channel broadcast), exactly the W8A8
        # dequant formula of the minimal-framework dump producer
        # (trace/minimal_infer_w8a8_stream.py dequantize()).  The stored
        # weight is transposed to [in, out] by the line above (the int8 gemm
        # layout); the scale stays [out, 1], so the dequant transposes back
        # to the golden [out, in] F.linear layout.  apply() then runs a
        # plain bf16 F.linear (fp32 accumulate), the same arithmetic as the
        # sglang_full reference run.  Env-gated: the production int8
        # scaled-gemm path is untouched when the gate is off.
        if os.environ.get("PORT_CUSTOM_DEQUANT") == "1":
            layer._dequantized_weight = (
                layer.weight.data.to(torch.bfloat16).t()
                * layer.weight_scale.data.to(torch.bfloat16)
            ).contiguous()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        initialize_shared_gate_up_m1(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        if use_intel_amx_backend(layer) or _is_cpu_arm64:
            return torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                x,
                layer.weight,
                layer.weight_scale,
                bias,
                x.dtype,
                True,  # is_vnni
            )
        # PORT_CUSTOM_DEQUANT (sglang_full alignment): bf16-dequantized
        # weight + plain F.linear -- the arithmetic of the minimal-framework
        # reference run (see process_weights_after_loading).
        if getattr(layer, "_dequantized_weight", None) is not None:
            return torch.nn.functional.linear(x, layer._dequantized_weight, bias)
        specialized = try_shared_gate_up_m1(layer, self.prefix, x, bias)
        if specialized is not None:
            return specialized
        x_q, x_scale = per_token_quant_int8(x)

        x_q_2d = x_q.view(-1, x_q.shape[-1])
        x_scale_2d = x_scale.view(-1, x_scale.shape[-1])
        output_shape = [*x_q.shape[:-1], layer.weight.shape[1]]

        output = quant_ops.triton_scaled_mm(
            x_q_2d,
            layer.weight,
            x_scale_2d,
            layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )

        return output.view(output_shape)


def _dequant_moe_param(layer: torch.nn.Module, w_name: str, s_name: str) -> None:
    """PORT_CUSTOM_DEQUANT helper: replace an int8 MoE weight param with its
    bf16 dequantization (int8.to(bf16) * scale.to(bf16), per-output-channel
    broadcast), chunked per expert to bound peak memory."""
    src = getattr(layer, w_name).data  # [E, ..., out_ch] int8
    scale = getattr(layer, s_name).data  # [E, ..., 1] fp32
    out = torch.empty(src.shape, dtype=torch.bfloat16, device=src.device)
    for e in range(src.shape[0]):
        out[e] = src[e].to(torch.bfloat16) * scale[e].to(torch.bfloat16)
    setattr(layer, w_name, torch.nn.Parameter(out, requires_grad=False))


class W8A8Int8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: W8A8Int8Config, prefix: str):
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)
        initialize_minimax_m3_moe_m1(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_hcu and self.runner.runner_backend.is_lightop():
            from sglang.srt.layers.moe.moe_runner.lightop import (
                process_weights_after_loading_lightop,
            )

            process_weights_after_loading_lightop(layer)
        elif _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
        else:
            layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
            layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        # PORT_CUSTOM_DEQUANT (sglang_full alignment): fold per-output-channel
        # scales into bf16 expert weights (dequant = int8.to(bf16) * scale.to(
        # bf16), per-channel broadcast) -- same formula as the minimal-
        # framework dump producer; get_triton_quant_info then hands the
        # standard bf16 (unquantized) FusedMoE kernel these weights, matching
        # the sglang_full reference arithmetic (bf16 gemm, fp32 accumulate).
        # The int8 params are REPLACED in place, chunked per expert (the full
        # [128, 2*3072, 6144] bf16 temp is ~9.7 GiB and OOMs the 64GB card
        # with the ~31GB model resident; per-expert chunks peak at ~0.25 GiB
        # and free the int8 storage right away).
        if os.environ.get("PORT_CUSTOM_DEQUANT") == "1":
            _dequant_moe_param(layer, "w13_weight", "w13_weight_scale")
            _dequant_moe_param(layer, "w2_weight", "w2_weight_scale")
        else:
            config = self.moe_runner_config
            install_minimax_m3_moe_m1(
                layer,
                self.prefix,
                runner_backend=self.runner.runner_backend,
                activation=config.activation,
                is_gated=config.is_gated,
                apply_router_weight_on_input=config.apply_router_weight_on_input,
                inplace=config.inplace,
                no_combine=config.no_combine,
                routed_scaling_factor=config.routed_scaling_factor,
                gemm1_alpha=config.gemm1_alpha,
                gemm1_clamp_limit=config.gemm1_clamp_limit,
                gate_up_interleaved=config.gate_up_interleaved,
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):

        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto():
            moe_runner_backend = MoeRunnerBackend.TRITON

        if moe_runner_backend.is_aiter() and _is_hcu:
            self.runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)
        elif moe_runner_backend.is_lightop() and _is_hcu:
            self.runner = MoeRunner(MoeRunnerBackend.LIGHTOP, moe_runner_config)
        elif moe_runner_backend.is_triton():
            self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)
        else:
            raise ValueError(f"Unsupported MoE runner backend: {moe_runner_backend}")

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        # PORT_CUSTOM_DEQUANT (sglang_full alignment): standard bf16
        # (unquantized) quant info on the dequantized (bf16-replaced) weights
        # -- same kernel and arithmetic as the sglang_full bf16 reference run.
        if (
            os.environ.get("PORT_CUSTOM_DEQUANT") == "1"
            and layer.w13_weight.dtype == torch.bfloat16
        ):
            return TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_int8_w8a8=False,
                per_channel_quant=False,
                w13_scale=None,
                w2_scale=None,
                a13_scale=None,
                a2_scale=None,
            )
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        if use_intel_amx_backend(layer) or _is_cpu_arm64:
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            topk_ids = topk_ids.int()
            x, topk_weights = apply_topk_weights_cpu(
                self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.INT8_W8A8,
                layer.w13_weight_scale,  # w1_scale
                layer.w2_weight_scale,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                None,  # w1 bias
                None,  # w3 bias
                None,  # alpha
                None,  # limit
                True,  # is_vnni
            )
            if bias is not None:
                output = output + bias
            return StandardCombineInput(hidden_states=output)

        topk_weights, topk_ids, _ = topk_output
        specialized = try_minimax_m3_moe_m1(
            layer,
            self.prefix,
            x,
            dispatch_output.hidden_states_scale,
            topk_ids,
            topk_weights,
            bias,
        )
        if specialized is not None:
            return StandardCombineInput(hidden_states=specialized)

        quant_info = self.get_triton_quant_info(layer)

        if _is_hcu and self.runner.runner_backend.is_aiter():
            from sglang.srt.layers.moe.moe_runner.aiter import (
                get_aiter_w8a8_int8_quant_info,
            )

            quant_info = get_aiter_w8a8_int8_quant_info(layer)
        elif _is_hcu and self.runner.runner_backend.is_lightop():
            from sglang.srt.layers.moe.moe_runner.lightop import get_lightop_quant_info

            quant_info = get_lightop_quant_info(layer)

        combine_input = self.runner.run(dispatch_output, quant_info)
        if bias is not None:
            return StandardCombineInput(
                hidden_states=combine_input.hidden_states + bias
            )
        return combine_input
