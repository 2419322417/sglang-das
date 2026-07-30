# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.models.llama_eagle3 import LlamaModel
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _IdentityDraftNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, hidden_states


class TestMiniMaxEagle3(unittest.TestCase):
    def test_hcu_draft_extend_uses_new_accept_count_fields(self):
        draft_extend = EagleDraftExtendInput(
            hidden_states=torch.ones(2, 4),
            num_correct_drafts=torch.tensor([1], dtype=torch.int32),
            num_accept_tokens=torch.tensor([2], dtype=torch.int32),
            num_accept_tokens_cpu=[2],
            input_ids=torch.tensor([101, 102]),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
            req_pool_indices=torch.tensor([3]),
        )
        batch = SimpleNamespace(
            spec_info=draft_extend,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )

        with patch(
            "sglang.srt.speculative.eagle_info.hcu_create_extend_after_decode_spec_info"
        ) as create_spec_info:
            draft_extend.prepare_extend_after_decode(batch, speculative_num_steps=3)

        call = create_spec_info.call_args.kwargs
        self.assertIs(call["accept_lens"], draft_extend.num_accept_tokens)
        self.assertIs(call["new_verified_id"], draft_extend.bonus_tokens)
        self.assertEqual(batch.extend_lens, [2])

    def test_unquantized_eagle3_draft_does_not_inherit_w8a8(self):
        model_config = object.__new__(ModelConfig)
        model_config.is_draft_model = True
        model_config.quantization = "w8a8_int8"
        model_config.hf_config = SimpleNamespace(
            architectures=["LlamaForCausalLMEagle3"]
        )

        model_config._resolve_eagle3_draft_quantization()

        self.assertIsNone(model_config.quantization)

    def test_quantized_eagle3_draft_keeps_w8a8(self):
        model_config = object.__new__(ModelConfig)
        model_config.is_draft_model = True
        model_config.quantization = "w8a8_int8"
        model_config.hf_config = SimpleNamespace(
            architectures=["LlamaForCausalLMEagle3"],
            quantization_config={"quant_method": "w8a8_int8"},
        )

        model_config._resolve_eagle3_draft_quantization()

        self.assertEqual(model_config.quantization, "w8a8_int8")

    def test_aux_hidden_states_are_cast_to_draft_projection_dtype(self):
        model = object.__new__(LlamaModel)
        nn.Module.__init__(model)
        model.is_mrope_enabled = False
        model.fc = nn.Linear(6, 2, bias=False, dtype=torch.bfloat16)
        model.fc_norm = None
        model.layers = nn.ModuleList()
        model.norm = _IdentityDraftNorm()
        model.norm_output = False

        forward_batch = SimpleNamespace(
            spec_info=SimpleNamespace(
                hidden_states=torch.ones(1, 6, dtype=torch.float32)
            )
        )
        input_embeds = torch.ones(1, 2, dtype=torch.bfloat16)

        hidden_states, _ = model(
            input_ids=None,
            positions=torch.zeros(1, dtype=torch.int64),
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        self.assertEqual(hidden_states.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
