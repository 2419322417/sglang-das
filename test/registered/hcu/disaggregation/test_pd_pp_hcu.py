# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci

try:
    from .pd_hcu_utils import HcuPDServerBase
except ImportError:
    from pd_hcu_utils import HcuPDServerBase


register_hcu_ci(est_time=900, suite="nightly-hcu", nightly=True)


class TestHcuPDPipelineParallel(HcuPDServerBase):
    pd_backend = os.environ.get("SGLANG_HCU_PD_BACKEND", "mooncake")
    required_gpus = 6
    port_delta = 100
    prefill_tp = 2
    prefill_pp = 2
    decode_tp = 2
    decode_base_gpu_id = 4
    disable_overlap_schedule = True

    def test_generate_smoke(self):
        self.assert_generate_smoke()


if __name__ == "__main__":
    unittest.main()
