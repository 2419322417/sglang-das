# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci

try:
    from .pd_hcu_utils import HcuPDServerBase
except ImportError:
    from pd_hcu_utils import HcuPDServerBase


register_hcu_ci(est_time=600, suite="nightly-hcu", nightly=True)


class TestHcuPDBasic(HcuPDServerBase):
    pd_backend = os.environ.get("SGLANG_HCU_PD_BACKEND", "mooncake")

    def test_generate_smoke(self):
        self.assert_generate_smoke()


if __name__ == "__main__":
    unittest.main()
