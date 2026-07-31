# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.test.ci.ci_register import register_hcu_ci

try:
    from .pd_hcu_utils import HcuPDServerBase, parse_backend_names
except ImportError:
    from pd_hcu_utils import HcuPDServerBase, parse_backend_names


register_hcu_ci(est_time=1200, suite="nightly-hcu", nightly=True)


def _make_backend_test(backend: str, index: int):
    class TestHcuPDTransferBackend(HcuPDServerBase):
        pd_backend = backend
        port_delta = 20 * (index + 1)

        def test_generate_smoke(self):
            self.assert_generate_smoke()

    TestHcuPDTransferBackend.__name__ = f"TestHcuPD{backend.title()}Transfer"
    TestHcuPDTransferBackend.__qualname__ = TestHcuPDTransferBackend.__name__
    TestHcuPDTransferBackend.__module__ = __name__
    return TestHcuPDTransferBackend


for _index, _backend in enumerate(parse_backend_names()):
    globals()[f"TestHcuPD{_backend.title()}Transfer"] = _make_backend_test(
        _backend,
        _index,
    )


if __name__ == "__main__":
    unittest.main()
