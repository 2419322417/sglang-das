# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import os
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_utils import assert_generate_non_empty
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    get_rdma_devices_args,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_hcu_ci(
    est_time=1,
    suite="nightly-hcu",
    nightly=True,
    disabled="Support module only; covered by test_pd_hcu_utils.py.",
)


DEFAULT_HCU_PD_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)
SUPPORTED_BACKENDS = ("mooncake", "nixl", "mori")
BACKEND_MODULES = {
    "mooncake": "mooncake",
    "nixl": "nixl",
    "mori": "mori",
}


def parse_backend_names(value: str | None = None) -> list[str]:
    raw = (
        os.environ.get("SGLANG_HCU_PD_BACKENDS", "mooncake") if value is None else value
    )
    names = list(
        dict.fromkeys(item.strip().lower() for item in raw.split(",") if item.strip())
    )
    unknown = [name for name in names if name not in BACKEND_MODULES]
    if unknown:
        raise ValueError(
            f"Unsupported HCU PD backend(s): {unknown}; "
            f"expected one of {list(SUPPORTED_BACKENDS)}"
        )
    if not names:
        raise ValueError("At least one HCU PD backend must be selected.")
    return names


def backend_module_name(backend: str) -> str:
    try:
        return BACKEND_MODULES[backend.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported HCU PD backend: {backend}") from exc


def transfer_backend_available(backend: str) -> bool:
    return importlib.util.find_spec(backend_module_name(backend)) is not None


def require_transfer_backend(backend: str):
    module_name = backend_module_name(backend)
    if not transfer_backend_available(backend):
        raise unittest.SkipTest(
            f"HCU PD backend '{backend}' is blocked: Python module "
            f"'{module_name}' is not installed in this image."
        )
    try:
        return importlib.import_module(module_name)
    except BaseException as exc:
        raise AssertionError(
            f"HCU PD backend '{backend}' module '{module_name}' exists "
            f"but failed to import: {type(exc).__name__}: {exc}"
        ) from exc


def resolve_model_path() -> str:
    model_path = os.environ.get("SGLANG_HCU_PD_MODEL", DEFAULT_HCU_PD_MODEL)
    if not Path(model_path).is_dir():
        if "SGLANG_HCU_PD_MODEL" in os.environ:
            raise AssertionError(
                f"SGLANG_HCU_PD_MODEL points to a missing directory: {model_path}"
            )
        raise unittest.SkipTest(f"Default HCU PD model is not available: {model_path}")
    return model_path


def active_rdma_devices() -> list[str]:
    ib_root = Path("/sys/class/infiniband")
    if not ib_root.is_dir():
        return []

    devices = []
    for device in sorted(ib_root.iterdir()):
        state_file = device / "ports" / "1" / "state"
        try:
            state = state_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if "ACTIVE" in state.upper():
            devices.append(device.name)
    return devices


def resolve_rdma_args() -> list[str]:
    devices = os.environ.get("SGLANG_TEST_RDMA_DEVICE")
    if not devices:
        active_devices = active_rdma_devices()
        devices = (
            ",".join(active_devices) if active_devices else get_rdma_devices_args()
        )
    return ["--disaggregation-ib-device", devices] if devices else []


def require_hcu_devices(minimum: int) -> list[str]:
    try:
        import torch
    except BaseException as exc:
        raise unittest.SkipTest(f"DTK PyTorch is not importable: {exc}") from exc

    if not torch.cuda.is_available():
        raise unittest.SkipTest("DTK PyTorch reports no visible HCU devices.")
    count = torch.cuda.device_count()
    if count < minimum:
        raise unittest.SkipTest(
            f"HCU PD smoke requires at least {minimum} visible devices; found {count}."
        )
    return [torch.cuda.get_device_name(index) for index in range(count)]


class HcuPDServerBase(PDDisaggregationServerBase):
    pd_backend = "mooncake"
    required_gpus = 2
    port_delta = 0
    prefill_tp = 1
    prefill_pp = 1
    decode_tp = 1
    decode_base_gpu_id = 1
    disable_overlap_schedule = False

    @classmethod
    def setUpClass(cls):
        require_hcu_devices(cls.required_gpus)
        require_transfer_backend(cls.pd_backend)
        super().setUpClass()

        cls.model = resolve_model_path()
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.pd_backend,
        ]
        cls.rdma_devices = resolve_rdma_args()
        cls._shift_ports()

        try:
            cls.launch_all()
        except BaseException:
            super().tearDownClass()
            raise

    @classmethod
    def _shift_ports(cls):
        if not cls.port_delta:
            return
        cls.lb_port = str(int(cls.lb_port) + cls.port_delta)
        cls.prefill_port = str(int(cls.prefill_port) + cls.port_delta)
        cls.decode_port = str(int(cls.decode_port) + cls.port_delta)
        cls.bootstrap_port = str(int(cls.bootstrap_port) + cls.port_delta)
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url

    @classmethod
    def _common_server_args(cls, mode: str) -> list[str]:
        return [
            "--trust-remote-code",
            "--disaggregation-mode",
            mode,
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--attention-backend",
            "fa3",
            "--page-size",
            "64",
            "--log-level",
            "warning",
            "--log-level-http",
            "warning",
        ]

    @classmethod
    def start_prefill(cls):
        prefill_args = cls._common_server_args("prefill") + [
            "--tp-size",
            str(cls.prefill_tp),
        ]
        if cls.prefill_pp > 1:
            prefill_args += ["--pp-size", str(cls.prefill_pp)]
        if cls.disable_overlap_schedule:
            prefill_args.append("--disable-overlap-schedule")
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = cls._common_server_args("decode") + [
            "--tp-size",
            str(cls.decode_tp),
            "--base-gpu-id",
            str(cls.decode_base_gpu_id),
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def assert_generate_smoke(self):
        output = assert_generate_non_empty(
            self.lb_url,
            text="The capital of China is",
            max_new_tokens=8,
        )
        self.assertGreater(len(output.strip()), 0)
