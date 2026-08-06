# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass

MINIMAX_M27_DEFAULT_MODEL_PATH = (
    "/public/opendas/DL_DATA/llm-models/MiniMax-M2.7-Channel-FP8-w8a8"
)
MINIMAX_M27_MODEL_ENV = "SGLANG_HCU_MINIMAX_M27_MODEL"

PREFILL_PORT = 30000
DECODE_PORT = 30001
ROUTER_PORT = 30002
BOOTSTRAP_PORT = 8998

MINIMAX_M27_COMMON_ENV = {
    "SGLANG_USE_MODELSCOPE": "1",
    "USE_HCU_CUSTOM_ALLREDUCE": "1",
    "SGLANG_USE_AITER_AR": "1",
    "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD": "0",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=0x40000",
    "SGLANG_USE_LIGHTOP": "1",
    "VLLM_USE_LIGHTOP_MOE_ALIGN": "1",
    "LMSLIM_USE_LIGHTOP": "1",
    "SGLANG_KVALLOC_KERNEL": "1",
    "SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO": "1",
    "SGLANG_ASSIGN_EXTEND_CACHE_LOCS": "1",
    "SGLANG_ASSIGN_REQ_TO_TOKEN_POOL": "1",
    "SGLANG_GET_LAST_LOC": "1",
    "SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON": "1",
    "SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES": "1",
    "ALLREDUCE_STREAM_WITH_COMPUTE": "1",
}


@dataclass(frozen=True)
class HcuPDRoleConfig:
    role: str
    host_ip: str
    ifname: str
    ib_device: str

    def __post_init__(self) -> None:
        if self.role not in {"prefill", "decode"}:
            raise ValueError(f"unsupported HCU PD role: {self.role}")
        for field_name in ("host_ip", "ifname", "ib_device"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")

    @property
    def port(self) -> int:
        return PREFILL_PORT if self.role == "prefill" else DECODE_PORT


def resolve_minimax_m27_model_path() -> str:
    return os.environ.get(MINIMAX_M27_MODEL_ENV, MINIMAX_M27_DEFAULT_MODEL_PATH).rstrip(
        "/"
    )


def minimax_m27_pd_env(
    role: HcuPDRoleConfig, *, gid_index: str = "3"
) -> dict[str, str]:
    env = dict(MINIMAX_M27_COMMON_ENV)
    env.update(
        {
            "SGLANG_HOST_IP": role.host_ip,
            "NCCL_SOCKET_IFNAME": role.ifname,
            "GLOO_SOCKET_IFNAME": role.ifname,
            "NCCL_IB_HCA": role.ib_device,
            "MC_GID_INDEX": gid_index,
        }
    )
    return env


def _common_server_args(role: HcuPDRoleConfig, model_path: str) -> list[str]:
    return [
        "--model-path",
        model_path,
        "--quantization",
        "w8a8_fp8",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--trust-remote-code",
        "--page-size",
        "64",
        "--dtype",
        "bfloat16",
        "--tool-call-parser",
        "minimax-m2",
        "--reasoning-parser",
        "minimax-append-think",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "fa3",
        "--numa-node",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "--max-running-requests",
        "512",
        "--context-length",
        "131072",
        "--watchdog-timeout",
        "3600",
        "--disaggregation-mode",
        role.role,
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        str(BOOTSTRAP_PORT),
        "--disaggregation-ib-device",
        role.ib_device,
        "--host",
        role.host_ip,
        "--port",
        str(role.port),
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]


def minimax_m27_server_args(role: HcuPDRoleConfig, model_path: str) -> list[str]:
    args = _common_server_args(role, model_path)
    if role.role == "prefill":
        args.extend(
            [
                "--tp-size",
                "2",
                "--pp-size",
                "4",
                "--dp-size",
                "1",
                "--chunked-prefill-size",
                "4096",
                "--load-balance-method",
                "round_robin",
            ]
        )
    else:
        args.extend(
            [
                "--tp-size",
                "8",
                "--pp-size",
                "1",
                "--dp-size",
                "1",
                "--prefill-round-robin-balance",
            ]
        )
    return args


def minimax_m27_server_command(role: HcuPDRoleConfig, model_path: str) -> list[str]:
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        *minimax_m27_server_args(role, model_path),
    ]


def minimax_m27_router_command(prefill_ip: str, decode_ip: str) -> list[str]:
    return [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--prefill",
        f"http://{prefill_ip}:{PREFILL_PORT}",
        str(BOOTSTRAP_PORT),
        "--decode",
        f"http://{decode_ip}:{DECODE_PORT}",
        "--policy",
        "cache_aware",
        "--host",
        prefill_ip,
        "--port",
        str(ROUTER_PORT),
    ]
