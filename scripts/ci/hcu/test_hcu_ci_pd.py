#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from hcu_ci_pd import (
    BOOTSTRAP_PORT,
    DECODE_PORT,
    PREFILL_PORT,
    ROLE_DECODE,
    ROLE_PREFILL,
    ROUTER_PORT,
    HcuPDRoleConfig,
    PDInfrastructureError,
    PDOrchestrator,
    PDRunContext,
    atomic_write_json,
    ensure_shared_dir,
    minimax_m27_pd_env,
    minimax_m27_router_command,
    minimax_m27_server_args,
    read_json,
    write_json_once,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

TEST_SHA = "1" * 40


def _shared_gid() -> int:
    return os.getgid() if hasattr(os, "getgid") else 0


def _context(root: Path, role: str = ROLE_PREFILL) -> PDRunContext:
    prefill_ip = "12.12.12.4"
    decode_ip = "12.12.12.36"
    return PDRunContext(
        role=role,
        run_id="1234",
        attempt="1",
        sha=TEST_SHA,
        target_ref="test-ref",
        runner_name=f"runner-{role}",
        hostname=f"host-{role}",
        image="registry/sglang:test",
        image_id="sha256:test",
        model_path="/models/MiniMax-M2.7",
        local_ip=prefill_ip if role == ROLE_PREFILL else decode_ip,
        peer_ip=decode_ip if role == ROLE_PREFILL else prefill_ip,
        prefill_ip=prefill_ip,
        decode_ip=decode_ip,
        ifname="eth0",
        ib_device="mlx5_0",
        gid_index="3",
        checkout=REPO_ROOT,
        shared_root=root / "hcu-pd",
        wheel_root=root / "hcu-wheels",
        shared_gid=_shared_gid(),
        peer_timeout=5,
        service_timeout=5,
        heartbeat_timeout=2,
        completion_timeout=5,
    )


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            body = b"OK"
        elif self.path == "/model_info":
            body = json.dumps({"model_path": "/models/MiniMax-M2.7"}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args) -> None:
        return


class TestHcuPDUtils(unittest.TestCase):
    def test_role_specific_server_topology(self) -> None:
        prefill = HcuPDRoleConfig(
            role=ROLE_PREFILL,
            host_ip="12.12.12.4",
            ifname="ens47f0np0",
            ib_device="mlx5_6",
        )
        decode = HcuPDRoleConfig(
            role=ROLE_DECODE,
            host_ip="12.12.12.36",
            ifname="eth0",
            ib_device="mlx5_0",
        )

        prefill_args = minimax_m27_server_args(prefill, "/models/m27")
        decode_args = minimax_m27_server_args(decode, "/models/m27")

        self.assertEqual(prefill_args[prefill_args.index("--tp-size") + 1], "2")
        self.assertEqual(prefill_args[prefill_args.index("--pp-size") + 1], "4")
        self.assertEqual(
            prefill_args[prefill_args.index("--port") + 1], str(PREFILL_PORT)
        )
        self.assertEqual(decode_args[decode_args.index("--tp-size") + 1], "8")
        self.assertEqual(decode_args[decode_args.index("--pp-size") + 1], "1")
        self.assertEqual(decode_args[decode_args.index("--port") + 1], str(DECODE_PORT))
        self.assertEqual(
            prefill_args[prefill_args.index("--disaggregation-ib-device") + 1],
            "mlx5_6",
        )
        self.assertEqual(
            decode_args[decode_args.index("--disaggregation-ib-device") + 1],
            "mlx5_0",
        )
        self.assertEqual(
            prefill_args[prefill_args.index("--watchdog-timeout") + 1], "3600"
        )
        self.assertEqual(
            decode_args[decode_args.index("--watchdog-timeout") + 1], "3600"
        )

    def test_role_network_environment(self) -> None:
        role = HcuPDRoleConfig(
            role=ROLE_PREFILL,
            host_ip="12.12.12.4",
            ifname="ens47f0np0",
            ib_device="mlx5_6",
        )
        env = minimax_m27_pd_env(role)
        self.assertEqual(env["SGLANG_HOST_IP"], "12.12.12.4")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "ens47f0np0")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "ens47f0np0")
        self.assertEqual(env["NCCL_IB_HCA"], "mlx5_6")
        self.assertEqual(env["MC_GID_INDEX"], "3")

    def test_router_contains_both_roles_and_bootstrap_port(self) -> None:
        command = minimax_m27_router_command("12.12.12.4", "12.12.12.36")
        self.assertIn(f"http://12.12.12.4:{PREFILL_PORT}", command)
        self.assertIn(f"http://12.12.12.36:{DECODE_PORT}", command)
        self.assertIn(str(BOOTSTRAP_PORT), command)
        self.assertEqual(command[command.index("--port") + 1], str(ROUTER_PORT))


class TestSharedState(unittest.TestCase):
    def test_atomic_json_and_first_writer_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "state" / "result.json"
            atomic_write_json(path, {"value": 1}, _shared_gid())
            self.assertEqual(read_json(path), {"value": 1})
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o664)

            once = root / "state" / "abort.json"
            self.assertTrue(write_json_once(once, {"role": "prefill"}, _shared_gid()))
            self.assertFalse(write_json_once(once, {"role": "decode"}, _shared_gid()))
            self.assertEqual(read_json(once)["role"], "prefill")

    @unittest.skipUnless(os.name == "posix", "POSIX permission semantics")
    def test_existing_peer_owned_directory_only_needs_group_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shared"
            path.mkdir(mode=0o775)
            path.chmod(0o2775)
            with mock.patch.object(Path, "chmod", side_effect=PermissionError):
                ensure_shared_dir(path, _shared_gid())

    @unittest.skipUnless(os.name == "posix", "POSIX permission semantics")
    def test_existing_directory_without_group_write_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shared"
            path.mkdir(mode=0o700)
            path.chmod(0o700)
            with mock.patch.object(Path, "chmod", side_effect=PermissionError):
                with self.assertRaisesRegex(
                    PDInfrastructureError, "incompatible ownership or mode"
                ):
                    ensure_shared_dir(path, _shared_gid())

    def test_peer_claim_rejects_same_physical_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary))
            peer = {
                **context.claim_payload(),
                "role": ROLE_DECODE,
                "runner_name": "runner-decode",
                "hostname": context.hostname,
                "ip": context.decode_ip,
            }
            with self.assertRaisesRegex(
                PDInfrastructureError, "different physical hosts"
            ):
                PDOrchestrator(context)._validate_peer_claim(peer)

    def test_peer_claim_rejects_different_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary))
            peer = {
                **context.claim_payload(),
                "role": ROLE_DECODE,
                "runner_name": "runner-decode",
                "hostname": "host-decode",
                "ip": context.decode_ip,
                "image_id": "sha256:different",
            }
            with self.assertRaisesRegex(PDInfrastructureError, "image_id"):
                PDOrchestrator(context)._validate_peer_claim(peer)

    def test_wheel_bundle_checks_all_three_wheels_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), ROLE_DECODE)
            orchestrator = PDOrchestrator(context)
            bundle = context.wheel_bundle
            bundle.mkdir(parents=True)
            wheels = []
            for kind in ("sglang", "sglang-kernel", "sglang-router"):
                wheel_path = bundle / f"{kind}.whl"
                wheel_path.write_bytes(kind.encode())
                wheels.append(
                    {
                        "kind": kind,
                        "path": wheel_path.name,
                        "sha256": hashlib.sha256(kind.encode()).hexdigest(),
                    }
                )
            (bundle / "READY").write_text(TEST_SHA)
            (bundle / "manifest.json").write_text(
                json.dumps({"commit_sha": TEST_SHA, "wheels": wheels})
            )

            manifest = orchestrator.ensure_shared_wheels()
            self.assertEqual(len(manifest["wheels"]), 3)

            wheels[0]["sha256"] = "0" * 64
            (bundle / "manifest.json").write_text(
                json.dumps({"commit_sha": TEST_SHA, "wheels": wheels})
            )
            with self.assertRaisesRegex(PDInfrastructureError, "checksum mismatch"):
                orchestrator.ensure_shared_wheels()

    def test_http_health_accepts_plain_text_and_model_info_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary))
            orchestrator = PDOrchestrator(context)
            server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                self.assertEqual(
                    orchestrator._http_json(f"{base_url}/health"),
                    {"text": "OK"},
                )
                self.assertEqual(
                    orchestrator._http_json(f"{base_url}/model_info")["model_path"],
                    "/models/MiniMax-M2.7",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
