"""Contract tests for the per-linear gfx928 W8A8 M=1 native ops.

Covers the four exact-shape integrations (qkv_proj / o_proj / dense_gate_up /
dense_down) against their sgl-kernel sources, registrations, build wiring,
environment gates, glue modules and hooks. All default tests are CPU-only
(source text assertions + reference pack round trips); the end-to-end
bit-exact native-kernel tests are skip-guarded on GPU availability so this
file collects on CPU-only runners.
"""

from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")

ROOT = Path(__file__).resolve().parents[5]
QUANT_DIR = ROOT / "sgl-kernel/csrc/quantization"
GLUE_DIR = ROOT / "python/sglang/srt/layers/quantization"
OPS_H = ROOT / "sgl-kernel/include/sgl_kernel_ops.h"
REGISTRATION = ROOT / "sgl-kernel/csrc/common_extension_rocm.cc"
BUILD = ROOT / "sgl-kernel/setup_rocm.py"
ENVIRON = ROOT / "python/sglang/srt/environ.py"
SERVER_ARGS = ROOT / "python/sglang/srt/server_args.py"
W8A8_INT8 = ROOT / "python/sglang/srt/layers/quantization/w8a8_int8.py"

# op -> (kernel base name, K, N, guard macro, packed shape/dtype, workspace
# buffer names, env name, glue module base name, prefix suffix)
OPS = {
    "qkv_proj": {
        "file": "w8a8_gfx928_qkv_proj_m1.hip",
        "k": 6144,
        "n": 1280,
        "macro": "SGL_W8A8_GFX928_QKV_PROJ_M1_BUILD",
        "packed": (1536, 1280),
        "packed_dtype": torch.int32,
        "glue": "rocm_w8a8_qkv_proj.py",
        "env": "SGLANG_ROCM_GFX928_W8A8_QKV_PROJ_M1",
        "prefix": "self_attn.qkv_proj",
        "two_launch": False,
        "workspace": ("acc", "counter"),
    },
    "o_proj": {
        "file": "w8a8_gfx928_o_proj_m1.hip",
        "k": 1024,
        "n": 6144,
        "macro": "SGL_W8A8_GFX928_O_PROJ_M1_BUILD",
        "packed": (192, 4096),
        "packed_dtype": torch.int64,
        "glue": "rocm_w8a8_o_proj.py",
        "env": "SGLANG_ROCM_GFX928_W8A8_O_PROJ_M1",
        "prefix": "self_attn.o_proj",
        "two_launch": False,
        "workspace": (),
    },
    "dense_gate_up": {
        "file": "w8a8_gfx928_dense_gate_up_m1.hip",
        "k": 6144,
        "n": 3072,
        "macro": "SGL_W8A8_GFX928_DENSE_GATE_UP_M1_BUILD",
        "packed": (1536, 3072),
        "packed_dtype": torch.int32,
        "glue": "rocm_w8a8_dense_gate_up.py",
        "env": "SGLANG_ROCM_GFX928_W8A8_DENSE_GATE_UP_M1",
        "prefix": "mlp.gate_up_proj",
        "two_launch": True,
        "workspace": ("qx", "act_scale", "acc", "counter"),
    },
    "dense_down": {
        "file": "w8a8_gfx928_dense_down_m1.hip",
        "k": 1536,
        "n": 6144,
        "macro": "SGL_W8A8_GFX928_DENSE_DOWN_M1_BUILD",
        "packed": (384, 6144),
        "packed_dtype": torch.int32,
        "glue": "rocm_w8a8_dense_down.py",
        "env": "SGLANG_ROCM_GFX928_W8A8_DENSE_DOWN_M1",
        "prefix": "mlp.down_proj",
        "two_launch": True,
        "workspace": ("qx", "act_scale", "acc", "counter"),
    },
}

OP_NAMES = list(OPS)


def _pack_k4n(weight_nk: torch.Tensor) -> torch.Tensor:
    n, k = weight_nk.shape
    words = weight_nk.t().contiguous().reshape(k // 4, 4, n).to(torch.int32)
    packed = (words[:, 0] & 0xFF) | ((words[:, 1] & 0xFF) << 8)
    packed = packed | ((words[:, 2] & 0xFF) << 16) | ((words[:, 3] & 0xFF) << 24)
    return packed.contiguous()


def _unpack_k4n(packed: torch.Tensor, k: int, n: int) -> torch.Tensor:
    assert packed.shape == (k // 4, n)
    lanes = torch.stack([(packed >> (8 * lane)) & 0xFF for lane in range(4)], dim=1)
    signed = torch.where(lanes >= 128, lanes - 256, lanes).to(torch.int8)
    return signed.reshape(k, n).t().contiguous()


def _pack_tile(weight_nk: torch.Tensor) -> torch.Tensor:
    n, k = weight_nk.shape
    k8n = k // 8
    lanes = weight_nk.t().contiguous().reshape(k8n, 8, n).to(torch.int64)
    k8n_packed = lanes[:, 0] & 0xFF
    for lane in range(1, 8):
        k8n_packed = k8n_packed | ((lanes[:, lane] & 0xFF) << (8 * lane))
    return (
        k8n_packed.reshape(k8n, n // 32, 32)
        .permute(1, 0, 2)
        .reshape(n // 32, k8n * 32)
        .contiguous()
    )


def _unpack_tile(packed: torch.Tensor, k: int, n: int) -> torch.Tensor:
    k8n = k // 8
    assert packed.shape == (n // 32, k8n * 32)
    p8 = packed.reshape(n // 32, k8n, 32).permute(1, 0, 2).reshape(k8n, n)
    lanes = torch.stack([(p8 >> (8 * lane)) & 0xFF for lane in range(8)], dim=1)
    signed = torch.where(lanes >= 128, lanes - 256, lanes).to(torch.int8)
    return signed.reshape(k, n).t().contiguous()


@pytest.mark.parametrize("op", ["qkv_proj", "dense_gate_up", "dense_down"])
def test_k4n_round_trip_preserves_signed_byte_order(op):
    n, k = OPS[op]["n"], OPS[op]["k"]
    weight = torch.zeros((n, k), dtype=torch.int8)
    weight[:, :4] = torch.tensor([-128, -1, 0, 127], dtype=torch.int8)
    packed = _pack_k4n(weight)
    assert packed.shape == OPS[op]["packed"]
    assert packed.is_contiguous()
    assert packed.dtype == torch.int32
    assert torch.equal(_unpack_k4n(packed, k, n), weight)


def test_tile_round_trip_preserves_signed_byte_order():
    n, k = OPS["o_proj"]["n"], OPS["o_proj"]["k"]
    weight = torch.zeros((n, k), dtype=torch.int8)
    weight[:, :8] = torch.tensor([-128, -1, 0, 127, 64, -64, 1, -2], dtype=torch.int8)
    packed = _pack_tile(weight)
    assert packed.shape == OPS["o_proj"]["packed"]
    assert packed.is_contiguous()
    assert packed.dtype == torch.int64
    assert torch.equal(_unpack_tile(packed, k, n), weight)


@pytest.mark.parametrize("op", OP_NAMES)
def test_native_abi_is_exact_shape_current_stream_single_or_two_stage(op):
    cfg = OPS[op]
    source = (QUANT_DIR / cfg["file"]).read_text()
    assert f"constexpr int kK = {cfg['k']};" in source
    assert f"constexpr int kN = {cfg['n']};" in source
    assert f"#if {cfg['macro']}" in source
    assert f"#endif  // {cfg['macro']}" in source
    assert "__builtin_amdgcn_sdot4" in source
    assert f"w8a8_gfx928_{op}_m1(" in source
    assert "OptionalHIPGuardMasqueradingAsCUDA" in source
    assert "getCurrentHIPStreamMasqueradingAsCUDA()" in source
    assert "hipMalloc" not in source
    assert "hipMemcpy" not in source
    assert "hipDeviceSynchronize" not in source
    assert "this_grid" not in source
    assert "hipLaunchCooperativeKernel" not in source
    assert f"w8a8_gfx928_{op}_m1 is only available on gfx928" in source
    if cfg["two_launch"]:
        assert "quantize_global_kernel" in source
        assert "hipLaunchKernelGGL" in source
    if op == "dense_gate_up":
        assert "fused_gridsync_kernel" in source
        assert "atomicExch(counter, 0)" not in source
        assert "atomicAdd(counter, 1) == kBlocks - 1" in source
        assert "counter[0] = 0" in source
    if op == "dense_down":
        assert "partial_pq_fused_kernel" in source
        assert "atomicExch(counter, 0)" not in source
        assert "atomicAdd(counter, 1) == kBlocks - 1" in source
        assert "counter[0] = 0" in source
    if op == "qkv_proj":
        assert "w8a8_gfx928_qkv_proj_m1_kernel" in source
        assert "atomicAdd(&accumulation[column]" in source or "atomicAdd(&acc[col0" in source
    if op == "o_proj":
        assert "w8a8_gfx928_o_proj_m1_kernel" in source
        assert "kGridX" in source


@pytest.mark.parametrize("op", OP_NAMES)
def test_registration_and_build_entries(op):
    cfg = OPS[op]
    declarations = OPS_H.read_text()
    registration = REGISTRATION.read_text()
    build = BUILD.read_text()
    assert f"at::Tensor w8a8_gfx928_{op}_m1(" in declarations
    assert f"m.def(\n      \"w8a8_gfx928_{op}_m1(" in registration
    assert f"m.impl(\"w8a8_gfx928_{op}_m1\", torch::kCUDA, &w8a8_gfx928_{op}_m1)" in registration
    assert f"csrc/quantization/w8a8_gfx928_{op}_m1.hip" in build
    assert f"-DSGL_W8A8_GFX928_{op.upper()}_M1_BUILD={{int(amdgpu_target == 'gfx928')}}" in build


def test_schema_strings_match_plan():
    registration = REGISTRATION.read_text()
    assert (
        "w8a8_gfx928_qkv_proj_m1(Tensor x, Tensor weight_k4n, Tensor weight_scale, Tensor? bias, Tensor! acc, "
        "Tensor! counter) -> Tensor" in registration
    )
    assert (
        "w8a8_gfx928_o_proj_m1(Tensor x, Tensor weight_tile, Tensor weight_scale, Tensor? bias) -> Tensor"
        in registration
    )
    assert (
        "w8a8_gfx928_dense_gate_up_m1(Tensor x, Tensor weight_k4n, Tensor weight_scale, Tensor? bias, Tensor! qx, "
        "Tensor! act_scale, Tensor! acc, Tensor! counter) -> Tensor" in registration
    )
    assert (
        "w8a8_gfx928_dense_down_m1(Tensor x, Tensor weight_k4n, Tensor weight_scale, Tensor? bias, Tensor! qx, "
        "Tensor! act_scale, Tensor! acc, Tensor! counter) -> Tensor" in registration
    )


@pytest.mark.parametrize("op", OP_NAMES)
def test_selector_envs_are_default_off_and_glue_is_metadata_only(op):
    cfg = OPS[op]
    environ = ENVIRON.read_text()
    glue = (GLUE_DIR / cfg["glue"]).read_text()
    assert f"{cfg['env']} = EnvBool(False)" in environ
    assert f"envs.{cfg['env']}.get()" in glue
    assert f"_PREFIX_SUFFIX = \"{cfg['prefix']}\"" in glue
    assert f"_INPUT_SHAPE = (1, {cfg['k']})" in glue
    assert f"_WEIGHT_SHAPE = ({cfg['n']}, {cfg['k']})" in glue
    assert f"_PACKED_SHAPE = {cfg['packed']}" in glue
    assert f"_OUTPUT_SHAPE = (1, {cfg['n']})" in glue
    assert f"_OP_NAME = \"w8a8_gfx928_{op}_m1\"" in glue
    assert "register_fake_if_exists" in glue
    assert "x.new_empty(_OUTPUT_SHAPE)" in glue
    assert "torch.ops.sgl_kernel.w8a8_gfx928_{}_m1(".format(op) in glue
    assert f"initialize_{op}_m1" in glue
    assert f"install_{op}_m1" in glue
    assert f"try_{op}_m1" in glue
    assert "torch.cuda.is_current_stream_capturing()" not in glue
    # workspace buffers registered lazily, created only at install time
    assert f"register_buffer(_PACKED_BUFFER, None, persistent=False)" in glue
    for workspace in cfg["workspace"]:
        assert f"_{workspace.upper()}_BUFFER" in glue
    if cfg["packed_dtype"] == torch.int32:
        assert "pack_signed_int8_k4n" in glue
    else:
        assert "pack_signed_int8_tile" in glue
    if op == "o_proj":
        assert "torch.zeros" not in glue
    else:
        assert "torch.zeros" in glue
    assert "persistent=False" in glue


def test_server_args_pd_multiplexing_assert_covers_all_gfx928_w8a8_envs():
    server_args = SERVER_ARGS.read_text()
    assert "PD-Multiplexing is not compatible with" in server_args
    for op in OP_NAMES:
        assert f"SGLANG_ROCM_GFX928_W8A8_{op.upper()}_M1" in server_args
    assert "SGLANG_ROCM_GFX928_W8A8_SHARED_GATE_UP_M1" in server_args
    # the shared-gate-up single-assert block must have been replaced by the loop
    assert "for _gfx928_w8a8_env in (" in server_args


def test_w8a8_int8_hooks_are_positioned_before_transpose_and_fallback():
    source = W8A8_INT8.read_text()
    for op in OP_NAMES:
        assert f"from sglang.srt.layers.quantization.rocm_w8a8_{op} import (" in source
        assert f"initialize_{op}_m1," in source
        assert f"install_{op}_m1," in source
        assert f"try_{op}_m1," in source
    installs = source.index("install_shared_gate_up_m1(layer, self.prefix)")
    transpose = source.index("layer.weight = Parameter(layer.weight.t(), requires_grad=False)")
    for op in OP_NAMES:
        install_pos = source.index(f"install_{op}_m1(layer, self.prefix)")
        assert installs < install_pos < transpose
    create_weights = source.index("def create_weights(")
    apply_def = source.index("def apply(")
    for op in OP_NAMES:
        initialize_pos = source.index(f"initialize_{op}_m1(layer)")
        assert create_weights < initialize_pos < apply_def
    quant_line = source.index("x_q, x_scale = per_token_quant_int8(x)")
    for op in OP_NAMES:
        try_pos = source.index(f"specialized = try_{op}_m1(layer, self.prefix, x, bias)")
        assert try_pos < quant_line


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
class TestGfx928DenseQkvNativeKernels:
    """End-to-end bit-exact checks against a float64/exact-int oracle.

    These only run on AMD CI GPU nodes where sgl_kernel is built for gfx928.
    The oracle replicates the kernel arithmetic exactly: magnitude absmax,
    scale = absmax/127 with the 1e-10 floor, q = clamp(round(x*recip)),
    exact int32 dot products, fp32 (sum*act_scale)*w_scale, single RNE cast.
    """

    @pytest.mark.parametrize("op", OP_NAMES)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_bit_exact_against_reference(self, op, dtype):
        import sgl_kernel  # noqa: F401 -- registers the custom ops

        cfg = OPS[op]
        if not hasattr(torch.ops.sgl_kernel, f"w8a8_gfx928_{op}_m1"):
            pytest.skip(f"w8a8_gfx928_{op}_m1 is not built in this sgl_kernel")
        device = torch.device("cuda")
        torch.manual_seed(1234 + OP_NAMES.index(op))
        k, n = cfg["k"], cfg["n"]
        x = (torch.randn(1, k, dtype=torch.float32) * 0.5).to(dtype).to(device)
        weight_nk = torch.randint(-127, 128, (n, k), dtype=torch.int8)
        weight_scale = (torch.rand(n, dtype=torch.float32) + 0.5).to(device)

        # per-token quantization oracle (identical to the kernel formula)
        vmax = x.float().abs().amax()
        absmax = torch.maximum(vmax, torch.tensor(1e-10, dtype=torch.float32, device=device))
        act_scale = absmax / 127.0
        recip = 127.0 / absmax
        q = torch.clamp(torch.round(x.float() * recip), -127, 127).to(torch.int8)

        # exact integer dot products (int32 products, int64 reduction)
        wt = weight_nk.t().contiguous().to(device)
        exact = (q[0].to(torch.int32).unsqueeze(1) * wt.to(torch.int32)).sum(
            dim=0, dtype=torch.int64
        )
        value = (exact.float() * act_scale) * weight_scale

        if op == "qkv_proj":
            packed = _pack_k4n(weight_nk).to(device)
            acc = torch.zeros(n, dtype=torch.int32, device=device)
            counter = torch.zeros(1, dtype=torch.int32, device=device)
            out = torch.ops.sgl_kernel.w8a8_gfx928_qkv_proj_m1(
                x, packed, weight_scale, None, acc, counter
            )
        elif op == "o_proj":
            packed = _pack_tile(weight_nk).to(device)
            out = torch.ops.sgl_kernel.w8a8_gfx928_o_proj_m1(
                x, packed, weight_scale, None
            )
        elif op == "dense_gate_up":
            packed = _pack_k4n(weight_nk).to(device)
            qx = torch.zeros(k // 4, dtype=torch.int32, device=device)
            act_scale_buf = torch.zeros(1, dtype=torch.float32, device=device)
            acc = torch.zeros(n, dtype=torch.int32, device=device)
            counter = torch.zeros(1, dtype=torch.int32, device=device)
            out = torch.ops.sgl_kernel.w8a8_gfx928_dense_gate_up_m1(
                x, packed, weight_scale, None, qx, act_scale_buf, acc, counter
            )
        else:  # dense_down
            packed = _pack_k4n(weight_nk).to(device)
            qx = torch.zeros(k // 4, dtype=torch.int32, device=device)
            act_scale_buf = torch.zeros(1, dtype=torch.float32, device=device)
            acc = torch.zeros(n, dtype=torch.int32, device=device)
            counter = torch.zeros(1, dtype=torch.int32, device=device)
            out = torch.ops.sgl_kernel.w8a8_gfx928_dense_down_m1(
                x, packed, weight_scale, None, qx, act_scale_buf, acc, counter
            )

        assert out.shape == (1, n)
        assert out.dtype == dtype
        assert torch.equal(out[0], value.to(dtype)), (
            f"w8a8_gfx928_{op}_m1 {dtype} output deviates from the exact oracle"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
