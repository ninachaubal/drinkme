"""the pack-id helper lives in a torch-free module.

The failure it guards: slotstore imports torch at module scope, so an
engine_mlx.load_compressed_mlx that imported serving/slotstore.py for
pack_id_from_meta would put a metal environment without torch (one made
before pyproject's `metal` group carried torch for packing — docs/metal.md)
at ModuleNotFoundError before the loader opened a prebuilt pack. The probe: a
fresh interpreter with a meta-path finder that rejects torch; engine_mlx
imports, and the loader must not fail on the blocked import.

Both tests run their subject in a FRESH interpreter (subprocess) so the
parent's own torch import cannot leak in and hide the regression. The mlx
test's toy must not be built with torch IN THIS PYTEST PROCESS — on a Mac
(no torch) that build would raise before the subprocess ever ran — so it
takes the toy the DRINKME_RADIX_MLX_TOY pattern uses everywhere else
(test_serving_engine_mlx.py's `toy` fixture): prebuilt, or built here
through the one shared builder when torch exists, or a by-name skip.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
sys.path.insert(0, BENCH_DIR)
from radix_mlx_toy_build import build_toy_qwen3  # noqa: E402

TOY_ENV = "DRINKME_RADIX_MLX_TOY"


def _fresh(code: str, timeout: int = 300) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=timeout, env=env)


def test_importing_the_identity_module_leaves_torch_out_of_sys_modules():
    r = _fresh("""
        import sys
        from drinkme.codec import identity
        from drinkme.serving import checkpoint
        assert callable(identity.pack_id_from_meta)
        assert identity.pack_id_from_meta({"manifestSha256": "x", "tensors": {}}).startswith("manifest")
        print(json.dumps({"torch": "torch" in sys.modules}) if (json := __import__("json")) else "")
        """)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == {"torch": False}


def _prebuilt_plain_toy(root: str) -> tuple[str, str] | None:
    """The prebuilt plain toy under DRINKME_RADIX_MLX_TOY (`model/` +
    `pack/`, bench/radix_mlx_toy_build.py's `--out` layout — the SAME toy
    test_serving_engine_mlx.py's `toy` fixture reads) or None."""
    model_dir, pack_dir = os.path.join(root, "model"), os.path.join(root, "pack")
    return (model_dir, pack_dir) if os.path.isdir(pack_dir) else None


def _mlx_toy(tmp_path_factory) -> tuple[str, str]:
    """(model_dir, pack_dir): prebuilt from DRINKME_RADIX_MLX_TOY, else
    built here through build_toy_qwen3 when torch is importable — the same
    builder test_serving_engine_mlx.py's `toy` fixture uses, so this is
    never a second toy — else a by-name skip."""
    root = os.environ.get(TOY_ENV)
    if root:
        found = _prebuilt_plain_toy(root)
        if found:
            return found
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("no torch to build the toy and DRINKME_RADIX_MLX_TOY names none "
                    "(bench/radix_mlx_toy_build.py)")
    d = tmp_path_factory.mktemp("toy_qwen3")
    return build_toy_qwen3(str(d / "model"), str(d / "pack"))


def test_the_compressed_mlx_loader_runs_with_torch_blocked(tmp_path_factory):
    """The ruler: a meta-path finder that rejects torch,
    then the compressed MLX loader on a prebuilt pack — all the way through
    a greedy generation on the reference path. Any failure must be about
    mlx, never torch — a module-scope torch import (slotstore's, say) shows
    up as `ModuleNotFoundError: import of torch halted` before the pack is
    opened."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    model_dir, pack_dir = _mlx_toy(tmp_path_factory)
    r = _fresh(f"""
        import json, sys, traceback

        # A torch-less box, faithfully: sys.modules['torch'] = None makes
        # `import torch` raise ImportError AND importlib.util.find_spec('torch')
        # return None — which is what transformers' own is_torch_available()
        # probes at import, so transformers comes up torch-free exactly as it
        # does in a metal environment without torch. (A meta-path finder that RAISES
        # from find_spec breaks transformers itself, which no real box does.)
        sys.modules["torch"] = None
        out = {{"stage": "import"}}
        try:
            from drinkme.serving import engine_mlx
            from drinkme.serving.engine import GenerationRequest, SampleParams, complete
            out["stage"] = "load"
            eng = engine_mlx.load_compressed_mlx({model_dir!r}, None, {pack_dir!r}, path="reference")
            out["packId"] = eng.meta["packId"]
            out["stage"] = "generate"
            res = complete(eng, GenerationRequest([{{"role": "user", "content": "hello world"}}],
                                                  SampleParams(temperature=0.0, max_tokens=4)))
            out["finish"] = res.finish_reason
            out["stage"] = "done"
        except BaseException as e:  # noqa: BLE001 — the verdict rides in the output
            out["error"] = f"{{type(e).__name__}}: {{e}}"
            out["trace"] = traceback.format_exc()
        out["torch_loaded"] = sys.modules.get("torch") is not None
        print("VERDICT " + json.dumps(out))
        """)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("VERDICT ")]
    assert lines, f"no verdict\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    v = json.loads(lines[-1][len("VERDICT "):])
    assert "torch" not in v.get("error", ""), \
        f"torch reached at stage {v['stage']}: {v.get('error')}\n{v.get('trace', '')}"
    assert v["stage"] == "done", f"failed at {v['stage']} (not torch): {v.get('error')}\n{v.get('trace', '')}"
    assert v["torch_loaded"] is False
    assert v["packId"].startswith("manifest")
    assert v["finish"] in ("length", "stop")
