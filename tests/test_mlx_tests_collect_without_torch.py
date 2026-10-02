"""Regression guard.

Two mlx-gated tests were unrunnable on a torch-less box for reasons that had
nothing to do with mlx: tests/test_pack_identity.py imported torch at module
scope, so the whole FILE failed collection; tests/test_identity_torch_free.py
built its toy with torch INSIDE the pytest process (`_toy_qwen3`) before the
torch-less subprocess it hands the toy to ever ran, so the toy build itself
raised ModuleNotFoundError. Neither is a regression in the code under test —
both were unrunnable from the day they were written — and nothing on Linux
catches it.

This test proves collection alone, torch blocked the way
test_identity_torch_free.py's `_fresh` blocks it (`sys.modules['torch'] =
None`, not a meta-path finder — matching what transformers' own
is_torch_available() probe sees), so a future module-scope `import torch`
in either file fails HERE, in a few seconds on any box, instead of on the
Mac a day later. It does not need mlx: --collect-only never imports mlx.core,
it only imports the test modules themselves.
"""

import os
import subprocess
import sys
import textwrap

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = ["test_pack_identity.py", "test_identity_torch_free.py"]
MLX_TEST_IDS = [
    "test_pack_identity.py::test_pack_from_A_served_with_B_is_refused_on_the_mlx_lane",
    "test_identity_torch_free.py::test_the_compressed_mlx_loader_runs_with_torch_blocked",
]


def test_both_mlx_gated_files_collect_with_torch_blocked():
    """`pytest --collect-only -q` on both files, torch blocked, from a fresh
    interpreter: exit 0, and both files' mlx test ids are in the collected
    list. A module-scope `import torch` in either file fails its COLLECTION
    outright."""
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    code = textwrap.dedent(f"""
        import sys
        sys.modules["torch"] = None
        import pytest
        raise SystemExit(pytest.main(["--collect-only", "-q", {FILES[0]!r}, {FILES[1]!r}]))
        """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=TESTS_DIR, env=env, timeout=120)
    assert r.returncode == 0, (
        f"collection failed with torch blocked (exit {r.returncode}):\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    for test_id in MLX_TEST_IDS:
        assert test_id in r.stdout, f"{test_id} missing from collection:\n{r.stdout}"
