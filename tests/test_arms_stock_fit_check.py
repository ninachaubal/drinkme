"""the stock-arm fit check's arms.py half — the live guard right before
`from_pretrained`, and the shape of `run_arms(run_stock=False)`'s branch.

These are pure-Python (no CUDA/HIP, no model download): `refuse_if_stock_wont_fit`
and `live_available_bytes`'s unified path touch only `/proc/meminfo` and
arithmetic; `torch.cuda.mem_get_info` is monkeypatched rather than called for
real, since this box's accelerator is deliberately masked for these tests
(CPU only). `run_arms` itself calls `.cuda()` throughout and a real
model download, which this suite cannot exercise (device-dependent bugs
are invisible to it by design) — its branch
structure is instead checked statically, AST-only, matching
test_bench_record_shape.py's `_assigned_dict_keys` style.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from drinkme import arms
from drinkme.fit import GB, GIB
from drinkme.suggest import MODELS, sizes

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ----------------------------------------------------- live_available_bytes --


def test_live_available_bytes_unified_reads_proc_meminfo():
    avail = arms.live_available_bytes("unified")
    assert avail is not None and avail > 0  # a real Linux box always has this


def test_live_available_bytes_discrete_none_when_mem_get_info_fails(monkeypatch):
    def boom():
        raise RuntimeError("No CUDA GPUs are available")

    monkeypatch.setattr(arms.torch.cuda, "mem_get_info", boom)
    assert arms.live_available_bytes("vram") is None


def test_live_available_bytes_discrete_reads_mem_get_info(monkeypatch):
    monkeypatch.setattr(arms.torch.cuda, "mem_get_info", lambda: (12 * GIB, 16 * GIB))
    assert arms.live_available_bytes("vram") == 12 * GIB


# --------------------------------------------------- refuse_if_stock_wont_fit --


def test_refuse_if_stock_wont_fit_passes_when_it_fits():
    assert arms.refuse_if_stock_wont_fit("toy/model", 10 * GB, "vram", 20 * GIB) is None


def test_refuse_if_stock_wont_fit_raises_when_unreadable():
    with pytest.raises(SystemExit, match="refusing to load"):
        arms.refuse_if_stock_wont_fit("toy/model", 10 * GB, "vram", None)


def test_refuse_if_stock_wont_fit_rx7600xt_qwen8b_discrete():
    """The original RX 7600 XT repro, at the arms.py guard: even if bench's own
    fit check were somehow bypassed, the live guard refuses by name before
    any allocation — never a bare torch.OutOfMemoryError three frames deep."""
    m = next(x for x in MODELS if x.name == "Qwen3-8B")
    with pytest.raises(SystemExit, match=r"refusing to load.*Qwen/Qwen3-8B"):
        arms.refuse_if_stock_wont_fit(m.hf_repo, sizes(m).bf16_gb * GB, "vram", 16.0 * GIB)


def test_refuse_if_stock_wont_fit_strix_halo_gemma_unified():
    """gemma (62.55 GB) is refused on a 124 GiB unified pool under the
    from_pretrained loader (the default charge, `direct=False`): its
    transient is 2x resident there even with device_map="cuda" (measured:
    MemAvailable 107 -> 5 GB). Under the streaming loader
    (`direct=True`) the same guard passes it — see
    test_stock_stream_loader.py."""
    m = next(x for x in MODELS if x.name == "gemma-4-31B-it")
    with pytest.raises(SystemExit, match="refusing to load"):
        arms.refuse_if_stock_wont_fit(m.hf_repo, sizes(m).bf16_gb * GB, "unified", 124.0 * GIB)
    with pytest.raises(SystemExit, match="refusing to load"):
        arms.refuse_if_stock_wont_fit(m.hf_repo, sizes(m).bf16_gb * GB, "unified", 124.0 * GIB,
                                      direct=False)
    assert arms.refuse_if_stock_wont_fit(m.hf_repo, sizes(m).bf16_gb * GB, "unified", 124.0 * GIB,
                                         direct=True) is None


def test_refuse_if_stock_wont_fit_strix_halo_27b_fits():
    """Contrast case: the same live guard, the same box, a model that
    actually fits — proves the guard isn't just always refusing."""
    m = next(x for x in MODELS if x.name == "Qwen3.8-27B")
    assert arms.refuse_if_stock_wont_fit(m.hf_repo, sizes(m).bf16_gb * GB, "unified",
                                         124.0 * GIB) is None


# ------------------------------------------------ run_arms's branch, statically --


def _run_arms_if_node() -> ast.If:
    tree = ast.parse((ROOT / "src" / "drinkme" / "arms.py").read_text())
    fn = next(n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "run_arms")
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "run_stock":
            return node
    raise AssertionError("run_arms has no `if run_stock:` branch")


def _dict_store_keys(nodes) -> set[str]:
    keys = set()
    for stmt in nodes:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    keys.add(sl.value)
    return keys


def test_run_arms_signature_carries_the_new_params():
    sig = next(n for n in ast.walk(ast.parse((ROOT / "src" / "drinkme" / "arms.py").read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "run_arms").args
    names = {a.arg for a in sig.args}
    assert {"run_stock", "stock_memory_kind", "stock_bf16_bytes"} <= names


def test_run_stock_true_branch_assigns_stock_fields_only_there():
    node = _run_arms_if_node()
    body_keys = _dict_store_keys(node.body)
    else_keys = _dict_store_keys(node.orelse)
    for field in ("stock_decode_tok_s", "stock_prefill_tok_s", "stock_ttft_s",
                 "vram_bf16_bytes"):
        assert field in body_keys, f"{field} not assigned in the run_stock branch"
        assert field not in else_keys, f"{field} leaked into the fit-point branch"


def test_run_stock_false_branch_never_assigns_stock_metrics_but_names_the_error():
    """`stock_outcome` (the outcome category) and
    `stock_error` are the two stock_* keys the fit-point branch may set —
    never a metric."""
    node = _run_arms_if_node()
    else_keys = _dict_store_keys(node.orelse)
    assert "stock_error" in else_keys and "stock_outcome" in else_keys
    assert not any(k.startswith("stock_") and k not in ("stock_error", "stock_outcome")
                   for k in else_keys)


def test_run_stock_true_branch_names_both_outcomes():
    """The run_stock branch ends as `measured` or, on a load failure the
    guard/allocator raised, `failed_load` — both spelled in that branch."""
    node = _run_arms_if_node()
    body_source = "\n".join(ast.unparse(n) for n in node.body)
    assert "'measured'" in body_source and "'failed_load'" in body_source
    assert "stock_load_failure(e)" in body_source


def test_device_map_loader_used_only_in_the_run_stock_branch():
    """the direct-to-device load (no CPU staging) belongs to pass 1
    only — pass 2/3 keep load_cpu's plain CPU path because they must swap
    Linears before the model ever touches the device."""
    node = _run_arms_if_node()
    body_source = "\n".join(ast.unparse(n) for n in node.body)
    else_source = "\n".join(ast.unparse(n) for n in node.orelse)
    assert 'device_map' in body_source
    assert 'device_map' not in else_source
