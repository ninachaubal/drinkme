"""The Metal loaders' fit check runs BEFORE a weight is materialised
(engine_mlx's FIT CHECK 1).

A loader that loads the model first — packed modules installed, the raw
remainder mx.eval'd tensor by tensor — and only then compares what it
holds against the working set fails on a Mac whose working set is smaller
than the model: the allocation failure or the swap storm comes before the
refusal. So the budget is estimated from descriptors alone
(estimate_load_bytes: safetensors headers + the pack's zip directories,
zero weight bytes) and judged first; the measured post-load check stays
as the second gate.

CPU only, network-free, and mlx-free: on a box without mlx (the Linux dev
box) a stub of the surface engine_mlx touches is installed for the test
(the module imports, the loaders run up to the trap); on a Mac the real
mlx is used and the same functions are monkeypatched. Every door a weight
could come through — _stream_checkpoint, make_module_mlx, mx.eval,
mx.array, mx.load — raises Materialized, so a loader that touches one
before refusing fails the test with that name rather than a SystemExit.
The toy is tests/test_serving_engine_mlx.py's (a real radix pack of a
2-layer Qwen3, built here with torch, or DRINKME_RADIX_MLX_TOY's).

The real Mac run — the loaders end to end on Metal with a working set
below a real model — is owed on the next Mac window.
"""

import importlib
import json
import os
import sys
import types

import pytest

try:
    import torch
except ImportError:  # pragma: no cover — the Mac
    torch = None

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
TOY_ENV = "DRINKME_RADIX_MLX_TOY"
GIB = 1024 ** 3

# the modules the fixture imports fresh against the (stub or real) mlx, and
# pops again so no later test meets an engine bound to the stub
_FRESH = ("drinkme.serving.engine_mlx", "drinkme.metal.gemv_radix", "drinkme.metal.dense_radix",
          "drinkme.metal.twin_radix")
_MLX = ("mlx", "mlx.core", "mlx.nn", "mlx.utils")


class Materialized(Exception):
    """A weight was materialised — the thing the fit check exists to precede."""


def _trap(what):
    def f(*a, **k):
        raise Materialized(what)
    return f


def _stub_mlx() -> dict:
    """The mlx surface engine_mlx and the metal kernels touch at import
    (dtype names, nn.Module, the metal_kernel constructors) and on the
    load path (metal.is_available, device_info) — enough to import and to
    reach the loaders' first materialising call, no more."""
    core = types.ModuleType("mlx.core")
    core.__version__ = "0.0-stub"
    for nm in ("bfloat16", "float16", "float32", "uint16", "uint32", "int32", "floating"):
        setattr(core, nm, nm)
    core.metal = types.SimpleNamespace(is_available=lambda: True)
    core.device_info = lambda: {"device_name": "stub Metal device",
                                "max_recommended_working_set_size": 16 * GIB}
    core.fast = types.SimpleNamespace(metal_kernel=lambda **kw: None)
    core.issubdtype = lambda a, b: False
    for nm in ("array", "eval", "load", "zeros", "concatenate", "matmul", "addmm"):
        setattr(core, nm, _trap(f"mx.{nm}"))

    nn = types.ModuleType("mlx.nn")

    class Module:
        def __init__(self):
            self.__dict__["_items"] = {}

        def __getitem__(self, k):
            return self._items[k]

        def __setitem__(self, k, v):
            self._items[k] = v

        def __contains__(self, k):
            return k in self._items

    class Linear(Module):
        pass

    nn.Module, nn.Linear = Module, Linear
    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda tree: []
    pkg = types.ModuleType("mlx")
    pkg.__path__ = []
    pkg.core, pkg.nn, pkg.utils = core, nn, utils
    return {"mlx": pkg, "mlx.core": core, "mlx.nn": nn, "mlx.utils": utils}


@pytest.fixture
def engine(request):
    """drinkme.serving.engine_mlx imported fresh — against the real mlx
    where there is one, else against the stub — and popped afterwards."""
    try:
        import mlx.core
        # conftest's DRINKME_TEST_HOST stand-in answers the host probes only
        stub = _stub_mlx() if getattr(mlx.core, "TEST_HOST_STANDIN", False) else None
    except ImportError:
        stub = _stub_mlx()
    saved = {n: sys.modules.pop(n) for n in _FRESH + _MLX if n in sys.modules}
    # an import also binds the module on its parent package, where `from
    # drinkme.serving import engine_mlx` finds it: put those back too, or a
    # later test gets the fresh engine's classes beside the original's
    parents = {n: getattr(sys.modules.get(n.rpartition(".")[0]), n.rpartition(".")[2], None)
               for n in _FRESH}

    def restore():
        for n in _FRESH + _MLX:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
        for n, mod in parents.items():
            pkg = sys.modules.get(n.rpartition(".")[0])
            if pkg is None:
                continue
            if mod is None:
                pkg.__dict__.pop(n.rpartition(".")[2], None)
            else:
                setattr(pkg, n.rpartition(".")[2], mod)

    request.addfinalizer(restore)
    if stub is not None:
        sys.modules.update(stub)
    else:
        sys.modules.update({n: saved[n] for n in _MLX if n in saved})
    return importlib.import_module("drinkme.serving.engine_mlx")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = os.environ.get(TOY_ENV)
    if root and os.path.isdir(os.path.join(root, "pack")):
        return os.path.join(root, "model"), os.path.join(root, "pack")
    if torch is None:
        pytest.skip(f"no torch to build the toy and {TOY_ENV} unset (bench/radix_mlx_toy_build.py)")
    sys.path.insert(0, BENCH_DIR)
    from radix_mlx_toy_build import build_toy_qwen3

    d = tmp_path_factory.mktemp("toy_qwen3_fit_check")
    return build_toy_qwen3(str(d / "model"), str(d / "pack"))


def _array_trap(real):
    """mx.array as a door: CALLING it raises Materialized, but it stays a
    type for isinstance — real mlx's nn.Module.__setattr__ asks
    isinstance(value, mx.array) on every attribute, so a plain trap
    function in its place breaks the stand-in skeleton before the loader
    reaches a door. The stub's mx.array is a trap function, not a type."""
    if not isinstance(real, type):
        return _trap("mx.array")

    class Meta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, real)

        def __call__(cls, *a, **k):
            raise Materialized("mx.array")

    return Meta("array", (), {})


def _arm_traps(engine, monkeypatch):
    """Every door a weight comes through raises Materialized; the skeleton
    is a stand-in (mlx-lm is not needed to reach the first door), and the
    module lookup hands back a Linear so the packed-module swap proceeds to
    make_module_mlx — where it is trapped."""
    nn = engine.nn
    monkeypatch.setattr(engine, "_stream_checkpoint", _trap("_stream_checkpoint"))
    monkeypatch.setattr(engine, "make_module_mlx", _trap("make_module_mlx"))
    monkeypatch.setattr(engine.mx, "array", _array_trap(engine.mx.array))
    for nm in ("eval", "load"):
        monkeypatch.setattr(engine.mx, nm, _trap(f"mx.{nm}"))
    monkeypatch.setattr(engine, "build_skeleton",
                        lambda config: (nn.Module(), types.SimpleNamespace(tie_word_embeddings=False)))
    monkeypatch.setattr(engine, "parameter_paths", lambda model: set())
    # an instance without __init__: the real mlx Linear wants its dims and draws a weight
    monkeypatch.setattr(engine, "_module_at", lambda model, path: nn.Linear.__new__(nn.Linear))
    monkeypatch.setattr(engine, "_host_budget", lambda: None)  # the host term: its own tests below


# ------------------------------------------------ the ordering, both arms --


def test_compressed_loader_refuses_on_the_estimate_before_any_weight_is_materialised(
        toy, engine, monkeypatch, capsys):
    model_dir, pack_dir = toy
    _arm_traps(engine, monkeypatch)
    monkeypatch.setattr(engine, "_working_set", lambda: 1 * GIB // 1024)  # 1 MiB: nothing fits
    with pytest.raises(SystemExit, match=r"refusing to load \(estimate\)"):
        engine.load_compressed_mlx(model_dir, None, pack_dir, path="reference")
    err = capsys.readouterr().err
    assert "fit check (measured)" not in err  # the second gate never ran: nothing was loaded


def test_stock_loader_refuses_on_the_estimate_before_any_weight_is_materialised(
        toy, engine, monkeypatch, capsys):
    model_dir, _ = toy
    _arm_traps(engine, monkeypatch)
    monkeypatch.setattr(engine, "_working_set", lambda: 1 * GIB // 1024)
    with pytest.raises(SystemExit, match=r"refusing to load \(estimate\)"):
        engine.load_stock_mlx(model_dir, None)
    assert "fit check (measured)" not in capsys.readouterr().err


def test_a_fitting_estimate_lets_both_loaders_proceed_to_materialise(toy, engine, monkeypatch, capsys):
    """The happy path: the estimate is under the ceiling, is printed as
    ok, and the loader goes on to its first materialising call — which
    here is the trap, so Materialized IS the proof it proceeded."""
    model_dir, pack_dir = toy
    _arm_traps(engine, monkeypatch)
    monkeypatch.setattr(engine, "_working_set", lambda: 16 * GIB)
    with pytest.raises(Materialized, match="make_module_mlx"):
        engine.load_compressed_mlx(model_dir, None, pack_dir, path="reference")
    err = capsys.readouterr().err
    assert "fit check (estimate): ok" in err
    with pytest.raises(Materialized, match="make_module_mlx"):  # the bench's twin arm, too
        engine.load_compressed_mlx(model_dir, None, pack_dir, path=engine.TWIN)
    assert "fit check (estimate): ok" in capsys.readouterr().err
    with pytest.raises(Materialized, match="_stream_checkpoint"):
        engine.load_stock_mlx(model_dir, None)
    assert "fit check (estimate): ok" in capsys.readouterr().err


# ---------------------------------------------------------- the estimate --


def test_estimate_reads_descriptors_only_and_matches_the_packers_measurement(toy, engine, monkeypatch):
    from drinkme.codec.pack import _shard_headers, iter_pack_descriptors, packed_resident_estimate

    model_dir, pack_dir = toy
    for nm in ("eval", "array", "load"):  # an estimate that reads a weight fails here
        monkeypatch.setattr(engine.mx, nm, _trap(f"mx.{nm}"))
    est = engine.estimate_load_bytes(model_dir, pack_dir, tie=False)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    headers = _shard_headers(model_dir)
    packed = {name: (sc, sizes) for name, sc, sizes in iter_pack_descriptors(pack_dir)}
    assert est.packed == meta["tensorCount"] == len(packed) == 10
    assert est.raw == len(headers) - est.packed
    # the raw remainder is the packer's own rawResidentBytes (same rule:
    # floats at bf16 width); the packed half is within the .npy headers of
    # the torch lane's measured runtime dicts
    raw = est.resident - sum(packed_resident_estimate(sc, sz) for sc, sz in packed.values())
    assert raw == meta["rawResidentBytes"]
    packed_measured = meta["residentBytes"] - meta["rawResidentBytes"]
    assert abs((est.resident - raw) - packed_measured) < 0.001 * packed_measured
    # the transient: the largest raw tensor staged at load (embed_tokens /
    # lm_head at 64 x 1024 bf16 here) vs the largest packed Linear's dense
    # bf16 plane at prefill (1024 x 1024 x 2) — the plane wins on this toy
    assert est.transient == 2 * 1024 * 1024
    # the stock arm: every header tensor, nothing packed, the largest raw
    # tensor (a 1024 x 1024 bf16 projection) is the whole transient
    stock = engine.estimate_load_bytes(model_dir, None, tie=False)
    assert stock.packed == 0 and stock.raw == len(headers)
    assert stock.resident == sum(2 * (sh[0] * (sh[1] if len(sh) > 1 else 1))
                                 for _, sh, _ in headers.values())
    assert stock.transient == 2 * 1024 * 1024
    # a tied head is not held: lm_head.weight leaves the sum
    tied = engine.estimate_load_bytes(model_dir, None, tie=True)
    assert tied.resident == stock.resident - 2 * 64 * 1024 and tied.raw == stock.raw - 1
    # the twin arm (RadixTwinLinear): each radix tensor resident as its bf16
    # plane, and the transient the decode of one tensor — its streams plus
    # one chunk of the decoder's planes — not a prefill plane it never builds
    from drinkme.metal.twin_radix import chunk_transient_bytes

    twin = engine.estimate_load_bytes(model_dir, pack_dir, tie=False, twin=True)
    assert twin.packed == est.packed and twin.raw == est.raw
    assert twin.resident == raw + sum(2 * int(sc["R"]) * int(sc["C"]) for sc, _ in packed.values())
    assert twin.transient == max(packed_resident_estimate(sc, sz)
                                 + chunk_transient_bytes(int(sc["R"]), int(sc["C"]))
                                 for sc, sz in packed.values())


def test_fit_check_charges_the_transient_in_fit_py_terms_and_names_the_stage(engine, capsys):
    """resident + transient + KV against working_set / FIT_HEADROOM: a
    model whose resident + KV fits but whose one staged tensor tips it is
    refused, with the transient in the printed arithmetic and the stage
    named; the bare accounting call (no transient) is unchanged."""
    from drinkme.suggest import FIT_HEADROOM

    pf = engine.fit_check
    ws = 11 * GIB
    kv = 4096 * 8192  # 32 MiB
    limit = ws / FIT_HEADROOM  # 10 GiB
    fits_alone = int(limit - kv - 0.5 * GIB)
    pf(fits_alone, 4096, 8192, ws, transient=0, stage="measured")
    assert "fit check (measured): ok" in capsys.readouterr().err
    with pytest.raises(SystemExit, match=r"refusing to load \(estimate\).*transient 1\.00 GiB") as ei:
        pf(fits_alone, 4096, 8192, ws, transient=1 * GIB, stage="estimate")
    assert "resident" in str(ei.value) and f"/ {FIT_HEADROOM} headroom" in str(ei.value)
    pf(fits_alone, 4096, 8192, None, transient=1 * GIB, stage="estimate")
    assert "fit check (estimate): no Metal device" in capsys.readouterr().err


# ------------------------------------------------ the host's memory, macOS --
#
# Recorded on the 24 GB M4 these checks were written for, with another
# user's session resident: `vm_stat` at 2026-09-17 16:37 and
# `memory_pressure` at 2026-09-18 22:28, verbatim — memory_pressure's
# counts carry a trailing space.

VM_STAT_M4 = (
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages free:                               46159.\n"
    "Pages active:                            466159.\n"
    "Pages inactive:                          586881.\n"
    "Pages speculative:                         6673.\n"
    "Pages throttled:                              0.\n"
    "Pages wired down:                        202902.\n"
    "Pages purgeable:                           1563.\n"
    "\"Translation faults\":                3933802562.\n"
    "Pages copy-on-write:                  120538523.\n"
    "Pages zero filled:                   2250737262.\n"
    "Pages reactivated:                    116730354.\n"
    "Pages purged:                          28072683.\n"
    "File-backed pages:                       312941.\n"
    "Anonymous pages:                         746772.\n"
    "Pages stored in compressor:             1443392.\n"
    "Pages occupied by compressor:            229397.\n"
    "Decompressions:                       184521100.\n"
    "Compressions:                         247608344.\n"
    "Pageins:                               75494652.\n"
    "Pageouts:                                572262.\n"
    "Swapins:                                2274656.\n"
    "Swapouts:                               4821464.\n"
)

MEMORY_PRESSURE_M4 = (
    "The system has 25769803776 (1572864 pages with a page size of 16384).\n"
    "\n"
    "Stats: \n"
    "Pages free: 78444 \n"
    "Pages purgeable: 30310 \n"
    "Pages purged: 31218052 \n"
    "\n"
    "Swap I/O:\n"
    "Swapins: 2315113 \n"
    "Swapouts: 4851198 \n"
    "\n"
    "Page Q counts:\n"
    "Pages active: 551413 \n"
    "Pages inactive: 393324 \n"
    "Pages speculative: 173370 \n"
    "Pages throttled: 0 \n"
    "Pages wired down: 165066 \n"
    "\n"
    "Compressor Stats:\n"
    "Pages used by compressor: 176739 \n"
    "Pages decompressed: 187165706 \n"
    "Pages compressed: 251409900 \n"
    "\n"
    "File I/O:\n"
    "Pageins: 81701446 \n"
    "Pageouts: 657137 \n"
    "\n"
    "System-wide memory free percentage: 77%\n"
)


def test_parse_vm_stat_reads_the_m4s_recorded_outputs(engine):
    """free + inactive + speculative + purgeable pages x the stated page
    size, from either tool's text; not "Pages purged", not the percentage
    line; None when a counter or the page size is missing."""
    assert engine.parse_vm_stat(VM_STAT_M4) == (46159 + 586881 + 6673 + 1563) * 16384
    assert engine.parse_vm_stat(MEMORY_PRESSURE_M4) == (78444 + 393324 + 173370 + 30310) * 16384
    assert round(engine.parse_vm_stat(VM_STAT_M4) / GIB, 2) == 9.79
    assert round(engine.parse_vm_stat(MEMORY_PRESSURE_M4) / GIB, 2) == 10.31
    assert engine.parse_vm_stat(VM_STAT_M4.replace("page size of 16384 bytes", "")) is None
    no_inactive = "\n".join(ln for ln in VM_STAT_M4.splitlines() if "inactive" not in ln)
    assert engine.parse_vm_stat(no_inactive) is None
    assert engine.parse_vm_stat("") is None


def test_host_budget_is_vm_stat_plus_what_mlx_already_holds(engine, monkeypatch):
    """On macOS: `vm_stat`'s available bytes plus mlx's active and cached
    bytes (at the measured check the loaded weights have left the free
    pages); elsewhere, or when vm_stat fails, None."""
    import subprocess

    runs = []

    def fake_run(argv, **kw):
        runs.append(argv)
        return types.SimpleNamespace(stdout=VM_STAT_M4, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(engine.mx, "get_active_memory", lambda: 5 * GIB, raising=False)
    monkeypatch.setattr(engine.mx, "get_cache_memory", lambda: 1 * GIB, raising=False)
    monkeypatch.setattr(engine.sys, "platform", "linux")
    assert engine._host_budget() is None and runs == []
    monkeypatch.setattr(engine.sys, "platform", "darwin")
    assert engine._host_budget() == engine.parse_vm_stat(VM_STAT_M4) + 6 * GIB
    assert runs == [["vm_stat"]]

    def broken(argv, **kw):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", broken)
    assert engine._host_budget() is None


def test_the_host_bounds_the_budget_and_the_refusal_names_both(engine, capsys):
    """The budget is min(working set, host); the refusal names both, says
    which bound, and what to do — the largest --ctx that fits (which does
    fit, and 1024 more does not) and, host-bound, closing apps."""
    from drinkme.suggest import FIT_HEADROOM

    pf = engine.fit_check
    ws, host = 16 * GIB, 10 * GIB
    kv, pre = 144 * 1024, 64 * 1024  # a 4B's KV per token; a prefill term per token
    resident, transient = int(5.5 * GIB), int(0.7 * GIB)
    with pytest.raises(SystemExit) as ei:
        pf(resident, kv, 32768, ws, transient=transient, stage="estimate",
           prefill_per_token=pre, host=host)
    msg = str(ei.value)
    assert "refusing to load (estimate)" in msg
    assert "GPU working set 16.00 GiB" in msg and "host available 10.00 GiB" in msg
    assert "The memory macOS has free for this process is the bound" in msg
    assert "close apps" in msg and "prefill 2.00 GiB" in msg
    fit_ctx = int(msg.split("--ctx ")[1].split()[0])
    assert fit_ctx % 1024 == 0 and 1024 <= fit_ctx < 32768
    pf(resident, kv, fit_ctx, ws, transient=transient, prefill_per_token=pre, host=host)
    assert "fit check (measured): ok" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        pf(resident, kv, fit_ctx + 1024, ws, transient=transient, prefill_per_token=pre, host=host)
    # the same need under a roomier host: the working set binds, and the line
    # says so without telling anyone to close their apps
    need = resident + transient + (kv + pre) * 65536
    assert need * FIT_HEADROOM > ws
    with pytest.raises(SystemExit) as ei:
        pf(resident, kv, 65536, ws, transient=transient, prefill_per_token=pre, host=64 * GIB)
    assert "The GPU working set is the bound" in str(ei.value)
    assert "close apps" not in str(ei.value)
    # nothing left for any context: a smaller model, by name
    with pytest.raises(SystemExit, match="no --ctx fits it: serve a smaller model"):
        pf(resident, kv, 8192, ws, transient=transient, prefill_per_token=pre, host=6 * GIB)
    # host unread (vm_stat failed): the working set alone, and the line says so
    pf(resident, kv, 8192, ws, transient=transient, prefill_per_token=pre, host=None)
    err = capsys.readouterr().err
    assert "host available unread" in err and "= 16.00 GiB / 1.1 headroom" in err


def test_both_loaders_refuse_on_the_hosts_memory_before_any_weight(toy, engine, monkeypatch,
                                                                    capsys):
    """A working set that fits and a host that does not: the estimate
    refuses on the host's number, before the first door — the same
    ordering as the working-set refusal, on the other budget."""
    model_dir, pack_dir = toy
    _arm_traps(engine, monkeypatch)
    monkeypatch.setattr(engine, "_working_set", lambda: 16 * GIB)
    monkeypatch.setattr(engine, "_host_budget", lambda: 1 * GIB // 1024)
    for load in (lambda: engine.load_compressed_mlx(model_dir, None, pack_dir, path="reference"),
                 lambda: engine.load_stock_mlx(model_dir, None)):
        with pytest.raises(SystemExit, match=r"refusing to load \(estimate\).*"
                                             r"The memory macOS has free for this process is the bound"):
            load()
        assert "fit check (measured)" not in capsys.readouterr().err


def test_the_reference_path_is_charged_every_dense_plane(toy, engine, monkeypatch):
    """The fused path's transient is one dense plane (or the largest raw
    tensor); the reference path decodes every plane while the graph is
    built, so the estimate check is charged their sum; the twin holds
    those planes as resident (decoded once at load, none built per
    forward), so it is charged them once, there, and its decode as the
    transient; and every path is charged the prefill's activations per
    token."""
    model_dir, pack_dir = toy
    _arm_traps(engine, monkeypatch)
    seen = []
    monkeypatch.setattr(engine, "fit_check", lambda *a, **k: seen.append((a, k)))
    est = engine.estimate_load_bytes(model_dir, pack_dir, tie=False)
    assert est.dense_total == 10 * 2 * 1024 * 1024  # the toy's ten 1024 x 1024 planes
    twin = engine.estimate_load_bytes(model_dir, pack_dir, tie=False, twin=True)
    streams = est.resident - (twin.resident - est.dense_total)  # the packed members twin drops
    assert twin.dense_total == 0 and streams > 0
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    for path, resident, want in (("fused", est.resident, est.transient),
                                 ("reference", est.resident, est.dense_total),
                                 (engine.TWIN, twin.resident, twin.transient)):
        seen.clear()
        with pytest.raises(Materialized):
            engine.load_compressed_mlx(model_dir, None, pack_dir, path=path)
        (a, k), = seen  # the estimate only: the first door comes before the measured check
        assert k["stage"] == "estimate" and a[0] == resident and k["transient"] == want, path
        assert k["prefill_per_token"] == engine.prefill_bytes_per_token(cfg)
