"""`drinkme bench`'s arms route alike.

Before this, the bench's stock arm (arms.load_stock_streaming / load_cpu,
timed by timed_decode) kept transformers' torch DeltaNet recurrence and
F.linear on the 48-row projections, while its compressed arm got
swap.NarrowLinear through engines.build_compressed_model and no fla route —
deltanet.route was called from engines.load_stock / load_compressed only,
never from run_arms. On the 27B that lopsides the stock/compressed ratio by
~7% in the compressed arm's favour.

Now ONE helper — serving/kernel_route.route_kernels(model, device) — binds
the recurrence kernel and adopts the narrow Linears, honoring
DRINKME_DELTANET_KERNEL and DRINKME_NARROW_GEMV, for engines.load_stock,
engines.load_compressed AND every arm run_arms times, and returns a plain
record of what it did. The bench prints one line per arm, keeps each arm's
record in raw (`raw.<arm>_routing`), and REFUSES to write a record whose
measured arms routed differently.

CPU only, except the one gated test at the end: the toy is
tests/test_serving_mtp.py's hybrid qwen3_5 (two linear_attention layers).
On CPU the recurrence is the torch reference on every arm — the PARITY is
what these tests pin, not fla. The modeling module's recurrence global is
process-wide by design and is put back after every test that binds it.
"""

from __future__ import annotations

import inspect
import types

import pytest
import torch

from drinkme import arms
from drinkme.serving import checkpoint, deltanet, deltanet_conv
from test_serving_mtp import _cfg

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

gpu_gate = pytest.mark.skipif(not torch.cuda.is_available(),
                              reason="fla's fused recurrence is triton; needs a device")

GB = 1_000_000_000
# what route_kernels does to the toy on the CPU: the torch reference (fla's
# kernel is triton), the narrow GEMV allowed but nothing adopted (the toy's
# in_proj_a/b are 4 x 64, under NarrowLinear.MIN_COLS; and it is not cuda)
TORCH_ON_CPU = {"deltanet_kernel": "torch", "deltanet_conv": "torch", "narrow_gemv": True, "narrow_count": 0,
                "stock_gemv": "linear", "raw_gemv": "stock"}
FLA_96 = {"deltanet_kernel": "fla", "deltanet_conv": "fla", "narrow_gemv": True, "narrow_count": 96}
TORCH_96 = {"deltanet_kernel": "torch", "deltanet_conv": "torch", "narrow_gemv": True, "narrow_count": 96}
STAT = {"name": "l", "shape": [1024, 1024], "numel": 1024 * 1024, "bits": 12 * 1024 * 1024,
        "bpw": 12.0, "format_version": 1, "codec": "radix", "profile": "sip",
        "dtype": None, "verified": True}


@pytest.fixture(scope="module")
def toy():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_cfg()).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def modeling(toy):
    """Put both process-wide DeltaNet callables back after each test."""
    mods = deltanet.deltanet_modules(toy)
    assert len(mods) == 2  # the toy's two linear_attention layers
    m = deltanet._modeling_module(mods[0])
    before = getattr(m, deltanet.NAME)
    before_conv = getattr(m, deltanet_conv.NAME)
    yield m
    setattr(m, deltanet.NAME, before)
    setattr(m, deltanet_conv.NAME, before_conv)


# ------------------------------------------------------------ the helper --


def test_route_kernels_record_on_the_toy_hybrid(toy, modeling, monkeypatch, capsys):
    """One call, one plain record of what it did, both knobs honored, the
    two boot lines printed as the loaders always printed them."""
    from drinkme.serving.kernel_route import describe, route_kernels

    monkeypatch.delenv(deltanet.ENV, raising=False)
    monkeypatch.delenv("DRINKME_NARROW_GEMV", raising=False)
    rec = route_kernels(toy, "cpu")
    assert rec == TORCH_ON_CPU
    assert describe(rec) == "deltanet torch, narrow 0, conv torch"
    assert "[drinkme] deltanet recurrence: torch reference (device cpu)" in capsys.readouterr().out
    assert getattr(modeling, deltanet.NAME) is deltanet.torch_reference(modeling)

    monkeypatch.setenv("DRINKME_NARROW_GEMV", "0")
    rec = route_kernels(toy, "cpu")
    assert rec == {"deltanet_kernel": "torch", "deltanet_conv": "torch", "narrow_gemv": False, "narrow_count": 0,
                   "stock_gemv": "linear", "raw_gemv": "stock"}
    assert describe(rec) == "deltanet torch, narrow off (DRINKME_NARROW_GEMV=0), conv torch"
    # the stock GEMV is named only when it is not F.linear
    assert describe(dict(rec, stock_gemv="mv")) == \
        "deltanet torch, narrow off (DRINKME_NARROW_GEMV=0), conv torch, stock gemv mv"
    # and the raw GEMV only when it is not the stock GEMV
    assert describe(dict(rec, stock_gemv="mv", raw_gemv="twin")) == \
        "deltanet torch, narrow off (DRINKME_NARROW_GEMV=0), conv torch, stock gemv mv, raw gemv twin"
    assert "[drinkme] narrow gemv: DRINKME_NARROW_GEMV=0" in capsys.readouterr().out

    # no DeltaNet layers (a dense model): nothing bound, and the record says so
    assert route_kernels(torch.nn.Linear(4, 4), "cpu")["deltanet_kernel"] == "none"


def test_the_loaders_and_the_bench_route_through_the_one_helper():
    """engines.load_stock, engines.load_compressed and arms.run_arms all go
    through kernel_route.route_kernels, and NOTHING else in either module
    calls deltanet.route or swap.install_narrow — the two calls whose
    placement let the paths drift. build_compressed_model hands back an
    unrouted tree: the bench's pack arm is routed where its other arms are
    (run_arms), never inside a loader only one arm uses."""
    from drinkme.serving import engines

    for fn in (engines.load_stock, engines.load_compressed):
        assert "route_kernels(model, device)" in inspect.getsource(fn), fn.__name__
    assert "route_kernels(" not in inspect.getsource(engines.build_compressed_model)
    assert "route_kernels(" in inspect.getsource(arms.run_arms) \
        or "_route_arm(" in inspect.getsource(arms.run_arms)
    for module in (engines, arms):
        src = inspect.getsource(module)
        assert "deltanet.route(" not in src, module.__name__
        assert "install_narrow(" not in src, module.__name__


# ---------------------------------------------------------- run_arms --


def _stub_device(monkeypatch, ids: torch.Tensor | None = None):
    """The tests' run_arms harness (tests/test_bench_one_source.py's):
    no device, no hub, no timing — every arm's loader hands back a real
    module tree and the routing runs for real on it."""
    class _Ids:
        shape = (1, 4)

        def cuda(self):
            return self

    class _Tok:
        def __call__(self, text, **kw):
            if kw.get("return_tensors"):
                return types.SimpleNamespace(input_ids=ids if ids is not None else _Ids())
            return types.SimpleNamespace(input_ids=[3, 4, 5])

    monkeypatch.setattr(checkpoint, "tokenizer", lambda path, revision: _Tok())
    monkeypatch.setattr(arms, "resolve_source", lambda m, r, p: ("/toy/snapshot", "a" * 40))
    monkeypatch.setattr(arms, "measure_bandwidth", lambda: {
        "probe_bytes": GB, "read_bytes_s": 200 * GB, "copy_bytes_s": 180 * GB, "device": "cuda"})
    monkeypatch.setattr(arms, "live_available_bytes", lambda kind: 100 * GB)
    if ids is None:  # the CPU harness: no device at all
        monkeypatch.setattr(arms.torch, "tensor", lambda *a, **k: _Ids())
        monkeypatch.setattr(arms.torch.cuda, "get_device_name", lambda i: "test-gpu")
        monkeypatch.setattr(arms, "free_all", lambda: None)
        monkeypatch.setattr(arms, "vram_bytes", lambda: 500_000_000)
        monkeypatch.setattr(arms, "timed_decode", lambda model, ids, **k: (None, [10.0, 10.0, 10.0]))
        monkeypatch.setattr(arms, "timed_prefill", lambda model, ids, **k: [500.0, 500.0, 500.0])
        monkeypatch.setattr(arms, "timed_ttft", lambda model, ids, **k: [0.05, 0.05, 0.05])


def _toy_loaders(monkeypatch, model):
    """Every arm's loader hands back the toy's tree: the arms differ in
    their Linears, which routing never touches (the recurrence is the
    modeling module's global; the narrow projections are the raw remainder
    on every arm)."""
    monkeypatch.setattr(arms, "load_stock_streaming", lambda *a, **k: model)
    monkeypatch.setattr(arms, "load_cpu", lambda *a, **k: model)
    monkeypatch.setattr(arms, "load_twin_streaming", lambda *a, **k: (model, []))
    monkeypatch.setattr(arms, "load_compressed_streaming", lambda *a, **k: (model, [dict(STAT)]))


def _run(**kw):
    # spec=False: the toy is a real tree, and the speculation pass would ask
    # the head loader about the stubbed snapshot path (tests/test_bench_spec.py
    # runs the pass over this toy with a head it can load)
    args = dict(run_stock=True, stock_memory_kind="vram", run_twin=True, stock_loader="stream",
                spec=False)
    args.update(kw)
    return arms.run_arms("org/toy-hybrid", None, "hi", **args)


def test_run_arms_routes_every_arm_alike_and_records_it(toy, modeling, monkeypatch, capsys):
    """The parity: stock, twin and compressed all go through route_kernels
    on the device the bench times on, land the SAME record in raw, and
    each says so on one line beside its pass. DRINKME_DELTANET_KERNEL=torch
    so the route decides before touching a device (the knob wins before
    any import, tests/test_serving_deltanet.py) — on the CPU suite there is
    none, and the parity is the subject, not fla."""
    monkeypatch.setenv(deltanet.ENV, "torch")
    monkeypatch.delenv("DRINKME_NARROW_GEMV", raising=False)
    _stub_device(monkeypatch)
    _toy_loaders(monkeypatch, toy)
    r = _run()
    assert r["stock_routing"] == TORCH_ON_CPU
    assert r["twin_routing"] == TORCH_ON_CPU
    assert r["compressed_routing"] == TORCH_ON_CPU
    assert r["stock_outcome"] == "measured" and r["twin_outcome"] == "measured"
    out = capsys.readouterr().out
    for arm in ("stock", "twin", "compressed"):
        assert f"{arm} arm: deltanet torch, narrow 0" in out, out
    # the from_pretrained stock loader is routed the same way
    r = _run(stock_loader="from_pretrained")
    assert r["stock_routing"] == r["compressed_routing"] == TORCH_ON_CPU


def test_run_arms_fit_point_records_the_compressed_arms_route_alone(toy, modeling, monkeypatch, capsys):
    """Neither stock nor twin ran: one arm, one record, nothing to compare
    — and no stock_routing / twin_routing beside the skipped outcomes."""
    monkeypatch.setenv(deltanet.ENV, "torch")
    _stub_device(monkeypatch)
    _toy_loaders(monkeypatch, toy)
    r = _run(run_stock=False, run_twin=False)
    assert r["compressed_routing"] == TORCH_ON_CPU
    assert "stock_routing" not in r and "twin_routing" not in r
    assert capsys.readouterr().out.count("arm: deltanet torch, narrow 0") == 1


def test_run_arms_refuses_a_record_whose_arms_routed_differently(toy, modeling, monkeypatch):
    """The stock arm routed torch (say fla's import failed for it, or the
    probe did) and the compressed arm (the second pass) routed fla: the
    bench REFUSES, by name, before the second arm is timed — a lopsided
    ratio is never written."""
    _stub_device(monkeypatch)
    _toy_loaders(monkeypatch, toy)
    routed, timed = [], []

    def lopsided(model, device):
        routed.append(device)
        return TORCH_96 if len(routed) == 1 else FLA_96

    monkeypatch.setattr(arms, "route_kernels", lopsided)
    monkeypatch.setattr(arms, "timed_decode",
                        lambda model, ids, **k: timed.append(1) or (None, [10.0, 10.0, 10.0]))
    with pytest.raises(SystemExit, match="refusing to write the record") as e:
        _run()
    msg = str(e.value)
    assert "stock deltanet torch, narrow 96" in msg and "compressed deltanet fla, narrow 96" in msg
    assert routed == ["cuda", "cuda"]  # refused at the compressed arm, before the twin loaded
    assert timed == [1]  # the stock arm was timed; the compressed arm never was

    # the twin skipped (a fit point with a stock arm): stock vs compressed
    routed.clear()
    with pytest.raises(SystemExit, match="stock deltanet torch, narrow 96.*compressed deltanet fla"):
        _run(run_twin=False)
    # and the stock arm skipped: compressed vs twin
    routed.clear()
    with pytest.raises(SystemExit, match="compressed deltanet torch, narrow 96.*twin deltanet fla"):
        _run(run_stock=False)


def test_run_arms_alike_under_the_knobs_is_not_refused(toy, modeling, monkeypatch):
    """Alike is alike whatever it is: three arms on the torch reference with
    the GEMV off (the A/B instrument's own configuration) write a record."""
    _stub_device(monkeypatch)
    _toy_loaders(monkeypatch, toy)
    off = {"deltanet_kernel": "torch", "deltanet_conv": "torch", "narrow_gemv": False, "narrow_count": 0}
    monkeypatch.setattr(arms, "route_kernels", lambda model, device: dict(off))
    r = _run()
    assert r["stock_routing"] == r["twin_routing"] == r["compressed_routing"] == off


def test_refuse_unless_routed_alike_is_pure_and_names_every_arm():
    arms.refuse_unless_routed_alike("m", {"stock": FLA_96, "twin": dict(FLA_96), "compressed": dict(FLA_96)})
    arms.refuse_unless_routed_alike("m", {"compressed": FLA_96})  # one arm: nothing to compare
    arms.refuse_unless_routed_alike("m", {})
    with pytest.raises(SystemExit) as e:
        arms.refuse_unless_routed_alike("m", {"stock": FLA_96, "twin": FLA_96, "compressed": TORCH_96})
    msg = str(e.value)
    assert msg.startswith("drinkme: refusing to write the record: m")
    assert "stock deltanet fla, narrow 96" in msg and "compressed deltanet torch, narrow 96" in msg
    # a different narrow count is a different route too
    with pytest.raises(SystemExit, match="narrow 96.*narrow 0"):
        arms.refuse_unless_routed_alike("m", {"stock": FLA_96, "compressed": dict(FLA_96, narrow_count=0)})
    # Prefill ratios also require the same convolution implementation.
    with pytest.raises(SystemExit, match="conv fla.*conv torch"):
        arms.refuse_unless_routed_alike(
            "m", {"stock": FLA_96, "compressed": dict(FLA_96, deltanet_conv="torch")})


def test_a_failed_stock_load_leaves_no_stock_route_to_compare(toy, modeling, monkeypatch):
    """failed_load drops every stock_* field, the route with them — the
    record compares the arms that ran."""
    monkeypatch.setenv(deltanet.ENV, "torch")
    _stub_device(monkeypatch)
    _toy_loaders(monkeypatch, toy)
    monkeypatch.setattr(arms, "load_stock_streaming", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("HIP out of memory. Tried to allocate 15.26 GiB")))
    r = _run(stock_bf16_bytes=15 * GB)
    assert r["stock_outcome"] == "failed_load" and "stock_routing" not in r
    assert r["twin_routing"] == r["compressed_routing"] == TORCH_ON_CPU


# ---------------------------------------------------------------- GPU --


@gpu_gate
def test_gpu_route_kernels_counts_the_narrow_linears_the_tree_holds(monkeypatch, capsys):
    """The 27B's shape (48 x 5120, two per DeltaNet layer) on a tree with
    no DeltaNet: narrow_count is what the tree HOLDS after the call — the
    same on a second call (nothing re-adopted; the record does not change
    under idempotence) — and 0 with the GEMV off, deltanet 'none'."""
    from drinkme.codec.swap import NarrowLinear
    from drinkme.serving.kernel_route import describe, route_kernels

    def lin(r, c, dtype=torch.bfloat16):
        return torch.nn.Linear(c, r, bias=False, dtype=dtype)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.big = lin(1024, 5120)   # the codec's shape: not narrow
            self.a = lin(48, 5120)       # narrow
            self.b = lin(48, 5120)       # narrow

    monkeypatch.delenv(deltanet.ENV, raising=False)
    monkeypatch.delenv("DRINKME_NARROW_GEMV", raising=False)
    m = torch.nn.Sequential(Block(), Block()).to("cuda")
    rec = route_kernels(m, "cuda")
    assert rec == {"deltanet_kernel": "none", "deltanet_conv": "none", "narrow_gemv": True, "narrow_count": 4}
    assert describe(rec) == "deltanet none, narrow 4, conv none"
    assert "[drinkme] narrow gemv: 4 x NarrowLinear (48x5120 bf16" in capsys.readouterr().out
    assert sum(isinstance(x, NarrowLinear) for x in m.modules()) == 4
    assert route_kernels(m, "cuda") == rec  # idempotent, and the record says the same
    monkeypatch.setenv("DRINKME_NARROW_GEMV", "0")
    fresh = torch.nn.Sequential(Block()).to("cuda")
    assert route_kernels(fresh, "cuda") == {"deltanet_kernel": "none", "deltanet_conv": "none", "narrow_gemv": False,
                                            "narrow_count": 0}


@gpu_gate
def test_gpu_run_arms_routes_fla_on_every_arm_of_the_toy_hybrid(toy, modeling, monkeypatch, capsys):
    """The GPU step: run_arms over the toy hybrid ON THE DEVICE —
    the loaders hand every arm the toy's tree on cuda, everything else is
    the bench's own (the route with fla's real probe on the toy's head
    shape, the narrow adoption, the timed decodes, prefill, TTFT, the
    record). Every arm line says `deltanet fla` and raw carries the same
    record three times. narrow is 0 on every arm: the toy's in_proj_a/b
    are 4 x 64, under NarrowLinear.MIN_COLS — the 27B's 96 is the RC
    sweep's to show."""
    monkeypatch.delenv(deltanet.ENV, raising=False)
    monkeypatch.delenv(deltanet_conv.ENV, raising=False)
    monkeypatch.delenv("DRINKME_NARROW_GEMV", raising=False)
    model = toy.to("cuda")
    try:
        _stub_device(monkeypatch, ids=torch.tensor([[3, 4, 5, 6]]))
        monkeypatch.setattr(arms, "build_prefill_ids", lambda tok, length=32: list(range(3, 3 + length)))
        _toy_loaders(monkeypatch, model)
        r = _run()
        fla = {"deltanet_kernel": "fla", "deltanet_conv": "fla" if torch.version.hip else "torch",
               "narrow_gemv": True, "narrow_count": 0}
        assert r["stock_routing"] == fla, r["stock_routing"]
        assert r["twin_routing"] == fla and r["compressed_routing"] == fla
        out = capsys.readouterr().out
        assert out.count("arm: deltanet fla, narrow 0") == 3, out
        assert "fused_recurrent_gated_delta_rule (Triton" in out
        assert r["stock_decode_tok_s"] > 0 and r["compressed_decode_tok_s"] > 0
        assert r["twin_decode_tok_s"] > 0 and r["compressed_prefill_tok_s"] > 0
        bound = getattr(modeling, deltanet.NAME)
        assert getattr(bound, "route", None) == "fla"  # what the timed decodes ran
    finally:
        toy.to("cpu")
