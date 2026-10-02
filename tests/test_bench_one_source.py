"""An unpinned benchmark loads every arm from ONE snapshot.

With `revision=None` (the menu rows Qwen3-0.6B / 14B / 32B, or a hub id
by hand), loaders that each resolve their own source would split: the
stock and twin arms on the current default ref, the compressed arm, under
--pack-dir, on the pack's recorded source commit, and the tokenizer from
the caller's coordinates a fourth time. A pack bound to an older commit
than the local default would then compare two revisions and publish the
menu's coordinate (None: no revision at all) as the record's
model.revision.

So arms.run_arms resolves the source once, before the tokenizer or any
arm loads (serving.checkpoint.resolve_source): the pack's bound snapshot
under --pack-dir, else the coordinates resolved now; every loader takes
that directory as `snap`, and the record's model.revision is the resolved
commit. Network-free: two toy "hub snapshots" laid out the way the HF
cache does (`models--org--toy/snapshots/<sha>/`), a fake snapshot_dir
whose default ref is the newer one, a pack cut from the older one.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch

from drinkme import arms, bench
from drinkme.codec import pack as P
from drinkme.serving import checkpoint

REPO = "org/toy"
SHA_A = "a" * 40  # the commit the pack is bound to
SHA_B = "b" * 40  # where the hub's default ref points now
GB = 1_000_000_000


class Two:
    """Two snapshots of one repo, a pack cut from the older, and the fake
    resolver that makes the newer the default ref."""

    def __init__(self, root: str):
        from tests.test_pack_identity import _toy_checkpoint

        cache = os.path.join(root, "models--org--toy", "snapshots")
        self.snap_a = os.path.join(cache, SHA_A)
        self.snap_b = os.path.join(cache, SHA_B)
        _toy_checkpoint(self.snap_a, seed=1)
        _toy_checkpoint(self.snap_b, seed=2)  # same shapes, different weights: "a later commit"
        self.pack_dir = os.path.join(root, "pack-at-A")

    def snapshot_dir(self, repo: str, revision: str | None) -> str:
        if os.path.isdir(repo):
            return repo
        assert repo == REPO, repo
        if revision in (None, "main"):
            return self.snap_b  # the moving default
        return {SHA_A: self.snap_a, SHA_B: self.snap_b}[revision]

    def patch(self, monkeypatch) -> None:
        monkeypatch.setattr(checkpoint, "snapshot_dir", self.snapshot_dir)
        monkeypatch.setattr(arms, "snapshot_dir", self.snapshot_dir)


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    t = Two(str(tmp_path_factory.mktemp("one_source")))
    orig = checkpoint.snapshot_dir
    checkpoint.snapshot_dir = t.snapshot_dir  # pack_model reads it off the module at call time
    try:
        P.pack_model(REPO, SHA_A, t.pack_dir, progress=lambda *_: None)
    finally:
        checkpoint.snapshot_dir = orig
    return t


def _embed(model) -> torch.Tensor:
    """A raw-remainder tensor every arm streams from the checkpoint."""
    return model.get_submodule("model.embed_tokens").weight.detach().float()


# ------------------------------------------------------------ the resolver --


def test_resolve_source_binds_the_run_to_the_packs_commit_or_the_ref_resolved_once(two, monkeypatch):
    two.patch(monkeypatch)
    snap, rev = checkpoint.resolve_source(REPO, None, two.pack_dir)
    assert snap == two.snap_a and rev == SHA_A
    snap, rev = checkpoint.resolve_source(REPO, None, None)
    assert snap == two.snap_b and rev == SHA_B  # no pack: the default ref, resolved now
    with pytest.raises(checkpoint.PackSourceMismatch, match="bound to"):
        checkpoint.resolve_source(REPO, SHA_B, two.pack_dir)  # a pinned disagreement refuses


# ------------------------------------------------------- the real loaders --


def test_every_torch_loader_reads_the_snapshot_it_is_handed(two, monkeypatch):
    """Each loader on the CPU, handed snap A while the default ref is B:
    the raw remainder it attaches is A's. Without `snap`, the same loader
    follows the default — the split run_arms now prevents."""
    two.patch(monkeypatch)
    a = _embed(arms.load_cpu(two.snap_a))
    b = _embed(arms.load_cpu(two.snap_b))
    assert not torch.equal(a, b)
    assert torch.equal(_embed(arms.load_stock_streaming(REPO, None, device="cpu", snap=two.snap_a)), a)
    assert torch.equal(_embed(arms.load_stock_streaming(REPO, None, device="cpu")), b)
    comp, stats = arms.load_compressed_streaming(REPO, None, device="cpu", snap=two.snap_a)
    assert torch.equal(_embed(comp), a)
    twin, _ = arms.load_twin_streaming(REPO, None, device="cpu", snap=two.snap_a,
                                       plan=arms.twin_plan(stats))
    assert torch.equal(_embed(twin), a)
    packed, stats, meta = arms.load_pack_compressed(REPO, None, two.pack_dir, device="cpu",
                                                    snap=two.snap_a)
    assert torch.equal(_embed(packed), a) and meta["source"]["revision"] == SHA_A and stats
    assert torch.equal(_embed(arms.load_cpu(REPO, None, snap=two.snap_a)), a)


# ---------------------------------------------------------- run_arms --


def _stub_device(monkeypatch):
    class _Ids:
        shape = (1, 4)

        def cuda(self):
            return self

    monkeypatch.setattr(arms.torch, "tensor", lambda *a, **k: _Ids())
    monkeypatch.setattr(arms.torch.cuda, "get_device_name", lambda i: "test-gpu")
    monkeypatch.setattr(arms, "measure_bandwidth", lambda: {
        "probe_bytes": GB, "read_bytes_s": 200 * GB, "copy_bytes_s": 180 * GB, "device": "cuda"})
    monkeypatch.setattr(arms, "free_all", lambda: None)
    monkeypatch.setattr(arms, "refuse_if_stock_wont_fit", lambda *a, **k: None)
    monkeypatch.setattr(arms, "refuse_if_compressed_wont_fit", lambda *a, **k: None)
    monkeypatch.setattr(arms, "live_available_bytes", lambda kind: 100 * GB)
    monkeypatch.setattr(arms, "vram_bytes", lambda: 500_000_000)
    monkeypatch.setattr(arms, "timed_decode", lambda model, ids, **k: (None, [10.0, 10.0, 10.0]))
    monkeypatch.setattr(arms, "timed_prefill", lambda model, ids, **k: [500.0, 500.0, 500.0])
    monkeypatch.setattr(arms, "timed_ttft", lambda model, ids, **k: [0.05, 0.05, 0.05])
    return _Ids


def _recording_loaders(two, monkeypatch, sources: dict):
    """Stub arms whose first line is the real loaders' first line: read
    from `snap` when handed one, else resolve as that loader always did —
    so what each arm WOULD have read is what gets recorded."""

    class _Model:
        def cuda(self):
            return self

        def parameters(self):
            return iter([torch.zeros(1024, 1024, dtype=torch.bfloat16)])

    stat = {"name": "l", "shape": [1024, 1024], "numel": 1024 * 1024, "bits": 12 * 1024 * 1024,
            "bpw": 12.0, "format_version": 1, "codec": "radix", "profile": "sip", "dtype": None, "verified": True}

    def stock(model_id, revision=None, device="cuda", snap=None):
        sources["stock"] = snap or arms.snapshot_dir(model_id, revision)
        return _Model()

    def from_pretrained(model_id, revision=None, config=None, device_map=None, snap=None):
        sources["stock"] = snap or arms.snapshot_dir(model_id, revision)
        return _Model()

    def twin(model_id, revision=None, device="cuda", snap=None, plan=None):
        sources["twin"] = snap or arms.snapshot_dir(model_id, revision)
        return _Model(), []

    def compressed(model_id, revision=None, device="cuda", snap=None):
        sources["compressed"] = snap or arms.snapshot_dir(model_id, revision)
        return _Model(), [dict(stat)]

    def packed(model_id, revision, pack_dir, device="cuda", snap=None):
        import json

        meta = json.load(open(os.path.join(pack_dir, "meta.json")))
        sources["compressed"] = snap or checkpoint.resolve_pack_source(model_id, revision, meta, pack_dir)
        return _Model(), [dict(stat)], meta

    _Ids = _stub_device(monkeypatch)

    class _Tok:
        def __call__(self, text, **kw):
            if kw.get("return_tensors"):
                return types.SimpleNamespace(input_ids=_Ids())
            return types.SimpleNamespace(input_ids=[1, 2, 3])

    def tokenizer(path, **kw):
        sources["tokenizer"] = two.snapshot_dir(path, kw.get("revision"))
        return _Tok()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer)))
    monkeypatch.setattr(arms, "load_stock_streaming", stock)
    monkeypatch.setattr(arms, "load_cpu", from_pretrained)
    monkeypatch.setattr(arms, "load_twin_streaming", twin)
    monkeypatch.setattr(arms, "load_compressed_streaming", compressed)
    monkeypatch.setattr(arms, "load_pack_compressed", packed)


def test_run_arms_hands_every_arm_and_the_tokenizer_the_packs_bound_snapshot(two, monkeypatch):
    two.patch(monkeypatch)
    sources: dict = {}
    _recording_loaders(two, monkeypatch, sources)
    r = arms.run_arms(REPO, None, "hi", run_stock=True, stock_memory_kind="vram",
                      stock_bf16_bytes=1 * GB, run_twin=True, stock_loader="stream",
                      pack_dir=two.pack_dir)
    assert sources == {k: two.snap_a for k in ("tokenizer", "stock", "twin", "compressed")}, sources
    assert r["resolved_revision"] == SHA_A
    assert r["revision"] is None  # the coordinate asked for, kept in raw
    assert r["verification"] == "pack"  # --pack-dir: the pack's hashes matched at load, no fresh comparison
    # the from_pretrained stock loader takes the same snapshot
    sources.clear()
    arms.run_arms(REPO, None, "hi", run_stock=True, stock_memory_kind="vram",
                  stock_bf16_bytes=1 * GB, run_twin=True, stock_loader="from_pretrained",
                  pack_dir=two.pack_dir)
    assert sources == {k: two.snap_a for k in ("tokenizer", "stock", "twin", "compressed")}, sources


def test_run_arms_without_a_pack_resolves_the_ref_once_for_every_arm(two, monkeypatch):
    two.patch(monkeypatch)
    sources: dict = {}
    _recording_loaders(two, monkeypatch, sources)
    r = arms.run_arms(REPO, None, "hi", run_stock=True, stock_memory_kind="vram",
                      stock_bf16_bytes=1 * GB, run_twin=True, stock_loader="stream")
    assert sources == {k: two.snap_b for k in ("tokenizer", "stock", "twin", "compressed")}, sources
    assert r["resolved_revision"] == SHA_B
    assert r["verification"] == "roundtrip"  # the in-memory arm: each tensor freshly compared


def test_run_arms_refuses_a_pinned_revision_the_pack_disagrees_with_before_any_arm(two, monkeypatch):
    two.patch(monkeypatch)
    sources: dict = {}
    _recording_loaders(two, monkeypatch, sources)
    with pytest.raises(checkpoint.PackSourceMismatch, match="bound to"):
        arms.run_arms(REPO, SHA_B, "hi", run_stock=True, stock_memory_kind="vram",
                      stock_bf16_bytes=1 * GB, run_twin=True, pack_dir=two.pack_dir)
    assert sources == {}  # nothing loaded, not even the tokenizer


# ------------------------------------------------------------- the record --


def test_the_record_carries_the_resolved_commit_not_the_menus_coordinate(linux_host, monkeypatch):
    from tests.test_bench_record_shape import _fake_hw, _fake_raw

    raw = _fake_raw()
    raw["revision"] = None  # an unpinned menu row
    raw["resolved_revision"] = SHA_A
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _unpinned_suggestion())
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: raw)
    r = bench.run(no_gemma=True)
    assert r["model"]["revision"] == SHA_A
    assert r["raw"]["revision"] is None and r["raw"]["resolved_revision"] == SHA_A
    # arms that report no resolved commit cannot make a record
    del raw["resolved_revision"]
    with pytest.raises(ValueError, match="resolved_revision"):
        bench.run(no_gemma=True)


def _unpinned_suggestion():
    from drinkme.suggest import Model, Suggestion

    m = Model(name="ToyModel", hf_repo="toy/model", revision=None)
    return Suggestion(ratio=m, ratio_extra=[], fit=None, fit_knife_edge=False,
                      fit_honest_negative=False, notes=[])


# ------------------------------------------------------------ the Mac lane --


def test_mlx_run_arms_hands_both_loaders_and_the_tokenizer_the_packs_bound_snapshot(two, monkeypatch):
    """arms_mlx.run_arms has the same three doors (tokenizer, stock,
    compressed) and takes the same rule: the engine loaders are stubbed
    (the mlx runtime needs mlx to build one; here it needs nothing) and record
    the snapshot they are handed, resolving as the real ones do when
    handed none."""
    from drinkme import arms_mlx

    two.patch(monkeypatch)
    sources: dict = {}
    core = types.ModuleType("mlx.core")
    core.metal = types.SimpleNamespace(is_available=lambda: False)
    core.synchronize = core.clear_cache = lambda: None  # run_arms hands the cache back between arms
    pkg = types.ModuleType("mlx")
    pkg.__path__, pkg.core = [], core
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    class _Tok:
        def encode(self, text):
            return [1, 2, 3]

    class _Eng:
        resident_bytes = 1000
        meta = {"profile": "sip"}
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[]))

    def tokenizer(path, revision):
        sources["tokenizer"] = two.snapshot_dir(path, revision)
        return _Tok()

    def stock(model_id, revision, ctx=None, snap=None):
        sources["stock"] = snap or checkpoint.snapshot_dir(model_id, revision)
        return _Eng()

    def compressed(model_id, revision, pack_dir, ctx=None, path=None, rope_scaling=None, snap=None):
        import json

        meta = json.load(open(os.path.join(pack_dir, "meta.json")))
        sources.setdefault("compressed", set()).add(
            snap or checkpoint.resolve_pack_source(model_id, revision, meta, pack_dir))
        sources.setdefault("paths", []).append(path)
        return _Eng()

    monkeypatch.setattr(checkpoint, "tokenizer", tokenizer)
    monkeypatch.setattr(arms_mlx, "measure_bandwidth", lambda: {
        "probe_bytes": GB, "read_bytes_s": 200 * GB, "copy_bytes_s": 180 * GB, "device": "metal"})
    monkeypatch.setattr(arms_mlx, "timed_decode", lambda eng, ids, n, reps: (None, [10.0, 10.0, 10.0]))
    monkeypatch.setattr(arms_mlx, "timed_prefill", lambda eng, ids, reps: [500.0, 500.0, 500.0])
    monkeypatch.setattr(arms_mlx, "timed_ttft", lambda eng, ids, reps: [0.05, 0.05, 0.05])
    monkeypatch.setattr(arms_mlx, "refuse_unless_verified", lambda model_id, swapped: None)
    monkeypatch.setattr(arms_mlx, "_packed_modules", lambda model: iter([1]))
    monkeypatch.setattr(arms_mlx, "_device", lambda: "test-device")
    # run_arms imports the two loaders from serving.engine_mlx at call time,
    # and that module needs mlx: a stand-in module carrying the stubs
    monkeypatch.setitem(sys.modules, "drinkme.serving.engine_mlx", types.SimpleNamespace(
        load_stock_mlx=stock, load_compressed_mlx=compressed, TWIN="twin"))
    r = arms_mlx.run_arms(REPO, None, "hi", pack_dir=two.pack_dir, fused=False, prefill_len=4)
    paths = sources.pop("paths")
    assert sources == {"tokenizer": two.snap_a, "stock": two.snap_a, "compressed": {two.snap_a}}, sources
    assert paths == ["twin", "reference"]  # the twin pass, then compressed on a machine without Metal
    assert r["resolved_revision"] == SHA_A and r["revision"] is None
    assert r["warmup_rep"] is arms_mlx.WARMUP_REP  # raw.warmup_rep, exactly as arms.run_arms writes it
