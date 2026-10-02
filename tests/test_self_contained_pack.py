"""Self-contained packs: a pack carries its checkpoint's small files and the
tensors the loaders read raw, and loads with no snapshot and no network
(docs/pack-format.md#the-embedded-checkpoint).

Every checkpoint is the toy Llama of tests/test_pack_identity.py
(bench/radix_mlx_toy_build.build_identity_checkpoint), as a local
directory or in a fabricated hub-cache layout. Offline is enforced, not
assumed: HF_HUB_OFFLINE=1, an empty HF_HOME, snapshot_dir replaced by a
function that fails the test, and the source checkpoint deleted before the
self-contained load. No GPU, no download.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import types

import pytest

from drinkme.codec import identity
from drinkme.serving import checkpoint

torch = pytest.importorskip("torch")

from drinkme.codec.pack import (  # noqa: E402
    EMBED_SUBDIR, REMAINDER_FILE, _shard_layout, embedded_dir, pack_model, verify_pack)
from tests.test_pack_identity import AAA, _toy_checkpoint, fake_hub, two_models  # noqa: E402,F401

MSGS = [{"role": "user", "content": "hello world the quick brown fox"}]


def _meta(pack_dir: str) -> dict:
    with open(os.path.join(pack_dir, "meta.json")) as f:
        return json.load(f)


def _quiet(*_a, **_k) -> None:
    pass


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A toy checkpoint as a local directory, packed self-contained."""
    root = tmp_path_factory.mktemp("sc")
    model = str(root / "model")
    _toy_checkpoint(model, seed=7)
    embedded = str(root / "pack-embedded")
    pack_model(model, None, embedded, progress=_quiet)
    return model, embedded


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No snapshot, no network: any snapshot resolution fails the test."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-home"))

    def no_snapshot(*a, **k):
        raise AssertionError(f"a self-contained load resolved a snapshot: {a}")

    from drinkme import arms

    monkeypatch.setattr(checkpoint, "snapshot_dir", no_snapshot)
    monkeypatch.setattr(arms, "snapshot_dir", no_snapshot)
    return no_snapshot


def _greedy(eng, n: int = 16) -> str:
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    res = complete(eng, GenerationRequest(MSGS, SampleParams(temperature=0.0, max_tokens=n)),
                   lambda _d: True)
    return res.text


def _load(repo, pack_dir):
    from drinkme.serving.engines import load_compressed

    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        return load_compressed(repo, None, pack_dir, device="cpu")
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


def _rewrite_hash(pack_dir: str, fn: str) -> None:
    """Re-record one embedded file's sha256 and the manifest digest, as a
    tamperer who edits meta.json too would: the hash gate then passes, and
    only the identity check stands between the edit and a load."""
    meta = _meta(pack_dir)
    with open(os.path.join(embedded_dir(pack_dir), fn), "rb") as f:
        meta["embedded"]["files"][fn] = hashlib.sha256(f.read()).hexdigest()
    meta["manifestSha256"] = identity.manifest_digest(meta)
    with open(os.path.join(pack_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


# ----------------------------------------------------------------- embed --


def test_pack_embeds_the_small_files_and_exactly_the_raw_tensors(source):
    model, embedded = source
    meta = _meta(embedded)
    d = embedded_dir(embedded)
    names = set(os.listdir(d))
    for f in ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", REMAINDER_FILE, identity.EMBEDDED_IDENTITY_FILE):
        assert f in names, f
    assert not any(n.endswith(".index.json") for n in names)
    # every embedded file is recorded, and nothing else
    assert set(meta["embedded"]["files"]) == names

    # the remainder is every checkpoint tensor the pack does not carry,
    # byte for byte, at its own dtype and shape
    src_path = os.path.join(model, "model.safetensors")
    src_base, src_hdr = _shard_layout(src_path)
    base, emb_hdr = _shard_layout(os.path.join(d, REMAINDER_FILE))
    packed = {f"{n}.weight" for n in meta["tensors"]}
    want = {k for k in src_hdr if k != "__metadata__"} - packed
    assert {k for k in emb_hdr if k != "__metadata__"} == want
    assert meta["embedded"]["tensorCount"] == len(want)
    src_blob = open(src_path, "rb").read()
    emb_blob = open(os.path.join(d, REMAINDER_FILE), "rb").read()
    for k in want:
        (a0, a1), (b0, b1) = src_hdr[k]["data_offsets"], emb_hdr[k]["data_offsets"]
        assert src_blob[src_base + a0:src_base + a1] == emb_blob[base + b0:base + b1], k
        assert (src_hdr[k]["dtype"], src_hdr[k]["shape"]) == (emb_hdr[k]["dtype"], emb_hdr[k]["shape"])
    from safetensors import safe_open

    with safe_open(os.path.join(d, REMAINDER_FILE), framework="pt") as f:
        assert set(f.keys()) == want  # a file the stock reader opens
    # the identity recomputes from the pack alone
    assert identity.embedded_digest(d) == meta["source"]["digest"] == identity.content_digest(model)
    # the manifest binds the embedded block
    assert identity.pack_manifest(meta)["embedded"] == meta["embedded"]


# ---------------------------------------------------------------- verify --


def test_verify_covers_embedded_files(source, tmp_path):
    _, embedded = source
    pack = str(tmp_path / "p")
    shutil.copytree(embedded, pack)
    lines = []
    verify_pack(pack, progress=lines.append)
    assert "embedded checkpoint:" in lines[0] and "identity matches" in lines[0]
    # one byte flipped in an embedded file: the hash gate
    p = os.path.join(embedded_dir(pack), REMAINDER_FILE)
    blob = bytearray(open(p, "rb").read())
    blob[-1] ^= 0x01
    open(p, "wb").write(bytes(blob))
    with pytest.raises(ValueError, match=f"hash mismatch: {EMBED_SUBDIR}/{REMAINDER_FILE}"):
        verify_pack(pack, progress=_quiet)


def test_verify_refuses_an_unrecorded_edit_to_the_embedded_map(source, tmp_path):
    _, embedded = source
    pack = str(tmp_path / "p")
    shutil.copytree(embedded, pack)
    meta = _meta(pack)
    del meta["embedded"]["files"]["generation_config.json"]
    with open(os.path.join(pack, "meta.json"), "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="manifest broken"):
        verify_pack(pack, progress=_quiet)


def test_embedded_file_names_cannot_leave_the_directory(source, tmp_path):
    _, embedded = source
    pack = str(tmp_path / "p")
    shutil.copytree(embedded, pack)
    meta = _meta(pack)
    meta["embedded"]["files"]["../meta.json"] = "0" * 64
    meta["manifestSha256"] = identity.manifest_digest(meta)
    with open(os.path.join(pack, "meta.json"), "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="not a bare file name"):
        verify_pack(pack, progress=_quiet)


# -------------------------------------------------------------- identity --


def test_an_embedded_tokenizer_that_is_not_the_sources_is_refused(source, tmp_path, offline):
    """A consistent tamper (file edited, its hash and the manifest
    re-recorded): the hashes pass, the recomputed identity does not."""
    _, embedded = source
    pack = str(tmp_path / "p")
    shutil.copytree(embedded, pack)
    with open(os.path.join(embedded_dir(pack), "tokenizer_config.json"), "a") as f:
        f.write("\n")
    _rewrite_hash(pack, "tokenizer_config.json")
    from drinkme.codec.pack import verify_hashes

    assert verify_hashes(pack)
    with pytest.raises(checkpoint.PackSourceMismatch, match="embedded checkpoint mismatch"):
        checkpoint.resolve_pack_source(_meta(pack)["hfRepo"], None, _meta(pack), pack)
    with pytest.raises(ValueError, match="embedded checkpoint mismatch"):
        verify_pack(pack, progress=_quiet)


def test_an_embedded_image_processor_config_is_part_of_the_identity(tmp_path):
    """A vision checkpoint's processor configs are embedded and bound by
    the source digest (codec/identity.IDENTITY_FILES): the digest
    recomputes from the pack alone, and a consistent tamper of the
    embedded preprocessor_config.json is refused the way a tokenizer's is."""
    model = str(tmp_path / "model")
    _toy_checkpoint(model, seed=13)
    for name, body in (("preprocessor_config.json", {"patch_size": 16, "merge_size": 2}),
                       ("processor_config.json", {"image_processor": {"patch_size": 16}})):
        with open(os.path.join(model, name), "w") as f:
            json.dump(body, f)
    pack = str(tmp_path / "pack")
    pack_model(model, None, pack, progress=_quiet)
    meta = _meta(pack)
    assert {"preprocessor_config.json", "processor_config.json"} <= set(meta["embedded"]["files"])
    assert identity.embedded_digest(embedded_dir(pack)) == meta["source"]["digest"] == \
        identity.content_digest(model)
    with open(os.path.join(embedded_dir(pack), "preprocessor_config.json"), "w") as f:
        json.dump({"patch_size": 14, "merge_size": 2}, f)
    _rewrite_hash(pack, "preprocessor_config.json")
    with pytest.raises(checkpoint.PackSourceMismatch, match="embedded checkpoint mismatch"):
        checkpoint.resolve_pack_source(_meta(pack)["hfRepo"], None, _meta(pack), pack)


def test_a_named_local_checkpoint_that_differs_is_still_refused(source, tmp_path):
    """--model naming an existing directory must be the pack's model even
    though a self-contained load does not read it."""
    model, embedded = source
    other = str(tmp_path / "other")
    shutil.copytree(model, other)
    cfg = json.load(open(os.path.join(other, "config.json")))
    cfg["rope_theta"] = 12345.0
    json.dump(cfg, open(os.path.join(other, "config.json"), "w"))
    with pytest.raises(checkpoint.PackSourceMismatch, match="config, tokenizer files or image processor configs"):
        checkpoint.resolve_pack_source(other, None, _meta(embedded), embedded)
    assert checkpoint.resolve_pack_source(model, None, _meta(embedded), embedded) == \
        embedded_dir(embedded)


# ----------------------------------------------------------- offline load --


def test_self_contained_pack_loads_offline_with_the_checkpoint_deleted(tmp_path, monkeypatch):
    """The acceptance shape on CPU: greedy output and the served embedding
    from a self-contained pack are the same whether its source checkpoint is
    present or gone and the network is off — a self-contained pack reads
    only its own `checkpoint/`, never the source, either way."""
    model = str(tmp_path / "model")
    _toy_checkpoint(model, seed=11)
    pack = str(tmp_path / "p")
    pack_model(model, None, pack, progress=_quiet)
    online = _load(model, pack)
    want_text = _greedy(online)
    want_embed = online.model.get_input_embeddings().weight.detach().clone()
    del online

    shutil.rmtree(model)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-home"))
    from drinkme import arms

    def no_snapshot(*a, **k):
        raise AssertionError("snapshot resolved")

    monkeypatch.setattr(checkpoint, "snapshot_dir", no_snapshot)
    monkeypatch.setattr(arms, "snapshot_dir", no_snapshot)
    offline = _load(model, pack)
    assert torch.equal(offline.model.get_input_embeddings().weight, want_embed)
    assert _greedy(offline) == want_text
    assert offline.meta["source"] == _meta(pack)["source"]


def test_a_pack_without_an_embedded_checkpoint_is_refused_at_load(tmp_path):
    """A pack without an embedded checkpoint (embed=False — no CLI flag
    writes this shape; pack_model's parameter exists only to build this one
    fixture) is refused at load, by name, rather than silently streaming its
    raw half from a checkpoint. The one place this repo builds a
    non-embedded pack."""
    model = str(tmp_path / "model")
    _toy_checkpoint(model, seed=13)
    pack = str(tmp_path / "p")
    pack_model(model, None, pack, progress=_quiet, embed=False)
    meta = _meta(pack)
    assert "embedded" not in meta
    with pytest.raises(checkpoint.PackSourceMismatch,
                       match="carries no embedded checkpoint"):
        checkpoint.resolve_pack_source(model, None, meta, pack)
    with pytest.raises(checkpoint.PackSourceMismatch,
                       match="carries no embedded checkpoint"):
        _load(model, pack)


def test_a_hub_pack_loads_offline_through_serves_front_door(two_models, fake_hub, tmp_path,
                                                             monkeypatch, capsys):
    """A hub-kind self-contained pack: serve's torch-free door and the
    loader both resolve the pack's own directory, never the snapshot; the
    boot line says where the raw tensors came from, and the engine reports
    the pack's commit."""
    from drinkme import arms, serve

    pack = str(tmp_path / "pack-A")
    pack_model("source/A", AAA, pack, progress=_quiet)  # while the (fake) hub answers
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-home"))

    def no_snapshot(*a, **k):
        raise AssertionError(f"snapshot resolved: {a}")

    monkeypatch.setattr(checkpoint, "snapshot_dir", no_snapshot)
    monkeypatch.setattr(arms, "snapshot_dir", no_snapshot)
    assert serve.ensure_pack("source/A", AAA, pack, auto_pack=False) == pack
    assert serve.ensure_pack("source/A", None, pack, auto_pack=False) == pack
    with pytest.raises(SystemExit, match="other/B"):
        serve.ensure_pack("other/B", None, pack, auto_pack=False)
    with pytest.raises(SystemExit, match="bbbb"):
        serve.ensure_pack("source/A", "b" * 40, pack, auto_pack=False)
    with pytest.raises(SystemExit, match="without the network"):
        serve.ensure_pack("source/A", "main", pack, auto_pack=False)
    capsys.readouterr()
    eng = _load("source/A", pack)
    out = capsys.readouterr().out
    assert "raw tensors from the pack's embedded checkpoint" in out
    assert eng.meta["resolvedRevision"] == AAA
    assert "streamed from" not in out


def test_the_bench_still_reads_the_real_snapshot(two_models, fake_hub, tmp_path):
    """bench's stock and twin arms need the packed Linears' source weights:
    resolve_source never hands them the embedded checkpoint."""
    _, a, _ = two_models
    pack = str(tmp_path / "pack-A")
    pack_model("source/A", AAA, pack, progress=_quiet)
    snap, rev = checkpoint.resolve_source("source/A", None, pack)
    assert snap == a and rev == AAA


# ------------------------------------------------------------------ MLX --


@pytest.fixture
def fake_mlx(monkeypatch):
    """engine_mlx imported over stand-in mlx / mlx_lm modules (no Mac here;
    the real path is tests/test_serving_engine_mlx.py's, on the Mac): the
    module's own resolution, estimate, tokenizer and generation-defaults
    code runs for real; the mlx arrays and modules are fakes."""
    class _Meta(type):
        def __getattr__(cls, n):
            if n == "__version__":
                return "fake"
            if n.startswith("__"):
                raise AttributeError(n)
            return _Dummy

    class _Dummy(metaclass=_Meta):
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return _Dummy()

        def __getattr__(self, n):
            return _Dummy()

    def fake(name):
        m = types.ModuleType(name)
        m.__getattr__ = lambda n: _Dummy
        m.__path__ = []
        return m

    for n in ("mlx", "mlx.core", "mlx.nn", "mlx.utils", "mlx_lm", "mlx_lm.models",
              "mlx_lm.models.qwen3", "mlx_lm.models.cache", "mlx_lm.models.base",
              "mlx_lm.utils", "mlx_lm.sample_utils"):
        monkeypatch.setitem(sys.modules, n, fake(n))
    for n in [k for k in sys.modules if k.startswith(("drinkme.serving.engine_mlx", "drinkme.metal"))]:
        monkeypatch.delitem(sys.modules, n)
    import importlib

    e = importlib.import_module("drinkme.serving.engine_mlx")
    yield e
    for n in [k for k in sys.modules if k.startswith(("drinkme.serving.engine_mlx", "drinkme.metal"))]:
        sys.modules.pop(n, None)


def test_mlx_loader_reads_everything_from_the_embedded_checkpoint(source, tmp_path, offline,
                                                                  fake_mlx):
    """load_compressed_mlx on a self-contained pack, offline: the snapshot
    it streams from, estimates from, and reads the tokenizer, config and
    generation defaults from is the pack's checkpoint/ directory."""
    e = fake_mlx
    model, embedded = source
    pack = str(tmp_path / "p")
    shutil.copytree(embedded, pack)
    seen = {}

    class Lin:
        def resident_bytes(self):
            return 0

    def stream(model_, snap, packed, tie, name=None):
        from drinkme.codec.pack import _shard_headers

        seen["snap"], seen["packed"] = snap, set(packed)
        seen["tensors"] = set(_shard_headers(snap))
        return 0, set()

    class Engine:
        def __init__(self, model_, tok, **kw):
            seen["tok"], seen["kw"] = tok, kw

    e.refuse_unsupported = lambda cfg: None
    e.fit_check = lambda *a, **k: None
    e.build_skeleton = lambda cfg: (object(), types.SimpleNamespace(tie_word_embeddings=False))
    e.parameter_paths = lambda m: set()
    e._module_at = lambda m, n: e.nn.Linear()
    e._set_at = lambda m, n, v: None
    e.make_module_mlx = lambda p, b, path=None, name=None: Lin()
    e._stream_checkpoint = stream
    e._check_materialized = lambda *a: None
    e.device_name = lambda: "fake"
    e.MLXEngine = Engine
    e.load_compressed_mlx(model, None, pack, path="reference")
    meta = _meta(pack)
    assert seen["snap"] == embedded_dir(pack)
    assert seen["packed"] == set(meta["tensors"])
    _, remainder = _shard_layout(os.path.join(embedded_dir(pack), REMAINDER_FILE))
    assert seen["tensors"] == {k for k in remainder if k != "__metadata__"}
    assert not any(f"{n}.weight" in seen["tensors"] for n in meta["tensors"])
    assert seen["tok"].encode("hello world") and seen["kw"]["meta"]["source"] == meta["source"]
    assert seen["kw"]["gen_defaults"] is not None


def test_a_tied_output_embedding_is_not_embedded(tmp_path):
    """A tied lm_head the checkpoint ships anyway (Qwen3-0.6B's 311 MB copy)
    is re-tied by the loaders, so it is not carried; the tied load still
    serves the source's embedding under both names."""
    from transformers import LlamaConfig

    from drinkme.arms import ckpt_to_skel, skeleton
    from drinkme.codec.pack import embedded_tensor_names, tied_output_weight

    cfg = LlamaConfig(vocab_size=64, hidden_size=1024, intermediate_size=1024, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    assert tied_output_weight(skeleton(cfg)) == "lm_head.weight"
    cfg.tie_word_embeddings = False
    assert tied_output_weight(skeleton(cfg)) is None

    model = str(tmp_path / "m")
    _toy_checkpoint(model, seed=3)
    c = json.load(open(os.path.join(model, "config.json")))
    c["tie_word_embeddings"] = True  # the toy ships lm_head.weight: the redundant copy
    json.dump(c, open(os.path.join(model, "config.json"), "w"))
    from transformers import AutoConfig

    tcfg = AutoConfig.from_pretrained(model)
    names = embedded_tensor_names(model, ckpt_to_skel(tcfg), set(), set(),
                                  tied_output_weight(skeleton(tcfg)))
    assert "lm_head.weight" not in names and "model.embed_tokens.weight" in names
    pack = str(tmp_path / "p")
    pack_model(model, None, pack, progress=_quiet)
    eng = _load(model, pack)
    emb = eng.model.get_input_embeddings().weight
    assert eng.model.get_output_embeddings().weight is emb
    from safetensors import safe_open

    with safe_open(os.path.join(model, "model.safetensors"), framework="pt") as f:
        assert torch.equal(emb.cpu(), f.get_tensor("model.embed_tokens.weight"))
