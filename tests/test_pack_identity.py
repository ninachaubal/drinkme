"""a pack is bound to ONE immutable checkpoint snapshot.

The failure it guards: a compressed loader that takes the pack's Linears
from `--pack-dir` and everything else (embeddings, norms, biases, config,
tokenizer) from whatever `--model`/revision resolves to on the serving box.
Compatible shapes load, and the mixed engine serves with a 200. The probe:
a verified pack identifying source/A@aaa, served against other/B@bbb.

Every checkpoint here is a toy Llama in a fabricated HF-cache layout
(`<cache>/models--org--name/snapshots/<sha>/`), so the identity the pack
records is the hub kind — a commit sha — exactly as a real pack's is; the
resolver is faked to map (repo, revision) onto those dirs. No network, no
real cache, no GPU.

No `import torch` at module scope: the torch-backed
tests below import it locally, or take it through
bench/radix_mlx_toy_build.py's build_identity_checkpoint. The Metal-lane
test at the bottom needs neither — it loads a prebuilt pack from
DRINKME_RADIX_MLX_TOY, or builds one itself where torch exists, so the file
COLLECTS and that one test RUNS on a torch-less box.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

from drinkme.codec import identity
from drinkme.serving import checkpoint

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
sys.path.insert(0, BENCH_DIR)
from radix_mlx_toy_build import (  # noqa: E402
    IDENTITY_AAA, IDENTITY_BBB, build_identity_checkpoint, build_identity_pack, identity_paths)

AAA = IDENTITY_AAA
BBB = IDENTITY_BBB
TOY_ENV = "DRINKME_RADIX_MLX_TOY"


def _snapshot(cache: str, repo: str, sha: str) -> str:
    return os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots", sha)


def _toy_checkpoint(model_dir: str, seed: int) -> None:
    """build_identity_checkpoint (bench/radix_mlx_toy_build.py) under its
    original name here: test_bits_per_weight.py, test_bench_arm_wiring.py,
    test_bench_one_source.py, test_pack_verify_manifest.py and
    test_pack_writer_transactional.py all import this by name for their own
    toy checkpoints, so it stays — as a one-line alias, not a second copy
    (a builder flag factored the toy itself out)."""
    build_identity_checkpoint(model_dir, seed)


@pytest.fixture(scope="module")
def two_models(tmp_path_factory):
    """source/A@AAA and other/B@BBB: same config, same tokenizer, same
    shapes, different weights: compatible shapes. Built
    through build_identity_checkpoint (bench/radix_mlx_toy_build.py), the
    same function --identity uses, so the toy is never duplicated."""
    pytest.importorskip("torch")
    cache = str(tmp_path_factory.mktemp("hubcache"))
    a, b = _snapshot(cache, "source/A", AAA), _snapshot(cache, "other/B", BBB)
    _toy_checkpoint(a, seed=1)
    _toy_checkpoint(b, seed=2)
    return cache, a, b


@pytest.fixture
def fake_hub(two_models, monkeypatch):
    """(repo, revision) -> the fabricated snapshot dir, in every place the
    packer and the loaders resolve one; a dir resolves to itself."""
    cache, a, b = two_models
    table = {("source/A", AAA): a, ("source/A", None): a, ("source/A", "main"): a,
             ("other/B", BBB): b, ("other/B", None): b, ("other/B", "main"): b}

    def resolve(repo, revision=None):
        if os.path.isdir(repo):
            return repo
        return table[(repo, revision)]

    from drinkme import arms

    monkeypatch.setattr(checkpoint, "snapshot_dir", resolve)
    monkeypatch.setattr(arms, "snapshot_dir", resolve)
    # Stubs for a loader that resolves config and tokenizer from the
    # caller's repo id on its own: a fake hub id reads its fabricated dir.
    # The loader under test never calls these with a hub id.
    from transformers import AutoConfig

    from drinkme.serving import engines

    orig_cfg = AutoConfig.from_pretrained.__func__

    def cfg_from(cls, name, *a, **kw):
        if isinstance(name, str) and (name, kw.get("revision")) in table:
            return orig_cfg(cls, table[(name, kw.get("revision"))])
        return orig_cfg(cls, name, *a, **kw)

    monkeypatch.setattr(AutoConfig, "from_pretrained", classmethod(cfg_from))
    orig_tok = checkpoint.tokenizer
    tok_from = lambda repo, revision=None: orig_tok(resolve(repo, revision), None)  # noqa: E731
    monkeypatch.setattr(engines, "_tokenizer", tok_from)
    monkeypatch.setattr(checkpoint, "tokenizer", tok_from)
    return resolve


@pytest.fixture
def pack_a(two_models, fake_hub, tmp_path):
    from drinkme.codec.pack import pack_model

    pack_dir = str(tmp_path / "pack-A")
    pack_model("source/A", AAA, pack_dir, progress=lambda *_: None)
    return pack_dir


@pytest.fixture
def identity_pack_a(tmp_path_factory):
    """pack_a's mlx-lane twin: the prebuilt identity fixture's pack from
    DRINKME_RADIX_MLX_TOY (bench/radix_mlx_toy_build.py --identity), else
    built here through build_identity_pack when torch is importable — the
    same helper pack_a builds through above, so a torch box's coverage
    never depends on which path ran — else a by-name skip, the
    DRINKME_RADIX_MLX_TOY pattern test_serving_engine_mlx.py already uses."""
    root = os.environ.get(TOY_ENV)
    if root:
        found = identity_paths(root)
        if found:
            return found["pack_a"]
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("no torch to build the identity toy and DRINKME_RADIX_MLX_TOY names "
                    "none (bench/radix_mlx_toy_build.py --identity)")
    return build_identity_pack(str(tmp_path_factory.mktemp("identity_142")))["pack_a"]


def _embedding(snap: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as sf:
        return sf.get_tensor("model.embed_tokens.weight")


# ------------------------------------------------------------- the ruler --


def test_pack_from_A_served_with_B_is_refused_not_mixed(two_models, fake_hub, pack_a):
    """The ruler: pack says source/A@AAA, the checkpoint named is
    other/B@BBB, shapes compatible. Refused: a PackSourceMismatch naming
    both, before any weight is allocated. A loader without the check builds
    a mixed engine — A's packed Linears under B's embeddings — that loads
    and serves, and this test fails on the embedding check below."""
    import torch

    from drinkme.serving.engines import load_compressed

    _, a, b = two_models
    refusal = getattr(checkpoint, "PackSourceMismatch", ())
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        try:
            eng = load_compressed("other/B", BBB, pack_a, device="cpu")
        except refusal as e:
            msg = str(e)
            assert "source/A" in msg and "other/B" in msg, msg
            meta = json.load(open(os.path.join(pack_a, "meta.json")))
            assert meta["source"]["repo"] == "source/A" and meta["source"]["revision"] == AAA
            return
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]
    served = eng.model.get_input_embeddings().weight.detach().cpu()
    from_b = torch.equal(served, _embedding(b)) and not torch.equal(served, _embedding(a))
    pytest.fail("MIXED ENGINE: pack cut from source/A loaded against other/B — "
                f"served embedding is B's: {from_b}; meta still says "
                f"{eng.model_meta()['hfRepo']}")


def test_pack_from_A_served_with_B_is_refused_on_the_mlx_lane(identity_pack_a):
    """The same refusal on the mlx runtime's loader, from the same check —
    torch-free: the refusal is check_pack_repo's meta.json string compare,
    before any snapshot is resolved (confirmed on a Strix Halo with
    torch AND mlx both unimportable)."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from drinkme.serving.engine_mlx import load_compressed_mlx

    with pytest.raises(checkpoint.PackSourceMismatch, match="source/A"):
        load_compressed_mlx("other/B", BBB, identity_pack_a, path="reference")


# --------------------------------------------------- the bound snapshot --


def test_pack_records_the_resolved_commit_and_the_loader_streams_from_it(
        two_models, fake_hub, pack_a):
    """revision=None at serve time means 'whatever the pack is bound to' —
    never a re-resolve of main. Config, tokenizer and raw tensors come
    from that snapshot; the engine's meta carries the binding."""
    import torch

    from drinkme.serving.engines import load_compressed

    _, a, _ = two_models
    meta = json.load(open(os.path.join(pack_a, "meta.json")))
    src = meta["source"]
    assert src["kind"] == "hub" and src["revision"] == AAA
    assert src["digest"] == identity.content_digest(a)
    assert src["identityVersion"] == identity.IDENTITY_VERSION
    assert meta["revision"] == AAA  # the caller's coordinate, unchanged
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        eng = load_compressed("source/A", None, pack_a, device="cpu")
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]
    assert torch.equal(eng.model.get_input_embeddings().weight.detach().cpu(), _embedding(a))
    assert eng.meta["source"] == src
    assert eng.model_meta()["hfRepo"] == "source/A"


def test_a_different_explicit_commit_is_refused_by_name(two_models, fake_hub, pack_a):
    meta = json.load(open(os.path.join(pack_a, "meta.json")))
    with pytest.raises(checkpoint.PackSourceMismatch, match=BBB):
        checkpoint.resolve_pack_source("source/A", BBB, meta, pack_a)


def test_a_branch_resolving_elsewhere_is_refused(two_models, fake_hub, pack_a, monkeypatch):
    """`main` moved on the hub since the pack was cut: an explicit branch
    that resolves to another commit is refused; None still means the pack's.
    This is bench's path (embedded=False): a normal load never reaches a
    snapshot at all, self-contained packs resolve to their own checkpoint/."""
    _, a, b = two_models
    meta = json.load(open(os.path.join(pack_a, "meta.json")))
    moved = {("source/A", AAA): a, ("source/A", "main"): b, ("source/A", None): b}
    monkeypatch.setattr(checkpoint, "snapshot_dir",
                        lambda repo, rev=None: repo if os.path.isdir(repo) else moved[(repo, rev)])
    with pytest.raises(checkpoint.PackSourceMismatch, match="resolves to commit"):
        checkpoint.resolve_pack_source("source/A", "main", meta, pack_a, embedded=False)
    assert checkpoint.resolve_pack_source("source/A", None, meta, pack_a, embedded=False) == a


def test_a_pack_cut_from_another_hub_repo_is_refused_on_the_name_alone(pack_a):
    meta = json.load(open(os.path.join(pack_a, "meta.json")))
    with pytest.raises(checkpoint.PackSourceMismatch, match="other/B"):
        checkpoint.check_pack_repo("other/B", meta, pack_a)
    unbound = {"hfRepo": "source/A", "revision": None}  # no `source` block at all
    with pytest.raises(checkpoint.PackSourceMismatch, match="source/A"):
        checkpoint.check_pack_repo("other/B", unbound, pack_a)


def test_a_local_directory_is_bound_by_content(two_models, tmp_path, capsys):
    """Local checkpoints: identity is the digest. A copy of the same files is
    the same model; the same shapes with a different config.json is not.
    The direct resolve_pack_source calls pass embedded=False (bench's path,
    resolve_source): a self-contained pack's normal load never touches a
    snapshot, so this is where local-directory identity is actually
    exercised; the final load_compressed call is the normal (embedded)
    path, and is refused the same way."""
    from drinkme.codec.pack import pack_model
    from drinkme.serving.engines import load_compressed

    _, a, _ = two_models
    local = str(tmp_path / "local-A")
    shutil.copytree(a, local)
    pack_dir = str(tmp_path / "pack-local")
    pack_model(local, None, pack_dir, progress=lambda *_: None)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    assert meta["source"]["kind"] == "local" and meta["source"]["revision"] is None
    assert meta["source"]["digest"] == identity.content_digest(local) == identity.content_digest(a)

    copy = str(tmp_path / "copy-of-A")
    shutil.copytree(local, copy)
    assert checkpoint.resolve_pack_source(copy, None, meta, pack_dir, embedded=False) == copy

    edited = str(tmp_path / "edited-A")
    shutil.copytree(local, edited)
    cfg = json.load(open(os.path.join(edited, "config.json")))
    cfg["rope_theta"] = 500000.0
    json.dump(cfg, open(os.path.join(edited, "config.json"), "w"))
    with pytest.raises(checkpoint.PackSourceMismatch, match="config, tokenizer files or image processor configs"):
        checkpoint.resolve_pack_source(edited, None, meta, pack_dir, embedded=False)
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        with pytest.raises(checkpoint.PackSourceMismatch):
            load_compressed(edited, None, pack_dir, device="cpu")
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


def test_a_pack_without_a_source_block_loads_with_one_warning_and_the_callers_coordinates(
        two_models, fake_hub, pack_a, capsys):
    """A tool-written pack (save_pack_dir with no identity; `drinkme pack`
    always records one) resolves the caller's coordinates, unpinned, and
    says so once, naming the pack and the fix (re-pack) — reachable only
    with embedded=False (bench's resolve_source): a pack without `source`
    also has no `embedded` block, and the normal (embedded) load refuses
    that outright rather than falling back to an unpinned snapshot."""
    _, a, _ = two_models
    meta = json.load(open(os.path.join(pack_a, "meta.json")))
    del meta["source"]
    assert checkpoint.resolve_pack_source("source/A", None, meta, pack_a, embedded=False) == a
    out = capsys.readouterr().out
    assert out.count("unbound pack (no source identity recorded)") == 1
    assert pack_a in out and "re-pack" in out and "drinkme verify" not in out


def test_serve_front_door_refuses_the_pair_before_torch(two_models, fake_hub, pack_a, monkeypatch):
    from drinkme import serve

    with pytest.raises(SystemExit, match="source/A"):
        serve.ensure_pack("other/B", BBB, pack_a, auto_pack=False)
    assert serve.ensure_pack("source/A", None, pack_a, auto_pack=False) == pack_a


# ------------------------------------------------------- the identity --


def test_checkpoint_identity_reads_the_hub_layout_and_digests_the_small_files(two_models, tmp_path):
    _, a, b = two_models
    ia, ib = identity.checkpoint_identity(a, "source/A"), identity.checkpoint_identity(b, "other/B")
    assert ia["kind"] == "hub" and ia["revision"] == AAA
    assert ib["kind"] == "hub" and ib["revision"] == BBB
    # same shapes, same config, same tokenizer: the headers agree, so the
    # digests do — the commit sha is what tells A from B on the hub kind
    assert ia["digest"] == ib["digest"]
    assert identity.hub_revision_of(str(tmp_path)) is None
    assert identity.is_commit_sha(AAA) and not identity.is_commit_sha("main")
    assert "source/A@" + AAA in identity.describe(ia)

    # every file class moves the digest: a tokenizer edit, a shard header edit
    t = str(tmp_path / "tok")
    shutil.copytree(a, t)
    with open(os.path.join(t, "tokenizer_config.json"), "a") as f:
        f.write("\n")
    assert identity.content_digest(t) != ia["digest"]
    s = str(tmp_path / "shard")
    shutil.copytree(a, s)
    blob = bytearray(open(os.path.join(s, "model.safetensors"), "rb").read())
    hdr_len = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8:8 + hdr_len])
    header["__metadata__"] = {"format": "pt", "note": "edited"}
    new_hdr = json.dumps(header).encode()
    new_hdr += b" " * ((8 - len(new_hdr) % 8) % 8)
    open(os.path.join(s, "model.safetensors"), "wb").write(
        len(new_hdr).to_bytes(8, "little") + new_hdr + bytes(blob[8 + hdr_len:]))
    assert identity.content_digest(s) != ia["digest"]


def test_the_image_processor_configs_are_identity_files(tmp_path):
    """preprocessor_config.json and processor_config.json define a vision
    checkpoint's image input (serving/vision.read_processor_config) the way
    config.json defines its skeleton: gaining one, or editing one, moves the
    digest. Torch-free: content_digest reads a safetensors header and small
    files, so a hand-written header is a whole checkpoint to it."""
    assert {"preprocessor_config.json", "processor_config.json"} <= set(identity.IDENTITY_FILES)
    ck = tmp_path / "ck"
    ck.mkdir()
    header = json.dumps({"w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    (ck / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(8))
    (ck / "config.json").write_text('{"model_type": "qwen3_5"}')
    seen = {identity.content_digest(str(ck))}
    for name, body in (("preprocessor_config.json", {"patch_size": 16}),
                       ("preprocessor_config.json", {"patch_size": 14}),
                       ("processor_config.json", {"image_processor": {"patch_size": 16}}),
                       ("processor_config.json", {"image_processor": {"patch_size": 16,
                                                                      "merge_size": 2}})):
        (ck / name).write_text(json.dumps(body))
        seen.add(identity.content_digest(str(ck)))
    assert len(seen) == 5
    assert identity.IDENTITY_VERSION == 1  # joined before any pack shipped: no bump
