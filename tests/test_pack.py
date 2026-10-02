"""The pack container (pack format 1) and the checkpoint skeleton helpers:
disk round trip, the streaming writer, the file hashes, eligibility on a meta
skeleton, ckpt_to_skel and the non-text tower pruning. The codec's own
round trip — THE product invariant, decode(encode(x)) == x bitwise — is
tests/test_radix.py and tests/test_radix_pack.py. All CPU/numpy."""

import json
import os

import numpy as np
import pytest

from drinkme.codec.pack import FORMAT_VERSION, load_pack_dir, save_pack_dir

from drinkme.codec import radix_pack as rp

from fixtures import radix_dict


def test_verification_is_not_strippable_by_python_O():
    """`assert` is removed under `python -O`; the invariant must not be.

    Runs a child process under `-O` so a bare `assert` in the radix
    encoder's round-trip check would be silently stripped and this test
    would go blind to it. Three things matter for the check to be real:
      - cwd is THIS tree's repo root (derived from __file__), not a
        hardcoded path — a stranger's checkout imports their own src/.
      - verify_pack_radix is called with the TRUE source tensor U (kept
        from before corruption), not the pack's own decode — passing a
        pack's decode back into itself is a tautology, always "matches".
      - the child prints exactly one verdict token and the assertion is an
        exact match, not a substring ("CAUGHT" in "NOT_CAUGHT" is True).
    """
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import numpy as np, sys; sys.path.insert(0,'src'); sys.path.insert(0,'tests');"
        "from drinkme.codec import radix_pack as rp; from fixtures import realistic_bf16_bits;"
        "U=realistic_bf16_bits(16,64,3,spread=True);"
        "p=rp.pack_array_radix(U,'sip',encoder='numpy'); p['rx_data']=p['rx_data'].copy(); p['rx_data'][3]^=1;"
        "\ntry:\n rp.verify_pack_radix(p, U); print('NOT_CAUGHT')\n"
        "except ValueError: print('CAUGHT')"
    )
    out = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True,
                         text=True, cwd=repo_root)
    assert out.stdout.strip() == "CAUGHT", out.stdout + out.stderr














def test_disk_roundtrip(tmp_path):
    tensors = {
        "layer0.q": radix_dict(64, 256, seed=5),
        "layer0.k": radix_dict(32, 128, seed=6, spread=True, profile="gulp"),
    }
    meta = {"hfRepo": "Qwen/Qwen3-8B", "revision": "abc123", "dtype": "bf16"}
    save_pack_dir(str(tmp_path / "pack"), tensors, meta)
    loaded, m2 = load_pack_dir(str(tmp_path / "pack"))
    assert m2["hfRepo"] == "Qwen/Qwen3-8B" and m2["formatVersion"] == FORMAT_VERSION == 1
    assert m2["profile"] == "sip" and "method" not in m2  # filled in off the first radix tensor
    for name, orig in tensors.items():
        assert np.array_equal(rp.decode_back(loaded[name]), rp.decode_back(orig))
        assert loaded[name]["profile"] == orig["profile"] and loaded[name]["format_version"] == 1
        assert loaded[name]["bpw"] == pytest.approx(orig["bpw"])


def test_disk_version_enforced(tmp_path):
    import json
    import os

    save_pack_dir(str(tmp_path / "p"), {"t": radix_dict(8, 64, seed=7)}, {})
    mp = os.path.join(tmp_path, "p", "meta.json")
    m = json.load(open(mp))
    m["formatVersion"] = 999
    json.dump(m, open(mp, "w"))
    with pytest.raises(ValueError):
        load_pack_dir(str(tmp_path / "p"))




# ---- the pack diet: streaming writer + meta-skeleton naming ----


def test_pack_writer_streaming_equals_batch(tmp_path):
    """PackWriter in any arrival order must produce byte-for-byte the same
    pack dir save_pack_dir produces — same filename assignment, same meta."""
    import json

    from drinkme.codec.pack import PackWriter

    tensors = {
        "b": radix_dict(8, 64, seed=21),
        "a": radix_dict(8, 64, seed=22, spread=True),
        "c": radix_dict(16, 64, seed=23),
    }
    meta = {"hfRepo": "t/t", "revision": None, "dtype": "bf16", "profile": "sip"}
    save_pack_dir(str(tmp_path / "batch"), tensors, meta)
    w = PackWriter(str(tmp_path / "stream"), list(tensors))
    for name in ("c", "a", "b"):  # arrival order != sorted order
        w.add(name, tensors[name])
    w.finish(meta)
    mb = json.load(open(tmp_path / "batch" / "meta.json"))
    ms = json.load(open(tmp_path / "stream" / "meta.json"))
    assert mb == ms
    tb, _ = load_pack_dir(str(tmp_path / "batch"))
    ts, _ = load_pack_dir(str(tmp_path / "stream"))
    for n in tensors:
        for k in rp._ARRAYS_RADIX:
            assert np.array_equal(tb[n][k], ts[n][k])
        assert tb[n]["widths"] == ts[n]["widths"]


def test_pack_writer_incomplete_withholds_meta(tmp_path):
    """meta.json is the commit point: a pack missing a declared tensor must
    raise AND leave no meta.json — a half-pack does not exist to the loader."""
    import os

    from drinkme.codec.pack import PackWriter

    w = PackWriter(str(tmp_path / "p"), ["a", "b"])
    w.add("a", radix_dict(8, 64, seed=24))
    with pytest.raises(ValueError, match="incomplete"):
        w.finish({})
    assert not os.path.exists(tmp_path / "p" / "meta.json")


def _skeleton_eligible_names(**cfg_kwargs):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from drinkme.arms import skeleton
    from drinkme.codec.swap import eligible_linears

    cfg = transformers.LlamaConfig(
        vocab_size=1024, hidden_size=1024, intermediate_size=1024,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        **cfg_kwargs,
    )
    model = skeleton(cfg)
    assert all(p.is_meta for p in model.parameters())  # zero weight bytes
    return {name for name, _, _, _ in eligible_linears(model)}


def test_eligible_linears_on_meta_skeleton_tied_vs_untied():
    """The data_ptr()==0 trap (probed): on a meta skeleton every
    tensor's data_ptr() is 0, so a storage-pointer tied check calls EVERY
    Linear tied and the pack comes out empty. Identity is the meta-safe check:
    only the genuinely tied head is excluded, and the untied root lm_head is
    emitted DOT-FREE ("lm_head", not the historical ".lm_head")."""
    tied = _skeleton_eligible_names(tie_word_embeddings=True)
    untied = _skeleton_eligible_names(tie_word_embeddings=False)
    assert untied - tied == {"lm_head"}
    # q,k,v,o,gate,up,down of the one layer all survive on the meta skeleton
    assert len(tied) == 7, sorted(tied)




# --- ckpt_to_skel: wrapped-checkpoint key translation (qwen3_5) ---

def test_ckpt_to_skel_qwen3_5():
    from types import SimpleNamespace

    from drinkme.arms import ckpt_to_skel

    f = ckpt_to_skel(SimpleNamespace(model_type="qwen3_5"))
    # text tensors lose the language_model wrap
    assert f("model.language_model.layers.0.linear_attn.in_proj_qkv.weight") == \
        "model.layers.0.linear_attn.in_proj_qkv.weight"
    assert f("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    # root lm_head passes through untouched
    assert f("lm_head.weight") == "lm_head.weight"
    # towers the text skeleton never builds are skipped, not renamed
    assert f("model.visual.blocks.0.attn.qkv.weight") is None
    assert f("mtp.layers.0.mlp.gate_proj.weight") is None


def test_ckpt_to_skel_qwen3_5_text_spelling():
    """The TEXT-config spelling (qwen3_5_text) translates exactly as the
    wrapper's does, through the same tables."""
    from types import SimpleNamespace

    from drinkme.arms import ckpt_to_skel

    f = ckpt_to_skel(SimpleNamespace(model_type="qwen3_5_text"))
    assert f("mtp.fc.weight") is None
    assert f("model.visual.blocks.0.attn.qkv.weight") is None
    assert f("model.language_model.layers.0.mlp.gate_proj.weight") == \
        "model.layers.0.mlp.gate_proj.weight"
    assert f("lm_head.weight") == "lm_head.weight"


def test_ckpt_to_skel_identity_for_unwrapped():
    from types import SimpleNamespace

    from drinkme.arms import ckpt_to_skel

    f = ckpt_to_skel(SimpleNamespace(model_type="qwen3"))
    assert f("model.layers.0.mlp.gate_proj.weight") == "model.layers.0.mlp.gate_proj.weight"
    assert f("lm_head.weight") == "lm_head.weight"
    g = ckpt_to_skel(SimpleNamespace())  # no model_type at all
    assert g("anything.weight") == "anything.weight"


# ------------------------------------------------------- the file hashes ----


def test_pack_writes_hashes(tmp_path):
    from drinkme.codec.pack import verify_hashes
    tensors = {"a": radix_dict(8, 64, seed=1),
               "b": radix_dict(8, 64, seed=2)}
    save_pack_dir(str(tmp_path / "p"), tensors, {"hfRepo": "t/t", "revision": None, "dtype": "bf16"})
    with open(tmp_path / "p" / "meta.json") as f:
        meta = json.load(f)
    assert set(meta["sha256"]) == set(meta["tensors"].values())
    assert all(len(h) == 64 for h in meta["sha256"].values())
    assert verify_hashes(str(tmp_path / "p")) is True


def test_hash_mismatch_on_a_bit_flip(tmp_path):
    from drinkme.codec.pack import verify_hashes
    save_pack_dir(str(tmp_path / "p"),
                  {"a": radix_dict(8, 64, seed=3)},
                  {"hfRepo": "t/t", "revision": None, "dtype": "bf16"})
    fn = os.path.join(tmp_path, "p", "t0000.npz")
    blob = bytearray(open(fn, "rb").read())
    blob[len(blob) // 2] ^= 0x01          # one bit, mid-file
    open(fn, "wb").write(bytes(blob))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hashes(str(tmp_path / "p"))


def test_a_pack_without_its_hashes_is_refused_not_served_on_less(tmp_path):
    """Every pack records its hashes at pack time, so a meta.json without the
    per-file hashes (or without the manifest digest) is not a pack this
    drinkme wrote: verify_hashes refuses it by name, with the re-pack command,
    and `drinkme verify` (verify_pack) says the same rather than hashing it now."""
    from drinkme.codec.pack import verify_hashes, verify_pack
    d = str(tmp_path / "p")
    save_pack_dir(d, {"a": radix_dict(8, 64, seed=4)},
                  {"hfRepo": "t/t", "revision": None, "dtype": "bf16"})
    assert verify_hashes(d) is True
    mp = os.path.join(d, "meta.json")
    original = json.load(open(mp))
    for key, needle in (("sha256", "not verifiable"), ("manifestSha256", "no manifest")):
        meta = dict(original)
        del meta[key]
        json.dump(meta, open(mp, "w"))
        for entry in (lambda: verify_hashes(d), lambda: verify_pack(d, progress=lambda *_: None)):
            with pytest.raises(ValueError) as e:
                entry()
            assert str(e.value).startswith(f"{needle}: {d}/meta.json")
            assert str(e.value).endswith("re-pack: `drinkme pack --model t/t --replace`")
        assert json.load(open(mp)) == meta  # verify never rewrites meta.json
    json.dump(original, open(mp, "w"))
    lines = []
    verify_pack(d, progress=lines.append)
    assert lines == [f"verified, manifest and all: {d} — 1 files verified, "
                     f"manifest {original['manifestSha256'][:16]}…; no embedded checkpoint "
                     "(refused at load: repack it)"]


def test_verify_refuses_missing_file(tmp_path):
    from drinkme.codec.pack import verify_hashes
    tensors = {"a": radix_dict(8, 64, seed=5),
               "b": radix_dict(8, 64, seed=6)}
    save_pack_dir(str(tmp_path / "p"), tensors, {"hfRepo": "t/t", "revision": None, "dtype": "bf16"})
    os.remove(os.path.join(tmp_path, "p", "t0001.npz"))
    with pytest.raises(ValueError, match="missing"):
        verify_hashes(str(tmp_path / "p"))


# --- muse_glimmer: the composite VLM whose text skeleton IS the pruned
#     wrapper (Muse-Glimmer-30B, transformers 5.15.0) ---
#
# Key shapes below are MEASURED from the snapshot's
# model.safetensors.index.json (1436 keys: 626 model.language_model.*, 806
# model.vision_tower.*, 2 model.vision_adapter.*, 1
# model.vision_projection.weight, lm_head.weight), not invented.

def test_ckpt_to_skel_muse_glimmer():
    from types import SimpleNamespace

    from drinkme.arms import ckpt_to_skel

    f = ckpt_to_skel(SimpleNamespace(model_type="muse_glimmer"))
    # text tensors keep their names — unlike qwen3_5, this skeleton IS the
    # wrapper, so it already calls them what the checkpoint calls them
    for k in ("model.language_model.layers.0.self_attn.gate_proj.weight",
              "model.language_model.layers.51.mlp.down_proj.weight",
              "model.language_model.layers.7.post_feedforward_layernorm.weight",
              "model.language_model.embed_tokens.weight",
              "model.language_model.norm.weight",
              "lm_head.weight"):
        assert f(k) == k
    # the three vision towers are skipped BY NAME, not renamed
    assert f("model.vision_tower.layers.0.attn.q_proj.weight") is None
    assert f("model.vision_tower.patch_embedder.patch_embedding.weight") is None
    assert f("model.vision_adapter.fc1.weight") is None
    assert f("model.vision_projection.weight") is None


def test_ckpt_to_skel_muse_glimmer_prefix_discipline():
    """A tower name is a PREFIX WITH ITS DOT — a longer sibling must survive.
    Without the dot, `model.vision_projection` would also swallow a
    hypothetical `model.vision_projection_gate`, silently dropping a text
    weight and turning it into a loader refusal three steps later."""
    from types import SimpleNamespace

    from drinkme.arms import ckpt_to_skel

    f = ckpt_to_skel(SimpleNamespace(model_type="muse_glimmer"))
    assert f("model.vision_projection_gate.weight") == "model.vision_projection_gate.weight"
    assert f("model.vision_tower_norm.weight") == "model.vision_tower_norm.weight"


def test_non_text_towers_table_drives_both_halves():
    """The prune list and the skip list are ONE table. Every tower named for
    an arch must be skipped by that arch's translator — if they ever drift,
    the loader refuses (homeless tensor, or a meta parameter that never got
    one), so pin the agreement here instead."""
    from types import SimpleNamespace

    from drinkme.arms import _NON_TEXT_TOWERS, _VISION_TOWERS, ckpt_to_skel

    assert "qwen3_5" in _NON_TEXT_TOWERS
    # Muse-Glimmer's tower is served: the other table's
    assert "muse_glimmer" not in _NON_TEXT_TOWERS and "muse_glimmer" in _VISION_TOWERS
    for model_type, towers in _NON_TEXT_TOWERS.items():
        f = ckpt_to_skel(SimpleNamespace(model_type=model_type))
        for tower in towers:
            assert f(f"{tower}.weight") is None, (model_type, tower)
            assert f(f"{tower}.layers.0.mlp.fc1.weight") is None, (model_type, tower)


def test_the_default_tree_keeps_glimmers_tower():
    """Muse-Glimmer's tower is in _VISION_TOWERS, not _NON_TEXT_TOWERS:
    _prune_non_text does not touch it, and the tree nobody asked about
    (vision=None, the bench's) KEEPS it too — every vision-capable
    architecture is in _TOWER_BY_DEFAULT, since a served model carries its
    tower and a measurement must describe the whole thing.
    drop_vision_tower (an explicit vision=False) removes it."""
    import torch
    from types import SimpleNamespace

    from drinkme.arms import _prune_non_text, _wants_tower, drop_vision_tower

    def wrapper():  # the shape muse_glimmer ships
        m = torch.nn.Module()
        m.model = torch.nn.Module()
        m.model.language_model = torch.nn.Linear(8, 8)
        m.model.vision_tower = torch.nn.Linear(8, 8)
        m.model.vision_adapter = torch.nn.Linear(8, 8)
        m.model.vision_projection = torch.nn.Linear(8, 8)
        m.lm_head = torch.nn.Linear(8, 8)
        return m

    cfg = SimpleNamespace(model_type="muse_glimmer", vision_config=object())
    assert _prune_non_text(wrapper(), cfg).model.vision_tower is not None
    assert _wants_tower(cfg, None)
    assert not _wants_tower(cfg, False)
    m = drop_vision_tower(wrapper(), cfg)
    assert m.model.vision_tower is None
    assert m.model.vision_adapter is None
    assert m.model.vision_projection is None
    assert isinstance(m.model.language_model, torch.nn.Linear)
    # the towers' parameters are gone from the tree, which is the whole point:
    # eligible_linears cannot name what named_modules no longer walks
    assert sorted(n for n, _ in m.named_parameters()) == [
        "lm_head.bias", "lm_head.weight",
        "model.language_model.bias", "model.language_model.weight"]


def test_prune_non_text_is_a_noop_when_the_tower_is_absent():
    """qwen3_5's text skeleton never builds model.visual/mtp, and an arch with
    no entry is untouched — the prune must not invent or demand anything."""
    import torch
    from types import SimpleNamespace

    from drinkme.arms import _prune_non_text

    def plain():
        m = torch.nn.Module()
        m.model = torch.nn.Linear(8, 8)
        m.lm_head = torch.nn.Linear(8, 8)
        return m

    for model_type in ("qwen3_5", "qwen3", None):
        m = _prune_non_text(plain(), SimpleNamespace(model_type=model_type))
        assert sorted(n for n, _ in m.named_parameters()) == [
            "lm_head.bias", "lm_head.weight", "model.bias", "model.weight"]


def _tiny_glimmer_config():
    """A 2-layer muse_glimmer, dimensioned so BOTH towers' Linears clear the
    codec's eligibility bar (min dim >= 1024) — otherwise "no vision tensors
    in the pack" would pass for the wrong reason."""
    pytest.importorskip("transformers.models.muse_glimmer",
                        reason="muse_glimmer arrived in transformers 5.15.0")
    from transformers.models.muse_glimmer import MuseGlimmerConfig

    return MuseGlimmerConfig(
        text_config=dict(vocab_size=1024, hidden_size=1024, intermediate_size=1024,
                         num_hidden_layers=2, num_attention_heads=8,
                         num_key_value_heads=2, head_dim=128,
                         max_position_embeddings=128, sliding_window=64),
        vision_config=dict(hidden_size=1024, intermediate_size=1024,
                           num_hidden_layers=2, num_attention_heads=8,
                           pos_emb_height=4, pos_emb_width=4, max_position_embeddings=16),
        out_hidden_size=1024, projector_hidden_size=1024,
    )


def test_skeleton_muse_glimmer_keeps_the_tower_by_default():
    import torch

    from drinkme.arms import skeleton
    from drinkme.codec.swap import eligible_linears

    cfg = _tiny_glimmer_config()
    m = skeleton(cfg)
    # 5.15.0 has no text-only causal class for this arch: the text skeleton is
    # MuseGlimmerForConditionalGeneration, and the bench's default tree
    # (vision=None) keeps its vision submodules too — every architecture
    # with a served tower is in _TOWER_BY_DEFAULT.
    assert type(m).__name__ == "MuseGlimmerForConditionalGeneration"
    assert m.model.vision_tower is not None
    assert m.model.vision_adapter is not None
    assert m.model.vision_projection is not None
    assert next(m.parameters()).is_meta  # zero weight bytes, still

    names = [n for n, _, _, _ in eligible_linears(m)]
    assert names, "nothing packable — the pack would be empty"
    assert all(n == "lm_head" or n.startswith(("model.language_model.", "model.vision_"))
              for n in names), names
    # untied head (text_config.tie_word_embeddings is False, and 5.15.0's
    # _tied_weights_keys declaration does not override it) -> lm_head packs
    assert "lm_head" in names
    # the tower's own Linears are eligible too, now that the tree holds it
    assert any(n.startswith("model.vision_") for n in names)
    # k_proj/v_proj are 2 kv heads * 128 = 256 rows: below the codec's floor,
    # exactly as in the 30B. They stream raw, they are not "missing".
    assert not any(n.endswith(("self_attn.k_proj", "self_attn.v_proj")) for n in names)

    off = skeleton(cfg, vision=False)
    assert (off.model.vision_tower, off.model.vision_adapter, off.model.vision_projection) \
        == (None, None, None)


def test_muse_glimmer_prune_and_skip_partition_the_checkpoint():
    """THE loader contract, at unit scale: with the tower asked off
    (vision=False — the served tree without images; the bench's default
    now KEEPS it, so this checks the opt-out explicitly rather than by
    default), the tensors that survive translation are exactly the
    parameters of the pruned tree. Anything else is a refusal — a
    checkpoint tensor with no module, or a meta parameter no tensor ever
    filled."""
    import torch
    from transformers import AutoModelForImageTextToText

    from drinkme.arms import ckpt_to_skel, skeleton
    from drinkme.codec.swap import eligible_linears

    cfg = _tiny_glimmer_config()
    # the UNPRUNED wrapper stands in for the checkpoint: its parameter names
    # are the key space the real shards carry (measured: same prefixes)
    with torch.device("meta"):
        full = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16)
    ckpt_keys = [n for n, _ in full.named_parameters()]
    assert any(k.startswith("model.vision_tower.") for k in ckpt_keys)
    # ...and unpruned, that tower WOULD be packed — so the prune does work
    assert any(n.startswith("model.vision_") for n, _, _, _ in eligible_linears(full))

    to_skel = ckpt_to_skel(cfg, vision=False)
    survivors = {to_skel(k) for k in ckpt_keys} - {None}
    skipped = [k for k in ckpt_keys if to_skel(k) is None]
    m = skeleton(cfg, vision=False)
    assert survivors == set(n for n, _ in m.named_parameters())
    assert skipped and all(k.startswith("model.vision_") for k in skipped)

    # and the pack's own key set: every eligible weight exists in the
    # checkpoint key space (pack names == loader asks, by construction)
    want = {f"{n}.weight" for n, _, _, _ in eligible_linears(m)}
    assert want <= set(ckpt_keys)
