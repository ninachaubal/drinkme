"""The Apple silicon bench records what one decode step reads.

arms_mlx.decode_read_bytes restates arms.decode_read_bytes over the MLX
engine's tree (arms.py imports torch), and arms_mlx.run_arms records it per
arm as `<arm>_bytes_per_token`, which bench turns into
`<arm>_decode_read_gb`, the denominator of the arm's bandwidth bound
(site/aggregator's F3 holds a record's decode speed to it).

No torch and no mlx at module scope, so the file collects on the Mac (no
torch) and in the Linux CPU suite (no mlx). The tally's arithmetic and
run_arms' wiring need neither. Qwen3-1.7B's stock tree is checked on each
runtime where that runtime is installed, both against the one dict its
config.json gives: the torch walk on the meta device, the MLX walk on
mlx-lm's unevaluated skeleton. The engine's own modules (radix, twin, raw
fallback) are walked on the toy in tests/test_serving_engine_mlx.py.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from drinkme import arms_mlx, bench
from drinkme.arms_mlx import decode_read_tally

# Qwen3-1.7B's config.json: hidden, intermediate, layers, attention heads,
# kv heads, head_dim, vocab; tie_word_embeddings true
H, I, L, NH, NKV, HD, V = 2048, 6144, 28, 16, 8, 128, 151936
QWEN3_1_7B = {"model_type": "qwen3", "hidden_size": H, "intermediate_size": I,
              "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": NKV,
              "head_dim": HD, "vocab_size": V, "rms_norm_eps": 1e-6,
              "max_position_embeddings": 40960, "rope_theta": 1000000.0,
              "tie_word_embeddings": True}


def _qwen3_1_7b_reads() -> list:
    """(kind, bytes) for each module a stock Qwen3-1.7B decode step reads,
    off the config alone: every layer's seven bf16 projections, the tied
    head (the embedding table as a Linear) and one embedding row."""
    proj = [(NH * HD, H), (NKV * HD, H), (NKV * HD, H), (H, NH * HD), (I, H), (I, H), (H, I)]
    reads = [("bf16", 2 * o * i) for _ in range(L) for o, i in proj]
    return reads + [("bf16", 2 * V * H), ("embedding_row", 2 * H)]


def test_the_tally_buckets_each_kind_as_the_torch_walk_does():
    """Codec and twin weights are packed_bytes; the raw fallback and plain
    bf16 are raw_linear_bytes; an embedding contributes its row and is not
    a Linear in `counts`."""
    got = decode_read_tally([("drinkme_codec", 700), ("drinkme_codec", 300), ("drinkme_twin", 2000),
                             ("drinkme_raw", 512), ("bf16", 4096), ("embedding_row", 128)])
    assert got == {"packed_bytes": 3000, "raw_linear_bytes": 4608, "embedding_row_bytes": 128,
                   "counts": {"drinkme_codec": 2, "drinkme_twin": 1, "drinkme_raw": 1, "bf16": 1},
                   "total_bytes": 7736}
    assert decode_read_tally([]) == {"packed_bytes": 0, "raw_linear_bytes": 0,
                                     "embedding_row_bytes": 0, "counts": {}, "total_bytes": 0}


def test_qwen3_1_7b_stock_reads_its_linears_the_tied_table_once_and_one_row():
    got = decode_read_tally(_qwen3_1_7b_reads())
    linears = L * (2 * NH * HD * H + 2 * NKV * HD * H + 3 * I * H)
    assert got["raw_linear_bytes"] == 2 * (linears + V * H)
    assert got["embedding_row_bytes"] == 2 * H
    assert got["counts"] == {"bf16": 7 * L + 1}
    assert got["total_bytes"] == 3_440_906_240  # bench publishes 3.441 GB
    assert bench._gb(got["total_bytes"]) == pytest.approx(3.441, abs=5e-4)


def test_the_torch_walk_gives_qwen3_1_7b_the_same_dict(tmp_path):
    """arms.decode_read_bytes on transformers' Qwen3-1.7B, built on the meta
    device: the dict the MLX walk is held to below."""
    torch = pytest.importorskip("torch")
    from transformers import AutoConfig, AutoModelForCausalLM

    from drinkme import arms

    cfg = AutoConfig.for_model("qwen3", **{k: v for k, v in QWEN3_1_7B.items() if k != "model_type"})
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert arms.decode_read_bytes(model, arms.tower_paths(model, tmp_path)) == \
        decode_read_tally(_qwen3_1_7b_reads())


def test_the_mlx_walk_gives_qwen3_1_7b_the_same_dict():
    """arms_mlx.decode_read_bytes on mlx-lm's Qwen3-1.7B skeleton, cast to
    bf16 and never evaluated (shapes and dtypes only, no weight allocated):
    no lm_head module (tied), the head counted off the embedding table."""
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from drinkme.serving.engine_mlx import build_skeleton

    model, _ = build_skeleton(dict(QWEN3_1_7B))
    model.set_dtype(mx.bfloat16)
    assert "lm_head" not in model
    before = mx.get_active_memory()
    assert arms_mlx.decode_read_bytes(model) == decode_read_tally(_qwen3_1_7b_reads())
    assert mx.get_active_memory() == before  # the walk read shapes, evaluated nothing


def test_run_arms_records_each_arms_read_while_its_engine_is_resident(tmp_path, monkeypatch):
    """Each arm's read is taken off that arm's engine, after its load and
    before its timed passes, as arms.run_arms takes the torch arms'. The
    mlx runtime, the loaders and the timing are stubbed: this is the
    wiring, not the walk."""
    from drinkme.codec import pack as P
    from drinkme.serving import checkpoint

    core = types.ModuleType("mlx.core")
    core.metal = types.SimpleNamespace(is_available=lambda: False)
    core.synchronize = core.clear_cache = lambda: None
    pkg = types.ModuleType("mlx")
    pkg.__path__, pkg.core = [], core
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    events: list = []

    class _Model:
        def __init__(self, arm):
            self.arm = arm

        def named_modules(self):  # what run_arms looks for: an mlx Module's tree
            return []

    class _Eng:
        resident_bytes = 1000
        meta = {"profile": "sip"}

        def __init__(self, arm):
            self.arm, self.model = arm, _Model(arm)

    def stock(model_id, revision, ctx=None, snap=None):
        events.append("load stock")
        return _Eng("stock")

    def compressed(model_id, revision, pack_dir, ctx=None, path=None, rope_scaling=None, snap=None):
        arm = "twin" if path == "twin" else "compressed"
        events.append(f"load {arm}")
        return _Eng(arm)

    def read(model):
        events.append(f"read {model.arm}")
        return {"total_bytes": len(events), "arm": model.arm}

    def decode(eng, ids, n, reps):
        events.append(f"decode {eng.arm}")
        return None, [10.0, 10.0, 10.0]

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "meta.json").write_text(json.dumps({"profile": "sip"}))
    monkeypatch.setattr(checkpoint, "resolve_source", lambda m, r, p: (str(tmp_path), "a" * 40))
    monkeypatch.setattr(checkpoint, "tokenizer",
                        lambda path, rev: types.SimpleNamespace(encode=lambda text: [1, 2, 3]))
    monkeypatch.setattr(P, "bpw_of_pack_dir", lambda path: (11.0, 11.2))
    monkeypatch.setattr(arms_mlx, "measure_bandwidth", lambda: {
        "probe_bytes": 10**9, "read_bytes_s": 10**11, "copy_bytes_s": 9 * 10**10, "device": "stub"})
    monkeypatch.setattr(arms_mlx, "decode_read_bytes", read)
    monkeypatch.setattr(arms_mlx, "timed_decode", decode)
    monkeypatch.setattr(arms_mlx, "timed_prefill", lambda eng, ids, reps: [500.0, 500.0, 500.0])
    monkeypatch.setattr(arms_mlx, "timed_ttft", lambda eng, ids, reps: [0.05, 0.05, 0.05])
    monkeypatch.setattr(arms_mlx, "_packed_modules", lambda model: iter([1]))
    monkeypatch.setattr(arms_mlx, "_device", lambda: "stub")
    monkeypatch.setattr(arms_mlx, "_settle_host", lambda: None)
    monkeypatch.setitem(sys.modules, "drinkme.serving.engine_mlx", types.SimpleNamespace(
        load_stock_mlx=stock, load_compressed_mlx=compressed, TWIN="twin"))
    r = arms_mlx.run_arms("org/toy", None, "hi", pack_dir=str(pack_dir), fused=False, prefill_len=4)
    assert events == ["load stock", "read stock", "decode stock",
                      "load twin", "read twin", "decode twin",
                      "load compressed", "read compressed", "decode compressed"]
    for arm in ("stock", "twin", "compressed"):
        assert r[f"{arm}_bytes_per_token"]["arm"] == arm
