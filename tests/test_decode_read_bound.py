"""The bandwidth bound's denominator is what one decode step reads.

bench.bound_fractions divides read_gb_s by each arm's `<arm>_decode_read_gb`
(arms.decode_read_bytes: the Linears, packed or raw, and one row of each
token embedding table, the vision tower's subtrees skipped), not by the
resident footprint `<arm>_weights_gb`, which also holds the tower and the
embedding rows a step never reads. On a vision model the resident figure
put the twin above its own bound: Muse-Glimmer-30B's twin decoded at
4.07 tok/s against 222.556 GB/s with 59.759 GB resident, 109% of a wall it
cannot pass.

The fixture is that model's tree, shape for shape (its checkpoint's
tensors: 52 text layers, the 50-layer ViT, the adapter and the projection,
an untied lm_head), built on the meta device, so no byte is allocated.
CPU only, no checkpoint.
"""
import pytest
import torch

from drinkme import arms, bench

BF16 = torch.bfloat16

# Muse-Glimmer-30B's text model and vision tower (the checkpoint's shapes)
H, I, L, V, Q, KV = 6656, 19968, 52, 202048, 4096, 256
VH, VI, VL = 1536, 8960, 50
# the twin arm's gulp record on gfx1151: decode tok/s, read_gb_s, twin_weights_gb
TWIN_TOK_S, READ_GB_S, TWIN_WEIGHTS_GB = 4.07, 222.556, 59.759


def _lin(o, i):
    return torch.nn.Linear(i, o, bias=False, device="meta", dtype=BF16)


def _glimmer_shaped() -> torch.nn.Module:
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    lm = root.model.language_model = torch.nn.Module()
    lm.embed_tokens = torch.nn.Embedding(V, H, device="meta", dtype=BF16)
    lm.layers = torch.nn.ModuleList()
    for _ in range(L):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        for name, (o, i) in dict(q_proj=(Q, H), gate_proj=(Q, H), k_proj=(KV, H), v_proj=(KV, H),
                                 o_proj=(H, Q)).items():
            setattr(layer.self_attn, name, _lin(o, i))
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj = _lin(I, H), _lin(I, H), _lin(H, I)
        lm.layers.append(layer)
    tower = root.model.vision_tower = torch.nn.Module()
    tower.patch_embedder = torch.nn.Module()
    tower.patch_embedder.patch_embedding = _lin(VH, 1176)
    tower.patch_embedder.position_embedding_table = torch.nn.Embedding(1024, VH, device="meta", dtype=BF16)
    tower.layers = torch.nn.ModuleList()
    for _ in range(VL):
        layer = torch.nn.Module()
        layer.attn = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "proj"):
            setattr(layer.attn, name, _lin(VH, VH))
        layer.mlp = torch.nn.Module()
        layer.mlp.fc1, layer.mlp.fc2 = _lin(VI, VH), _lin(VH, VI)
        tower.layers.append(layer)
    root.model.vision_adapter = torch.nn.Module()
    root.model.vision_adapter.fc1, root.model.vision_adapter.fc2 = _lin(4096, 6144), _lin(4096, 4096)
    root.model.vision_projection = _lin(H, 4096)
    root.lm_head = _lin(V, H)
    return root


TOWER = arms._VISION_TOWERS["muse_glimmer"]
TEXT_LINEAR_PARAMS = L * (3 * Q * H + 2 * KV * H + 3 * I * H) + V * H


def test_a_decode_step_reads_the_text_linears_and_one_embedding_row():
    got = arms.decode_read_bytes(_glimmer_shaped(), TOWER)
    assert got["raw_linear_bytes"] == 2 * TEXT_LINEAR_PARAMS
    assert got["embedding_row_bytes"] == 2 * H  # one row of the table, not 2.69 GB of it
    assert got["packed_bytes"] == 0
    assert got["total_bytes"] == 2 * TEXT_LINEAR_PARAMS + 2 * H == 53_017_129_984
    assert got["counts"] == {"bf16": 8 * L + 1}
    assert got["skipped"] == list(TOWER)


def test_the_tower_counts_toward_memory_and_not_toward_the_read():
    """Without the skip the tower's 304 Linears are counted (and the
    position table's one row): the resident tree holds them, a text decode
    step never runs them."""
    tree = _glimmer_shaped()
    everything = arms.decode_read_bytes(tree)
    text = arms.decode_read_bytes(tree, TOWER)
    tower_linears = 2 * (VL * (4 * VH * VH + 2 * VI * VH) + VH * 1176 + 4096 * 6144 + 4096 * 4096 + H * 4096)
    assert everything["total_bytes"] - text["total_bytes"] == tower_linears + 2 * VH
    assert everything["counts"]["bf16"] - text["counts"]["bf16"] == 6 * VL + 4
    assert "skipped" not in everything


def test_the_fraction_stays_under_the_wall_at_the_twins_real_rate():
    """The twin's record, read two ways: over its resident footprint it
    claims 109% of the wall; over what a decode step reads, 97%."""
    read_gb = bench._gb(arms.decode_read_bytes(_glimmer_shaped(), TOWER)["total_bytes"])
    metrics = [bench._metric("read_gb_s", READ_GB_S, "GB/s"),
               bench._metric("twin_decode_tok_s", TWIN_TOK_S, "tok/s"),
               bench._metric("twin_weights_gb", TWIN_WEIGHTS_GB, "GB"),
               bench._metric("twin_decode_read_gb", read_gb, "GB")]
    assert TWIN_TOK_S / (READ_GB_S / TWIN_WEIGHTS_GB) > 1.09  # the resident denominator
    [(arm, frac)] = bench.bound_fractions(metrics)
    assert arm == "twin" and frac == pytest.approx(0.9696, abs=1e-4) and frac <= 1


def test_a_record_without_the_decode_read_metric_has_no_bound():
    """A record without `<arm>_decode_read_gb` (an arm that walks no module
    tree) carries only the resident footprint; bound_fractions states no
    bound for it rather than dividing by the footprint."""
    metrics = [bench._metric("read_gb_s", READ_GB_S, "GB/s"),
               bench._metric("twin_decode_tok_s", TWIN_TOK_S, "tok/s"),
               bench._metric("twin_weights_gb", TWIN_WEIGHTS_GB, "GB")]
    assert bench.bound_fractions(metrics) == []


def test_a_tied_head_reads_the_whole_table_and_the_lookup_one_row():
    """The tied lm_head is a Linear over the embedding's own Parameter: the
    head's call reads every row, the lookup one more."""
    tree = torch.nn.Module()
    tree.embed = torch.nn.Embedding(1000, 64, dtype=BF16)
    tree.body = torch.nn.Linear(64, 64, bias=False, dtype=BF16)
    tree.head = torch.nn.Linear(64, 1000, bias=False, dtype=BF16)
    tree.head.weight = tree.embed.weight
    got = arms.decode_read_bytes(tree)
    assert got["raw_linear_bytes"] == 2 * (64 * 64 + 1000 * 64)
    assert got["embedding_row_bytes"] == 2 * 64


class _Cfg:
    model_type = "muse_glimmer"
    vision_config = {"hidden_size": VH}


def test_tower_paths_names_the_subtrees_the_tree_holds(tmp_path):
    """Judged against the tree's own config when the source has no
    config.json (the tests' sources), and only the subtrees still in the
    tree: a pruned tower (DRINKME_VISION=0, drop_vision_tower) skips
    nothing."""
    tree = _glimmer_shaped()
    tree.config = _Cfg()
    assert arms.tower_paths(tree, tmp_path) == TOWER
    arms.drop_vision_tower(tree, tree.config)
    assert arms.tower_paths(tree, tmp_path) == ()
    text = torch.nn.Module()
    text.config = type("C", (), {"model_type": "qwen3", "vision_config": None})()
    assert arms.tower_paths(text, tmp_path) == ()
