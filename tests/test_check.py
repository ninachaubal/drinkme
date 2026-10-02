"""`drinkme check` verdicts: pure assess() over fabricated headers — refusals name
their condition; estimates approximate what the swap will actually touch."""

import pytest

from drinkme.check import PACKED_RATIO, Verdict, assess

GIB = 1024**3


def linear(r=4096, c=4096, dtype="BF16"):
    return (dtype, [r, c])


def qwen_like(tied=False):
    t = {
        "model.embed_tokens.weight": linear(151936, 4096),
        "lm_head.weight": linear(151936, 4096),
        "model.layers.0.self_attn.q_proj.weight": linear(),
        "model.layers.0.mlp.gate_proj.weight": linear(12288, 4096),
        "model.layers.0.mlp.down_proj.weight": linear(4096, 12288),
        "model.norm.weight": ("BF16", [4096]),
    }
    cfg = {"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": tied}
    return cfg, t


def test_bf16_repo_is_eligible_with_honest_estimate():
    cfg, t = qwen_like()
    p = assess("x/y", None, cfg, t)
    assert p.ok
    # embeddings excluded; untied lm_head INCLUDED; norm (1D) excluded
    assert p.n_packable == 4
    packable = 2 * (151936 * 4096 + 4096 * 4096 + 12288 * 4096 + 4096 * 12288)
    total = packable + 2 * (151936 * 4096 + 4096)
    assert p.packable_pct == pytest.approx(100 * packable / total)
    assert p.est_packed_gib == pytest.approx(
        (packable * PACKED_RATIO + (total - packable)) / GIB)


def test_tied_head_is_excluded_only_when_tied():
    cfg, t = qwen_like(tied=True)
    p = assess("x/y", None, cfg, t)
    assert p.ok and p.n_packable == 3  # lm_head out under the tie rule


def test_f16_checkpoint_refused_by_name():
    cfg, t = qwen_like()
    t = {k: ("F16", s) for k, (d, s) in t.items()}
    p = assess("x/y", None, cfg, t)
    assert not p.ok and "F16" in p.reason and "BF16" in p.reason


def test_quantized_checkpoint_refused():
    cfg, t = qwen_like()
    t = {k: ("U8", s) for k, (d, s) in t.items()}
    p = assess("x/y", None, cfg, t)
    assert not p.ok and "quantized" in p.reason


def test_no_tensors_pass_shape_gates():
    cfg = {}
    t = {"tiny.weight": ("BF16", [64, 64])}
    p = assess("x/y", None, cfg, t)
    assert not p.ok and "shape gates" in p.reason


def test_unknown_architecture_is_a_note_not_a_refusal():
    cfg, t = qwen_like()
    cfg["architectures"] = ["FrobnicatorForCausalLM"]
    p = assess("x/y", None, cfg, t)
    assert p.ok
    assert any("Frobnicator" in n for n in p.notes)


def test_moe_3d_tensors_ride_raw_with_a_note():
    cfg, t = qwen_like()
    t["model.layers.0.mlp.experts.w2"] = ("BF16", [128, 4096, 1408])
    p = assess("x/y", None, cfg, t)
    assert p.ok
    assert any("3D" in n for n in p.notes)


def test_refusal_line_names_the_condition():
    p = Verdict("a/b", None, False, "no safetensors in this repo")
    assert "refused" in p.line() and "safetensors" in p.line()


def test_the_verdict_line_speaks_decimal_gb_while_the_fields_stay_gib():
    """The fields are GiB (bytes / 1024**3, kept for --json), but the
    line a person reads says GB, so its numbers are converted, not just
    relabelled: 1 GiB is 1.07 GB."""
    p = Verdict("x/y", None, True, None, total_gib=1.0, packable_pct=50.0,
                  est_packed_gib=0.5)
    assert p.line() == "check x/y: eligible — 50.0% of 1.07 GB packs, est. 0.54 GB"
    assert "GiB" not in p.line()


def test_depthwise_conv_weights_are_named_as_conv_not_fused_moe():
    """A dense hybrid's DeltaNet conv1d weight is [C, 1, K]; it rides
    raw like a fused/MoE 3D tensor, but it is not one, and the note says so."""
    cfg, t = qwen_like()
    t["model.layers.0.linear_attn.conv1d.weight"] = ("BF16", [8192, 1, 4])
    t["model.layers.1.linear_attn.conv1d.weight"] = ("BF16", [8192, 1, 4])
    p = assess("x/y", None, cfg, t)
    assert p.ok and "2 conv weights [C, 1, K] ride raw" in p.notes
    assert not any("fused/MoE" in n for n in p.notes)
    t["model.layers.0.mlp.experts.gate_up_proj"] = ("BF16", [8, 4096, 2048])
    p = assess("x/y", None, cfg, t)
    assert "1 3D bf16 tensors (fused/MoE layout) ride raw" in p.notes
    assert "2 conv weights [C, 1, K] ride raw" in p.notes
