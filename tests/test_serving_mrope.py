"""serving/mrope.py: Qwen3.5's M-RoPE positions, pinned against
transformers' own `Qwen3_5Model.get_rope_index` (built from a tiny config
on the meta device; the method reads no weights) over synthetic id streams:
text only, one image, several, images at the very start and end,
back-to-back, interleaved, odd grids, and a seeded fuzz. The decode rule
past the prompt is pinned against the reference's
`compute_3d_position_ids`."""

from __future__ import annotations

import numpy as np
import pytest

from drinkme.serving import mrope

PAD, START, END = 299, 297, 298  # image pad, <|vision_start|>, <|vision_end|> in the toy vocab


@pytest.fixture(scope="module")
def reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    cfg = Qwen3_5Config(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=32, vocab_size=300,
                         layer_types=["linear_attention", "full_attention"]),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, spatial_merge_size=2, patch_size=16),
        image_token_id=PAD, vision_start_token_id=START, vision_end_token_id=END)
    with torch.device("meta"):
        model = Qwen3_5Model(cfg)
    return torch, model


def _ids(segments, rng) -> tuple[list[int], list[tuple[int, int, int]]]:
    """[("text", n) | ("image", (t, h, w)) | ("wrapped", (t, h, w))] -> ids
    and grids. "wrapped" is the template's <|vision_start|> run
    <|vision_end|>; a bare "image" is a run with nothing around it."""
    ids, grids = [], []
    for kind, arg in segments:
        if kind == "text":
            ids += [int(x) for x in rng.integers(1, 290, size=arg)]
            continue
        n = mrope.image_token_count(arg)
        run = [PAD] * n
        ids += [START, *run, END] if kind == "wrapped" else run
        grids.append(arg)
    return ids, grids


def _reference(reference, ids, grids):
    torch, model = reference
    t = torch.tensor([ids])
    pos, delta = model.get_rope_index(t, (t == PAD).int(),
                                      image_grid_thw=torch.tensor(grids) if grids else None)
    return pos[:, 0].numpy(), int(delta.item())


CASES = {
    "text only": [("text", 11)],
    "one image": [("text", 4), ("wrapped", (1, 4, 6)), ("text", 3)],
    "two images": [("text", 5), ("wrapped", (1, 8, 4)), ("text", 2), ("wrapped", (1, 6, 10)),
                   ("text", 7)],
    "image at the very start": [("image", (1, 6, 6)), ("text", 9)],
    "image at the very end": [("text", 9), ("image", (1, 2, 12))],
    "only an image": [("image", (1, 4, 4))],
    "back to back": [("text", 3), ("wrapped", (1, 4, 4)), ("wrapped", (1, 2, 8)),
                     ("wrapped", (1, 10, 2)), ("text", 3)],
    "interleaved": [("wrapped", (1, 4, 2)), ("text", 6), ("wrapped", (1, 2, 2)), ("text", 1),
                    ("wrapped", (1, 12, 6)), ("text", 4)],
    "odd grids": [("text", 2), ("wrapped", (1, 2, 6)), ("text", 2), ("wrapped", (1, 6, 2)),
                  ("text", 2), ("wrapped", (1, 14, 10))],
    "a screenshot's grid": [("text", 30), ("wrapped", (1, 68, 120)), ("text", 40)],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_positions_equal_get_rope_index(reference, name):
    ids, grids = _ids(CASES[name], np.random.default_rng(1))
    pos4, delta = mrope.rope_positions(ids, grids, PAD)
    ref, ref_delta = _reference(reference, ids, grids)
    assert pos4.dtype == np.int64 and pos4.shape == (4, len(ids))
    assert np.array_equal(pos4[1:], ref) and delta == ref_delta
    assert np.array_equal(pos4[0], np.arange(len(ids)))


def test_positions_equal_get_rope_index_fuzz(reference):
    rng = np.random.default_rng(7)
    for _ in range(150):
        segs = []
        for _ in range(int(rng.integers(1, 7))):
            if rng.random() < 0.5:
                segs.append(("text", int(rng.integers(1, 9))))
            else:
                grid = (1, 2 * int(rng.integers(1, 9)), 2 * int(rng.integers(1, 9)))
                segs.append(("wrapped" if rng.random() < 0.8 else "image", grid))
        # two bare runs side by side are one run to the reference (it groups
        # maximal runs); the template never emits that, so neither does this
        for (a, _), (b, _) in zip(segs, segs[1:]):
            if a == b == "image":
                break
        else:
            ids, grids = _ids(segs, rng)
            pos4, delta = mrope.rope_positions(ids, grids, PAD)
            ref, ref_delta = _reference(reference, ids, grids)
            assert np.array_equal(pos4[1:], ref), segs
            assert delta == ref_delta, segs


def test_a_worked_example():
    """Four text tokens, a 2x3 merged image, three text tokens: the image
    takes one T position, H over two rows, W over three columns, and the
    text after it resumes at 4 + max(2, 3) = 7."""
    pos4, delta = mrope.rope_positions([1, 2, 3, 4] + [PAD] * 6 + [5, 6, 7], [(1, 4, 6)], PAD)
    assert pos4[1].tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 7, 8, 9]
    assert pos4[2].tolist() == [0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 7, 8, 9]
    assert pos4[3].tolist() == [0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9]
    assert delta == -3


@pytest.mark.parametrize("n", [0, 1, 17, 4096])
def test_text_only_is_exactly_todays_positions(n):
    """No image: every row is arange(T) (what the text model builds itself
    when it is given no position_ids) and delta is 0."""
    ids = list(np.random.default_rng(n).integers(1, 290, size=n))
    pos4, delta = mrope.rope_positions(ids, [], PAD)
    assert np.array_equal(pos4, np.broadcast_to(np.arange(n), (4, n))) and delta == 0


def test_past_the_prompt_positions_continue_with_delta(reference):
    """Decode, verify and re-arm take [p, p + delta, p + delta, p + delta]:
    the reference's compute_3d_position_ids with its stored rope_deltas and
    a cache holding the prompt, and also what the prompt's own trailing
    text already holds."""
    torch, model = reference
    ids, grids = _ids(CASES["two images"], np.random.default_rng(3))
    pos4, delta = mrope.rope_positions(ids, grids, PAD)
    n = len(ids)
    last_img = max(i for i, t in enumerate(ids) if t == PAD)
    for p in range(last_img + 1, n):
        assert pos4[1:, p].tolist() == [p + delta] * 3

    class Seen:
        def get_seq_length(self):
            return n

    model.rope_deltas = torch.tensor([[delta]])
    try:
        for k in (1, 5):
            ref = model.compute_3d_position_ids(
                input_ids=torch.full((1, k), 7), inputs_embeds=torch.zeros(1, k, 64),
                past_key_values=Seen())
            step = mrope.step_positions(n, k, delta)
            assert np.array_equal(step[1:], ref[:, 0].numpy())
            assert np.array_equal(step[0], np.arange(n, n + k))
    finally:
        model.rope_deltas = None


def test_a_prompt_and_images_that_disagree_are_refused():
    run = [PAD] * mrope.image_token_count((1, 4, 4))
    with pytest.raises(ValueError, match="shorter than"):
        mrope.rope_positions([1, *run[:-1], 2], [(1, 4, 4)], PAD)
    with pytest.raises(ValueError, match="have no image"):
        mrope.rope_positions([1, *run, 2, PAD], [(1, 4, 4)], PAD)
    with pytest.raises(ValueError, match="no placeholder run left"):
        mrope.rope_positions([1, *run, 2], [(1, 4, 4), (1, 2, 2)], PAD)
    with pytest.raises(ValueError, match="no tokens"):
        mrope.rope_positions([1, 2], [(1, 1, 1)], PAD)
