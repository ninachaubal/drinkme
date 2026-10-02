"""Every vision tower's ViT attention goes through vision.bounded_attention,
and the boot self-test checks that attention on the device, CPU.

ROUTING. AOTriton on gfx1151 is wrong at head width 72 (serving/vision.py
HEAD_ALIGN), which is the ViT head of Qwen3.5 AND of gemma-4, and
bounded_attention zero-pads it to 80 on ROCm. The served engine routes
gemma-4's and Muse-Glimmer's ViT attention through it: every attention
module, every call, and no SDPA call in the tower outside it, at the real
checkpoints' head widths (gemma-4's 72, padded on ROCm; Muse-Glimmer's 96, a
multiple of HEAD_ALIGN, left as it is). ROCm is simulated on the CPU
(vision._rocm), as in test_serving_vision_tower.py. The real configs, read
from the HF cache when present, pin those widths and that bound() routes
every layer.

THE BOOT SELF-TEST. vision.attention_self_test runs the tower's attention as
its modules call it against fp32 matmul and softmax, and HFEngine refuses
images when the two disagree, so a ROCm wheel that breaks the padded kernel
refuses images at boot instead of serving NaN features. A fake broken
kernel (NaN at the padded width, or only when masked) stands in for the
wheel; the accelerator is simulated (vision._on_accelerator).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from drinkme.serving import vision
from drinkme.serving.engine import GenerationRequest, SampleParams, complete

import test_serving_gemma_vision as gv
import test_serving_glimmer_vision as glv

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# the real checkpoints' ViT head widths (config.json; pinned below)
HEADS = {"gemma4": 72, "muse_glimmer": 96}


def _padded(d: int) -> int:
    return d + -d % vision.HEAD_ALIGN


def _toy(arch: str):
    """(test module, reference, served copy, tower): the architecture's toy
    from its own test file, its ViT at the real checkpoint's head width."""
    if arch == "gemma4":
        from transformers import Gemma4ForConditionalGeneration as cls

        mod, cfg = gv, gv.config(16, vision_head=HEADS[arch])
    else:
        from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration as cls

        mod, cfg = glv, glv.config(16, vision_width=2 * HEADS[arch], vision_heads=2)
    torch.manual_seed(0)
    ref = cls(cfg).eval().float()
    if arch == "gemma4":
        with torch.no_grad():  # the standardization buffers start empty
            ref.model.vision_tower.std_bias.normal_()
            ref.model.vision_tower.std_scale.uniform_(0.5, 1.5)
    for p in ref.parameters():
        p.requires_grad_(False)
    return mod, ref, copy.deepcopy(ref), mod._tower(cfg)


def _image(arch: str):
    return gv.image(40, 24, 1) if arch == "gemma4" else glv.image(48, 32, 3)


def _reference(arch: str, ref, img):
    """transformers' get_image_features for one image, as the processor
    batches it."""
    with torch.inference_mode():
        if arch == "gemma4":
            pv = torch.tensor(np.array(img.pixel_values))[None]
            pos = torch.tensor(vision.patch_positions(img.grid_thw, pv.shape[1]))[None]
            return ref.model.get_image_features(pv, pos).pooler_output[0]
        pv, grid = glv._batch([img])
        return ref.model.get_image_features(pv, grid).pooler_output[0]


@pytest.fixture
def registry(monkeypatch):
    """transformers' attention-function registry, restored afterwards: the
    tests below register spies under bound()'s names."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    cls = type(ALL_ATTENTION_FUNCTIONS)
    monkeypatch.setattr(cls, "_global_mapping", dict(cls._global_mapping))
    return ALL_ATTENTION_FUNCTIONS


# ------------------------------------------------------------ ROUTING --

@pytest.mark.parametrize("rocm", [False, True], ids=["cpu", "rocm"])
@pytest.mark.parametrize("arch", ["gemma4", "muse_glimmer"])
def test_every_vit_attention_call_goes_through_bounded_attention(arch, rocm, monkeypatch,
                                                                 registry):
    """The engine routes every ViT attention module of gemma-4's and
    Muse-Glimmer's towers through vision.bounded_attention (gemma's one
    masked call per layer, Glimmer's window and full calls), and every SDPA
    call the tower makes happens inside it. On ROCm the SDPA calls see
    gemma's head padded to 80 and Glimmer's 96 as it is; the features are
    transformers' get_image_features either way (bit for bit when nothing
    is padded)."""
    from transformers.integrations import sdpa_attention as sd

    mod, ref, model, tower = _toy(arch)
    want = _reference(arch, ref, _image(arch))
    depth, entered, kernels = [0], [], []
    real_bounded, real_sdpa = vision.bounded_attention, sd.sdpa_attention_forward

    def spy(module, query, *a, **kw):
        entered.append((id(module), depth[0]))
        depth[0] += 1
        try:
            return real_bounded(module, query, *a, **kw)
        finally:
            depth[0] -= 1

    def rec(module, query, *a, **kw):
        kernels.append((query.shape[-1], depth[0]))
        return real_sdpa(module, query, *a, **kw)

    monkeypatch.setattr(vision, "bounded_attention", spy)  # before bound() registers it
    monkeypatch.setattr(sd, "sdpa_attention_forward", rec)
    monkeypatch.setattr(vision, "_rocm", lambda x: rocm)
    eng = mod.engine((ref, model, mod._tokenizer(), tower))
    with torch.inference_mode():
        got = tower.features(eng.model, _image(arch))
    attn = [m for m in tower.module(eng.model).modules()
            if type(m).__name__ in vision._BOUNDED_TYPES[arch]]
    assert attn and all(m.config._attn_implementation == vision.BOUNDED_NAME for m in attn)
    assert {i for i, d in entered if d == 0} == {id(m) for m in attn}
    assert kernels and all(d > 0 for _w, d in kernels)
    head = HEADS[arch]
    assert {w for w, _d in kernels} == {_padded(head) if rocm else head}
    if rocm and head % vision.HEAD_ALIGN:
        torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)
    else:
        assert torch.equal(got, want)


def test_the_real_towers_have_these_head_widths_and_route_every_layer():
    """From the real checkpoints' configs (HF cache, meta device, no
    weights): the ViT head widths above, which of them HEAD_ALIGN pads (the
    Qwen3.5 and gemma-4 ViTs' 72, not Muse-Glimmer's 96), and bound()
    routing every one of each tower's attention layers."""
    from transformers import AutoConfig

    from drinkme import arms

    real = {"qwen3_5": (glv.VISION_ROWS["qwen3_5"], 27, 72),
            "gemma4": (glv.VISION_ROWS["gemma4"], 27, HEADS["gemma4"]),
            "muse_glimmer": (glv.VISION_ROWS["muse_glimmer"], 50, HEADS["muse_glimmer"])}
    for arch, ((repo, rev), layers, head) in real.items():
        cfg = AutoConfig.from_pretrained(glv._cached(repo, rev, filename="config.json"))
        tree = arms.skeleton(cfg, vision=True)
        tower = vision.Tower(arch, arms.vision_tower_path(cfg), 0, 1)
        attn = [m for m in tower.module(tree).modules()
                if type(m).__name__ in vision._BOUNDED_TYPES[arch]]
        assert len(attn) == layers and {m.head_dim for m in attn} == {head}, arch
        assert bool(head % vision.HEAD_ALIGN) == (arch != "muse_glimmer")
        assert vision.bound(tower, tree) == layers


# ------------------------------------------------- THE BOOT SELF-TEST --

@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("kernel", ["sound", *gv.BROKEN])
def test_attention_error_passes_a_sound_kernel_and_fails_a_broken_one(kernel, masked):
    """vision.attention_error, the convention the boot self-test and the
    gates' --fp32-check share (fp32 by matmul and softmax, rounded to the
    served dtype; sdpa.MAX_ULPS of the peak on the max, one bf16 eps on the
    mean): bf16 SDPA at gemma's head width and the self-test's logit scale
    (std ~9) is within it, and each wrong kernel (x3, rows permuted, noise,
    one NaN) is not. The two mask forms (sdpa's bool, eager's additive) give
    the same reference."""
    import math

    import torch.nn.functional as F

    g = torch.Generator().manual_seed(0)
    h, n, keys, d = 4, 192, 256, HEADS["gemma4"]
    amp = math.sqrt(9.0 / (d ** -0.5 * math.sqrt(d)))

    def rand(rows, a=1.0):
        return (torch.randn(1, h, rows, d, generator=g) * a).bfloat16()

    q, k, v = rand(n, amp), rand(keys, amp), rand(keys)
    mask = (torch.arange(keys) < keys - keys // 8).expand(1, 1, n, keys) if masked else None
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if kernel != "sound":
        out = gv.BROKEN[kernel](out)
    t = vision.attention_error(out, q, k, v, mask)
    assert t.ok == (kernel == "sound"), t.to_dict()
    if masked:
        additive = torch.zeros(mask.shape).masked_fill(~mask, float("-inf"))
        assert torch.equal(vision.fp32_attention(q, k, v, mask),
                           vision.fp32_attention(q, k, v, additive))


def _broken_when(pred):
    """F.scaled_dot_product_attention returning NaN when pred(query,
    attn_mask) holds: a ROCm wheel whose kernel breaks at one shape."""
    real = torch.nn.functional.scaled_dot_product_attention

    def sdpa(query, key, value, attn_mask=None, *a, **kw):
        out = real(query, key, value, attn_mask, *a, **kw)
        return out * float("nan") if pred(query, attn_mask) else out

    return sdpa


@pytest.mark.parametrize("arch", ["gemma4", "muse_glimmer"])
def test_the_self_test_agrees_on_a_sound_kernel_and_fails_on_a_broken_one(arch, monkeypatch,
                                                                         registry):
    """attention_self_test on a (simulated) ROCm device, at the tower's own
    head width, heads and scaling: a sound kernel AGREES; a kernel broken
    at the width the tower's calls reach fails, and one broken elsewhere
    does not; gemma-4's masked form is checked too (its ViT masks padding
    patches), so a kernel broken only under a mask fails gemma and not
    Glimmer; a kernel that raises fails with the error. On the CPU there is
    nothing to check."""
    _mod, _ref, model, tower = _toy(arch)
    vision.bound(tower, model)
    assert vision.attention_self_test(tower, model) is None  # the CPU: not checked
    monkeypatch.setattr(vision, "_on_accelerator", lambda x: True)
    monkeypatch.setattr(vision, "_rocm", lambda x: True)
    head, masked = HEADS[arch], arch == "gemma4"

    def check():
        return vision.attention_self_test(tower, model, rows=48, keys=96)

    ok = check()
    assert ok.ok and ok.error is None
    assert [form for form, _t in ok.tests] == (["unmasked", "masked"] if masked else ["unmasked"])
    assert all(t.max_abs < 1e-4 for _f, t in ok.tests)  # fp32 on the CPU
    assert ok.shape.startswith(f"2 heads x {head}" + (" (padded to 80)" if masked else ","))
    assert ok.lines()[0].endswith("AGREES") and len(ok.lines()) == 1
    served = _padded(head)
    with monkeypatch.context() as m:
        m.setattr(torch.nn.functional, "scaled_dot_product_attention",
                  _broken_when(lambda q, _m: q.shape[-1] == served))
        bad = check()
    assert not bad.ok and all(not t.ok for _f, t in bad.tests)
    assert "self-test failed" in "\n".join(bad.lines()) and "DISAGREES" in "\n".join(bad.lines())
    assert bad.reason.startswith("the vision tower's attention failed its boot self-test")
    with monkeypatch.context() as m:  # broken at a width this tower never sends
        m.setattr(torch.nn.functional, "scaled_dot_product_attention",
                  _broken_when(lambda q, _m: q.shape[-1] == served + 16))
        assert check().ok
    with monkeypatch.context() as m:
        m.setattr(torch.nn.functional, "scaled_dot_product_attention",
                  _broken_when(lambda _q, mask: mask is not None))
        only_masked = check()
    assert only_masked.ok == (not masked)
    if masked:
        assert [t.ok for _f, t in only_masked.tests] == [True, False]

    def boom(*a, **k):
        raise RuntimeError("HIP error: invalid device function")

    with monkeypatch.context() as m:
        m.setattr(torch.nn.functional, "scaled_dot_product_attention", boom)
        raised = check()
    assert not raised.ok and "invalid device function" in raised.reason


def test_a_failing_self_test_refuses_images_at_boot_and_serves_text(monkeypatch, capsys,
                                                                   registry):
    """HFEngine runs the self-test when it builds a tower on an accelerator:
    with gemma-4's head (72, padded to 80 on ROCm) and a kernel broken at
    80, the boot log carries the loud block and `image input: off`, an image
    request is refused naming the self-test, and text is served. With a
    sound kernel the same boot prints AGREES and serves images."""
    mod, ref, model, tower = _toy("gemma4")
    monkeypatch.setattr(vision, "_on_accelerator", lambda x: True)
    monkeypatch.setattr(vision, "_rocm", lambda x: True)
    monkeypatch.setattr(vision, "SELFTEST_ROWS", 48)
    monkeypatch.setattr(vision, "SELFTEST_KEYS", 96)
    tok = mod._tokenizer()

    def ask(eng, images):
        msgs = [mod.user(*(["what is in"] + [None] * len(images)))]
        return complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0, max_tokens=3),
                                               images=tuple(images)))

    good = mod.engine((ref, model, tok, tower))
    boot = capsys.readouterr().out
    assert "image input: attention self-test @ 2 heads x 72 (padded to 80)" in boot
    assert "AGREES" in boot and good.vision is not None
    assert ask(good, [_image("gemma4")]).text is not None
    with monkeypatch.context() as m:
        m.setattr(torch.nn.functional, "scaled_dot_product_attention",
                  _broken_when(lambda q, _m: q.shape[-1] == 80))
        bad = mod.engine((ref, copy.deepcopy(ref), tok, tower))
    boot = capsys.readouterr().out
    assert "Vision tower attention self-test failed — image input refused." in boot
    assert "[drinkme] image input: off — the vision tower's attention failed its boot " \
           "self-test" in boot
    assert bad.vision is None and bad._tower is None
    with pytest.raises(ValueError, match="failed its boot self-test"):
        ask(bad, [_image("gemma4")])
    assert ask(bad, []).text is not None


def test_the_selftest_instrument_runs_the_boot_check_alone(tmp_path, monkeypatch, registry):
    """bench/vision_selftest.py, from a config.json alone (gemma-4's toy at
    the 31B's head width): on the CPU nothing is checked; on a (simulated)
    ROCm device the padded check AGREES, a kernel broken at 72 never sees
    72, and --no-pad hands it 72, so the check FAILs (the GPU negative
    control) and HEAD_ALIGN is restored afterwards."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench"))
    import vision_selftest as st

    gv.config(16, vision_head=HEADS["gemma4"]).save_pretrained(tmp_path)
    snap = str(tmp_path)
    assert st.run(snap, "cpu")["verdict"] == "NOT CHECKED"
    monkeypatch.setattr(vision, "_on_accelerator", lambda x: True)
    monkeypatch.setattr(vision, "_rocm", lambda x: True)
    v = st.run(snap, "cpu", rows=48, keys=96)
    assert v["verdict"] == "PASS" and v["routed_modules"] == 1
    assert "2 heads x 72 (padded to 80)" in v["shape"] and set(v["tests"]) == {"unmasked",
                                                                              "masked"}
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention",
                        _broken_when(lambda q, _m: q.shape[-1] == 72))
    assert st.run(snap, "cpu", rows=48, keys=96)["verdict"] == "PASS"
    v = st.run(snap, "cpu", no_pad=True, rows=48, keys=96)
    assert v["verdict"] == "FAIL" and "padded" not in v["shape"]
    assert vision.HEAD_ALIGN == 16
