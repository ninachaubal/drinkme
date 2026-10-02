"""The SDPA backend probe: arithmetic pinned to the measured 56.13 GiB OOM, shape
extraction off real config shapes, and — the part that matters for a stranger —
the whole module degrading on a box with no GPU instead of crashing a load.

No test here needs an accelerator. The probe runs on CPU (torch ships a CPU
flash path, so a laptop gets a real answer rather than an error), and every
branch that only exists on ROCm is exercised through a fabricated SdpaReport.
"""

import pytest

from drinkme import sdpa
from drinkme.sdpa import EXPERIMENTAL_ENV, AttnShape, SdpaReport, SelfTest

GIB = 1024**3
QWEN38 = AttnShape(head_dim=256, n_heads=24, n_kv_heads=4)


@pytest.fixture(autouse=True)
def no_flag(monkeypatch):
    """The flag leaks in from a developer's shell otherwise, and half these
    assertions are about which branch an unset flag takes."""
    monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)


def fabricate(**kw) -> SdpaReport:
    """A report as if probed, without probing — the ROCm-only branches have to
    be assertable from a CPU box."""
    base = dict(shape=QWEN38, device="cuda", arch="gfx1151",
                torch_version="2.12.0a0+rocm7.13.0a20260411",
                total_memory=133143986176,
                backends={"flash": False, "mem_efficient": False,
                          "cudnn": False, "math": True})
    base.update(kw)
    return SdpaReport(**base)


# -- the arithmetic, pinned to the crash ------------------------------------

def test_score_matrix_reproduces_the_2026_08_24_allocation():
    """The prompt tokenized to 25,057 and the OOM asked for 56.13 GiB. If this
    ever stops matching, the story in the module docstring is wrong."""
    b = sdpa.score_matrix_bytes(n_heads=24, seq_len=25057)
    assert round(b / GIB, 2) == 56.13


def test_score_matrix_is_quadratic_not_linear():
    at_8k = sdpa.score_matrix_bytes(24, 8192)
    assert sdpa.score_matrix_bytes(24, 16384) == 4 * at_8k


def test_max_prompt_before_inverts_the_score_matrix():
    wall = sdpa.max_prompt_before(24, 133143986176)
    assert sdpa.score_matrix_bytes(24, wall) <= 133143986176
    assert sdpa.score_matrix_bytes(24, wall + 1) > 133143986176


def test_max_prompt_before_refuses_nonsense_instead_of_dividing_by_zero():
    assert sdpa.max_prompt_before(0, 1 << 40) == 0
    assert sdpa.max_prompt_before(24, 0) == 0


# -- shapes off configs -----------------------------------------------------

class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_explicit_head_dim_wins():
    s = sdpa.shape_from_config(Cfg(num_attention_heads=24, num_key_value_heads=4,
                                   head_dim=256, hidden_size=5120))
    assert s == QWEN38  # 5120/24 is not 256; the config's own head_dim rules


def test_head_dim_derived_when_the_config_omits_it():
    # Qwen2.5-72B's shape: no head_dim key, 8192/64 = 128
    s = sdpa.shape_from_config(Cfg(num_attention_heads=64, num_key_value_heads=8,
                                   hidden_size=8192))
    assert s == AttnShape(128, 64, 8)


def test_kv_heads_default_to_query_heads_for_plain_mha():
    s = sdpa.shape_from_config(Cfg(num_attention_heads=16, hidden_size=2048))
    assert s == AttnShape(128, 16, 16)


def test_wrapped_multimodal_config_is_unwrapped():
    """Qwen3.8-27B is a Qwen3_5ForConditionalGeneration: the head counts live
    under text_config, and reading the outer config gets the vision tower."""
    text = Cfg(num_attention_heads=24, num_key_value_heads=4, head_dim=256)
    outer = Cfg(hidden_size=1152, get_text_config=lambda: text)
    assert sdpa.shape_from_config(outer) == QWEN38


def test_plain_dict_config_works():
    assert sdpa.shape_from_config(
        {"num_attention_heads": 24, "num_key_value_heads": 4,
         "head_dim": 256}) == QWEN38


def test_unrecognizable_config_returns_none_rather_than_raising():
    assert sdpa.shape_from_config(None) is None
    assert sdpa.shape_from_config(Cfg()) is None
    assert sdpa.shape_from_config(Cfg(num_attention_heads=24)) is None  # no dim


# gemma-4-31B-it's text config (config.json): 50 sliding layers at head_dim
# 256 with 16 kv heads, 10 full layers at 512 with 4, full every sixth layer.
GEMMA4_KINDS = (["sliding_attention"] * 5 + ["full_attention"]) * 10
GEMMA4_TEXT = {"num_hidden_layers": 60, "hidden_size": 5376, "num_attention_heads": 32,
               "num_key_value_heads": 16, "num_global_key_value_heads": 4,
               "head_dim": 256, "global_head_dim": 512, "sliding_window": 1024,
               "attention_k_eq_v": True, "layer_types": GEMMA4_KINDS}
GEMMA4_SHAPES = [(AttnShape(256, 32, 16), "50 of 60 layers (sliding_attention)"),
                 (AttnShape(512, 32, 4), "10 of 60 layers (full_attention)")]


def test_gemma4_per_layer_config_yields_both_head_widths():
    """transformers 5 refuses `head_dim` on gemma-4's whole config
    (AmbiguousGlobalPerLayerAttributeError); each layer's own config has
    it. The real class, built from the fields, no download."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("this transformers has no gemma-4")
    text = transformers.Gemma4TextConfig(**GEMMA4_TEXT)
    assert sdpa.shapes_from_config(text) == GEMMA4_SHAPES
    outer = transformers.Gemma4Config(text_config=text.to_dict())
    assert sdpa.shapes_from_config(outer) == GEMMA4_SHAPES


def test_gemma4_config_json_dict_yields_both_head_widths():
    assert sdpa.shapes_from_config({"text_config": GEMMA4_TEXT}) == GEMMA4_SHAPES
    assert sdpa.shapes_from_config(GEMMA4_TEXT) == GEMMA4_SHAPES
    # without attention_k_eq_v the full layers keep the plain kv heads, as
    # Gemma4TextConfig builds them
    plain = sdpa.shapes_from_config(dict(GEMMA4_TEXT, attention_k_eq_v=False))
    assert plain[1][0] == AttnShape(512, 32, 16)


def test_a_config_that_refuses_the_whole_model_field_is_read_per_layer():
    class PerLayer(Cfg):
        def __getattr__(self, key):
            if key in ("head_dim", "num_key_value_heads"):
                raise RuntimeError(f"{key!r} is a per-layer attribute")
            raise AttributeError(key)

    layers = [Cfg(num_attention_heads=32, num_key_value_heads=16, head_dim=256)] * 5 + [
        Cfg(num_attention_heads=32, num_key_value_heads=4, head_dim=512)]
    cfg = PerLayer(num_attention_heads=32, per_layer_config=layers,
                   layer_types=GEMMA4_KINDS[:6])
    assert sdpa.shapes_from_config(cfg) == [
        (AttnShape(256, 32, 16), "5 of 6 layers (sliding_attention)"),
        (AttnShape(512, 32, 4), "1 of 6 layers (full_attention)")]
    assert sdpa.shape_from_config(cfg) == AttnShape(256, 32, 16)
    assert sdpa.shapes_from_config(PerLayer(num_attention_heads=32)) == []  # never raises


def test_a_uniform_model_is_one_shape_with_no_label():
    """Qwen3.5's DeltaNet layers run no SDPA: their per-layer head fields do
    not count as a second shape."""
    full = Cfg(num_attention_heads=24, num_key_value_heads=4, head_dim=256)
    delta = Cfg(num_attention_heads=16, num_key_value_heads=16, head_dim=128)
    cfg = Cfg(per_layer_config=[delta, delta, delta, full] * 2,
              layer_types=["linear_attention"] * 3 + ["full_attention"]
              + ["linear_attention"] * 3 + ["full_attention"])
    assert sdpa.shapes_from_config(cfg) == [(QWEN38, "")]
    assert sdpa.shapes_from_config(Cfg(num_attention_heads=24, num_key_value_heads=4,
                                       head_dim=256)) == [(QWEN38, "")]


# -- the flag ---------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    ("1", True), ("y", True), ("YES", True), ("true", True), ("On", True),
    ("0", False), ("", False), ("maybe", False),
])
def test_flag_parsed_the_way_torch_parses_it(monkeypatch, val, expected):
    monkeypatch.setenv(EXPERIMENTAL_ENV, val)
    assert sdpa.flag_is_set() is expected
    assert sdpa.flag_value() == val


def test_flag_absent_is_not_set(monkeypatch):
    assert sdpa.flag_value() is None
    assert sdpa.flag_is_set() is False


def test_report_carries_the_flag_verbatim(monkeypatch):
    monkeypatch.setenv(EXPERIMENTAL_ENV, "1")
    rep = sdpa.probe_backends(AttnShape(64, 4, 4), device="cpu", seq_len=16)
    assert rep.to_dict()[EXPERIMENTAL_ENV] == "1"
    assert rep.flag_set is True


# -- probing, on whatever box this is -------------------------------------

def test_probe_on_cpu_answers_instead_of_crashing():
    """The laptop case: someone reading the README runs the probe
    with no GPU in the machine. It must produce a verdict."""
    rep = sdpa.probe_backends(AttnShape(64, 4, 4), device="cpu", seq_len=16)
    assert rep.probed and rep.device == "cpu"
    assert rep.backends["math"] is True  # math is always there
    assert set(rep.backends) == set(sdpa.BACKENDS)
    assert all(isinstance(v, bool) for v in rep.backends.values())


def test_unavailable_backends_record_why():
    rep = sdpa.probe_backends(AttnShape(64, 4, 4), device="cpu", seq_len=16)
    for name, ok in rep.backends.items():
        assert (name in rep.errors) is (not ok)
    for reason in rep.errors.values():
        assert reason and "\n" not in reason  # one line, for one log line


def test_probe_without_the_torch_attention_api_says_so(monkeypatch):
    """A torch older than 2.3 has no torch.nn.attention. Report it; do not
    make it the load's problem."""
    monkeypatch.setattr(sdpa, "_backend_enums", lambda: None)
    rep = sdpa.probe_backends(QWEN38)
    assert rep.probed is False and "torch" in rep.note
    assert rep.has_efficient is False
    assert sdpa.report_lines(rep) == [f"SDPA: not probed — {rep.note}"]


def test_gqa_is_probed_the_way_transformers_dispatches():
    """Grouped kv + head_dim <= 256 with no mask is enable_gqa=True upstream,
    and on ROCm that choice changes which backends answer — so the probe has to
    make the same choice, and record the other one too."""
    grouped = sdpa.probe_backends(AttnShape(64, 8, 2), device="cpu", seq_len=16)
    assert grouped.gqa is True
    assert set(grouped.backends_repeat_kv or {}) == set(sdpa.BACKENDS)

    even = sdpa.probe_backends(AttnShape(64, 8, 8), device="cpu", seq_len=16)
    assert even.gqa is False and even.backends_repeat_kv is None

    # head_dim past 256 disqualifies enable_gqa upstream, mask or no mask
    wide = sdpa.probe_backends(AttnShape(512, 8, 2), device="cpu", seq_len=16)
    assert wide.gqa is False


def test_efficient_is_ordered_and_excludes_math():
    rep = fabricate(backends={"flash": True, "mem_efficient": True,
                              "cudnn": False, "math": True})
    assert rep.efficient == ["flash", "mem_efficient"]
    assert rep.has_efficient is True
    assert fabricate().has_efficient is False  # math alone is not efficient


def test_to_dict_is_json_shaped():
    import json

    rep = sdpa.probe_backends(AttnShape(64, 4, 4), device="cpu", seq_len=16)
    json.dumps(rep.to_dict())  # the module's __main__ contract


# -- the loud block ---------------------------------------------------------

def text(lines):
    return "\n".join(lines)


def test_warning_names_the_variable_and_the_arithmetic():
    body = text(sdpa.warning_lines(fabricate(), ctx=262144))
    assert EXPERIMENTAL_ENV in body
    assert "quadratically" in body
    assert "55.9 GiB" in body       # 25,000 tokens at 24 heads, fp32
    assert "[1, 24, T, T]" in body
    assert "advertises: 262,144 tokens" in body


def test_warning_states_where_this_box_runs_out():
    body = text(sdpa.warning_lines(fabricate(), ctx=262144))
    wall = sdpa.max_prompt_before(24, 133143986176)
    assert f"~{wall:,} tokens" in body and "124.0 GiB" in body


def test_warning_ladder_never_quotes_a_prompt_the_server_cannot_accept():
    body = text(sdpa.warning_lines(fabricate(), ctx=8192))
    assert "8,192 tokens" in body
    assert "25,000 tokens" not in body and "65,536 tokens" not in body


def test_warning_ladder_falls_back_to_the_ctx_itself_when_it_is_tiny():
    assert "4,096 tokens" in text(sdpa.warning_lines(fabricate(), ctx=4096))


def test_warning_blames_the_head_dim_cap_when_that_is_the_real_reason():
    """Flag on, still no efficient backend, head_dim past the gfx11xx cap: the
    user must not go hunting for a second env var. Nothing moves this."""
    rep = fabricate(shape=AttnShape(512, 24, 4), flag="1")
    body = text(sdpa.warning_lines(rep, ctx=8192))
    assert rep.gfx11_head_dim_capped
    assert "changed nothing" in body and "head_dim 256" in body
    assert "drinkme serve" not in body  # do not advise what is already done


def test_warning_does_not_blame_the_cap_when_the_head_dim_is_inside_it():
    rep = fabricate(flag="1")
    body = text(sdpa.warning_lines(rep, ctx=8192))
    assert not rep.gfx11_head_dim_capped
    assert "no efficient backend available" in body or "made no efficient" in body
    assert "caps the" not in body


def test_warning_tells_a_box_without_the_flag_how_to_set_it():
    body = text(sdpa.warning_lines(fabricate(), ctx=8192))
    assert f"{EXPERIMENTAL_ENV}=1 drinkme serve" in body
    assert "before the process" in body


def test_a_healthy_box_gets_one_quiet_line():
    rep = fabricate(backends={"flash": True, "mem_efficient": False,
                              "cudnn": False, "math": True})
    lines = sdpa.report_lines(rep, ctx=262144)
    assert lines == ["SDPA: flash available — attention is O(T) in memory."]


def test_the_quiet_line_credits_the_flag_when_the_flag_is_why():
    rep = fabricate(flag="1", backends={"flash": True, "mem_efficient": False,
                                        "cudnn": False, "math": True})
    assert EXPERIMENTAL_ENV in sdpa.report_lines(rep)[0]


# -- the self-test ----------------------------------------------------------

def test_selftest_seq_len_stays_inside_its_memory_budget():
    for heads in (4, 24, 64, 128):
        t = sdpa.selftest_seq_len(heads)
        assert sdpa.score_matrix_bytes(heads, t) <= sdpa.SELFTEST_BUDGET
        assert t <= sdpa.SELFTEST_MAX_T and t >= 128


def test_selftest_seq_len_shrinks_when_the_device_is_nearly_full():
    roomy = sdpa.selftest_seq_len(24, free_bytes=100 * GIB)
    cramped = sdpa.selftest_seq_len(24, free_bytes=64 * 1024**2)
    assert cramped < roomy and cramped >= 128


def test_selftest_agreement_is_measured_not_asserted():
    """Runs for real on CPU: flash vs math over identical inputs. Small shape,
    fp32, so this is a couple of milliseconds and no GPU."""
    st = sdpa.self_test(AttnShape(64, 4, 4), "flash", device="cpu", seq_len=128)
    if st is None:
        pytest.skip("no CPU flash backend in this torch")
    assert st.ok and st.max_abs <= st.tol_max and st.mean_abs <= st.tol_mean
    assert st.seq_len == 128 and st.backend == "flash"


def test_selftest_declines_comparisons_it_cannot_make():
    assert sdpa.self_test(QWEN38, "math", device="cpu") is None
    assert sdpa.self_test(QWEN38, "nonsense", device="cpu") is None


def test_selftest_tolerance_scales_with_the_reference_magnitude():
    st = sdpa.self_test(AttnShape(64, 4, 4), "flash", device="cpu", seq_len=128)
    if st is None:
        pytest.skip("no CPU flash backend in this torch")
    assert st.tol_max == pytest.approx(
        sdpa.MAX_ULPS * sdpa.BF16_EPS * max(1.0, st.ref_max))


def failed_test(**kw):
    base = dict(backend="flash", seq_len=1024, max_abs=1.9, mean_abs=0.4,
                ref_max=3.3, tol_max=0.104, tol_mean=sdpa.MEAN_TOL, ok=False)
    base.update(kw)
    return SelfTest(**base)


def test_a_passing_selftest_prints_the_numbers_it_measured():
    rep = fabricate(flag="1", backends={"flash": True, "mem_efficient": False,
                                        "cudnn": False, "math": True})
    st = SelfTest("flash", 1024, 0.015625, 9.86e-05, 3.33, 0.104,
                  sdpa.MEAN_TOL, True)
    body = text(sdpa.report_lines(rep, 262144, st))
    assert "AGREES" in body and "1.56e-02" in body and "9.86e-05" in body


def test_a_failing_selftest_is_loud_and_still_serves():
    rep = fabricate(flag="1", backends={"flash": True, "mem_efficient": False,
                                        "cudnn": False, "math": True})
    body = text(sdpa.report_lines(rep, 262144, failed_test()))
    assert "self-test failed" in body
    assert "Serving anyway" in body
    assert "Unset it" in body


# -- what the server actually calls -----------------------------------------

class FakeEngine:
    def __init__(self, cfg, ctx=8192):
        self.model = type("M", (), {"config": cfg})()
        self.ctx = ctx


def test_announce_probes_the_models_real_shape(monkeypatch):
    seen, said = {}, []
    monkeypatch.setattr(sdpa, "probe_backends",
                        lambda shape, device=None: seen.update(
                            shape=shape, device=device) or fabricate(shape=shape))
    eng = FakeEngine(Cfg(num_attention_heads=24, num_key_value_heads=4,
                         head_dim=256), ctx=262144)
    sdpa.announce_for_engine(eng, "cuda", emit=said.append)
    assert seen == {"shape": QWEN38, "device": "cuda"}
    assert any("No efficient attention backend" in s for s in said)
    assert any("advertises: 262,144 tokens" in s for s in said)


def test_announce_warns_for_gemma4s_full_layers_past_the_gfx11_cap(monkeypatch):
    """gfx1151 with the flag set: the 256-wide sliding layers get flash, the
    512-wide full layers only MATH. The warning prints for those layers,
    blames the head_dim cap, and names which layers they are."""
    probed, said = [], []

    def probe(shape, device=None):
        probed.append(shape)
        ok = shape.head_dim <= sdpa.GFX11_HEAD_DIM_CAP
        return fabricate(shape=shape, flag="1", backends={
            "flash": ok, "mem_efficient": False, "cudnn": False, "math": True})

    monkeypatch.setattr(sdpa, "probe_backends", probe)
    monkeypatch.setattr(sdpa, "self_test", lambda *a, **k: None)
    eng = FakeEngine({"text_config": GEMMA4_TEXT}, ctx=8192)
    reps = sdpa.announce_for_engine(eng, "cuda", emit=said.append)
    assert probed == [s for s, _ in GEMMA4_SHAPES]
    assert [r.layers for r in reps] == [lbl for _, lbl in GEMMA4_SHAPES]
    body = text(said)
    assert said[0].startswith("SDPA: flash available at head_dim 256, 32 q heads / 16 kv, "
                              "on 50 of 60 layers (sliding_attention)")
    assert "No efficient attention backend" in body
    assert "on 10 of 60 layers (full_attention)" in body
    assert "head_dim 256 and these layers are 512" in body
    assert "[1, 32, T, T]" in body


def test_announce_is_silent_when_there_is_no_config_to_read(monkeypatch):
    """FakeEngine and the serve wiring tests hand build_engine an object with
    no model. Nothing honest can be said about attention, so say nothing."""
    said = []
    monkeypatch.setattr(sdpa, "probe_backends",
                        lambda *a, **k: pytest.fail("must not probe"))
    assert sdpa.announce_for_engine(object(), "cuda", emit=said.append) is None
    assert said == []


def test_selftest_runs_only_when_the_flag_is_set(monkeypatch):
    calls = []
    monkeypatch.setattr(sdpa, "self_test",
                        lambda *a, **k: calls.append(a) or None)
    efficient = {"flash": True, "mem_efficient": False, "cudnn": False,
                 "math": True}

    sdpa.announce(fabricate(backends=efficient), emit=lambda _l: None)
    assert calls == []  # nothing was opted into, so there is nothing to verify

    sdpa.announce(fabricate(flag="1", backends=efficient), emit=lambda _l: None)
    assert len(calls) == 1 and calls[0][1] == "flash"

    calls.clear()
    sdpa.announce(fabricate(flag="1"), emit=lambda _l: None)
    assert calls == []  # no efficient backend: nothing to compare against math


def test_serve_never_lets_the_probe_kill_a_load(monkeypatch, capsys):
    """A diagnostic that can take down a server start is worse than none."""
    from drinkme import serve

    def boom(*a, **k):
        raise RuntimeError("hip blew up")

    monkeypatch.setattr(sdpa, "announce_for_engine", boom)
    serve._announce_attention(object(), "cuda")  # must not raise
    assert "SDPA probe failed" in capsys.readouterr().err
