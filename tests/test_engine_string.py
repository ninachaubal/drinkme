"""`environment.engine` names the kernels.

One grammar, slashes between segments, each segment starting with its
noun, greppable by segment; every kernel segment is `<noun> <impl>[ <version>]`:

    drinkme <semver> / torch <version>[+<accel tag>] / deltanet <fla <ver> | torch | none> / conv <fla <ver> | torch | none> / narrow-gemv <triton | torch> / stock-gemv <mv | linear | triton> / raw-gemv <twin | stock>
    drinkme <semver> / mlx <version>

The kernel segments are composed from the routing record — every arm
routed alike or no record exists, so there is ONE answer per record — and
parse back to it (bench.parse_engine). `raw.<arm>_routing` stays as the
per-arm detail. The lexicon and docs/bench.md carry the grammar; the
lexicon's `raw` description names the routing among raw's contents.
CPU only: the routing record comes from route_kernels on the toy hybrid
(deltanet torch) and on a dense tree (deltanet none).
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from drinkme import __version__, arms, bench
from drinkme.publish import validate as V
from drinkme.serving import deltanet, deltanet_conv
from drinkme.serving.kernel_route import route_kernels
from test_serving_mtp import _cfg

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAMMAR_TORCH = ("drinkme <semver> / torch <version>[+<accel tag>] / deltanet "
                 "<fla <ver> | torch | none> / conv <fla <ver> | torch | none> "
                 "/ narrow-gemv <triton | torch> / stock-gemv <mv | linear | triton> / raw-gemv <twin | stock>")
GRAMMAR_MLX = "drinkme <semver> / mlx <version>"
# the routing record's fields the torch string carries
ROUTED = ("deltanet_kernel", "deltanet_conv", "narrow_gemv", "stock_gemv", "raw_gemv")


@pytest.fixture
def toy_routing(monkeypatch):
    """route_kernels on the hybrid toy on the CPU: the torch reference, the
    narrow GEMV allowed, nothing adopted — the recurrence global put back
    afterwards (it is process-wide)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    monkeypatch.delenv(deltanet.ENV, raising=False)
    monkeypatch.delenv(deltanet_conv.ENV, raising=False)
    monkeypatch.delenv("DRINKME_NARROW_GEMV", raising=False)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_cfg()).eval()
    mods = deltanet.deltanet_modules(model)
    modeling = deltanet._modeling_module(mods[0])
    before = getattr(modeling, deltanet.NAME)
    before_conv = getattr(modeling, deltanet_conv.NAME)
    try:
        yield route_kernels(model, "cpu")
    finally:
        setattr(modeling, deltanet.NAME, before)
        setattr(modeling, deltanet_conv.NAME, before_conv)


# ------------------------------------------------------------- the grammar --


def test_the_string_composes_from_the_routing_record_and_parses_back(toy_routing):
    # conv torch: the toy hybrid on CPU (deltanet_conv.route falls back off-CUDA)
    assert toy_routing == {"deltanet_kernel": "torch", "deltanet_conv": "torch", "narrow_gemv": True, "narrow_count": 0,
                           "stock_gemv": "linear", "raw_gemv": "stock"}
    s = bench.engine_string("torch", "2.12.0+rocm7.1", toy_routing)
    assert s == (f"drinkme {__version__} / torch 2.12.0+rocm7.1 / deltanet torch "
                "/ conv torch / narrow-gemv triton / stock-gemv linear / raw-gemv stock")
    p = bench.parse_engine(s)
    assert p["drinkme"] == __version__
    assert p["runtime"] == "torch" and p["runtime_version"] == "2.12.0+rocm7.1"
    assert {k: p[k] for k in ROUTED} == {k: toy_routing[k] for k in ROUTED}

    # conv none: a dense tree has no DeltaNet layers, and both segments say none
    dense = route_kernels(torch.nn.Linear(4, 4), "cpu")
    assert dense["deltanet_conv"] == "none"
    s = bench.engine_string("torch", "2.12.0+cu128", dense)
    assert s == (f"drinkme {__version__} / torch 2.12.0+cu128 / deltanet none / conv none / narrow-gemv triton"
                 " / stock-gemv linear / raw-gemv stock")
    p = bench.parse_engine(s)
    assert p["deltanet_kernel"] == "none" and p["deltanet_conv"] == "none"

    # conv fla: fla carries its version in every segment it runs; the GEMV knob off is torch's
    # F.linear; gfx1151's stock GEMV is torch.mv and its raw GEMV the twin's kernel
    fla = {"deltanet_kernel": "fla", "deltanet_conv": "fla", "narrow_gemv": False, "narrow_count": 0,
           "stock_gemv": "mv", "raw_gemv": "twin"}
    s = bench.engine_string("torch", "2.12.0a0+rocm7.13.0a20260411", fla, fla_version="0.5.2")
    assert s == (f"drinkme {__version__} / torch 2.12.0a0+rocm7.13.0a20260411 / "
                 "deltanet fla 0.5.2 / conv fla 0.5.2 / narrow-gemv torch / stock-gemv mv / raw-gemv twin")
    p = bench.parse_engine(s)
    assert p["deltanet_kernel"] == "fla" and p["fla_version"] == "0.5.2"
    assert p["deltanet_conv"] == "fla"
    assert p["narrow_gemv"] is False and p["stock_gemv"] == "mv" and p["raw_gemv"] == "twin"

    # conv fla beside the torch recurrence: the conv segment alone carries fla's version
    mixed = {"deltanet_kernel": "torch", "deltanet_conv": "fla", "narrow_gemv": True, "narrow_count": 2,
             "stock_gemv": "linear", "raw_gemv": "stock"}
    s = bench.engine_string("torch", "2.12.0+rocm7.1", mixed, fla_version="0.5.2")
    assert s == (f"drinkme {__version__} / torch 2.12.0+rocm7.1 / "
                 "deltanet torch / conv fla 0.5.2 / narrow-gemv triton / stock-gemv linear / raw-gemv stock")
    p = bench.parse_engine(s)
    assert (p["deltanet_kernel"], p["deltanet_conv"], p["fla_version"]) == ("torch", "fla", "0.5.2")


def test_the_engine_string_refuses_a_routing_dict_with_no_deltanet_conv():
    """A routing dict route_kernels always fills in; a record must not
    silently say `conv torch` for a run that never routed the convolution."""
    with pytest.raises(ValueError, match="deltanet_conv"):
        bench.engine_string("torch", "2.12.0", {"deltanet_kernel": "torch",
                                                "narrow_gemv": True, "narrow_count": 0, "stock_gemv": "linear",
                                                "raw_gemv": "stock"})


def test_the_engine_string_names_the_stock_gemv_or_refuses():
    """The stock GEMV is the sixth segment, read off route_kernels'
    stock_gemv, whose values are swap.STOCK_GEMV_MODES (triton among them:
    gfx1151's stock GEMV, the twin's kernel); a routing dict without it,
    or with another value, writes no string."""
    from drinkme.codec.swap import STOCK_GEMV_MODES

    assert set(bench.STOCK_GEMV_IMPLS) == set(STOCK_GEMV_MODES) == {"linear", "mv", "triton"}
    base = {"deltanet_kernel": "none", "deltanet_conv": "none", "narrow_gemv": True, "narrow_count": 0,
            "raw_gemv": "stock"}
    for mode in STOCK_GEMV_MODES:
        s = bench.engine_string("torch", "2.12.0", dict(base, stock_gemv=mode))
        assert s.endswith(f" / narrow-gemv triton / stock-gemv {mode} / raw-gemv stock")
        assert bench.parse_engine(s)["stock_gemv"] == mode
    with pytest.raises(ValueError, match="stock_gemv"):
        bench.engine_string("torch", "2.12.0", base)
    with pytest.raises(ValueError, match="stock_gemv"):
        bench.engine_string("torch", "2.12.0", dict(base, stock_gemv="tunableop"))
    p = bench.parse_engine("drinkme 1.0.0 / torch 2.12.0+rocm7.13 / deltanet fla 0.5.2 / conv fla 0.5.2"
                           " / narrow-gemv triton / stock-gemv triton / raw-gemv twin / decode eager")
    assert p["stock_gemv"] == "triton" and p["raw_gemv"] == "twin" and p["narrow_gemv"] is True


def test_the_engine_string_names_the_raw_gemv_or_refuses():
    """A codec tree's raw Linears' one-row call is the last segment, read
    off route_kernels' raw_gemv, whose values are swap.RAW_GEMV_MODES; a
    routing dict without it, or with another value, writes no string."""
    from drinkme.codec.swap import RAW_GEMV_MODES

    assert set(bench.RAW_GEMV_IMPLS) == set(RAW_GEMV_MODES)
    base = {"deltanet_kernel": "none", "deltanet_conv": "none", "narrow_gemv": True, "narrow_count": 0,
            "stock_gemv": "mv"}
    for mode in RAW_GEMV_MODES:
        s = bench.engine_string("torch", "2.12.0", dict(base, raw_gemv=mode))
        assert s.endswith(f" / stock-gemv mv / raw-gemv {mode}")
        assert bench.parse_engine(s)["raw_gemv"] == mode
    with pytest.raises(ValueError, match="raw_gemv"):
        bench.engine_string("torch", "2.12.0", base)
    with pytest.raises(ValueError, match="raw_gemv"):
        bench.engine_string("torch", "2.12.0", dict(base, raw_gemv="narrow"))


def test_the_mlx_form_and_the_segments_are_greppable():
    s = bench.engine_string("mlx", "0.32.2")
    assert s == f"drinkme {__version__} / mlx 0.32.2"
    p = bench.parse_engine(s)
    assert p == {"drinkme": __version__, "runtime": "mlx", "runtime_version": "0.32.2"}
    # every segment starts with its noun: a grep for the noun finds the segment
    for s in (bench.engine_string("mlx", "0.32.2"),
              bench.engine_string("torch", "2.12.0", {"deltanet_kernel": "torch", "deltanet_conv": "torch",
                                                      "narrow_gemv": True, "narrow_count": 3,
                                                      "stock_gemv": "mv", "raw_gemv": "twin"})):
        segments = s.split(" / ")
        assert segments[0].startswith("drinkme ")
        assert all(seg.split(" ")[0] for seg in segments)
        assert " / " not in "".join(seg for seg in segments)  # the separator is the grammar's own


def test_anything_outside_the_grammar_does_not_parse():
    for s in ("drinkme 1.0.0 / torch 2.12.0+rocm7.1",  # the two-segment string, torch
              # five and six segments: every torch string names stock-gemv and raw-gemv
              "drinkme 1.0.0 / torch 2.12.0+rocm7.1 / deltanet fla 0.5.2 / conv fla 0.5.2 / narrow-gemv triton",
              "drinkme 1.0.0 / torch 2.12.0+rocm7.1 / deltanet fla 0.5.2 / conv fla 0.5.2 / narrow-gemv triton"
              " / stock-gemv mv",
              # the four-segment torch form (no conv route named)
              "drinkme 1.0.0 / torch 2.12.0 / deltanet fla 0.5.2 fused / narrow-gemv on",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet fla 0.5.2 / narrow-gemv triton",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet fla / conv fla 0.5.2 / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",  # fla without a version
              "drinkme 1.0.0 / torch 2.12.0 / deltanet fla 0.5.2 / conv fla 0.5.3 / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",  # two flas
              "drinkme 1.0.0 / torch 2.12.0 / deltanet cuda / conv fla 0.5.2 / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv cuda / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",  # conv not fla|torch|none
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv on-ish"
              " / stock-gemv mv / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv none"
              " / stock-gemv mv / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet torch fused / conv torch / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",  # no adjectives
              "drinkme 1.0.0 / mlx 0.32.2 / metal / extra",
              "torch 2.12.0 / drinkme 1.0.0 / deltanet none / conv none / narrow-gemv triton",
              # a deltanet or conv segment with an adjective, or an fla without its version
              "drinkme 1.0.0 / torch 2.12.0 / deltanet torch reference / conv torch / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet fla 0.5.2 / conv fla / narrow-gemv on"
              " / stock-gemv mv / raw-gemv twin",
              # deltanet/conv disagree on whether the model has DeltaNet layers
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv fla 0.5.2 / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv torch / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet torch / conv none / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet fla 0.5.2 / conv none / narrow-gemv triton"
              " / stock-gemv mv / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv fla / narrow-gemv on"
              " / stock-gemv mv / raw-gemv twin",
              # the stock-gemv segment: only after narrow-gemv, only mv | linear | triton
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv triton"
              " / stock-gemv hipblaslt / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none"
              " / stock-gemv mv / narrow-gemv triton / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv triton"
              " / stock-gemv mv / x y",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv on"
              " / stock-gemv mv / raw-gemv twin",
              # the raw-gemv segment: only after stock-gemv, only twin | stock
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv triton / stock-gemv mv"
              " / raw-gemv triton",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv triton / raw-gemv twin",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv triton / raw-gemv twin"
              " / stock-gemv mv",
              "drinkme 1.0.0 / torch 2.12.0 / deltanet none / conv none / narrow-gemv triton / stock-gemv mv"
              " / raw-gemv twin / x y",
              ""):
        with pytest.raises(ValueError):
            bench.parse_engine(s)
    with pytest.raises(ValueError):
        bench.engine_string("torch", "2.12.0", {"deltanet_kernel": "cuda", "deltanet_conv": "torch",
                                                "narrow_gemv": True, "narrow_count": 0, "stock_gemv": "linear",
                                                "raw_gemv": "stock"})
    with pytest.raises(ValueError):
        bench.engine_string("torch", "2.12.0", {"deltanet_kernel": "torch", "deltanet_conv": "cuda",
                                                "narrow_gemv": True, "narrow_count": 0, "stock_gemv": "linear",
                                                "raw_gemv": "stock"})


# ------------------------------------------------------------- the writer --


def test_the_string_the_bench_writes_parses_back_to_the_routing_record(toy_routing, monkeypatch):
    """bench.run through the real _run_point, the arms stubbed with the
    toy's own routing on every measured arm: environment.engine parses
    back to that record, and raw.<arm>_routing is untouched beside it."""
    from test_bench_record_shape import _build_record, _fake_raw

    raw = _fake_raw()
    for arm in ("stock", "twin", "compressed"):
        raw[f"{arm}_routing"] = dict(toy_routing)
    monkeypatch.setattr("test_bench_record_shape._fake_raw", lambda: raw)
    r = _build_record(monkeypatch)
    p = bench.parse_engine(r["environment"]["engine"])
    assert {k: p[k] for k in ROUTED} == {k: toy_routing[k] for k in ROUTED}
    assert p["runtime"] == "torch" and p["runtime_version"] == torch.__version__
    assert r["environment"]["platform"] in ("cuda", "rocm")
    for arm in ("stock", "twin", "compressed"):
        assert r["raw"][f"{arm}_routing"] == toy_routing
    assert V.validate(bench.lexicon_safe(r)) == []


def test_a_fit_point_reads_the_route_off_the_one_arm_that_ran(monkeypatch):
    from test_bench_record_shape import _build_record, _fake_raw

    raw = _fake_raw()
    for arm in ("stock", "twin"):  # a fit point: only the compressed arm ran and routed
        raw.pop(f"{arm}_routing")
    raw["compressed_routing"] = {"deltanet_kernel": "none", "deltanet_conv": "none",
                                 "narrow_gemv": False, "narrow_count": 0, "stock_gemv": "mv", "raw_gemv": "twin"}
    monkeypatch.setattr("test_bench_record_shape._fake_raw", lambda: raw)
    r = _build_record(monkeypatch)
    p = bench.parse_engine(r["environment"]["engine"])
    assert p["deltanet_kernel"] == "none" and p["deltanet_conv"] == "none" and p["narrow_gemv"] is False
    assert p["stock_gemv"] == "mv" and p["raw_gemv"] == "twin"


def test_arms_that_report_no_route_or_two_routes_make_no_record(monkeypatch):
    """The arms refuse a record whose arms routed differently before bench
    sees it (arms.refuse_unless_routed_alike); bench holds the same line
    at its own boundary, and a torch report with no route at all cannot
    name its kernels."""
    from test_bench_record_shape import _build_record, _fake_raw

    raw = _fake_raw()
    for arm in ("stock", "twin", "compressed"):
        raw.pop(f"{arm}_routing", None)
    monkeypatch.setattr("test_bench_record_shape._fake_raw", lambda: raw)
    with pytest.raises(ValueError, match="routing"):
        _build_record(monkeypatch)
    raw = _fake_raw()
    raw["stock_routing"] = {"deltanet_kernel": "torch", "narrow_gemv": True, "narrow_count": 96}
    raw["twin_routing"] = raw["compressed_routing"] = {"deltanet_kernel": "fla", "narrow_gemv": True,
                                                       "narrow_count": 96}
    monkeypatch.setattr("test_bench_record_shape._fake_raw", lambda: raw)
    with pytest.raises(ValueError, match="alike"):
        _build_record(monkeypatch)


def test_the_mlx_record_writes_the_metal_form(monkeypatch):
    from test_bench_record_shape import _build_record_mlx

    r = _build_record_mlx(monkeypatch)
    assert r["environment"]["engine"] == f"drinkme {__version__} / mlx 0.32.2"
    assert bench.parse_engine(r["environment"]["engine"])["runtime"] == "mlx"
    assert r["environment"]["platform"] == "metal"


# --------------------------------------------------------- the public words --


def test_the_lexicon_carries_the_grammar():
    lex = json.loads((ROOT / "lexicons" / "wtf.petrichor.drinkme.measurement.json").read_text())
    engine = lex["defs"]["environment"]["properties"]["engine"]["description"]
    assert GRAMMAR_TORCH in engine and GRAMMAR_MLX in engine, engine
    raw = lex["defs"]["main"]["record"]["properties"]["raw"]["description"]
    assert "routing" in raw, raw


def test_raw_routing_is_untouched_per_arm():
    """The per-arm detail stays where it was: run_arms still assigns
    <arm>_routing for each arm, and bench does not rewrite it."""
    import inspect

    src = inspect.getsource(arms.run_arms)
    for arm in ("stock", "twin", "compressed"):
        assert f'report["{arm}_routing"]' in src, arm
