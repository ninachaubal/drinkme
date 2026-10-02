"""bench/ngram_gpu_ab.py's command line and its per-cell summary — CPU only.

The tool itself is a GPU instrument; what is pinned here is the part that
decides WHAT it runs and HOW a cell reads: `--prompts` picks and orders the
prompts, `--reps` repeats every prompt x arm on one engine build, and a cell
keeps the single-run receipt's fields (so older readers still parse it) with
the median, the spread and every run's raw numbers beside them. One end-to-end
run drives main() over the MTP tests' toy engine to hold the loop's shape.
"""

import importlib.util
import json
import os

import pytest

from drinkme.serving import mtp

from test_serving_mtp import _torch_chunk_on_cpu, toy  # noqa: F401 — fixtures by name

HERE = os.path.dirname(os.path.abspath(__file__))
AB = os.path.join(HERE, "..", "bench", "ngram_gpu_ab.py")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def ab():
    spec = importlib.util.spec_from_file_location("ngram_gpu_ab_under_test", AB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_importing_the_tool_leaves_the_environment_alone():
    """The arm toggle writes DRINKME_SPEC; that belongs to main(), not to
    import, or a test that imports the tool would leak it into the suite."""
    keys = ("DRINKME_SPEC", "DRINKME_PREFIX_SLOTS", "DRINKME_NGRAM_MIN")
    before = {k: os.environ.get(k) for k in keys}
    spec = importlib.util.spec_from_file_location("ngram_gpu_ab_import_only", AB)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert {k: os.environ.get(k) for k in keys} == before


def test_defaults_are_the_old_single_run_over_every_prompt(ab):
    cfg = ab.parse_args([])
    assert cfg["reps"] == 1
    assert cfg["prompts"] == list(ab.PROMPTS)
    assert cfg["arms"] == ["off", "ngram"]
    assert cfg["n_tokens"] == 256 and cfg["stock"] is False


def test_prompts_and_reps_parse_in_the_order_given(ab):
    cfg = ab.parse_args(["--prompts", "code,chat", "--reps", "3", "--stock",
                         "--arms", "off,mtp", "--tokens", "32"])
    assert cfg["prompts"] == ["code", "chat"]
    assert cfg["reps"] == 3
    assert cfg["stock"] is True
    assert cfg["arms"] == ["off", "mtp"]
    assert cfg["n_tokens"] == 32


@pytest.mark.parametrize("argv, msg", [
    (["--prompts", "chat,poetry"], "unknown prompt"),
    (["--prompts", ""], "unknown prompt"),
    (["--prompts", "chat,chat"], "twice"),
    (["--reps", "0"], "at least 1"),
    (["--reps", "two"], "whole number"),
    (["--reps"], "needs a value"),
    (["--rep", "3"], "unknown arguments"),
])
def test_bad_arguments_stop_before_any_engine_is_built(ab, argv, msg):
    with pytest.raises(SystemExit, match=msg):
        ab.parse_args(argv)


def _run(tps, text="t", mode="mtp"):
    return {"text": text, "finish": "length", "tokens": 8, "wall_s": 1.0,
            "ttft_s": 0.1, "decode_toks_per_s": tps,
            "spec": {"mode": mode, "cycles": int(tps * 10)}, "audit": None}


def test_a_cell_is_the_median_with_every_run_kept(ab):
    runs = [_run(10.0), _run(12.0), _run(11.0)]
    cell = ab.summarize_cell(runs)
    assert cell["decode_toks_per_s"] == 11.0
    # the other fields belong to ONE real run: the median one
    assert cell["spec"]["cycles"] == 110
    assert cell["decode_toks_per_s_runs"] == [10.0, 12.0, 11.0]
    assert cell["spread"] == {"min": 10.0, "max": 12.0, "rel_range": round(2 / 11, 4)}
    assert [r["decode_toks_per_s"] for r in cell["runs"]] == [10.0, 12.0, 11.0]
    assert all("text" not in r for r in cell["runs"]) and "text" not in cell
    # the single-run receipt's fields are all still there
    for k in ("finish", "tokens", "wall_s", "ttft_s", "decode_toks_per_s", "spec", "audit"):
        assert k in cell


def test_an_even_number_of_reps_takes_the_mean_of_the_middle_two(ab):
    cell = ab.summarize_cell([_run(10.0), _run(13.0), _run(11.0), _run(12.0)])
    assert cell["decode_toks_per_s"] == 11.5
    assert cell["spec"]["cycles"] == 110  # the lower-median run


def test_main_on_the_toy_repeats_each_prompt_and_arm_on_one_build(ab, toy, monkeypatch, tmp_path):
    """--reps 2 --prompts repetitive,chat over the toy: one build, both
    prompts in the order given, two runs per arm, identity per rep, and the
    old fields in every cell."""
    import drinkme.serve as serve_mod
    from drinkme.serving.engines import HFEngine

    model, tok, head, _ = toy
    builds = []

    def fake_build(model_id, revision, pack, stock=False, ctx=0):
        builds.append((model_id, pack, stock))
        os.environ["DRINKME_PREFIX_SLOTS"] = "0"
        try:
            return HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=512,
                            mtp_head=head)
        finally:
            del os.environ["DRINKME_PREFIX_SLOTS"]

    monkeypatch.setattr(serve_mod, "build_engine", fake_build)
    # main() writes these and wraps Speculator.stats; put them all back after
    for k in ("DRINKME_SPEC", "DRINKME_PREFIX_SLOTS", "DRINKME_MTP_DEPTH"):
        monkeypatch.setenv(k, os.environ.get(k, ""))
        if not os.environ[k]:
            monkeypatch.delenv(k)
    monkeypatch.setenv("DRINKME_MTP_DEPTH", "3")
    monkeypatch.setattr(mtp.Speculator, "stats", mtp.Speculator.stats)
    out = tmp_path / "ab.json"
    code = ab.main(["--prompts", "repetitive,chat", "--reps", "2", "--arms", "off,mtp",
                    "--tokens", "6", "--json", str(out)])
    assert code == 0
    assert len(builds) == 1
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "PASS"
    # one untimed request per arm, on the first prompt, before the timed reps
    assert rep["warmup"]["requests_per_arm"] == 1 and rep["warmup"]["prompt"] == "repetitive"
    assert rep["warmup"]["timed"] is False and list(rep["warmup"]["decode_toks_per_s"]) == ["off", "mtp"]
    assert rep["reps"] == 2 and rep["prompt_names"] == ["repetitive", "chat"]
    assert list(rep["prompts"]) == ["repetitive", "chat"]
    for name in ("repetitive", "chat"):
        cell_off, cell_on = rep["prompts"][name]["off"], rep["prompts"][name]["mtp"]
        for c in (cell_off, cell_on):
            assert len(c["runs"]) == 2 and [r["rep"] for r in c["runs"]] == [1, 2]
            assert len(c["decode_toks_per_s_runs"]) == 2
        assert all(r["spec"]["mode"] == "mtp" for r in cell_on["runs"])
        assert all(r["identity"] in ("IDENTICAL",) or r["identity"]["verdict"] == "FORK"
                   for r in cell_on["runs"])
        assert "identity" in cell_on and "speedup" in cell_on


def test_the_warm_up_is_one_untimed_request_per_arm_before_any_rep(ab):
    """Each arm's first request pays its compiles (17.6 against 143 tok/s on
    the 0.6B): warm_up runs it once per arm, skips an MTP arm with no head,
    and leaves the arm set for the next request to the timed loop."""
    class Engine:
        _spec_plan = _mtp_depth = None

    seen = []

    def fake_run(engine, prompt, n):
        seen.append((os.environ["DRINKME_SPEC"], prompt == ab.PROMPTS["chat"], n))
        return {"decode_toks_per_s": 17.6}

    old = os.environ.get("DRINKME_SPEC")
    try:
        w = ab.warm_up(Engine(), ["off", "mtp", "ngram", "ngram+mtp"], "chat", 8, has_head=False, run=fake_run)
    finally:
        if old is None:
            os.environ.pop("DRINKME_SPEC", None)
        else:
            os.environ["DRINKME_SPEC"] = old
    assert seen == [("off", True, 8), ("ngram", True, 8)]
    assert w == {"requests_per_arm": 1, "prompt": "chat", "n_tokens": 8, "timed": False,
                 "decode_toks_per_s": {"off": 17.6, "ngram": 17.6}}
