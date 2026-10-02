"""`drinkme bench`'s speculation pass (src/drinkme/spec_pass.py, arms.
time_speculation): decode timed with the checkpoint's MTP head proposing, as
`drinkme serve --spec mtp` runs it, recorded as six metrics beside the plain
decode.

What is pinned here, on the CPU:

  * the prompts are one copy, shared with bench/ngram_gpu_ab.py;
  * the pass over the MTP tests' toy (a qwen3_5 hybrid whose head loads from
    a checkpoint through the loader serve uses): serve's engine, the head
    the decision serve makes loads, one untimed warm-up per prompt
    discarded, the timed reps' samples, accepted drafts per verify step;
  * each arm asks for the head serve would load (the pack's under
    --pack-dir, packed in memory for the re-packed compressed arm);
  * run_arms runs the pass after each measured stock and compressed arm's
    plain timings, never on the twin, and --no-spec skips it;
  * the record: the six names and their presence rules (no head, mlx, a
    stock arm that was not measured), raw.spec's shape both ways, the
    summary lines.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest
import torch

from drinkme import arms, bench, spec_pass
from drinkme.codec import swap
from drinkme.serving import engines, mtp

from test_serving_mtp import _torch_chunk_on_cpu, _write_head_ckpt, toy  # noqa: F401 — fixtures by name

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES = ("stock_spec_decode_tok_s_agent", "stock_spec_decode_tok_s_chat", "stock_spec_decode_tok_s_code",
         "compressed_spec_decode_tok_s_agent", "compressed_spec_decode_tok_s_chat",
         "compressed_spec_decode_tok_s_code")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ------------------------------------------------------------ the prompts --


def test_the_prompts_are_one_copy_shared_with_ngram_gpu_ab():
    path = ROOT / "bench" / "ngram_gpu_ab.py"
    spec = importlib.util.spec_from_file_location("ngram_gpu_ab_spec_prompts", path)
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    assert ab.PROMPTS is spec_pass.PROMPTS
    src = path.read_text()
    for text in spec_pass.PROMPTS.values():
        assert text[:40] not in src  # no second copy of any prompt's text
    # bench times three of the four, under the record's names; repetitive is ngram_gpu_ab's alone
    assert spec_pass.BENCH_PROMPTS == {"agent": "agent-transcript", "chat": "chat", "code": "code"}
    assert [spec_pass.metric_name(a, p) for a in spec_pass.ARMS for p in spec_pass.BENCH_PROMPTS] \
        == list(NAMES)


# ----------------------------------------------- the pass, on the toy head --


# The toy's native window is 512 tokens and its vocabulary twenty words: the
# real agent transcript alone is longer than the window, so the toy times
# three short prompts over a short budget. The pass is the same code.
SHORT = {"agent-transcript": "alpha beta gamma alpha beta gamma alpha beta gamma",
         "chat": "hello world the quick brown fox",
         "code": "delta epsilon zeta over the lazy dog"}
TOY_TOKENS = 12


@pytest.fixture
def short_prompts(monkeypatch):
    monkeypatch.setattr(spec_pass, "PROMPTS", {**spec_pass.PROMPTS, **SHORT})
    monkeypatch.setattr(spec_pass, "TOKENS", TOY_TOKENS)
    monkeypatch.delenv("DRINKME_MTP_DEPTH", raising=False)
    mtp._warned.clear()


def test_the_pass_times_the_heads_decode_through_serves_engine(toy, tmp_path, monkeypatch, short_prompts):
    """The toy's head streams from a checkpoint through engines._mtp_head,
    the engine is serve's HFEngine over the arm's tree, and every request
    it serves is the head proposing (arms._spec_rep refuses anything
    else). One untimed request per prompt comes first and is discarded;
    the timed reps are the samples; accepted_per_step pools the timed reps'
    accepted drafts over their verify steps. An operator's DRINKME_SPEC=off
    does not stop the pass (it measures `--spec mtp`), and is handed back."""
    model, tok, head, _ = toy
    _write_head_ckpt(tmp_path, head)
    monkeypatch.setenv("DRINKME_SPEC", "off")
    monkeypatch.setattr(arms, "WARMUP_REP", True)
    reps = []
    real = arms._spec_rep
    monkeypatch.setattr(arms, "_spec_rep", lambda *a: reps.append(real(*a)) or reps[-1])
    r = arms.time_speculation(model, "stock", "toy", tok, str(tmp_path), device="cpu")
    assert os.environ["DRINKME_SPEC"] == "off"
    assert r["ctx"] == 512  # the engine's own default context: the toy's native window
    assert list(r["prompts"]) == list(r["timed"]) == ["agent", "chat", "code"]
    per = 1 + arms.DECODE_REPS
    assert len(reps) == 3 * per
    for i, (name, key) in enumerate(spec_pass.BENCH_PROMPTS.items()):
        warm, *timed = reps[i * per:(i + 1) * per]
        cell = r["timed"][name]
        assert cell["samples"] == [x["tok_s"] for x in timed]  # the warm-up is not a sample
        assert all(x > 0 for x in cell["samples"])
        assert all(x["tokens"] == TOY_TOKENS for x in [warm, *timed])
        cycles = sum(x["cycles"] for x in timed)
        assert cycles > 0
        assert cell["accepted_per_step"] == round(sum(x["accepted"] for x in timed) / cycles, 3)
        assert 0 <= cell["accepted_per_step"] <= mtp.DEFAULT_DEPTH
        assert r["prompts"][name] == {"sha256": spec_pass.sha256(SHORT[key]),
                                      "tokens": timed[0]["prompt_tokens"]}
    assert set(r) == {"ctx", "prompts", "timed"}


def test_without_a_head_the_pass_is_skipped_before_any_engine_is_built(toy, tmp_path, monkeypatch,
                                                                       short_prompts):
    """A checkpoint that carries no mtp.* tensors (MiMo's config claims a
    head and ships none): the loader serve uses finds none, and the pass
    stops there."""
    from safetensors.torch import save_file

    save_file({"model.embed_tokens.weight": torch.zeros(4, 4)}, str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(engines, "HFEngine", lambda *a, **k: pytest.fail("no head, no engine"))
    monkeypatch.delenv("DRINKME_SPEC", raising=False)
    model, tok, _, _ = toy
    for arm in spec_pass.ARMS:
        assert arms.time_speculation(model, arm, "toy", tok, str(tmp_path), device="cpu") \
            == {"skipped": "no MTP head"}
    assert "DRINKME_SPEC" not in os.environ


class _Stop(Exception):
    pass


def test_each_arm_asks_for_the_head_serve_would_load(toy, monkeypatch):
    """The stock arm: the raw head (serve --stock). The compressed arm off a
    pack: the pack's (the head diet). The in-memory compressed arm: the raw
    head packed in memory, as the pack would carry it. Each under
    DRINKME_SPEC=mtp, which is what serve's --spec mtp writes."""
    model = toy[0]
    calls, diets = [], []

    def head(m, repo, revision, device, pack_dir=None):
        calls.append((repo, revision, device, pack_dir, os.environ.get("DRINKME_SPEC")))
        return toy[2]

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(engines, "_mtp_head", head)
    monkeypatch.setattr(arms, "_diet_in_memory", lambda h, device: diets.append((h, device)))
    monkeypatch.setattr(engines, "HFEngine", stop)
    for arm, pack in (("stock", None), ("compressed", "/packs/p"), ("compressed", None)):
        with pytest.raises(_Stop):
            arms.time_speculation(model, arm, "toy", None, "/snap", pack_dir=pack, device="cpu")
    assert calls == [("/snap", None, "cpu", None, "mtp"), ("/snap", None, "cpu", "/packs/p", "mtp"),
                     ("/snap", None, "cpu", None, "mtp")]
    assert diets == [(toy[2], "cpu")]  # the in-memory compressed arm's alone


def _bf16_head():
    torch.manual_seed(0)
    head = torch.nn.Module()
    head.fc = torch.nn.Linear(1024, 1024, bias=False).to(torch.bfloat16)  # eligible: >= 1024 each way
    head.small = torch.nn.Linear(64, 64, bias=False).to(torch.bfloat16)   # under the bar: stays raw
    return head


def test_the_in_memory_head_diet_packs_what_the_pack_would(monkeypatch):
    monkeypatch.delenv("DRINKME_MTP_DIET", raising=False)
    head = _bf16_head()
    assert arms._diet_in_memory(head, "cpu") == 1
    assert isinstance(head.fc, swap.SWAPPED_LINEAR_TYPES)
    assert isinstance(head.small, torch.nn.Linear) and not isinstance(head.small, swap.SWAPPED_LINEAR_TYPES)
    # DRINKME_MTP_DIET=0 is serve's opt-out, and the pass honors it
    monkeypatch.setenv("DRINKME_MTP_DIET", "0")
    head = _bf16_head()
    assert arms._diet_in_memory(head, "cpu") == 0
    assert not isinstance(head.fc, swap.SWAPPED_LINEAR_TYPES)


def test_running_out_of_memory_is_the_passes_outcome_not_the_arms(monkeypatch):
    monkeypatch.setattr(arms, "free_all", lambda: None)

    def oom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(arms, "time_speculation", oom)
    results: dict = {}
    arms._speculate(results, object(), "stock", "m", None, "/snap", None)
    assert results == {"stock": {"skipped": "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"}}

    def bug(*a, **k):
        raise ValueError("the speculation pass asked for 'mtp' and the engine served 'serially'")

    monkeypatch.setattr(arms, "time_speculation", bug)
    with pytest.raises(ValueError, match="served 'serially'"):
        arms._speculate(results, object(), "compressed", "m", None, "/snap", None)


# ------------------------------------------------------------- run_arms --


def _timed(arm: str) -> dict:
    """time_speculation's result for a toy arm: rates that say which arm."""
    base = {"stock": 20.0, "compressed": 30.0}[arm]
    return {"ctx": 8192,
            "prompts": {p: {"sha256": spec_pass.sha256(spec_pass.PROMPTS[k]), "tokens": 100 + i}
                        for i, (p, k) in enumerate(spec_pass.BENCH_PROMPTS.items())},
            "timed": {p: {"samples": [base + i, base + i + 1, base + i - 1], "accepted_per_step": 1.5 + i / 10}
                      for i, p in enumerate(spec_pass.BENCH_PROMPTS)}}


def _stub_run_arms(monkeypatch, heads=("stock", "compressed")):
    """run_arms on tests/test_stream_compressed_arms.py's stubbed device,
    with an event log: each arm's plain timings and each speculation pass,
    in order."""
    from test_stream_compressed_arms import STAT, _stub_run_arms_device

    _stub_run_arms_device(monkeypatch)
    monkeypatch.setattr(arms, "live_available_bytes", lambda kind: 100 * 10 ** 9)
    monkeypatch.setattr(arms, "refuse_if_stock_wont_fit", lambda *a, **k: None)
    events = []

    class _Model:
        def __init__(self, arm):
            self.arm = arm

        def parameters(self):
            return iter([torch.zeros(4, dtype=torch.bfloat16)])

    monkeypatch.setattr(arms, "load_stock_streaming", lambda *a, **k: _Model("stock"))
    monkeypatch.setattr(arms, "load_compressed_streaming", lambda *a, **k: (_Model("compressed"), [dict(STAT)]))
    monkeypatch.setattr(arms, "load_twin_streaming", lambda *a, **k: (_Model("twin"), []))
    monkeypatch.setattr(arms, "timed_decode",
                        lambda model, ids, **k: events.append(f"{model.arm} decode") or (None, [10.0] * 3))
    monkeypatch.setattr(arms, "timed_ttft",
                        lambda model, ids, **k: events.append(f"{model.arm} ttft") or [0.05] * 3)

    def speculate(model, arm, model_id, tok, snap, pack_dir=None, **k):
        assert model.arm == arm  # the arm's own loaded tree
        events.append(f"{arm} spec")
        return _timed(arm) if arm in heads else {"skipped": spec_pass.SKIP_NO_HEAD}

    monkeypatch.setattr(arms, "time_speculation", speculate)
    return events


def _run_arms(**kw):
    args = dict(run_stock=True, stock_memory_kind="vram", stock_bf16_bytes=10 * 10 ** 9, run_twin=True)
    args.update(kw)
    return arms.run_arms("toy/model", None, "hi", **args)


def test_run_arms_runs_the_pass_after_each_measured_arms_timings_and_never_on_the_twin(monkeypatch):
    events = _stub_run_arms(monkeypatch)
    r = _run_arms()
    assert events == ["stock decode", "stock ttft", "stock spec",
                      "compressed decode", "compressed ttft", "compressed spec", "twin decode"]
    assert r["stock_decode_tok_s"] == r["compressed_decode_tok_s"] == 10.0  # the plain numbers, untouched
    spec = r["spec"]
    assert set(spec) == {"mode", "tokens", "reps", "ctx", "prompts", "arms"}
    assert (spec["mode"], spec["tokens"], spec["reps"], spec["ctx"]) == ("mtp", 256, arms.DECODE_REPS, 8192)
    assert list(spec["prompts"]) == ["agent", "chat", "code"]
    assert set(spec["prompts"]["agent"]) == {"sha256", "tokens"}
    assert set(spec["arms"]) == {"stock", "compressed"}
    assert set(spec["arms"]["compressed"]["chat"]) == {"samples", "accepted_per_step"}


def test_run_arms_without_a_head_records_why_and_nothing_else_moves(monkeypatch):
    events = _stub_run_arms(monkeypatch, heads=())
    r = _run_arms()
    assert r["spec"] == {"skipped": "no MTP head"}
    assert r["stock_decode_tok_s"] == r["compressed_decode_tok_s"] == 10.0
    assert events.count("stock spec") == events.count("compressed spec") == 1


def test_run_arms_under_no_spec_never_runs_the_pass(monkeypatch):
    events = _stub_run_arms(monkeypatch)
    r = _run_arms(spec=False)
    assert r["spec"] == {"skipped": "--no-spec"}
    assert not any(e.endswith(" spec") for e in events)


def test_run_arms_times_no_stock_pass_for_a_stock_arm_that_did_not_measure(monkeypatch):
    events = _stub_run_arms(monkeypatch)
    r = _run_arms(run_stock=False)  # predicted not to fit
    assert "stock spec" not in events and list(r["spec"]["arms"]) == ["compressed"]

    def oom(*a, **k):
        raise RuntimeError("HIP out of memory. Tried to allocate 15.26 GiB")

    events.clear()
    monkeypatch.setattr(arms, "load_stock_streaming", oom)
    r = _run_arms()  # failed_load
    assert r["stock_outcome"] == "failed_load"
    assert "stock spec" not in events and list(r["spec"]["arms"]) == ["compressed"]


def test_an_arm_the_pass_could_not_time_is_named_beside_the_one_it_did(monkeypatch):
    _stub_run_arms(monkeypatch, heads=("compressed",))  # the stock arm's head yielded, say
    r = _run_arms()
    assert list(r["spec"]["arms"]) == ["compressed"]
    assert r["spec"]["skipped_arms"] == {"stock": "no MTP head"}


# ------------------------------------------------------------ the record --


def _spec_raw(spec: dict | None) -> dict:
    from tests.test_bench_record_shape import _fake_raw

    raw = _fake_raw()
    if spec is not None:
        raw["spec"] = spec
    return raw


def _record(monkeypatch, raw: dict, **run_kw) -> dict:
    from hosts import pin_linux_host
    from tests.test_bench_record_shape import _fake_hw, _fake_suggestion

    pin_linux_host(monkeypatch)
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _fake_suggestion())
    seen = {}
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: seen.update(k) or raw)
    r = bench.run(no_gemma=True, **run_kw)
    r["_run_arms_kw"] = seen
    return r


def _full_spec() -> dict:
    return spec_pass.assemble({"stock": _timed("stock"), "compressed": _timed("compressed")},
                              arms.DECODE_REPS)


def test_the_record_carries_the_six_names_with_the_median_and_every_sample(monkeypatch, capsys):
    r = _record(monkeypatch, _spec_raw(_full_spec()))
    assert r["_run_arms_kw"]["spec"] is True
    by = {m["name"]: m for m in r["metrics"]}
    assert [m["name"] for m in r["metrics"] if "_spec_" in m["name"]] == list(NAMES)
    agent = by["compressed_spec_decode_tok_s_agent"]
    assert agent == {"name": "compressed_spec_decode_tok_s_agent", "value": "30.0", "unit": "tok/s",
                     "samples": ["30.0", "31.0", "29.0"]}
    assert by["stock_spec_decode_tok_s_code"]["value"] == "22.0"
    # nothing else moved: the plain metrics are the ones a record without the pass carries
    plain = [m for m in _record(monkeypatch, _spec_raw(None))["metrics"]]
    assert [m for m in r["metrics"] if "_spec_" not in m["name"]] == plain
    out = capsys.readouterr().out
    assert "stock speculation (mtp): agent 20.0 · chat 21.0 · code 22.0 tok/s, 1.6 accepted/step" in out
    assert "compressed speculation (mtp): agent 30.0 · chat 31.0 · code 32.0 tok/s, 1.6 accepted/step" in out


def test_a_record_without_a_head_carries_none_and_says_why(monkeypatch, capsys):
    r = _record(monkeypatch, _spec_raw({"skipped": "no MTP head"}))
    assert not [m for m in r["metrics"] if "_spec_" in m["name"]]
    assert r["raw"]["spec"] == {"skipped": "no MTP head"}
    assert "speculation: skipped (no MTP head)" in capsys.readouterr().out


def test_no_spec_reaches_run_arms_and_the_record_says_so(monkeypatch, capsys):
    r = _record(monkeypatch, _spec_raw({"skipped": "--no-spec"}), no_spec=True)
    assert r["_run_arms_kw"]["spec"] is False
    assert not [m for m in r["metrics"] if "_spec_" in m["name"]]
    assert "speculation: skipped (--no-spec)" in capsys.readouterr().out


def test_no_spec_parses_on_the_command_line(monkeypatch):
    import types

    from drinkme import bootstrap, cli

    seen = {}
    monkeypatch.setattr(bootstrap, "ensure_accelerator", lambda *a, **k: types.SimpleNamespace(ok=True))
    monkeypatch.setattr(bench, "main_json", lambda *a, **k: seen.update(k))
    assert cli.main(["bench", "--model", "Qwen3.8-27B", "--no-spec"]) == 0
    assert seen["no_spec"] is True
    assert cli.main(["bench", "--model", "Qwen3.8-27B"]) == 0
    assert seen["no_spec"] is False


def test_a_stock_arm_that_was_not_measured_publishes_no_stock_spec_metric(monkeypatch):
    raw = _spec_raw(_full_spec())
    for k in [k for k in raw if k.startswith("stock_") and k != "stock_loader"] + ["vram_bf16_bytes"]:
        raw.pop(k, None)
    raw["stock_outcome"] = "skipped_predicted_nonfit"
    raw["stock_error"] = "skipped by bench's stock-arm fit check"
    r = _record(monkeypatch, raw)
    names = [m["name"] for m in r["metrics"] if "_spec_" in m["name"]]
    assert names == [n for n in NAMES if n.startswith("compressed_")]


def test_metal_records_carry_none_and_say_the_runtime_does_not_speculate(monkeypatch):
    from tests.test_bench_record_shape import _build_record_mlx

    r = _build_record_mlx(monkeypatch)
    assert not [m for m in r["metrics"] if "_spec_" in m["name"]]
    assert r["raw"]["spec"] == {"skipped": "runtime does not speculate"}


def test_the_record_validates_against_the_shipped_lexicon(monkeypatch):
    from drinkme.publish import validate as V

    r = _record(monkeypatch, _spec_raw(_full_spec()))
    r.pop("_run_arms_kw")
    assert V.validate(bench.lexicon_safe(r)) == []


def test_the_lexicon_describes_the_metrics_and_raw_spec():
    import json

    lex = json.loads((ROOT / "lexicons" / "wtf.petrichor.drinkme.measurement.json").read_text())
    props = lex["defs"]["main"]["record"]["properties"]
    metrics = props["metrics"]["description"]
    assert "<arm>_spec_decode_tok_s_<prompt>" in metrics
    assert f"{spec_pass.TOKENS} new tokens" in metrics
    for prompt in spec_pass.BENCH_PROMPTS:
        assert f"prompt {prompt} (" in metrics or f", {prompt} (" in metrics or f"and {prompt} (" in metrics
    raw = props["raw"]["description"]
    assert "spec — the speculation pass" in raw
    for why in (spec_pass.SKIP_NO_HEAD, "does not speculate", spec_pass.SKIP_FLAG):
        assert why in raw, why


# ---------------------------------------------------------- spec_pass --


def test_assemble_both_ways():
    assert spec_pass.assemble({}, 3) == {"skipped": "no MTP head"}
    assert spec_pass.assemble({"stock": {"skipped": "no MTP head"},
                               "compressed": {"skipped": "no MTP head"}}, 3) == {"skipped": "no MTP head"}
    # the head loaded and the pass ran out of memory on every arm: it ran, and says so
    oom = spec_pass.assemble({"compressed": {"skipped": "OutOfMemoryError: ..."}}, 3)
    assert oom == {"mode": "mtp", "tokens": 256, "reps": 3, "arms": {},
                   "skipped_arms": {"compressed": "OutOfMemoryError: ..."}}
    assert spec_pass.metrics(oom, True) == []


def test_metrics_refuse_an_arm_missing_a_prompt():
    spec = _full_spec()
    del spec["arms"]["compressed"]["code"]
    with pytest.raises(ValueError, match="compressed arm without its 'code' prompt"):
        spec_pass.metrics(spec, True)


def test_the_median_is_the_plain_decodes_median():
    import numpy as np

    for xs in ([29.07, 22.41, 24.6], [1.0, 2.0, 3.0, 4.0], [10.005, 10.015, 10.0]):
        assert spec_pass.median(xs) == round(float(np.median(xs)), 2)
