"""bench/agreement.py's near-tie rule and the verdicts of the gates that use
it, on synthetic inputs — CPU only, no model.

The gates themselves need a GPU and a real model. What is pinned here is how
each one reads two runs: the same output passes, a part at a near-tie is
reported and passes, a part above agreement.NEAR_TIE_MARGIN fails, and a
bookkeeping or format failure fails whatever the text says.
"""

import importlib.util
import json
import os
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "bench")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", os.path.join(BENCH, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def agreement():
    return _load("agreement")


# ------------------------------------------------------------ agreement --


def test_the_margin_is_half_a_logit(agreement):
    assert agreement.NEAR_TIE_MARGIN == 0.5


def test_first_difference(agreement):
    assert agreement.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert agreement.first_difference([1, 2, 3], [1, 5, 3]) == 1
    assert agreement.first_difference([1, 2], [1, 2, 3]) == 2
    assert agreement.first_difference("abc", "abd") == 2


def test_a_fork_at_a_near_tie_is_reported_and_passes(agreement):
    same = agreement.fork([1, 2, 3], [1, 2, 3], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0])
    assert same == {"at": None}
    assert agreement.label(same) == "agree" and not agreement.above_margin(same)

    tie = agreement.fork([1, 2, 3], [1, 7, 3], [3.0, 0.125, 1.0], [3.0, 0.25, 1.0])
    assert tie["at"] == 1 and tie["tokens"] == [2, 7] and tie["margins"] == [0.125, 0.25]
    assert tie["near_tie"] is True and agreement.label(tie) == "near-tie"
    assert not agreement.above_margin(tie)
    assert agreement.describe(tie) == "part at token 1: 2 (margin 0.125) against 7 (margin 0.25) -> near-tie"


def test_a_fork_above_the_margin_fails(agreement):
    # one run was sure of its pick: not a near-tie, whatever the other says
    rec = agreement.fork([1, 2, 3], [1, 7, 3], [3.0, 4.0, 1.0], [3.0, 0.25, 1.0])
    assert rec["near_tie"] is False
    assert agreement.above_margin(rec) and agreement.label(rec) == "ABOVE NEAR-TIE MARGIN"
    edge = agreement.fork([1, 2], [1, 7], [1.0, agreement.NEAR_TIE_MARGIN], [1.0, 0.0])
    assert edge["near_tie"] is True  # at the margin is still a near-tie


def test_a_fork_with_no_margin_is_reported_not_failed(agreement):
    rec = agreement.fork([1, 2], [1, 2, 3])  # one run stopped: no margin past its end
    assert rec["at"] == 2 and rec["near_tie"] is None
    assert agreement.label(rec) == "no margin" and not agreement.above_margin(rec)
    assert agreement.is_near_tie(None, None) is None


def test_row_fork_reads_both_picks_in_both_rows(agreement):
    same = agreement.row_fork(torch.tensor([0.0, 5.0, 1.0]), torch.tensor([0.0, 5.5, 1.0]))
    assert same == {"picks": [1, 1], "agree": True} and not agreement.above_margin(same)

    tie = agreement.row_fork(torch.tensor([0.0, 5.0, 4.875]), torch.tensor([0.0, 4.875, 5.0]))
    assert tie["picks"] == [1, 2] and tie["gaps"] == [0.125, 0.125]
    assert tie["near_tie"] is True and not agreement.above_margin(tie)

    # the second row's pick is third in the first row: its top-2 margin would
    # be small, but the gap between the two picks is not
    far = agreement.row_fork(torch.tensor([4.9, 5.0, 1.0]), torch.tensor([0.0, 1.0, 1.1]))
    assert far["picks"] == [1, 2] and far["gaps"][0] == pytest.approx(4.0)
    assert agreement.above_margin(far)
    assert agreement.describe(far).endswith("-> ABOVE NEAR-TIE MARGIN")
    assert agreement.row_fork([1.0, 9.0], [1.0, 9.5])["agree"]  # plain lists too


def test_text_part(agreement):
    assert agreement.text_part("same", "same") is None
    p = agreement.text_part("hello world", "hello there", context=3)
    assert p == {"at_char": 6, "a": "lo wor", "b": "lo the"}


def test_pick_tap_records_by_position_and_restores(agreement, monkeypatch):
    from drinkme.serving import engines

    def fake(logits, params, generator=None, prev_ids=None, gen_ids=None, **kw):
        return int(logits.argmax())

    monkeypatch.setattr(engines, "sample_next", fake)
    with agreement.PickTap() as tap:
        assert engines.sample_next is not fake
        # positions out of call order: the record is keyed by len(gen_ids)
        engines.sample_next(torch.tensor([0.0, 3.0, 1.0]), None, gen_ids=[9])
        engines.sample_next(torch.tensor([2.0, 0.0, 1.5]), None, gen_ids=[])
        ids, margins = tap.take()
        assert ids == [0, 1] and margins == [0.5, 2.0]
        assert tap.take() == ([], [])
    assert engines.sample_next is fake


# ------------------------------------------------- prefix_cache_verify --


def _turn(text, ids=(), margins=(), cached=0, prompt=100):
    return {"text": text, "ids": list(ids), "margins": list(margins),
            "cached_tokens": cached, "prompt_tokens": prompt}


def test_prefix_cache_compare_turns():
    verify = _load("prefix_cache_verify")
    a = [_turn("x", [1, 2]), _turn("y", [3, 4], [2.0, 0.25]), _turn("z", [5])]
    b = [_turn("x", [1, 2]), _turn("y'", [3, 6], [2.0, 0.125]), _turn("z'", [7])]
    rows = verify.compare_turns(a, b, own_history=True)
    assert rows[0] == {"compared": True, "same_text": True, "fork": None}
    assert rows[1]["fork"]["at"] == 1 and rows[1]["fork"]["near_tie"] is True
    assert rows[2] == {"compared": False}  # each arm conditioned on its own reply
    replayed = verify.compare_turns(a, b, own_history=False)
    assert replayed[2]["compared"] and replayed[2]["fork"]["near_tie"] is None

    sure = [_turn("x", [1, 2], [3.0, 3.0])]
    other = [_turn("x'", [1, 9], [3.0, 0.1])]
    from agreement import above_margin

    assert above_margin(verify.compare_turns(sure, other, own_history=True)[0]["fork"])


def test_prefix_cache_bookkeeping():
    verify = _load("prefix_cache_verify")
    cold = [_turn("a"), _turn("b"), _turn("c")]
    warm = [_turn("a", cached=0, prompt=100), _turn("b", cached=96, prompt=180),
            _turn("c", cached=176, prompt=260)]
    assert verify.cache_bookkeeping(cold, warm) == []

    assert verify.cache_bookkeeping([_turn("a", cached=5)], warm[:1]) == [
        "cold turn 1: cached 5, expected 0"]
    flat = [warm[0], _turn("b", cached=96, prompt=180), _turn("c", cached=96, prompt=260)]
    assert verify.cache_bookkeeping(cold, flat) == ["warm turn 3: cached 96, not more than turn 2's 96"]
    whole = [warm[0], _turn("b", cached=180, prompt=180)]
    assert any("at least one re-runs" in p for p in verify.cache_bookkeeping(cold, whole))
    assert verify.cache_bookkeeping(cold, [_turn("a", cached=3)]) == [
        "warm turn 1: cached 3, expected 0 after the reset"]


# ------------------------------------------------- prefix_slots_verify --


def _slot_rec(logits, text="t", ids=(1, 2), margins=(2.0, 2.0), cached=10, expect="warm"):
    return {"conv": "A", "turn": 2, "expect": expect, "text": text, "prompt_tokens": 50,
            "cached_tokens": cached, "wall": 0.1, "logits": logits,
            "ids": list(ids), "margins": list(margins)}


def test_prefix_slots_verdict():
    verify = _load("prefix_slots_verify")
    row = torch.tensor([0.0, 5.0, 4.9])
    near = torch.tensor([0.0, 4.9, 5.0])
    far = torch.tensor([0.0, 1.0, 5.0])

    v = verify.verdict(verify.compare([_slot_rec(row)], [_slot_rec(row.clone())]))
    assert v["pass"] and v["first_token_agrees"] and v["same_text"]

    rows = verify.compare([_slot_rec(row, margins=(0.25, 2.0))],
                          [_slot_rec(near, text="u", ids=(2, 1), margins=(0.1, 2.0))])
    v = verify.verdict(rows)
    assert v["pass"] and not v["first_token_agrees"] and not v["same_text"]
    assert rows[0]["first_token"]["near_tie"] and rows[0]["text_fork"]["near_tie"]

    v = verify.verdict(verify.compare([_slot_rec(row)], [_slot_rec(far)]))
    assert not v["pass"] and v["parts_above_near_tie_margin"] == ["A2"]

    v = verify.verdict(verify.compare([_slot_rec(row, text="t")],
                                      [_slot_rec(row.clone(), text="u", ids=(1, 3), margins=(2.0, 3.0))]))
    assert not v["pass"]  # the text parted where the model was sure

    v = verify.verdict(verify.compare([_slot_rec(row, cached=0)], [_slot_rec(row.clone())]))
    assert not v["pass"] and not v["all_cache_as_expected"]  # expected warm, got no reuse

    v = verify.verdict(verify.compare([_slot_rec(None)], [_slot_rec(row)]))
    assert not v["pass"] and not v["all_logits_seen"]  # the tap missed the prefill row


# ------------------------------------------------------- vision_smoke --


def test_vision_smoke_warm_vs_cold():
    smoke = _load("vision_smoke")
    from agreement import above_margin

    reply = lambda r: r["text"]  # noqa: E731
    w = [{"text": "a", "ids": [1]}, {"text": "b", "ids": [2, 3], "margins": [1.0, 0.0625]},
         {"text": "c", "ids": [4]}]
    c = [{"text": "a", "ids": [1]}, {"text": "b'", "ids": [2, 5], "margins": [1.0, 0.125]},
         {"text": "c'", "ids": [6]}]
    rows = smoke._warm_vs_cold(w, c, reply)
    assert rows[0] == {"compared": True, "same": True, "fork": None}
    assert rows[1]["fork"]["near_tie"] is True and rows[2] == {"compared": False}
    assert not any(above_margin(r.get("fork")) for r in rows)

    c[1]["margins"] = [1.0, 6.0]
    assert any(above_margin(r.get("fork")) for r in smoke._warm_vs_cold(w, c, reply))
    assert smoke._chat_reply({"text": "x", "reasoning": "r"}) == ("r", "x")


# ----------------------------------------------------- cuda_graph_gate --


def test_cuda_graph_gate_reads_rows_against_the_margin():
    gate = _load("cuda_graph_gate")
    a = {"text": "abc", "rows": [[1, 3.0], [2, 0.125], [3, 2.0]]}
    b = {"text": "abd", "rows": [[1, 3.0], [9, 0.25], [3, 2.0]]}
    f = gate.rows_fork(a["rows"], b["rows"])
    assert f["at"] == 1 and f["near_tie"] is True
    assert gate.text_and_rows(a, a) == "same text (3 chars)"
    assert gate.text_and_rows(a, b) == (
        "text parts at char 2; part at emitted row 1: 2 (margin 0.125) against 9 (margin 0.25) -> near-tie")
    b["rows"][1][1] = 5.0
    assert gate.label(gate.rows_fork(a["rows"], b["rows"])) == "ABOVE NEAR-TIE MARGIN"


# --------------------------------------------------------- genreq_gate --


def _case(content="391", status=200, prompt=20, obs=1, reasoning=""):
    transcript = {"reasoning": reasoning, "content": content, "tool_calls": [], "finish": "stop",
                  "usage": {"prompt": prompt, "completion": 3}}
    return {"status": status, "sha256": json.dumps(transcript, sort_keys=True), "transcript": transcript,
            "server_ttft_observations": obs, "server_ttft_s": 0.1, "client_ttft_s": None}


def _receipts(tmp_path, before, after):
    paths = []
    for name, cases in (("before", before), ("after", after)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"cases": cases}))
        paths.append(str(p))
    return paths


def test_genreq_compare(tmp_path, capsys):
    gate = _load("genreq_gate")
    schema = '{"title": "The Keeper"}'
    base = {"chat/think=off/json": _case(), "chat/json_schema": _case(schema)}

    assert gate.compare(*_receipts(tmp_path, base, base)) == 0
    assert "2/2 the same sha" in capsys.readouterr().out

    parted = {**base, "chat/think=off/json": _case("392")}
    assert gate.compare(*_receipts(tmp_path, base, parted)) == 0  # reported, not failed
    out = capsys.readouterr().out
    assert "content parts at char 2" in out and "VERDICT: PASS" in out

    # a reply that parts can change how many TTFTs the server observes
    shape = {**base, "chat/think=off/json": _case("392", obs=0)}
    assert gate.compare(*_receipts(tmp_path, base, shape)) == 0
    capsys.readouterr()

    for after, why in (({**base, "chat/think=off/json": _case(obs=2)}, "for the same reply"),
                       ({**base, "chat/think=off/json": _case(prompt=21)}, "rendered differently"),
                       ({**base, "chat/think=off/json": _case(status=500)}, "answered 500"),
                       ({**base, "chat/json_schema": _case('{"title": 3}')}, "does not parse"),
                       ({"chat/json_schema": _case(schema)}, "missing after")):
        assert gate.compare(*_receipts(tmp_path, base, after)) == 1, why
        out = capsys.readouterr().out
        assert "VERDICT: FAIL" in out and why in out


# -------------------------------------------------------- vision_bitpin --


def _bitpin_files(tmp_path, rows: dict):
    feats = torch.ones(4, 3, dtype=torch.bfloat16)
    arm = {"grid": [1, 2, 2], "ids": [1, 2, 3], "digest": "d", "patch_embed_routed": True,
           "features_unbounded": feats, "features_bounded": feats}
    torch.save({**arm, "logits_unbounded": rows["stock_unbounded"], "logits_bounded": rows["stock_bounded"],
                "token_unbounded": 1, "token_bounded": 1}, tmp_path / "stock.pt")
    torch.save({**arm, "logits_unbounded": rows["comp_unbounded"], "logits_bounded": rows["comp_bounded"],
                "token_unbounded": 1, "token_bounded": 1}, tmp_path / "comp.pt")
    torch.save({"processor_pixels_equal": True, "processor_ids_equal": True, "features_ref": feats,
                "features_ref_served_unbounded": feats, "features_ref_served_bounded": feats,
                "features_fp32": feats.float(), "logits_ref": rows["ref"],
                "logits_ref_served_bounded": rows["ref_served"]}, tmp_path / "ref.pt")
    return types.SimpleNamespace(out=str(tmp_path), model="27b")


@pytest.mark.parametrize("comp_row, first_token_ok", [
    (torch.tensor([0.0, 5.0, 4.0]), True),     # every row picks token 1
    (torch.tensor([0.0, 4.95, 5.0]), True),    # the compressed rows part at a near-tie
    (torch.tensor([0.0, 1.0, 5.0]), False),    # ... or where they were sure of another
])
def test_vision_bitpin_first_token(tmp_path, capsys, comp_row, first_token_ok):
    bitpin = _load("vision_bitpin")
    row = torch.tensor([0.0, 5.0, 4.9])
    rows = {"ref": row, "ref_served": row, "stock_unbounded": row, "stock_bounded": row,
            "comp_unbounded": comp_row, "comp_bounded": comp_row}
    bitpin.compare(_bitpin_files(tmp_path, rows))
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert v["pass"]["first_token_agrees_or_near_tie"] is first_token_ok
    assert v["pass"]["tower_bits"]  # the tower pins are untouched by any of this
    assert v["verdict"] == ("PASS" if first_token_ok else "FAIL")
    assert f"VERDICT {v['verdict']}" in capsys.readouterr().out
