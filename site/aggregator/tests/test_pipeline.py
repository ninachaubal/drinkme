"""The pipeline: build_points' validation, filters F2-F5, the vote collapse, the output."""
import datetime as dt
import json
import os
import pathlib

import pytest

from aggregator import pipeline
from aggregator.pipeline import Settings, bp
from records import DID_A, DID_B, edit, fitpoint, rec, uri


def run(*records, **settings):
    return pipeline.run(list(records), Settings(**settings))


def only_exclusion(*records, **settings):
    points, excluded = run(*records, **settings)
    assert points == [] and len(excluded) == 1, (points, excluded)
    return excluded[0]


def test_a_clean_record_and_a_fit_point_are_admitted():
    points, excluded = run(rec(), fitpoint(rkey="3fit", name="Qwen3-14B"))
    assert excluded == []
    assert sorted(p["model"]["name"] for p in points) == ["Qwen3-14B", "Qwen3-8B"]


# ------------------------------------------------ validation is build_points' --

def test_validation_is_build_points_admit_by_import():
    assert pipeline.bp.admit is bp.admit and bp.__file__.endswith("site/data/build_points.py")


def test_an_incompatible_version_is_excluded_by_build_points_policy():
    e = only_exclusion(edit(rec(), lambda v: v.update(version="0.0.1")))
    assert e["filter"] == "validate" and "not comparable with drinkme" in e["reason"]


def test_a_record_missing_an_identity_field_is_excluded_by_name():
    e = only_exclusion(edit(rec(), lambda v: v["model"].pop("revision")))
    assert e["filter"] == "validate" and "model.revision" in e["reason"]


def test_a_record_of_another_type_or_collection_is_not_admitted():
    assert "$type" in only_exclusion(edit(rec(), lambda v: v.update({"$type": "app.bsky.feed.post"})))["reason"]
    other = {**rec(), "uri": f"at://{DID_A}/app.bsky.feed.post/3aaa"}
    assert only_exclusion(other)["filter"] == "validate"


def test_a_malformed_record_is_excluded_and_never_stops_the_run():
    bad = edit(rec(rkey="3bad"), lambda v: v.update(metrics=["not an object"], stock={"outcome": "measured"}))
    points, excluded = run(bad, rec(rkey="3good", name="Qwen3-4B"))
    assert [p["model"]["name"] for p in points] == ["Qwen3-4B"]
    assert [e["source"] for e in excluded] == [uri(rkey="3bad")]


# ------------------------------------------------- F1 (retired 2026-09-29) --

def test_a_metal_record_is_kept_like_rocm_and_cuda():
    for platform in ("metal", "rocm", "cuda"):
        points, excluded = run(rec(platform=platform))
        assert excluded == [] and len(points) == 1, platform


# ------------------------------------------------------------------- F2 --

@pytest.mark.parametrize("kw, needle", [
    (dict(verified=252, swapped=253), "252 of 253"),
    (dict(verified=0, swapped=0), "0 of 0"),
    (dict(verified=None), "unrecorded"),
    (dict(verified="253"), "unrecorded"),
    (dict(verification="skipped"), "'skipped'"),
])
def test_f2_excludes_a_record_that_did_not_pass_the_bit_exact_gate(kw, needle):
    e = only_exclusion(rec(**kw))
    assert e["filter"] == "F2" and needle in e["reason"], e


def test_f2_excludes_a_record_without_raw():
    assert only_exclusion(edit(rec(), lambda v: v.pop("raw")))["filter"] == "F2"


@pytest.mark.parametrize("how", ["pack", "roundtrip", None])
def test_f2_passes_equal_counts_verified_either_way(how):
    r = rec(verification=how)
    if how is None:
        r["value"]["raw"].pop("verification")
    assert run(r)[1] == []


# ------------------------------------------------------------------- F3 --

def test_f3_excludes_a_compressed_arm_faster_than_its_bandwidth_allows():
    # 239.3 GB/s over 12.096 GB is 19.78 tok/s; 1.10 x that is 21.76
    e = only_exclusion(rec(comp="22.0"))
    assert e["filter"] == "F3" and e["numbers"]["arm"] == "compressed"
    assert e["numbers"]["bound_tok_s"] == pytest.approx(239.3 / 12.096, abs=1e-3)
    assert e["numbers"]["decode_tok_s"] == 22.0 and e["numbers"]["tolerance"] == 1.10
    assert run(rec(comp="21.7"))[1] == []  # inside the tolerance


def test_f3_excludes_a_stock_arm_faster_than_its_bandwidth_allows():
    # 239.3 / 16.384 = 14.61 tok/s; 1.10 x = 16.07
    e = only_exclusion(rec(stock="16.2"))
    assert e["filter"] == "F3" and e["numbers"]["arm"] == "stock"
    assert run(rec(stock="16.0"))[1] == []


def test_f3_excludes_a_twin_arm_faster_than_its_bandwidth_allows():
    # 239.3 / 16.417 = 14.58 tok/s; 1.10 x = 16.03
    e = only_exclusion(rec(twin="16.2"))
    assert e["filter"] == "F3" and e["numbers"]["arm"] == "twin"
    assert e["numbers"]["bound_tok_s"] == pytest.approx(239.3 / 16.417, abs=1e-3)
    assert run(rec(twin="12.36"))[1] == []
    # uncheckable without its decode-read bytes
    e = only_exclusion(edit(rec(twin="12.36"), lambda v: v.__setitem__(
        "metrics", [m for m in v["metrics"] if m["name"] != "twin_decode_read_gb"])))
    assert e["filter"] == "F3" and "uncheckable" in e["reason"] and e["numbers"]["arm"] == "twin"


def test_f3_tolerance_is_a_setting():
    assert run(rec(comp="21.7"), tolerance=1.05)[1][0]["filter"] == "F3"


@pytest.mark.parametrize("fn", [
    lambda v: v["metrics"].__setitem__(1, {"name": "read_gb_s", "value": "0", "unit": "GB/s"}),
    lambda v: v.__setitem__("metrics", [dict(m, value="NaN") if m["name"] == "compressed_decode_read_gb"
                                        else m for m in v["metrics"]]),
    lambda v: v.__setitem__("metrics", [m for m in v["metrics"] if m["name"] != "stock_decode_read_gb"]),
])
def test_f3_excludes_an_arm_whose_claim_cannot_be_checked(fn):
    e = only_exclusion(edit(rec(), fn))
    assert e["filter"] == "F3" and "uncheckable" in e["reason"]


def test_f3_checks_a_fit_points_compressed_arm_only():
    assert run(fitpoint())[1] == []
    assert only_exclusion(fitpoint(comp="30"))["filter"] == "F3"


# ------------------------------------------------- F3: speculation metrics --

# a directory of real spec-bench records, named by DRINKME_SPEC_RECORDS; the test below skips without it
SPEC_DIR = pathlib.Path(os.environ["DRINKME_SPEC_RECORDS"]) if os.environ.get("DRINKME_SPEC_RECORDS") else None
# H100-like: stock decode-read 51.244 GB and compressed 36.044 GB over 3073.185 GB/s
# bound 59.98 and 85.26 tok/s; the depth-4 limit is 5 x 1.10 x that.
H100_SPEC = {"stock_spec_decode_tok_s_agent": "28.55", "stock_spec_decode_tok_s_chat": "30.79",
             "stock_spec_decode_tok_s_code": "36.34", "compressed_spec_decode_tok_s_agent": "21.42",
             "compressed_spec_decode_tok_s_chat": "24.77", "compressed_spec_decode_tok_s_code": "28.88"}


def spec_rec(drop=(), **over):
    vals = {**H100_SPEC, **over}
    r = rec(read="3073.185", comp="13.28", stock="15.86", comp_gb="36.044", stock_gb="51.244")
    return edit(r, lambda v: v.__setitem__("metrics", [
        m for m in v["metrics"] if m["name"] not in drop] + [
        {"name": k, "value": x, "unit": "tok/s"} for k, x in vals.items()]))


def test_f3_admits_a_record_with_h100_like_speculation_metrics():
    assert run(spec_rec())[1] == []


def test_f3_excludes_a_speculation_metric_at_six_times_its_arms_bound():
    bound = 3073.185 / 36.044
    limit = 5 * 1.10 * bound
    e = only_exclusion(spec_rec(compressed_spec_decode_tok_s_chat=f"{6 * bound:.2f}"))
    assert e["filter"] == "F3" and e["numbers"]["arm"] == "compressed"
    assert e["numbers"]["metric"] == "compressed_spec_decode_tok_s_chat"
    assert e["numbers"]["bound_tok_s"] == pytest.approx(bound, abs=1e-3)
    assert e["numbers"]["spec_depth"] == 4
    assert e["numbers"]["limit_tok_s"] == pytest.approx(limit, abs=1e-3)
    assert "compressed_spec_decode_tok_s_chat" in e["reason"]
    assert run(spec_rec(compressed_spec_decode_tok_s_chat=f"{limit - 0.1:.2f}"))[1] == []
    assert run(spec_rec(compressed_spec_decode_tok_s_chat=f"{limit + 0.1:.2f}"))[1][0]["filter"] == "F3"


def test_f3_a_speculation_metric_with_no_bound_for_its_arm_is_uncheckable():
    # a fit-point-shaped stock arm (no plain decode, no bound) that still claims speculation
    r = spec_rec(drop=("stock_decode_read_gb", "stock_decode_tok_s"))
    e = only_exclusion(edit(r, lambda v: v.__setitem__("stock", {"outcome": "skipped_predicted_nonfit"})))
    assert e["filter"] == "F3" and "uncheckable" in e["reason"]
    assert e["numbers"]["arm"] == "stock" and "stock_spec_decode_tok_s" in e["reason"]


def test_f3_a_record_without_speculation_metrics_is_unchanged():
    assert run(rec())[1] == []
    assert only_exclusion(rec(comp="22.0"))["numbers"]["decode_tok_s"] == 22.0


@pytest.mark.skipif(SPEC_DIR is None or not SPEC_DIR.is_dir(), reason="DRINKME_SPEC_RECORDS unset or not a directory")
@pytest.mark.parametrize("name", sorted(p.name for p in SPEC_DIR.glob("*.json")) if SPEC_DIR else [])
def test_f3_admits_the_real_speculation_records(name):
    value = json.loads((SPEC_DIR / name).read_text())
    points, excluded = run({"uri": uri(rkey="3real"), "cid": "bafyreal", "value": value})
    assert excluded == [] and len(points) == 1


# Muse-Glimmer-30B gulp on Strix Halo: the twin decoded at 4.07
# tok/s against a 222.556 GB/s read probe. Its resident footprint is 59.759
# GB, vision tower included; one decode step reads 53.017 GB (the text
# Linears and one embedding row; tests/test_decode_read_bound.py derives the
# bytes from the model's tree).
GLIMMER = dict(name="Muse-Glimmer-30B", read="222.556", comp="3.04", comp_gb="41.498",
               stock="3.63", stock_gb="59.646", twin="4.07", twin_gb="59.759")
GLIMMER_READ = {"stock": "53.017", "twin": "53.017", "compressed": "37.48"}


def test_f3_divides_by_the_bytes_a_decode_step_reads_when_the_record_carries_them():
    """The Glimmer twin sits at 0.970 of the bound over its decode-read bytes."""
    assert run(rec(**GLIMMER, decode_read=GLIMMER_READ))[1] == []
    # the limit is unchanged: 1.10 x 222.556 / 53.017 = 4.618 tok/s
    e = only_exclusion(rec(**{**GLIMMER, "twin": "4.63"}, decode_read=GLIMMER_READ))
    assert e["filter"] == "F3" and e["numbers"]["arm"] == "twin"
    assert e["numbers"]["twin_decode_read_gb"] == 53.017
    assert e["numbers"]["bound_tok_s"] == pytest.approx(222.556 / 53.017, abs=1e-3)
    assert "twin_decode_read_gb" in e["reason"]


@pytest.mark.parametrize("bad", ["0", "NaN", "-1", "fast"])
def test_f3_an_unusable_decode_read_is_uncheckable(bad):
    """An arm is checked against its `<arm>_decode_read_gb`: a value that
    is not a positive number makes the arm uncheckable."""
    e = only_exclusion(rec(**GLIMMER, decode_read={**GLIMMER_READ, "twin": bad}))
    assert e["filter"] == "F3" and "uncheckable" in e["reason"] and "twin_decode_read_gb" in e["reason"]


def test_f5_judges_the_baseline_against_the_same_bound_as_f3():
    b = pipeline.baseline_efficiency(rec(**GLIMMER, decode_read=GLIMMER_READ)["value"])
    assert b["bound_tok_s"] == pytest.approx(222.556 / 53.017, abs=1e-3)


# ------------------------------------------------------------------- F4 --

def test_f4_denylist_is_empty_by_default_and_read_from_a_file(tmp_path):
    assert Settings().denylist == frozenset() and pipeline.load_denylist(None) == frozenset()
    f = tmp_path / "deny.txt"
    f.write_text(f"# the deploy's list\n{DID_B}\n\n{uri(DID_A, '3bad')}  # one record\n")
    deny = pipeline.load_denylist(f)
    assert deny == {DID_B, uri(DID_A, "3bad")}
    points, excluded = pipeline.run([rec(DID_B, "3x"), rec(DID_A, "3bad"), rec(DID_A, "3ok", name="Qwen3-4B")],
                                    Settings(denylist=deny))
    assert [p["site"]["source"] for p in points] == [uri(DID_A, "3ok")]
    assert {(e["source"], e["filter"], e["reason"]) for e in excluded} == {
        (uri(DID_B, "3x"), "F4", "denylisted contributor"), (uri(DID_A, "3bad"), "F4", "denylisted record")}


# ------------------------------------------------------------------- F5 --
# The fixture's stock bound is 239.3 / 16.384 = 14.6057 tok/s.

BOUND = 239.3 / 16.384


def at(eff):
    """A stock tok/s at this fraction of the fixture's bound."""
    return f"{eff * BOUND:.4f}"


def did(i):
    return f"did:plc:{chr(ord('c') + i) * 24}"


def cell(*effs, **kw):
    """One record per contributor, each at its baseline efficiency."""
    return [rec(did(i), f"3c{i}", stock=at(e), **kw) for i, e in enumerate(effs)]


def admitted(points):
    return sorted(p["site"]["contributor"] for p in points)


def test_f5_defaults_live_in_settings():
    s = Settings()
    assert (s.baseline_arm, s.baseline_floor, s.baseline_relative, s.baseline_min_contributors) == \
        ("stock", 0.10, 0.6, 3)


def test_f5_admits_the_dev_small_models_launch_bound_stock_arm():
    # the dev Qwen3-0.6B on Strix Halo: 41.49 tok/s against 238.48 / 1.192 = 200.07, 21%
    r = rec(comp="85.24", stock="41.49", read="238.48", comp_gb="0.973", stock_gb="1.192")
    assert run(r)[1] == []
    e = only_exclusion(r, baseline_floor=0.25)  # the floor is a setting
    assert e["filter"] == "F5" and "absolute floor" in e["reason"]
    n = e["numbers"]
    assert n["decode_tok_s"] == 41.49 and n["bound_tok_s"] == pytest.approx(200.067, abs=1e-3)
    assert n["efficiency"] == pytest.approx(0.2074, abs=1e-4) and n["floor"] == 0.25 and n["arm"] == "stock"
    assert "41.49" in e["reason"] and "200.1" in e["reason"]


def test_f5_drops_a_lone_record_under_the_absolute_floor():
    e = only_exclusion(*cell(0.08))
    assert e["filter"] == "F5" and "absolute floor" in e["reason"] and "whatever its peers show" in e["reason"]
    assert e["numbers"]["efficiency"] == pytest.approx(0.08, abs=1e-4) and e["numbers"]["floor"] == 0.10
    assert run(*cell(0.10))[1] == []


def test_f5_the_absolute_floor_applies_in_a_full_cell_too():
    points, excluded = run(*cell(0.09, 0.10, 0.11))
    assert admitted(points) == [did(1), did(2)]
    assert [(e["source"], "absolute floor" in e["reason"]) for e in excluded] == [(uri(did(0), "3c0"), True)]


def test_f5_drops_a_contributor_under_six_tenths_of_five_honest_peers():
    """Five honest contributors and one at 0.3x their median: the median of all six
    is (0.50 + 0.52) / 2 = 0.51, and 0.156 < 0.6 x 0.51 = 0.306."""
    points, excluded = run(*cell(0.48, 0.50, 0.52, 0.55, 0.58, 0.3 * 0.52))
    assert admitted(points) == [did(i) for i in range(5)]
    [e] = excluded
    assert e["source"] == uri(did(5), "3c5") and e["filter"] == "F5" and "relative" in e["reason"]
    n = e["numbers"]
    assert n["efficiency"] == pytest.approx(0.156, abs=1e-4) and n["cell_median"] == pytest.approx(0.51, abs=1e-4)
    assert n["contributors"] == 6 and n["relative"] == 0.6 and n["arm"] == "stock"
    assert "15.6%" in e["reason"] and "51.0%" in e["reason"] and "6 contributors" in e["reason"]


def test_f5_admits_a_low_contributor_when_the_cell_has_too_few_peers():
    """Two contributors, one at 0.3x the other: not enough peers to judge."""
    points, excluded = run(*cell(0.5, 0.15))
    assert excluded == [] and admitted(points) == [did(0), did(1)]
    # the threshold is a setting: at 2 the low one goes
    points, excluded = run(*cell(0.5, 0.15), baseline_min_contributors=2)
    assert admitted(points) == [did(0)] and excluded[0]["numbers"]["contributors"] == 2


def test_f5_the_relative_multiple_is_a_setting():
    assert run(*cell(0.48, 0.50, 0.52, 0.55, 0.58, 0.156), baseline_relative=0.3)[1] == []


def test_f5_counts_contributors_not_records():
    """One contributor's three runs are one vote: with one other contributor the cell
    has two, too few to judge, however many records it holds."""
    runs = [rec(DID_A, f"3r{i}", stock=at(0.15), created=f"2026-09-2{i}T00:00:00+00:00") for i in range(3)]
    points, excluded = run(*runs, rec(DID_B, "3b", stock=at(0.5)))
    assert excluded == [] and len(points) == 2


def test_f5_drops_every_record_a_dropped_point_was_medianed_from():
    runs = [rec(DID_A, f"3r{i}", stock=at(0.15), created=f"2026-09-2{i}T00:00:00+00:00") for i in range(3)]
    points, excluded = run(*runs, *cell(0.5, 0.5))
    assert admitted(points) == [did(0), did(1)]
    assert sorted(e["source"] for e in excluded) == [uri(DID_A, f"3r{i}") for i in range(3)]
    assert {e["numbers"]["contributors"] for e in excluded} == {3}


def test_f5_one_wild_contributor_cannot_get_honest_ones_dropped():
    """A contributor near F3's ceiling (1.05 of the bound) among honest ones at 0.45-0.55.
    The median moves by at most one rank, so the honest stay."""
    for effs in [(0.45, 0.50, 1.05), (0.45, 0.50, 0.52, 1.05), (0.45, 0.48, 0.50, 0.52, 0.55, 1.05)]:
        points, excluded = run(*cell(*effs))
        assert excluded == [], (effs, excluded)


def test_f5_the_median_includes_the_judged_contributor():
    """Why not leave-one-out: with three contributors at 0.45, 0.50 and 1.05, the
    median of the other two for the 0.45 one is their mean, 0.775, and 0.45 would be
    under 0.6 x 0.775 = 0.465. The median of all three is 0.50, and 0.45 stays."""
    points, _ = run(*cell(0.45, 0.50, 1.05))
    assert admitted(points) == [did(0), did(1), did(2)]
    # the judged contributor's own value can only keep a low outlier in at the margin:
    # 0.25 among 0.5, 0.5 is still dropped (the median of all three is 0.5)
    points, excluded = run(*cell(0.5, 0.5, 0.25))
    assert admitted(points) == [did(0), did(1)] and excluded[0]["numbers"]["cell_median"] == 0.5


def test_f5_judges_within_a_cell_only():
    """Low peers in another cell (here another profile) do not count."""
    points, excluded = run(*cell(0.5, 0.5), rec(did(2), "3g", stock=at(0.2), profile="gulp"))
    assert excluded == [] and len(points) == 3


@pytest.mark.parametrize("outcome", ["skipped_predicted_nonfit", "failed_load"])
def test_f5_leaves_fit_points_alone(outcome):
    """A fit point has no stock speed: at any floor, among any peers, it stays, a didn't-load row."""
    rs = [rec(did(i), f"3f{i}", outcome=outcome, comp="1.0") for i in range(4)]
    for r in rs:
        if outcome == "failed_load":
            r["value"]["stock"] = {"outcome": "failed_load", "error": "HIP out of memory"}
    points, excluded = run(*rs, baseline_floor=0.99, baseline_relative=0.99)
    assert excluded == [] and len(points) == 4 and {p["stock"]["outcome"] for p in points} == {outcome}


def test_f5_the_arm_is_a_parameter():
    assert pipeline.baseline_efficiency(rec(stock=at(0.5))["value"])["efficiency"] == pytest.approx(0.5, abs=1e-4)
    assert pipeline.baseline_efficiency(fitpoint()["value"]) is None
    with pytest.raises(KeyError):
        pipeline.baseline_efficiency(rec()["value"], "nonesuch")
    with pytest.raises(KeyError):  # the twin is a record's data, never a baseline
        pipeline.baseline_efficiency(rec()["value"], "twin")


# F5 judges the stock arm, the page's denominator, whatever else a record carries.
# The fixture's twin bound is 239.3 / 16.417 = 14.5764 tok/s.
TWIN_BOUND = 239.3 / 16.417


def twin_at(eff):
    return f"{eff * TWIN_BOUND:.4f}"


def test_f5_judges_stock_when_the_record_carries_the_twin_too():
    """A record's twin at 80% of its bound does not keep a stock arm under
    the floor, and a twin under the floor does not drop an honest stock arm."""
    e = only_exclusion(rec(stock=at(0.08), twin=twin_at(0.8)))
    assert e["filter"] == "F5" and e["numbers"]["arm"] == "stock" and "the stock arm's" in e["reason"]
    assert run(rec(stock=at(0.5), twin=twin_at(0.05)))[1] == []


def test_f5_relative_judges_stock_peers_twin_or_not():
    """Four contributors, two of whose records carry the twin: one cell, one
    median of four stock efficiencies (0.48, 0.50, 0.52, 0.2 -> 0.49)."""
    rs = [rec(did(i), f"3t{i}", stock=at(e), **({"twin": twin_at(0.8)} if i % 2 else {}))
          for i, e in enumerate((0.48, 0.50, 0.52, 0.2))]
    points, excluded = run(*rs)
    assert admitted(points) == [did(0), did(1), did(2)]
    [e] = excluded
    assert e["numbers"]["arm"] == "stock" and e["numbers"]["contributors"] == 4
    assert e["numbers"]["cell_median"] == pytest.approx(0.49, abs=1e-4)


def test_f5_leaves_a_fit_point_whose_twin_ran_alone():
    """Stock skipped, the twin ran: no stock speed to judge, a didn't-load row."""
    assert run(fitpoint(twin=twin_at(0.05)), baseline_floor=0.99)[1] == []


def test_records_with_and_without_the_twin_are_one_cell():
    """One contributor's runs with and without the twin arm are one vote: the
    page divides both by stock."""
    points, _ = run(rec(rkey="3old", created="2026-09-22T00:00:00+00:00"),
                    rec(rkey="3new", twin="12.36", created="2026-09-24T00:00:00+00:00"))
    assert len(points) == 1 and len(points[0]["site"]["provenance"]) == 2


# -------------------------------------------------------- the vote collapse --

def test_one_contributors_runs_of_one_cell_collapse_to_the_median():
    runs = [rec(rkey=f"3r{i}", comp=c, stock=s, created=f"2026-09-2{i}T00:00:00+00:00")
            for i, (c, s) in enumerate([("16.0", "9.0"), ("17.0", "10.0"), ("15.0", "11.0"), ("19.0", "10.4")])]
    points, excluded = run(*runs)
    assert excluded == [] and len(points) == 1
    p = points[0]
    m = {x["name"]: x for x in p["metrics"]}
    assert m["compressed_decode_tok_s"]["value"] == "16.5"   # median of 15, 16, 17, 19
    assert m["stock_decode_tok_s"]["value"] == "10.2"        # median of 9, 10, 10.4, 11
    assert "samples" not in m["compressed_decode_tok_s"]     # one run's samples no longer describe a median
    assert p["site"]["source"] == uri(rkey="3r3")            # the latest record
    assert p["site"]["provenance"] == [uri(rkey=f"3r{i}") for i in range(4)]
    assert p["site"]["contributor"] == DID_A
    assert "raw" not in p


def _with_spec(r, **values):
    return edit(r, lambda v: v["metrics"].extend(
        {"name": n, "value": x, "unit": "tok/s"} for n, x in values.items()))


def test_speculation_metrics_pass_through_and_a_record_without_them_is_unchanged():
    names = [f"{a}_spec_decode_tok_s_{p}" for a in ("compressed", "stock") for p in ("agent", "chat", "code")]
    with_m = _with_spec(rec(), **{n: f"{i + 1}.5" for i, n in enumerate(names)})
    plain = rec(did=DID_B, rkey="3bbb")
    points, excluded = run(with_m, plain)
    assert excluded == [] and len(points) == 2
    got = {p["site"]["contributor"]: {m["name"]: m["value"] for m in p["metrics"]} for p in points}
    assert {n: got[DID_A][n] for n in names} == {n: f"{i + 1}.5" for i, n in enumerate(names)}
    assert not any("spec" in n for n in got[DID_B])


def test_a_later_run_without_speculation_does_not_drop_an_earlier_runs_metrics():
    a = _with_spec(rec(rkey="3s1", created="2026-09-20T00:00:00+00:00"), compressed_spec_decode_tok_s_chat="10")
    b = _with_spec(rec(rkey="3s2", created="2026-09-21T00:00:00+00:00"), compressed_spec_decode_tok_s_chat="14")
    c = rec(rkey="3s3", created="2026-09-22T00:00:00+00:00")
    points, _ = run(a, b, c)
    m = {x["name"]: x["value"] for x in points[0]["metrics"]}
    assert m["compressed_spec_decode_tok_s_chat"] == "12"


def test_two_contributors_of_one_cell_are_two_points():
    points, _ = run(rec(DID_A, "3a"), rec(DID_B, "3b", comp="18.0"))
    assert len(points) == 2 and {p["site"]["contributor"] for p in points} == {DID_A, DID_B}


@pytest.mark.parametrize("kw", [dict(name="Qwen3-4B"), dict(device="Radeon RX 7600 XT"), dict(mem=68_000_000_000),
                                dict(profile="gulp"), dict(platform="cuda"), dict(outcome="skipped_predicted_nonfit")])
def test_one_contributor_in_two_cells_is_two_points(kw):
    """D4: many models or different machines per person all count."""
    points, _ = run(rec(rkey="3a"), rec(rkey="3b", **kw))
    assert len(points) == 2


def test_a_single_record_keeps_its_samples_and_is_its_own_provenance():
    p = run(rec())[0][0]
    assert p["site"]["provenance"] == [uri()] and "samples" in p["metrics"][0]


def test_the_same_at_uri_twice_is_one_record():
    points, _ = run(rec(comp="16.0"), rec(comp="17.0"))
    assert len(points) == 1 and points[0]["metrics"][0]["value"] == "17.0"


# --------------------------------------------------------------- output --

def test_the_document_is_build_points_bundle_plus_generated_at_and_cursor():
    points, excluded = run(rec(), rec(rkey="3m", verified=252, swapped=253))
    doc = pipeline.document(points, excluded, dt.datetime(2026, 9, 23, 21, 0, tzinfo=dt.timezone.utc), cursor=42)
    shape = bp.bundle(points, excluded)
    assert set(shape) <= set(doc) and doc["records"] == points and doc["excluded"] == excluded
    assert doc["version"] == shape["version"] and doc["lexicon"] == shape["lexicon"]
    assert doc["generatedAt"] == "2026-09-23T21:00:00Z" and doc["cursor"] == 42 and doc["skipped"] == []


def test_write_is_atomic_and_keeps_a_dated_copy(tmp_path, monkeypatch):
    out = tmp_path / "data" / "points.json"
    doc = pipeline.document([], [], dt.datetime(2026, 9, 23, 23, 59, tzinfo=dt.timezone.utc))
    dated = pipeline.write_atomic(out, doc)
    assert dated == tmp_path / "data" / "points-2026-09-23.json"
    assert json.loads(out.read_text()) == doc == json.loads(dated.read_text())
    assert sorted(p.name for p in out.parent.iterdir()) == ["points-2026-09-23.json", "points.json"]

    # every write goes through <name>.tmp and a rename; a failure before the
    # rename leaves the old file whole
    replaced = []
    real = os.replace
    monkeypatch.setattr(pipeline.os, "replace", lambda a, b: (replaced.append((str(a), str(b))), real(a, b)))
    doc2 = {**doc, "cursor": 7}
    pipeline.write_atomic(out, doc2)
    assert (str(out) + ".tmp", str(out)) in replaced
    assert json.loads(out.read_text())["cursor"] == 7

    def boom(a, b):
        raise OSError("disk full")
    monkeypatch.setattr(pipeline.os, "replace", boom)
    with pytest.raises(OSError):
        pipeline.write_atomic(out, {**doc, "cursor": 8})
    assert json.loads(out.read_text())["cursor"] == 7
