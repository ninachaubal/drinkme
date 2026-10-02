"""The results site's comparison identity.

Grouping plotted records by device + model name only would put an
old-version `sip` record of one checkpoint and a current-version `gulp`
record of another in one cell, n = 2, one combined median ratio. So
site/data/build_points.py validates each record and applies
docs/versioning.md's compatibility policy (the same major version),
naming every exclusion; and the page groups by the full identity —
version compatibility + device + hfRepo + resolved revision +
compression profile + platform — with labels that show what differs.
Each identity field is varied here, one at a time, and the
page's grouping block is lifted out of index.html and executed under
Node. The record lexicon is untouched: only fields records carry today
are read.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "site" / "index.html"

spec = importlib.util.spec_from_file_location("build_points", ROOT / "site" / "data" / "build_points.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

CUR = getattr(bp, "CURRENT_VERSION", None) or __import__("drinkme").__version__


def rec(version=CUR, repo="Qwen/Qwen3-8B", rev="aaaa1111", profile="sip", platform="rocm",
        comp="20", stock="10", read="200", device="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", name="Qwen3-8B",
        outcome="measured", **over):
    r = {"$type": "wtf.petrichor.drinkme.measurement", "createdAt": "2026-09-20T00:00:00Z",
         "version": version, "model": {"name": name, "hfRepo": repo, "revision": rev},
         "compression": {"profile": profile, "bitsPerWeight": "12.0"},
         "environment": {"deviceClass": device, "platform": platform, "memoryBytes": 128_000_000_000,
                         "memoryKind": "unified"},
         "stock": {"outcome": outcome} if outcome == "measured" else {"outcome": outcome, "error": "x"},
         "metrics": [{"name": "compressed_decode_tok_s", "value": comp, "unit": "tok/s"},
                     {"name": "stock_decode_tok_s", "value": stock, "unit": "tok/s"},
                     {"name": "read_gb_s", "value": read, "unit": "GB/s"},
                     {"name": "stock_weights_gb", "value": "16", "unit": "GB"},
                     {"name": "compressed_weights_gb", "value": "12", "unit": "GB"}]}
    for k, v in over.items():
        r[k] = v
    return r


# ------------------------------------------------------------ the build --


def _build(tmp_path, records):
    paths = []
    for i, r in enumerate(records):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(r))
        paths.append(str(p))
    return bp.build(paths, root=tmp_path)


def test_the_build_excludes_an_incompatible_version_and_names_it(tmp_path):
    """The reproduction's first half: the old-version sip record is not
    bundled; the current gulp record is; the exclusion says why."""
    old = rec(version="0.1.0", profile="sip")
    cur = rec(version=CUR, repo="Qwen/Qwen3-8B-other", rev="bbbb2222", profile="gulp")
    recs, excluded = _build(tmp_path, [old, cur])
    assert [r["model"]["hfRepo"] for r in recs] == ["Qwen/Qwen3-8B-other"]
    assert len(excluded) == 1 and excluded[0]["source"] == "r0.json"
    assert "not comparable with drinkme " + CUR + " (same major version)" in excluded[0]["reason"]


def test_compatible_is_the_documented_policy():
    assert bp.compatible("1.0.0", "1.0.0") and bp.compatible("1.0.1", "1.0.0")
    assert bp.compatible("1.3.0", "1.0.0") and bp.compatible("1.0.0", "1.9.2")
    assert not bp.compatible("2.0.0", "1.9.2") and not bp.compatible("1.9.2", "2.0.0")
    # a lower major compares with nothing of this one either
    assert not bp.compatible("0.2.0", "1.0.0") and not bp.compatible("0.9.0", "1.0.0")
    assert not bp.compatible("garbage", "1.0.0") and not bp.compatible(None, "1.0.0")


def test_this_drinkme_admits_its_own_major_only():
    assert bp.compatible(CUR, CUR)
    major = int(CUR.split(".")[0])
    assert bp.compatible(f"{major}.99.0", CUR)
    assert not bp.compatible(f"{major + 1}.0.0", CUR) and not bp.compatible("0.2.0", CUR)


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r.pop("version"), "missing or non-string version"),
    (lambda r: r.__setitem__("version", "v2"), "not a semver"),
    (lambda r: r["model"].pop("revision"), "missing or non-string model.revision"),
    (lambda r: r["model"].pop("hfRepo"), "model.hfRepo"),
    (lambda r: r["environment"].pop("platform"), "environment.platform"),
    (lambda r: r.pop("compression"), "without compression.profile"),
    (lambda r: r["metrics"].pop(2), "missing metrics ['read_gb_s']"),
    (lambda r: r.pop("stock"), "not a new-shape record"),
])
def test_the_build_refuses_a_record_missing_an_identity_field(tmp_path, mutate, needle):
    r = rec()
    mutate(r)
    recs, excluded = _build(tmp_path, [r, rec()])
    assert len(recs) == 1 and len(excluded) == 1
    assert needle in excluded[0]["reason"], excluded[0]["reason"]


def test_a_non_measured_record_needs_no_profile_or_metrics(tmp_path):
    r = rec(outcome="failed_load")
    del r["compression"]
    r["metrics"] = []
    recs, excluded = _build(tmp_path, [r])
    assert len(recs) == 1 and excluded == []


def test_the_bundle_carries_its_version_and_its_exclusions(tmp_path):
    recs, excluded = _build(tmp_path, [rec(version="0.0.1"), rec()])
    out = tmp_path / "points.json"
    bp.write(out, recs, excluded)
    d = json.loads(out.read_text())
    assert d["version"] == CUR and len(d["records"]) == 1
    assert d["excluded"][0]["version"] == "0.0.1"
    assert d["lexicon"] == "lexicons/wtf.petrichor.drinkme.measurement.json"


# ------------------------------------------------ the page's grouping --


def _grouping_js() -> str:
    """The page's grouping, lifted out of index.html: the marked pure block
    (groupCells + identity + labels), or — on a tree without the markers —
    boot()'s inline grouping loop wrapped as groupCells."""
    html = INDEX.read_text()
    helpers = "".join(re.search(pat, html).group(0) for pat in (
        r"const median = .*?;\n", r"function metric\(r, name\) \{.*?\}\n",
        r"const DEVICE_LABELS = .*?;\n", r"const deviceLabel = .*?;\n"))
    block = re.search(r"// -- grouping \(pure.*?\n(.*?)// -- end grouping --", html, re.S)
    if block:
        return helpers + block.group(1)
    loop = re.search(r"\nfunction boot\(\) \{\n  const groups(.*?)\n  cells\.sort", html, re.S)
    assert loop, "index.html has neither the grouping markers nor boot()'s inline loop"
    return (helpers + "function groupCells(records) { const DATA = {records}, fits = [], cells = [];\n"
            "  const groups" + loop.group(1) + "\n  for (const c of cells) c.label = "
            "`${deviceLabel(c.box)} · ${c.model} · BF16`;\n  return cells; }\n")


def group_under_node(records):
    """Run the site's own grouping over `records` under Node and return the cells (label, n, ratio, identity)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = (_grouping_js() + "\nconst cells = groupCells(" + json.dumps(records) + ");\n"
          "console.log(JSON.stringify(cells.map(c => ({label: c.label, short: c.short, n: c.n, "
          "ratio: c.ratio, hfRepo: c.hfRepo, revision: c.revision, profile: c.profile, "
          "platform: c.platform, version: c.version, device: c.device, model: c.model}))));\n")
    r = subprocess.run([node, "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_two_records_of_different_checkpoints_are_two_cells_under_node():
    """The reproduction's second half: the page itself, given the two
    records the build would admit if they were both current, keeps them
    apart and labels what differs."""
    a = rec(profile="sip")
    b = rec(repo="Qwen/Qwen3-8B-other", rev="bbbb2222", profile="gulp")
    cells = group_under_node([a, b])
    assert [c["n"] for c in cells] == [1, 1]  # never one cell, n = 2, ratio 2
    assert sorted(c["ratio"] for c in cells) == [2.0, 2.0]
    assert {c["label"] for c in cells} == {"Strix Halo · Max+ 395 · Qwen3-8B · BF16 sip",
                                           "Strix Halo · Max+ 395 · Qwen3-8B · BF16 gulp"}


def test_identical_identity_is_one_cell_with_a_median():
    cells = group_under_node([rec(comp="20", stock="10"), rec(comp="40", stock="10"), rec(comp="30", stock="10")])
    assert len(cells) == 1 and cells[0]["n"] == 3 and cells[0]["ratio"] == 3.0
    assert cells[0]["label"] == "Strix Halo · Max+ 395 · Qwen3-8B · BF16 sip" and cells[0]["short"] == ""


@pytest.mark.parametrize("field, other, shown", [
    ("rev", "bbbb2222", "@bbbb222"),
    ("repo", "Qwen/Qwen3-8B-fork", "Qwen/Qwen3-8B-fork"),
    ("profile", "gulp", "BF16 gulp"),
    ("platform", "cuda", "cuda"),
    ("version", "2.0.0", "v2"),  # two majors in one bundle (the build would exclude one; the page still splits)
    ("device", "RX 7900 XTX", "RX 7900 XTX"),
])
def test_each_identity_field_splits_the_cell_and_is_visible_in_the_label(field, other, shown):
    base = rec(version="1.0.0")
    varied = rec(**{"version": "1.0.0", field: other})
    cells = group_under_node([base, varied])
    assert [c["n"] for c in cells] == [1, 1], cells
    labels = [c["label"] for c in cells]
    assert len(set(labels)) == 2, labels
    assert any(shown in lb for lb in labels), (shown, labels)
    if field != "device":
        assert all(c["short"] for c in cells), cells  # the chart's point labels differ too


def test_the_minors_of_one_major_are_one_cell():
    """Minors and patches compare: versionKey folds 1.3.0 and 1.4.1 together,
    and keeps another major apart."""
    cells = group_under_node([rec(version="1.3.0"), rec(version="1.4.1")])
    assert len(cells) == 1 and cells[0]["n"] == 2
    cells = group_under_node([rec(version="1.4.0"), rec(version="2.0.0")])
    assert len(cells) == 2


def test_a_non_measured_record_is_not_a_cell():
    cells = group_under_node([rec(outcome="failed_load")])
    assert cells == []


def test_an_old_shape_record_with_the_detectors_environment_notes_still_builds(tmp_path):
    """Records published before the detector's working notes moved to
    raw.detector carry them in `environment` (budgetGB, heuristic,
    evidence, detectorVersion), which the lexicon never declared. The
    build still reads such a record rather than crashing on it."""
    old = rec()
    old["environment"].update({"budgetGB": "124.0", "heuristic": True,
                               "evidence": ["pci id table: 0x1586"], "detectorVersion": 2})
    recs, excluded = _build(tmp_path, [old])
    assert excluded == [] and len(recs) == 1
    assert recs[0]["environment"]["deviceClass"] == old["environment"]["deviceClass"]


def test_a_record_wrapped_in_another_object_is_refused_by_name(tmp_path):
    """The build reads the record file as `drinkme bench` writes it. The
    `{"receipt": record}` wrapping is not read: such a file is
    excluded with its reason, not plotted and not a crash."""
    recs, excluded = _build(tmp_path, [{"receipt": rec()}])
    assert recs == [] and len(excluded) == 1
    assert excluded[0]["source"] == "r0.json" and "not a new-shape record" in excluded[0]["reason"]


# ------------------------------------------- the chart's frame and table --


def page_js(records, expr):
    """Evaluate `expr` under Node after the page's pure block, with
    `records` and `cells = groupCells(records)` in scope; return its JSON."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = (_grouping_js() + "\nconst records = " + json.dumps(records) + ";\n"
          "const cells = groupCells(records);\n"
          "console.log(JSON.stringify(" + expr + "));\n")
    r = subprocess.run([node, "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


STRIX = "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"
RX = "Radeon RX 7600 XT"


def nonfit(**kw):
    """A record whose uncompressed arm the fit check skipped: no stock_* metrics."""
    r = rec(outcome="skipped_predicted_nonfit", **kw)
    r["metrics"] = [m for m in r["metrics"] if not m["name"].startswith("stock_")]
    return r


@pytest.mark.parametrize("records", [
    # the dev records' spread: the 1.7B's 2.75x sat above the old fixed frame (ymax 2.6)
    [rec(name="Qwen3-1.7B", repo="Qwen/Qwen3-1.7B", comp="44.02", stock="16.03", read="239.193"),
     rec(name="Qwen3-0.6B", repo="Qwen/Qwen3-0.6B", comp="85.24", stock="41.49", read="238.48"),
     rec(name="Qwen3-4B", repo="Qwen/Qwen3-4B", profile="gulp", comp="12.48", stock="16.2", read="280.902", device=RX),
     rec(name="Qwen3-4B", repo="Qwen/Qwen3-4B", profile="gulp", comp="16.77", stock="16.17", read="233.931")],
    # one point: the frame still has width and height
    [rec(comp="30", stock="10", read="500")],
    # a wide spread: an APU beside a discrete GPU, and a ratio far below 1x
    [rec(read="120", comp="50", stock="10"), rec(read="1008", comp="3", stock="12", device="RTX 4090", platform="cuda")],
])
def test_every_plotted_point_lies_inside_the_axis_domain(records):
    out = page_js(records, "{dom: domainOf(cells), pts: cells.map(c => [c.x, c.ratio]), "
                           "tx: [tx(domainOf(cells).x[0]), tx(domainOf(cells).x[1]), ...cells.map(c => tx(c.x))], "
                           "xt: niceTicks(domainOf(cells).x[0], domainOf(cells).x[1], 5), "
                           "yt: niceTicks(domainOf(cells).y[0], domainOf(cells).y[1], 5, 1, true)}")
    (x0, x1), (y0, y1) = out["dom"]["x"], out["dom"]["y"]
    t0, t1, *tpts = out["tx"]
    assert out["pts"]
    assert y0 >= 0   # a speed ratio is never negative, so neither is the axis
    for (x, y), t in zip(out["pts"], tpts):
        # strictly inside, with room for a marker at the edge: 1.5% of the span, measured on each axis's own
        # scale (x: the page's power transform tx; y: linear)
        m, n = (t1 - t0) * 0.015, (y1 - y0) * 0.015
        assert t0 + m < t < t1 - m, (x, out["dom"])
        assert y0 + n < y < y1 - n, (y, out["dom"])
    assert y0 < 1 < y1
    assert 1 in out["yt"]
    assert len(out["xt"]) >= 2 and all(x0 <= v <= x1 for v in out["xt"])
    assert all(y0 <= v <= y1 for v in out["yt"])


def test_a_narrow_bandwidth_range_gets_round_ticks():
    records = [rec(read="233.9"), rec(read="281.3", device=RX)]
    out = page_js(records, "(() => { const d = domainOf(cells).x; return [d, niceTicks(d[0], d[1], 5), niceTicks(d[0], d[1], 3)]; })()")
    dom, wide, phone = out
    assert 200 < dom[0] < 233.9 and 281.3 < dom[1] < 320
    # the frame pads 5% of the power-transformed span (fitX), so 230 sits just outside it
    assert wide == [240, 250, 260, 270, 280]
    assert phone == [240, 260, 280]
    # the page's own x ticks: only 250 of the fixed set is inside, so it falls back to these round ones
    assert page_js(records, "(() => { const d = domainOf(cells).x; return X_TICKS.filter((v) => v >= d[0] && v <= d[1]); })()") == [250]


def test_a_nonfit_record_is_a_table_row_and_not_a_point():
    """Qwen3-8B on the RX 7600 XT: the fit check skipped uncompressed BF16,
    so there is no ratio to plot, but the table says it didn't load."""
    measured = rec(name="Qwen3-4B", repo="Qwen/Qwen3-4B", device=RX, comp="19.56", stock="16.28", read="281.187")
    skipped = nonfit(device=RX, comp="13.45", read="281.083")
    out = page_js([measured, skipped],
                  f"{{cells: cells.map(c => c.model), rows: rowsFor(records, cells, {json.dumps(RX)}, 'sip')"
                  ".map(r => ({name: r.name, point: r.cell != null, ratio: r.ratio, stock: r.stock, tok: r.tok}))}")
    assert out["cells"] == ["Qwen3-4B"]
    assert [r["name"] for r in out["rows"]] == ["Qwen3-4B", "Qwen3-8B"]
    four, eight = out["rows"]
    assert four["point"] and four["ratio"] == pytest.approx(19.56 / 16.28)
    assert not eight["point"] and eight["ratio"] is None and eight["stock"] is None
    assert eight["tok"] == pytest.approx(13.45)


def test_the_table_keeps_to_one_machine_and_profile_smallest_first():
    small = rec(name="Qwen3-0.6B", repo="Qwen/Qwen3-0.6B")
    small["metrics"][4]["value"] = "0.97"
    records = [rec(), small, rec(profile="gulp"), rec(device=RX)]
    out = page_js(records, f"rowsFor(records, cells, {json.dumps(STRIX)}, 'sip').map(r => r.name)")
    assert out == ["Qwen3-0.6B", "Qwen3-8B"]


def with_twin(r, tok, gb="16.03"):
    """`r` with bench's twin arm: records carry it as data, the page never reads it."""
    r["metrics"] += [{"name": "twin_decode_tok_s", "value": tok, "unit": "tok/s"},
                     {"name": "twin_weights_gb", "value": gb, "unit": "GB"}]
    return r


def test_the_ratio_divides_by_stock_even_when_a_record_carries_the_twin():
    """Qwen3-8B on Strix Halo, quiet: compressed 16.06, stock 10.86,
    twin 13.79. The point and the row are compressed / stock, and the row's
    uncompressed is the stock arm."""
    r = with_twin(rec(comp="16.06", stock="10.86", read="238.976"), "13.79")
    out = page_js([r], f"[cells.map(c => c.ratio), rowsFor(records, cells, {json.dumps(STRIX)}, 'sip')"
                       ".map(r => [r.stock, r.ratio])]")
    assert out[0] == [pytest.approx(16.06 / 10.86)]
    assert out[1] == [[pytest.approx(10.86), pytest.approx(16.06 / 10.86)]]


def test_a_cell_medians_every_records_stock_ratio_twin_or_not():
    """Two contributors in one cell, one whose record carries the twin: one
    baseline, so both records' compressed / stock are medianed."""
    records = [by("did:plc:a", comp="16.1", stock="10.8"), by("did:plc:b", comp="16.0", stock="10.9")]
    with_twin(records[0], "13.8")
    out = page_js(records, "cells.map(c => [c.ratio, c.n])")
    assert out == [[pytest.approx((16.1 / 10.8 + 16.0 / 10.9) / 2), 2]]


def test_a_fit_point_whose_twin_ran_is_a_row_not_a_point():
    """Stock skipped, the twin ran: no stock speed, so no ratio; the table
    says the uncompressed arm didn't load."""
    r = with_twin(nonfit(device=RX, comp="13.45", read="281.083"), "10.0")
    out = page_js([r], f"[cells.length, rowsFor(records, cells, {json.dumps(RX)}, 'sip').map(r => [r.ratio, r.stock])]")
    assert out == [0, [[None, None]]]


def test_the_key_is_the_legend_alone_and_the_page_has_one_baseline():
    """The key under the chart is its legend and nothing else: the notes on
    what uncompressed means and on run-to-run spread came off the landing
    page (they live in docs/bench.md). The page has no other
    baseline: bench's diagnostic twin arm is not on it."""
    html = INDEX.read_text()
    key = re.search(r"\nfunction drawKey\(any\) \{\n(.*?)\n\}\n", html, re.S).group(1)
    assert "key-note" not in key
    assert "BASELINE_NOTE" not in html and "SPREAD_NOTE" not in html
    assert "twin" not in html.lower()


def test_the_default_machine_has_the_most_records():
    records = [rec(device=RX), rec(), rec(profile="gulp"), nonfit()]
    out = page_js(records, "machinesOf(records).map(m => [m.label, m.n, m.profiles])")
    assert out == [["Strix Halo · Max+ 395", 3, ["sip", "gulp"]], ["Radeon RX 7600 XT", 1, ["sip"]]]


def by(did, **kw):
    """A record as the aggregator writes it: collapsed, and naming its contributor."""
    r = rec(**kw)
    r["site"] = {"contributor": did, "source": f"at://{did}/wtf.petrichor.drinkme.measurement/{len(kw)}"}
    return r


def test_a_cells_count_is_its_distinct_contributors():
    """Two people are two contributors; one person's two boxes of different
    memory are two records (two votes) but one contributor; a record
    without site.contributor counts as its own."""
    records = [by("did:plc:a"), by("did:plc:b", comp="22"), by("did:plc:c", device=RX),
               by("did:plc:c", device=RX, comp="18"),
               rec(name="Qwen3-4B", repo="Qwen/Qwen3-4B"), rec(name="Qwen3-4B", repo="Qwen/Qwen3-4B", comp="21")]
    out = page_js(records, "cells.map(c => [c.device, c.model, c.n, c.contributors])")
    assert sorted(map(tuple, out)) == sorted([(STRIX, "Qwen3-8B", 2, 2), (RX, "Qwen3-8B", 2, 1),
                                              (STRIX, "Qwen3-4B", 2, 2)])
    assert page_js([], "[1, 2].map(contributorsText)") == ["1 contributor", "2 contributors"]


def test_the_tables_rows_and_the_tooltip_carry_the_contributor_count():
    records = [by("did:plc:a"), by("did:plc:b", comp="22"), nonfit(device=STRIX, name="Qwen3-32B", repo="Qwen/Qwen3-32B")]
    out = page_js(records, f"rowsFor(records, cells, {json.dumps(STRIX)}, 'sip').map(r => [r.name, r.contributors])")
    assert sorted(out) == [["Qwen3-32B", 1], ["Qwen3-8B", 2]]
    html = INDEX.read_text()
    body = re.search(r"\nfunction renderTable\(\) \{\n(.*?)\n\}\n", html, re.S).group(1)
    assert "contributorsText(row.contributors)" in body
    tip = re.search(r"\nfunction tipText\(c\) \{\n(.*?)\n\}\n", html, re.S).group(1)
    assert "contributorsText(c.contributors)" in tip


def test_the_default_machine_has_the_most_contributors_then_strix_halo():
    # one person's three Strix records against two people on the RX: the RX leads
    records = [by("did:plc:a"), by("did:plc:a", profile="gulp"), by("did:plc:a", name="Qwen3-4B", repo="Qwen/Qwen3-4B"),
               by("did:plc:b", device=RX), by("did:plc:c", device=RX)]
    out = page_js(records, "machinesOf(records).map(m => [m.device, m.contributors, m.n])")
    assert out == [[RX, 2, 2], [STRIX, 1, 3]]
    # a tie goes to Strix Halo, even with fewer records
    records = [by("did:plc:a"), by("did:plc:b", device=RX), by("did:plc:b", device=RX, name="Qwen3-4B", repo="Qwen/Qwen3-4B")]
    assert page_js(records, "machinesOf(records)[0].device") == STRIX


PCIE, SXM4 = "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"


def test_the_a100_variants_are_one_machine_but_never_one_cell():
    # Modal's A100 80GB pool hands out both; the picker shows one machine with both models
    records = [rec(device=PCIE, platform="cuda", name="Qwen3.8-27B", repo="Qwen/Qwen3.8-27B", read="1667"),
               rec(device=SXM4, platform="cuda", read="1734")]
    assert page_js(records, "machinesOf(records).map(m => [m.device, m.n])") == [["NVIDIA A100 80GB", 2]]
    rows = page_js(records, "rowsFor(records, cells, 'NVIDIA A100 80GB', 'sip').map(r => [r.model, r.env.deviceClass])")
    assert sorted(rows) == [["Qwen3-8B", SXM4], ["Qwen3.8-27B", PCIE]]
    # the same model on both variants stays two cells, each named by its own deviceClass
    records.append(rec(device=PCIE, platform="cuda", read="1666"))
    out = page_js(records, "cells.filter(c => c.model === 'Qwen3-8B').map(c => [c.device, c.n, c.label])")
    assert sorted((d, n) for d, n, _ in out) == [(PCIE, 1), (SXM4, 1)]
    assert all(d in label for d, _, label in out)
    # a link naming a member's slug lands on the grouped machine
    assert page_js(records, f"machineOf({json.dumps(PCIE)}) === machineOf({json.dumps(SXM4)})") is True


# ------------------------------------------------ anchors and links --


def test_every_in_page_link_names_a_section_and_none_is_numbered():
    """The section anchors are named for their sections (#fit, #speed…),
    and every href="#…" on the page lands on an element that exists."""
    html = INDEX.read_text()
    assert not re.search(r"#p[0-9]\b|id=\"p[0-9]\"", html)
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    for section in ("intro", "fit", "speed", "start", "measure"):
        assert section in ids, section
    targets = re.findall(r'href="#([^"]+)"', html)
    assert targets and all(t in ids for t in targets), set(targets) - ids


def _heading_slugs(md: str) -> set[str]:
    """The anchors a Markdown file's headings get: lower-cased, code ticks
    and punctuation dropped, spaces to hyphens."""
    return {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")
            for h in re.findall(r"^#+ (.+)$", md, re.M)}


def test_every_repo_doc_link_names_a_file_and_heading_in_this_tree():
    """The setup links under the Caterpillar point into docs/ on main: each
    file exists here, and a #fragment is one of its headings."""
    links = re.findall(r'href="https://tangled\.org/ninachaubal\.com/drinkme/blob/main/([^"#]+)(?:#([^"]+))?"',
                       INDEX.read_text())
    assert {path for path, _ in links} >= {"docs/models.md", "docs/cli.md"}
    for path, fragment in links:
        doc = ROOT / path
        assert doc.is_file(), path
        if fragment:
            assert fragment in _heading_slugs(doc.read_text()), (path, fragment)


def test_the_setup_links_open_with_the_readme_then_the_docs():
    """The first link under the Caterpillar is the repo's own page, which
    shows README.md (present in this tree); the rest go into docs/."""
    foot = re.search(r'<p class="setup-links">(.*?)</p>', INDEX.read_text()).group(1)
    hrefs = re.findall(r'href="([^"]+)"', foot)
    assert hrefs[0] == "https://tangled.org/ninachaubal.com/drinkme"
    assert (ROOT / "README.md").is_file()
    assert all(h.startswith("https://tangled.org/ninachaubal.com/drinkme/blob/main/docs/") for h in hrefs[1:]), hrefs


def test_a_machine_link_names_the_machine_by_a_slug():
    out = page_js([], f"[machineSlug({json.dumps(STRIX)}), machineSlug({json.dumps(RX)})]")
    assert out == ["amd-ryzen-ai-max-395-w-radeon-8060s", "radeon-rx-7600-xt"]


# ------------------------------------------------------ the fit table --


def test_the_fit_table_is_the_pages_own_rows_largest_first_with_a_dash_for_a_missing_profile():
    """The fit table is hard-coded (a footprint is the model's, not the
    machine's): the rows render the same with no records at all, in order
    of uncompressed size, largest first, with a dash and no percentage for
    a profile with no record. The values are the records' (Strix Halo, drinkme 1.0.0), so every
    cell is filled; the dash is checked on its own."""
    assert page_js([], "[sizeText(null), smaller(null, 145.412)]") == ["—", None]
    out = page_js([], "FIT_ROWS.map(r => [r.model, sizeText(r.bf16), sizeText(r.sip), sizeText(r.gulp), "
                      "smaller(r.sip, r.bf16), smaller(r.gulp, r.bf16)])")
    assert out == [
        ["Qwen2.5-72B", "145.4 GB", "103.5 GB", "99.1 GB", 29, 32],
        ["Qwen3-32B", "65.5 GB", "47.1 GB", "45.2 GB", 28, 31],
        ["gemma-4-31B-it", "62.6 GB", "45.2 GB", "43.4 GB", 28, 31],
        ["Muse-Glimmer-30B", "59.6 GB", "43.0 GB", "41.5 GB", 28, 30],
        ["Qwen3.8-27B", "54.7 GB", "39.5 GB", "38.0 GB", 28, 31],
        ["MiMo-V2.6-Distill-Qwen-9B", "18.8 GB", "14.0 GB", "13.6 GB", 25, 28],
        ["Qwen3-8B", "16.4 GB", "12.2 GB", "11.7 GB", 26, 28],
        ["Qwen3-4B", "8.1 GB", "6.1 GB", "5.9 GB", 25, 27],
        ["Qwen3-1.7B", "3.4 GB", "2.7 GB", "2.7 GB", 21, 23],
        ["Qwen3-0.6B", "1.2 GB", "1.1 GB", "1.0 GB", 12, 13],
    ]
    bf16 = page_js([], "FIT_ROWS.map(r => r.bf16)")
    assert bf16 == sorted(bf16, reverse=True)


def test_the_fit_table_renders_from_its_rows_not_from_points_json():
    """renderSizes reads FIT_ROWS and runs before the fetch resolves; the
    footnote is kept."""
    html = INDEX.read_text()
    body = re.search(r"\nfunction renderSizes\(\) \{\n(.*?)\n\}\n", html, re.S).group(1)
    assert "FIT_ROWS" in body and "DATA" not in body and "records" not in body
    assert "fitRows" not in html
    boot = re.search(r"\nfunction boot\(\) \{\n(.*?)\n\}\n", html, re.S).group(1)
    assert "renderSizes" not in boot
    assert "These are the weights alone. You’ll also need room for the KV cache and runtime." in html


def test_the_speed_table_has_no_weights_column():
    """Sizes are the fit table's job; the speed table is compressed,
    uncompressed and the ratio."""
    head = re.search(r'<table class="results" id="results".*?<thead>(.*?)</thead>', INDEX.read_text(), re.S).group(1)
    assert [re.sub(r"<[^>]+>", "", h) for h in re.findall(r"<th[^>]*>.*?</th>", head)] == [
        "Model", "compressed", "uncompressed", "speed"]


SPEC_NAMES = [f"{arm}_spec_decode_tok_s_{p}" for arm in ("compressed", "stock") for p in ("agent", "chat", "code")]


def with_spec(r, comp=None, stock=None):
    """`r` plus per-prompt speculative metrics: comp / stock are (agent, chat, code) or None."""
    r = json.loads(json.dumps(r))
    for arm, vals in (("compressed", comp), ("stock", stock)):
        for p, v in zip(("agent", "chat", "code"), vals or ()):
            if v is not None:
                r["metrics"].append({"name": f"{arm}_spec_decode_tok_s_{p}", "value": str(v), "unit": "tok/s"})
    return r


def spec_lines(records):
    return page_js([], f"specLines({json.dumps(records)})")


def test_the_speculation_lines_are_up_to_the_best_prompt():
    r = with_spec(rec(), comp=(9.23, 10.58, 12.96), stock=(6.78, 8.38, 10.17))
    assert spec_lines([r]) == {"lines": {"compressed": "up to 13.0 tok/s", "uncompressed": "up to 10.2 tok/s"},
                               "said": "compressed"}


def test_there_is_no_speculation_ratio_line():
    # tok/s lines only. code has no stock: its 30 tok/s still counts for compressed
    r = with_spec(rec(), comp=(10, 20, 30), stock=(9, 10, None))
    assert spec_lines([r])["lines"] == {"compressed": "up to 30.0 tok/s", "uncompressed": "up to 10.0 tok/s"}


def test_without_stock_only_the_compressed_line_says_the_words():
    out = spec_lines([with_spec(rec(), comp=(22.41, 24.57, 29.07))])
    assert out == {"lines": {"compressed": "up to 29.1 tok/s", "uncompressed": None}, "said": "compressed"}


def test_a_prompt_missing_from_the_records_is_skipped():
    out = spec_lines([with_spec(rec(), comp=(None, 12, None), stock=(None, 8, None))])
    assert out["lines"] == {"compressed": "up to 12.0 tok/s", "uncompressed": "up to 8.0 tok/s"}


def test_a_row_without_the_metrics_gets_no_lines():
    assert spec_lines([rec()]) is None and spec_lines([]) is None


def test_the_row_uses_the_median_across_its_records():
    rs = [with_spec(rec(), comp=(c, c, c), stock=(5, 5, 5)) for c in (10, 12, 20)]
    assert spec_lines(rs)["lines"]["compressed"] == "up to 12.0 tok/s"


def test_a_records_speculation_metrics_pass_through_the_build_untouched(tmp_path):
    with_metrics = with_spec(rec(), comp=(9.23, 10.58, 12.96), stock=(6.78, 8.38, 10.17))
    plain = rec(repo="Qwen/Qwen3-4B", rev="bbbb2222", name="Qwen3-4B")
    recs, excluded = _build(tmp_path, [with_metrics, plain])
    assert excluded == []
    by = {r["model"]["name"]: r for r in recs}
    assert [m for m in by["Qwen3-8B"]["metrics"] if "spec" in m["name"]] == [m for m in with_metrics["metrics"] if "spec" in m["name"]]
    assert {m["name"] for m in by["Qwen3-8B"]["metrics"]} >= set(SPEC_NAMES)
    assert by["Qwen3-4B"]["metrics"] == plain["metrics"]
    assert by["Qwen3-8B"]["site"]["at"]["compressed_spec_decode_tok_s_code"].startswith("/metrics/")


def test_the_page_has_no_hand_kept_speculation_table():
    html = INDEX.read_text()
    assert "SPEC_RANGES" not in html and "specText" not in html
    assert 'const SPEC_WORDS = "with speculative decoding";' in html
    body = re.search(r"\nfunction renderTable\(\) \{\n(.*?)\n\}\n", html, re.S).group(1)
    assert "specLines(row.records)" in body
    assert '[["compressed", tok], ["uncompressed", bf16]]' in body and "visually-hidden" in body
    render = re.search(r"\nfunction render\(\) \{\n(.*?)\n\}\n", html, re.S).group(1)
    assert "specLines" not in render


# ------------------------------------------------ emoji presentation --

# Characters iOS may draw as colour emoji even though they have a text form:
# the Unicode Emoji=Yes property outside plain ASCII (emoji-data.txt). A
# following U+FE0E asks for the text form instead.
EMOJI_PRONE = re.compile(
    "[©®‼⁉™ℹ↔-↙↩↪⌚⌛⌨⏏"
    "⏩-⏳⏸-⏺Ⓜ▪▫▶◀◻-◾☀-➿"
    "⤴⤵⬅-⬇⬛⬜⭐⭕〰〽㊗㊙\U0001f000-\U0001faff]"
    "(?!︎)")


def test_the_page_has_no_character_ios_draws_as_an_emoji():
    """The hero's ↗ rendered as an emoji on iPhone; arrows like ↓ → are
    text-only and fine."""
    html = INDEX.read_text()
    assert not EMOJI_PRONE.findall(html)


def test_the_hero_quotes_carroll_verbatim():
    """Alice's Adventures in Wonderland (1865), Chapter I, as in Project
    Gutenberg #11: his semicolon and curly quotes, not the page's style."""
    q = re.search(r'<figure class="carroll-quote" id="quote1">\s*<blockquote><p>(.*?)</p></blockquote>', INDEX.read_text(), re.S)
    assert q and q.group(1) == "“What a curious feeling!” said Alice; “I must be shutting up like a telescope.”"


def test_the_fit_section_quotes_the_nursery_alice_verbatim():
    """The Nursery "Alice" (1890), as in Project Gutenberg #55040: his
    _Now_ is emphasised (roman inside the italic quote), with his curly
    quotes and exclamation mark."""
    html = INDEX.read_text()
    fig = re.search(r'<figure class="carroll-quote" id="quote3">\s*<blockquote><p>(.*?)</p></blockquote>\s*'
                    r'<figcaption>(.*?)</figcaption>', html, re.S)
    assert fig and fig.group(1) == "“<em>Now</em> I’m the right size to get through the little door!”"
    assert fig.group(2) == "Lewis Carroll, <cite>The Nursery “Alice”</cite> (1890)"
    fit = re.search(r'<section[^>]*id="fit".*?</section>', html, re.S).group(0)
    assert fit.index("section-intro") < fit.index('id="quote3"') < fit.index('id="sizes"')


# ------------------------------------------------------ the Caterpillar --


def test_the_caterpillar_has_alt_text_and_every_file_it_loads_has_a_provenance_sidecar():
    """#start's Caterpillar is sourced art: each webp it can load sits in
    site/art with a sidecar naming the Commons file and the book, and the
    image says what the plate shows, like the hero's alt."""
    html = INDEX.read_text()
    start = re.search(r'<section[^>]*id="start".*?</section>', html, re.S).group(0)
    img = re.search(r'<figure class="caterpillar" id="caterpillar5">\s*<img ([^>]*)>', start, re.S).group(1)
    alt = re.search(r'alt="([^"]+)"', img).group(1)
    assert alt == ("The Caterpillar, blue, on a mushroom, smoking a hookah, with Alice peeking over the edge. "
                   "The Nursery Alice, hand-coloured by John Tenniel, 1890.")
    files = set(re.findall(r'art/(caterpillar-[\w-]+\.webp)', img))
    assert files == {"caterpillar-400.webp", "caterpillar-700.webp"}
    for name in files:
        assert (ROOT / "site" / "art" / name).is_file()
        side = json.loads((ROOT / "site" / "art" / f"{name}.json").read_text())
        assert "c06543 03.jpg" in side["prompt"] and "Nursery" in side["prompt"]
        assert side["source"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert "no generation" in side["prompt"].lower() or "nothing drawn, painted or generated" in side["prompt"].lower()


def test_the_smoke_is_decoration_behind_the_words():
    """The drawn smoke is hidden from assistive tech, sits under the
    section's words, and does nothing while the section is off-screen."""
    html = INDEX.read_text()
    start = re.search(r'<section[^>]*id="start".*?</section>', html, re.S).group(0)
    svg = re.search(r'<svg class="smoke" id="smoke5"([^>]*)>', start).group(1)
    assert 'aria-hidden="true"' in svg
    assert start.index('id="smoke5"') < start.index('class="sheet"')
    assert re.search(r"\.smoke\{[^}]*z-index:0", html) and ".install-page .sheet{z-index:1}" in html
    script = re.search(r"<script>\n// The Caterpillar's smoke.*?</script>", html, re.S).group(0)
    assert "IntersectionObserver" in script and "prefers-reduced-motion: reduce" in script
