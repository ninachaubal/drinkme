"""The model menu (suggest.py): the pick per budget, and where each row's
sizes come from. A row carries no size: its BF16 size is its checkpoint's
safetensors bytes (tests/menu_checkpoints.py here, through conftest's
checkpoint_table) and its compressed size a pack's own, else that times the
profile's ratio. Each test gets an empty DRINKME_HOME, so no pack exists
unless the test writes one."""

import dataclasses
import json
import os
import statistics

import pytest

from drinkme import suggest as sg
from drinkme.suggest import COMP_RATIO, MODELS, Model, serve_pick, sizes, suggest

# the real lookup, taken at import: conftest's checkpoint_table replaces
# suggest.checkpoint_bytes for the length of each test
REAL_CHECKPOINT_BYTES = sg.checkpoint_bytes


@pytest.fixture(autouse=True)
def _no_packs(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    monkeypatch.delenv("DRINKME_COMPRESSION_PROFILE", raising=False)
    monkeypatch.delenv("DRINKME_VISION", raising=False)


def names(s):
    return (
        s.ratio.name if s.ratio else None,
        s.fit.name if s.fit else None,
    )


def row(name):
    return next(m for m in MODELS if m.name == name)


def write_pack(m, resident_bytes, profile="sip", revision=None, **meta):
    """A pack's meta.json in `m`'s default pack directory (DRINKME_HOME)."""
    from drinkme.codec.pack import default_pack_dir

    d = default_pack_dir(m.hf_repo, revision or m.revision, profile)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"formatVersion": 1, "hfRepo": m.hf_repo, "revision": revision or m.revision,
                   "profile": profile, "residentBytes": resident_bytes, **meta}, f)
    return d


def test_showcase_lineup_leads_the_menu_presentation(): # showcase lineup
    """Qwen3.8-27B, Qwen3-8B, gemma first — the RC receipt matrix order —
    pure presentation: suggest()'s own tests
    below assert the fit-based SELECTION is unmoved by this ordering."""
    assert [m.name for m in MODELS[:3]] == ["Qwen3.8-27B", "Qwen3-8B", "gemma-4-31B-it"]


def test_8gb_vram():
    s = suggest(8, memory_gb=8, memory_kind="vram")
    assert names(s) == ("Qwen3-1.7B", "Qwen3-4B")
    assert not s.fit_knife_edge and not s.fit_honest_negative


def test_12gb_vram_knife_edge():
    # A 12 GiB card (e.g. a 12GB RTX 3060) = 12.88 GB decimal: Qwen3-8B's
    # compressed estimate, 11.91 GB (16.38 GB checkpoint x 0.727), clears the
    # card with no headroom to spare (1.0x) but not with FIT_HEADROOM's
    # 1.10x (13.10 GB) — the knife-edge itself.
    s = suggest(12.88, memory_gb=12.88, memory_kind="vram")
    assert names(s) == ("Qwen3-4B", "Qwen3-8B")
    assert s.fit_knife_edge
    assert any("an estimated 11.91GB into 12.88GB" in n for n in s.notes)


def test_16gb_vram_validated_on_a_7600xt():
    # RX 7600 XT 15.98GB (16 GiB card = 17.18 GB decimal): stock Qwen3-8B
    # OOMed, compressed (11.77GB) ran at 11.92 tok/s — the validated fit row.
    s = suggest(17.18, memory_gb=17.18, memory_kind="vram")
    assert names(s) == ("Qwen3-4B", "Qwen3-8B")
    assert not s.fit_knife_edge


def test_24gb_vram():
    # 24 GiB card = 25.77 GB decimal.
    s = suggest(25.77, memory_gb=25.77, memory_kind="vram")
    assert names(s) == ("Qwen3-8B", "Qwen3-14B")


def test_48gb_vram_fit_is_the_27b_either_gate_state():
    # The 27B (compressed estimate 40.73 GB; its sip record held 39.25) is
    # the 48GB fit pick with clean headroom, where gemma would be a
    # knife-edge pick.
    s = suggest(48, memory_gb=48, memory_kind="vram", gemma_ok=True)
    assert names(s) == ("Qwen3-14B", "Qwen3.8-27B")
    assert not s.fit_knife_edge


def test_48gb_vram_gate_closed_same_fit():
    s = suggest(48, memory_gb=48, memory_kind="vram", gemma_ok=False)
    assert s.ratio.name == "Qwen3-14B"
    # Before the 27B joined the menu this tier had NO ungated fit at all.
    assert s.fit.name == "Qwen3.8-27B"


def test_mac_8gb_fit_cliff():
    # M1 Air 8GB, default working set ~5.4GB: 4B compressed (5.90 GB
    # estimate) exceeds the budget but not the machine — the honest-negative
    # cliff experiment.
    s = suggest(5.4, memory_gb=8, memory_kind="unified")
    assert names(s) == ("Qwen3-1.7B", "Qwen3-4B")
    assert s.fit_honest_negative


def test_mac_64gb():
    s = suggest(48, memory_gb=64, memory_kind="unified", gemma_ok=True)
    assert names(s) == ("Qwen3-14B", "Qwen3.8-27B")


def test_128gb_unified_validated_395():
    # The 395: 128 GiB installed RAM (137.44 GB decimal), 124 GiB GTT budget
    # (133.14 GB decimal). The 72B's sip pack held 103.32 GB.
    s = suggest(133.14, memory_gb=137.44, memory_kind="unified", gemma_ok=True)
    assert names(s) == ("Qwen3-32B", "Qwen2.5-72B")
    assert not s.fit_knife_edge
    assert [m.name for m in s.ratio_extra] == ["gemma-4-31B-it"]


def test_gated_model_never_in_menu_when_closed():
    for budget in (8, 48, 110, 124):
        s = suggest(budget, gemma_ok=False)
        for m in [s.ratio, s.fit, *s.ratio_extra]:
            assert m is None or m.name != "gemma-4-31B-it"


def test_estimates_are_flagged():
    s = suggest(25.77, memory_gb=25.77, memory_kind="vram")
    assert any("Qwen3-14B" in n and "estimate" in n for n in s.notes)


@pytest.mark.parametrize("name", ["Muse-Glimmer-30B", "MiMo-V2.6-Distill-Qwen-9B"])
def test_a_row_not_auto_eligible_resolves_by_name_but_never_picks_itself(name):
    """Muse-Glimmer-30B: on the menu so `bench --model` /
    `serve --model` resolve it, off every automatic pick — a little larger
    than the 27B showcase, it would otherwise outrank it on the very budgets
    the tests above validate. MiMo-V2.6-Distill-Qwen-9B is the
    same kind of row: it would otherwise take the 8B's seat on a 16-24 GB
    budget."""
    from drinkme.cli import resolve_model

    g = row(name)
    assert not g.auto_eligible and g.revision
    assert resolve_model(name, "serve") == (g.hf_repo, g.revision)
    assert resolve_model(g.hf_repo, "serve") == (g.hf_repo, g.revision)
    for budget in (8, 48, 52, 110, 124):
        for gemma_ok in (False, True):
            s = suggest(budget, memory_gb=128, memory_kind="unified", gemma_ok=gemma_ok)
            assert g not in (s.ratio, s.fit) and g not in s.ratio_extra
            assert serve_pick(budget, gemma_ok=gemma_ok)[0] is not g


def test_auto_eligible_is_an_explicit_opt_in():
    """auto_eligible defaults to False, so a new menu row stays out of the
    automatic pick until marked. The rows left out today, by name."""
    assert not Model("x", "o/x", None).auto_eligible
    assert {m.name for m in MODELS if not m.auto_eligible} == {
        "Muse-Glimmer-30B", "granite-4.2-8b", "MiMo-V2.6-Distill-Qwen-9B"}


# ---- where a row's sizes come from ----


def test_a_menu_row_carries_no_size():
    assert {f.name for f in dataclasses.fields(Model)} == {
        "name", "hf_repo", "revision", "gated", "auto_eligible", "model_type"}


@pytest.mark.parametrize("m", MODELS, ids=lambda m: m.name)
def test_before_a_pack_the_compressed_size_is_the_checkpoint_times_the_ratio(m, checkpoint_table):
    n = checkpoint_table[m.hf_repo]
    sz = sizes(m)
    assert sz.bf16_gb == n / 1e9 and sz.profile == "sip"
    assert sz.comp_gb == pytest.approx(n * COMP_RATIO["sip"] / 1e9)
    assert sz.estimated and sz.pack_dir is None
    assert sz.comp_label().endswith(f"(estimate: {n / 1e9:.2f} GB checkpoint x 0.727, the sip ratio)")
    # the whole checkpoint counts, tower included, images on or off
    assert sizes(m, vision=False) == sz


def test_the_ratio_is_the_profile_a_pack_would_be_written_at(monkeypatch, checkpoint_table):
    m = row("Qwen3-8B")
    n = checkpoint_table[m.hf_repo]
    assert sizes(m, profile="gulp").comp_gb == pytest.approx(n * 0.700 / 1e9)
    monkeypatch.setenv("DRINKME_COMPRESSION_PROFILE", "gulp")
    assert sizes(m).profile == "gulp" and sizes(m).comp_gb == pytest.approx(n * 0.700 / 1e9)
    # balanced has no record: it is charged sip's ratio, the larger
    assert sizes(m, profile="balanced").comp_gb == pytest.approx(n * 0.727 / 1e9)


def test_each_ratio_is_its_profiles_median_over_the_1_0_0_records(checkpoint_table):
    """COMP_RATIO against the records its comment names, the bundle's 1.0.0
    records (the ones a pack was built for, so they carry residentBytes):
    each profile's median, ONE value per model, of residentBytes over the
    checkpoint's bytes, to the constant's three places. A model's pack is the
    same bytes on every machine, so its records must agree; a second machine's
    record of a model is not a second sample."""
    path = os.path.join(os.path.dirname(__file__), "..", "site", "data", "points.json")
    with open(path) as f:
        records = [r for r in json.load(f)["records"] if "residentBytes" in r["compression"]]
    assert records, "the bundle no longer holds records with residentBytes"
    by = {}
    for r in records:
        c, repo = r["compression"], r["model"]["hfRepo"]
        by.setdefault(c["profile"], {}).setdefault(repo, set()).add(c["residentBytes"] / checkpoint_table[repo])
    for profile, models in by.items():
        for repo, ratios in models.items():
            assert len(ratios) == 1, f"{profile} {repo}: records disagree on residentBytes"
    medians = {p: round(statistics.median(next(iter(v)) for v in m.values()), 3) for p, m in by.items()}
    assert medians == COMP_RATIO
    assert (len(by["sip"]), len(by["gulp"])) == (10, 10)


def test_once_a_pack_exists_its_own_size_is_used():
    m = row("Qwen3-8B")
    d = write_pack(m, 12_006_244_888)
    sz = sizes(m)
    assert sz.comp_gb == 12.006244888 and not sz.estimated and sz.pack_dir == d
    assert sz.bf16_gb == 16.38147072  # still the checkpoint's
    assert sz.comp_label() == "12.01 GB (the sip pack's own size)"


def test_a_packs_size_leaves_its_tower_out_when_images_are_off():
    m = row("Muse-Glimmer-30B")
    write_pack(m, 42_899_585_080, vision={"residentBytes": 2_726_680_032})
    assert sizes(m).comp_gb == 42.89958508
    assert sizes(m, vision=False).comp_gb == pytest.approx(40.172905048)


@pytest.mark.parametrize("why, pack", [
    ("another profile's pack", dict(profile="gulp")),
    ("a pack of another repo", dict(hfRepo="o/other")),
    ("a pack this build refuses", dict(formatVersion=99)),
    ("a pack without residentBytes", dict(residentBytes=None)),
])
def test_a_pack_that_is_not_this_rows_leaves_the_estimate(why, pack):
    from drinkme.codec.pack import default_pack_dir

    m = row("Qwen3-8B")
    d = default_pack_dir(m.hf_repo, m.revision, "sip")
    os.makedirs(d)
    meta = {"formatVersion": 1, "hfRepo": m.hf_repo, "revision": m.revision, "profile": "sip",
            "residentBytes": 11_000_000_000, **pack}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    assert sizes(m).estimated, why


def test_a_gulp_pack_sizes_the_gulp_profile():
    m = row("Qwen3-8B")
    d = write_pack(m, 11_575_275_304, profile="gulp")
    assert sizes(m, profile="gulp").pack_dir == d and sizes(m).estimated


def test_an_explicit_pack_dir_is_read_whatever_it_is_called(tmp_path):
    m = row("Qwen3-8B")
    d = tmp_path / "anywhere"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"formatVersion": 1, "hfRepo": m.hf_repo,
                                             "profile": "gulp", "residentBytes": 11_575_275_304}))
    sz = sizes(m, pack_dir=str(d))
    assert sz.comp_gb == 11.575275304 and sz.pack_dir == str(d)


def test_an_unpinned_rows_pack_is_found_by_the_commit_the_cache_has_for_main(monkeypatch):
    from drinkme.serving import checkpoint

    m = row("Qwen3-0.6B")
    assert m.revision is None
    sha = "c1899de289a04d12100db370d81485cdf75e47ca"
    monkeypatch.setattr(checkpoint, "_cached_commit", lambda repo, rev: sha if rev == "main" else None)
    d = write_pack(m, 1_245_571_120, revision=sha)
    assert sizes(m).pack_dir == d and sizes(m).comp_gb == 1.24557112


def test_the_pick_uses_the_packs_own_size():
    """With the estimate, Qwen3-8B is a 12 GiB card's knife-edge fit; with
    a pack on the machine holding 11.0 GB, it fits with the headroom and
    no estimate is noted."""
    write_pack(row("Qwen3-8B"), 11_000_000_000)
    s = suggest(12.88, memory_gb=12.88, memory_kind="vram")
    assert names(s) == ("Qwen3-4B", "Qwen3-8B") and not s.fit_knife_edge
    assert not s.sizes["Qwen3-8B"].estimated
    assert not any("estimate" in n for n in s.notes)


def test_a_row_whose_checkpoint_cannot_be_sized_is_left_out_and_said(checkpoint_table):
    del checkpoint_table["Qwen/Qwen3-14B"]
    s = suggest(25.77, memory_gb=25.77, memory_kind="vram")
    assert names(s) == ("Qwen3-8B", None)
    assert any(n.startswith("not considered: Qwen3-14B") for n in s.notes)
    pick, notes = serve_pick(25.77)
    assert pick.name == "Qwen3-8B" and any("not considered: Qwen3-14B" in n for n in notes)
    checkpoint_table.clear()
    pick, notes = serve_pick(25.77)
    assert pick is None and "no menu model could be sized" in notes[-1]


def test_bench_refuses_a_point_it_cannot_size(checkpoint_table):
    from drinkme import bench

    del checkpoint_table["Qwen/Qwen3-8B"]
    with pytest.raises(SystemExit, match="cannot size Qwen3-8B"):
        bench._sizes_or_exit(row("Qwen3-8B"))


# ---- checkpoint_bytes: the checkpoint's own metadata ----


def _shard(path, tensors):
    """A safetensors file with only its header: {name: (dtype, shape)}."""
    header = json.dumps({"__metadata__": {"format": "pt"},
                         **{k: {"dtype": d, "shape": s, "data_offsets": [0, 0]}
                            for k, (d, s) in tensors.items()}}).encode()
    with open(path, "wb") as f:
        f.write(len(header).to_bytes(8, "little") + header)


@pytest.fixture
def lookup(monkeypatch):
    """The real checkpoint_bytes with an empty per-process memory."""
    monkeypatch.setattr(sg, "_CHECKPOINT_BYTES", {})
    return REAL_CHECKPOINT_BYTES


def test_checkpoint_bytes_sums_a_local_snapshots_headers(tmp_path, lookup):
    _shard(tmp_path / "model-00001-of-00002.safetensors",
           {"a.weight": ("BF16", [1024, 512]), "b.weight": ("BF16", [7])})
    _shard(tmp_path / "model-00002-of-00002.safetensors", {"c.A_log": ("F32", [48])})
    assert lookup(str(tmp_path), None) == 1024 * 512 * 2 + 7 * 2 + 48 * 4


def test_checkpoint_bytes_reads_a_cached_index_total(tmp_path, monkeypatch, lookup):
    import huggingface_hub

    from drinkme.serving import checkpoint

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 55562855904.0}, "weight_map": {}}))
    monkeypatch.setattr(checkpoint, "cached_snapshot_dir", lambda repo, rev: None)
    seen = []
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, name, **kw: seen.append((repo, name, kw)) or str(index))
    assert lookup("Qwen/Qwen3.8-27B", "1d4bf0f2") == 55_562_855_904
    assert seen == [("Qwen/Qwen3.8-27B", "model.safetensors.index.json", {"revision": "1d4bf0f2"})]


def test_checkpoint_bytes_asks_the_hub_anonymously_and_remembers_the_answer(monkeypatch, lookup):
    import huggingface_hub

    from drinkme.serving import checkpoint

    monkeypatch.setattr(checkpoint, "cached_snapshot_dir", lambda repo, rev: None)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    calls = []

    def model_info(self, repo, **kw):
        calls.append((repo, kw))
        st = type("ST", (), {"parameters": {"BF16": 8_190_735_360, "F32": 10}})()
        return type("Info", (), {"safetensors": st})()

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    assert lookup("Qwen/Qwen3-8B", "b968826d") == 16_381_470_720 + 40
    assert lookup("Qwen/Qwen3-8B", "b968826d") == 16_381_470_720 + 40
    assert len(calls) == 1
    repo, kw = calls[0]
    assert repo == "Qwen/Qwen3-8B" and kw["revision"] == "b968826d"
    assert kw["token"] is False and kw["expand"] == ["safetensors"] and kw["timeout"] == sg.HUB_TIMEOUT_S


def test_checkpoint_bytes_is_none_when_nothing_answers_and_asks_again_later(monkeypatch, lookup):
    import huggingface_hub

    from drinkme.serving import checkpoint

    monkeypatch.setattr(checkpoint, "cached_snapshot_dir", lambda repo, rev: None)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    # conftest makes HfApi.model_info raise, as an offline Hub does
    assert lookup("Qwen/Qwen3-8B", None) is None
    st = type("ST", (), {"parameters": {"BF16": 5}})()
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info",
                        lambda self, repo, **kw: type("Info", (), {"safetensors": st})())
    assert lookup("Qwen/Qwen3-8B", None) == 10


# ---- serve's no-args pick (menu wiring) ----


def test_serve_pick_largest_comfortable_compressed_fit():
    assert serve_pick(8)[0].name == "Qwen3-4B"
    assert serve_pick(16)[0].name == "Qwen3-8B"
    assert serve_pick(133.14)[0].name == "Qwen2.5-72B"  # the 395's GTT budget, 124 GiB decimal
    pick, notes = serve_pick(1)
    assert pick is None and "--model" in notes[0]


def test_serve_pick_gated_only_when_gate_known_open():
    """serve never HEAD-probes the gemma gate by itself (that probe sends the
    user's HF token; bench owns it, with --no-gemma). Gated models join the
    pick only when the caller already knows the gate is open.

    51.54 GB (a 48 GiB card): wide enough for gemma's headroom-adjusted
    compressed estimate (45.85 x 1.10 = 50.43 GB) but not Qwen3-32B's
    (48.03 x 1.10 = 52.83 GB) — the window where the gate, not size, decides
    the pick (Qwen3-32B is ungated and bigger than gemma, so it would win
    over gemma on size alone once it also fit)."""
    assert serve_pick(51.54)[0].name == "Qwen3.8-27B"
    assert serve_pick(51.54, gemma_ok=True)[0].name == "gemma-4-31B-it"


def test_serve_pick_marks_estimates():
    pick, notes = serve_pick(8)  # no pack of Qwen3-4B here: its size is the estimate
    assert pick.name == "Qwen3-4B" and any("estimate" in n for n in notes)


def test_serve_pick_charges_a_packs_tower_only_with_images_on(monkeypatch, checkpoint_table):
    """A pack holds its tower resident (fit.served_resident_bytes): a row
    whose text fits and whose tower does not is picked only with images
    off."""
    tower = Model("Tower-8B", "o/tower-8b", "r", auto_eligible=True)
    checkpoint_table["o/tower-8b"] = 16_000_000_000
    monkeypatch.setattr(sg, "MODELS", [tower])
    write_pack(tower, 11_000_000_000, vision={"residentBytes": 1_000_000_000})
    budget = 10.5 * sg.FIT_HEADROOM  # the text with headroom, not the text and the tower
    pick, notes = serve_pick(budget)
    assert pick is None and "11.00 GB (the sip pack's own size)" in notes[0]
    monkeypatch.setenv("DRINKME_VISION", "0")
    assert serve_pick(budget)[0] is tower


# ---- the picker's choice for every machine class it knows ----
#
# machine, budget GB, memory GB, memory kind, runtime,
#   bench's (ratio, fit, flag) with gemma's gate closed,
#   bench's (ratio, ratio_extra, fit, flag) with it open,
#   serve_pick with the gate closed, with it open, and serve's no-arg pick
#   (packs.build_candidates with no pack on the machine, filtered for the
#   runtime, weights only).
# Every size is derived (checkpoint x COMP_RATIO["sip"]). The rows marked
# "Changed" picked differently with the hand-kept sizes this replaced; the
# comment says what moved.
PICKS = [
    ('6 GiB card', 6.44, 6.44, 'vram', 'torch',
     ('Qwen3-1.7B', 'Qwen3-4B', None), ('Qwen3-1.7B', (), 'Qwen3-4B', None),
     'Qwen3-4B', 'Qwen3-4B', 'Qwen3-4B'),
    ('8 GiB card', 8.59, 8.59, 'vram', 'torch',
     ('Qwen3-1.7B', 'Qwen3-4B', None), ('Qwen3-1.7B', (), 'Qwen3-4B', None),
     'Qwen3-4B', 'Qwen3-4B', 'Qwen3-4B'),
    ('10 GiB card', 10.74, 10.74, 'vram', 'torch',
     ('Qwen3-4B', None, None), ('Qwen3-4B', (), None, None),
     'Qwen3-4B', 'Qwen3-4B', 'Qwen3-4B'),
    ('11 GiB card', 11.81, 11.81, 'vram', 'torch',
     ('Qwen3-4B', None, None), ('Qwen3-4B', (), None, None),
     'Qwen3-4B', 'Qwen3-4B', 'Qwen3-4B'),
    ('12 GiB card', 12.88, 12.88, 'vram', 'torch',
     ('Qwen3-4B', 'Qwen3-8B', 'knife'), ('Qwen3-4B', (), 'Qwen3-8B', 'knife'),
     'Qwen3-4B', 'Qwen3-4B', 'Qwen3-4B'),
    ('16 GiB card', 17.18, 17.18, 'vram', 'torch',
     ('Qwen3-4B', 'Qwen3-8B', None), ('Qwen3-4B', (), 'Qwen3-8B', None),
     'Qwen3-8B', 'Qwen3-8B', 'Qwen3-8B'),
    ('20 GiB card', 21.47, 21.47, 'vram', 'torch',
     ('Qwen3-8B', None, None), ('Qwen3-8B', (), None, None),
     'Qwen3-8B', 'Qwen3-8B', 'Qwen3-8B'),
    ('24 GiB card', 25.77, 25.77, 'vram', 'torch',
     ('Qwen3-8B', 'Qwen3-14B', None), ('Qwen3-8B', (), 'Qwen3-14B', None),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    ('32 GiB card', 34.36, 34.36, 'vram', 'torch',
     ('Qwen3-14B', None, None), ('Qwen3-14B', (), None, None),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    ('40 GiB card', 42.95, 42.95, 'vram', 'torch',
     ('Qwen3-14B', 'Qwen3.8-27B', 'knife'), ('Qwen3-14B', (), 'Qwen3.8-27B', 'knife'),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    # Changed from the hand-kept sizes: serve_pick with the gate open was
    # Qwen3.8-27B. gemma-4-31B-it's 48.35 GB x 1.10 = 53.19 GB missed the
    # budget; its estimate, 45.85 GB (its sip record held 44.90), fits.
    ('48 GiB card', 51.54, 51.54, 'vram', 'torch',
     ('Qwen3-14B', 'Qwen3.8-27B', None), ('Qwen3-14B', (), 'Qwen3.8-27B', None),
     'Qwen3.8-27B', 'gemma-4-31B-it', 'Qwen3.8-27B'),
    ('80 GiB card', 85.9, 85.9, 'vram', 'torch',
     ('Qwen3-32B', None, None), ('Qwen3-32B', ('gemma-4-31B-it',), None, None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('96 GiB card', 103.08, 103.08, 'vram', 'torch',
     ('Qwen3-32B', None, None), ('Qwen3-32B', ('gemma-4-31B-it',), None, None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('8 GB card', 8.0, 8.0, 'vram', 'torch',
     ('Qwen3-1.7B', 'Qwen3-4B', None), ('Qwen3-1.7B', (), 'Qwen3-4B', None),
     'Qwen3-4B', 'Qwen3-4B', 'Qwen3-4B'),
    # Changed: bench's fit was a knife-edge and serve's picks Qwen3-4B.
    # Qwen3-8B's hand-kept 12.73 GB x 1.10 = 14.003 GB missed 14.0 GB by
    # 3 MB; its estimate, 12.01 GB (its sip record held 12.01), fits.
    ('14 GB card', 14.0, 14.0, 'vram', 'torch',
     ('Qwen3-4B', 'Qwen3-8B', None), ('Qwen3-4B', (), 'Qwen3-8B', None),
     'Qwen3-8B', 'Qwen3-8B', 'Qwen3-8B'),
    ('20 GB card', 20.0, 20.0, 'vram', 'torch',
     ('Qwen3-8B', None, None), ('Qwen3-8B', (), None, None),
     'Qwen3-8B', 'Qwen3-8B', 'Qwen3-8B'),
    ('30 GB card', 30.0, 30.0, 'vram', 'torch',
     ('Qwen3-8B', 'Qwen3-14B', None), ('Qwen3-8B', (), 'Qwen3-14B', None),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    ('40 GB card', 40.0, 40.0, 'vram', 'torch',
     ('Qwen3-14B', None, None), ('Qwen3-14B', (), None, None),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    ('44 GB card', 44.0, 44.0, 'vram', 'torch',
     ('Qwen3-14B', 'Qwen3.8-27B', 'knife'), ('Qwen3-14B', (), 'Qwen3.8-27B', 'knife'),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    ('48 GB card', 48.0, 48.0, 'vram', 'torch',
     ('Qwen3-14B', 'Qwen3.8-27B', None), ('Qwen3-14B', (), 'Qwen3.8-27B', None),
     'Qwen3.8-27B', 'Qwen3.8-27B', 'Qwen3.8-27B'),
    # Changed: serve_pick was Qwen3.8-27B (gemma-4-31B-it with the gate open)
    # and serve's no-arg pick Qwen3.8-27B. Qwen3-32B's hand-kept 50.4 GB
    # (0.77 x 65.5, it has no record) x 1.10 = 55.44 GB missed the budget;
    # the ratio's estimate, 48.03 GB, fits.
    ('54 GB card', 54.0, 54.0, 'vram', 'torch',
     ('Qwen3-14B', 'Qwen3.8-27B', None), ('Qwen3-14B', (), 'Qwen3.8-27B', None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('56 GB card', 56.0, 56.0, 'vram', 'torch',
     ('Qwen3-14B', 'Qwen3.8-27B', None), ('Qwen3-14B', (), 'Qwen3.8-27B', None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('60 GB card', 60.0, 60.0, 'vram', 'torch',
     ('Qwen3-14B', 'Qwen3.8-27B', None), ('Qwen3-14B', (), 'Qwen3.8-27B', None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('64 GB card', 64.0, 64.0, 'vram', 'torch',
     ('Qwen3.8-27B', 'Qwen3-32B', None), ('Qwen3.8-27B', (), 'gemma-4-31B-it', None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('70 GB card', 70.0, 70.0, 'vram', 'torch',
     ('Qwen3.8-27B', 'Qwen3-32B', None), ('Qwen3.8-27B', (), 'gemma-4-31B-it', None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('Mac 8 GB', 5.4, 8.59, 'unified', 'mlx',
     ('Qwen3-1.7B', 'Qwen3-4B', 'cliff'), ('Qwen3-1.7B', (), 'Qwen3-4B', 'cliff'),
     'Qwen3-1.7B', 'Qwen3-1.7B', 'Qwen3-1.7B'),
    ('Mac 16 GB', 12.88, 17.18, 'unified', 'mlx',
     ('Qwen3-4B', 'Qwen3-8B', 'knife'), ('Qwen3-4B', (), 'Qwen3-8B', 'knife'),
     'Qwen3-4B', 'Qwen3-4B', 'Qwen3-4B'),
    ('Mac 18 GB', 14.5, 19.33, 'unified', 'mlx',
     ('Qwen3-4B', 'Qwen3-8B', None), ('Qwen3-4B', (), 'Qwen3-8B', None),
     'Qwen3-8B', 'Qwen3-8B', 'Qwen3-8B'),
    ('Mac 24 GB', 19.33, 25.77, 'unified', 'mlx',
     ('Qwen3-8B', 'Qwen3-14B', 'cliff'), ('Qwen3-8B', (), 'Qwen3-14B', 'cliff'),
     'Qwen3-8B', 'Qwen3-8B', 'Qwen3-8B'),
    ('Mac 32 GB', 25.77, 34.36, 'unified', 'mlx',
     ('Qwen3-8B', 'Qwen3-14B', None), ('Qwen3-8B', (), 'Qwen3-14B', None),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    ('Mac 36 GB', 28.99, 38.65, 'unified', 'mlx',
     ('Qwen3-8B', 'Qwen3-14B', None), ('Qwen3-8B', (), 'Qwen3-14B', None),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    ('Mac 48 GB', 38.66, 51.54, 'unified', 'mlx',
     ('Qwen3-14B', 'Qwen3.8-27B', 'cliff'), ('Qwen3-14B', (), 'Qwen3.8-27B', 'cliff'),
     'Qwen3-14B', 'Qwen3-14B', 'Qwen3-14B'),
    # Changed: serve_pick with the gate open was Qwen3.8-27B, as on the
    # 48 GiB card (gemma-4-31B-it's 45.85 GB estimate now fits 51.54 GB).
    ('Mac 64 GB', 51.54, 68.72, 'unified', 'mlx',
     ('Qwen3-14B', 'Qwen3.8-27B', None), ('Qwen3-14B', (), 'Qwen3.8-27B', None),
     'Qwen3.8-27B', 'gemma-4-31B-it', 'Qwen3-14B'),
    ('Mac 96 GB', 77.31, 103.08, 'unified', 'mlx',
     ('Qwen3-32B', None, None), ('Qwen3-32B', ('gemma-4-31B-it',), None, None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('Mac 128 GB', 103.08, 137.44, 'unified', 'mlx',
     ('Qwen3-32B', 'Qwen2.5-72B', 'cliff'), ('Qwen3-32B', ('gemma-4-31B-it',), 'Qwen2.5-72B', 'cliff'),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('Mac 192 GB', 154.62, 206.16, 'unified', 'mlx',
     ('Qwen3-32B', 'Qwen2.5-72B', None), ('Qwen3-32B', ('gemma-4-31B-it',), 'Qwen2.5-72B', None),
     'Qwen2.5-72B', 'Qwen2.5-72B', 'Qwen3-32B'),
    ("Mac 64 GB (tests' 48 budget)", 48.0, 64.0, 'unified', 'mlx',
     ('Qwen3-14B', 'Qwen3.8-27B', None), ('Qwen3-14B', (), 'Qwen3.8-27B', None),
     'Qwen3.8-27B', 'Qwen3.8-27B', 'Qwen3-14B'),
    ('Strix Halo 128 GiB (GTT 124 GiB)', 133.14, 137.44, 'unified', 'torch',
     ('Qwen3-32B', 'Qwen2.5-72B', None), ('Qwen3-32B', ('gemma-4-31B-it',), 'Qwen2.5-72B', None),
     'Qwen2.5-72B', 'Qwen2.5-72B', 'Qwen2.5-72B'),
    ("Strix Halo 128 GiB (record's GTT)", 133.68, 137.44, 'unified', 'torch',
     ('Qwen3-32B', 'Qwen2.5-72B', None), ('Qwen3-32B', ('gemma-4-31B-it',), 'Qwen2.5-72B', None),
     'Qwen2.5-72B', 'Qwen2.5-72B', 'Qwen2.5-72B'),
    ('Strix Halo 64 GiB (GTT 60 GiB)', 64.42, 68.72, 'unified', 'torch',
     ('Qwen3.8-27B', 'Qwen3-32B', None), ('Qwen3.8-27B', (), 'gemma-4-31B-it', None),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    ('Strix Halo 96 GiB GTT', 103.08, 137.44, 'unified', 'torch',
     ('Qwen3-32B', 'Qwen2.5-72B', 'cliff'), ('Qwen3-32B', ('gemma-4-31B-it',), 'Qwen2.5-72B', 'cliff'),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
    # Changed: the 72B fit was the fit-cliff experiment. Its hand-kept
    # 110.28 GB exceeded the budget; its estimate, 106.59 GB (its sip record
    # held 103.32), fits without the headroom: a knife-edge attempt.
    ('unified 110 GB', 110.0, 128.0, 'unified', 'torch',
     ('Qwen3-32B', 'Qwen2.5-72B', 'knife'), ('Qwen3-32B', ('gemma-4-31B-it',), 'Qwen2.5-72B', 'knife'),
     'Qwen3-32B', 'Qwen3-32B', 'Qwen3-32B'),
]


@pytest.mark.parametrize("machine, budget, memory, kind, runtime, closed, opened, sp_closed, sp_open, noarg",
                         PICKS, ids=[p[0] for p in PICKS])
def test_the_picker_on_every_machine_class(machine, budget, memory, kind, runtime,
                                           closed, opened, sp_closed, sp_open, noarg):
    from drinkme import packs

    def name(m):
        return m.name if m else None

    for gate, want in ((False, closed), (True, opened)):
        s = suggest(budget, memory, kind, gemma_ok=gate)
        flag = "knife" if s.fit_knife_edge else ("cliff" if s.fit_honest_negative else None)
        got = ((name(s.ratio), tuple(m.name for m in s.ratio_extra), name(s.fit), flag) if gate
               else (name(s.ratio), name(s.fit), flag))
        assert got == want, gate
    assert name(serve_pick(budget)[0]) == sp_closed
    assert name(serve_pick(budget, gemma_ok=True)[0]) == sp_open
    cands, _ = packs.filter_for_runtime(packs.build_candidates([]), runtime)
    ranked = packs.rank_fits(budget * 1e9 / packs.GIB, cands, packs.ctx_estimate(None), 1)
    assert next((f.candidate.name for f in ranked if f.fits), None) == noarg
