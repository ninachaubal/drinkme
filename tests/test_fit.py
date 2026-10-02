"""ONE fit primitive. bench's suggest()/serve_pick() (decimal-GB model
sizes) and the serve picker's packs.rank_fits (GiB pack measurements) both
route the resident+kv, headroom-multiplied, against-a-budget check through
drinkme.fit.fits — bytes throughout, so there is exactly one place this
arithmetic lives instead of two, each in its own units."""

from __future__ import annotations

from drinkme.fit import GB, GIB, fits, stock_arm_fits, stock_transient_bytes


def test_fits_pure_arithmetic():
    assert fits(100, 0, 110, 1.0)
    assert not fits(100, 0, 90, 1.0)


def test_kv_bytes_count_toward_the_budget():
    assert fits(100, 0, 115, 1.1)
    assert not fits(100, 10, 115, 1.1)


def test_headroom_flips_a_knife_edge_case():
    """The original bug's scenario: fits clean at 1.0x, not at 1.1x headroom."""
    resident, budget = 100 * GB, 105 * GB
    assert fits(resident, 0, budget, 1.0)
    assert not fits(resident, 0, budget, 1.1)


def test_fit_headroom_constant_produces_the_same_flip():
    from drinkme.suggest import FIT_HEADROOM

    resident, budget = 100 * GB, 105 * GB
    assert fits(resident, 0, budget, 1.0)
    assert not fits(resident, 0, budget, FIT_HEADROOM)


def test_gib_and_gb_are_not_the_same_byte_count():
    assert GIB != GB
    assert GIB > GB  # 1024**3 > 1e9


# --------------------------------------------- both real callers agree --


def test_suggest_and_packs_agree_on_the_same_candidate():
    """bench's serve_pick (decimal-GB suggest.Sizes) and the serve picker's
    rank_fits (GiB resident_gib) must reach the same verdict for the same
    model at equivalent budgets, once each converts to bytes at its own
    boundary — two callers, one fit rule."""
    from drinkme import packs
    from drinkme.suggest import MODELS, serve_pick, sizes

    m = next(x for x in MODELS if x.name == "Qwen3-8B")
    comp_gb = sizes(m).comp_gb
    # 13.1 GiB: comfortably above the 12.30 GiB crossover where Qwen3-8B's
    # compressed estimate (12.01 GB) x 1.10 headroom clears the budget in
    # either unit convention.
    budget_gib = 13.1
    budget_gb = budget_gib * GIB / GB  # what bench's decimal-GB suggest() wants

    bench_pick, _ = serve_pick(budget_gb)
    assert bench_pick.name == "Qwen3-8B"

    cand = packs.Candidate(m.name, m.hf_repo, m.revision, comp_gb * packs.GB_TO_GIB,
                           True, None, True, None, None)
    ranked = packs.rank_fits(budget_gib, [cand], ctx=8192, slots=1)
    assert ranked[0].fits


def test_suggest_and_packs_agree_the_knife_edge_flips_the_same_way():
    """One byte short of Qwen3-8B's 1.1x-headroom line: neither caller's fit
    check should let it through, in either unit convention — the
    '1.1x headroom on a knife-edge case flips' pin, at both call sites."""
    from drinkme import packs
    from drinkme.suggest import MODELS, serve_pick, sizes

    m = next(x for x in MODELS if x.name == "Qwen3-8B")
    comp_gb = sizes(m).comp_gb
    budget_bytes = comp_gb * GB * 1.10 - 1  # one byte short of the exact headroom line
    budget_gb = budget_bytes / GB
    budget_gib = budget_bytes / GIB

    bench_pick, _ = serve_pick(budget_gb)
    assert bench_pick is None or bench_pick.name != "Qwen3-8B"

    cand = packs.Candidate(m.name, m.hf_repo, m.revision, comp_gb * packs.GB_TO_GIB,
                           True, None, True, None, None)
    ranked = packs.rank_fits(budget_gib, [cand], ctx=8192, slots=1)
    assert not ranked[0].fits


# ------------------------------------------- the stock-arm fit check --


def test_stock_transient_doubles_only_on_unified():
    """The default charge is from_pretrained's: 2x on unified (measured
    device_map="cuda" or not), 1x on discrete. `direct=True`
    is the STREAMING loader's charge now (resident +
    one tensor — tests/test_stock_stream_loader.py has the full set)."""
    from drinkme.fit import STAGING_SHARD_BYTES
    assert stock_transient_bytes(100, 0, "unified") == 200
    assert stock_transient_bytes(100, 0, "unified", direct=False) == 200
    assert stock_transient_bytes(100, 0, "unified", direct=True) == 100 + STAGING_SHARD_BYTES
    assert stock_transient_bytes(100, 0, "unified", direct=True, tensor_bytes=3) == 103
    assert stock_transient_bytes(100, 0, "vram") == 100


def test_stock_transient_adds_kv_once_not_doubled():
    assert stock_transient_bytes(100, 10, "unified") == 210
    assert stock_transient_bytes(100, 10, "unified", direct=False) == 210
    assert stock_transient_bytes(100, 10, "vram") == 110


def test_rx7600xt_qwen8b_stock_does_not_fit_16gb_discrete():
    """The original bug's repro (a): an RX 7600 XT, 16 GB VRAM.
    `--detect-only` already got this right (Qwen3-8B's stock arm does not
    fit) — the bug was that `--model` ran it anyway, not a wrong prediction;
    this pins the prediction itself."""
    from drinkme.suggest import MODELS, RATIO_HEADROOM, sizes

    m = next(x for x in MODELS if x.name == "Qwen3-8B")
    ceiling = 16.0 * GIB  # detect.py's discrete budget_gb, GiB
    assert not stock_arm_fits(sizes(m).bf16_gb * GB, 0.0, ceiling, "vram", RATIO_HEADROOM)


def test_strix_halo_gemma_stock_does_not_fit_124gb_unified_under_from_pretrained():
    """Strix Halo, 124 GB unified. The naive single-copy check (62.55 GB *
    1.15 = 71.93 GB) says yes; the transient-charged check says no — and,
    MEASURED, from_pretrained loading straight to the device does not change
    that (MemAvailable 107 -> 5 GB in ~30 s, watchdog kill). Under the
    streaming stock loader's one-tensor charge (`direct=True`) gemma is a
    3-arm ratio point."""
    from drinkme.suggest import MODELS, RATIO_HEADROOM, sizes

    m = next(x for x in MODELS if x.name == "gemma-4-31B-it")
    ceiling = 124.0 * GIB
    assert fits(sizes(m).bf16_gb * GB, 0.0, ceiling, RATIO_HEADROOM)  # the naive check alone would pass it
    assert not stock_arm_fits(sizes(m).bf16_gb * GB, 0.0, ceiling, "unified", RATIO_HEADROOM)
    assert not stock_arm_fits(sizes(m).bf16_gb * GB, 0.0, ceiling, "unified", RATIO_HEADROOM, direct=False)
    assert stock_arm_fits(sizes(m).bf16_gb * GB, 0.0, ceiling, "unified", RATIO_HEADROOM, direct=True)


def test_strix_halo_27b_squeaked_through_the_same_formula():
    """The 27B that "squeaked through minutes earlier" in the original
    bug report: same formula, same box, less margin used — not a different
    rule, just a smaller model against the same ceiling."""
    from drinkme.suggest import MODELS, RATIO_HEADROOM, sizes

    m = next(x for x in MODELS if x.name == "Qwen3.8-27B")
    ceiling = 124.0 * GIB
    assert stock_arm_fits(sizes(m).bf16_gb * GB, 0.0, ceiling, "unified", RATIO_HEADROOM)


def test_stock_arm_fits_matches_plain_fits_on_discrete():
    """No doubling on discrete: stock_arm_fits and the plain primitive must
    agree exactly (same bytes go into the same inequality)."""
    from drinkme.suggest import RATIO_HEADROOM

    resident, ceiling = 100 * GB, 115 * GB
    assert stock_arm_fits(resident, 0.0, ceiling, "vram", RATIO_HEADROOM) == \
        fits(resident, 0.0, ceiling, RATIO_HEADROOM)


def test_served_resident_bytes_leaves_the_vision_tower_out_only_when_images_are_off():
    """A pack's residentBytes charges its vision tower like any resident
    tensor; DRINKME_VISION=0 builds no tower, so the fit leaves the
    `vision` block's share out. A pack without a block charges whole."""
    from drinkme.fit import served_resident_bytes

    meta = {"residentBytes": 1000, "vision": {"tower": "model.visual", "residentBytes": 300}}
    assert served_resident_bytes(meta, True) == 1000
    assert served_resident_bytes(meta, False) == 700
    assert served_resident_bytes({"residentBytes": 1000}, False) == 1000
    assert served_resident_bytes({}, True) is None
