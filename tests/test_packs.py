"""drinkme.packs: local pack discovery and the KV-aware fit arithmetic behind
`drinkme serve` with no `--model` (cli.py's no-model branch). Discovery tests
build a fake packs root under tmp_path with DRINKME_HOME pointed at it, never
touching a real cache; fit-arithmetic tests are pure, fabricated configs."""

from __future__ import annotations

import json

import pytest

from drinkme import packs


def _write_pack(root, dirname, hf_repo, revision, npz_sizes=(100, 200),
                extra_meta=None):
    """A minimal on-disk pack: meta.json + N .npz files of known byte sizes,
    so packed_bytes has a value the test can assert exactly."""
    d = root / "packs" / dirname
    d.mkdir(parents=True)
    meta = {"hfRepo": hf_repo, "revision": revision, "formatVersion": 1}
    meta.update(extra_meta or {})
    (d / "meta.json").write_text(json.dumps(meta))
    for i, size in enumerate(npz_sizes):
        (d / f"t{i:04d}.npz").write_bytes(b"x" * size)
    return d


# ------------------------------------------------------------- discovery --


def test_local_packs_finds_canonical_and_experiment_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "fake--repo@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(1000, 2000))
    # An experiment dir (`pack -o`): same identity, a name default_pack_dir
    # would never produce.
    _write_pack(tmp_path, "fake--repo-v3d8@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(500,))
    # A malformed meta.json: skipped, never raised.
    bad = tmp_path / "packs" / "broken@main"
    bad.mkdir(parents=True)
    (bad / "meta.json").write_text("{not json")
    # An empty dir: no meta.json at all.
    (tmp_path / "packs" / "empty@main").mkdir(parents=True)

    found = packs.local_packs()
    assert len(found) == 2  # the malformed and empty dirs contribute nothing

    canonical = [p for p in found if p.canonical]
    experiment = [p for p in found if not p.canonical]
    assert len(canonical) == 1 and len(experiment) == 1
    assert canonical[0].packed_bytes == 3000
    assert experiment[0].packed_bytes == 500
    assert canonical[0].hf_repo == experiment[0].hf_repo == "fake/repo"


def test_local_packs_skips_malformed_meta_with_one_stderr_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    bad = tmp_path / "packs" / "broken@main"
    bad.mkdir(parents=True)
    (bad / "meta.json").write_text("not even json")

    found = packs.local_packs()
    assert found == []
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "broken@main" in err


def test_local_packs_does_not_offer_a_pack_of_another_format_and_says_why_once(tmp_path, monkeypatch, capsys):
    """A pack of a format version this build does not read is not on the
    menu — the loader would refuse it by name — and the scan says so once,
    with the fix; a pack declaring no version at all is the same case."""
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "old--repo@main", "old/repo", None, extra_meta={"formatVersion": 7})
    _write_pack(tmp_path, "new--repo@main", "new/repo", None)
    bare = _write_pack(tmp_path, "bare--repo@main", "bare/repo", None)
    meta = json.loads((bare / "meta.json").read_text())
    del meta["formatVersion"]
    (bare / "meta.json").write_text(json.dumps(meta))
    found = packs.local_packs()
    assert [p.hf_repo for p in found] == ["new/repo"]
    err = capsys.readouterr().err
    assert err.count("\n") == 2
    assert ("old--repo@main: unsupported pack format 7 (this drinkme reads format 1); "
            "re-pack with `drinkme pack --model old/repo --replace`") in err
    assert ("bare--repo@main: unsupported pack format (none declared) (this drinkme reads "
            "format 1); re-pack with `drinkme pack --model bare/repo --replace`") in err
    assert err.count("old--repo@main") == 1  # the path once per line


def test_local_packs_root_param_overrides_env(tmp_path, monkeypatch):
    # DRINKME_HOME points somewhere else entirely; the explicit root param wins,
    # and the canonical check is computed against THAT root, not the env's.
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path / "not-used"))
    other = tmp_path / "elsewhere"
    _write_pack(other, "fake--repo@abc123def456", "fake/repo", "abc123def456")

    found = packs.local_packs(root=str(other))
    assert len(found) == 1 and found[0].canonical


def test_local_packs_reads_residentbytes_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "fake--repo@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(1000,), extra_meta={"residentBytes": 1600})

    found = packs.local_packs()
    assert len(found) == 1
    assert found[0].packed_bytes == 1000
    assert found[0].resident_bytes == 1600
    assert found[0].resident_bytes_estimated is False


def test_local_packs_charges_the_vision_tower_unless_images_are_off(tmp_path, monkeypatch):
    """DRINKME_VISION=0 builds no tower, so the picker charges the pack
    less its `vision` block's resident bytes (fit.served_resident_bytes)."""
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "Qwen--Qwen3.8-27B@main", "Qwen/Qwen3.8-27B", None, npz_sizes=(1000,),
                extra_meta={"residentBytes": 1600, "vision": {"residentBytes": 400}})
    monkeypatch.delenv("DRINKME_VISION", raising=False)
    assert packs.local_packs()[0].resident_bytes == 1600
    monkeypatch.setenv("DRINKME_VISION", "0")
    assert packs.local_packs()[0].resident_bytes == 1200


def test_local_packs_falls_back_to_npz_sum_and_marks_the_estimate(tmp_path, monkeypatch):
    """A tool-written pack without residentBytes (`drinkme pack` always
    records it). The picker must not silently treat the npz-sum fallback as
    a real measurement."""
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "fake--repo@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(1000, 500))

    found = packs.local_packs()
    assert len(found) == 1
    assert found[0].resident_bytes == 1500  # == packed_bytes, the estimate
    assert found[0].resident_bytes_estimated is True


def test_local_packs_matches_menu_by_hf_repo_and_revision(tmp_path, monkeypatch):
    from drinkme.suggest import MODELS

    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    m = next(x for x in MODELS if x.name == "Qwen3-8B")
    slug = m.hf_repo.replace("/", "--")
    _write_pack(tmp_path, f"{slug}@{m.revision[:12]}", m.hf_repo, m.revision)

    found = packs.local_packs()
    assert len(found) == 1
    assert found[0].menu is m


# --------------------------------------------------------- build_candidates --


def test_experiment_pack_is_never_offered_as_the_menu_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "fake--repo@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(4000,))
    _write_pack(tmp_path, "fake--repo-v3d8@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(999999,))  # much bigger, must NOT win by size alone

    local = packs.local_packs()
    cands = packs.build_candidates(local)
    matches = [c for c in cands if c.hf_repo == "fake/repo"]
    assert len(matches) == 1
    assert matches[0].pack_dir.endswith("fake--repo@abc123def456")
    assert matches[0].resident_gib == pytest.approx(4000 / packs.GIB)


def test_gemma_excluded_even_when_packed_locally_without_gate_probe(tmp_path, monkeypatch):
    """serve never HEAD-probes gemma's gate on the no-model path; a canonical
    pack already on disk must not become silent consent to serve it."""
    from drinkme.suggest import MODELS

    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    gemma = next(x for x in MODELS if x.gated)
    slug = gemma.hf_repo.replace("/", "--")
    _write_pack(tmp_path, f"{slug}@{gemma.revision[:12]}", gemma.hf_repo, gemma.revision)

    local = packs.local_packs()
    assert not any(c.hf_repo == gemma.hf_repo for c in packs.build_candidates(local))
    assert any(c.hf_repo == gemma.hf_repo
              for c in packs.build_candidates(local, gemma_ok=True))


def test_a_row_not_auto_eligible_is_never_a_no_arg_candidate_even_when_packed(tmp_path, monkeypatch):
    """A row without suggest.Model.auto_eligible (Muse-Glimmer-30B): listed for
    `--model`, never the automatic pick — not from the menu loop, and not
    from a canonical pack of it already on disk either (the gemma rule's
    shape, without a gate to open)."""
    from drinkme.suggest import MODELS

    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    named = next(x for x in MODELS if not x.auto_eligible)
    slug = named.hf_repo.replace("/", "--")
    _write_pack(tmp_path, f"{slug}@{named.revision[:12]}", named.hf_repo, named.revision)

    local = packs.local_packs()
    assert any(p.hf_repo == named.hf_repo and p.canonical for p in local)
    for gemma_ok in (False, True):
        assert not any(c.hf_repo == named.hf_repo
                       for c in packs.build_candidates(local, gemma_ok=gemma_ok))


def test_build_candidates_uses_residentbytes_over_npz_sum_when_present(tmp_path, monkeypatch):
    """the whole point — a pack's npz sum under-counts (the untied-embed
    case); build_candidates must charge the real residentBytes, not the
    on-disk packed bytes, once a pack records one."""
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "fake--repo@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(4000,), extra_meta={"residentBytes": 5200})

    cands = packs.build_candidates(packs.local_packs())
    c = next(c for c in cands if c.hf_repo == "fake/repo")
    assert c.resident_gib == pytest.approx(5200 / packs.GIB)
    assert not c.resident_estimated


def test_build_candidates_marks_the_npz_fallback_as_estimated(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "fake--repo@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(4000,))  # no residentBytes: an old-style pack

    cands = packs.build_candidates(packs.local_packs())
    c = next(c for c in cands if c.hf_repo == "fake/repo")
    assert c.resident_gib == pytest.approx(4000 / packs.GIB)
    assert c.resident_estimated


def test_an_unpacked_menu_row_is_charged_its_whole_checkpoint_times_the_ratio(tmp_path, monkeypatch,
                                                                             checkpoint_table):
    """Before a pack exists a menu row's charge is suggest.estimate: the
    checkpoint's safetensors bytes x the profile's ratio, tower included,
    so DRINKME_VISION=0 does not change it (a pack's `vision` block is
    left out once the pack exists: fit.served_resident_bytes)."""
    from drinkme import suggest

    row = suggest.Model("Tower-8B", "o/tower-8b", "r", auto_eligible=True)
    checkpoint_table["o/tower-8b"] = 16_000_000_000
    monkeypatch.setattr(suggest, "MODELS", [row])
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    for env in (None, "0"):
        if env is None:
            monkeypatch.delenv("DRINKME_VISION", raising=False)
        else:
            monkeypatch.setenv("DRINKME_VISION", env)
        (c,) = packs.build_candidates(packs.local_packs())
        assert not c.packed and not c.measured
        assert c.resident_gib == pytest.approx(16.0 * suggest.COMP_RATIO["sip"] * packs.GB_TO_GIB)


def test_a_menu_row_that_cannot_be_sized_is_not_offered_and_said(tmp_path, monkeypatch, capsys,
                                                                 checkpoint_table):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    del checkpoint_table["Qwen/Qwen3-14B"]
    cands = packs.build_candidates(packs.local_packs())
    assert "Qwen3-14B" not in {c.name for c in cands} and "Qwen3-8B" in {c.name for c in cands}
    assert "not offering Qwen3-14B — the checkpoint's size could not be read" in capsys.readouterr().err


def test_rank_fits_flips_on_the_true_residentbytes_a_npz_sum_alone_would_miss():
    """The picker's fit decision uses residentBytes when present: a
    pack whose npz sum alone fits a budget with headroom must NOT fit once
    its true (measured) residentBytes — bigger, by the raw-streamed share —
    is what gets charged."""
    npz_gib = 10.0
    resident_gib = 10.9  # the untied-embed-style undercount, measured
    budget_gib = 11.5  # npz*1.1=11.0 fits; resident*1.1=11.99 does not

    npz_only = _cand("x", npz_gib, packed=True)
    true_resident = _cand("x", resident_gib, packed=True)

    assert packs.rank_fits(budget_gib, [npz_only], ctx=8192, slots=1)[0].fits
    assert not packs.rank_fits(budget_gib, [true_resident], ctx=8192, slots=1)[0].fits


def test_format_summary_carries_the_estimate_caveat_for_a_pack_without_resident_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "fake--repo@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(4000,))  # no residentBytes

    local = packs.local_packs()
    cands = packs.build_candidates(local)
    ranked = packs.rank_fits(1000.0, cands, ctx=8192, slots=1)
    lines = packs.format_summary(1000.0, "s", ranked, local, "/root", 8192, 1)
    joined = "\n".join(lines)
    assert "fake/repo" in joined
    assert "(estimated — repack to measure)" in joined


def test_format_summary_has_no_caveat_when_residentbytes_is_measured(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    _write_pack(tmp_path, "fake--repo@abc123def456", "fake/repo", "abc123def456",
               npz_sizes=(4000,), extra_meta={"residentBytes": 4200})

    local = packs.local_packs()
    cands = packs.build_candidates(local)
    ranked = packs.rank_fits(1000.0, cands, ctx=8192, slots=1)
    lines = packs.format_summary(1000.0, "s", ranked, local, "/root", 8192, 1)
    assert "(estimated — repack to measure)" not in "\n".join(lines)


def test_menu_model_not_packed_locally_is_an_estimate_and_says_so(tmp_path, monkeypatch):
    from drinkme.suggest import MODELS, sizes

    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    local = packs.local_packs()
    cands = packs.build_candidates(local)
    m = next(x for x in MODELS if x.name == "Qwen3-8B")
    c = next(c for c in cands if c.name == "Qwen3-8B")
    assert not c.packed and c.config is None and not c.measured
    assert c.resident_gib == pytest.approx(sizes(m).comp_gb * packs.GB_TO_GIB)
    ranked = packs.rank_fits(1000.0, [c], ctx=8192, slots=1)
    (line,) = [x for x in packs.format_summary(1000.0, "s", ranked, local, "/root", 8192, 1)
               if "Qwen3-8B" in x]
    assert f"{c.resident_gib:.2f} GiB (estimate)" in line and "(will download+pack)" in line


# ---------------------------------------------------------- unit conversion --


def test_gb_to_gib_matches_a_real_pack_measurement():
    # Qwen3.8-27B: docstring says "pack on disk 38.44 GB" (suggest.py); a real
    # pack directory on the dev box sums to 38,441,646,161 bytes == 35.80 GiB.
    assert 38.44 * packs.GB_TO_GIB == pytest.approx(35.80, abs=0.01)


def test_detect_budget_is_treated_as_gib_not_gb(monkeypatch):
    """detect.Hardware.budget_gb is decimal GB (bytes / 1e9); packs.py is
    GiB-native. hardware_budget() must charge the box's budget_bytes in
    GiB — 124 decimal GB is 115.48 GiB, not 124 — or the no-`--model` serve
    pick is ~7.4% too generous (a units defect). The box is Linux, pinned
    where hardware_budget reads it: on a Mac it takes MLX's working set
    instead of detect's budget."""
    from hosts import pin_linux_host

    from drinkme.detect import Hardware

    pin_linux_host(monkeypatch)

    budget_bytes = 124 * 10**9
    fake = Hardware(device_class="d", memory_gb=128.0, memory_kind="unified",
                    budget_gb=124.0, cpu_info=None, gpu_info=None,
                    memory_bytes=128 * 10**9, budget_bytes=budget_bytes)
    monkeypatch.setattr("drinkme.detect.detect", lambda: fake)
    # the capacity's units are the question here, so what is free right now
    # (the live reading of this test box) is held at more than the capacity
    monkeypatch.setattr("drinkme.detect.live_free_bytes", lambda hw: (2 * budget_bytes, "MemAvailable"))
    budget_gib, source = packs.hardware_budget()
    assert budget_gib == pytest.approx(budget_bytes / packs.GIB)  # 115.48, not 124.0
    assert budget_gib == pytest.approx(115.48, abs=0.01)
    assert budget_gib < fake.budget_gb
    assert "115.5 GiB" in source and "GB" not in source.replace("GiB", "")


# -------------------------------------------------------------- kv arithmetic --


_DENSE_CFG = {"num_hidden_layers": 4, "num_key_value_heads": 2, "head_dim": 8,
             "hidden_size": 16, "num_attention_heads": 2}


def test_kv_gib_per_token_dense():
    # K+V, bf16: 2 * layers * kv_heads * head_dim * 2 bytes = 2*4*2*8*2 = 256 B/token
    assert packs.kv_gib_per_token(_DENSE_CFG) == pytest.approx(256 / packs.GIB)


def test_kv_gib_per_token_derives_head_dim_when_absent():
    cfg = dict(_DENSE_CFG)
    del cfg["head_dim"]  # hidden_size // num_attention_heads = 16 // 2 = 8, same as above
    assert packs.kv_gib_per_token(cfg) == pytest.approx(256 / packs.GIB)


def test_kv_gib_per_token_hybrid_counts_only_full_attention_layers():
    cfg = dict(_DENSE_CFG, num_hidden_layers=4,
              layer_types=["linear_attention", "full_attention",
                           "linear_attention", "full_attention"])
    # Only 2 of 4 layers carry a KV cache -> half the dense charge.
    assert packs.kv_gib_per_token(cfg) == pytest.approx(128 / packs.GIB)


def test_kv_gib_per_token_unwraps_text_config():
    nested = {"model_type": "qwen3_5", "text_config": _DENSE_CFG}
    assert packs.kv_gib_per_token(nested) == packs.kv_gib_per_token(_DENSE_CFG)


# -------------------------------------------------------------------- ctx/slots --


def test_ctx_estimate_defaults_to_ctx_cap():
    from drinkme.serving.checkpoint import CTX_CAP

    assert packs.ctx_estimate(None) == CTX_CAP


def test_ctx_estimate_explicit_wins():
    assert packs.ctx_estimate(4096) == 4096


def test_ctx_estimate_reads_env(monkeypatch):
    monkeypatch.setenv("DRINKME_CTX", "2048")
    assert packs.ctx_estimate(None) == 2048


def test_slots_estimate_default_and_explicit():
    assert packs.slots_estimate(None) == 1
    assert packs.slots_estimate(3) == 3
    assert packs.slots_estimate(0) == 1  # garbage falls back, no warning here


def test_slots_estimate_reads_env(monkeypatch):
    monkeypatch.setenv("DRINKME_PREFIX_SLOTS", "5")
    assert packs.slots_estimate(None) == 5


# ------------------------------------------------------------------ rank_fits --


def _cand(name, resident_gib, config=None, packed=False, mtp_head_gib=None):
    return packs.Candidate(name, f"org/{name}", None, resident_gib, packed,
                           None, packed, config, mtp_head_gib)


def test_rank_fits_charges_headroom_and_marks_fit():
    cands = [_cand("big", 100.0), _cand("small", 1.0)]
    ranked = packs.rank_fits(budget_gib=2.0, candidates=cands, ctx=8192, slots=1)
    by_name = {f.candidate.name: f for f in ranked}
    assert by_name["small"].fits and not by_name["big"].fits
    from drinkme.suggest import FIT_HEADROOM

    assert by_name["small"].charged_gib == pytest.approx(1.0 * FIT_HEADROOM)


def test_rank_fits_orders_largest_resident_first_fits_before_non_fits():
    cands = [_cand("a", 5.0), _cand("b", 50.0), _cand("c", 1.0)]
    ranked = packs.rank_fits(budget_gib=100.0, candidates=cands, ctx=8192, slots=1)
    assert [f.candidate.name for f in ranked] == ["b", "a", "c"]


def test_rank_fits_prefers_already_packed_on_a_tie():
    cands = [_cand("unpacked", 10.0, packed=False), _cand("packed", 10.0, packed=True)]
    ranked = packs.rank_fits(budget_gib=100.0, candidates=cands, ctx=8192, slots=1)
    assert ranked[0].candidate.name == "packed"


def test_rank_fits_kv_note_when_no_config():
    ranked = packs.rank_fits(1000.0, [_cand("x", 1.0)], ctx=8192, slots=1)
    assert ranked[0].note == "KV not counted — config not cached"
    assert ranked[0].kv_gib == 0.0


def test_rank_fits_kv_scales_with_ctx_and_slots():
    cands = [_cand("x", 1.0, config=_DENSE_CFG)]
    f1 = packs.rank_fits(1000.0, cands, ctx=1000, slots=1)[0]
    f2 = packs.rank_fits(1000.0, cands, ctx=2000, slots=1)[0]
    f3 = packs.rank_fits(1000.0, cands, ctx=1000, slots=4)[0]
    assert f2.kv_gib == pytest.approx(f1.kv_gib * 2)
    assert f3.kv_gib == pytest.approx(f1.kv_gib * 4)
    assert f1.note is None


def test_rank_fits_includes_mtp_head():
    cands = [_cand("x", 1.0, mtp_head_gib=0.5)]
    f = packs.rank_fits(1000.0, cands, ctx=8192, slots=1)[0]
    assert f.total_gib == pytest.approx(1.5)


# ----------------------------------------------------------------- summary --


def test_format_summary_marks_the_pick_and_collapses_non_fits():
    cands = [_cand("big", 90.0), _cand("mid", 10.0), _cand("tiny", 1.0)]
    ranked = packs.rank_fits(budget_gib=15.0, candidates=cands, ctx=8192, slots=1)
    lines = packs.format_summary(15.0, "test 15.0 GiB", ranked, [], "/root", 8192, 1)
    joined = "\n".join(lines)
    assert "budget: test 15.0 GiB" in joined
    assert "-> mid:" in joined
    assert "tiny:" in joined
    assert "1 more don't fit at this ctx" in joined
    assert "big" not in joined


def test_format_summary_reports_canonical_and_experiment_counts():
    local = [
        packs.LocalPack("a/a", None, "/root/a@main", 1, True, None),
        packs.LocalPack("b/b", None, "/root/b-gulp@main", 1, False, None),
    ]
    ranked = packs.rank_fits(100.0, [_cand("x", 1.0)], ctx=8192, slots=1)
    lines = packs.format_summary(100.0, "s", ranked, local, "/root", 8192, 1)
    joined = "\n".join(lines)
    assert "2 pack(s) found under /root" in joined
    assert "1 canonical, 1 experiment" in joined


def test_format_summary_omits_local_line_when_nothing_found():
    ranked = packs.rank_fits(100.0, [_cand("x", 1.0)], ctx=8192, slots=1)
    lines = packs.format_summary(100.0, "s", ranked, [], "/root", 8192, 1)
    assert not any("pack(s) found" in l for l in lines)
