"""codec/radix_schedule.py: the per-(box class, profile family, shape class)
launch table and its
overrides. CPU-only — the table is a pure function of (box class, R, C,
widths) and the DRINKME_RADIX_SCHEDULE override; what each row is WORTH is
the rotation sweeps' output (bench/parity_rotation.py, bench/modal_nvidia2.py),
not this file's."""
import json
from types import SimpleNamespace

import pytest

from drinkme.codec import radix_schedule as rs

QWEN3_8B = ((151936, 4096), (1024, 4096), (4096, 12288), (12288, 4096), (4096, 4096))


@pytest.fixture
def gfx1151(monkeypatch):
    """The gfx1151 tables, whatever box runs the suite."""
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "rocm")
    monkeypatch.setattr(rs, "_ARCH", "gfx1151")
    monkeypatch.delenv("DRINKME_RADIX_SCHEDULE", raising=False)
    rs._per_class_table.cache_clear()


@pytest.fixture
def gfx1102(monkeypatch):
    """The RX 7600 XT's own tables (a ROCm build whose device reports
    gfx1102), whatever box runs the suite."""
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "rocm")
    monkeypatch.setattr(rs, "_ARCH", "gfx1102")
    monkeypatch.delenv("DRINKME_RADIX_SCHEDULE", raising=False)
    rs._per_class_table.cache_clear()


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "cuda")
    monkeypatch.setattr(rs, "_ARCH", "")
    monkeypatch.delenv("DRINKME_RADIX_SCHEDULE", raising=False)
    rs._per_class_table.cache_clear()


@pytest.mark.parametrize("shape,cls", [
    ((151936, 4096), "head"), ((32768, 1024), "head"),
    ((1024, 4096), "narrow"), ((512, 8192), "narrow"),
    ((4096, 12288), "long"), ((4096, 8192), "long"),
    ((12288, 4096), "wide"), ((8192, 4096), "wide"),
    ((4096, 4096), "square"), ((2048, 7168), "square"),
])
def test_shape_classes(shape, cls):
    assert rs.shape_class(*shape) == cls


def test_families_by_tier_count():
    assert rs.family((3, 8)) == "sip"
    assert rs.family((2, 3, 8)) == "scheduled"
    assert rs.family((2, 2, 4, 8)) == "scheduled"
    assert rs.family([3, 8]) == "sip"


def test_the_table_is_box_class_x_family_x_class():
    """Every box class — the two AMD gfx targets with their own rows and
    cuda — carries both families over every class, every row a valid
    launch; the gfx1151 and cuda sip rows are whole-row at one and four warps,
    their scheduled rows one block at one warp, mc t1/w1 (the gfx1102 rows
    are pinned by their own test below)."""
    assert set(rs.TABLES) == set(rs.BOX_CLASSES) == {"gfx1151", "gfx1102", "cuda"}
    assert set(rs.BACKEND_FAMILIES) == {"rocm", "cuda"}
    for box, table in rs.TABLES.items():
        assert set(table) == set(rs.FAMILIES) == {"sip", "scheduled"}
        for fam, rows in table.items():
            assert set(rows) == set(rs.CLASSES)
            for row in rows.values():
                assert set(row) == set(rs.SPIKE)
                rs._validate(row)
                if box == "gfx1102":
                    continue
                assert (row["mc_tiles"], row["mc_warps"]) == (1, 1)
                if fam == "sip":
                    assert row["gemv_tiles"] == 0
                    assert row["gemv_warps"] == (1 if box == "gfx1151" else 4)
                else:
                    assert (row["gemv_tiles"], row["gemv_warps"]) == (1, 1)


def test_gfx1151_sip_is_the_row_program_and_scheduled_is_one_block(gfx1151):
    """The gfx1151 rotation sweep: the whole-row program at one warp wins every class
    for sip and loses 1.3-2.3x for the profiles that carry the schedule
    header; the mc arm is one block per program at one warp for both."""
    for R, C in QWEN3_8B:
        sip = rs.select(R, C, 1024, (3, 8))
        assert (sip.family, sip.box_class, sip.source) == ("sip", "gfx1151", "table")
        assert (sip.gemv_tiles, sip.gemv_warps, sip.mc_tiles, sip.mc_warps) == (0, 1, 1, 1)
        for widths in ((2, 3, 8), (2, 2, 4, 8)):
            sch = rs.select(R, C, 1024, widths)
            assert (sch.family, sch.box_class, sch.source) == ("scheduled", "gfx1151", "table")
            assert (sch.gemv_tiles, sch.gemv_warps, sch.mc_tiles, sch.mc_warps) == (1, 1, 1, 1)
    assert sip.describe() == "gemv row/w1 mc 1/w1"
    assert sch.describe() == "gemv 1/w1 mc 1/w1"


def test_cuda_sip_is_the_row_program_at_four_warps(cuda):
    """Every NVIDIA card's sip row is the whole row at
    four warps; the scheduled family keeps its gfx1151 row there."""
    for R, C in QWEN3_8B:
        sip = rs.select(R, C, 1024, (3, 8))
        assert (sip.family, sip.box_class, sip.source) == ("sip", "cuda", "table")
        assert (sip.gemv_tiles, sip.gemv_warps, sip.mc_tiles, sip.mc_warps) == (0, 4, 1, 1)
        sch = rs.select(R, C, 1024, (2, 2, 4, 8))
        assert (sch.family, sch.box_class) == ("scheduled", "cuda")
        assert (sch.gemv_tiles, sch.gemv_warps, sch.mc_tiles, sch.mc_warps) == (1, 1, 1, 1)
    assert sip.describe() == "gemv row/w4 mc 1/w1"


def test_gfx1102_is_its_own_box_class(gfx1102):
    """The RX 7600 XT rotation sweep (sip and gulp at
    the grid): sip is one block per program at one warp on square / wide /
    long / head and the whole row at two warps on narrow; gulp is one block
    at two warps everywhere; mc t1/w1. `table=gfx1151` on that box is the
    gfx1151 row — the control arm of the model-level pair."""
    picks = {"square": (1, 1), "wide": (1, 1), "long": (1, 1), "head": (1, 1), "narrow": (0, 2)}
    for R, C in QWEN3_8B:
        sip = rs.select(R, C, 1024, (3, 8))
        assert (sip.family, sip.box_class, sip.source) == ("sip", "gfx1102", "table")
        assert (sip.gemv_tiles, sip.gemv_warps) == picks[sip.shape_class]
        assert (sip.mc_tiles, sip.mc_warps) == (1, 1)
        gulp = rs.select(R, C, 1024, (2, 2, 4, 8))
        assert (gulp.family, gulp.box_class, gulp.source) == ("scheduled", "gfx1102", "table")
        assert (gulp.gemv_tiles, gulp.gemv_warps, gulp.mc_tiles, gulp.mc_warps) == (1, 2, 1, 1)
        control = rs.select(R, C, 1024, (3, 8), override="table=gfx1151")
        assert (control.gemv_tiles, control.gemv_warps, control.box_class, control.source) == (0, 1, "gfx1151", "table:gfx1151")
    assert rs.select(4096, 4096).describe() == "gemv 1/w1 mc 1/w1"
    assert rs.select(1024, 4096).describe() == "gemv row/w2 mc 1/w1"
    assert rs.select(4096, 4096, 1024, (2, 2, 4, 8)).describe() == "gemv 1/w2 mc 1/w1"
    assert rs.TABLES["gfx1102"]["sip"]["narrow"] == dict(gemv_tiles=0, gemv_warps=2, mc_tiles=1, mc_warps=1)


@pytest.mark.parametrize("build", ["rocm", "cuda"])
def test_the_gfx1102_table_by_name_from_another_box(monkeypatch, build):
    """table=gfx1102 names the RX 7600 XT's rows whatever the build (the
    cross-box arm the other way round)."""
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", build)
    monkeypatch.setattr(rs, "_ARCH", "gfx1151" if build == "rocm" else "")
    x = rs.select(4096, 4096, 1024, (3, 8), override="table=gfx1102")
    assert (x.gemv_tiles, x.gemv_warps, x.box_class, x.source) == (1, 1, "gfx1102", "table:gfx1102")
    x = rs.select(1024, 4096, 1024, (2, 3, 8), override="table=gfx1102")
    assert (x.gemv_tiles, x.gemv_warps, x.box_class, x.source) == (1, 2, "gfx1102", "table:gfx1102")


def test_default_widths_are_sip(gfx1151):
    assert rs.select(4096, 4096).family == "sip"


def test_the_other_box_class_by_name(gfx1151):
    """table=cuda / table=gfx1151 names the other table whatever the build; the
    Launch says whose row it is."""
    x = rs.select(4096, 4096, 1024, (3, 8), override="table=cuda")
    assert (x.gemv_tiles, x.gemv_warps, x.box_class, x.source) == (0, 4, "cuda", "table:cuda")
    x = rs.select(4096, 4096, 1024, (2, 3, 8), override="table=cuda")
    assert (x.gemv_tiles, x.gemv_warps, x.box_class, x.source) == (1, 1, "cuda", "table:cuda")
    x = rs.select(4096, 4096, 1024, (3, 8), override="table=gfx1151")
    assert (x.gemv_tiles, x.gemv_warps, x.box_class, x.source) == (0, 1, "gfx1151", "table:gfx1151")
    with pytest.raises(ValueError, match="unknown key"):  # not a table name: parsed as key=value
        rs.select(4096, 4096, override="table=metal")


def test_table_follows_the_build(monkeypatch):
    monkeypatch.delenv("DRINKME_RADIX_SCHEDULE", raising=False)
    monkeypatch.setattr(rs, "_ARCH", "")
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "cuda")
    assert (rs.select(4096, 4096).gemv_warps, rs.select(4096, 4096).box_class) == (4, "cuda")
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "rocm")
    assert (rs.select(4096, 4096).gemv_warps, rs.select(4096, 4096).box_class) == (1, "gfx1151")


def test_an_amd_part_without_its_own_rows_serves_the_gfx1151_table(monkeypatch):
    """gfx1151 (the rows' own box), an RDNA3 part the sweep has not seen,
    and a ROCm build with no device to read all land on the gfx1151 table."""
    monkeypatch.delenv("DRINKME_RADIX_SCHEDULE", raising=False)
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "rocm")
    for arch in ("gfx1151", "gfx1100", "gfx942", ""):
        monkeypatch.setattr(rs, "_ARCH", arch)
        assert rs.box_class() == "gfx1151"
        assert rs.select(4096, 4096).box_class == "gfx1151"
    # a CUDA build never consults the arch
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "cuda")
    monkeypatch.setattr(rs, "_ARCH", "gfx1102")
    assert rs.box_class() == "cuda"


def test_arch_is_torch_gcnarchname_without_its_feature_suffix(monkeypatch):
    """_arch() reads the device's gcnArchName once on a ROCm build ("" on
    cuda / no device) and drops the ':sramecc+:xnack-' feature suffix some
    builds append, so the table key is the bare target."""
    import torch
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "rocm")
    monkeypatch.setattr(rs, "_ARCH", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(gcnArchName="gfx1102:sramecc+:xnack-"))
    assert rs._arch() == "gfx1102" and rs.box_class() == "gfx1102"
    monkeypatch.setattr(rs, "_ARCH", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert rs._arch() == "" and rs.box_class() == "gfx1151"
    monkeypatch.setattr(rs, "_ARCH", None)
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert rs._arch() == "" and rs.box_class() == "cuda"


def test_backend_detection_matches_torch(monkeypatch):
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", None)
    monkeypatch.setattr(rs, "_ARCH", None)
    import torch
    assert rs._backend_family() == ("rocm" if getattr(torch.version, "hip", None) else "cuda")
    assert rs.box_class() in rs.TABLES
    assert rs.table_for() is rs.TABLES[rs.box_class()]
    assert rs.table_for("cuda") is rs.TABLES["cuda"]
    assert rs.table_for("gfx1102") is rs.TABLES["gfx1102"]


def test_spike_override_is_the_same_for_every_family_and_box(monkeypatch):
    monkeypatch.setenv("DRINKME_RADIX_SCHEDULE", "spike")
    for box in rs.BOX_CLASSES:
        monkeypatch.setattr(rs, "_BACKEND_FAMILY", "cuda" if box == "cuda" else "rocm")
        monkeypatch.setattr(rs, "_ARCH", "" if box == "cuda" else box)
        for widths in ((3, 8), (2, 2, 4, 8)):
            l = rs.select(4096, 4096, 1024, widths)
            assert (l.source, l.box_class) == ("spike", box)
            assert (l.gemv_tiles, l.gemv_warps, l.mc_tiles, l.mc_warps) == (1, 2, 1, 2)


def test_explicit_override_parses_row_and_partial_keys(gfx1151, monkeypatch):
    monkeypatch.setenv("DRINKME_RADIX_SCHEDULE", "tiles=row,warps=2")
    l = rs.select(4096, 4096, 1024, (2, 2, 4, 8))
    assert l.source == "env" and (l.gemv_tiles, l.gemv_warps) == (0, 2)
    assert (l.mc_tiles, l.mc_warps) == (1, 1)  # the family's row for the keys not given
    l = rs.select(4096, 4096, 1024, (3, 8), override="tiles=2,warps=4,mc_tiles=1,mc_warps=2")
    assert (l.gemv_tiles, l.gemv_warps, l.mc_tiles, l.mc_warps) == (2, 4, 1, 2)


def test_override_is_validated_by_name(gfx1151):
    with pytest.raises(ValueError, match="unknown key"):
        rs.select(4096, 4096, override="blocks=2")
    with pytest.raises(ValueError, match="must be 1, 2, 4 or 8"):
        rs.select(4096, 4096, override="warps=3")
    with pytest.raises(ValueError, match="non-negative"):
        rs.select(4096, 4096, override="tiles=-1")


def test_per_class_inline_json(gfx1151):
    spec = json.dumps({"head": {"gemv_tiles": 2, "gemv_warps": 2}, "square": {"gemv_warps": 4, "mc_warps": 2}})
    head = rs.select(151936, 4096, override=spec)
    assert (head.gemv_tiles, head.gemv_warps, head.mc_tiles, head.mc_warps, head.source) == (2, 2, 1, 1, "env")
    square = rs.select(4096, 4096, override=spec)
    assert (square.gemv_tiles, square.gemv_warps, square.mc_tiles, square.mc_warps) == (0, 4, 1, 2)
    # an unnamed class falls back to the table's row
    wide = rs.select(12288, 4096, override=spec)
    assert (wide.gemv_tiles, wide.gemv_warps) == (0, 1)
    # ... for the tensor's family: a scheduled tensor keeps its one-block row where the spec is silent
    square_c = rs.select(4096, 4096, 1024, (2, 2, 4, 8), override=spec)
    assert (square_c.gemv_tiles, square_c.gemv_warps, square_c.mc_warps) == (1, 4, 2)


def test_per_class_file(tmp_path, gfx1151):
    p = tmp_path / "table.json"
    p.write_text(json.dumps({"narrow": {"gemv_tiles": "row", "gemv_warps": 2}}))
    n = rs.select(1024, 4096, override="@" + str(p))
    assert (n.gemv_tiles, n.gemv_warps) == (0, 2)


def test_per_class_falls_back_to_this_box_row(cuda):
    x = rs.select(1024, 4096, override='{"head": {"gemv_warps": 2}}')
    assert (x.gemv_tiles, x.gemv_warps, x.box_class) == (0, 4, "cuda")


@pytest.mark.parametrize("spec", [
    '{"nope": {"gemv_warps": 1}}',
    '{"head": {"warps": 1}}',
    '{"head": {"gemv_warps": 3}}',
    '[1, 2]',
])
def test_per_class_rejects(gfx1151, spec):
    with pytest.raises(ValueError):
        rs.select(4096, 4096, override=spec)


def test_to_device_radix_threads_the_widths(gfx1151):
    """swap.to_device_radix picks the family off the tensor's own widths and
    the backend off the build."""
    from drinkme.codec import radix_pack as rp
    from drinkme.codec.swap import to_device_radix
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from fixtures import realistic_bf16_bits

    U = realistic_bf16_bits(64, 2048, seed=3)
    sip = to_device_radix(rp.pack_array_radix(U, "sip", encoder="numpy"), "cpu")
    gulp = to_device_radix(rp.pack_array_radix(U, "gulp", encoder="numpy"), "cpu")
    assert sip["rx_launch"]["family"] == "sip" and sip["rx_launch"]["gemv_tiles"] == 0
    assert sip["rx_launch"]["box_class"] == "gfx1151"
    assert gulp["rx_launch"]["family"] == "scheduled" and gulp["rx_launch"]["gemv_tiles"] == 1


# ------------------------------------------------------ launch-table honesty --


def test_the_boot_log_names_the_row_that_stood_in_for_an_unmeasured_part():
    """The table is keyed by the backend, so an RX 9070 (gfx1201) or a 4090
    takes the rows measured on gfx1151/gfx1102 or on the five Modal cards
    without anyone having measured them there. The boot line says so, and
    says nothing on a measured part (an "NVIDIA L4" is the L4; an "NVIDIA
    L40S" is its own measured card, not a token match on "L4")."""
    assert rs.MEASURED == {"gfx1151": ("gfx1151", "gfx1102"),
                           "cuda": ("L4", "A10", "A10G", "L40S", "H100")}
    for arch in ("gfx1151", "gfx1102"):
        assert rs.measured_note("gfx1151", arch) is None
    for name in ("NVIDIA L4", "NVIDIA L40S", "NVIDIA A10G", "NVIDIA H100 80GB HBM3"):
        assert rs.measured_note("cuda", name) is None
    note = rs.measured_note("gfx1151", "gfx1201")
    assert note == "no measured row for gfx1201: the gfx1151 rows (measured on gfx1151, gfx1102) stand in"
    note = rs.measured_note("cuda", "NVIDIA GeForce RTX 4090")
    assert note.startswith("no measured row for NVIDIA GeForce RTX 4090: the cuda rows (measured on L4, ")
    assert "this device" in rs.measured_note("cuda", None)
    # the cross-box A/B arm (table=cuda on a gfx1151 box) is unmeasured by construction
    assert rs.measured_note("cuda", "gfx1151") is not None


def test_a_pack_carries_no_box_class_or_launch_schedule(tmp_path):
    """The schedule is chosen at load (rx_launch in the runtime dict), never
    written to the pack, so renaming a box class cannot strand an existing
    pack: neither meta.json nor any tensor file names one."""
    import numpy as np
    from fixtures import radix_dict

    from drinkme.codec.pack import save_pack_dir

    d = tmp_path / "p"
    save_pack_dir(str(d), {"a": radix_dict(8, 64, seed=1),
                           "b": radix_dict(8, 64, seed=2, profile="gulp")},
                  {"hfRepo": "t/t", "revision": None, "dtype": "bf16"})
    meta = (d / "meta.json").read_text()
    for word in ("rx_launch", "box_class", "boxClass", *rs.BOX_CLASSES, "rocm"):
        assert word not in meta, word
    for f in d.glob("*.npz"):
        with np.load(f, allow_pickle=False) as z:
            assert not {"rx_launch", "box_class"} & set(z.files), (f, z.files)
