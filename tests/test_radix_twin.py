"""The bench's order-matched twin (swap.RadixTwinLinear over
swap.to_device_twin), CPU only: that it is launched the way the compressed
tensor is — at any one launch schedule the same program decomposition and
the same scalars into the kernel, the same route at every M — so the only
difference left is the weight read; and that its own launch row is
radix_schedule.select_twin's (the box class's twin row for the gemv where
it has one, the compressed tensor's row otherwise). Whether the kernels'
outputs are then bitwise equal is a device question:
bench/radix_twin_bitpin.py."""
import numpy as np
import pytest
import torch

from drinkme import arms
from drinkme.codec import radix_pack as rp
from drinkme.codec import radix_schedule as rs
from drinkme.codec import swap
from drinkme.codec.ops import MC_MAX
from drinkme.codec.swap import (RadixCompressedLinear, RadixTwinLinear, RawLinear, make_module,
                                make_twin, to_device_radix, to_device_twin)

PROFILES = ("sip", "balanced", "gulp")
SCHEDULES = ("table", "table=gfx1151", "table=gfx1102", "table=cuda", "spike",
             "tiles=2,warps=4,mc_tiles=1,mc_warps=2", "tiles=row,warps=8")


@pytest.fixture(params=("gfx1151", "gfx1102", "cuda"))
def box(request, monkeypatch):
    """Each box class's tables, whatever box runs the suite."""
    monkeypatch.setattr(rs, "_BACKEND_FAMILY", "cuda" if request.param == "cuda" else "rocm")
    monkeypatch.setattr(rs, "_ARCH", "" if request.param == "cuda" else request.param)
    monkeypatch.delenv("DRINKME_RADIX_SCHEDULE", raising=False)
    monkeypatch.delenv(rs.TWIN_ENV, raising=False)
    rs._per_class_table.cache_clear()
    _clear_decoder()
    yield request.param
    _clear_decoder()


def _clear_decoder():
    """radix_ops._decoder is cached per widths and reads the backend family
    the fixture sets: a box's tests see that family's decoders."""
    try:
        from drinkme.codec import radix_ops
    except ImportError:  # no triton: the tests that launch skip
        return
    radix_ops._decoder.cache_clear()


def _weight(R, C, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(R, C, generator=g) * 0.02).to(torch.bfloat16)


def _pair(w, profile):
    """The compressed module and its twin for one tensor, on the CPU."""
    pack = rp.pack_weight_radix(w, profile, encoder="numpy")
    assert pack["codec"] == "radix"
    comp = make_module(pack, None, "cpu")
    twin, _ = make_twin(w, None, "cpu", codec="radix", widths=pack["widths"])
    return pack, comp, twin


def _twin_row(radix_launch: dict, schedule: str, box: str, mode: str) -> dict:
    """What select_twin should give: the compressed tensor's row, with the
    gemv fields of the twin table the schedule names when there is one."""
    table_box = box if schedule == "table" else schedule[6:] if schedule.startswith("table=") else None
    if mode == "served" or table_box not in rs.TWIN_TABLES:
        return radix_launch
    row = rs.TWIN_TABLES[table_box][radix_launch["shape_class"]]
    return dict(radix_launch, gemv_tiles=row["gemv_tiles"], gemv_warps=row["gemv_warps"], box_class=table_box,
                source="twin-table" if schedule == "table" else f"twin-table:{table_box}")


@pytest.mark.parametrize("mode", ("twin", "served"))
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_the_twin_gets_select_twins_row(box, profile, schedule, mode, monkeypatch):
    """to_device_twin's rx_launch is select_twin's: to_device_radix's row
    for the same tensor, except the gemv fields where the schedule is a
    table with twin rows (this box's under "table", the named box's under
    "table=<box>") and DRINKME_TWIN_SCHEDULE is not "served"; the mc
    fields are always the compressed tensor's. Every box class, profile
    family and DRINKME_RADIX_SCHEDULE form the gate sweeps."""
    monkeypatch.setenv("DRINKME_RADIX_SCHEDULE", schedule)
    monkeypatch.setenv(rs.TWIN_ENV, mode)
    for shape in ((1024, 1024), (2048, 1024), (1024, 3584)):  # narrow, square, a ragged row
        w = _weight(*shape)
        pack = rp.pack_weight_radix(w, profile, encoder="numpy")
        radix = to_device_radix(pack, "cpu")
        twin = to_device_twin(w, pack["widths"], "cpu")
        assert twin["rx_launch"] == _twin_row(radix["rx_launch"], schedule, box, mode), (shape, profile, schedule)
        assert (twin["rx_launch"]["mc_tiles"], twin["rx_launch"]["mc_warps"]) == \
            (radix["rx_launch"]["mc_tiles"], radix["rx_launch"]["mc_warps"])
        assert (twin["R"], twin["C"], twin["widths"], twin["block_size"], twin["NBK"]) == \
            (radix["R"], radix["C"], radix["widths"], radix["block_size"], radix["NBK"])


@pytest.mark.parametrize("profile", PROFILES)
def test_every_shape_class_selects_the_twin_row(box, profile):
    """The shape classes too big to pack in a unit test (head, wide, long):
    to_device_twin makes select_twin's call, which is select()'s row with
    the box's twin gemv row over it."""
    widths = rp.widths_of(profile)
    for R, C in ((32768, 1024), (8192, 1024), (1024, 8192)):
        w = torch.zeros(R, C, dtype=torch.bfloat16)
        want = _twin_row(rs.select(R, C, rp.RADIX_BLOCK, widths).as_dict(), "table", box, "twin")
        assert to_device_twin(w, widths, "cpu")["rx_launch"] == want


def test_the_gfx1151_twin_row_is_the_whole_row_at_eight_warps(monkeypatch):
    """The measured row (radix_schedule.TWIN_TABLES' comment): every shape
    class, sip or scheduled, the whole row at eight warps; no other box
    class has twin rows, so they run the twin at the compressed row."""
    monkeypatch.delenv("DRINKME_RADIX_SCHEDULE", raising=False)
    monkeypatch.delenv(rs.TWIN_ENV, raising=False)
    assert set(rs.TWIN_TABLES) == {"gfx1151"}
    for cls in rs.CLASSES:
        assert rs.TWIN_TABLES["gfx1151"][cls] == {"gemv_tiles": 0, "gemv_warps": 8}
    for widths in ((3, 8), (2, 2, 4, 8)):
        got = rs.select_twin(4096, 4096, 1024, widths, override="table=gfx1151")
        assert (got.gemv_tiles, got.gemv_warps, got.source) == (0, 8, "twin-table:gfx1151")
        assert rs.select_twin(4096, 4096, 1024, widths, override="table=gfx1102") == \
            rs.select(4096, 4096, 1024, widths, override="table=gfx1102")


def test_the_twin_schedule_knob_refuses_a_typo(monkeypatch):
    monkeypatch.setenv(rs.TWIN_ENV, "severd")
    with pytest.raises(ValueError, match="DRINKME_TWIN_SCHEDULE"):
        rs.select_twin(4096, 4096, 1024, (3, 8), override="table=gfx1151")


def test_the_twin_holds_the_raw_weight_row_major():
    w = _weight(1024, 2048)
    p = to_device_twin(w, (3, 8), "cpu")
    assert p["weight"].dtype == torch.bfloat16 and p["weight"].is_contiguous()
    assert torch.equal(p["weight"].view(torch.int16), w.view(torch.int16))
    assert p["codec"] == swap.TWIN


def test_the_twin_is_the_compressed_modules_routing():
    """RadixTwinLinear overrides only the weight producers (dense, CPU):
    the router, the fused M=1 / mc forward, the epilogue and the GEMV arm
    bodies are RadixCompressedLinear's own functions."""
    for attr in ("_route", "_route_rows", "_forward", "_epilogue", "_gemv_row", "_gemv_cols",
                 "forward"):
        assert getattr(RadixTwinLinear, attr) is getattr(RadixCompressedLinear, attr), attr
    assert issubclass(RadixTwinLinear, swap.SWAPPED_LINEAR_TYPES)


@pytest.mark.parametrize("dense_min", (None, "0", "4"))
def test_routes_match_at_every_m(dense_min, monkeypatch):
    if dense_min is None:
        monkeypatch.delenv("DRINKME_PREFILL_DENSE_MIN", raising=False)
    else:
        monkeypatch.setenv("DRINKME_PREFILL_DENSE_MIN", dense_min)
    monkeypatch.setattr(swap, "_DENSE_MIN", None)
    _, comp, twin = _pair(_weight(1024, 1024), "sip")
    for m in range(1, 3 * MC_MAX):
        assert twin._route_rows(m) == comp._route_rows(m), m
    x = torch.zeros(1, 5, 1024, dtype=torch.bfloat16)
    assert twin._route(x) == comp._route(x) == "cpu"


@pytest.mark.parametrize("profile", PROFILES)
def test_dense_and_cpu_weights_are_the_raw_weight(profile):
    """The dense arm (M >= GEMM_MIN_ROWS) is F.linear over the resident raw
    weight — the bytes the compressed arm's decode produces — with no
    transient copy; on the CPU both modules are stock's F.linear over the
    exact weight, so their outputs are bitwise equal at every M."""
    w = _weight(1024, 1024)
    b = _weight(1, 1024, seed=1)[0]
    pack = rp.pack_weight_radix(w, profile, encoder="numpy")
    comp = make_module(pack, b, "cpu")
    twin, _ = make_twin(w, b, "cpu", codec="radix", widths=pack["widths"])
    assert twin._dense_weight() is twin.p["weight"]
    assert torch.equal(twin._cpu_weight().view(torch.int16), comp._cpu_weight().view(torch.int16))
    for m in (1, 2, MC_MAX, MC_MAX + 1, 64):
        x = _weight(m, 1024, seed=m)
        assert torch.equal(twin(x), comp(x)), m


def test_a_raw_fallback_slot_is_the_compressed_arms_raw_linear():
    w = _weight(1024, 1024)
    U = w.view(torch.int16).numpy().view(np.uint16)
    comp = make_module(rp.raw_dict(U), None, "cpu")
    twin, stat = make_twin(w, None, "cpu", codec="raw")
    assert stat is None and type(twin) is RawLinear is type(comp)
    assert torch.equal(twin.p["weight"].view(torch.int16), comp.p["weight"].view(torch.int16))


def test_make_twin_refuses_what_it_cannot_match():
    w = _weight(1024, 1024)
    with pytest.raises(ValueError, match="needs its widths"):
        make_twin(w, None, "cpu", codec="radix")
    with pytest.raises(ValueError, match="no twin for codec"):
        make_twin(w, None, "cpu", codec="fp8")
    with pytest.raises(ValueError, match="to_device_twin"):
        RadixTwinLinear(to_device_radix(rp.pack_weight_radix(w, "sip", encoder="numpy"), "cpu"), None)


def test_twin_plan_is_each_slots_codec_and_widths():
    stats = [{"name": "a", "codec": "radix", "profile": "gulp", "widths": [2, 2, 4, 8]},
             {"name": "b", "codec": "raw", "profile": None, "widths": None}]
    assert arms.twin_plan(stats) == {"a": {"codec": "radix", "widths": [2, 2, 4, 8]},
                                     "b": {"codec": "raw", "widths": None}}


def test_the_twin_is_counted_as_its_raw_weight():
    _, _, twin = _pair(_weight(1024, 1024), "sip")
    assert arms.linear_kind(twin) == "drinkme_twin"
    model = torch.nn.Sequential(twin)
    got = arms.decode_read_bytes(model)
    assert got["packed_bytes"] == 1024 * 1024 * 2 and got["counts"] == {"drinkme_twin": 1}


# -- the launch arguments: radix_ops (imports triton; nothing runs on a device) --


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_the_kernels_get_the_same_scalars_and_decomposition(profile, schedule, box, monkeypatch):
    """radix_ops._args / _splits for a twin dict: the kernel's constexpr
    scalars (C, mantissa/exponent bits, WIDTHS, B) are the compressed
    tensor's; DECODER is the scheduled decoder's for the twin (under
    RAW=True no decoder runs; the index only picks radix_kernel_gpu._widen's
    path) and radix_ops._decoder's for the compressed tensor, a lean one
    where it applies; at the compressed tensor's launch row so is
    the program decomposition (tiles, splits, warps) per arm — at its own
    row the twin decomposes by that row; RAW is set, and `data` is the raw
    weight."""
    pytest.importorskip("triton")
    from drinkme.codec import radix_ops as ro

    monkeypatch.setenv("DRINKME_RADIX_SCHEDULE", schedule)
    for shape in ((1024, 1024), (1024, 3584), (2048, 5120)):
        w = _weight(*shape)
        pack = rp.pack_weight_radix(w, profile, encoder="numpy")
        radix = to_device_radix(pack, "cpu")
        twin = to_device_twin(w, pack["widths"], "cpu")
        assert ro._raw(twin) and not ro._raw(radix)
        ta, ra = ro._args(twin), ro._args(radix)
        assert ta[5:-1] == ra[5:-1], shape
        assert ta[-1] == ro.DECODER_SCHEDULED
        assert ra[-1] == ro._decoder(tuple(pack["widths"]), shape[1] % rp.RADIX_BLOCK == 0)
        assert ta[0] is twin["weight"] and ra[0] is radix["rx_data"]
        at_radix = dict(twin, rx_launch=radix["rx_launch"])
        for arm in ("gemv", "mc"):
            assert ro._splits(at_radix, arm) == ro._splits(radix, arm), (shape, arm)
        own = twin["rx_launch"]
        nb = -(-shape[1] // rp.RADIX_BLOCK)
        tiles = min(own["gemv_tiles"] or nb, nb)
        assert ro._splits(twin, "gemv") == (nb, tiles, -(-nb // tiles), own["gemv_warps"])
