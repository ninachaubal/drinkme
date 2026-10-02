"""The per-tensor launch schedule of the radix M=1 GEMV and mc kernels:
which (blocks-per-program, warps) a tensor's gemv and mc run at,
chosen at load from the BOX CLASS (the AMD gfx target whose rows serve this
device — gfx1151 or gfx1102 — or cuda), the tensor's PROFILE FAMILY and
its shape class. Triton-free on purpose —
swap.to_device_radix calls it for CPU dicts too, and the CPU route ignores
it.

WHY A TABLE AND NOT A LOAD-TIME MICROBENCH. Every config is a separate
Triton specialization (TILES / SPLITS / num_warps are constexprs), so a
per-tensor microbench at load would compile ~9–12 variants per shape class
and time them while the loader is streaming 12 GB through the same memory
bus at unsettled clocks: slow, noisy, and — worse — non-reproducible: two
boots of the same pack could serve at different launch configs and the
performance record would not be attributable to the code. The table is
measured once per box class under the rotation protocol
(bench/parity_rotation.py; docs/pack-format.md#the-launch-schedule has the
rows and where they were measured), is inspectable, and makes the served
configuration a function of (pack, code) only. The checkpoint bytes are
untouched either way: the schedule is a launch-time choice, never a pack
field.

THE BOX CLASSES (the same sweep in one Modal
container per NVIDIA card — L4, A10, A10G, L40S, H100 — and on the 27B's
shapes on the L40S). On gfx1151 the whole row at ONE warp won every class
for the sip profile; on every NVIDIA card the whole row at FOUR warps won
every class (the L4: two, four within 1–3%), and one warp — gfx1151's row —
is the slowest whole-row config there (1.0–1.3x the t1/w2 time where four
warps are 0.83–0.94x). The rows were measured, not derived: Triton's AMD
backend gives gfx1151 (gfx11) a 32-lane warp like the NVIDIA cards
(`warp_size = 32 if gfx_major >= 10 else 64`, triton 3.7.0), so the two
rows put 32 vs 128 lanes on a row's 1024-weight blocks; why each box class
prefers its count is not established, and the rotation sweeps' outputs are the
table's only justification. The table is keyed by the BOX CLASS first:
on a ROCm torch build (torch.version.hip) an AMD gfx target with its own
measured rows by that target's name ("gfx1151", the Strix Halo; "gfx1102",
the RX 7600 XT: box_class() reads the device's gcnArchName), and "gfx1151"
for any AMD part without rows of its own; "cuda" otherwise.
At the model level the CUDA row is about +17% over the gfx1151 row at M=1
on the A10 and +2–6% on the L4/A10G/H100, the L40S pair inside its own
sample spread.

THE gfx1102 ROW (the same sweep on the RX 7600 XT, sip and gulp at the grid). A third answer: for sip
the whole-row program loses at every warp count and ONE BLOCK per program
at ONE warp — split-K partials and the _finish launch — wins square,
wide, long and head, the narrow class the whole row at two warps; the
gfx1151 row is 1.04–1.08x the pick's time per class. For gulp the `spike`
schedule (one block, two warps) wins every class, the shared scheduled
row 5% behind it. The part has 8.8 GB/s of bandwidth per CU (gfx1151:
6.0), so the decode inside the GEMV binds on compute sooner, and the
picks are the ones that keep more programs in flight.

THE PROFILE FAMILIES (the same rotation sweep run with balanced and then
gulp at the grid on gfx1151). The whole-row-at-one-warp program that won every
class for sip is 1.3x (balanced) to 2.3x (gulp) SLOWER than one block
per program for the profiles with more than two tiers: their decoder
carries the per-block schedule header and one more scan level per tier,
and a program holding a whole row of that at one warp spills. So each
box class's table has two rows per class:

  sip         two tiers (3,8): no schedule header    -> the row program
                (one warp on gfx1151, four on cuda); one block per program
                at one warp on gfx1102 (the whole row at two warps for
                the narrow class)
  scheduled   three or more tiers (balanced (2,3,8), gulp (2,2,4,8)):
              one block per program at one warp, on both platforms — the
              gfx1151 sweep's row; two warps on gfx1102 (gulp at the
              grid there); the family has not been swept on the 8B's
              shapes on an NVIDIA card (the L40S 27B sweep
              put gulp's best at trow/w4 with t1/w1 1.1–1.2x behind
              per class: a candidate row, not a measured 8B one)

`family(widths)` is the rule; a profile with widths this table has not
measured falls into one of the two by its tier count.

THE SHAPE CLASSES (R rows, NB = ceil(C / 1024) blocks per row), named for
Qwen3-8B's tensors but defined by the rule, so other checkpoints fall into
one of them:

  head    R >= 32768                the vocabulary projection (151,936 x 4,096)
  narrow  R <= 1024                 k_proj / v_proj (1,024 x 4,096)
  long    NB >= 8                   down_proj (4,096 x 12,288): a long row
  wide    R >= 8192                 gate_proj / up_proj (12,288 x 4,096)
  square  everything else           q_proj / o_proj (4,096 x 4,096)

tiles = 0 means the whole row in one program (no split-K partials, no
_finish launch, bias fused in-kernel); tiles = t > 0 means t consecutive
blocks per program with split-K partials reduced by _finish (the `spike`
override is tiles=1, warps=2 for every tensor and profile).

DRINKME_RADIX_SCHEDULE overrides the table for A/B runs:
  spike               tiles=1, warps=2 everywhere — the fixed-schedule
                      control arm of the rotation sweeps
  table               this box class's table (the default): the gfx1102
                      rows on that part, the gfx1151 rows on any other ROCm
                      torch build, the CUDA rows otherwise
  table=gfx1151 | table=gfx1102 | table=cuda
                      another box class's table (the cross-box A/B arm:
                      table=gfx1151 on the RX 7600 XT)
  tiles=<n|row>,warps=<n>[,mc_tiles=<n|row>,mc_warps=<n>]
                      one explicit config for every tensor (experiments)
  {"head": {...}, "square": {...}, ...}   or   @/path/to/table.json
                      a per-class table (another box's row, e.g. the one
                      bench/modal_nvidia2.py sweeps per NVIDIA card),
                      applied to every profile family: each class names
                      any subset of gemv_tiles / gemv_warps / mc_tiles /
                      mc_warps (0 or "row" = whole row); what a class
                      leaves unsaid comes from this box's row for the
                      tensor's family
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, asdict

SPIKE = dict(gemv_tiles=1, gemv_warps=2, mc_tiles=1, mc_warps=2)

CLASSES = ("head", "narrow", "long", "wide", "square")
FAMILIES = ("sip", "scheduled")
BACKEND_FAMILIES = ("rocm", "cuda")  # the torch build's hardware family (torch.version.hip or not)
# the tables' keys — BOX CLASSES: an AMD gfx target with its own measured rows
# (box_class() below picks it off the device's gcnArchName; gfx1151 stands in
# for any other AMD part), or cuda
BOX_CLASSES = ("gfx1151", "gfx1102", "cuda")

# sip on gfx1151 — filled from the rotation sweeps (bench/parity_rotation.py;
# Strix Halo, gfx1151, rotation protocol with the pass primer,
# 20 passes x 3 repeats, the `spike` schedule t1/w2 as the in-session
# control).
#   gemv: the whole row in one program at ONE warp won every class —
#         0.936x (square) / 0.903x (narrow) / 0.916x (wide) / 0.950x (long,
#         tied with t2/w1 within 0.1%) / 0.907x (head) of t1/w2's time.
#   mc:   one block per program at ONE warp beat t1/w2's two warps at
#         every M and class (M=2 0.97-0.99x, M=4 0.94x, M=6 0.86-0.91x,
#         M=8 0.87-0.92x); multi-block mc programs spill their eight
#         accumulators (t2/w1: 6-13x slower at M >= 4).
_SIP_GFX1151 = dict(gemv_tiles=0, gemv_warps=1, mc_tiles=1, mc_warps=1)
# sip on cuda — filled from the rotation sweeps on five Modal cards (the
# same protocol):
#   gemv: the whole row at FOUR warps on the A10 (three draws), A10G, L40S,
#         H100 and the 27B's shapes on the L40S, every class; the L4 picked
#         two with four within 1-3%.
#   mc:   t1/w1 on every card (the H100's picks wander among t1/w2, t1/w4,
#         t1/w8 within 1-3% of it).
_SIP_CUDA = dict(gemv_tiles=0, gemv_warps=4, mc_tiles=1, mc_warps=1)
# scheduled — from the gfx1151 rotation sweep with balanced and with gulp
# at the grid, same protocol.
#   gemv: one block per program at ONE warp is balanced's pick on
#         square/narrow/wide/long (0.95x its t1/w2; head: trow/w2 by
#         0.2%, t1/w1 within 1%) and ties t1/w2 within 1% on every class for
#         gulp; the whole-row program at one warp is 1.34x (balanced) and
#         2.3x (gulp) slower per layer, 2.35x on gulp's lm_head.
#   mc:   t1/w1 again: balanced 0.95/0.91/0.90/0.87x its t1/w2 at M=2/4/6/8.
# The same row stands in for cuda (module docstring: not swept on the 8B
# there).
_SCHEDULED_ROW = dict(gemv_tiles=1, gemv_warps=1, mc_tiles=1, mc_warps=1)
# gfx1102 (Navi33: the RX 7600 XT / 7600 / 7700S / W7600) — filled from
# a rotation sweep on an RX 7600 XT 16 GB
# (the same protocol: primer, 20 passes x 3 repeats, layers 0 + 18 + lm_head,
# tiles {1, 2, 4, row} x warps {1, 2, 4, 8}, sip AND gulp at the grid, the
# `spike` schedule in-session; wall
# 277.8 GB/s).
#   sip gemv: ONE block per program at ONE warp (split-K partials + _finish)
#         won square / wide / long / head at 0.977-0.981x t1/w2, and the
#         narrow class (k/v_proj) the whole row at TWO warps at 0.972x with
#         t1/w1 1.6% behind; gfx1151's whole-row-at-one-warp row is the
#         wrong row here — 1.062x (square) / 1.035x (narrow) / 1.079x (wide) /
#         1.058x (long) / 1.084x (head) the pick's time, 6.4% per token at
#         the kernel level (36 layers + lm_head). Eight warps lose 40-80%
#         on every class. The card decodes at 0.57-0.65 of its wall at the
#         pick (gfx1151: 0.8+): the decode binds on compute sooner on 32 CUs
#         behind 278 GB/s.
#   sip mc:  t1/w1 again, 0.88-0.91x t1/w2 summed over M = 2, 4, 8 (M=8:
#         0.79-0.89x); t2/w1 spills as on gfx1151 (6-59x at M=8).
#   scheduled (gulp at the grid): t1/w2 — the `spike` schedule — won every
#         class, the gfx1151 scheduled row t1/w1 1.050-1.059x behind it and
#         the whole row at one warp 2.0-2.1x; gulp's best is 1.77-1.91x
#         sip's per class (0.25-0.33 of the wall): the profile is
#         compute-bound on this part. mc as sip's pick (measured for gulp
#         at t1/w1 only, as the control).
_SIP_GFX1102 = {cls: dict(gemv_tiles=1, gemv_warps=1, mc_tiles=1, mc_warps=1) for cls in CLASSES}
_SIP_GFX1102["narrow"] = dict(gemv_tiles=0, gemv_warps=2, mc_tiles=1, mc_warps=1)
_SCHEDULED_GFX1102 = dict(gemv_tiles=1, gemv_warps=2, mc_tiles=1, mc_warps=1)
TABLES = {
    "gfx1151": {
        "sip": {cls: dict(_SIP_GFX1151) for cls in CLASSES},
        "scheduled": {cls: dict(_SCHEDULED_ROW) for cls in CLASSES},
    },
    "gfx1102": {
        "sip": {cls: dict(row) for cls, row in _SIP_GFX1102.items()},
        "scheduled": {cls: dict(_SCHEDULED_GFX1102) for cls in CLASSES},
    },
    "cuda": {
        "sip": {cls: dict(_SIP_CUDA) for cls in CLASSES},
        "scheduled": {cls: dict(_SCHEDULED_ROW) for cls in CLASSES},
    },
}

# THE TWIN'S OWN ROWS: the bench's order-matched twin (swap.to_device_twin:
# the radix kernels with RAW=True over the raw bf16 weight) at the gemv
# schedule that reads raw bf16 fastest on this box class, per shape class.
# The rows above are tuned for the DECODER; a raw bf16 row wants its own
# (see select_twin). Filled from bench/parity_rotation.py --radix-arms sip
# twin; a box class without rows here runs the twin at select()'s row.
#   gfx1151 (Strix Halo, the same protocol: primer, 20 passes x
#   3 repeats, tiles {1, 2, 4, row} x warps {1, 2, 4, 8}, sip beside the
#   twin; every layer of Qwen3-0.6B and -1.7B, layers 0/9/18/27 + lm_head
#   of Qwen3-8B on a quiet box, layers 0-3 + lm_head of Qwen3.8-27B beside
#   a CPU build, its least-disturbed repeat): the whole row at EIGHT warps
#   won or tied every class of every model. Summed over the set it is
#   0.671x (0.6B), 0.630x (1.7B), 0.814x (8B) and 0.953x (27B) the time
#   of sip's row (the whole row at one warp), within 0.7% of each model's
#   per-class best (one block per program at eight warps on four of the
#   8B's classes); four warps are within 1.5% of it. sip's own row stayed
#   its best on every model (the decoder binds on compute, a raw load does
#   not).
TWIN_TABLES: dict = {
    "gfx1151": {cls: dict(gemv_tiles=0, gemv_warps=8) for cls in CLASSES},
}

# The parts each box class's rows were MEASURED on. The table is keyed by the
# box class, so every other part of that class takes the same rows
# unmeasured; measured_note() says so in the boot log rather than letting a
# row measured on a Strix Halo pass for a measurement on an RX 9070.
#   gfx1151: Strix Halo (gfx1151) and
#            RX 7600 XT 16 GB (gfx1102) — the two
#            AMD machines that ran the rotation protocol
#   cuda: the five Modal cards (L4, A10, A10G, L40S, H100)
MEASURED = {
    "gfx1151": ("gfx1151", "gfx1102"),
    "cuda": ("L4", "A10", "A10G", "L40S", "H100"),
}

_BACKEND_FAMILY: str | None = None
_ARCH: str | None = None


def box_arch() -> str | None:
    """The part the launch table is being applied to: the gfx target on a
    ROCm build (gcnArchName), the device name on CUDA; None without torch or
    a device. Read once at load, never at import."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        if getattr(torch.version, "hip", None):
            return getattr(props, "gcnArchName", None) or props.name
        return props.name
    except Exception:  # noqa: BLE001 — a name is a nicety, not a gate
        return None


def is_measured(box: str, arch: str | None) -> bool:
    """Was this box class's table measured on `arch`? gfx targets match
    exactly; a CUDA device name matches on a whole token ("NVIDIA L4" is
    the L4, "NVIDIA L40S" is not)."""
    if not arch:
        return False
    measured = MEASURED.get(box, ())
    if box.startswith("gfx"):
        return arch in measured
    tokens = set(arch.replace("-", " ").split())
    return any(m in tokens for m in measured)


def measured_note(box: str, arch: str | None) -> str | None:
    """None when the table's rows were measured on this part; else the
    boot-log clause naming the class that stood in — `no measured row for
    gfx1201: the gfx1151 rows (measured on gfx1151, gfx1102) stand in`."""
    if is_measured(box, arch):
        return None
    measured = ", ".join(MEASURED.get(box, ()))
    return (f"no measured row for {arch or 'this device'}: the {box} rows "
            f"(measured on {measured}) stand in")


def _backend_family() -> str:
    """The torch build's hardware family: "rocm" on a ROCm build, else
    "cuda" — resolved once, lazily, so this module stays importable without
    torch (the CPU route ignores the schedule either way)."""
    global _BACKEND_FAMILY
    if _BACKEND_FAMILY is None:
        try:
            import torch
            _BACKEND_FAMILY = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        except Exception:  # noqa: BLE001 — no torch: the schedule is moot
            _BACKEND_FAMILY = "cuda"
    return _BACKEND_FAMILY


def _arch() -> str:
    """The device's gfx target on a ROCm build ("gfx1151", "gfx1102": the
    `gcnArchName` torch reports, its feature suffix after ':' dropped), ""
    when there is no device or the build is not ROCm. Resolved once."""
    global _ARCH
    if _ARCH is None:
        _ARCH = ""
        if _backend_family() == "rocm":
            try:
                import torch
                if torch.cuda.is_available():
                    name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
                    _ARCH = name.split(":", 1)[0]
            except Exception:  # noqa: BLE001 — no device to read: the family's row
                _ARCH = ""
    return _ARCH


def box_class() -> str:
    """The table this machine serves from: on a ROCm build its gfx target's
    own when the table has one (gfx1151, gfx1102), else the gfx1151 rows;
    cuda otherwise."""
    if _backend_family() == "rocm":
        arch = _arch()
        return arch if arch in TABLES else "gfx1151"
    return "cuda"


def table_for(box: str | None = None) -> dict:
    """The (family -> class -> row) table of a box class; this box's by
    default."""
    return TABLES[box or box_class()]


@dataclass(frozen=True)
class Launch:
    gemv_tiles: int  # 0 = whole row per program
    gemv_warps: int
    mc_tiles: int
    mc_warps: int
    shape_class: str
    family: str  # "sip" | "scheduled"
    box_class: str  # the box class whose table the row came from (or would have): "gfx1151" | "gfx1102" | "cuda"
    source: str  # "table" | "table:<box class>" | "spike" | "env"

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        """`gemv row/w1 mc 1/w1` — the boot-log form."""
        g = "row" if self.gemv_tiles == 0 else str(self.gemv_tiles)
        m = "row" if self.mc_tiles == 0 else str(self.mc_tiles)
        return f"gemv {g}/w{self.gemv_warps} mc {m}/w{self.mc_warps}"


def shape_class(R: int, C: int, block_size: int = 1024) -> str:
    nb = (int(C) + int(block_size) - 1) // int(block_size)
    R = int(R)
    if R >= 32768:
        return "head"
    if R <= 1024:
        return "narrow"
    if nb >= 8:
        return "long"
    if R >= 8192:
        return "wide"
    return "square"


def family(widths) -> str:
    """Two tiers (the sip profile) -> "sip"; three or more (balanced,
    gulp: the profiles whose runtime dict carries rx_schedule) ->
    "scheduled"."""
    return "sip" if len(tuple(widths)) <= 2 else "scheduled"


def _parse_override(spec: str) -> dict:
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if key in ("tiles", "gemv_tiles"):
            out["gemv_tiles"] = 0 if value == "row" else int(value)
        elif key in ("warps", "gemv_warps"):
            out["gemv_warps"] = int(value)
        elif key == "mc_tiles":
            out["mc_tiles"] = 0 if value == "row" else int(value)
        elif key == "mc_warps":
            out["mc_warps"] = int(value)
        else:
            raise ValueError(f"DRINKME_RADIX_SCHEDULE: unknown key {key!r} in {spec!r}")
    return out


def select(R: int, C: int, block_size: int = 1024, widths=(3, 8),
           override: str | None = None) -> Launch:
    """The launch config for a (R, C) radix tensor at `widths`: this box
    class's table row for its (profile family, shape class), or the
    DRINKME_RADIX_SCHEDULE override."""
    cls = shape_class(R, C, block_size)
    fam = family(widths)
    box = box_class()
    spec = os.environ.get("DRINKME_RADIX_SCHEDULE", "table") if override is None else override
    spec = (spec or "table").strip()
    if spec == "table":
        return Launch(**TABLES[box][fam][cls], shape_class=cls, family=fam, box_class=box, source="table")
    if spec.startswith("table=") and spec[6:] in TABLES:
        other = spec[6:]
        return Launch(**TABLES[other][fam][cls], shape_class=cls, family=fam, box_class=other, source="table:" + other)
    if spec == "spike":
        return Launch(**SPIKE, shape_class=cls, family=fam, box_class=box, source="spike")
    cfg = dict(TABLES[box][fam][cls])
    if spec.startswith("{") or spec.startswith("@"):
        cfg.update(_per_class_table(spec).get(cls, {}))
    else:
        cfg.update(_parse_override(spec))
    _validate(cfg)
    return Launch(**cfg, shape_class=cls, family=fam, box_class=box, source="env")


TWIN_ENV = "DRINKME_TWIN_SCHEDULE"


def select_twin(R: int, C: int, block_size: int = 1024, widths=(3, 8),
                override: str | None = None) -> Launch:
    """The bench's twin's launch schedule. Its gemv fields come from this
    box class's TWIN_TABLES row for the shape class, and its mc fields from
    the compressed tensor's select() row: the bench times the twin's
    decode at M=1 only. It falls back to select()'s row, the compressed
    tensor's own, when the box class has no twin rows, when
    DRINKME_TWIN_SCHEDULE=served asks for that (the A/B against the
    order-matched-at-the-served-row twin), and when DRINKME_RADIX_SCHEDULE
    names anything but the table (an experiment's schedule applies to both
    arms). `override` is select()'s: "table=<box>" names another box
    class's twin rows (the gate's cross-box rows)."""
    base = select(R, C, block_size, widths, override)
    spec = os.environ.get("DRINKME_RADIX_SCHEDULE", "table") if override is None else override
    spec = (spec or "table").strip()
    if spec == "table":
        box = base.box_class
    elif spec.startswith("table=") and spec[6:] in TABLES:
        box = spec[6:]
    else:
        return base
    mode = os.environ.get(TWIN_ENV, "twin").strip() or "twin"
    if mode not in ("twin", "served"):
        raise ValueError(f"{TWIN_ENV}={mode!r}: twin (the twin's own rows) or served (the compressed row)")
    if mode == "served" or box not in TWIN_TABLES:
        return base
    row = TWIN_TABLES[box][base.shape_class]
    return Launch(gemv_tiles=row["gemv_tiles"], gemv_warps=row["gemv_warps"], mc_tiles=base.mc_tiles,
                  mc_warps=base.mc_warps, shape_class=base.shape_class, family=base.family, box_class=box,
                  source="twin-table" if spec == "table" else "twin-table:" + box)


@functools.lru_cache(maxsize=8)
def _per_class_table(spec: str) -> dict:
    """A per-class override (module docstring): inline JSON or @path. Cached
    on the spec string — the loader calls select() once per tensor."""
    text = spec
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            text = f.read()
    table = json.loads(text)
    if not isinstance(table, dict):
        raise ValueError("DRINKME_RADIX_SCHEDULE: a per-class table must be a JSON object")
    out = {}
    for cls, row in table.items():
        if cls not in CLASSES:
            raise ValueError(f"DRINKME_RADIX_SCHEDULE: unknown shape class {cls!r} (known: {sorted(CLASSES)})")
        if not isinstance(row, dict):
            raise ValueError(f"DRINKME_RADIX_SCHEDULE: class {cls!r} must map to an object")
        cfg = {}
        for key, value in row.items():
            if key not in SPIKE:
                raise ValueError(f"DRINKME_RADIX_SCHEDULE: unknown key {key!r} for class {cls!r}")
            cfg[key] = 0 if value == "row" else int(value)
        _validate({**SPIKE, **cfg})  # the given keys' values; the rest is filled per family at select()
        out[cls] = cfg
    return out


def _validate(cfg: dict) -> None:
    for key in ("gemv_tiles", "mc_tiles"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"radix schedule: {key} must be a non-negative int (0 = whole row)")
    for key in ("gemv_warps", "mc_warps"):
        if cfg[key] not in (1, 2, 4, 8):
            raise ValueError(f"radix schedule: {key} must be 1, 2, 4 or 8")


for _table in TABLES.values():
    for _rows in _table.values():
        for _cfg in _rows.values():
            _validate(_cfg)
for _box, _rows in TWIN_TABLES.items():
    if _box not in TABLES or set(_rows) != set(CLASSES):
        raise ValueError(f"TWIN_TABLES[{_box!r}]: a known box class with a row for every shape class")
    for _cfg in _rows.values():
        _validate({**SPIKE, **_cfg})
_validate(SPIKE)
