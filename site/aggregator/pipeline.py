"""The aggregator's rules: published measurement records -> the site's points.json.

    records -> validate -> filters F2-F5 -> vote collapse -> points.json

Pure functions, no network: sources.py fetches the records, follow.py keeps
them current, and both hand them here. The same validator decides the
committed site build and the live service, because validation IS
site/data/build_points.py's `admit` (identity fields, then the version policy
of docs/versioning.md), imported, not copied; and each surviving record is
turned into a plotted point by build_points' `wrap_measurement`.

Every record that does not reach the page is named in `excluded` with the
step that dropped it (`filter`), the reason, and where a number decided it,
the numbers (`numbers`).

The filters, in order (a record is excluded by the first it fails):

- F1 was retired on 2026-09-29: it kept `environment.platform == "metal"` off
  the chart until Apple silicon's whole-model results were released, and they
  are (docs/metal.md). The numbers F2-F5 keep are not renumbered, and no
  record is excluded under "F1" any more.
- F2 bit-exact gate: `raw.verified_tensors == raw.swapped_linears`, both
  counts present, and `raw.verification` (when present) one of the ways
  bench verifies ("pack", "roundtrip"). bench writes no record otherwise
  (bench.py prints the gate), so a record that says anything else did not
  come from an unmodified bench.
- F3 physics: a decode step reads the arm's weights once, so an arm's decode
  tok/s cannot exceed its bandwidth bound, the machine's own read bandwidth
  over the bytes one decode step reads, by more than `tolerance` (default
  1.10). Speculation emits several tokens per step: a speculative metric
  (`<arm>_spec_decode_tok_s_<prompt>`) is held to (SPEC_DEPTH + 1) x its
  arm's bound x `tolerance`, the most one verify step can emit. The bytes are the record's
  `<arm>_decode_read_gb` (the Linears plus one embedding row: not a vision
  tower, which decode never reads; the lexicon's `metrics` description).
  A record without that metric (the Apple silicon bench does not record it
  yet) is checked against the arm's resident footprint, `<arm>_weights_gb`:
  that is the larger number, so its bound is the stricter one. A
  `<arm>_decode_read_gb` that is present but not a positive number gets no
  fallback; the arm is uncheckable. Checked for every
  arm the record claims a decode speed for: the compressed arm and, when
  they ran, the stock arm and the twin (bench's diagnostic arm, which the
  page does not show; a record with any arm faster than physics allows did
  not come from a working bench). An arm that claims a decode speed without
  the numbers to check it against is excluded as uncheckable.
- F4 denylist: DIDs and at-uris listed in a file the deployment supplies;
  empty by default.
- F5 baseline suspect, in two parts. A record's baseline efficiency is its
  baseline arm's decode tok/s over that arm's bandwidth bound (F3's bound,
  the same denominator); the baseline arm is the stock arm (`Settings.baseline_arm`),
  the one the page divides by. The speed ratio is all network data feeds
  (the fit table is written in the page), and a baseline that is broken makes
  it meaningless. But a slow baseline is not by itself broken: on small
  models decode is launch-bound, not bandwidth-bound. On Strix Halo an
  honest stock arm reaches about 68% of its bound on a 1.7B and 87% on an
  8B through gfx1151's stock GEMV, the twin's Triton kernel
  (docs/serve-kernels.md); about 39%, 50% and 75% on a 0.6B, 1.7B and 8B
  through torch.mv under hipBLASLt (DRINKME_STOCK_GEMV=mv, the stock GEMV
  before it); and about 21%, 23% and 50% where the one-row call is
  PyTorch's default F.linear (DRINKME_STOCK_GEMV=linear).
  So a baseline is judged against its peers:
  - the absolute floor, per record, before the collapse: under
    `baseline_floor` (default 0.10) of its bound is implausible whatever the
    peers show, and the record is dropped;
  - relative, after the collapse: in a cell with at least
    `baseline_min_contributors` (default 3) contributors, a contributor's
    collapsed point under `baseline_relative` (default 0.6) x the cell's
    median efficiency is dropped, with every record it was medianed from.
    The median is over one point per contributor, the judged one included:
    with three contributors the leave-one-out median is the mean of the
    other two, which one wild contributor can drag, while the median of all
    three moves by at most one rank. In a smaller cell there are not enough
    peers to judge, and the record stays.
  Only a baseline arm that loaded is judged: a fit point (stock.outcome
  skipped_predicted_nonfit or failed_load) has no stock speed and stays, as
  the table's "didn't load" row.

The vote collapse: one vote per contributor (DID) per cell. The cell is the
model (name, hfRepo, revision), the machine (deviceClass, memoryBytes,
memoryKind), the compression profile, the version-compatibility class, the
platform, and whether the stock arm was measured (a measured record and a fit
point are different rows on the page, and a median across them would mix a
stock speed with its absence). Each metric of a (DID, cell) group is the
median of that DID's records; everything else comes from the group's latest
record, whose at-uri is `site.source`; `site.provenance` lists every at-uri
the point was medianed from. After the collapse, the records sharing a page
cell are distinct contributors or distinct machines, so the page's count of
records in a cell is its contributor count.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

SITE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE / "data"))
import build_points as bp  # noqa: E402 — site/data/build_points.py, the site build's validator

NSID = "wtf.petrichor.drinkme.measurement"
CURRENT_VERSION = bp.CURRENT_VERSION
VERIFIED_BY = ("pack", "roundtrip")
# MTP speculation drafts at depth 4 (serving/mtp.py DEFAULT_DEPTH, what AUTO
# resolves to), so one verify step emits at most SPEC_DEPTH + 1 = 5 tokens.
# Stated here, not imported: mtp.py imports torch, which the aggregator does not.
SPEC_DEPTH = 4
SPEC_ARMS = ("stock", "compressed")
SPEC_PROMPTS = ("agent", "chat", "code")


@dataclass(frozen=True)
class Settings:
    tolerance: float = 1.10       # F3: an arm may claim up to this multiple of its bandwidth bound
    baseline_arm: str = "stock"            # F5: the arm whose efficiency is judged (BASELINE_ARMS)
    baseline_floor: float = 0.10           # F5: under this fraction of its bound drops the record, peers or not
    baseline_relative: float = 0.6         # F5: under this multiple of its cell's median efficiency drops it...
    baseline_min_contributors: int = 3     # F5: ...when the cell has at least this many contributors
    denylist: frozenset = field(default_factory=frozenset)  # F4: DIDs and at-uris


def load_denylist(path: str | os.PathLike | None) -> frozenset:
    """One DID or at-uri per line; blank lines and `#` comments ignored."""
    if not path:
        return frozenset()
    out = set()
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return frozenset(out)


def parse_uri(uri) -> tuple[str, str, str] | None:
    """at://<did>/<collection>/<rkey> -> (did, collection, rkey), else None."""
    if not isinstance(uri, str) or not uri.startswith("at://"):
        return None
    parts = uri[len("at://"):].split("/")
    if len(parts) != 3 or not all(parts) or not parts[0].startswith("did:"):
        return None
    return parts[0], parts[1], parts[2]


def _num(v) -> float | None:
    """A metric's decimal string (or a number) as a finite float, else None."""
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _metrics(value: dict) -> dict:
    return {m["name"]: m.get("value") for m in value.get("metrics") or []
            if isinstance(m, dict) and isinstance(m.get("name"), str)}


# ----------------------------------------------------------- the filters --
# Each takes the record (and what it needs) and returns None, or
# (reason, numbers) when the record is excluded.

def f2_bit_exact(value: dict):
    raw = value.get("raw")
    raw = raw if isinstance(raw, dict) else {}
    v, s = raw.get("verified_tensors"), raw.get("swapped_linears")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (v, s)):
        return "bit-exact gate unrecorded: raw.verified_tensors / raw.swapped_linears missing or not integers", {}
    if s <= 0 or v != s:
        return (f"bit-exact gate failed: {v} of {s} swapped tensors verified",
                {"verified_tensors": v, "swapped_linears": s})
    how = raw.get("verification")
    if how is not None and how not in VERIFIED_BY:
        return (f"bit-exact gate: raw.verification is {how!r}, not one of {', '.join(VERIFIED_BY)}",
                {"verification": how})
    return None


def _twin_ran(value: dict) -> bool:
    """The record publishes the twin arm (bench does on torch)."""
    return "twin_decode_tok_s" in _metrics(value)


def _arms(value: dict):
    """The arms that claim a decode speed: (arm, tok/s field)."""
    arms = [("compressed", "compressed_decode_tok_s")]
    if (value.get("stock") or {}).get("outcome") == "measured":
        arms.append(("stock", "stock_decode_tok_s"))
    if _twin_ran(value):
        arms.append(("twin", "twin_decode_tok_s"))
    return arms


def bound_bytes_name(m: dict, arm: str) -> str:
    """The metric an arm's bandwidth bound divides by: `<arm>_decode_read_gb`,
    the bytes one decode step reads, when the record carries it; else
    `<arm>_weights_gb`, the arm's resident footprint (which also counts a
    vision tower and the embedding rows a step does not read, so the bound
    is stricter). Apple silicon records carry no decode-read figure yet."""
    read = f"{arm}_decode_read_gb"
    return read if read in m else f"{arm}_weights_gb"


def f3_physics(value: dict, tolerance: float):
    m = _metrics(value)
    for arm, tok_name in _arms(value):
        if tok_name not in m and arm == "compressed":
            continue  # claims no compressed speed: plots nothing, nothing to check
        gb_name = bound_bytes_name(m, arm)
        tok, read, gb = _num(m.get(tok_name)), _num(m.get("read_gb_s")), _num(m.get(gb_name))
        if tok is None or read is None or gb is None or read <= 0 or gb <= 0:
            return (f"physics uncheckable: the {arm} arm claims {tok_name} without a positive "
                    f"read_gb_s and {gb_name}",
                    {"arm": arm, tok_name: m.get(tok_name), "read_gb_s": m.get("read_gb_s"),
                     gb_name: m.get(gb_name)})
        bound = read / gb
        if tok > tolerance * bound:
            return (f"physics: the {arm} arm's {tok:g} tok/s exceeds {tolerance:g} x its bandwidth bound "
                    f"{bound:.4g} tok/s (read_gb_s {read:g} / {gb_name} {gb:g})",
                    {"arm": arm, "decode_tok_s": tok, "read_gb_s": read, gb_name: gb,
                     "bound_tok_s": round(bound, 4), "tolerance": tolerance})
    return _f3_speculation(m, tolerance)


def _f3_speculation(m: dict, tolerance: float):
    """Every `<arm>_spec_decode_tok_s_<prompt>` the record carries against
    (SPEC_DEPTH + 1) x the arm's plain-decode bound x tolerance."""
    for arm in SPEC_ARMS:
        gb_name = bound_bytes_name(m, arm)
        for prompt in SPEC_PROMPTS:
            name = f"{arm}_spec_decode_tok_s_{prompt}"
            if name not in m:
                continue
            tok, read, gb = _num(m.get(name)), _num(m.get("read_gb_s")), _num(m.get(gb_name))
            if tok is None or read is None or gb is None or read <= 0 or gb <= 0:
                return (f"physics uncheckable: the {arm} arm claims {name} without a positive "
                        f"read_gb_s and {gb_name}",
                        {"arm": arm, name: m.get(name), "read_gb_s": m.get("read_gb_s"),
                         gb_name: m.get(gb_name)})
            bound = read / gb
            limit = (SPEC_DEPTH + 1) * tolerance * bound
            if tok > limit:
                return (f"physics: the {arm} arm's {name} {tok:g} tok/s exceeds {SPEC_DEPTH + 1} x "
                        f"{tolerance:g} x its bandwidth bound {bound:.4g} tok/s "
                        f"(read_gb_s {read:g} / {gb_name} {gb:g}), the most a depth-{SPEC_DEPTH} "
                        f"verify step emits",
                        {"arm": arm, "metric": name, "spec_tok_s": tok, "read_gb_s": read, gb_name: gb,
                         "bound_tok_s": round(bound, 4), "spec_depth": SPEC_DEPTH,
                         "limit_tok_s": round(limit, 4), "tolerance": tolerance})
    return None


def f4_denylist(uri: str, did: str, denylist: frozenset):
    if did in denylist:
        return "denylisted contributor", {}
    if uri in denylist:
        return "denylisted record", {}
    return None


@dataclass(frozen=True)
class BaselineArm:
    loaded: object  # value -> bool: the arm ran, so its speed metrics are present
    tok_s: str      # its decode tok/s metric


# The arms F5 can judge a baseline by: the stock arm, the one the page
# divides by (compressed ÷ stock).
BASELINE_ARMS = {
    "stock": BaselineArm(lambda v: (v.get("stock") or {}).get("outcome") == "measured",
                         "stock_decode_tok_s"),
}


def baseline_efficiency(value: dict, arm: str = "stock") -> dict | None:
    """The baseline arm's decode tok/s, its bandwidth bound (F3's, through
    bound_bytes_name) and their ratio, or None when the arm did not load (a
    fit point) or lacks the numbers."""
    a = BASELINE_ARMS[arm]
    if not a.loaded(value):
        return None
    m = _metrics(value)
    gb_name = bound_bytes_name(m, arm)
    tok, read, gb = _num(m.get(a.tok_s)), _num(m.get("read_gb_s")), _num(m.get(gb_name))
    if tok is None or read is None or gb is None or read <= 0 or gb <= 0:
        return None
    bound = read / gb
    return {"arm": arm, "decode_tok_s": tok, "bound_tok_s": round(bound, 4), "efficiency": tok / bound}


def f5_floor(value: dict, arm: str, floor: float):
    """F5's absolute floor, one record at a time."""
    b = baseline_efficiency(value, arm)
    if b is None or b["efficiency"] >= floor:
        return None
    return (f"baseline implausible (absolute floor): the {arm} arm's {b['decode_tok_s']:g} tok/s is "
            f"{b['efficiency']:.1%} of its bandwidth bound {b['bound_tok_s']:.4g} tok/s, under the "
            f"{floor:.0%} floor whatever its peers show",
            {**b, "efficiency": round(b["efficiency"], 4), "floor": floor})


def f5_relative(points: list[tuple[str, tuple, dict]], settings: Settings) -> dict:
    """F5 against peers. points: one (did, cell, collapsed value) per
    contributor per cell. Returns {index into points: (reason, numbers)} for
    each point dropped."""
    cells = {}
    for i, (did, cell, value) in enumerate(points):
        b = baseline_efficiency(value, settings.baseline_arm)
        if b is not None:
            cells.setdefault(cell, []).append((i, b))
    out = {}
    for judged in cells.values():
        n = len(judged)
        if n < settings.baseline_min_contributors:
            continue  # too few peers to judge
        median = statistics.median(b["efficiency"] for _, b in judged)
        for i, b in judged:
            if b["efficiency"] < settings.baseline_relative * median:
                out[i] = (f"baseline suspect (relative): the {b['arm']} arm's {b['decode_tok_s']:g} tok/s is "
                          f"{b['efficiency']:.1%} of its bandwidth bound {b['bound_tok_s']:.4g} tok/s, under "
                          f"{settings.baseline_relative:g} x the cell's median {median:.1%} over {n} contributors",
                          {**b, "efficiency": round(b["efficiency"], 4), "cell_median": round(median, 4),
                           "contributors": n, "relative": settings.baseline_relative})
    return out


# ----------------------------------------------------------- the pipeline --

def _excluded(uri, value, step, reason, numbers=None) -> dict:
    e = {"source": uri, "version": value.get("version") if isinstance(value, dict) else None,
         "filter": step, "reason": reason}
    if numbers:
        e["numbers"] = numbers
    return e


def judge(uri: str, value, settings: Settings, current: str = CURRENT_VERSION) -> dict | None:
    """None when the record reaches the collapse; else its exclusion."""
    parsed = parse_uri(uri)
    if parsed is None or parsed[1] != NSID:
        return _excluded(uri, value, "validate", f"not an at-uri in {NSID}")
    if not isinstance(value, dict):
        return _excluded(uri, value, "validate", "not a JSON object")
    if value.get("$type") != NSID:
        return _excluded(uri, value, "validate", f"$type is {value.get('$type')!r}, not {NSID}")
    why = bp.admit(value, uri, current)
    if why is not None:
        return _excluded(uri, value, "validate", why)
    for step, check in (("F2", lambda: f2_bit_exact(value)),
                        ("F3", lambda: f3_physics(value, settings.tolerance)),
                        ("F4", lambda: f4_denylist(uri, parsed[0], settings.denylist)),
                        ("F5", lambda: f5_floor(value, settings.baseline_arm, settings.baseline_floor))):
        hit = check()
        if hit is not None:
            return _excluded(uri, value, step, *hit)
    return None


def version_class(v: str) -> str:
    """docs/versioning.md's comparison class: the major version (the page's
    versionKey)."""
    return v.split(".")[0]


def cell_key(value: dict) -> tuple:
    model, env = value["model"], value["environment"]
    return (model["name"], model["hfRepo"], model["revision"],
            env["deviceClass"], env.get("memoryBytes"), env.get("memoryKind"),
            (value.get("compression") or {}).get("profile"), version_class(value["version"]),
            env["platform"], value["stock"]["outcome"] == "measured")


def _median(xs: list[Decimal]) -> Decimal:
    s = sorted(xs)
    k = len(s) // 2
    return s[k] if len(s) % 2 else (s[k - 1] + s[k]) / 2


def _dec(v) -> Decimal | None:
    try:
        d = Decimal(str(v))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def collapse_group(group: list[tuple[str, dict]]) -> tuple[str, dict, list[str]]:
    """One DID's records of one cell -> (the latest record's at-uri, the
    record with each metric the median across the group, every at-uri)."""
    group = sorted(group, key=lambda ur: (str(ur[1].get("createdAt", "")), ur[0]))
    uri, latest = group[-1]
    if len(group) == 1:
        return uri, latest, [uri]
    # the latest record's metrics, then any an earlier record of the group carried that it did not
    # (a later run without the speculative arms must not drop them)
    metrics_in = list(latest["metrics"])
    seen = {m["name"] for m in metrics_in}
    for _, r in group[:-1]:
        for m in r.get("metrics") or []:
            if isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"] not in seen:
                seen.add(m["name"]); metrics_in.append(m)
    metrics = []
    for m in metrics_in:
        vals = [_dec(v) for _, r in group for n, v in _metrics(r).items() if n == m["name"]]
        vals = [v for v in vals if v is not None]
        # the value is now a median across records, so one record's samples no longer describe it
        out = {k: v for k, v in m.items() if k != "samples"}
        if vals:
            out["value"] = format(_median(vals), "f")
        metrics.append(out)
    return uri, {**latest, "metrics": metrics}, [u for u, _ in group]


def run(records, settings: Settings = Settings(), current: str = CURRENT_VERSION) -> tuple[list, list]:
    """records: an iterable of {"uri", "value"} (listRecords / getRecord
    shape) -> (points for the page, exclusions). The same at-uri twice is
    one record: the last one wins."""
    by_uri = {}
    for r in records:
        by_uri[r.get("uri")] = r.get("value")
    groups, excluded = {}, []
    for uri, value in by_uri.items():
        try:
            bad = judge(uri, value, settings, current)
        except Exception as exc:  # a malformed record never stops the run
            bad = _excluded(uri, value, "validate", f"malformed record ({type(exc).__name__}: {exc})")
        if bad is not None:
            excluded.append(bad)
            continue
        did = parse_uri(uri)[0]
        groups.setdefault((did, cell_key(value)), []).append((uri, value))
    collapsed = []
    for (did, cell), group in groups.items():
        uri, value, provenance = collapse_group(group)
        collapsed.append((did, cell, uri, value, provenance))
    # F5 relative: needs every contributor's collapsed point in the cell
    dropped = f5_relative([(did, cell, value) for did, cell, _, value, _ in collapsed], settings)
    points = []
    for i, (did, cell, uri, value, provenance) in enumerate(collapsed):
        if i in dropped:
            excluded.extend(_excluded(u, by_uri[u], "F5", *dropped[i]) for u in provenance)
            continue
        try:
            point = bp.wrap_measurement(value, uri)
        except Exception as exc:
            excluded.extend(_excluded(u, value, "validate", f"malformed record ({type(exc).__name__}: {exc})")
                            for u in provenance)
            continue
        point["site"]["contributor"] = did
        point["site"]["provenance"] = provenance
        points.append(point)
    points.sort(key=lambda p: (p["environment"]["deviceClass"], p["model"]["name"],
                               (p.get("compression") or {}).get("profile") or "", p["site"]["source"]))
    excluded.sort(key=lambda e: (str(e["filter"]), str(e["source"])))
    return points, excluded


# --------------------------------------------------------------- output --

def document(points: list, excluded: list, generated_at: _dt.datetime, cursor=None,
             skipped: list | None = None, current: str = CURRENT_VERSION) -> dict:
    """build_points' bundle, plus when it was generated, the follow cursor
    it reflects, and the repositories that could not be read."""
    doc = bp.bundle(points, excluded, current)
    doc["_"] = ("Every point the site plots, as a wtf.petrichor.drinkme.measurement record read from "
                "the network. Generated by site/aggregator from every published record: validated by "
                "build_points.admit, filtered (F2-F5, site/aggregator/pipeline.py), and collapsed to "
                "one point per contributor per cell, each metric the median of that contributor's "
                "records (site.provenance). `excluded` names every record left out and why.")
    doc["generatedAt"] = generated_at.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    doc["cursor"] = cursor
    doc["skipped"] = skipped or []
    return doc


def _replace(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_atomic(path, doc: dict) -> pathlib.Path:
    """Write `doc` to `path` through `<path>.tmp` and a rename, so a reader
    sees the old file or the new one, never half of either; and the same to
    the day's dated copy, `<stem>-YYYY-MM-DD.json` (the day of generatedAt),
    which ends each day as its last write. Returns the dated copy's path."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    day = doc["generatedAt"][:10]
    dated = path.with_name(f"{path.stem}-{day}{path.suffix}")
    _replace(dated, text)
    _replace(path, text)
    return dated
