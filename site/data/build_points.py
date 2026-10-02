#!/usr/bin/env python3
"""Build site/data/points.json: every plotted point as a lexicon-shaped record.

The chart shows `wtf.petrichor.drinkme.measurement` records in the shape
`drinkme bench` writes (lexicons/wtf.petrichor.drinkme.measurement.json):
`stock {outcome, budgetBytes?, error?}`, `environment.memoryBytes`,
`compression.profile`, the metrics in decimal GB, `raw` beside them. Each
record is wrapped by `wrap_measurement` and pointed at ITSELF: every
metric's pointer is `/metrics/<i>/value` — the literal field in the
record file (`site.source`) — so nothing is converted or re-derived here,
with one exception:
`site.allocatableGB`, the capacity chart's door line, is
`environment.memoryBytes` in decimal GB (÷ 1e9); check_points.py divides
the same way when it checks it.

The records come from `measurements/`, where `drinkme bench` writes them:

    python3 site/data/build_points.py                          # measurements/*.json
    python3 site/data/build_points.py measurements/a.json      # only the files named

With no records the bundle is the empty set and the page says so ("No
results for this version of drinkme yet"). The site displays records of
the current major version only (docs/versioning.md).

THE COMPARISON POLICY IS APPLIED HERE (docs/versioning.md "Measurement
compatibility"): every record is validated (validate_record — the identity
fields the site groups on must be present and well-formed), and a record
whose `version` is not compatible with this drinkme's (`compatible`: the
same major version) is EXCLUDED and named on stdout, never bundled —
`measurements/` naturally holds old runs and several profiles, and without
this rule an old-version sip record and a current gulp record of two
different checkpoints became one plotted cell with a combined median. The
bundle records the version it was built for (`version`) and every
exclusion (`excluded`). The page then groups by
the full comparison identity — version compatibility, hfRepo, resolved
revision, compression profile, platform, device — and labels each
group so differing groups are visible, never medianed together.

Anything not in a record file is not here. `check_points.py` re-resolves every
pointer and fails on drift. Run both from the repo root:

    python3 site/data/build_points.py && python3 site/data/check_points.py
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "data" / "points.json"
MEASUREMENTS = ROOT / "measurements"

sys.path.insert(0, str(ROOT / "src"))
from drinkme import __version__ as CURRENT_VERSION  # noqa: E402 — drinkme/__init__ imports nothing


def load(src, root=ROOT):
    return json.loads((root / src).read_text())


# ------------------------------------------------ the comparison policy --

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def parse_version(v) -> tuple[int, int, int] | None:
    m = _SEMVER.match(v) if isinstance(v, str) else None
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def compatible(record_version: str, current: str = CURRENT_VERSION) -> bool:
    """docs/versioning.md: compare within the same major version. A version
    that does not parse is not comparable with anything."""
    r, c = parse_version(record_version), parse_version(current)
    if r is None or c is None:
        return False
    return r[0] == c[0]


# The fields the site's comparison identity is built from (index.html
# identityKey): a record missing one cannot be grouped honestly and is
# refused by name, not plotted under a guess.
IDENTITY_FIELDS = (("version",), ("model", "name"), ("model", "hfRepo"), ("model", "revision"),
                   ("environment", "deviceClass"), ("environment", "platform"), ("stock", "outcome"))
MEASURED_METRICS = ("compressed_decode_tok_s", "stock_decode_tok_s", "read_gb_s")


def _get(record: dict, path: tuple[str, ...]):
    cur = record
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def validate_record(record: dict, src: str) -> str | None:
    """None when `record` carries a well-formed comparison identity; else
    the reason (naming the field) it cannot be plotted."""
    if not is_new_shape(record):
        return f"not a new-shape record (no {NEW_SHAPE_MARKER!r})"
    for path in IDENTITY_FIELDS:
        v = _get(record, path)
        if not isinstance(v, str) or not v:
            return f"missing or non-string {'.'.join(path)}"
    if parse_version(record["version"]) is None:
        return f"version {record['version']!r} is not a semver"
    if not isinstance(record.get("metrics"), list):
        return "missing metrics"
    names = {m.get("name") for m in record["metrics"] if isinstance(m, dict)}
    if record["stock"]["outcome"] == "measured":
        profile = _get(record, ("compression", "profile"))
        if not isinstance(profile, str) or not profile:
            return "measured record without compression.profile"
        missing = [n for n in MEASURED_METRICS if n not in names]
        if missing:
            return f"measured record missing metrics {missing}"
    if not isinstance(_get(record, ("environment", "memoryBytes")), (int, float)):
        return "missing environment.memoryBytes"
    return None


def admit(record: dict, src: str, current: str = CURRENT_VERSION) -> str | None:
    """None when the record belongs in this version's bundle; else why not
    (validation, then the compatibility policy)."""
    bad = validate_record(record, src)
    if bad is not None:
        return bad
    if not compatible(record["version"], current):
        return f"version {record['version']} is not comparable with drinkme {current} (same major version)"
    return None


# A record carries the stock arm's outcome as one object, `stock`
# (outcome: measured | skipped_predicted_nonfit | failed_load; the budget a
# skip was judged against; the error a failed load raised) — each field
# copied through with a pointer at itself.
NEW_SHAPE_MARKER = "stock"
STOCK_FIELDS = ("outcome", "budgetBytes", "error")


def is_new_shape(record: dict) -> bool:
    return NEW_SHAPE_MARKER in record


def wrap_measurement(record: dict, src: str) -> dict:
    """A `wtf.petrichor.drinkme.measurement` record -> the plotted point,
    verbatim (minus `raw`), with `site.at` pointing every number at the
    literal field in its record file `src` (`site.source`). No unit
    conversion and no re-derivation: bench already converted bytes ->
    decimal GB at its record boundary, and that is the number the site
    shows. The one derived number is site.allocatableGB (memoryBytes /
    1e9), pointed at the bytes."""
    if not is_new_shape(record):
        raise ValueError(f"{src}: not a new-shape record (no {NEW_SHAPE_MARKER!r})")
    at = {}
    for i, m in enumerate(record["metrics"]):
        at[m["name"]] = f"/metrics/{i}/value"
    comp = record.get("compression") or {}
    for field in ("bitsPerWeight", "meanTensorBitsPerWeight"):
        if field in comp:
            at[field] = f"/compression/{field}"
    stock = record["stock"]
    for field in STOCK_FIELDS:
        if field in stock:
            at[f"stock.{field}"] = f"/stock/{field}"
    measured = stock["outcome"] == "measured"
    if not measured:
        # the capacity chart's sentence for a stock arm that did not run:
        # a failed load's own message (stock.error); for a fit-check skip
        # the public field is the budget, a number, so bench's reason in
        # raw (raw.stock_error) is the sentence
        at["stockError"] = f"/stock/error" if "error" in stock else f"/raw/stock_error"
    at["allocatableGB"] = f"/environment/memoryBytes"
    out = {k: v for k, v in record.items() if k != "raw"}
    out["site"] = {
        "source": src,
        "at": at,
        "allocatableGB": record["environment"]["memoryBytes"] / 1e9,
        "harness": "drinkme bench",
    }
    if not measured:
        out["site"]["stockError"] = stock.get("error") or (record.get("raw") or {}).get("stock_error")
    return out


def build(paths, root=ROOT, current: str = CURRENT_VERSION) -> tuple[list, list]:
    """Record files (paths under `root` — check_points resolves
    `site.source` relative to it) -> (plotted points, exclusions). Every
    record goes through `admit`: an invalid or incompatible one is not
    bundled, and is returned as {source, version, reason} so the build
    says what it left out. A file is the record itself, as `drinkme bench`
    writes it; anything else (a record wrapped in another object) fails
    `admit` by name."""
    out, excluded = [], []
    for path in paths:
        full = pathlib.Path(path).resolve()
        if not full.is_relative_to(root):
            raise SystemExit(f"{path}: a record file must live under the repo root {root}")
        src = str(full.relative_to(root))
        record = load(src, root)
        why = admit(record, src, current) if isinstance(record, dict) else "not a JSON object"
        if why is not None:
            version = _get(record, ("version",)) if isinstance(record, dict) else None
            excluded.append({"source": src, "version": version, "reason": why})
            continue
        out.append(wrap_measurement(record, src))
    return out, excluded


def bundle(recs: list, excluded: list | None = None, current: str = CURRENT_VERSION) -> dict:
    """The points.json document: the records, the version they were admitted
    for, and every exclusion. `write` saves it for the site build; the
    aggregator (site/aggregator) adds its own fields and writes it atomically."""
    return {
        "_": ("Every point the site plots, as a wtf.petrichor.drinkme.measurement record. "
              "Generated by build_points.py from the records in measurements/; "
              "`site.at` holds an RFC-6901 pointer for every number so check_points.py can re-derive it. "
              "Nothing here is interpolated, projected, or typed from memory. "
              "Only records comparable with `version` are bundled "
              "(docs/versioning.md); `excluded` names every record the build left out and why."),
        "lexicon": "lexicons/wtf.petrichor.drinkme.measurement.json",
        "version": current,
        "excluded": excluded or [],
        "records": recs,
    }


def write(out_path: pathlib.Path, recs: list, excluded: list | None = None,
          current: str = CURRENT_VERSION) -> None:
    out = bundle(recs, excluded, current)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}: "
          f"{len(recs)} records for drinkme {current}, {len(excluded or [])} excluded")
    for e in excluded or []:
        print(f"  excluded {e['source']}: {e['reason']}")
    for r in recs:
        m = {x["name"]: x["value"] for x in r["metrics"]}
        print(f"  {r['environment']['deviceClass']:14s} {r['model']['name']:16s} "
              f"stock {m.get('stock_decode_tok_s','—'):>6s} -> drinkme {m.get('compressed_decode_tok_s'):>6s} tok/s"
              f"  {m.get('stock_weights_gb','—')}->{m.get('compressed_weights_gb')} GB"
              f"  read {m.get('read_gb_s','—')}  stock={r['stock']['outcome']}")


def main(argv) -> None:
    # record files named: only those; none: every record in measurements/
    paths = argv or sorted(str(p) for p in MEASUREMENTS.glob("*.json"))
    recs, excluded = build(paths)
    write(OUT, recs, excluded)


if __name__ == "__main__":
    main(sys.argv[1:])
