#!/usr/bin/env python3
"""Re-derive every number in site/data/points.json from its named record file.

Each record's `site.at` maps a field to an RFC-6901 pointer into
`site.source`; this resolves every pointer and compares it with the value
the record carries. Drift exits nonzero and names the field.

    python3 site/data/check_points.py        # from the repo root
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
POINTS = ROOT / "site" / "data" / "points.json"

errors, checked, sources, _cache = [], 0, set(), {}


def load(src):
    if src not in _cache:
        _cache[src] = json.loads((ROOT / src).read_text())
    sources.add(src)
    return _cache[src]


def resolve(doc, pointer):
    cur = doc
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        cur = cur[int(token)] if isinstance(cur, list) else cur[token]
    return cur


def same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def check(where, src, pointer, expected, convert=None):
    global checked
    checked += 1
    try:
        got = resolve(load(src), pointer)
    except (KeyError, IndexError, TypeError, ValueError, FileNotFoundError) as exc:
        errors.append(f"{where}: {src}{pointer} does not resolve ({exc!r})")
        return
    if convert is not None:
        got = convert(got)
    if expected is None:
        return  # existence only
    if isinstance(expected, bool):
        if bool(got) != expected:
            errors.append(f"{where}: record says {expected}, {src} says {got!r}")
    elif not same(got, expected):
        errors.append(f"{where}: record says {expected!r}, {src}{pointer} says {got!r}")


data = json.loads(POINTS.read_text())
for r in data["records"]:
    site = r["site"]
    src = site["source"]
    tag = f"{r['environment']['deviceClass']}/{r['model']['name']}"
    metrics = {m["name"]: m["value"] for m in r["metrics"]}
    for field, ptr in site["at"].items():
        if field in metrics:
            check(f"{tag}.{field}", src, ptr, metrics[field])
        elif field in ("bitsPerWeight", "meanTensorBitsPerWeight"):
            check(f"{tag}.{field}", src, ptr, r["compression"][field])
        elif field.startswith("stock."):
            # the stock arm's OUTCOME (measured | skipped_predicted_nonfit |
            # failed_load), the budget a skip was judged against, the error
            # a failed load raised — the record's `stock` object, each field
            # pointed at itself
            check(f"{tag}.{field}", src, ptr, r["stock"][field[len("stock."):]])
        elif field == "allocatableGB":
            # the one number build_points derives: environment.memoryBytes
            # in decimal GB, divided the same way here
            check(f"{tag}.allocatableGB", src, ptr, site["allocatableGB"], convert=lambda b: b / 1e9)
        elif field == "stockError":
            check(f"{tag}.stockError", src, ptr, site.get("stockError"))
        else:
            errors.append(f"{tag}: pointer for unknown field {field!r}")
    if r.get("compression") and "bitsPerWeight" not in site["at"]:
        errors.append(f"{tag}: bitsPerWeight has no pointer")
    if "stock.outcome" not in site["at"]:
        errors.append(f"{tag}: a record must point at its stock outcome")
    if r["stock"]["outcome"] != "measured" and "stockError" not in site["at"]:
        errors.append(f"{tag}: a fit point must point at its stock error")
    for name in metrics:
        if name not in site["at"]:
            errors.append(f"{tag}: metric {name} has no pointer")

print(f"{checked} values checked against {len(sources)} record files")
for src in sorted(sources):
    print(f"  · {src}")
if errors:
    print(f"\n{len(errors)} MISMATCH(ES):", file=sys.stderr)
    for e in errors:
        print(f"  ✗ {e}", file=sys.stderr)
    sys.exit(1)
print("all clear — every plotted number is a literal field in its record file")
