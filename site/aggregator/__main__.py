"""The aggregator's command line. Run from site/ (or with site/ on PYTHONPATH):

    python -m aggregator backfill -o out/points.json
    python -m aggregator backfill --repos did:plc:a,did:web:b --pds https://dev-host.example -o /tmp/points.json
    python -m aggregator follow --state /srv/state -o /srv/data/points.json
    python -m aggregator reconcile --state /srv/state -o /srv/data/points.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import threading

from . import follow, pipeline, sources


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--out", required=True, help="points.json to write (atomically; a dated copy beside it)")
    p.add_argument("--repos", help="comma-separated DIDs: skip discovery on the relay")
    p.add_argument("--pds", help="read every repo from this PDS instead of resolving DIDs (development)")
    p.add_argument("--relay", action="append",
                   help=f"relay to discover repos on, in order (default: {', '.join(sources.RELAYS)})")
    p.add_argument("--plc", default=sources.PLC, help="PLC directory for did:plc documents")
    p.add_argument("--denylist", help="file of DIDs and at-uris to exclude, one per line (F4; default none)")
    p.add_argument("--tolerance", type=float, default=pipeline.Settings.tolerance,
                   help="F3: the multiple of an arm's bandwidth bound a decode speed may claim")
    p.add_argument("--baseline-arm", choices=sorted(pipeline.BASELINE_ARMS), default=pipeline.Settings.baseline_arm,
                   help="F5: the arm whose decode speed over its bandwidth bound is judged")
    p.add_argument("--baseline-floor", type=float, default=pipeline.Settings.baseline_floor,
                   help="F5: a baseline under this fraction of its bound drops the record, whatever its peers show")
    p.add_argument("--baseline-relative", type=float, default=pipeline.Settings.baseline_relative,
                   help="F5: a baseline under this multiple of its cell's median drops the record")
    p.add_argument("--baseline-min-contributors", type=int, default=pipeline.Settings.baseline_min_contributors,
                   help="F5: the contributors a cell needs before the relative rule judges it")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds for each HTTP request")
    p.add_argument("--repo-deadline", type=float, default=60.0, help="seconds one repository may take")


def source_of(a) -> sources.Source:
    return sources.Source(http=sources.Http(timeout=a.timeout), relays=tuple(a.relay or sources.RELAYS), plc=a.plc,
                          repos=tuple(d.strip() for d in a.repos.split(",") if d.strip()) if a.repos else None,
                          pds=a.pds, repo_deadline=a.repo_deadline)


def settings_of(a) -> pipeline.Settings:
    return pipeline.Settings(tolerance=a.tolerance, baseline_arm=a.baseline_arm, baseline_floor=a.baseline_floor,
                             baseline_relative=a.baseline_relative,
                             baseline_min_contributors=a.baseline_min_contributors,
                             denylist=pipeline.load_denylist(a.denylist))


def summarize(records: list, points: list, excluded: list, skipped: list) -> None:
    admitted = sum(len(p["site"]["provenance"]) for p in points)
    print(f"read {len(records)} records; {admitted} admitted, collapsed to {len(points)} points; "
          f"{len(excluded)} excluded; "
          f"{len(skipped)} repos skipped")
    for e in excluded:
        print(f"  excluded [{e['filter']}] {e['source']}: {e['reason']}")
    for s in skipped:
        print(f"  skipped {s['did']}: {s['reason']}")


def cmd_backfill(a) -> int:
    try:
        snap = sources.backfill(source_of(a))
    except sources.FetchError as exc:
        print(f"backfill failed: {exc}", file=sys.stderr)
        return 1
    points, excluded = pipeline.run(snap.records, settings_of(a))
    doc = pipeline.document(points, excluded, _dt.datetime.now(_dt.timezone.utc), cursor=None,
                            skipped=snap.skipped)
    dated = pipeline.write_atomic(a.out, doc)
    print(f"wrote {a.out} and {dated}")
    summarize(snap.records, points, excluded, snap.skipped)
    return 0


def follower_of(a) -> follow.Follower:
    return follow.Follower(source_of(a), a.out, a.state, settings_of(a), jetstream=a.jetstream,
                           reconcile_every=a.reconcile_hours * 3600)


def cmd_follow(a) -> int:
    stop = threading.Event()
    if a.duration:
        timer = threading.Timer(a.duration, stop.set)
        timer.daemon = True
        timer.start()
    try:
        follower_of(a).run(stop)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_reconcile(a) -> int:
    return 0 if follower_of(a).reconcile() else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aggregator", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="read every published record once and write points.json")
    _common(b)
    b.set_defaults(fn=cmd_backfill)
    for name, fn, text in (("follow", cmd_follow, "tail Jetstream, rewriting points.json on every change, "
                                                  "with a full reconcile on a timer"),
                           ("reconcile", cmd_reconcile, "replace the follow state's record set with a full "
                                                        "backfill and rewrite points.json")):
        c = sub.add_parser(name, help=text)
        _common(c)
        c.add_argument("--state", required=True, help="directory for the stream cursor and the record set")
        c.add_argument("--jetstream", default=follow.JETSTREAM, help="Jetstream subscribeEvents URL")
        c.add_argument("--reconcile-hours", type=float, default=24.0, help="hours between full reconciles")
        c.add_argument("--duration", type=float, help="follow: stop after this many seconds")
        c.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
