# Aggregator

The results site shows the `wtf.petrichor.drinkme.measurement` records
published on atproto; the `data/points.json` committed in the repository is a
snapshot of such records. This service keeps the page's `points.json` current.
Anyone can publish a record, so this service decides which ones the page
shows. It finds the records, validates and filters them, collapses them to one
point per contributor per cell, and writes the `points.json` that
`../index.html` reads.

The rules live in [`pipeline.py`](pipeline.py) and its docstring lists them.
Validation is `site/data/build_points.py`'s `admit`, imported rather than copied,
so the committed site build and the live service cannot disagree about what a
valid, comparable record is. Filters F1 to F5 follow it: Apple silicon, the
bit-exact gate, the bandwidth bound, a denylist, and a stock arm too slow for its
ratio to mean anything: under 10% of its bandwidth bound, or, in a cell with at
least three contributors, under 0.6 times the cell's median. Every excluded record is listed in `excluded` with its
filter, its reason, and the numbers behind the decision.

[`sources.py`](sources.py) reads the network with read-only GETs, each with a
timeout. [`follow.py`](follow.py) keeps the record set current.

## Setup

The aggregator is the site's backend, not part of the drinkme package, so its one
runtime dependency, a websocket client, stays out of drinkme's `pyproject.toml`.
Give it its own environment in this directory (`.venv/` is gitignored):

```sh
cd site/aggregator
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt       # runtime
uv pip install --python .venv/bin/python -r requirements-test.txt  # and pytest, for the tests
.venv/bin/python -m pytest -q tests
```

## Commands

Run the commands from `site/`, where the package is importable:

```sh
aggregator/.venv/bin/python -m aggregator backfill -o /srv/data/points.json
aggregator/.venv/bin/python -m aggregator backfill --repos did:plc:… --pds https://dev-host.example -o /tmp/points.json
```

`backfill` discovers every repository that holds the collection with
`com.atproto.sync.listReposByCollection` on the relay (us-east, falling back to
us-west). It resolves each DID to its PDS and reads that PDS's records. `--repos`
skips discovery. `--pds` skips DID resolution, for a PDS used in development. A PDS that
is slow or failing is skipped, and its reason appears in the output's `skipped`
list. `--denylist FILE` names DIDs and at-uris to exclude. `--tolerance` tunes
F3 (which also holds each speculative decode metric to 5 x its arm's bound, the most a depth-4 verify step emits); `--baseline-floor`, `--baseline-relative`, `--baseline-min-contributors` and
`--baseline-arm` tune F5.

```sh
aggregator/.venv/bin/python -m aggregator follow --state /srv/state -o /srv/data/points.json
aggregator/.venv/bin/python -m aggregator reconcile --state /srv/state -o /srv/data/points.json
```

`follow` subscribes to Jetstream for this collection and rewrites `points.json`
whenever the record set changes. It keeps the stream cursor and the record set
in `--state`, so a restart resumes from where it stopped. The default endpoint
is Jetstream v2 (`collections=`, cursor `seq`); a v1 `/subscribe` URL gets
`wantedCollections=` and its cursor is `time_us`. An event only tells the
follower where to look. For a create or update, it fetches the record from the
author's PDS and admits that copy, never the payload inside the event. A delete
drops the record.

Every `--reconcile-hours` (default 24), a full backfill replaces the record
set. A repository that cannot be read during a reconcile keeps its previous
records. `reconcile` runs the same step once, from the command line.
`--duration` stops `follow` after a set number of seconds.

## Output

The output has the shape `build_points.write` produces (`version`, `lexicon`,
`records`, `excluded`) plus three fields: `generatedAt`, the stream `cursor`
the output reflects (null after a plain backfill), and `skipped`. Each record's
`site.source` is the at-uri of its contributor's latest record in that cell,
`site.provenance` lists every at-uri its metrics were medianed from, and
`site.contributor` is the contributor's DID.

A write goes to `points.json.tmp` first and is then renamed into place, so a
reader always gets a whole file. Every write also updates the dated copy for
that day, `points-YYYY-MM-DD.json`, which ends up holding the day's last write.
