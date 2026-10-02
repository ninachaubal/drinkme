# Versioning

Drinkme uses semantic versioning. The running version is `__version__` in
[`src/drinkme/__init__.py`](../src/drinkme/__init__.py), repeated as the
project version in `pyproject.toml` and in `uv.lock`'s entry for drinkme.
`drinkme --version` prints it, and every measurement record carries it twice:
as `version`, the lexicon's one version field, and as the first segment of
`environment.engine` (`drinkme <semver>`).

| Change | Version increment | Packs | Published records |
|---|---|---|---|
| Bug fix | Patch | Unchanged | Still comparable |
| Kernel optimization or platform addition with the same pack format | Minor | Unchanged | Still comparable; new metrics and engine-string segments may appear |
| Codec or on-disk format change; a new meaning for an existing metric or engine-string segment | Major | Rebuilt | The previous major's records leave the site |

The model menu is defined in [`suggest.py`](../src/drinkme/suggest.py).

## Packs

The pack format is 1 (`FORMAT_VERSION` in
[`codec/pack.py`](../src/drinkme/codec/pack.py), `formatVersion` in
`meta.json`). Drinkme 1.x writes and reads only format 1. Every pack entry
point rejects any other version with a re-pack command, and there is no
migration ([pack format](pack-format.md)). A format 2 would ship as 2.0.0,
and every pack would be rebuilt with `drinkme pack --model <repo> --replace`.

A change to the tensor encoding or to an existing `meta.json` field is a new
format. Launch schedules are chosen at load and never stored in a pack, so a kernel
or schedule change needs no re-pack
([launch schedule](pack-format.md#the-launch-schedule)).

The persisted prefix-slot store has its own format number (`FORMAT_VERSION`
in [`serving/slotstore.py`](../src/drinkme/serving/slotstore.py)). A slot file
with another number is skipped and recomputed, never migrated.

## Measurement compatibility

Compare records within the same major version. `compatible` in
[`site/data/build_points.py`](../site/data/build_points.py) is the rule: a
record is comparable with this drinkme when its major version matches.
The site build applies the rule to every record and lists each
excluded record and its reason in `excluded` and on stdout. The
[aggregator](../site/aggregator/README.md) imports the same check (`admit`),
so the committed snapshot and the live service admit the same records.

Within a major version:

- An existing metric name keeps its meaning; a minor may add metrics.
- `environment.engine` may gain segments at its end; existing segments keep
  their meaning, and `bench.parse_engine` then also reads a string that
  ends before the new segment ([engine string](bench.md#the-engine-string)).
- Records from different minor and patch releases of one identity share a
  cell, since the chart does not split on them. `environment.engine` names
  the kernels each record ran, which is how a reader tells records from
  before and after a kernel change apart.

### The comparison identity

The chart medians records into one cell only when they agree on every term
of `identityKey` in [`site/index.html`](../site/index.html):

| Term | Record field |
|---|---|
| Major version | `version` |
| Device | `environment.deviceClass` |
| Checkpoint | `model.hfRepo` at its resolved `model.revision` |
| Compression profile | `compression.profile` |
| Platform | `environment.platform` |

Each cell has its own label, median and contributor count; where two cells
share a device, model and profile, the terms that differ are added to the
label. Before grouping, `admit` refuses a record, by name, when:

- it lacks an identity field (`IDENTITY_FIELDS`: `version`, `model.name`,
  `model.hfRepo`, `model.revision`, `environment.deviceClass`,
  `environment.platform`, `stock.outcome`), or its `version` is not a semver;
- its stock arm was measured but it has no `compression.profile`, or lacks
  `compressed_decode_tok_s`, `stock_decode_tok_s` or `read_gb_s`;
- it has no `environment.memoryBytes`.

The aggregator then collapses each contributor's records of one cell to one
record, their median. Its cell (`cell_key` in
[`site/aggregator/pipeline.py`](../site/aggregator/pipeline.py)) also
separates `environment.memoryBytes`, `environment.memoryKind`, and whether
the stock arm was measured. `drinkme publish` refuses a record whose
`model.revision` is not a resolved Hub commit
([local records](bench.md#local-records)).

So a record is comparable when a release of the current major made it
against a Hub commit and it carries every identity field and plotted metric.

The benchmark must use the packed tensors and kernels served by that version.
[`test_bench_arm_wiring.py`](../tests/test_bench_arm_wiring.py) checks the
compressed arm against a `pack_model` pack. The torch harness runs its own
loop over the Hugging Face forward pass; it does not measure HTTP or
`Engine.generate`. See [benchmark scope](bench.md#torch-benchmark-scope).

## What to re-run

Every change runs the CPU suite and the checks in its row of
[checks by change type](checks.md). A change that touches packs or records
adds these:

- **The codec or the pack writer.** Rerun the codec gates, re-pack a menu
  model, and run `drinkme verify` and `drinkme bench --pack-dir` on it
  ([codec / pack format](checks.md#codec--pack-format)). A new format is a
  major: every pack is rebuilt, and the site has no records for the new
  major until they are benchmarked and published with it.
- **A kernel or a launch-schedule row, same pack format.** A minor. Rerun the
  kernel gates, and `drinkme bench --pack-dir` on a real pack for a speed
  claim ([kernel / launch schedule](checks.md#kernel--launch-schedule)).
  Packs are unchanged. New records measure the new kernel and share cells
  with the major's earlier records.
- **A new metric.** A minor. Name it in the lexicon's `metrics` description;
  `tests/test_bench_record_shape.py` holds the metrics a record carries to
  that list. Then run the
  [benchmarking and publishing](checks.md#benchmarking-and-publishing)
  checks: `drinkme bench --dry-run`, a real `drinkme bench`, and
  `bench/publish_e2e.py` on a test PDS.
- **An existing metric's meaning or name.** A major. The page reads metrics
  by name, so a record that carries the old name cannot be plotted beside
  one that carries the new.
- **The engine string.** A segment appended at the end is a minor; a new
  meaning for an existing segment is a major. `bench.engine_string` composes
  it and `bench.parse_engine` inverts it; the lexicon's `engine` description
  and [the grammar](bench.md#the-engine-string) state it, and
  `tests/test_engine_string.py` holds the lexicon to it.
- **The comparison identity or admission.** `admit` and `IDENTITY_FIELDS`
  in `build_points.py`, `identityOf` and `identityKey` in `site/index.html`,
  and `cell_key` in the aggregator change together.
  `tests/test_site_points.py` runs the page's grouping block under Node, and
  the aggregator has its own tests (`site/aggregator/tests`).
