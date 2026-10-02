# Measurement lexicon

`wtf.petrichor.drinkme.measurement` defines one benchmark observation, published
by `drinkme publish` to the participant's own atproto PDS. Each record includes
its hardware, model, and measurements so consumers can interpret it without
joining separate device records.

## Fields

| Field | Contents |
|---|---|
| `model` | Name, HF repo, and the resolved commit every arm loaded (`revision`, required: the 40-hex sha; with `--pack-dir`, the pack's bound commit). A bench of a local checkpoint directory has no commit to name: its record is written locally without `model.revision` and is not publishable |
| `compression` | The compression profile (`sip` or `gulp`), `bitsPerWeight` and `meanTensorBitsPerWeight`, and the loaded pack's packed/resident bytes when the arm ran off a pack on disk |
| `environment` | `deviceClass` (the normalized device name), memory capacity/kind, raw CPU/GPU identity, OS, `platform` (the compute platform: `cuda`, `rocm`, `metal`), and `engine` — the runtime and the kernels every arm ran, in one grammar (below) |
| `metrics[]` | Named values, units, and samples |
| `stock` | The stock arm's `outcome` (`measured`, `skipped_predicted_nonfit` or `failed_load`), with the `budgetBytes` a skip was judged against or the `error` a failed load raised |
| `raw` | Benchmark diagnostics (samples, bandwidth bytes, bytes per token, pack counts, twin arm, `device_source`, the detector's working notes as `detector`, the per-arm kernel routing as `<arm>_routing`, the host's 1-minute load around each arm's timing and its CPU count as `host_load`, the speculation pass's settings, prompts and per-arm samples as `spec` (or why it did not run), `unpublishable` on a local record); excluded from aggregation |
| `createdAt`, `version` | Timestamp and drinkme version |

The canonical metrics are `stock_decode_tok_s`, `compressed_decode_tok_s`,
`stock_prefill_tok_s`, `compressed_prefill_tok_s`, `stock_ttft_s`,
`compressed_ttft_s`, `read_gb_s`, `copy_gb_s`, `stock_weights_gb` and
`compressed_weights_gb`; their units and per-platform scope are specified in
the [schema](wtf.petrichor.drinkme.measurement.json). The arms depend on
the runtime. Stock is the runtime's own BF16 path: drinkme's on torch,
mlx-lm's on MLX. The twin on torch is the compressed arm's own kernels
reading the uncompressed BF16 weight, a diagnostic the results site does
not plot (the site's ratio is compressed ÷ stock). On MLX the twin is
drinkme's reference compute path over the pack decoded once at load, a
correctness check. On torch it also
writes `twin_decode_tok_s` and `twin_weights_gb`; `metrics` is an open list,
so they validate, and the schema's metrics description names them. On MLX
the twin publishes no metric, and its decode stays in `raw`. Its outcome
stays in `raw` on both.

When the checkpoint has an MTP head, bench on torch also times decode with
speculation on, as `drinkme serve --spec mtp` runs it, and writes
`<arm>_spec_decode_tok_s_<prompt>` for arms `stock` and `compressed` and
prompts `agent`, `chat` and `code`: tok/s over 256 new tokens, the median
of the prompt's timed repetitions. They are present only when the checkpoint
has a head, the arm was measured, and the runtime speculates; MLX does not,
so metal records carry none. `raw.spec` says why when they are absent
([bench](../docs/bench.md#speculation)).

`model.revision` is required: the resolved HF commit (the 40-hex sha of the
one snapshot every arm and the tokenizer loaded), never a branch name or a
local directory's structural digest. A record without one has no immutable
source identity — nobody can group it with another run of the same weights or
reproduce it — so it is not publishable: `drinkme publish` refuses it by name
and says the fix (bench against a hub repo). `drinkme bench` still writes such
a record locally, marked `raw.unpublishable`, so the operator keeps their
numbers ([bench](../docs/bench.md#local-records)).

`environment.engine` is one string in one grammar, slash-separated segments
each starting with its noun, so it is greppable by segment. After the
version-carrying segments (drinkme, then the runtime), every kernel segment is
`<noun> <impl>[ <version>]`: the implementation that ran, or `none`, with the
version wherever that implementation has one:

```
drinkme <semver> / torch <version>[+<accel tag>] / deltanet <fla <ver> | torch | none> / conv <fla <ver> | torch | none> / narrow-gemv <triton | torch> / stock-gemv <mv | linear | triton> / raw-gemv <twin | stock>[ / decode <eager | cudagraph>]
drinkme <semver> / mlx <version>
```

The torch form's kernel segments come from the routing record the arms agreed
on (`raw.<arm>_routing`; arms that routed differently write no record): the
DeltaNet decode recurrence — fla's fused Triton kernel with fla's version, the
torch recurrence, or `none` for a model without DeltaNet layers — the prefill
convolution kernel — fla's time-major Triton kernel with fla's version, the
torch conv1d fallback, or `none`, which must agree with the deltanet segment —
what ran the raw Linears under the codec's row threshold: `triton` for
drinkme's Triton GEMV, `torch` for `F.linear` under `DRINKME_NARROW_GEMV=0` —
the one-row call of the raw Linears at or above it (every Linear of
the stock arm): `mv` for `torch.mv` with hipBLASLt preferred, `linear` for
`F.linear` on PyTorch's default library, `triton` for the twin arm's Triton
GEMV over the raw BF16 weight — and the one-row call of the raw
Linears at or above it that the compressed and twin arms hold (a tied
`lm_head`): `twin` for the twin arm's Triton GEMV, `stock` for the
`stock-gemv` call. The decode segment names the step path the decode figures
ran on (a replayed CUDA graph, or eager); a record without it decoded eager.
The runtime version is `torch.__version__` verbatim, accelerator tag included.
The mlx form stops at the runtime; `environment.platform` says `metal`. For
example, `drinkme 1.0.0 / torch 2.12.0a0+rocm7.13.0a20260411 / deltanet fla
0.5.2 / conv fla 0.5.2 / narrow-gemv triton / stock-gemv triton / raw-gemv twin`
is a hybrid model on fla's kernels under ROCm on gfx1151, and `drinkme 1.0.0
/ torch 2.12.0+cu128 / deltanet none / conv none / narrow-gemv triton /
stock-gemv linear / raw-gemv stock` a dense model under CUDA. `bench.parse_engine` is
the inverse. Within a major version, new segments may be appended to the end
of either form; existing segments do not change meaning.

Four fields are closed enums: `compression.profile` (`sip`, `gulp`),
`environment.memoryKind` (`vram`, `unified`), `environment.platform` (`cuda`,
`rocm`, `metal`) and `stock.outcome` (`measured`, `skipped_predicted_nonfit`,
`failed_load`); a record naming any other value does not validate.

GB and GB/s are decimal (10^9). Fractional values — metric values and
`bitsPerWeight` — use decimal strings because the atproto data model has no
floating-point type; byte counts (`environment.memoryBytes`,
`stock.budgetBytes`, `compression.packedBytes`/`residentBytes`) are integers. Comparison rules are in [versioning](../docs/versioning.md). Round-trip verification is a write-time
requirement, so it is not a record field.

## Publishing and consumers

The CLI validates locally, authorizes through loopback OAuth, creates the record,
and reads it back for comparison. A published record is public and permanent,
so nothing local goes with it: publish drops `raw.pack_dir` (the local pack path
benches before 1.0.0's release wrote) and refuses a record with a home-directory
path in any field. It requests create access to this collection:
`repo:wtf.petrichor.drinkme.measurement?action=create`. An authorization server
returning `invalid_scope` triggers an announced fallback to the broader
`transition:generic` scope. See [publishing](../docs/bench.md#drinkme-publish).

Consumers can index this collection to build hardware comparisons or other
views. Compare matching model revisions within one drinkme major version;
distinguish predicted non-fits from attempted load failures. Records are self-reported.
The results site shows the records published in this collection.

This directory is the schema source. The schema is also published as a
`com.atproto.lexicon.schema` record, resolvable the standard way: the
`_lexicon.drinkme.petrichor.wtf` TXT record names
`did:plc:t3wzk4ypgx2ooi4zwjjioaet`, whose repository holds it at
`at://did:plc:t3wzk4ypgx2ooi4zwjjioaet/com.atproto.lexicon.schema/wtf.petrichor.drinkme.measurement`.
A schema change here is published there in the same release. A published
permission set could also provide application-specific consent wording.
