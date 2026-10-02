# Bench and publish

`drinkme bench` measures local inference and writes a record.
`drinkme publish` shares that record on your atproto PDS. Publishing is optional;
benchmarking needs no atproto account or authentication.

Use the [installation instructions](../README.md#run-the-server) first. In an
existing environment:

```sh
uv run --no-sync drinkme bench --dry-run
uv run --no-sync drinkme bench --model Qwen3-8B
uv run --no-sync drinkme publish --handle you.example.com
```

On Apple silicon, `drinkme bench --runtime mlx` is not yet practical at model
scale (the reference twin decodes on the CPU); see [Metal status](metal.md).

## `drinkme bench`

A run detects the device, chooses a model, measures device read bandwidth,
and loads each benchmark arm separately. Decode uses 64 greedy tokens: one
untimed warm-up repetition (Triton compile, first touch, allocator growth),
then three timed repetitions, reporting the median and the timed samples
only. `raw.warmup_rep`
records that the warm-up ran; `DRINKME_BENCH_WARMUP=0` opts out and records
`false`. Stock and compressed arms also measure prefill and time to first
token (TTFT), with the same warm-up. Packed tensors must pass a bit-exact
round trip before a record is written.

Records go to `measurements/<model>_<device>_<date>.json` under the current
directory; a record made with a compression profile other than the default
appends it (`…_<date>_gulp.json`). Repeated runs add `_2`, `_3`, and so on. `-o FILE` chooses another
path when `--model` is given; without `--model` it is ignored with a warning,
since a menu run can produce two records. Records follow the [measurement lexicon](../lexicons/README.md).

The model download can be tens of gigabytes. `--model Qwen3-1.7B` selects a
smaller model. Qwen downloads need no login. Gemma is offered only if an HF
HEAD request confirms access using your token; `--no-gemma` skips that check.

`--pack-dir DIR` (with `--model`, torch runtime) times the compressed arm
using an existing pack through the serving loader instead of re-packing in
memory. Before any arm runs, the shared validation code checks the pack without
importing torch and rejects unsupported format versions or checkpoint mismatches.

All arms read the same snapshot. Before loading the tokenizer or models,
`serving.checkpoint.resolve_source` resolves one source commit. With
`--pack-dir`, it uses the pack's bound commit and rejects a pack bound to a
different commit than the menu entry's pinned revision, naming both identities. Otherwise it resolves the menu's revision
once, so a moving Hub ref cannot give different arms different checkpoints.
The record's `model.revision` contains the measured commit, including for
unpinned menu entries such as Qwen3-0.6B, 14B, and 32B.

<a id="local-records"></a>

### Local records

`model.revision` is the resolved hub commit — the 40-hex sha of the one
snapshot every arm loaded — and the lexicon requires it. A checkpoint that is
not a hub snapshot (a local directory) has no commit to name: `resolve_source`
gives it a structural digest (`local-…`, over headers, config and tokenizer
files — not the weight bytes), which identifies it on this machine but lets
nobody else group or reproduce the run. Such a bench still writes its record —
the operator's own numbers — with no `model.revision`, the digest in
`raw.resolved_revision`, and `raw.unpublishable` saying why; the run's last
line says `drinkme publish` will refuse the file. Bench writes without
validating against the lexicon; publish does the refusing.

Metal records are ordinary records: bench on Darwin/arm64 writes one with
`environment.platform` `metal`, `drinkme publish` checks and sends it like any
other, and the results site's aggregator charts it.

### Torch benchmark scope

The CUDA/ROCm harness in [`arms.py`](../src/drinkme/arms.py) runs three module
trees through the same Hugging Face `model.forward`:

| Arm | Implementation | Purpose |
|---|---|---|
| Stock | Raw BF16 weights through drinkme's runtime: the same [DeltaNet recurrence kernel](serve-kernels.md#deltanet-recurrence-kernel) and [narrow GEMV](serve-kernels.md#narrow-gemv) as the other arms and the [stock GEMV](serve-kernels.md#stock-gemv) for every other Linear, no codec — not transformers' eager path | Baseline runtime |
| Compressed | `pack_weight_served` and `swap.make_module`, using the packer's format and the serving kernels | Measure the compressed representation |
| Twin | `RadixTwinLinear`: the uncompressed BF16 weight through the compressed arm's own kernels, with the decode replaced by a load of the raw weight, at the box's twin rows; a raw-fallback tensor is the same `RawLinear` as in the compressed arm | The order-matched control: the same kernels, reading raw BF16 |

The results site plots compressed/stock: the stock arm is closest to what a
user runs without drinkme's compression. The twin/compressed comparison
isolates the effect of reading fewer bytes, and stock/twin the effect of
changing kernels; the twin is for tuning and diagnosis, and records carry
its numbers as data. On small models the stock arm is bound by kernel
launches rather than bandwidth, so compressed/stock there includes the
compressed arm's kernels as well as its smaller reads. Bench prints both
ratios, and each arm's fraction of its bandwidth bound (`read_gb_s` over the
bytes one of the arm's decode steps reads, `<arm>_decode_read_gb`), beside
them.

Repeated runs move a little. In the 1.0.0 records from one Strix Halo, each
model's stock arm ran twice, once per compression profile, and every pair
agrees within about 0.5% (the widest is Qwen3.8-27B, 3.90 and 3.92 tok/s).
Other work on the machine moves them more: CPU memory traffic shares the
bandwidth the GPU reads, and lowers the read probe with it. Both arms of one
run share the conditions, so the ratio moves less than either arm's tok/s.

The twin is order-matched: its gemv and multi-column kernels are the
compressed arm's (`RAW=True` in `radix_kernel_gpu._gemv` and
`radix_ops._gemv_mc`), with the same program decomposition, accumulators,
reductions, split-K pass and bias fusion at any one launch schedule. It
runs its M=1 GEMV at the box's own twin rows
([launch schedule](pack-format.md#the-launch-schedule)): the compressed
rows are tuned for the decoder, and on gfx1151 the one-warp whole-row
program they choose reads raw BF16 1.05–1.6× slower than the eight-warp one
the twin rows name. A box without twin rows runs the twin at the
compressed tensor's row, as does `DRINKME_TWIN_SCHEDULE=served`. It routes by M as the compressed module does, and
its dense route is `F.linear` over the raw weight. In the source only the
weight read differs, so the two arms' outputs should be bitwise equal. The
compiler can still treat the twin's smaller loop differently: on a row whose
width is not a multiple of 1024 it may fuse a multiply and an add that the
served kernel keeps apart, which changes the last bits.
[`bench/radix_twin_bitpin.py`](../bench/radix_twin_bitpin.py) checks bitwise
equality on a GPU at every launch-table row and twin row and records a wider
tiles × warps grid; its `--compile` mode compares the compiled kernels on the
CPU. The twin's weight load is capped at the activation load's vector width,
because Triton takes the loop's register layout from its widest vectorized
load and `tl.sum` adds each thread's share in that layout; the
decoder's loads are gathers, so in the served kernel the activation sets it.
At the twin rows the twin is the fastest this kernel reads raw BF16 on the
box, not the fastest BF16 kernel for the device. It is built from the compressed arm's per-tensor codec and profile,
so it runs after the compressed arm. Bench times the twin's decode only.
[`test_bench_arm_wiring.py`](../tests/test_bench_arm_wiring.py) checks that the
compressed arm's tensors match `pack_model` output byte for byte.

The stock arm is drinkme's own eager BF16 forward, its one-row Linear calls
through the box's [stock GEMV](serve-kernels.md#stock-gemv): the fastest
one-row BF16 call drinkme has on the box, which `drinkme serve --stock` runs
too. On gfx1151 that is the twin arm's own Triton GEMV over the raw weight,
faster on every measured shape than `torch.mv` under hipBLASLt and than
PyTorch's default `F.linear`, which goes to rocBLAS at about a quarter of the
wall on small models
([`bench/stock_blas_knobs.py`](../bench/stock_blas_knobs.py) measures the
knobs). There the stock arm and the twin make the same one-row call on every
Linear, and their decode figures agree. On hosts where decode is launch-bound
the stock arm still runs below the bandwidth wall, so compressed/stock is a
ratio against this runtime's BF16 path, not against the fastest BF16 engine
for the device. The raw Linears the
compressed and twin arms hold, a tied `lm_head`, make their one-row call
through the box's [raw GEMV](serve-kernels.md#raw-gemv), the twin
arm's own kernel on gfx1151. For a comparison against another engine, run an
external engine such as llama.cpp at BF16 on the same model
(`bench/llamacpp_bench.py`).

The harness uses its own greedy loop. It does not exercise the server's
sampler, prefix cache, or HTTP path; only the [speculation pass](#speculation)
runs the server's engine. For client-visible timing,
use `drinkme serve` with the [endpoint instruments](../bench/README.md).

<a id="speculation"></a>

### Speculation

On torch, when the checkpoint has an MTP head (on the menu, Qwen3.8-27B;
MiMo's config names a head but its checkpoint carries no `mtp.*` tensors),
bench also times decode with speculation on, the way `drinkme serve --spec
mtp` runs it: the checkpoint's head drafts and the model verifies. The pass
runs on each measured stock and compressed arm after that arm's plain
timings, over the same loaded weights, through the server's engine with the
head serve would load for that arm: the raw BF16 head for stock, the pack's
compressed head under `--pack-dir`, and otherwise the same head packed in
memory as `drinkme pack` writes it. Whether there is a head is the server's
own decision (`engines._mtp_head`), not a reading of the config. The pass
times three of [`ngram_gpu_ab.py`](../bench/ngram_gpu_ab.py)'s prompts (an
agent transcript, a chat turn, a code edit; the text is in
[`spec_pass.py`](../src/drinkme/spec_pass.py)), greedy, 256 new tokens, one
untimed warm-up and then three timed repetitions per prompt, with the prefix
cache off and thinking off. Decode tok/s counts the tokens after the first
over the time after the first, so the prompt's prefill is not in it.

The record carries `stock_spec_decode_tok_s_<prompt>` and
`compressed_spec_decode_tok_s_<prompt>` for `agent`, `chat` and `code`, and
`raw.spec` carries the settings, each prompt's hash and token count, and per
arm and prompt the samples and the mean number of drafts each verify step
accepted. Without a head, on the MLX runtime (which does not speculate), or
under `--no-spec`, there are no such metrics and `raw.spec.skipped` says why.
The twin arm has none, and a stock arm that was not measured has none. The
plain decode metrics are the same with and without the pass.

The two decode numbers come from different harnesses: plain decode is
bench's own 64-token loop on a short prompt, and the speculation pass is the
server's engine at 256 tokens. Compare `compressed_spec_decode_tok_s_*` with
`stock_spec_decode_tok_s_*`, not with `compressed_decode_tok_s`.

<a id="kernel-routing"></a>

### Kernel routing

After each arm's weights are on the device, the harness applies the same
routine `drinkme serve` applies to a loaded model
([`serving/kernel_route.py`](../src/drinkme/serving/kernel_route.py)): the
DeltaNet decode recurrence is bound to FLA's fused kernel when the startup
probe passes, large ROCm prefill convolutions use FLA after their own probe,
raw Linears below the codec's row threshold use the Triton GEMV, and the
raw Linears at or above it make their one-row call through the box's
stock GEMV, or in the compressed and twin arms through its
[raw GEMV](serve-kernels.md#raw-gemv). `DRINKME_DELTANET_KERNEL`,
`DRINKME_DELTANET_CONV`, `DRINKME_NARROW_GEMV`, `DRINKME_STOCK_GEMV` and
`DRINKME_RAW_GEMV` apply to every arm.
The run prints one line per arm, for example `stock arm: deltanet fla,
narrow 96, conv fla`, and a `routing:` summary beside the ratio; each arm's route is
recorded as `raw.<arm>_routing` with `deltanet_kernel` (`fla`, `torch`, or
`none` for a model without DeltaNet layers), `deltanet_conv` (the same three
values for [prefill convolution](serve-kernels.md#deltanet-prefill-convolution)), `narrow_gemv` (false only under
`DRINKME_NARROW_GEMV=0`), `narrow_count`, `stock_gemv` (`linear`,
`mv` or `triton`) and `raw_gemv` (`twin` or `stock`). The two GEMV fields are the box's
modes, the same in every arm: the stock arm holds no codec tensor for the
raw GEMV to apply to.

Arms that route differently produce no record: the run stops with
`drinkme: refusing to write the record`, naming each arm's route, before the differing arm
is timed. A ratio between a stock arm on the torch recurrence and a compressed
arm on the fused kernel would measure the runtime, not the weight read (about
7% on Qwen3.8-27B in maintainer runs). The twin is routed the same way, since its decode figure is
read against both neighbours.

<a id="the-engine-string"></a>

### The engine string

Because the arms route alike, a record has one routing, and the typed
`environment.engine` names it beside the runtime, in one grammar. Slashes
separate segments; each segment starts with its noun, so the string is
greppable by segment. After the version-carrying segments (drinkme, then the
runtime), every kernel segment is `<noun> <impl>[ <version>]`: the
implementation that ran, or `none`, with the version wherever that
implementation has one:

```
drinkme <semver> / torch <version>[+<accel tag>] / deltanet <fla <ver> | torch | none> / conv <fla <ver> | torch | none> / narrow-gemv <triton | torch> / stock-gemv <mv | linear | triton> / raw-gemv <twin | stock> / decode <eager | cudagraph>
drinkme <semver> / mlx <version>
```

The runtime version is `torch.__version__` verbatim, accelerator tag included.
The `deltanet` segment is the recurrence kernel every arm ran — fla's fused
Triton kernel with fla's version, the torch recurrence, or `none` for a model
without DeltaNet layers; `conv` is the prefill convolution kernel every arm
ran — fla's time-major Triton kernel with fla's version, the torch conv1d
fallback, or `none`, which must agree with the `deltanet` segment;
`narrow-gemv` is what ran the sub-threshold raw Linears — `triton` for
drinkme's Triton GEMV, `torch` for `F.linear` under `DRINKME_NARROW_GEMV=0`;
`stock-gemv` is the one-row call of the raw Linears at or above the
threshold, every Linear of the stock arm ([stock GEMV](serve-kernels.md#stock-gemv)) —
`mv` for `torch.mv` with hipBLASLt preferred, `linear` for `F.linear` on
PyTorch's default library, `triton` for the twin arm's Triton GEMV over the
raw BF16 weight; `raw-gemv` is the one-row call of the raw
Linears at or above the threshold that the compressed and twin arms hold,
a tied `lm_head` ([raw GEMV](serve-kernels.md#raw-gemv)) — `twin` for the
twin arm's Triton GEMV, `stock` for the `stock-gemv` call; `decode` is the
step path the decode figures ran on: `cudagraph` when every arm replayed a
captured CUDA graph ([CUDA graphs](serve-kernels.md#cuda-graphs)), `eager`
otherwise. Records written before the decode segment existed have seven
segments and decoded eager. A run whose arms took different step paths
(a capture that failed on one arm only) is written locally, marked
`raw.unpublishable`, and refused by `drinkme publish`.
For example, `drinkme 1.0.0 / torch 2.12.0a0+rocm7.13.0a20260411 / deltanet
fla 0.5.2 / conv fla 0.5.2 / narrow-gemv triton / stock-gemv triton /
raw-gemv twin` is a hybrid model on fla's kernels under ROCm on gfx1151; `drinkme
1.0.0 / torch 2.12.0+cu128 / deltanet none / conv none / narrow-gemv triton /
stock-gemv linear / raw-gemv stock` a dense model under CUDA. The segments are composed
from `raw.<arm>_routing` by `bench.engine_string`; `bench.parse_engine`
inverts them, refuses a `deltanet`/`conv` disagreement on whether DeltaNet
layers exist, and rejects anything outside the grammar. The MLX arms are not
routed (both routes are torch concepts), so the mlx form names the runtime
only; `environment.platform` says `metal`. Within a major version, new
segments may be appended to the end of either form; existing segments do not
change meaning.

### Host load

Other work on the host slows the arms unevenly. On an APU the CPU and GPU
share one power budget and one memory bus, and a small model's decode waits
on the host's kernel launches. Bench records the host's 1-minute load
average before and after each arm's timed passes, with the CPU count, as
`raw.host_load`, and prints them under the ratios. When an arm was timed
with the load above a fifth of the CPU count, the summary adds a warning; the
record is still written. On a 32-thread Strix Halo the line is 6.4:
runs at or under it were within 3% of a quiet run on every arm, and Qwen3-8B
runs at loads of 11–27 lost 11–17% on the compressed arm against 2–10% on the
twin and under 6% on stock, so a ratio from a busy host is biased against
compression.
The comment above `probe.HOST_LOAD_WARN_PER_CPU` lists the runs. For a
comparable record, stop other work and run again.

<a id="the-served-ruler"></a>

### Server timing

Server measurements use the default prefix cache and slot count and state the
`--ctx` value. This includes per-token [cache overhead](serve-prefix-slots.md)
that `drinkme bench`'s forward loop does not measure.

Label any changes to serving defaults, including disabling the cache.
[`bench/serve_drive.py`](../bench/serve_drive.py) enforces
this by default: it reads the running server's `/health` and refuses to
record a run whose prefix cache is off unless `--allow-cache-off` is passed,
in which case the label gets a `-cache-off` suffix.

### Loaders

All three BF16 arms default to streaming: build the module tree on the meta
device, read one checkpoint tensor, install it on the device, and release the
host copy before reading the next. The compressed arm packs each eligible
Linear as it arrives; the twin installs its BF16 replacement. Embeddings,
norms, biases, and ineligible weights use the shared checkpoint walker.
Tied weights are re-tied, and computed buffers are created by the module.
A vision model is measured with its vision tower, as `drinkme serve` holds
it: all three arms build the tower (`arms._TOWER_BY_DEFAULT`), so the weights
and resident figures and both bits-per-weight figures include it. The record
says so in `raw.vision_tower_included`, and a `--pack-dir` record carries the
pack's `vision` block as `raw.pack.vision`.

On unified memory this limits host loading overhead to one tensor beyond
resident weights. `--stock-loader from_pretrained` selects the alternative
stock loader, whose measured loading transient is approximately twice the
model's resident size on unified memory. Its fit check uses that larger charge.

CPU tests compare logits bitwise between the streaming loaders and the
whole-model ones (transformers' `from_pretrained`; `load_cpu` + `swap_linears`):
[`test_stock_stream_loader.py`](../tests/test_stock_stream_loader.py) and
[`test_stream_compressed_arms.py`](../tests/test_stream_compressed_arms.py).
These cover tied heads, shards, biases, computed buffers, and wrappers with
a vision tower. GPU correctness still requires hardware acceptance tests.

### Apple silicon

The MLX harness in [`arms_mlx.py`](../src/drinkme/arms_mlx.py) loads the
packs the mlx runtime serves — coded BF16 with its raw fallbacks: stock is
mlx-lm, twin is drinkme's reference path over the pack decoded once at load,
and compressed is its fused path.
Every timed forward, prefill, TTFT and decode alike, applies the output head
to the last position only, as serve does (`engine_mlx.last_row_logits`) and
as the torch arms do with `logits_to_keep=1`, so both sides time the same
work. Tokens are unchanged: the last row is the full head's row.
So on MLX, stock is the runtime's own BF16 path but not drinkme's, and the
twin is a correctness check rather than a bandwidth control. Its speed
is not published as a
comparable kernel metric. It decodes the pack once at load with an MLX
decoder that is bitwise the CPU oracle, then runs the reference path's
matmul over the BF16 weights. Measurements include minimum, median, and
thermal drift. See [Metal status and verification](metal.md).

### Automatic model selection

Without `--model`, bench runs the menu's ratio point (stock and compressed fit)
and, when available, a larger fit point (compressed fits and stock is predicted
not to). Each produces a separate record. `--model` selects one menu model, by
name or by its HF repo id. Bench refuses a repository that is not on the menu,
although `serve`, `pack` and `check` accept any HF repo.

A menu entry carries no sizes. A model's BF16 size is its checkpoint's
safetensors bytes, read from the headers when the checkpoint is in the local
Hugging Face cache, and otherwise from the Hub's safetensors metadata for that
revision (one anonymous request, no weights). Its compressed size is the
`residentBytes` of a pack of it on this machine (`--pack-dir`, or the
profile's default pack directory). Without a pack, it is estimated as the BF16
size times the profile's ratio (`suggest.COMP_RATIO`, measured from Strix
Halo records), and the menu and fit-check lines mark it as an estimate.
Bench can't size a model whose checkpoint is neither cached nor reachable, so
it leaves that model out of the menu and says so.

### Memory fit check and fit-point records

Before loading, each arm is charged against available memory with headroom:

| Arm / loader | Unified-memory loading charge |
|---|---|
| Stock, streaming | BF16 checkpoint bytes + largest checkpoint tensor |
| Stock, `from_pretrained` | Approximately 2× BF16 checkpoint bytes |
| Twin, BF16 | BF16 checkpoint bytes + largest checkpoint tensor |
| Compressed | Compressed resident bytes (a pack's own, or the estimate) + largest checkpoint tensor |

The largest tensor comes from cached safetensors headers; without them, the
plan uses a 5 GiB bound. Discrete GPUs use a separate VRAM pool, so stock's
host transient is not charged to VRAM. The checks run again immediately before
allocation against the memory then available. `raw.stock_loader` records the
selected stock loader.

| `stock.outcome` | Meaning | Beside it in `stock` |
|---|---|---|
| `measured` | Stock ran successfully | the `stock_*` metrics are present |
| `skipped_predicted_nonfit` | Fit policy skipped the load | `budgetBytes`, the ceiling it was judged against |
| `failed_load` | The live guard or an out-of-memory error prevented loading | `error`, the message |

Unmeasured stock metrics are omitted. A predicted non-fit is a policy decision;
a different loader or margin might fit, so a skip does not establish that
hardware ran out of memory. Non-memory failures, such as missing dependencies
or an unsupported config, still fail the run.

On torch the BF16 twin publishes its decode as `twin_decode_tok_s` and its
weight footprint as `twin_weights_gb`, beside stock's metrics. The results
site divides the compressed decode by stock's, not the twin's. Its outcome
(`raw.twin_outcome`, `measured` or `skipped_predicted_nonfit`) stays in `raw`.
The twin needs one BF16 copy, so it can run where stock was skipped. When both
stock and twin are skipped,
the record contains compressed metrics only. For example, a Qwen3-8B record
on a 16 GB card has this form.

The compressed arm uses 1.10× headroom. A model fitting only without headroom
is attempted with that reduced margin; a model exceeding the hard budget is refused
before loading. A compressed-arm failure produces no record.
[`fit72b_acceptance.sh`](../bench/fit72b_acceptance.sh) adds an external memory
watchdog for the large unified-memory fit case.

### Units

`raw` values are bytes and bytes per second. Public metrics use decimal GB and
GB/s (10^9), as decimal strings because atproto records do not support
floating-point numbers; `environment.memoryBytes`, `stock.budgetBytes` and the
pack byte counts are integers.

Memory metric scope depends on the runtime: torch reports the caching allocator's
live allocation after loading; MLX reports engine resident-weight accounting.
Neither is total process memory or a working set including KV cache.

`compression.packedBytes` and `compression.residentBytes` are the loaded pack's
`meta.json` numbers (the ones `drinkme pack` wrote): present when the compressed
arm loaded a pack from disk (`--pack-dir` on torch; every MLX run), absent when
the torch arm re-packed in memory. For a vision model both include the
tower.

### `--dry-run`

```sh
drinkme bench --dry-run --model Qwen3-8B
```

Prints JSON with models, predicted sizes, arm loaders, memory charges and
headroom, fit verdicts, record shape, and output filenames. It does not import
torch, bootstrap dependencies, download weights, probe bandwidth, or write files.
For a menu model whose checkpoint is not cached, it reads the checkpoint's
safetensors metadata from the Hub, without a token.
`environment` is the block a record would carry, `detector` the detector's
notes (a record's `raw.detector`). Sizes are decimal GB (`bf16GB`,
`compressedGB`, `…ChargeGB`). `compressedGBEstimated` is true when
`compressedGB` is the ratio estimate rather than a pack's own size, and
`compressedGBFrom` says which. `compressedArmFitsWithoutHeadroom` is true when
the compressed arm fits only without the fit headroom, the knife-edge case.
`--detect-only` stops earlier, after hardware identity and the model menu.

### Naming a discrete AMD card

[`detect.py`](../src/drinkme/detect.py) looks up the PCI device ID and revision
in `_AMD_PCI_NAMES`, using names from libdrm's `amdgpu.ids` and gfx targets from
LLVM's processor table. The table covers RDNA3/RDNA4 discrete GPUs, recent
APUs, and CDNA Instinct devices. Detection coverage is broader than tested
runtime coverage; see [hardware support](hardware.md).

If the table misses, detection tries sysfs `product_name`, `rocminfo`, and
`rocm-smi --showproductname`, then falls back to a name such as `amdgpu 0x7480`.
The table wins on conflicts. A fallback-only name blocks record writing unless
`--allow-unknown-device` is given; filenames retain the PCI ID.

## `drinkme publish`

```sh
drinkme publish --handle you.example.com
drinkme publish                         # newest record, remembered identity
drinkme publish measurements/qwen3-8b_nvidia-geforce-rtx-4090_2026-09-20.json
drinkme publish measurements/*.json     # a set: each record checked, the good ones sent
```

Publish writes a `wtf.petrichor.drinkme.measurement` record to **your own PDS**,
attributed to your DID. The record is self-contained so other atproto apps can
collect, compare, and display records without a drinkme-specific service.
The [lexicon](../lexicons/README.md) defines the fields.

The first run opens your PDS's authorization page; later runs reuse the stored
session. Each step is printed, and a failure stops the operation:

1. **Validate:** a record without a resolved hub commit in `model.revision`
   is refused by name, with the fix in the reason — bench against a hub repo;
   a [local record](#local-records) has only a structural digest and stays
   local. Then the record is checked against the lexicon, including required
   fields, enums, datetime format, and decimal-string numbers. Incomplete runs
   are refused. Refusals are per record: in a set,
   the other records still go, the refused are listed on stderr at the end,
   and the exit status is 3 ([exit codes](cli.md#exit-codes)); a set with
   nothing publishable stops here, before any network. The refusal for a
   local record reads:

   ```
   drinkme publish: measurements/qwen3.8-27b_amd-ryzen-ai-max-395-w-radeon-8060s_2026-09-20.json: no model.revision — the record cannot name the commit its arms loaded, so nobody can group or reproduce it; bench against a hub repo (a --model menu entry, or a --pack-dir bound to a hub commit) so the revision resolves to the 40-hex sha — a local checkpoint directory has only a structural digest (raw.resolved_revision), and its record stays local
   ```
2. **Resolve identity:** resolve the handle to a DID, confirm the DID document
   claims that handle, and find its PDS. `--did` accepts a DID directly;
   `--plc URL` overrides the default `https://plc.directory` for did:plc.
3. **Authorize:** use loopback OAuth with PAR, PKCE, and DPoP. You sign in on the
   PDS's page; drinkme does not receive your password. The redirect must reach
   the temporary listener on this machine's `127.0.0.1`. `--no-browser` prints
   the URL without opening it.
4. **Write:** call `com.atproto.repo.createRecord` and print the `at://` URI.
   Validation is local; the write uses `validate: false` because the PDS may
   not have the lexicon.
5. **Read back:** retrieve the record and compare it with the submitted value
   after JSON canonicalization. A mismatch fails the publish. Steps 4 and 5
   run once per record of a set; identity and authorization happen once.

### Permissions and local state

The requested scopes are `atproto` and
`repo:wtf.petrichor.drinkme.measurement?action=create`. If the authorization
server returns `invalid_scope`, publish retries with `transition:generic`,
which grants broader repository write access, and prints that change before
opening the URL. The publisher itself writes only the record.

The client appears as a localhost development client on the consent screen.
Identity settings are saved in `~/.config/drinkme/publish.json`. Tokens and the
DPoP key are stored in `~/.config/drinkme/session.json` with mode 0600;
`--logout` removes the session. `DRINKME_CONFIG_DIR` relocates these files.

### Site integration

The results site shows the measurement records published on atproto
(`wtf.petrichor.drinkme.measurement`). Its
[aggregator](../site/aggregator/README.md) reads them from the network,
validates and filters them, and writes the page's data, so a record you
publish reaches the page with no further step. `site/data/points.json`,
committed in this repository, is a snapshot of such records. Charts group
comparable records and show median ratios and sample counts. Bandwidth-based
plausibility checks can flag outliers but do not independently verify a
submitted measurement.

## Bandwidth interpretation

The bandwidth wall is the measured read bandwidth divided by the bytes one
decode step reads: the model's Linears, packed or raw, and one row of the
token embedding. It is the decode rate if every token read those bytes once.
A vision model's tower and the embedding rows a step does not look up are
resident, so they count toward memory (`<arm>_weights_gb`), but a text
decode step never reads them, so they do not count toward the wall. Each arm
records its own denominator as `<arm>_decode_read_gb`
(`arms.decode_read_bytes`), and bench prints each arm's decode rate as a
fraction of `read_gb_s / <arm>_decode_read_gb`. Kernel efficiency and other
work keep achieved throughput below the wall. Speculative decoding can
exceed it by producing several tokens per weight read. See
[method](method.md).
