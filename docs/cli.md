# CLI reference

Commands assume an activated environment; otherwise prefix them with
`uv run --no-sync`. Flags override their corresponding environment variables.
Use `drinkme <verb> --help` to check the installed version's interface.

## Exit codes

Every verb (`check`, `pack`, `verify`, `serve`, `bench`, `publish`,
`bootstrap`) and every usage error share one table:

| code | meaning | examples |
|---|---|---|
| 0 | success | `check` of an eligible repo this machine can pack; a clean `serve` shutdown |
| 1 | a bug: an uncaught exception, traceback printed | file an issue |
| 2 | usage: the command line is wrong | a missing or conflicting flag, an unknown menu model name, `pack` onto an existing pack directory without `--replace` |
| 3 | refused: a definite verdict about the model, pack or record | an FP8/quantized/non-BF16 checkpoint, `check` refused, `verify`'s hash check failed, a `verify --upstream` MISMATCH or NOT COVERED, a `publish` record that fails the lexicon |
| 4 | can't run here: this machine or its environment | no C++ compiler on PATH, not enough memory for `--ctx`, no usable accelerator device, HF/network unreachable, a port already in use, an OAuth/auth failure |

A refusal about the model beats an environment problem: a refused repo on a
machine with no compiler exits 3, not 4. Warnings (a busy host, thermal
throttling, measurement drift) never change the code — bench's numbers are
the contributor's. `check` on an eligible repo this machine cannot currently
pack prints the verdict (exit-independent) and exits 4 rather than 0; its
`--json` verdict carries `encoderProblem` so a script can tell that case
apart from `check`'s own "could not check" (also 4, a network problem, not a
verdict).

## `drinkme serve`

Without `--model`, the server ranks cached packs and menu models by estimated
memory use and proposes the largest fit. Confirm with Enter/`y`, cancel with
`n`/`q`/EOF/Ctrl-C (exit 2), or enter another model name or HF repo. `--yes` and
non-interactive stdin accept the choice. Missing packs are created before serving.
See [models](models.md) for selection details, [hardware](hardware.md) for setup,
and [clients](clients.md) for base URLs, keys, and client recipes.

| flag | default | what it does |
|---|---|---|
| `--model MODEL` | detect + confirm | menu model name or HF repo |
| `--yes` | off | skip the confirm prompt when no `--model` is given; serve the pick |
| `--port PORT` | 3215 | listen port |
| `--host HOST` | 127.0.0.1 | listen host; `0.0.0.0` to serve the LAN, and then set `--auth` and `--no-image-urls` |
| `--stock` | off | serve uncompressed BF16 through the same serving stack for A/B comparisons |
| `--runtime R` | host | `torch` (CUDA/ROCm/CPU, `serving/engines.py`) or `mlx` (Apple silicon, `serving/engine_mlx.py`); default `mlx` on Darwin/arm64, `torch` elsewhere. MLX is in development. On `mlx`: no prefix cache yet (every request re-prefills); `--prefix-slots` above 1, `--rope-scaling`, `--spec` and sleep/wake are refused or 501 ([metal.md](metal.md)) |
| `--pack-dir DIR` | cache | serve this pack instead of the cached one |
| `--no-auto-pack` | off | fail instead of packing when the cache misses; nothing is downloaded either |
| `--no-hub-pack` | off | on a cache miss, download the BF16 checkpoint and pack it here without looking for a published pack ([published packs](serve.md#published-packs)) |
| `--ctx N` | 8192 cap | context to allocate (a static cache bound); memory is linear in N — Qwen3-8B at its native 40k preallocates ~6 GB |
| `--rope-scaling SPEC` | off | opt-in YaRN: `yarn:<factor>[:<original>]`, the vendor recipe's pre-extension window (Qwen3's card: `yarn:4:32768` → 131072); `--ctx` alone never turns it on |
| `--auth TOKEN` | none | require `Authorization: Bearer TOKEN` on `/v1/*`; `/health` stays open |
| `--prefix-slots N` | 1 | whole prefix-cache states retained in accelerator memory; `0` turns the prefix cache off (a per-request cache); counts that exceed the budget are rejected at load with a memory estimate |
| `--ctx-checkpoints N` | 32 | context checkpoints kept per prefix slot (llama.cpp's flag): a prompt that parts from a slot is served up to their shared prefix; `0` keeps the extends-only rule ([context checkpoints](serve-prefix-slots.md#context-checkpoints)) |
| `--spec MODE` | auto | speculative proposer: `off`, `mtp` (the checkpoint's head), `ngram` (prompt lookup, no weights), `ngram+mtp` (chained), `auto` (mtp if the checkpoint has a head, else ngram) |
| `--advertised-ctx N` | real ctx | report a smaller window on `/v1/messages` so Claude Code's auto-compact fires before the real one is exhausted; input-token usage is scaled; engine accounting is unchanged |
| `--served-model-name NAME...` | none | extra ids this server also answers to, listed on `/v1/models`; a menu name given to `--model` is added automatically |
| `--profile NAME=KEY:VAL,...` | none | named generation profile containing sampling and chat-template overrides, served as `<id>:NAME`; repeatable; overrides `profiles.json` beside the pack. Compression uses `drinkme pack --sip`/`--gulp` |
| `--sleep-on-idle SEC` | 0 = never | park the model in host RAM after SEC seconds without a generation |
| `--image-max-pixels N\|WxH` | 2560x1440 | the largest image a vision model reads: a bigger one is resized down to this many pixels. A request's `detail: "low"` asks for 512x512; nothing a request sends can raise the cap ([pixel cap](serve.md#the-pixel-cap)) |
| `--no-image-urls` | off: URLs are fetched | do not fetch http(s) image URLs; a request must send the image as base64 (or a `file://` path under `--media-path`). Set it on a server other machines can reach: a fetch has a 10 s timeout and a 20 MiB cap but no address filtering ([sources](serve.md#sources)) |
| `--media-path DIR` | unset | serve `file://` image paths relative to this existing directory; absolute paths, `..`, and symlinks that resolve outside it are refused. Unset, `file://` is refused. A directory that does not exist is a usage error |

Equivalent environment variables: `DRINKME_CTX`, `DRINKME_ROPE_SCALING`, `DRINKME_AUTH_TOKEN`,
`DRINKME_PREFIX_SLOTS`, `DRINKME_SPEC`, `DRINKME_ADVERTISED_CTX`,
`DRINKME_SERVED_MODEL_NAMES` (space- or comma-separated), `DRINKME_SLEEP_ON_IDLE_S`,
`DRINKME_IMAGE_MAX_PIXELS`, `DRINKME_IMAGE_URLS` (`0` for `--no-image-urls`),
`DRINKME_MEDIA_PATH`. An unparseable `DRINKME_IMAGE_MAX_PIXELS` or a
`DRINKME_MEDIA_PATH` that is not a directory warns at boot and falls back to
the default.

## `drinkme bench`

Bench measures BF16 checkpoints. MLX benchmarking is still in development;
see [bench status](bench.md).

| flag | what it does |
|---|---|
| `-o, --out FILE` | write the record here instead of `./measurements/<model>_<device>_<date>[_<compression profile>][_<n>].json`; with `--model` only (a menu run ignores it with a warning) |
| `--model MODEL` | override the menu with a menu model, by its name or its HF repo id. Menu models only: an off-menu repo is refused, although `serve`, `pack` and `check` accept one |
| `--detect-only` | hardware identity and the menu only; no torch, no probe |
| `--dry-run` | print JSON with each model, its sizes, its arms, stock-arm fit estimate, record shape, and filename; no torch import, weight download, bandwidth probe, or file write ([bench.md](bench.md#--dry-run)) |
| `--allow-unknown-device` | write a record even when the device name is only the pci-id fallback (`amdgpu 0x7480`), which is otherwise refused ([bench.md](bench.md#naming-a-discrete-amd-card)) |
| `--no-gemma` | skip the Gemma license check (a HEAD request to huggingface.co using your HF token, if set); omit Gemma from selection |
| `--runtime R` | `torch` runs the torch arms; `mlx` runs the MLX arms — mlx-lm bf16 as stock, the MLX engine's reference path as the twin, the fused path as compressed, each arm's min and drift beside the median (default: `mlx` on Darwin/arm64, `torch` elsewhere) |
| `--stock-loader L` | `stream` (default) or `from_pretrained`, torch runtime only: how the stock bf16 arm loads and what its fit check charges — streaming moves one tensor at a time from the shards (charge: weights + the largest tensor), `from_pretrained` is transformers' loader (charge: 2x on unified memory, measured). `--dry-run` shows the plan under either ([bench.md](bench.md)) |
| `--pack-dir DIR` | time the compressed arm using this pack through the serving loader; requires `--model` and the torch runtime. Before running any arm, validate the format version and checkpoint identity without importing torch |
| `--no-spec` | skip the [speculation pass](bench.md#speculation), which times decode with the checkpoint's MTP head proposing on the stock and compressed arms of a model with a head (torch runtime); the record says `raw.spec.skipped` = `--no-spec` |

## `drinkme pack`

| flag | what it does |
|---|---|
| `--model MODEL` | menu model name or HF repo (required) |
| `-o, --out DIR` | pack directory (default `~/.cache/drinkme/packs/...`) |
| `--replace` | re-pack over an existing pack directory |
| `--sip` | default compression profile (tiers 3,8); faster decoding |
| `--gulp` | the smaller compression profile (tiers 2,2,4,8): ~4% fewer bytes than sip, slower on bandwidth-bound devices; mutually exclusive with `--sip`. Without `-o`, writes to `<org>--<model>-gulp@<rev>`; serve it with `--pack-dir` |

`serve` packs automatically; `pack` runs that CPU step separately. Each tensor
must round-trip bit for bit before writing. Output is staged and published
transactionally, with one writer per destination. `--replace` retains the old
pack until the new one is complete. Do not replace a live server's pack.
A pack declaring any other format version is refused at load and must be
packed again. See [pack format](pack-format.md).

The pack is self-contained: `checkpoint/` holds the checkpoint's config,
tokenizer, generation defaults and every tensor the loaders read raw, and
serving it needs no checkpoint and no network. A vision model's pack carries
its [vision tower](pack-format.md#the-vision-tower) too.
See [the embedded checkpoint](pack-format.md#the-embedded-checkpoint).

## `drinkme check`

| flag | what it does |
|---|---|
| `--model MODEL` | menu model name or HF repo (required) |
| `--json` | machine-readable verdict |

Eligibility from the repo's metadata alone, no weight download: the same
gates `pack` and `serve` apply ([models.md](models.md)). Also reports THIS
MACHINE's radix encoder: the compiler path and version when the native
encoder is available, or — printed separately, on stderr, after the verdict —
that this machine can't pack it yet, with install lines, when the encoder is
missing or failed to build. That case is independent of the repo's own
eligibility (a repo can be eligible on a machine that cannot currently pack
it) and exits 4, not 0; see [exit codes](#exit-codes).

## `drinkme bootstrap`

| flag | what it does |
|---|---|
| `--dry-run` | print the host probe, the lane and the exact install and self-test that would run; install nothing, run nothing |
| `--detect-only` | the probe and the lane only (no torch, no ROCm needed); install nothing |
| `--gfx TARGET` | select an AMD gfx target (e.g. `gfx1201`) for `--dry-run` or `--detect-only` |

Runs dependency setup explicitly. It probes the host (on AMD: Linux, the gfx
target from KFD sysfs or the PCI table, the `amdgpu` module, and access to
`/dev/kfd` and `/dev/dri/renderD*`), selects an installation lane, and installs
its packages into the project environment. The automatic installation command is
`uv sync --no-default-groups --group <group>`, where the group is the
lane's dependency group, not the lane's name: lane `cuda` installs group
`cuda`, `official` group `rocm-official`, `therock` group `rocm` (gfx1151
only; other TheRock targets get the install line printed instead), and
`metal` group `metal`. A child interpreter then runs
seven self-test stages, printing each result.

`serve`, `pack`, `bench`, and `verify` also trigger setup when torch is missing
or detects no device. See [hardware setup](hardware.md). See
[exit codes](#exit-codes): 0 on success, 2 for `--gfx` used without `--dry-run`
or `--detect-only`, 4 for anything this machine's environment stops (an
unsupported host or target, a refused device, a failed sync, or a failed
self-test).

## `drinkme verify`

| flag | what it does |
|---|---|
| `--model MODEL` | resolve the cached pack for this model |
| `--pack-dir DIR` | check this pack directory instead |
| `--upstream` | also rebuild every source weight file from the pack and compare its SHA-256 with the one the Hugging Face Hub publishes for it at the pack's commit, and write the receipt ([upstream verification](pack-format.md#upstream-verification)) |

Checks each file against its recorded SHA-256 and the tensor manifest against
`meta.json`, for both the trunk and any MTP sub-pack. For a self-contained
pack it also checks every embedded file, and that the embedded config,
tokenizer and shard headers still digest to the pack's source identity. It prints the result
without loading a model or rewriting files. Every pack records its hashes when
written. Unsupported formats, mismatched hashes, and missing hashes produce the
same errors as the loader. See [pack verification](pack-format.md#verification).

With `--upstream`, after those checks pass, it prints one line per source
file (`MATCH`, `MISMATCH`, or `NOT COVERED` with the reason), the verdict,
and the receipt's path. It exits 0 only when every file matches, 3 for any
MISMATCH or NOT COVERED or for a pack with no Hub commit to compare with,
and 4 when the Hub cannot be asked (offline); the file-hash check has
passed in that case, and the message says so.

## `drinkme publish`

```
drinkme publish [FILE ...] [--handle H | --did D] [--plc URL] [--logout] [--no-browser]
```

| flag | what it does |
|---|---|
| `FILE ...` | the record(s) to publish (default: the newest file under `./measurements/`); a record without a resolved hub commit in `model.revision`, or with `environment.platform` `metal`, is refused by name and the others still go ([bench.md](bench.md#local-records)) |
| `--handle H` | your atproto handle; remembered in `~/.config/drinkme/publish.json` after the first successful publish |
| `--did D` | your DID instead of a handle |
| `--plc URL` | the PLC directory to resolve a did:plc through (default `https://plc.directory`); remembered |
| `--logout` | delete the stored session (tokens + DPoP key, `~/.config/drinkme/session.json`) and stop, unless a record is also given |
| `--no-browser` | do not try to open a browser; only print the authorization URL (headless machines: open it anywhere, the redirect must reach this machine's `127.0.0.1`) |

Publishes each validated measurement to your own PDS through loopback OAuth,
then reads it back for comparison; exit status is 0 only when every record
given went, 3 when at least one was refused (a bad revision, an incomplete
run, a lexicon failure — see [exit codes](#exit-codes)) after the others have
gone. See [publishing](bench.md#drinkme-publish)
for scopes and local state. `DRINKME_CONFIG_DIR` relocates configuration.
`DRINKME_PUBLISH_ALLOW_INSECURE=1` permits non-loopback plain HTTP for test PDS
or authorization servers and emits a warning.

## Environment variables

A variable that takes a duration ends in `_S` and is in seconds. An
operator-sized quota ends in `_GIB` and is in GiB (2^30 bytes); a request-size
cap ends in `_BYTES`. Records use decimal GB, never GiB.

Serving:

| variable | default | what it does |
|---|---|---|
| `DRINKME_MTP_DEPTH` | unset | `k` = pin the MTP draft depth (and load the head even where the fit check would skip it); unset = 4 when the checkpoint has a head. It is not an off switch: `--spec off` / `DRINKME_SPEC=off` is |
| `DRINKME_MTP_DIET` | on | serve the MTP head from its reduced sub-pack (~0.6 GiB); `0` serves the raw ~0.75 GiB head |
| `DRINKME_NGRAM_TOKENS` / `_MAX` / `_MIN` | 5 / 4 / 3 | n-gram proposer settings, named after vLLM's ([serve-speculation.md](serve-speculation.md)) |
| `DRINKME_PREFILL_CHUNK` | 4096 | tokens per prefill forward pass, which bounds how long one GPU kernel runs ([serve.md](serve.md#chunked-prefill)); `0` = the whole prompt in one pass |
| `DRINKME_SLOT_DIR` / `_DISK_GIB` / `_RAM_GIB` | `~/.cache/drinkme/slots` / 64 / 0 | the on-disk slot tier ([serve-prefix-slots.md](serve-prefix-slots.md)); `DRINKME_SLOT_DIR=off` disables it |
| `DRINKME_CTX_CHECKPOINTS` | 32 | context checkpoints per prefix slot; `--ctx-checkpoints` wins ([serve-prefix-slots.md](serve-prefix-slots.md#context-checkpoints)) |
| `DRINKME_MAX_TOKENS_CLAMP` | on | clamp `max_tokens` to the room left in the window; `0` = refuse with a 400 instead ([serve.md](serve.md)) |
| `DRINKME_SSE_KEEPALIVE_S` | 10 | `: keep-alive` SSE comments while a stream waits in queue or prefill; `0` disables |
| `DRINKME_IGNORE_GENERATION_CONFIG` | unset | `1` = plain OpenAI sampling defaults instead of the checkpoint's `generation_config.json` |
| `DRINKME_SLEEP_PIN` | off | stage sleep's host copies in pinned memory (faster wake; on unified memory it reduces memory available to other models) |
| `DRINKME_DEVICE` | detect | force `cuda` or `cpu`; `cpu` says a CPU serve is deliberate and silences the no-GPU warning |
| `DRINKME_TOOLS_UNTESTED` | unset | `1` = let `tools` requests reach every row that has a parser, not only the tested tier; announced at boot. Used for hardware validation before setting a row’s `tested` flag ([serve-tool-formats.md](serve-tool-formats.md)) |
| `DRINKME_MTP_SUFFIX_SDPA` | on | `0` = keep explicit masks for MTP suffix attention, for A/B comparison ([serve-kernels.md](serve-kernels.md#mtp-suffix-attention)) |
| `DRINKME_SPEC_AUDIT` | on | `0` = silence the speculative-decoding acceptance audit printed every 50 cycles, for either proposer |
| `DRINKME_MTP_BAIL_WINDOW` | 64 | the adaptive bail's window, in drafted positions (a rejected cycle's unreached positions count as rejected); an integer ≥ 1, anything else is the default with one stderr line ([serve-speculation.md](serve-speculation.md#adaptive-bail-and-re-arm)) |
| `DRINKME_MTP_BAIL_FLOOR` | 0.08 | the acceptance rate over a full window below which the request bails to serial decoding; 0..1, `0` = never bail; anything else is the default with one stderr line |
| `DRINKME_MTP_REARM` | 64 | tokens a request decodes serially after the adaptive bail trips before speculation re-arms on a fresh window; doubles on a re-trip (cap 512); `0` = never re-arm, the rest of the request stays serial ([serve-speculation.md](serve-speculation.md#adaptive-bail-and-re-arm)) |
| `DRINKME_MAX_BODY_BYTES` | 128 MiB | cap on a request body (`Content-Length` above it is a 413) |
| `DRINKME_READ_TIMEOUT_S` | 30 | seconds a client may go silent while its request is being read before the connection is closed; never applies to the response side |
| `DRINKME_DELTANET_KERNEL` | detect | DeltaNet recurrence implementation: `torch` forces the reference; `fla` requires the fused kernel. Unset selects FLA on an accelerator if the startup probe passes, otherwise torch. The boot log reports the choice. Rounding differences can change near-tie token choices ([details](serve-kernels.md#deltanet-recurrence-kernel)) |
| `DRINKME_DELTANET_CONV` | detect | DeltaNet prefill convolution: `torch` retains upstream dispatch; `fla` requires the Triton kernel. Unset selects FLA for large ROCm windows if the startup probe passes. Small MTP windows retain their existing path ([details](serve-kernels.md#deltanet-prefill-convolution)) |
| `DRINKME_NARROW_GEMV` | on | `0` uses BF16 `F.linear` instead of Triton GEMV for raw Linears below the codec's row threshold at M ≤ 8. The boot log reports `[drinkme] narrow gemv:`. Intended for A/B measurements; see [narrow GEMV](serve-kernels.md#narrow-gemv) |
| `DRINKME_STOCK_GEMV` | by target | the one-row call of raw BF16 Linears at or above the codec's row threshold (a tied `lm_head`; every Linear under `--stock`): `linear` = `F.linear`, `mv` = `torch.mv` with hipBLASLt preferred for that call, `triton` = the twin arm's Triton GEMV over the raw weight. Unset selects `triton` on gfx1151 and gfx1102 and `linear` elsewhere; the boot log reports `[drinkme] stock gemv:`. See [stock GEMV](serve-kernels.md#stock-gemv) |
| `DRINKME_RAW_GEMV` | by target | the one-row call of the raw BF16 Linears at or above the codec's row threshold that a pack leaves raw (a tied `lm_head`) in `drinkme serve` over a pack and the bench's compressed and twin arms: `twin` = the twin arm's Triton GEMV over the raw weight, `stock` = the stock GEMV. Unset selects `twin` on gfx1151 and `stock` elsewhere; the boot log reports `[drinkme] raw gemv:`. See [raw GEMV](serve-kernels.md#raw-gemv) |
| `DRINKME_CUDA_GRAPHS` | on (CUDA) | `0` = decode eager in `drinkme serve` and `drinkme bench`. Unset, the decode and speculative verify steps replay CUDA graphs on CUDA for the model families `serving/cudagraph.py` has verified; ROCm and Metal never do. The boot log reports `[drinkme] decode step:` and what the graphs hold per cache; the fit checks charge that memory unless this is `0` ([CUDA graphs](serve-kernels.md#cuda-graphs)) |
| `DRINKME_CUDNN_SDPA` | off | `1` = let torch choose cuDNN attention on CUDA, for A/B comparison only: every fresh request thread pays cuDNN's plan builds again, the MTP head's included. Unset, `drinkme serve` and every `drinkme bench` arm run attention without cuDNN SDPA on CUDA, for decode and prefill; ROCm, Metal and the CPU keep torch's choice ([attention backend on CUDA](serve-kernels.md#attention-backend-on-cuda)) |
| `DRINKME_VISION` | on | `0` = build no vision tower: its resident share goes to the KV cache, its pack tensors are never read, and images are refused naming the switch ([image input](serve.md#when-images-are-refused)) |
| `DRINKME_VISION_BOUNDED` | on | `0` = run the vision tower's attention as one call per image instead of dispatches of at most 33,554,432 query-key pairs; a reference for the bit-exactness gates, head-padded on ROCm either way ([ROCm](rocm.md#the-vision-tower-on-gfx1151)) |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | unset | ROCm parts AOTriton classifies experimental (gfx1151, gfx1102 measured), set before the process starts ([ROCm](rocm.md#attention-setup)) |

Install and cache:

| variable | default | what it does |
|---|---|---|
| `DRINKME_HOME` | `~/.cache/drinkme` | root of the pack cache (slots have their own `DRINKME_SLOT_DIR`) |
| `DRINKME_PACK_HUB` | unset | where `serve` looks for published packs: a Hub namespace (`<ns>/<model>-drinkme`) or a whole repo id; `off` disables the lookup. Unset, `hub_packs.PACK_HUB_NAMESPACE` decides: `drinkme-packs` ([published packs](serve.md#published-packs)) |
| `DRINKME_NO_HUB_PACK` | unset | `1` = `serve --no-hub-pack` |
| `DRINKME_NO_AUTO_DEPS` | unset | `1` = never install accelerator packages on first run, and no host probe or self-test; you manage torch yourself ([hardware.md](hardware.md)) |
| `DRINKME_RUNTIME` | host | `torch` or `mlx`; `--runtime` wins |
| `DRINKME_MLX_PATH` | by Metal | `fused` or `reference`: the MLX engine's compute path (the bench's twin arm forces `reference`); reported as `drinkme.computePath` on `/v1/models` |
| `DRINKME_MLX_KEEP_CACHE` | unset | `1` = the MLX engine keeps MLX's buffer cache between requests instead of returning it after each prefill and request ([Metal](metal.md#engine-and-loading)) |

BF16 codec settings (default compression profile: sip; default schedule: device table):

| variable | what it does |
|---|---|
| `DRINKME_COMPRESSION_PROFILE` | the compression profile `pack` uses when neither `--sip` nor `--gulp` is given (`sip` unset); a flag wins over it |
| `DRINKME_RADIX_SCHEDULE` | `table` (default: this box class's rows) / `table=gfx1151` / `table=gfx1102` / `table=cuda` (another box class's) / `spike` / an explicit `tiles=…,warps=…` launch config / a per-class JSON object or `@file` for the decode kernels ([pack-format.md](pack-format.md#the-launch-schedule)) — an A/B instrument, never set in production |
| `DRINKME_RADIX_DECODER` | unset (default: the lean sip decoder on CUDA for sip's whole-block rows, a lean gulp decoder for gulp's whole-block rows (`_decode_gulp_lean` on ROCm, `_decode_gulp_lean_cuda` on CUDA), the scheduled decoder otherwise) / `scheduled` / `lean` for the radix kernels' block decoder; the decoded bits are the same ([pack-format.md](pack-format.md#the-launch-schedule)) — an A/B instrument, never set in production |
| `DRINKME_TWIN_SCHEDULE` | the bench's twin arm's launch rows: `twin` (default: the box class's twin rows where it has them, [pack-format.md](pack-format.md#the-launch-schedule)) or `served` (the compressed tensor's row, the order-matched-at-the-served-row twin) — an A/B instrument |
| `DRINKME_RADIX_ENCODER` | `native` (the default; requires a C++ compiler on PATH — [install lines](../README.md)) or `numpy` (opt-in only, much slower — hours for an 8B model, and prints a line saying so). With no compiler (or one that fails to build the encoder) and no explicit `numpy`, packing refuses (exit 4) rather than silently falling back; `drinkme check` reports the same condition and exits 4 too. Both encoders write the same bytes |

Instrument-only variables (`DRINKME_REFERENCE`, `DRINKME_HOTLOOP_OFF`,
`DRINKME_DETOK_VERIFY`, `DRINKME_NO_WARMUP`, `DRINKME_FAKE_ENGINE`,
`DRINKME_AB_PACK`, `DRINKME_PREFILL_DENSE_MIN`, `DRINKME_MTP_DRAFT_VOCAB`, ...)
are developer-only controls, described in [../bench/README.md](../bench/README.md).
`DRINKME_MTP_DRAFT_VOCAB` has MTP drafts argmax over a subset of the
vocabulary; it acts only where the head's vocabulary projection is an
uncompressed Linear (`--stock`, or a tied `lm_head`). A packed `lm_head` has no
row-subset kernel, so there it keeps the full vocabulary and says so on stderr.
