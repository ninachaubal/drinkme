# Architecture and vocabulary

This page defines project terminology, traces serving, packing, and
benchmarking, and maps those operations to their modules. It also identifies
public flags and fields that use different names from the implementation.

## Vocabulary

| Concept | Definition | Where the name lives |
|---|---|---|
| **Source checkpoint** | The Hugging Face snapshot — a repo at one resolved revision, or a local directory — that a pack is cut from. It supplies the packed tensors at pack time, and the raw remainder, config, and tokenizer too, all copied into the pack's `checkpoint/`; needed only to write a pack (`drinkme pack`) and by `drinkme bench`'s stock and twin comparison arms, never to serve or verify one. Four identities that are not interchangeable: the repo name; the requested revision (`meta.json` `hfRepo`/`revision`, the caller's coordinates); the resolved revision (`source.revision`, the snapshot commit); and the structural digest (`source.digest`, over safetensors headers, config, tokenizer files and image processor configs — never tensor payloads). | [`codec/identity.py`](../src/drinkme/codec/identity.py), [`serving/checkpoint.py`](../src/drinkme/serving/checkpoint.py) (`snapshot_dir`, `resolve_pack_source`) |
| **Pack** | A directory: `meta.json` plus one `.npz` per eligible Linear, an `mtp/` sub-pack when the checkpoint has an MTP head, and `checkpoint/`, the **embedded checkpoint**. Every pack `drinkme pack` writes is **self-contained**: `checkpoint/` holds the raw remainder, config, tokenizer and generation defaults, so it loads with no source checkpoint and no network. | [`codec/pack.py`](../src/drinkme/codec/pack.py) writes; [`packs.py`](../src/drinkme/packs.py) finds |
| **Pack format** | The versioned container contract: `meta.json` plus the tensor files, at one **pack format version** (below). | `codec/pack.py`, [pack format](pack-format.md) |
| **Tensor codec** (`codec`) | What one tensor inside a pack is: `codec: "radix"` (the BF16 exponent-tier code) or `codec: "raw"` (a raw fallback, below). In code it is `tensor_codec` on `CompressedLinear` and the first key of the registry table. "Raw" has four senses — the raw fallback, the raw remainder (both below), the torch twin's uncompressed weights, and a record's `raw` object (Record, below) — say which. | [`codec/registry.py`](../src/drinkme/codec/registry.py), [`codec/swap.py`](../src/drinkme/codec/swap.py) |
| **Raw fallback** | A packed tensor stored as its BF16 bits verbatim (`codec: "raw"`) because radix would have expanded it; `RawLinear` serves it through plain `F.linear`. *Where:* `rawFallbackTensorCount` / `rawFallbackBytes` in `meta.json`. | [`codec/radix_pack.py`](../src/drinkme/codec/radix_pack.py), [pack format](pack-format.md) |
| **Raw remainder** | The source tensors the codec does not pack — embeddings, norms, biases, projections too small to pack — stored byte for byte in the embedded checkpoint's `remainder.safetensors`. Not counted in bits per weight. *Where:* `rawTensorCount` / `rawResidentBytes` and `embedded` in `meta.json`. | [`serving/engines.py`](../src/drinkme/serving/engines.py) (`stream_checkpoint`), [pack format](pack-format.md#the-embedded-checkpoint) |
| **Compression profile** | The tier widths every coded tensor in a pack is encoded at: `sip` (3, 8; the default) or `gulp` (2, 2, 4, 8). Fixed at pack time. Always spelled with its adjective in prose and on the CLI. *Where:* `--sip`/`--gulp` on `drinkme pack` (`DRINKME_COMPRESSION_PROFILE` in the env); `profile` in `meta.json` (the pack's own file); `compression.profile` in the record; `compressionProfile` on `/v1/models`; `compression_profile` in code. (`balanced` exists in `radix.PROFILES` behind `DRINKME_COMPRESSION_PROFILE`; it is not public.) | [`codec/radix_pack.py`](../src/drinkme/codec/radix_pack.py) (`resolve_compression_profile`), `cli.py` |
| **Generation profile** | A named sampling/chat-template overlay applied per request on the one engine. Bare "profile" on the CLI and in prose means this one. *Where:* `drinkme serve --profile NAME=KEY:VAL,...` or a `profiles.json` beside the pack; `<id>:NAME` on `/v1/models` with `sampling.profile: NAME`; `generation_profiles` / `generation_profile_flags` in code. | [`serving/generation_profiles.py`](../src/drinkme/serving/generation_profiles.py) |
| **Runtime** | Which implementation serves and benches: `torch` (`serving/engines.py`; CUDA, ROCm, CPU) or `mlx` (`serving/engine_mlx.py`; Apple silicon). Resolved once, before anything is loaded or ranked: the flag, then the env, then the host (Darwin/arm64 is `mlx`). Say "runtime" for `torch` / `mlx`; "engine" is the `Engine` object a runtime builds (or an external engine such as llama.cpp), and `fused` / `reference` are compute paths. *Where:* `--runtime torch\|mlx` on `serve` and `bench`; `DRINKME_RUNTIME`; `runtime=` in `serve.run`, `build_engine`, `bench`; `drinkme.runtime` on `/v1/models`; the second segment of the record's `environment.engine`. | [`serve.py`](../src/drinkme/serve.py) (`resolve_runtime`), [`runtimes.py`](../src/drinkme/runtimes.py) |
| **Host** | The machine drinkme runs on, as the runtime's last fallback: with no `--runtime` and no `DRINKME_RUNTIME`, Darwin/arm64 is `mlx` and everything else `torch`. `--host` (the listen address) and host RAM are the ordinary word. | [`serve.py`](../src/drinkme/serve.py) (`resolve_runtime`) |
| **Platform** | The compute platform that executed the decode: `cuda`, `rocm`, or `metal`, read off the runtime that ran the arms. torch reports ROCm through `torch.cuda`, so a boot log's `device: cuda` is a torch device string, not this. *Where:* the record's `environment.platform` (a closed enum in the lexicon; `bench._engine_env`); the site's comparison identity. Not on the CLI, not in the env. | [`bench.py`](../src/drinkme/bench.py), [lexicon](../lexicons/README.md) |
| **Box class** | The launch table's key: `gfx1151`, `gfx1102`, or `cuda`. On ROCm, `box_class()` reads the device's `gcnArchName`; an AMD target without rows of its own uses `gfx1151`'s. *Where:* code and `DRINKME_RADIX_SCHEDULE=table=<box class>` only. | [`codec/radix_schedule.py`](../src/drinkme/codec/radix_schedule.py) |
| **Device class** | The record's normalized device name — how owners self-identify the machine (`Apple M3 Max`, `NVIDIA GeForce RTX 4090`, or on a Linux APU the CPU brand string, `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`) — which the site groups and labels by. *Where:* the record's `environment.deviceClass` (`detect.Hardware.device_class` → `as_record_env`), the `<device>` in a measurement's filename. `/health`'s and `/v1/models`' `device` (`cuda:0`, `cpu`) is torch's device string, a different thing. | [`detect.py`](../src/drinkme/detect.py), [lexicon](../lexicons/README.md) |
| **Pack format version** | The backward-compatibility check on a pack; today it only ever says 1 and other versions are refused (re-pack; there is no migration). *Where:* `formatVersion` in `meta.json`; `FORMAT_VERSION` / `SUPPORTED_FORMAT_VERSIONS` and `format_version` (every loaded tensor dict, `PackWriter`) in code. | `codec/pack.py` (`_check_format_version`), [pack format](pack-format.md) |
| **Bits per weight** | Two figures over the packed population, by the lexicon's names. `bitsPerWeight` is weight-weighted: total encoded payload bits over the population's weights (`weighted_bpw`). `meanTensorBitsPerWeight` is the unweighted mean of per-tensor bpw. *Where:* the record's `compression.bitsPerWeight` / `compression.meanTensorBitsPerWeight` (decimal strings); `drinkme.bitsPerWeight` / `drinkme.meanTensorBitsPerWeight` on `/v1/models` (numbers); in `meta.json` the same two are `weightedBpw` / `meanBpw` (the pack's own file, unchanged). | `codec/pack.py`, [pack format](pack-format.md#metajson) |
| **Engine string** vs `Engine` | Two different things. The **engine string** is the record's `environment.engine`, naming what ran the arms — `drinkme <semver> / <runtime> <version> / …`, with the kernel routes as segments on torch ([bench](bench.md#the-engine-string)). `Engine` is the serving Protocol every runtime's engine implements (`HFEngine`, `MLXEngine`, `FakeEngine`), the seam bench's A/B and the HTTP layer share. | [`bench.py`](../src/drinkme/bench.py) (`engine_string`, `parse_engine`), [`serving/engine.py`](../src/drinkme/serving/engine.py) |
| **SDPA backend** | Which of torch's scaled-dot-product-attention kernels (`flash`, `mem_efficient`, `cudnn`, `math`) the machine actually runs — torch's own word, kept with its adjective everywhere it appears. *Where:* `sdpa.py` (`probe_backends`, the self-test), the boot log, the [ROCm](rocm.md#attention-setup) page. | [`sdpa.py`](../src/drinkme/sdpa.py) |
| **Kernel** (`kernel`) / compute path | The registry's `kernel` key: `triton` (torch on CUDA and ROCm), `cpu` (the CPU route, `F.linear` over the decoded weight), or `metal` (MLX). The registry's `op` — `dense`, `gemv`, `mc`, `row_subset` — is the M dispatch within a kernel. **Compute path** is the MLX engine's `fused` / `reference` switch (`RadixLinear.path`, `DRINKME_MLX_PATH`, `drinkme.computePath`); both run the `metal` kernel. | `codec/registry.py`, [`serving/engine_mlx.py`](../src/drinkme/serving/engine_mlx.py) |
| **Kernel route** | The five kernel choices `serving/kernel_route.py` makes once per model load, the same for `serve`, `serve --stock` and every bench arm: the DeltaNet recurrence, the DeltaNet prefill convolution, the narrow GEMV, the stock GEMV and the raw GEMV. Each has an env var, a boot line, an engine-string segment and a raw routing key ([names by surface](serve-kernels.md#kernel-routes-by-name)). | [`serving/kernel_route.py`](../src/drinkme/serving/kernel_route.py), [serving kernels](serve-kernels.md) |
| **Installation lane** | Which source of accelerator packages the bootstrap installs and self-tests: `cuda`, `therock` (TheRock wheels), `official` (PyTorch's ROCm wheel), or `metal` (mlx, with torch's CPU wheel for packing). The lane names a source; the uv dependency group it installs is named in `LANES`, and for two lanes it is not the lane's name: `cuda` → `cuda`, `therock` → `rocm` (gfx1151 only; other TheRock targets get an install line, not a group), `official` → `rocm-official`, `metal` → `metal`. The self-test checks runtime and kernel operation on that machine. Model support and performance need separate tests. | [`bootstrap.py`](../src/drinkme/bootstrap.py) (`LANES`, `choose_lane`), [hardware](hardware.md), [ROCm](rocm.md#install-lanes) |
| **Menu** | The models drinkme knows by name: `suggest.MODELS`, each with an HF repo and a pinned revision. A menu entry carries no sizes: a pick reads a model's BF16 size from its checkpoint's safetensors metadata, and its compressed size from a pack on the machine, or estimates it as the BF16 size times the profile's ratio (`suggest.sizes`). `--model` takes a menu name or any HF repo; a bare `serve` or `bench` ranks the menu against the machine's memory, and only models marked `auto_eligible` are candidates; the rest run only when asked for by name. "Suite" means the CPU test suite, not this. | [`suggest.py`](../src/drinkme/suggest.py), [models](models.md) |
| **Benchmark arm** | One of the three representations `drinkme bench` loads in isolation: `stock`, `twin`, `compressed`. What each arm is depends on the runtime. **Stock** is the runtime's own BF16 path over the unmodified checkpoint: drinkme's own on torch (with the box's [stock GEMV](serve-kernels.md#stock-gemv) for one-row calls), mlx-lm's on MLX. **Twin** on torch is `RadixTwinLinear` — raw BF16 through the compressed arm's own kernels with the decode replaced by a load of the raw weight, so at any one launch schedule it is order-matched to the codec's kernel; it runs at the box's twin rows where they are measured ([launch schedule](pack-format.md#the-launch-schedule)), the compressed tensor's row elsewhere; on MLX it is `RadixTwinLinear`, the reference compute path's matmul over the pack decoded once at load (bitwise the CPU oracle), a correctness check, not a bandwidth control. The torch twin publishes `twin_decode_tok_s` and `twin_weights_gb`, a diagnostic the results site does not plot (it divides by stock); the MLX twin publishes no metric. **Compressed** is the pack on drinkme's runtime. `/v1/models` also reports `drinkme.arm`: `compressed`, or `stock` under `--stock` on either runtime. Bench times its own greedy loop, never the server. | [`arms.py`](../src/drinkme/arms.py), [`arms_mlx.py`](../src/drinkme/arms_mlx.py), [`codec/swap.py`](../src/drinkme/codec/swap.py), [bench](bench.md) |
| **Prefix cache / prefix slot** | The **prefix cache** is the feature: a later prompt that extends a cached conversation, or shares a prefix with it, reuses its state instead of re-prefilling; a **context checkpoint** is a snapshot of the state a rewind cannot reach, which serves the second case. A **prefix slot** is its unit: one whole cache state — a `StaticCache` plus the token ids whose KV it holds. `--prefix-slots N` keeps N slots warm, and `0` turns the cache off (`/health` reports `slots: 0`); an evicted slot goes to disk (`DRINKME_SLOT_DIR`). Slots are not batching: the HTTP layer holds one generation lock and a second request waits. | [`serving/engines.py`](../src/drinkme/serving/engines.py), [`serving/slotstore.py`](../src/drinkme/serving/slotstore.py), [`serving/ctx_checkpoints.py`](../src/drinkme/serving/ctx_checkpoints.py), [prefix cache](serve-prefix-slots.md) |
| **Speculation** | Drafting several tokens and verifying them in one forward. A **proposer** drafts: the MTP head or n-gram lookup, picked by `--spec off\|mtp\|ngram\|ngram+mtp\|auto` (`DRINKME_SPEC`; `off` is the only off switch). Drafts are **proposed**; those the verify pass keeps are **accepted** (`drinkme_spec_proposed_tokens_total` / `_accepted_`, both proposers). The adaptive **bail** hands a request to serial decode when the MTP head's rolling acceptance falls below the floor, and **re-arm** resumes drafting after a serial stretch. The `DRINKME_MTP_*` variables and the `drinkme_mtp_*` metrics are MTP-only. | [`serving/mtp.py`](../src/drinkme/serving/mtp.py), [`serving/ngram.py`](../src/drinkme/serving/ngram.py), [speculation](serve-speculation.md) |
| **Thinking** | drinkme's word for a model's reasoning channel. `reasoning*` appears only in OpenAI wire fields (`reasoning_content`, `reasoning`, `reasoning_effort`); Anthropic's is `thinking`. `/v1/models`' `capabilities.thinking` is `open`, `closed`, `always` or `none` ([defined there](serve.md#get-v1models)), and `thinkingSwitch` names the request field that turns it off. | [`serving/capability.py`](../src/drinkme/serving/capability.py), [`serving/think.py`](../src/drinkme/serving/think.py) |
| **Vision tower, image run** | The **vision tower** is a model's image encoder: the ViT and whatever maps its output into the text model's embeddings (gemma-4's `embed_vision`; Muse-Glimmer's adapter and projection). It is built beside the text model in the served tree (`arms._VISION_TOWERS`, every subtree at the checkpoint's own path), packed with it, and left out under `DRINKME_VISION=0`. An image's **run** is the placeholder tokens its expansion puts in the prompt, markers excluded; the tower's output replaces their embeddings. A run is **causal** or **bidirectional** in the text model (`vision.Tower.bidirectional`), and that one fact decides how it may be prefilled. *Where:* `capabilities.vision` / `imageInput` on `/v1/models`; the `vision` block in `meta.json`; `[drinkme] image input:` at boot. | [`serving/vision.py`](../src/drinkme/serving/vision.py), [`serving/image_prompt.py`](../src/drinkme/serving/image_prompt.py), [`arms.py`](../src/drinkme/arms.py), [image input](serve.md#image-input) |
| **Sleep level** | `POST /sleep?level=1` parks the weights in host RAM; `level=2` also frees the host copies and reloads them on wake. `/health` reports the `level`. The torch runtime only; MLX returns 501. | [`serving/sleep.py`](../src/drinkme/serving/sleep.py), [sleep/wake](serve-sleep.md) |
| **Pack hashes** | The per-file SHA-256s and `manifestSha256` (over name → file → hash → shape/dtype/codec descriptors, plus `source`) stored in `meta.json` and rechecked at every load and by `drinkme verify`. Matching hashes establish consistency with the pack's own metadata. They do not authenticate the producer, repeat the source comparison, or establish that streams decode correctly. `radix.validate` bounds descriptors; validating data-dependent streams requires decoding. | `codec/pack.py` (`verify_hashes`, `verify_pack`), [verification](pack-format.md#verification) |
| **Check words** | **Self-test**: bootstrap's seven stages (`bootstrap_selftest.RUNGS`) in a child interpreter after an install. **Gate**: a check that refuses — it prints a verdict and fails the run or the write on a violation: the bit-exact gate before bench writes a record, the codec gates (`bench/*_bitpin.py`: a kernel's output pinned bit for bit against the CPU reference), the lexicon check before publish. **Hardware acceptance**: the run on real hardware a change type requires ([checks](checks.md)); its gates are the instruments it runs. **Smoke**: a short end-to-end run of a real model on a real server (`bench/*_smoke*`); it prints a verdict like a gate but covers a whole path. Gemma's Hugging Face license acceptance is the **license check**, not a gate. | [checks](checks.md), [`bench/`](../bench/README.md) |
| **Record** | The output of one benchmark run: the JSON object `drinkme bench` writes to `measurements/<model>_<device>_<date>….json` and `drinkme publish` sends. "Measurement" is the lexicon's and the collection's name for it (`wtf.petrichor.drinkme.measurement`); the site calls a plotted record a point. Its `raw` object is bench's diagnostics bag — declared `unknown` in the lexicon, published but never aggregated — the fourth sense of "raw". | [`bench.py`](../src/drinkme/bench.py), [lexicon](../lexicons/README.md), [bench](bench.md) |
| **Bandwidth wall** | `read_gb_s` ÷ the bytes one decode step reads: the decode rate, in tok/s, if every token read the model's Linears once, plus one embedding row, at the measured read bandwidth. The vision tower and the embedding rows a step does not look up count toward memory (`<arm>_weights_gb`), not toward the wall; bench records the per-arm denominator as `<arm>_decode_read_gb`. Achieved decode sits below it (kernel efficiency, launches, other work); speculation can exceed it by emitting several tokens per weight read. "Bandwidth ceiling" is the same thing; say wall. | [bench](bench.md#bandwidth-interpretation), [method](method.md) |
| **Ratio point / fit point** | The two models a menu run of `drinkme bench` picks for a machine. At the **ratio point** stock and compressed both fit, so the record carries compressed ÷ stock. At the **fit point** a larger model fits only compressed: the stock arm is skipped (`stock.outcome: skipped_predicted_nonfit`) and the record shows that it loads and runs. | [`suggest.py`](../src/drinkme/suggest.py), [bench](bench.md#memory-fit-check-and-fit-point-records) |
| **Fit check, charge, headroom, budget** | The **fit check** predicts, before loading, whether an arm fits. It **charges** the arm's resident bytes plus its loader's transient (the largest tensor when streaming; 2× the model for `from_pretrained` on unified memory), times the **headroom** (`suggest.RATIO_HEADROOM` 1.15 for stock and twin, `FIT_HEADROOM` 1.10 for compressed), against the **budget**: the detector's usable memory, or physical memory on unified machines (the record's `stock.budgetBytes` when the stock arm is skipped). `--dry-run` prints each arm's `…ChargeGB`. It is not `drinkme check`. | [`fit.py`](../src/drinkme/fit.py), [`bench.py`](../src/drinkme/bench.py) |
| **Check** | `drinkme check`: a repo's eligibility verdict from its metadata (config, safetensors headers), with no weight download; `serve` runs it before packing a model it has not seen. It is not the fit check, and not [docs/checks.md](checks.md). | [`check.py`](../src/drinkme/check.py), [models](models.md) |
| **Trunk, MTP head, sub-pack, diet** | The **trunk** is the model's main stack; the **MTP head** is the checkpoint's multi-token-prediction layer that drafts tokens for speculation. A pack stores the trunk's tensors at its top level and the head's in the `mtp/` **sub-pack**. The **diet** serves the head from that sub-pack, compressed (on by default; `DRINKME_MTP_DIET=0` serves the raw BF16 head). | [`serving/mtp.py`](../src/drinkme/serving/mtp.py), [pack format](pack-format.md), [speculation](serve-speculation.md) |
| **Dialect** | One of the three HTTP APIs one engine serves: OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses (`/v1/responses`), Anthropic Messages (`/v1/messages`). A model's tool-call wire format is its **tool format** (`toolFormat`), not a dialect. | [`serving/http.py`](../src/drinkme/serving/http.py), [serve](serve.md) |
| **Generation lock** | The one lock the HTTP layer holds while a request generates; a second request waits for it. `/health`, `/metrics`, `/v1/models` and `/tokenizer_info` do not take it. | [`serving/http.py`](../src/drinkme/serving/http.py) |
| **Launch schedule / launch table** | The **launch schedule** is the (blocks per program, warps) each radix tensor's M=1 GEMV and mc kernels run at, chosen at load (`radix_schedule.select`) and never stored in a pack. The **launch table** is its measured rows, keyed by box class, compression profile family and shape class. `DRINKME_RADIX_SCHEDULE` overrides it for A/B runs. | [`codec/radix_schedule.py`](../src/drinkme/codec/radix_schedule.py), [pack format](pack-format.md#the-launch-schedule) |
| **Near-tie** | Two top logits close enough that a different accumulation order can flip the greedy pick. It is why two correct greedy runs over the same weights (say, compressed and `--stock`, or speculation on and off) can produce different transcripts. | [method](method.md#numerical-behavior) |

## Flows

**Serving** — `drinkme serve [--model M] [--runtime torch|mlx] [--pack-dir D] [--stock]`:

```text
cli.main
├─ bootstrap.ensure_accelerator   torch (or mlx) present and launching? else choose the lane,
│                                 install it, self-test in a child interpreter
├─ no --model: packs.*            hardware_budget → local_packs + menu
│                                 → filter_for_runtime → rank_fits → confirm
├─ cli.resolve_model              menu name or HF repo → (repo, pinned revision)
└─ serve.run
   ├─ runtimes.refusal_for_repo   check model-family support before downloading weights or packing
   ├─ serve.ensure_pack           cache hit → check_pack_dir (format, then identity)
   │                              miss → check → codec.pack.pack_model (the pack flow below)
   ├─ serve.build_engine          resolve_runtime
   │     torch: resolve_device → engines.load_compressed | engines.load_stock   → HFEngine
   │     mlx:   engine_mlx.load_compressed_mlx | load_stock_mlx                 → MLXEngine
   └─ serving.http.start_server   one Engine, one generation lock, three dialects:
                                  /v1/chat/completions (http.py), /v1/responses (responses.py),
                                  /v1/messages (messages.py)
```

**An image** — from a request part to the forwards (torch runtime):

```text
the dialect (http.py, messages.py, responses.py), outside the generation lock
├─ vision.check_image_count       more than 32 → too_many_images, before anything decodes
├─ vision.parse_image_urls        data URL inline; http(s) fetched (one turn's concurrently);
│  | parse_base64                 file:// under --media-path → EncodedImage (format from the magic bytes)
├─ Vision.prepare                 decode (bomb guard, first frame, EXIF, RGB) → the architecture's
│                                 preprocessor, looked up by model_type → PreparedImage
│                                 (pixel_values, grid, tokens, content digest)
└─ GenerationRequest.images       template order; the turn keeps {"type": "image"} parts
HFEngine.generate
├─ ImagePrompt                    each placeholder expanded to its run (with gemma-4's and
│                                 Muse-Glimmer's markers); key_ids: each run → its prefix_key;
│                                 Qwen3.5's M-RoPE positions (mrope.rope_positions)
├─ pick_slot, cold tier           on key_ids; an image inside the reused prefix never runs the tower;
│                                 a context checkpoint at each media end (ImagePrompt.media_ends)
├─ prefill.spans(whole=…)         a causal run longer than the chunk is cut like text;
│                                 a bidirectional run stays in one span
├─ prefill.run(image=…)           per span: inputs_embeds (embed_tokens, the tower's rows over the
│                                 run), position_ids (Qwen3.5), the attention mask (gemma-4). The
│                                 tower runs once per distinct image, its attention through
│                                 vision.bounded_attention, and the pixels are released after;
│                                 an image it read before comes from tower_cache.TowerCache
└─ decode, verify, MTP head       position p + delta on Qwen3.5 (mrope.step_positions)
```

**Packing** — `codec.pack.pack_model` (CPU only; `drinkme pack`, or `serve` on a cache miss):

```text
source checkpoint  (serving.checkpoint.snapshot_dir: the HF cache snapshot or a local dir)
├─ refuse_checkpoint            reject unsupported dtypes (including FP8 / quantized)
│                                 before importing torch
├─ arms.skeleton (meta device)  the served tree, vision tower included → swap.eligible_linears
│     eligible = nn.Linear, bf16, C % 4 == 0, min(R, C) ≥ 1024, and not weight-tied to an Embedding
├─ eligible Linears             → radix_pack.pack_weight_radix at the compression profile (raw fallback if
│                                 radix would expand it) → decode == source, bit for bit
├─ everything else              = the raw remainder (embeddings, norms, biases, small projections)
│                               → write_embedded: copied byte for byte, with config, tokenizer
│                                 and generation defaults, into checkpoint/; digest rechecked
└─ PackWriter                   staged, hashed, renamed into place   = the pack
                                (meta.json's `vision` block when the tree has a tower)

At load, resolve_pack_source hands the loaders the pack's checkpoint/ (identity rechecked
from the embedded files; never the source checkpoint, never the network), and the raw
remainder streams through engines.stream_checkpoint / engine_mlx._stream_checkpoint.
```

**Measuring** — `drinkme bench [--model M] [--runtime torch|mlx] [--pack-dir D]`, then `drinkme publish`:

```text
bench.run
├─ detect.detect → suggest (ratio point, fit point) → confirm → probe.measure_bandwidth
├─ arms.run_arms (torch: stock → compressed → twin) | arms_mlx.run_arms (mlx: stock → twin → compressed), a fresh load each
│     torch compressed: re-pack in memory or load --pack-dir; MLX compressed: load a pack
├─ bench.write_record            measurements/<model>_<device>_<date>[_<compression profile>][_<n>].json (the lexicon's shape)
└─ drinkme publish [FILE ...]    a separate, opt-in verb: validate (a record with no resolved
                                 hub commit is refused) → resolve identity → loopback OAuth →
                                 createRecord on your PDS → getRecord and compare, per record
```

## Ownership boundaries

- **`serving/engine.py` vs `engines.py` vs `engine_mlx.py`.** `engine.py` is the
  shared interface: `SampleParams`, `GenResult`, the `Engine` protocol,
  the sampling validator, and `FakeEngine` (`DRINKME_FAKE_ENGINE=1`, canned generation for
  protocol tests). `engines.py` is the torch implementation (`HFEngine`, both
  arms, prefix slots, MTP). `engine_mlx.py` is the MLX implementation
  (`MLXEngine`, `RadixLinear`/`RawLinear`). The full request-context
  contract at this seam is [below](#the-engine-seam-the-generation-contract).
- **`codec/pack.py` vs `packs.py`.** `codec/pack.py` owns the container and
  the model packing flow (writer, hashes, `iter_pack_dir`, refusals).
  `packs.py` owns cache discovery and the no-`--model` picker
  (`local_packs`, `build_candidates`, `rank_fits`) and imports no torch.
- **`arms.py` holds production loaders too.** Besides the benchmark arms it
  defines `skeleton`, `ckpt_to_skel`, and `load_cpu`, which `codec/pack.py`
  and `serving/engines.py` import; `snapshot_dir`/`resolve_source` are
  re-exported from `serving/checkpoint.py`. Packing, serving, and bench walk
  the same module tree.
- **`serve.py` vs `serving/http.py`.** `serve.py` validates inputs and constructs
  the engine through `check_pack_dir`, `ensure_pack`, `resolve_device`,
  `resolve_runtime`, `build_engine`. `http.py` is transport, the generation
  lock, and the shared generation core; `responses.py` and `messages.py`
  adapt requests and responses for their APIs.
- **Capability checks.** `codec/registry.py` records implementations for each
  tensor codec, operation, and kernel. `serving/capability.py`
  detects thinking channels and tool formats from model templates, and
  reads image input off the engine (`engine.vision`, a structural fact of the
  served tree, not a template probe).
  [`runtimes.py`](../src/drinkme/runtimes.py) defines model-family support for
  each runtime. The model picker, explicit model requests, and MLX loader use
  that shared check; torch leaves family support to its loader.
- **Modules that do not import torch.** `bootstrap.py` (lane choice, install, self-test),
  `detect.py` (hardware identity and budget), `fit.py` (the one fit
  primitive), `suggest.py` (the menu), `check.py` (eligibility from repo
  metadata), `packs.py`, `runtimes.py`, `serving/checkpoint.py`,
  `serving/mrope.py`, `serving/tower_cache.py`, `serving/vision.py` (whose
  tower half imports torch inside its functions) and `serving/video.py` (torch
  for the resize, PyAV for the decode, both inside its functions) import no
  torch at module level.
  This lets Metal setup, `--dry-run`/`--detect-only`, checks before weight
  downloads, and image parsing in the dialects run without torch. `probe.py` (bandwidth) needs the device.

## Where the rest lives

- **Top level.** `src/drinkme/` is the package; `tests/` the CPU suite
  ([testing](testing.md)); `bench/` the developer instruments
  ([bench/README](../bench/README.md)); `lexicons/` the record schema;
  `site/` the results page; `docs/` these pages; `measurements/` is
  `drinkme bench`'s untracked local output.
- **`publish/`**: the publish verb — identity resolution, loopback OAuth,
  PDS writes, record validation.
- **`upstream.py`**: upstream verification, a pack's source shards rebuilt
  and hashed against the Hub's published SHA-256, and its receipt
  ([pack format](pack-format.md#upstream-verification)). **`hub_packs.py`**:
  published packs, the lookup, download and install `serve` runs on a cache
  miss ([published packs](serve.md#published-packs)).
- **`codec/radix_gpu.py`, `radix_kernel_gpu.py`, `radix_ops.py`**: the Triton
  radix kernels and the ops that launch them. **`codec/radix_native.*`**: the
  C++ CPU encoder. **`codec/decode.py`**: the narrow Linears' BF16 GEMV.
  **`ops.py`**: `MC_MAX`, importable without Triton.
- **`metal/`**: the Metal radix kernels.
- **`serving/prefill.py`, `segmented_attention.py`**: chunked prefill
  ([serve](serve.md#chunked-prefill)). **`suffix_attention.py`**: MTP suffix
  attention. **`deltanet.py`, `deltanet_conv.py`, `kernel_route.py`**: the
  DeltaNet kernels and the routine that picks every kernel
  ([serving kernels](serve-kernels.md)).
- **`serving/mtp.py`, `ngram.py`, `speculative.py`, `draft_vocab.py`**:
  speculative decoding ([speculation](serve-speculation.md)).
- **`serving/kvcache.py`, `slotstore.py`, `ctx_checkpoints.py`, `sleep.py`**:
  the prefix slot's cache, its disk tier, its context checkpoints, and
  sleep/wake ([prefix slots](serve-prefix-slots.md), [sleep](serve-sleep.md)).
- **`serving/tools.py`, `tool_formats.py`, `control.py`, `think.py`,
  `detok.py`, `constrain.py`**: tool calls, control tokens, the reasoning
  split, streaming detokenization, and constrained JSON output
  ([tool formats](serve-tool-formats.md)).
- **`serving/kernel_route.py`, `deltanet.py`, `deltanet_conv.py`**: the kernel
  route for each loaded operation, including the DeltaNet recurrence and
  convolution.
- **`serving/vision.py`**: image input, torch-free up to the tower: the
  sources (data URLs, the http(s) fetch, `file://` paths), the limits and
  their named errors, decoding, and each architecture's preprocessing, a
  registry keyed by `config.json`'s `model_type` (Qwen3.5, gemma-4,
  Muse-Glimmer), each reproducing its transformers processor bit for bit.
  Then the engine's side: `Tower` (where the tower sits, its placeholder and
  markers, M-RoPE or 1-D positions, and `bidirectional`),
  `bounded_attention` (the ViT's attention in dispatches of at most
  `WORK_V` query-key pairs, its head zero-padded to a multiple of 16 on
  ROCm), the ROCm patch-embedding GEMM, and the boot self-test of the
  tower's attention ([ROCm](rocm.md#the-vision-tower-on-gfx1151)).
- **`serving/video.py`**: video input for Qwen3.5: the `video_url` sources
  (vision.py's rules with a video's caps), PyAV decoding (the optional
  `drinkme[video]` extra), transformers' `Qwen3VLVideoProcessor` reproduced
  bit for bit (frame sampling by the clip's real rate, the whole-clip pixel
  budget, the frame pairs), and each pair's timestamp text, tokenized by the
  engine's tokenizer. A `VideoInput` rides on `Vision.video`.
- **`serving/image_prompt.py`**: one request's images and videos as the
  forwards need them (a video is one run per frame pair, `Frames`): the expanded ids, the prefix-cache keys, the positions, the
  per-span embeddings and gemma-4's mask. Its docstring's "HOW A RUN MAY BE
  PREFILLED" is the one rule `Tower.bidirectional` decides: a causal run
  (Qwen3.5, Muse-Glimmer) may be cut by chunked prefill and restored
  mid-run; a bidirectional one (gemma-4) never is. `media_ends` is where the
  context checkpoints take one per image and video.
- **`serving/tower_cache.py`**: the tower's output per image and video in
  host RAM, keyed by the digest of what the tower reads, LRU under
  `--tower-cache-gib` ([tower output cache](serve.md#the-tower-output-cache)).
- **`serving/mrope.py`**: Qwen3.5's M-RoPE positions for a prompt with
  images, a port of transformers' `get_rope_index`. A text request passes no
  `position_ids` at all.
- **`mtp.trunk_ids`**: the text model's own forwards (every prefill span
  but the last, speculative verify) embed an image placeholder id as the
  model's wrapper does.

## The Engine seam: the generation contract

What one generation carries across `serving/http.py` → `Engine`. There is
exactly **one** channel: a frozen
`GenerationRequest` value goes in, a stream of typed events comes out.
Nothing travels beside the call — no thread-local, no context variable —
so a caller that is not `http.py` (the warm-up in `serve.py`, the `bench/`
drivers, a direct test) gets precisely what it asked for, and is told what
it got. `tests/test_serving_generation_request.py` holds the direct and wire
renders of two model families equal.

### In: `GenerationRequest` (`serving/engine.py`)

| Field | Type | Built by | Read by |
|---|---|---|---|
| `messages` | `list[dict]`, OpenAI-shaped; string content, or a parts list (`{"type": "text", …}`, `{"type": "image"}`) for a turn carrying images | the dialect: `http._normalize_messages` (chat), `messages.convert_messages` (Anthropic: `system` folded in), `responses.convert_input` | `template.render_prompt` (every real engine), `FakeEngine` (echoes the last user turn) |
| `sampling` | `SampleParams` | `engine.validated_sample_params` at each boundary, after `gen_config.resolve_sampling` (request > profile > `generation_config.json` > OpenAI default). `output_schema`/`output_schema_validated` are set afterwards by the dialect (`response_format`, `text.format`, `output_config.format`). The request-time clamp (`DRINKME_MAX_TOKENS_CLAMP`) produces a **new** request via `dataclasses.replace`; nothing is mutated. | the samplers (`sampling.sample_next`, `constrain.pick_token`), `StopScanner(sampling.stop)`, `mtp.maybe_speculate` (temperature and seed decide the sampled path), the length stop, the constraint (`JsonConstraint(output_schema)`), and the render: a constrained request renders with `enable_thinking=False` unless a kwargs layer says otherwise |
| `tools` | validated OpenAI tool definitions, a missing or null `description` given `""` and missing or null `parameters` `{}` (llama.cpp's defaults), or `None` | `tools.validate_tools` (chat), `messages.convert_tools`, `responses.convert_tools`; `tool_choice: none` zeroes it at the boundary | `render_prompt` (rendered by the family's template), `ToolCallScanner(tools, capability.tool_format)` |
| `template_kwargs` | `dict` or `None` — the REQUEST layer of chat-template kwargs | the dialect: the profile's template keys as the base, then the request's `reasoning_effort` / `chat_template_kwargs` (chat), `thinking` and `output_config.effort` (Messages), `reasoning.effort` (Responses); `enable_thinking: false` whenever a JSON schema is requested | `template.effective_kwargs(engine_kwargs, request_kwargs, constrained=…)` inside the engine's one render: the engine's own `generation_config.json` defaults underneath, the request winning. `/tokenize` passes none: that route renders with the engine's defaults only. |
| `stream` | `bool` | the dialect's `stream` field | nothing in the engine — it is the adapter's fact, carried so the value is complete |
| `request_id` | `str` | minted when the value is built: `chatcmpl-…`, `msg_…`, `resp_…` | the reply's `id`; the same generation on the wire and in a log line |
| `images` | `tuple[vision.PreparedImage, ...]`, template order; empty for text | the dialect: `vision.parse_image_url` / `parse_base64`, then the engine's `Vision.prepare`, outside the generation lock | the engine: the vision tower's input, the M-RoPE positions (`serving/mrope.py`), and the prefix-cache key (`PreparedImage.prefix_key`) |

Sibling calls on the same seam, lockless, tokenizer-only:
`count_tokens(req)` (the request-time context check,
`/v1/messages/count_tokens`, the streamed Messages reply's up-front
`input_tokens`) renders `req` exactly as `generate(req)` will (one `_render`
per engine); `tokenize(prompt|messages, tools, template_kwargs)` is
`/tokenize`.

### Out: the event stream — `Engine.generate(req) -> Iterator[StreamStart | Delta | Finished]`

| Event | When | Carries | Read by |
|---|---|---|---|
| `StreamStart` | **first, always** — the prompt is rendered, the prefix cache has answered, no forward pass has run | `prompt_tokens`; `opens_think` (did the rendered prompt end inside an open think tag — `template.render_prompt`'s read-back, or `FakeEngine.opens_think`); `cached_tokens` (prefix-slot reuse, `HFEngine` only) | `http._generate` builds the `ThinkSplitter` from `opens_think` before the first token exists |
| `Delta` | per delivered piece, after tool-call extraction and stop scanning | `text` | `http._generate` feeds it through the splitter into `push` (the channels, TTFT, inter-token) and `emit` (the wire). The consumer answers it with `send(False)` when the client is gone; the engine stops promptly and finishes with `finish_reason "abort"`. Plain iteration (`send(None)`) continues. |
| `Finished` | last, always, abort included | `result: GenResult` (delivered text, finish reason, token counts, parsed tool calls, `cached_tokens`, `stop_sequence`) | `http._generate` turns it into a `Turn` that each dialect renders; `engine.complete(eng, req, on_delta)` is the driver for every non-streaming caller (bench, tests, warm-up) |

Both engines yield inside their own loops (`HFEngine.generate` inside
`torch.inference_mode()`, `MLXEngine.generate` over its per-request cache),
so a consumer's code between events runs on the generating thread with the
engine suspended.

### The stream-start fact and TTFT

`StreamStart` marks the start of the stream; it is **not** the TTFT mark. TTFT is defined in `serving/metrics.py`:
`http._generate` takes `t0` before requesting `gen_lock`, and the first
`Delta` that carries reasoning **or** content after the think split records
`metrics.observe_ttft(now - t0)` — queueing time is in the number, matching
`bench/serve_timing_ab.py`'s client-side definition — and later deltas
record `observe_inter_token`. That is the only server-side TTFT, measured
in one place for both engines. The SSE keep-alive stops on `emit`'s first
call, which is the same first delta. Client-side, the bench drivers time
their own first delta;
`drinkme bench` times prefill directly (`arms.timed_ttft`) and never goes
through this seam.
