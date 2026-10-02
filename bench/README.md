# Developer instruments

This directory contains performance experiments, GPU probes, and acceptance
gates. The user-facing benchmark is `drinkme bench`; these scripts are not
part of installation or CI. Most need a real pack and accelerator. Read a
script's header and `--help` before running it, and follow
[AGENTS.md](../AGENTS.md).

## Instruments

| Area | Scripts | Purpose |
|---|---|---|
| Fresh installation | `modal_fresh_box.py` | Install and run `drinkme bench` on a clean NVIDIA Modal instance |
| Release sweep on Modal | `modal_sweep.py` (+ `modal_sweep_drive.py`) | `drinkme bench --model <M>` as a stranger runs it (`uv sync`, the accelerator bootstrap, bench) on one rented NVIDIA card per cell; the record comes back with bench's own name and bytes. `--profile gulp` cuts a gulp pack in the container, verifies it and benches off it (`--pack-dir`). The driver deploys per card and spawns / collects in bounded waits |
| Speculation range on Modal | `modal_spec_range.py` | The 27B sip on an L40S: a fresh pack, `ngram_gpu_ab.py` off/mtp over its four prompts with reps, and the served default's peak GPU memory |
| Endpoint timing | `serve_timing_ab.py`, `modal_serve_ab.py`, `serve_drive.py` | Measure client-visible stock/compressed performance; `serve_drive.py` drives one fixed prompt against an already-running server, speculation on or off (the Modal drivers' served step) |
| Live-cache attention | `live_cache_attention_ab.py` | Interleave two single-token mask policies on one loaded model, over a private HTTP port; record context-dependent throughput and transcript agreement with resident prefix caching enabled |
| MTP suffix attention | `mtp_suffix_attention_ab.py` | Interleave explicit and native lower-right causal attention on one resident model; HTTP throughput, draft acceptance, transcript agreement, and optional separate cycle phase timings |
| Cold prefill | `prefill_conv_ab.py` | Interleave upstream and FLA prefill convolution over exact context lengths, with serial/MTP HTTP timing and transcript agreement; optionally compare a BLAS preference or saved GEMM cache |
| Prefill dispatch length | `prefill_dispatch.py`, `prefill_chunk_8b.py` | The longest single GPU dispatch of a prefill (or decode) forward under `rocprofv3 --kernel-trace`: a two-layer model at Qwen3.8-27B shapes swept over new tokens and cached context under a predict-then-launch safety ladder, and the Qwen3-8B pack through the real engine (dispatch windows, prefill tok/s per chunk size, chunked-vs-whole transcripts); [chunked prefill](../docs/serve.md#chunked-prefill) |
| Image input | `vision_dispatch.py`, `vision_attention_check.py`, `vision_selftest.py`, `vision_bitpin.py`, `vision_smoke.py` (screenshots from `vision_screens.py`) | The vision tower's longest dispatch under `rocprofv3` over an image-size ladder, alone or the whole served request (`--engine`), which calibrates `vision.WORK_V`; every SDPA backend at the ViT's head width against fp32; the engine's boot self-test of the tower's attention alone, for any checkpoint's ViT, with an unpadded negative control (`--no-pad`); the tower and first-token logits against transformers, stock and compressed; the screenshot smoke, pixel-cap ladder, prefix cache, speculation and sleep/wake with images over the wire (`vision_smoke.py --model 27b\|mimo\|gemma\|glimmer`; `vision_dispatch.py`, `vision_attention_check.py` and `vision_bitpin.py` are Qwen3.5-only, the per-model gates below cover the rest). What each proves: [checks](../docs/checks.md#image-input) |
| Offline ROCm GEMM tuning | `prefill_gemm_tune.py` | Measure the 27B's dominant 8K/32K projection shapes, check candidate arithmetic, and export a device/runtime-validated TunableOp cache; [workflow](#offline-rocm-prefill-gemm-tuning) |
| Adaptive speculation | `mtp_ask.py`, `bail_sweep.sh`, `bail_sweep_table.py` | Sample one prompt repeatedly with `mtp_ask.py`; sweep window, floor, and re-arm settings with `bail_sweep.sh`; format results with `bail_sweep_table.py` ([serve-speculation.md](../docs/serve-speculation.md#adaptive-bail-and-re-arm)) |
| Kernel correctness | `radix_gemv_bitpin.py`, `radix_mc_bitpin.py`, `radix_schedule_bitpin.py`, `radix_twin_bitpin.py`, `mtp_head_bitpin.py`, `*_gpu_acceptance.py`, `*_gate.py` | Device checks required for kernel and format changes: the three BF16-codec gates (every bf16 bit pattern, the multi-column arm against the float64 oracle, every launch-table row), the bench twin against the compressed module (bitwise, every launch-table row; `--compile` on the CPU) and the MTP head sub-pack |
| Lean decoders | `radix_lean_gate.py` | The lean decoder (sip on CUDA, gulp on ROCm and CUDA) against the scheduled one on every radix tensor of a real pack (dense decode, M=1 and multi-column GEMV bitwise, and the checkpoint's own bits), then each decoder's GEMV time per shape class; `--selftest` must FAIL |
| CUDA graphs | `cuda_graph_gate.py`, `cuda_graph_serve.py`, `modal_cuda_graphs.py` | The correctness bar for graph-mode decode, one arm per process: the static step's attention against a masked SDPA reference, graph replay against the eager static step (tokens and final logits, bitwise, per verify width), the bench loop eager and graph, the step at long live lengths against today's eager step, serve's own generate with speculation off and on, and `--compare` across arms; `--only-warm` repeats serve's request on one slot and prints the logit margin where warm and cold runs part. `cuda_graph_serve.py` compares one streamed chat request eager and graph over HTTP on a test port; `modal_cuda_graphs.py` runs both on an L4 or H100 ([CUDA graphs](../docs/serve-kernels.md#cuda-graphs)) |
| Decoders on the CPU | `radix_lean_interp.py` | Every requested decoder under the Triton interpreter, no device: the dense decode against the source bits and the GEMV and multi-column arms against the scheduled decoder, on tensors that reach every stream field; `--selftest` must FAIL |
| Instruction counts | `radix_sass_count.py` | The radix GEMV's SASS for NVIDIA targets compiled on the CPU: the tile loop's instructions per weight per thread per decoder, its opcode histogram, registers and resident warps (`--diff A,B`, `--dump`); a static count, not a timing |
| Host floor | `host_floor.py`, `mtp_cycle_floor.py` | The stub method: every compressed Linear's kernel replaced by a no-op of the same shape, so what remains is the host+framework floor — of the serial decode loop, and of one MTP draft/verify cycle (trunk step, draft chain, verify, the drafter's cost coefficient `c`; runs on the CPU toy with `--toy`); `mtp_cycle_floor.py --stock --arms real` also times each cycle's phases (draft, verify, accept, restore, rebuild) with device events on the BF16 checkpoint or the pack |
| Verify cost per shape | `verify_rows_linear.py` | Every Linear of a loaded model at M=1 and at the verify's M rows, rotated over its real layers with an event pair per call: `gemv_fused` against `gemv_mc` on the pack, `F.linear` on BF16 |
| Launch schedule | `parity_rotation.py`, `parity_common.py`, `roofline_wall.py` | Interleaved timing sweeps and bandwidth probes used to populate `codec/radix_schedule.py` for each box class |
| Raw Linears' GEMV | `stock_blas_knobs.py`, `narrow_linear_knobs.py`, `narrow_linear_numerics.py`, `modal_narrow_ab.py`, `raw_head_gemv.py`, `stock_token_identity.py` | Every BLAS knob for the raw BF16 Linears' one-row call beside the twin's Triton GEMV (the [stock GEMV](../docs/serve-kernels.md#stock-gemv) per box), the [narrow GEMV](../docs/serve-kernels.md#narrow-gemv) against `torch.mv` and `F.linear` on rocBLAS and hipBLASLt at M = 1..8 on every narrow shape of the menu (on an NVIDIA L4 through Modal as well), its row identity and float64 error on the menu's real narrow weights, and a tied `lm_head`'s one-row call under the stock GEMV against the twin kernel and the narrow GEMV's kernel (the [raw GEMV](../docs/serve-kernels.md#raw-gemv)), with its bitwise gate, and whether the arms' greedy tokens on the bench prompt agree under each stock GEMV (tokens, top-2 margins, logits bitwise) |
| NVIDIA on Modal | `modal_radix.py`, `modal_nvidia2.py` (+ `modal_radix_drive.py`, `modal_nvidia2_drive.py`) | Pack, `bench --pack-dir`, the rotation sweep and a served pair per rented card; the served step runs `serve_drive.py` in the same container |
| Apple silicon | `radix_bitpin_mlx.py`, `radix_metal_timing.py`, `mlx_twin_bitpin.py` | The BF16-codec tensor gate on Metal, the fused GEMV's timing, and the bench twin's MLX decoder against the CPU oracle on a real pack |
| API integration | `serve_dialect_smoke.py`, `responses_sdk_smoke.mjs`, `responses_codex_smoke.sh` | Protocol and real-client tests against a running server |
| Engine-seam exactness | `genreq_gate.py` | A fixed 14-case matrix (three dialects × thinking on/off × stream on/off, one JSON-schema and one tool-call request), greedy, against a running server: one transcript sha per case and the server's own TTFT observation count from `/metrics`; `--compare before.json after.json` holds two servers sha-identical case by case (the seam-change gate, [architecture.md](../docs/architecture.md#the-engine-seam-the-generation-contract)) |
| Publish | `publish_e2e.py`, `publish_consent.cjs` | OAuth, record write, independent read-back, and cleanup on a test PDS |
| Cache and sleep correctness | `prefix_cache_verify.py`, `prefix_slots_verify.py`, `slot_restore_verify.py`, `sleep_wake_verify.py`, `mtp_prefix_slots_acceptance.py` | Warm vs cold logits and transcripts across one slot, N slots, a process restart, and a sleep/wake cycle |
| Context checkpoints | `ctx_checkpoint_probe.py`, `ctx_checkpoint_sizes.py` | On the CPU: a slot spy over the real tokenizers and chat templates of gemma-4, Qwen3.8-27B and Muse-Glimmer through the HTTP dialects, with scripted replies, before and after checkpoints; and one checkpoint's bytes per menu model, measured at the real shapes against the fit formula ([context checkpoints](../docs/serve-prefix-slots.md#context-checkpoints)) |
| Output identity | `greedy_ab_smoke.sh`, `logits_row_identity.py`, `hotloop_gpu_ab.sh` | Compressed vs stock transcripts and logits; hot-loop on/off byte equality |
| Image input (Muse-Glimmer) | `glimmer_vision_gate.py`, `glimmer_preprocess_pin.py` | Muse-Glimmer-30B's vision acceptance, `gemma_vision_gate.py`'s steps over Glimmer's tower (the ViT, adapter and projection) and reference (`MuseGlimmerModel.get_image_features`, forward with `image_grid_thw`), with `--fp32-check --no-fp32-tower` checking every attention call at the cap, where a full layer sees 16,320 keys, without the CPU tower; and the preprocessing pinned against transformers' torchvision-only processor, run in a throwaway venv that has torchvision (the suite's golden digests come from it). `vision_smoke.py --model glimmer` runs the over-the-wire phases |
| Image input (gemma-4) | `gemma_vision_gate.py` | gemma-4-31B-it's vision acceptance, one step per process: the tower alone against transformers' `get_image_features` over drinkme's attention route, stock and compressed, byte for byte, and with `--fp32-check` every ViT attention call against fp32 on its own inputs, with the boot self-test's tolerances, plus the tower in float32 on the CPU with eager attention for finiteness (and the dispatch gate's worker under `rocprofv3`); the first-token logits row of an image prompt, transformers' forward against the engine's, stock and compressed; the longest dispatch in a trace; a nonce screenshot through a running server's two dialects |
| Fixtures | `radix_fixture_build.py`, `radix_mlx_toy_build.py` | Real-tensor fixture for the Metal gate; toy packs for the mlx tests (`DRINKME_RADIX_MLX_TOY`) |
| External engines | `llamacpp_bench.py`, `ollama_bench.py` | The same model on llama.cpp's `llama-server` and on Ollama, for a comparison outside drinkme's own stock arm |

Other scripts are one-off probes; read their headers.

The speculation probe records per-run `/metrics` deltas (tokens per step,
acceptance, trips, and re-arms), TTFT, decode tokens/s, and p50/p95 inter-token
latency. `bail_sweep.sh` runs each `DRINKME_MTP_BAIL_WINDOW`,
`DRINKME_MTP_BAIL_FLOOR`, and `DRINKME_MTP_REARM` configuration on port 3299 in turn. It holds the GPU lock and can be
re-run without repeating completed configurations.

`modal shell bench/modal_fresh_box.py::fresh_box` opens the fresh-box environment
with the repository at `/root/drinkme`. It provisions remote infrastructure;
use it as a deliberate hardware test. The `modal_radix.py` and `modal_sweep.py`
images mount only files git tracks, so private working files kept beside the
checkout never reach Modal.

`fake_server.py` provides canned CPU responses for protocol tests. Follow those
with a model-backed smoke when validating generation behavior. The Node SDK
checks use dependencies in `bench/.node`; install them there during environment
setup. `node_modules` is ignored.

The publish end-to-end test needs Node/Playwright (`--browser-dir`) and a test
account. Supply credentials through `--env-file` and `--password-env`, never as
literal command-line arguments. `--handle` and `--plc` (both required) and the
optional `--pds` select the test service. The test writes a record and deletes it afterward.

## Large-model runs

`fit72b_acceptance.sh` checks the cached Qwen2.5-72B snapshot, requires a
compressed-only dry-run plan, checks live available memory, then runs the bench
under a watchdog. The watchdog targets only that benchmark PID and reports
minimum available memory. Read the output verdict: TheRock's exit handler can
mask a failure with exit code zero.

Shell drivers often assume particular packs and devices. Keep test servers on
separate ports and never use broad process-kill commands; 3215 is
`drinkme serve`'s default port, so a server there may be one someone relies on. Generated records and logs are local development evidence and
are not tracked.

## Offline ROCm prefill GEMM tuning

`bench/prefill_gemm_tune.py` measures the four dominant BF16 MLP projection
shapes in Qwen's 27B model at 8K and 32K context. It can save PyTorch TunableOp
choices for those shapes. This is a separate, optional experiment: serving
does not enable TunableOp by default. Other shapes keep normal dispatch, so
the cache does not promise a benefit at arbitrary prompt lengths.

Run tuning offline, with other GPU work stopped, using the existing
interpreter. For example, from the repository root:

```sh
PYTHONPATH=src flock /tmp/drinkme-gpu.lock .venv/bin/python \
  bench/prefill_gemm_tune.py \
  --out verification/prefill-gemm.json \
  --tuning-file verification/prefill-gemm0.csv
```

Tuning takes several minutes per shape. The instrument checks candidate
outputs against FP64 dot products, then writes and reads back a cache with
PyTorch's device/runtime validators. Regenerate it after a hardware or runtime
change. In a worktree, use the original checkout's interpreter as described
in [AGENTS.md](../AGENTS.md#cpu-tests).

To load that cache on GPU 0, set these variables before starting the server:

```sh
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_FILENAME="$PWD/verification/prefill-gemm.csv"
```

PyTorch inserts the device ordinal into the environment-specified filename,
so `prefill-gemm.csv` loads `prefill-gemm0.csv`. Keep tuning disabled while
serving: missing shapes should use their defaults without an online search.
Before adopting a cache, measure it with `bench/prefill_conv_ab.py
--reference-conv-fla --tuning-file verification/prefill-gemm0.csv` and the
usual pack/output arguments. This holds convolution routing constant and
compares the cache against normal GEMM dispatch on uncached HTTP requests.

## Instrument-only settings

`DRINKME_REFERENCE`, `DRINKME_HOTLOOP_OFF`, `DRINKME_DETOK_VERIFY`,
`DRINKME_NO_WARMUP`, `DRINKME_FAKE_ENGINE`, `DRINKME_AB_PACK` and
`DRINKME_MTP_DRAFT_VOCAB` (`draft_vocab_build.py` builds its subset) are
developer controls. Their source comments define their
behavior; they are not supported operator options. Operator settings are in
[the CLI reference](../docs/cli.md).
