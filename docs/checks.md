# Checks by change type

Every code change runs the [CPU suite](../AGENTS.md#cpu-tests) plus the checks
in its row below. Device tests skip in the isolated CPU run, so a change to
device execution also needs hardware acceptance
([developer instruments](../bench/README.md)). Read each gate's printed
verdict, because the process exit status can be misleading
([known behavior](dev-environment.md#known-behavior)). Server-based gates use a
test port, never 3215. Script names without a directory are in `bench/`;
module paths are under `src/drinkme/`, tests under `tests/`.
"Hardware acceptance" says whether a change type needs a run on real
hardware; "Gates" are the instruments that run there and print a pass/fail
verdict ([check words](architecture.md#vocabulary)).
[Versioning](versioning.md#what-to-re-run) says which changes need a new
minor or major version, and what they do to packs and published records.

| Change type | Modules | CPU tests | Hardware acceptance | Gates |
|---|---|---|---|---|
| [Codec / pack format](#codec--pack-format) | `codec/radix.py`, `radix_pack.py`, `radix_native.cpp`, `pack.py`, `identity.py`, the module classes in `swap.py` | `test_radix.py`, `test_radix_pack.py`, `test_radix_bounds.py`, `test_pack.py`, `test_pack_*.py`, `test_codec_registry.py`, `test_bits_per_weight.py`, `test_bench_arm_wiring.py`, `test_unsupported_format.py`, `test_stream_compressed_arms.py`, `test_serving_pack_iter.py` | Required | `radix_gemv_bitpin.py`, `radix_mc_bitpin.py`, `radix_schedule_bitpin.py`; `radix_bitpin_mlx.py` on Metal |
| [Kernel / launch schedule](#kernel--launch-schedule) | `codec/radix_kernel_gpu.py`, `radix_gpu.py`, `radix_ops.py`, `ops.py`, `radix_schedule.py`, `decode.py` (the narrow GEMV) | `test_codec_mc.py`, `test_codec_bias_rounding.py`, `test_radix_twin.py`, `test_radix_bounds.py`, `test_radix_schedule.py`, `test_swap_prefill_dense.py`, `test_hotloop_equivalence.py` (the kernel-dispatching parts skip) | Required | the codec gates, `parity_rotation.py`, `hotloop_gpu_acceptance.py`, `greedy_ab_smoke.sh`, `logits_row_identity.py` |
| [Serving loop](#serving-loop) | `serving/engines.py`, `kvcache.py`, `slotstore.py`, `ctx_checkpoints.py`, `sleep.py`, `mtp.py`, `ngram.py`, `speculative.py`, `sampling.py`, `template.py`, `think.py`, `prefill.py`, `segmented_attention.py`, `suffix_attention.py`, `cudagraph.py`, `deltanet.py`, `deltanet_conv.py`, `kernel_route.py`, `draft_vocab.py` | `test_serving_engines.py`, `test_serving_kvcache.py`, `test_serving_prefix_slots.py`, `test_serving_ctx_checkpoints.py`, `test_serving_slotstore.py`, `test_serving_sleep.py`, `test_serving_mtp*.py`, `test_serving_ngram.py`, `test_serving_speculative.py`, `test_serving_gemma_spec.py`, `test_spec_agree.py`, `test_serving_sampling*.py`, `test_serving_template.py`, `test_serving_think.py`, `test_hotloop_equivalence.py`, `test_serving_prefill_chunk.py`, `test_serving_prefill_bound.py`, `test_serving_suffix_attention.py`, `test_serving_deltanet*.py`, `test_cudagraph_policy.py` | Cache, speculation and sleep changes | one per feature (below) |
| [Image input](#image-input) | `serving/vision.py`, `image_prompt.py`, `mrope.py`, the tower tables in `arms.py`, the image paths of `serving/engines.py`, `prefill.py`, `mtp.py` and the dialects | `test_serving_vision.py`, `test_serving_vision_urls.py`, `test_serving_vision_tower.py`, `test_serving_vision_attention.py`, `test_serving_image_prompt.py`, `test_serving_mrope.py`, `test_serving_gemma_vision.py`, `test_serving_glimmer_vision.py`, `test_serving_head_logits.py` | Tower, attention, patch-embedding, prefill and cache changes | `vision_dispatch.py`, `vision_attention_check.py`, `vision_selftest.py`, `vision_bitpin.py`, `gemma_vision_gate.py`, `glimmer_vision_gate.py`, `vision_smoke.py` |
| [HTTP adapter](#http-adapter) | `serving/http.py`, `responses.py`, `messages.py`, `tools.py`, `tool_formats.py`, `constrain.py`, `detok.py`, `control.py`, `capability.py` | `test_serving_http.py`, `test_serving_responses.py`, `test_serving_messages.py`, `test_serving_tools.py`, `test_serving_tool_formats.py`, `test_serving_glimmer_channel.py`, `test_serving_constrain.py`, `test_serving_detok.py`, `test_serving_control.py`, `test_serving_capability.py`, `test_serving_advertised_ctx.py`, `test_serving_overflow_wording.py`, `test_serving_wiring.py` | Not for protocol structure and validation | `serve_dialect_smoke.py` for a dialect change; the real-client smokes |
| [MLX runtime](#mlx-runtime) | `serving/engine_mlx.py`, `arms_mlx.py`, `metal/gemv_radix.py`, `metal/dense_radix.py`, `metal/twin_radix.py` | Linux: `test_radix_mlx_toy_reference.py`, `test_serving_engine_mlx_fit_check.py`, `test_twin_radix.py`. Mac: `test_serving_engine_mlx.py`, `test_metal_radix.py`, the mlx halves of `test_identity_torch_free.py` and `test_pack_identity.py` | Required, on a Mac | `radix_bitpin_mlx.py --fixture`, `radix_metal_timing.py`, `serve_dialect_smoke.py` |
| [Benchmarking and publishing](#benchmarking-and-publishing) | `bench.py`, the timing half of `arms.py`, `fit.py`, `detect.py`, `suggest.py`, `probe.py`, `publish/`, `lexicons/` | `test_bench*.py`, `test_fit.py`, `test_detect.py`, `test_suggest.py`, `test_arms_*.py`, `test_publish_*.py`, `test_modal_sweep.py`, `test_bench_script_references.py` | When measured values change | `drinkme bench --dry-run`, a real `drinkme bench`, `publish_e2e.py`, `fit72b_acceptance.sh`, `modal_fresh_box.py` |
| [Bootstrap, CLI, selection](#bootstrap-cli-selection) | `bootstrap.py`, `bootstrap_selftest.py`, `cli.py`, `packs.py`, `check.py`, `serve.py` input validation, `serving/checkpoint.py` | `test_bootstrap.py`, `test_cli_entry.py`, `test_cli_serve_no_model.py`, `test_packs.py`, `test_check.py`, `test_unsupported_format.py`, `test_serving_checkpoint.py`, `test_serving_wiring.py` | When what gets installed changes | `drinkme bootstrap --dry-run` and `--detect-only` |
| [Docs](#docs) | `docs/`, `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `bench/README.md` | none | No | links, names and numbers (below) |

## Codec / pack format

`test_bench_arm_wiring.py` checks that bench's arm is `pack_model`'s bytes.

- `bench/radix_gemv_bitpin.py`: all 65,536 BF16 patterns through every
  compression profile, the fused epilogue.
- `bench/radix_mc_bitpin.py --pack-dir <pack>`: the multi-column arm
  against the float64 oracle on real tensors.
- `bench/radix_schedule_bitpin.py`: every launch-table row.
- On Metal, `bench/radix_bitpin_mlx.py --fixture`, with the fixture from
  `radix_fixture_build.py` on a machine with torch.

A format or container change also re-packs a menu model and runs
`drinkme verify` and `drinkme bench --pack-dir` on it. A change to the
embedded checkpoint (`write_embedded`, `resolve_pack_source`) also serves
the new pack with `HF_HUB_OFFLINE=1` and an empty `HF_HOME`, and compares a
greedy transcript with the same pack served with its source checkpoint
still present.

## Kernel / launch schedule

`test_swap_prefill_dense.py` covers the M router and dense numerics.

- Run the three codec gates above before enabling a registry row
  (`codec/registry.py`).
- A schedule-row change reruns `bench/parity_rotation.py` on that class of
  machine.
- A change to `_gemv`, `_gemv_mc`, their launch or the twin's rows reruns
  `bench/radix_twin_bitpin.py`, which checks the bench's twin (`RAW=True`)
  against the compressed module, bitwise at every launch-table row and twin
  row, with BF16 and FP32 activations. Its `--compile` mode runs on the CPU
  at the device's pointer specialization and checks that the twin's
  accumulator layout is the served kernel's in every specialization.
- A twin-row change reruns `bench/parity_rotation.py --radix-arms sip twin`
  on that class of machine.
- A change to the lean sip decoder (`radix_kernel_gpu._decode_sip_lean`)
  runs `bench/radix_lean_gate.py` on a real sip pack (its bits and GEMVs
  against the scheduled decoder's, bitwise) with the codec gates, the
  twin gate above on an NVIDIA card, and `bench/radix_lean_interp.py` on
  the CPU first; `bench/radix_sass_count.py` compares its instruction
  count per weight.
- A change to the lean gulp decoder (`radix_kernel_gpu._decode_gulp_lean`)
  runs the same gates on a ROCm machine: `bench/radix_lean_interp.py
  --profile gulp --decoders 2,4` on the CPU first, then
  `bench/radix_lean_gate.py` on a real gulp pack, the twin gate with
  `--profiles gulp --real-arms gulp`, and `bench/parity_rotation.py` with
  gulp at the grid. A change to the CUDA one (`_decode_gulp_lean_cuda`)
  runs them on an NVIDIA card (`--decoders 2,4,5` on the CPU), with
  `bench/radix_sass_count.py --widths 2,2,4,8 --decoders 2,4,5 --warps 1
  --tiles 1` for its count and the CUDA-graph gate (`bench/cuda_graph_gate.py`)
  on a gulp pack in place of the rotation.
- A change to the narrow GEMV (`radix_ops.gemv_narrow`, `swap.NarrowLinear`),
  the stock GEMV (`swap.STOCK_GEMV`) or the raw GEMV (`swap.RAW_GEMV`,
  `swap.stock_linear`) reruns `bench/narrow_linear_knobs.py`,
  `bench/stock_blas_knobs.py` or `bench/raw_head_gemv.py` on that machine;
  each checks every route's output against a float64 reference, and
  `raw_head_gemv.py` checks the routed Linear bitwise against the twin
  module and the stock route, in a codec tree and in a stock tree. A
  stock GEMV change also runs `bench/stock_token_identity.py`, the arms'
  greedy tokens on the bench prompt. A narrow GEMV change also runs
  `bench/narrow_linear_numerics.py`, the row identity and float64 error on
  the menu's real narrow weights.
- A hot-loop change runs `bench/hotloop_gpu_acceptance.py` through
  `hotloop_gpu_ab.sh` (both arms, byte-equal results).
- Output identity: `bench/greedy_ab_smoke.sh` (compares compressed and
  `--stock` transcripts; investigate differences with the
  [numerical caveat](method.md#numerical-behavior)) and
  `bench/logits_row_identity.py`.
- A speed claim: `drinkme bench --pack-dir` on the real pack.

## Serving loop

Cache, speculation, and sleep changes require hardware acceptance. CPU tests
suffice for sampling, template, and thinking-parser changes when they cover the
affected token IDs.

- Prefix cache: `bench/prefix_cache_verify.py` (warm vs cold over the wire,
  one slot), `bench/prefix_slots_verify.py` (N slots, logits),
  `bench/slot_restore_verify.py` (the on-disk tier across a process death).
- Context checkpoints: `bench/ctx_checkpoint_probe.py` (CPU: where each
  re-rendered turn parts from its slot and what it reuses, on the real
  tokenizers and templates), `bench/ctx_checkpoint_sizes.py` (CPU: one
  checkpoint's bytes per menu model against the fit formula), and on the
  device `bench/vision_smoke.py`'s cache and agent phases, whose re-rendered
  turns reuse the previous prompt.
- Sleep/wake: `bench/sleep_wake_verify.py` (memory freed, logits
  bit-identical after wake).
- MTP: `bench/mtp_gpu_acceptance.py`, `bench/mtp_prefix_slots_acceptance.py`,
  `bench/mtp_head_bitpin.py` (the reduced head matches checkpoint weights bit
  for bit).
- N-gram: `bench/ngram_agreement_gate.py` on gemma and Qwen3-8B.
- A served-speed claim: `bench/serve_drive.py` or `serve_timing_ab.py` against
  a test port with the cache on ([server timing](bench.md#server-timing)).
- Chunked prefill: `bench/prefill_dispatch.py`, `bench/prefill_chunk_8b.py`.
- CUDA graphs (`serving/cudagraph.py`, and any change to what the decode or
  verify step runs on CUDA): `bench/cuda_graph_gate.py` per arm on a CUDA
  card, then `--compare` (graph replay bitwise equal to the eager static
  step at each verify width, the step at 4k and 16k live, serve's own
  generate eager and graph with speculation off and on), and
  `bench/cuda_graph_serve.py` over HTTP. A family joins `VERIFIED` only on
  those verdicts ([CUDA graphs](serve-kernels.md#cuda-graphs)).

## Image input

The CPU tests hold each preprocessor to its transformers processor bit for
bit on a fixture set (Muse-Glimmer's against golden digests from a
torchvision run), the M-RoPE positions to transformers' `get_rope_index`,
and toy models of each architecture to their transformers reference: prefill
logits whole and chunked, greedy transcripts with speculation on and off,
the prefix cache keyed on image content, the pack's `vision` block and an
offline load whose tower features equal `get_image_features`. They also
check that every ViT attention call goes through `bounded_attention`, padded
on a simulated ROCm device, and that the boot self-test fails on a broken
kernel. They cannot see a device kernel. On hardware, run the dispatch gate
first, from the smallest image up, before anything reads an answer:

- `bench/vision_dispatch.py` (Qwen3.5 models): the tower's longest dispatch
  under `rocprofv3` over an image-size ladder, alone or the whole served
  request (`--engine`). It refuses a step predicted over 250 ms and stops at
  60 ms or a ring timeout. It proves no tower dispatch comes near the
  graphics-ring timeout [chunked prefill](serve.md#chunked-prefill) guards
  against, and it is what calibrates `vision.WORK_V`.
- `bench/vision_attention_check.py` (Qwen3.5 models): every SDPA backend at
  the ViT's head width, on the tower's own activations, layer by layer
  against fp32. It proves the kernel the tower runs is right on this device;
  the per-model gates' `--fp32-check` asks the same of gemma-4 and
  Muse-Glimmer.
- `bench/vision_selftest.py --snap SNAP`: the engine's boot self-test alone,
  for any checkpoint's ViT, without loading weights; `--no-pad` is its
  negative control and must FAIL at head width 72 on gfx1151.
- `bench/vision_bitpin.py` (Qwen3.5 models): transformers' tower over
  drinkme's two routes, drinkme's stock tower and its compressed tower are
  byte-equal, bounded and unbounded; the served tower is no further from an
  fp32 tower than transformers' own bf16 tower (1.5× its mean error at
  most); the first-token logits agree; the processor's ids and pixels equal
  transformers'. It proves the codec changes nothing in the tower.
- `bench/gemma_vision_gate.py`, `bench/glimmer_vision_gate.py`: the same
  questions per model, one step per process: `tower` (the byte pin, plus
  `--fp32-check`: every ViT attention call against fp32 on its own inputs,
  with the self-test's tolerances, which a byte pin cannot do because every
  arm calls the same kernel; Muse-Glimmer's `--no-fp32-tower` runs it at the
  cap), `row` (the first-token logits, transformers' forward against the
  engine's), `compare` (one argmax; greedy answers equal, or parting where
  each arm's two candidates are at most one bf16 step apart), `trace` (the
  longest dispatch in a trace) and `smoke` (a nonce read through two
  dialects of a running server).
- `bench/glimmer_preprocess_pin.py`: Muse-Glimmer's preprocessing against
  transformers' torchvision-only processor in a separate environment that
  has torchvision; it regenerates the suite's golden digests.
- `bench/vision_smoke.py --model 27b|mimo|gemma|glimmer`: images over the
  wire. Screenshot nonces through chat, messages, a `tool_result` and
  responses, and a Retina capture; the pixel-cap ladder; the prefix cache
  (warm equals cold, the tower skipped for a resent image, two same-size
  images kept apart, a cold-tier restore); speculation (each arm matches
  serial decode or parts at a near-tie); sleep and wake, with the tower's
  bytes unchanged; and an agent loop.

A preprocessing change reruns the gate's `row` step or `vision_bitpin.py`;
a change to the tower's attention, `HEAD_ALIGN`, `WORK_V` or the patch
embedding reruns the dispatch gate, the attention check, the self-test with
`--no-pad`, and the gates' `tower` step with `--fp32-check`; a prefill,
position or cache change reruns the `row` step and the smoke's `cache` and
`spec` phases.

## HTTP adapter

The CPU tests all run over `FakeEngine` on a real localhost socket;
`test_serving_messages.py` holds the three dialects to one engine transcript.
They suffice for protocol structure and validation.

- A new or changed tool-call dialect: `bench/serve_dialect_smoke.py` against a
  model-backed server is what flips a row's `tested` flag; gemma additionally
  `bench/gemma_control_tokens_gate.py`. A reasoning split that the next turn's
  history must render back (Muse-Glimmer's `atem`) also needs a two-turn check
  that the second prompt extends the slot: `bench/vision_smoke.py --phases cache`.
- Real clients: `bench/responses_sdk_smoke.mjs`, `responses_codex_smoke.sh`.
- `bench/fake_server.py` covers protocol only; follow it with a model-backed
  smoke when generation behavior changed.

## MLX runtime

On Linux without `mlx`, `test_serving_engine_mlx_fit_check.py` stubs the mlx
surface and the other `mlx` tests skip. On a Mac, the four mlx tests read
`DRINKME_RADIX_MLX_TOY` for their toy (`bench/radix_mlx_toy_build.py`; the
last two want the plain toy and the `--identity` toy respectively), and none of
the four needs torch at collection or in-process to reach it.

- `bench/radix_bitpin_mlx.py --fixture` on the Mac (`--selftest` must FAIL; it
  checks that the gate can fail).
- `bench/mlx_twin_bitpin.py --pack <pack>` wherever mlx runs: the twin's
  decoder against the CPU oracle on every tensor of a real pack (`--selftest`
  must FAIL).
- Timing through `bench/radix_metal_timing.py`.
- `bench/serve_dialect_smoke.py` against a Mac test port.

Follow [Metal hardware verification](metal.md#hardware-verification) and record
the device, MLX version, and `computePath: fused`.

## Benchmarking and publishing

CPU tests suffice for benchmark planning, record structure, and validation.
Changes to measured values also require hardware checks.

- `drinkme bench --dry-run` (no torch) for the plan.
- A real `drinkme bench --model Qwen3-1.7B` on the machine for the record.
- `bench/publish_e2e.py` on a test PDS (credentials via `--env-file`, never on
  the command line).
- `bench/fit72b_acceptance.sh` for the large fit point.
- `bench/modal_fresh_box.py` for a fresh install end to end (rents hardware).

## Bootstrap, CLI, selection

CPU tests cover decision logic with mocked installations.

- `drinkme bootstrap --dry-run` and `--detect-only` on the machine (install
  nothing).
- The real self-test
  (`test_bootstrap.py::test_the_real_self_test_passes_on_this_box`) runs only
  with a visible accelerator.
- A change to what gets installed is verified on a fresh environment in a
  separate human session, or on `bench/modal_fresh_box.py`.

## Docs

No test reads a Markdown doc ([invariants](../AGENTS.md#invariants));
`tests/test_vocabulary.py` holds `src/` and the lexicon JSON to
[architecture](architecture.md)'s vocabulary table, and docs follow the same
table by review. Runtime tests are not needed for prose-only changes.

- Every relative Markdown link you touched resolves.
- Every flag, field, and module name matches the tree (check the source; see
  [architecture](architecture.md) for terminology).
- A number states where it was measured or says it is maintainer-reported.
