# Apple silicon: MLX and Metal

The BF16 kernels pass bit-exact checks on Apple silicon, and whole-model BF16
serving runs on it. On an M4 with 24 GB, Qwen3-1.7B decodes 1.21× and
Qwen3-4B 1.25× as fast as MLX-LM's BF16 (34.24 vs 28.32 and 15.53 vs 12.42
tok/s), bit-exact. Other model families, the 8B and larger on this machine,
and speculation on MLX remain in progress. Metal records publish like any other
([local records](bench.md#local-records)). This page
is for developers testing or extending the MLX runtime and its Metal kernels.

## Supported scope

The mlx runtime supports dense Qwen3 text models (0.6B, 1.7B, 4B, and 8B)
from BF16 packs ([pack format](pack-format.md); `drinkme pack --sip` /
`--gulp`; see [The BF16 codec on Metal](#the-bf16-codec-on-metal)). Each
packed tensor installs the module its scalars name — `RadixLinear`,
`RawLinear` for the raw fallback — after the capability registry has
confirmed Metal support for that tensor. The loader rejects unsupported pack
versions before reading tensors. It also rejects FP8 checkpoints and the
hybrid DeltaNet 27B. CUDA, ROCm, and Metal use the same
on-disk pack format.

The engine defaults to MLX on Darwin/arm64. `--runtime mlx` or
`DRINKME_RUNTIME=mlx` selects it explicitly. `/v1/models` reports
`drinkme.runtime: "mlx"` and `drinkme.computePath`; hardware runs should report
the `fused` path.

Current limitations:

- KV cache is per request. Prefix reuse and persisted slots are unavailable;
  `cached_tokens` is zero and `--prefix-slots` above 1 is refused.
- Sleep/wake returns 501. MTP and n-gram speculation are unavailable.
- `--rope-scaling` is refused.
- No image input: every image part is refused by name
  ([image input](serve.md#image-input) is torch-only).
- No multi-column kernel: M > 1 uses a decoded BF16 transient and `mx.matmul`.
- Packing runs on the CPU through torch. Loading a pack with MLX does not use it.

## Setup

For a new Mac environment, follow the [project setup](../README.md#run-the-server).
First-run bootstrap installs the Metal dependency group: `mlx`, `mlx-lm`, and
PyPI's torch wheel, which `drinkme pack` reads checkpoints through, so `serve`
can pack a model on first use. drinkme's MLX code does not use torch, but
transformers loads it once installed (mlx-lm imports its tokenizer classes),
which adds about 0.2 GB and 0.7 s to the server's start on an M4. An
environment set up before the group included torch gets it from
`uv run --no-sync drinkme bootstrap`; without it, serve a pack copied from
another machine. A pack carries its raw weights and tokenizer and needs
nothing else — see
[the embedded checkpoint](pack-format.md#the-embedded-checkpoint).

Check the runtime before testing:

```sh
.venv/bin/python -c 'import mlx.core as mx; print(mx.__version__, mx.metal.is_available(), mx.device_info())'
```

Metal should be available. Record the device name, memory size, and
`max_recommended_working_set_size`. Use `uv run --no-sync` for subsequent
commands; read [AGENTS.md](../AGENTS.md) before changing an existing environment.

```sh
uv run --no-sync drinkme serve --model Qwen3-4B --pack-dir /path/to/pack --port 3216
curl http://127.0.0.1:3216/health
curl http://127.0.0.1:3216/v1/models
```

If `computePath` is `reference`, check `DRINKME_MLX_PATH` and Metal availability.
The MLX reference compute path may run on either CPU or GPU; it is not the fused-kernel
performance path. If the fit check refuses the load, its message names the
budget that bound and the largest `--ctx` that fits; otherwise choose a smaller
model or close applications ([engine and loading](#engine-and-loading)).

## Engine and loading

[`engine_mlx.py`](../src/drinkme/serving/engine_mlx.py) implements the shared
Engine interface over mlx-lm. Each packed Linear becomes a `RadixLinear` or
`RawLinear` by its tensor's scalars, after the registry has confirmed the
tensor has a Metal path. Embeddings, norms, and ineligible
projections stream through `mx.load`, one tensor evaluated at a time.
The loader preserves tied weights and refuses parameters left uninitialized.
It maps Transformers' `rope_parameters` to mlx-lm's rotary config fields.

When a checkpoint includes bias, every module adds it to the fp32 accumulator
before rounding once to BF16, matching plain `F.linear` and torch's
`CompressedLinear`. The fused GEMV does this in-kernel; dense and reference
paths, `RawLinear`, and `BiasedLinear` use `engine_mlx.biased_matmul`.

`mx.addmm` gives this behavior at M ≥ 2. At M=1, MLX's GEMV rounds to BF16
before adding bias. On M4 with MLX 0.32.2, the cancellation case
`1 + 1/256` with bias `-1` returned `0.0` instead of `0.00390625`.
For a single row, drinkme therefore uses fp32 matmul, requiring a transient
4 bytes per weight. Dense Qwen3 models have no bias and do not incur this cost.

mlx-lm's Qwen3 creates Linears without bias slots regardless of `attention_bias`.
For biased checkpoints, both benchmark arms replace unpacked projections with
`BiasedLinear` as their biases load. `bench/radix_mlx_toy_build.py --bias --reference`
builds a biased toy and a torch reference for Mac tests of both compute paths.
`tests/test_serving_engine_mlx.py` checks the cancellation case at unit level.

The tokenizer, chat template, pack verification, source binding, capability
probe, stop scanning, tool parsers, and checkpoint sampling defaults are shared
with the torch runtime. Sampling reproduces the torch pipeline with float32
NumPy operations and a per-request seeded generator. Structured output uses
the same grammar with a NumPy ban-and-resample loop.

A request's memory on unified memory:

- The prefill applies the output head to the last hidden row only.
  mlx-lm's `Model.__call__` applies it to every prompt position, and MLX
  does not push a later slice back through the matmul, so an 8k prompt to
  Qwen3-4B built 2.2 GiB of logits (prompt × vocab × 2 bytes) to keep one
  row. On MLX CPU the new row is bitwise the old one; on Metal an M=1 head
  may round differently from row T−1 of an M=T GEMM.
- MLX caches every buffer it frees, on Metal up to its memory limit (1.5×
  the working set by default) for the life of the process: an M4 server
  held the 12 GB peak of an 8k request until it exited. The engine calls
  `mx.clear_cache()` after a request's first token, when the prefill's
  transients are free and the fused path's decode never reuses them, and
  again when the request ends, including when the client disconnects. The
  next request re-allocates what it uses, about 0.11 s per GiB on MLX CPU;
  the Metal cost is not yet measured. `DRINKME_MLX_KEEP_CACHE=1` keeps
  MLX's default, for comparing the two.
- Each request logs MLX's own account:
  `[drinkme.engine] mlx memory: <n>-token prompt, peak <x> GiB active;
  cache returned: <y> GiB after prefill, <z> GiB at the end`. The peak is
  `mx.get_peak_memory()`, reset when the request starts.

The fit check uses `fit.py`'s accounting: one memory pool containing
resident weights, one transient tensor, the KV cache and the prefill's
activations at the full context, and headroom (`suggest.FIT_HEADROOM`).
The budget is the smaller of Metal's recommended working set and the memory
macOS would give the process now: `vm_stat`'s free, inactive, speculative
and purgeable pages, plus what MLX already holds. The working set alone is
16 GiB on a 24 GB M4 however much of it other applications use. The serve
picker without `--model` (`packs.hardware_budget`) takes the same minimum,
reading `vm_stat` through `fit.host_available_bytes`, and its one-line
description names the bound: `host available 10.0 GiB, the smaller of mlx
working set 16.0 GiB and host available 10.0 GiB`. The check runs at two
stages:

1. Before allocating weights, `engine_mlx.estimate_load_bytes` reads pack member
   sizes from zip directories and raw tensor sizes from safetensors headers,
   using the loader's BF16 conversion rules. It adds the larger of the biggest
   raw tensor staged during loading and the biggest packed Linear's dense
   M>1 BF16 transient; on the `reference` path, which decodes every plane
   while the graph is built, it adds the sum of the planes. A model
   exceeding the budget is rejected before allocation.
   The benchmark's twin path charges each coded tensor's BF16 weight as
   resident and one tensor's decode as the transient. Those are the planes
   the reference path builds on every forward, held once, so the twin is
   not charged their sum again.
2. After loading, the engine repeats the calculation using actual module storage.

The prefill term is 8 × (hidden + intermediate) bytes per token
(`engine_mlx.prefill_bytes_per_token`). The MLP's live set, measured on MLX
CPU, is 4 × (hidden + intermediate) bytes per token; Metal holds each
command buffer's buffers until the buffer completes, and an M4 measured
7.9 × (hidden + intermediate) bytes per token in an 8k prefill of
Qwen3-4B. For the same reason several dense transients can be alive at
once: a Python variable going out of scope does not free its GPU buffer.
MLX CPU's attention fallback builds the full attention scores, which
Metal's fused kernel does not, so the formula does not describe the CPU.

Boot logs label the stages `fit check (estimate)` and `(measured)` and print
both budgets. A refusal names the one that bound and the largest `--ctx`
that fits; when the host's free memory is the bound, closing applications
also helps. Without Metal, the engine skips the check and reports that it
did so.

<a id="the-codecs-dense-arm-m1-on-metal"></a>

## BF16 kernels

The BF16 codec's Metal kernels are
[The BF16 codec on Metal](#the-bf16-codec-on-metal) below. The raw fallback
(a tensor the codec would have expanded, stored as its BF16 bits) needs no
kernel: `RawLinear` runs the bits through `mx.matmul`.

## The BF16 codec on Metal

[`metal/gemv_radix.py`](../src/drinkme/metal/gemv_radix.py) and
[`metal/dense_radix.py`](../src/drinkme/metal/dense_radix.py) serve the BF16
codec (`codec/radix_pack.py`; the CPU oracle is `codec/radix.py`) two ways:
a fused decode+GEMV at M=1 and a decode-to-BF16 transient for M>1 through
`mx.matmul`. `RadixLinear` in
`serving/engine_mlx.py` holds one resident dict (the block streams, the word
directory, the padded per-tier palette tables) and dispatches on the same
`fused`/`reference` switch; `make_module_mlx` picks it by the tensor's
`codec` scalar. The registry rows `(radix, 0, dense|gemv, metal)` are enabled
on the evidence under [Existing hardware evidence](#existing-hardware-evidence).

The GEMV program is one SIMD group per output row. Each of the 32 threads
holds 32 consecutive weights of a 1024-weight block, so its literals are eight
consecutive words and its tier-0 codes exactly `W0` consecutive words (three
for the sip compression profile) — decoded from registers. The rank of an escape flag (its
position in the next stream) is a popcount within the thread's escape mask
plus a `simd_prefix_exclusive_sum` across the SIMD group. A thread's escaped
lanes hold consecutive ranks, so their codes (and terminal exponents) are
consecutive bits of the next stream: the words holding them are read once,
before the lanes, and each escaped lane takes the next field off a cursor,
with one load per lane only for the rare thread whose codes overrun those
words. Chained tiers (gulp) repeat the scan per tier, and a chained
profile's terminal tier is skipped for a SIMD step none of whose blocks
carries a terminal stream. Gulp's 4-bit tier 2 runs with no branch per lane, as
the ROCm lean gulp decoder does: a thread's lanes are four groups of eight, each
group's codes are the 32 bits of the stream from its first one, and a lane takes
the code at its rank within the group. Gulp's tiers leave each lane a palette
index, and one byte load per lane reads every lane's exponent after tier 2. Each thread uses four fp32 accumulators and a fixed
`simd_shuffle_down` reduction tree. Bias is added in fp32, followed by one
rounding to BF16 using integer operations. `x` is read as BF16 bits. The kernel
supports blocks of 32..1024 lanes and 2..4 tiers and rejects other configurations.
The dense kernel uses the same chunk decoder with a store in place of the multiply.

`bench/radix_bitpin_mlx.py` and `tests/test_metal_radix.py` check all 65,536 BF16
patterns through every compression profile at 32- and 1024-weight blocks. They also cover
ragged and production shapes against a float64 dot product with a
cancellation-aware bound, the fused epilogue, and module dispatch.
With `--fixture`, the gate compares real Qwen3-8B tensors against safetensors
digests; build the fixture on a torch machine with `bench/radix_fixture_build.py`.
The gate's `--selftest` must report FAIL after deliberately flipping a decoded bit.

`bench/radix_metal_timing.py` compares fused GEMV with `mx.matmul` per tensor,
measures bandwidth, and computes the parity report's projection
([results](#existing-hardware-evidence)).

The reference path for a coded tensor decodes on the CPU
(`radix_pack.decode_back_radix`, native when a C++ compiler is present) on
every forward. This reference is intended for correctness checks and is
unsuitable for serving: on an M4 it spent 0.85 s per token on Qwen3-0.6B
with the native decoder, and the numpy decoder that runs without a
compiler is far slower.

The benchmark's twin arm uses a third path, `twin`. `RadixTwinLinear` in
`serving/engine_mlx.py` decodes each coded tensor once, at load, with
[`metal/twin_radix.py`](../src/drinkme/metal/twin_radix.py): the radix
decode in generic MLX ops, vectorised over the tensor's blocks, sharing no
code with the fused kernel. It keeps the BF16 weight and runs the reference
path's matmul over it, so its outputs are the reference path's bit for bit
and it holds the stock arm's 2 bytes per weight. `tests/test_twin_radix.py`
checks the decoder against the CPU oracle on every BF16 pattern, ragged
blocks and every profile, through MLX and through a NumPy stand-in in the
CPU suite. `bench/mlx_twin_bitpin.py --pack <dir>` checks every tensor of a
real pack against the oracle; its `--selftest` must report FAIL.

## Hardware verification

Run on a separate server port. CPU tests alone do not validate Metal kernels.

First build the toy pack, which needs torch, and copy it to the Mac:

```sh
PYTHONPATH=src python bench/radix_mlx_toy_build.py --out /path/to/radix_toy --reference
```

Then, on the Mac (the metal lane does not install pytest; add it with
`uv pip install pytest`):

```sh
DRINKME_RADIX_MLX_TOY=/path/to/radix_toy \
  uv run --no-sync pytest tests/test_metal_radix.py tests/test_serving_engine_mlx.py -v
uv run --no-sync python bench/serve_dialect_smoke.py \
  --base-url http://127.0.0.1:3216 -o smoke-metal.json
```

Check that the pytest run reports no skips: without `DRINKME_RADIX_MLX_TOY`,
the end-to-end tests skip and the run still looks green.

Additional gates:

- `bench/radix_bitpin_mlx.py`: the BF16 codec's kernels (all BF16 patterns, GEMV bound,
  module routes, real 8B tensors with `--fixture`) and a corruption self-test.
- `DRINKME_RADIX_MLX_TOY`: a toy Qwen3 BF16 pack (`model/` + `pack/`) for
  `test_metal_radix.py`'s end-to-end load; the test skips when unset.

Do not loosen tolerances to make a new device pass. For a failure, record the
tensor, seed, MLX version, and device, then compare fused and reference paths
with corrections enabled and disabled. Check the textual verdict and skipped
tests as well as the process status.

### Existing hardware evidence

A sequential timing sweep on an M1 Air drifted 6.51 to 9.32 ms
under thermal load, which is why the timing instruments interleave their
measurements.

Maintainer run on an M4 with 24 GB, MLX 0.32.2: the BF16 codec's gate passed
(all 65,536 BF16 patterns through every compression profile, eight real Qwen3-8B tensors
bitwise the safetensors digests), and the fused GEMV
reached 81–91% of the measured 105 GB/s read bandwidth on the 8B's large
tensors. Whole-model serving on the M4 is measured below and in the published records.

### Benchmark development

`drinkme bench --runtime mlx --model <row>` runs stock mlx-lm, the
twin, and the fused compute path over a BF16 pack. The twin decodes the pack
once at load ([above](#the-bf16-codec-on-metal)), outside the timed passes,
and holds BF16 weights, so it needs the memory the stock arm needs. Before
it did, the twin decoded every tensor on the CPU per step: 165 of the 178
seconds of an M4's Qwen3-0.6B bench. Record the compute path, memory,
minimum/median samples, and drift band; a difference smaller than the
drift band is unresolved.

Remaining work: whole-model measurements of the 8B and larger on a Mac
with enough memory and disk, checking the fit accounting's prefill term against the per-request
`mlx memory` line at 2k, 8k and 16k prompts, and producing comparable
stock-versus-drinkme records with controlled thermal drift.
