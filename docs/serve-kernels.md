# Serving kernels

How `drinkme serve` picks the attention paths and kernels around the compressed
weights, and how to compare each choice against its reference. Operators
need only [serving operations](serve.md) and the boot log; each choice below
prints a `[drinkme]` line at startup, and each has an environment variable
([CLI reference](cli.md#environment-variables)) for A/B runs. None of them changes a
weight byte; several change floating-point order, so near-tie greedy token
choices can differ ([numerical behavior](method.md#numerical-behavior)).

## Single-token attention over the live cache

For full-attention and DeltaNet models, the prefix cache returns only the
rows written so far and reports its changing shape to Transformers. An
unpadded, single-token decode can therefore omit the causal mask: every
returned row is visible to that query. This lets SDPA use grouped KV heads
directly instead of copying K and V for every query-head group. The speed
benefit depends on context length and the installed SDPA backend.
Speculative verification still uses multi-token passes, so requests with
high draft acceptance may see little throughput change.

Padding and multi-token suffixes (including speculative verification) keep
their required masks. Mixed caches containing other static attention layer
types, such as sliding-window layers, keep the masked path;
those layers can still return unwritten capacity before their buffers fill.
Packed weights and KV storage are unchanged. SDPA backend selection can
change floating-point accumulation order and near-tie token choices.

`bench/live_cache_attention_ab.py` compares the masked and unmasked
single-token policies on one loaded model over a private HTTP test port, with resident
prefix caching enabled. It records transcript agreement separately from
timing and supports both serial and default speculative decoding.

## MTP suffix attention

Qwen3.5-family MTP verification and head rebuilding use PyTorch's
`causal_lower_right` bias for short GPU passes (2–8 tokens). Each query sees
the existing cache and its own prefix of the new tokens. This lets native
SDPA keep grouped KV heads instead of Transformers expanding them for an
explicit mask. Ordinary `is_causal=True` is upper-left aligned and would
be incorrect for this cached suffix.

The trunk path requires live full-attention cache layers and the Qwen text
model's unpadded MTP entry point. Other model families, padding, sliding
windows, ordinary static caches, and longer passes retain their existing
attention path. PyTorch selects the SDPA backend and materializes an equivalent
mask if a fused SDPA backend is unavailable. Weight tensors and cache storage
do not change; attention rounding can still change near-tie token choices.

Set `DRINKME_MTP_SUFFIX_SDPA=0` to retain explicit suffix masks for comparison.
`bench/mtp_suffix_attention_ab.py` alternates both paths on one loaded model
over a private HTTP port. `--phases` adds separate synchronized draft,
verification, and cycle measurements; these are excluded from throughput
timing.

## Segmented attention over a long cache

Chunking the queries does not bound attention over a long cached prefix:
each kernel program still reads every cached key. When one chunk's attention
would cover more than 8,388,608 query-key pairs, drinkme splits the cached
keys into segments of 33,554,432 query-key pairs each and merges the partial
results by their log-sum-exp, as flash-decoding does. Segments read grouped
KV heads in place, with no explicit mask. Shorter extends keep the
unsegmented attention path. With the defaults, the longest prefill kernel measured at
27B shapes is about 58 ms at every cached length up to 262,144 tokens.

## DeltaNet recurrence kernel

Hybrid models decode each token through a per-layer recurrence over fp32
state. Qwen3.8-27B uses gated DeltaNet in 48 of its 64 layers. Transformers
looks up the recurrence kernel by name when importing FLA. With the FLA
version shipped here, that name is missing, so Transformers falls back to
about 40 torch launches per layer per token. On gfx1151, this accounted for
6% of the 27B's serial step and 9% of an MTP cycle.

At load time, drinkme selects FLA's fused Triton implementation, which uses
one launch per layer per token, if it passes the startup probe. Example log:

```
[drinkme] deltanet recurrence: fla 0.5.2 fused_recurrent_gated_delta_rule (Triton, one launch per step; probe max |Δ| vs the torch reference 4.8e-07 fp32, 4.4e-03 bfloat16; default — DRINKME_DELTANET_KERNEL=torch for the reference)
```

Before serving, the probe runs both implementations at the model's head shape
in fp32 and in the model's dtype. It checks shapes and dtypes, compares fp32
results against an error threshold, and reports the model-dtype difference
without using it as an acceptance threshold. A compilation, launch, or
validation failure selects the torch reference (Transformers' own recurrence in
torch ops, the ~40 launches above) and logs the reason.

`DRINKME_DELTANET_KERNEL=torch` forces the reference. `=fla` requires the fused
kernel and stops startup if it is unavailable or fails the probe. CPU uses
the reference by default and rejects an explicit `=fla` request.

The implementations use different rounding. The reference normalizes q/k in
the model's dtype (BF16) and reduces with torch `sum` kernels; the fused kernel
normalizes and reduces in fp32 registers. Both read identical weights, but
near-tie greedy token choices can differ. MTP verification calls the selected
recurrence function for each position.

## DeltaNet prefill convolution

On ROCm, large DeltaNet convolution windows use FLA's Triton kernel after a
startup numerical check. It reads the projection's time-major layout directly,
avoiding the expensive contiguous conversion in the torch conv1d fallback.
Windows shorter than 128 rows, including ordinary MTP verification, retain
their existing path. Single-step decode keeps its separate update kernel.

The convolution still rounds to the model's dtype before SiLU, and the model
continues to own all convolution history updates and MTP rollback. The startup
probe compares the model's actual convolution weights in each distinct
shape/dtype against the upstream function, including a cropped history view.
The boot line starts with `[drinkme] deltanet prefill convolution:`.

`DRINKME_DELTANET_CONV=torch` retains the upstream implementation. `=fla`
requires the Triton implementation and refuses startup if the probe fails.
With the variable unset, ROCm uses FLA when its probe passes; CPU and NVIDIA
retain upstream dispatch. Near-tie token choices can differ because arithmetic
order changes, while weight bytes remain identical.

FLA can compile and autotune a new context-size bucket on its first use. The
startup probe warms a small window, so a previously unseen larger bucket can
still add setup latency to one request. Uncached-prefill measurements with
populated kernel caches do not measure that first-use cost.

## Narrow GEMV

On CUDA and ROCm, drinkme uses Triton GEMV at M ≤ 8 for eligible raw BF16
Linears below the codec's row threshold. Larger batches and CPU execution use
`F.linear`. `DRINKME_NARROW_GEMV=0` selects BF16 `F.linear` for A/B measurements;
startup reports the choice as `[drinkme] narrow gemv:`.

Qwen3.8-27B's 48-row `in_proj_a` and `in_proj_b` require 96 launches per token.
On gfx1151, PyTorch's default `F.linear` (rocBLAS) assigns each projection to
one workgroup and takes 120–145 µs per launch.
[`bench/narrow_linear_knobs.py`](../bench/narrow_linear_knobs.py) times the
Triton GEMV against `F.linear` on rocBLAS and on hipBLASLt and against
`torch.mv`, at M = 1 to 8, on every narrow shape in the model menu: the 27B's,
MiMo-V2.6-Distill-Qwen-9B's 32×4096 and Muse-Glimmer-30B's 256×6656 `k_proj`
and `v_proj`. On gfx1151 the Triton GEMV was fastest at every M on every
shape, in about half of hipBLASLt's time and a tenth of rocBLAS's. On the 27B
that is 1.3 ms per token less than hipBLASLt and 12.5 ms less than rocBLAS,
and about the same for each M = 5 verification. The Triton and BLAS results
differed by at most one BF16 ULP in the measured checks because of fp32
reduction order.

## Stock GEMV

A raw BF16 Linear at or above the codec's row threshold makes its one-row
(decode) call through the box's stock GEMV. These are every Linear under
`drinkme serve --stock` and in the bench's stock arm, and the Linears a
pack leaves raw (a tied `lm_head`) where the box's [raw GEMV](#raw-gemv)
is `stock`. The stock arm is the baseline the published ratio divides by,
so the stock GEMV is the fastest one-row BF16 call drinkme has on the box.
On gfx1151 that is `triton`, the twin arm's Triton GEMV over the raw weight
(`radix_ops.gemv_fused` with `RAW=True`, at the twin's launch row for the
shape: the whole row at eight warps), the same kernel as the gfx1151
[raw GEMV](#raw-gemv). Its launch row is the twin's, so
`DRINKME_TWIN_SCHEDULE=served` (the compressed tensor's own row) and a
`DRINKME_RADIX_SCHEDULE` experiment move the stock arm's row as they move
the twin's ([launch schedule](pack-format.md#the-launch-schedule)). On
gfx1102 (RX 7600 XT) it is `triton` too: there the kernel took 0.36-0.97x
rocBLAS `F.linear`'s time on every decode shape of Qwen3-0.6B, 1.7B, 4B and
8B, and made the stock arm decode 1.4x faster on the 1.7B and 4B. gfx1102 has
no twin rows of its own, so it runs at the compressed tensor's row; on the
0.6B, whose decode is host-bound there, it measured 4% slower than
`F.linear`. Every other target keeps `F.linear`, as does every call of more
than one row. `DRINKME_STOCK_GEMV=linear` selects `F.linear`
everywhere, `=mv` forces `torch.mv` with hipBLASLt as the preferred BLAS
library for that call only, and `=triton` forces the twin kernel, on any
accelerator. When the route is on, startup reports it as
`[drinkme] stock gemv:`.

On gfx1151, PyTorch's default `F.linear` goes to rocBLAS (hipBLASLt is not
this build's default for the target), which answers a one-row problem with a
128x128-tile GEMM. [`bench/stock_blas_knobs.py`](../bench/stock_blas_knobs.py)
times every knob that needs no kernel of drinkme's across a model's decode
Linears (rocBLAS or hipBLASLt `F.linear`, `torch.mv` under either, and
PyTorch TunableOp), and beside them the `triton` mode. Among the BLAS knobs,
on Qwen3-0.6B, 1.7B and 8B, `torch.mv` under hipBLASLt was the fastest, at
0.66–0.76 of the bandwidth wall against rocBLAS `F.linear`'s 0.26–0.49, with
no tuning step. TunableOp came second there: 7–24 s of first-run tuning per
model, cached in a CSV per box. It came first only on the 27B's shapes, by
3% after 58 s of tuning. The `triton` mode is faster than `torch.mv` on every
shape of Qwen3-1.7B, Qwen3-8B and MiMo-V2.6-Distill-Qwen-9B (layers 0–3 and
`lm_head`), at 0.63–0.80 of its time per shape. Per token's Linears:

| Model | `torch.mv` (hipBLASLt) | `triton` | Fraction of the wall, `torch.mv` → `triton` |
|---|---|---|---|
| Qwen3-1.7B | 21.95 ms | 15.10 ms | 0.66 → 0.95 |
| Qwen3-8B | 83.00 ms | 63.73 ms | 0.76 → 0.99 |
| MiMo-V2.6-Distill-Qwen-9B | 84.82 ms | 66.24 ms | 0.80 → 1.02 |

Device time (rotation protocol, per-call medians summed) against the
bench's measured 239 GB/s read wall, which the kernel slightly exceeds on
MiMo's shapes. In `drinkme bench` the stock arm's decode went from 34.9 to
47.2 tok/s on Qwen3-1.7B, 10.9 to 13.8 on Qwen3-8B and 10.5 to 13.0 on
MiMo-V2.6-Distill-Qwen-9B, the twin's figure on each. The stock GEMVs are
different kernels, so their results differ in the last bits.

## Raw GEMV

In a codec tree (the compressed arm, the twin, and `drinkme serve` over a
pack) the raw BF16 Linears at or above the row threshold, a tied `lm_head`
or a raw-fallback tensor, make their one-row call through the box's raw
GEMV instead of the stock GEMV. On gfx1151 that is the kernel the twin arm
reads raw BF16 with (`radix_ops.gemv_fused` with `RAW=True`, at the twin's
launch row for the shape: the whole row at eight warps), which is also
gfx1151's stock GEMV. Every other target keeps the stock GEMV, as does every
call of more than one row, the stock arm and `drinkme serve --stock`, which
hold no codec tensor.
`DRINKME_RAW_GEMV=stock` selects the stock GEMV and `=twin` forces the twin
kernel on any accelerator. When the route is on, startup reports it as
`[drinkme] raw gemv:`.

[`bench/raw_head_gemv.py`](../bench/raw_head_gemv.py) times the tied
`lm_head` of Qwen3-0.6B, 1.7B and 4B and gemma-4-31B-it under the stock GEMV,
`F.linear` on either library, the twin kernel across a tiles × warps grid
and the narrow GEMV's kernel. On gfx1151 the twin kernel at its own row took
0.84, 0.67, 0.93 and 0.94 of `torch.mv`'s time, within 2% of the grid's best
on each; hipBLASLt `F.linear` took 0.88–0.96. The instrument also checks
that the routed Linear's one-row output is bitwise the twin module's over
the same weight, and that every other row count and the `stock` mode are
bitwise the stock route's. The twin kernel and `torch.mv` reduce in a
different order, so the logits differ from the stock GEMV's in the last
bits, within the float64 reference bound.

These selections, the [DeltaNet recurrence](#deltanet-recurrence-kernel), and
the [prefill convolution](#deltanet-prefill-convolution) are made by one routine
(`serving/kernel_route.py`) for `drinkme serve`,
`drinkme serve --stock`, and every arm of `drinkme bench`, which refuses a record
whose arms were routed differently ([kernel routing](bench.md#kernel-routing)).

## CUDA graphs

On NVIDIA an eager decode step is limited by the host: a Qwen3-8B token
launches about 2,000 kernels. On CUDA, `drinkme serve` and every arm of
`drinkme bench` capture the decode step once per cache as a
`torch.cuda.CUDAGraph` and replay it. Speculative verify steps are captured
too, one graph per width. Prefill stays eager. The boot log reports
`[drinkme] decode step:` with the mode. `DRINKME_CUDA_GRAPHS=0` forces eager.
ROCm, Metal and the CPU never capture a graph.

Graph mode is on only for the model families and modes on
`serving/cudagraph.py`'s `VERIFIED` list. A family goes on the list once
`bench/cuda_graph_gate.py` has shown three things on a CUDA card: graph
replay is bitwise equal to the same step run eagerly, the compressed arm
matches stock token for token in graph mode, and the step is no slower than
the eager step at 4k and 16k live tokens. The list is keyed by family, not
by compression profile: each family on it passed on a sip pack and on a
gulp pack, the two profiles `drinkme pack` writes (`GATED_PROFILES` in
`serving/cudagraph_fit.py`). A family that is not on the list,
a cache layout the step cannot take (sliding-window layers), and a failed
capture each decode eager, with one line saying so. A capture failure
never fails a request or a bench run.

The captured step reads the whole KV allocation. Attention gets the live
length as a device tensor through the `seqused_k` argument of torch's flash
kernel, so it reads only the written rows and handles grouped-query heads
without copying K and V. A masked `StaticCache` would attend across the
whole allocation, which is the cost `serving/kvcache.py` avoids. The prefix
cache's rewinds, context checkpoints and slot restores write the same
tensors in place, so a slot's graphs survive them. Sleep drops the slots
and their graphs with them.

All of a cache's graphs share one memory pool, and a DeltaNet hybrid's
verify graphs write each row's recurrent state into one buffer per cache.
A cache's graphs then hold the pool (at most 98 MiB measured) plus, on a
hybrid, 144 MiB per verify row but the last: 674 MiB for Qwen3.8-27B at MTP
depth 4, where one private pool per graph held 2.4 GiB. The server captures
every step a request can replay right after its prefill, so the warm-up
request captures them for the first slot and no width is captured
mid-stream. `serving/cudagraph_fit.py` owns the arithmetic. The load check
charges it per cache beside the prefix slots, and graph mode that would
leave less than the fit check's 10% margin decodes eager with a line
saying so. The serve picker and bench's fit point charge the same bytes on
an NVIDIA card. `DRINKME_CUDA_GRAPHS=0` charges nothing.

Against the eager step, graph mode changes the attention kernel (the flash
kernel over the allocation instead of SDPA over the live window). Near-tie
greedy tokens can therefore differ, the class described in
[numerical behavior](method.md#numerical-behavior). Bench records carry the
mode in the engine string's `decode` segment
([the engine string](bench.md#the-engine-string)), and both arms of a run
take the same step path.

## Attention backend on CUDA

On an H100 (sm90) torch picks cuDNN attention, which builds an execution
plan for every new sequence length and caches it per thread. `drinkme
serve` runs each request on a fresh HTTP handler thread, so an eager
decode paid a plan build at every step: every step is a new length.
Qwen3-8B served eager at 21–41 tok/s, against 71–80 with cuDNN excluded.
Prefill on a fresh thread was slower with cuDNN at 1k, 4k and 16k prompt
tokens on Qwen3-8B and Qwen3.8-27B. On CUDA drinkme therefore turns
torch's cuDNN SDPA flag off for the process (`sdpa.route`, called from
`kernel_route.route_kernels`, so serve and every bench arm run the same
attention), and torch takes flash, then mem-efficient, then math, for
decode and prefill alike. The boot log says `[drinkme] attention: cuDNN
SDPA excluded on CUDA`. ROCm, Metal and the CPU keep torch's choice. On
an L4 torch offers cuDNN attention but picks flash at Qwen3-8B's decode
and prefill shapes, so nothing changes there.

The captured CUDA-graph steps and the MTP suffix bias never took cuDNN,
so graph-mode decode is unchanged; eager decode, prefill and the MTP head
change backend. That changes floating-point order on H100, so near-tie
greedy tokens can differ ([numerical behavior](method.md#numerical-behavior)).
`DRINKME_CUDNN_SDPA=1` keeps torch's choice for A/B runs, everywhere: the
MTP head's attention pays the plan builds again too, and the served
Qwen3.8-27B drops from 68 to 19 tok/s in graph mode;
`bench/serve_fixes.py --what ttft` measures prefill both ways on a fresh
thread.

## Kernel routes by name

Each of the five kernel routes has one name per surface:

| Route | Environment variable | Boot line | `environment.engine` segment | `raw.<arm>_routing` keys |
|---|---|---|---|---|
| [DeltaNet recurrence](#deltanet-recurrence-kernel) | `DRINKME_DELTANET_KERNEL` | `[drinkme] deltanet recurrence:` | `deltanet` | `deltanet_kernel` |
| [DeltaNet prefill convolution](#deltanet-prefill-convolution) | `DRINKME_DELTANET_CONV` | `[drinkme] deltanet prefill convolution:` | `conv` | `deltanet_conv` |
| [Narrow GEMV](#narrow-gemv) | `DRINKME_NARROW_GEMV` | `[drinkme] narrow gemv:` | `narrow-gemv` | `narrow_gemv`, `narrow_count` |
| [Stock GEMV](#stock-gemv) | `DRINKME_STOCK_GEMV` | `[drinkme] stock gemv:` | `stock-gemv` | `stock_gemv` |
| [Raw GEMV](#raw-gemv) | `DRINKME_RAW_GEMV` | `[drinkme] raw gemv:` | `raw-gemv` | `raw_gemv` |
