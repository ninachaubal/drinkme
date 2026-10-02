# Pack format

A pack is self-contained: besides eligible Linear weights and metadata, it
carries an [embedded checkpoint](#the-embedded-checkpoint) — the source
checkpoint's config, tokenizer, generation defaults, and the tensors the
codec does not pack (embeddings, norms, biases, ineligible projections) — so
it loads with no source checkpoint and no network. One pack serves every
platform: CUDA, ROCm, and Metal read the same files ([Apple
silicon](metal.md)). The NumPy codec is torch-free at import; model-level
packing uses torch to read checkpoints. The source checkpoint is needed only
to write a pack (`drinkme pack`) and by `drinkme bench`'s stock and twin
comparison arms, which measure against the uncompressed weights on purpose
([below](#the-embedded-checkpoint)); once a pack exists, serving and
`verify` never touch it, so it can be deleted.

This is **pack format 1** (`formatVersion: 1` in `meta.json`; `format_version`
on every loaded tensor dict). The BF16 codec
uses exponent tiers and stores the sign and mantissa unchanged, with decoding
inside the GEMV kernel. See [method](method.md#the-method). The two compression
profiles are **sip** (the default, faster) and **gulp** (smaller).
`meta.json` records the choice as `profile: "sip"` or `profile: "gulp"`.

All pack entry points reject missing or unsupported format versions. This
includes `iter_pack_dir`, `load_pack_dir`, the pack menu, `serve`,
`bench --pack-dir`, and `verify`. The serving loader checks the version before
hashing the files or resolving a snapshot. Packs must be rebuilt; there is no
migration. Each entry point returns the same error:

```text
<dir>: unsupported pack format <n> (this drinkme reads format 1); re-pack with `drinkme pack --model <repo> --replace`
```

where `<n>` is the version seen (`(none declared)` when the field is missing)
and `<repo>` is the pack's own `hfRepo`.

## Per tensor

Every tensor is one `.npz` whose `scalars` blob names its kind. A pack holds
two kinds:

| Kind | Discriminator | Arrays | Scalars |
|---|---|---|---|
| coded | `codec: "radix"` (the tensor-encoding tag: an exponent-radix coded tensor) | `rx_palette` (uint8, the exponents in code order, unpadded), `rx_offsets` (uint32 word directory, one per 1024-weight block plus a final sentinel), `rx_data` (uint32, the concatenated block streams) | `R`, `C`, `bpw`, `profile`, `widths`, `block_size` (1024) |
| raw | `codec: "raw"` | `raw_bits` (uint16 `[R, C]`, the BF16 bit patterns verbatim) | `R`, `C`, `bpw` (16.0) |

**Source dtype.** Packed tensors preserve the checkpoint's BF16 bits without
casting. `drinkme check`, packing, and loading reject a checkpoint if any otherwise
eligible tensor uses another dtype, naming the tensor. This includes FP16 and
FP32 releases and BF16 releases with a single wider eligible Linear. Tensors
outside the packed set (embeddings, norms, biases, and small projections) may
use wider dtypes; the loader converts them to BF16 as
`from_pretrained(dtype=bf16)` would.

A **coded** block is 1024 consecutive weights of one row. Each block stores the
sign+mantissa bytes of its weights literally, then one code stream per
exponent tier: a tier of width *w* has 2^*w* − 1 palette codes for the most
frequent exponents and one escape code to the next tier; the last tier stores
the 8-bit exponent itself. The `widths` tuple is the compression profile; the per-tensor
palette orders the exponents by frequency. Every bit pattern — signed zeros,
subnormals, inf, every NaN payload — round-trips, and every block decodes
independently (the directory is what lets one program own one block, or one
row).

### Compression profiles

Choose one compression profile when packing. `meta.json` stores it as `profile` and
`profileWidths`, and each coded tensor's manifest descriptor repeats it.
Re-encoding a tensor with another compression profile fails both the manifest and file-hash
checks.

| Compression profile | `widths` | Qwen3-8B, weighted bpw | Qwen3-8B resident bytes | How to select | Default pack directory |
|---|---|---|---|---|---|
| sip (default) | (3, 8) | 11.374 | 12,006,244,888 | `drinkme pack --sip`, or nothing | `<org>--<model>@<rev>` |
| gulp | (2, 2, 4, 8) | 10.903 | 11,575,275,304 | `drinkme pack --gulp` | `<org>--<model>-gulp@<rev>` |

Sip has two tiers and no per-block schedule. Gulp has four tiers and uses
about 4% fewer bytes. In maintainer measurements, sip was the faster compression profile
at M=1 on the bandwidth-bound devices tested (Strix Halo, L4, A10G), and
gulp's decoding cost was better absorbed by L40S and H100. On a 48 GB card both
fit Qwen3.8-27B (gulp 37.8 GB resident, sip 39.2 GB), and sip was faster. Use
gulp when sip exceeds the available memory.

The two flags are mutually exclusive. `serve`, `verify`, and the MLX bench
resolve a model to sip's directory (the torch bench re-packs in memory unless
given `--pack-dir`); use `--pack-dir` to serve or bench a gulp pack.

The stored `bpw` covers the payload. Device allocations also include padded
per-tier lookup tables, a zero-filled buffer after the streams, and, for
compression profiles with more than two tiers, a 2-byte-per-block schedule. The schedule
contains nonterminal stream lengths computed at load time. The zero-filled
buffer is as large as the widest possible block; see
[verification](#verification).

`residentBytes` in `meta.json` counts the loader's runtime arrays and all
weights loaded raw.

### Raw fallback

If either the stored or device representation would reach the size of the raw
BF16 tensor, the encoder stores that tensor raw. `swap.RawLinear` serves it
through plain `F.linear` on every route. `meta.json` records
`rawFallbackTensorCount` and `rawFallbackBytes`. No tested real checkpoint has
needed this fallback (0 of 253 tensors on Qwen3-8B at every compression profile); an
adversarial toy in the test suite exercises it.

`meanBpw` is the unweighted average of per-tensor costs. `weightedBpw` is
total payload bits divided by packed weight count. Both cover the trunk's
packed population — coded and raw-fallback tensors; they exclude raw weights
and the separately accounted `mtp/` head. Use the weighted figure when
estimating the packed population's size. Outside `meta.json` they go by the
[lexicon](../lexicons/README.md)'s names: `weightedBpw` is `bitsPerWeight`
and `meanBpw` is `meanTensorBitsPerWeight`, in a measurement record's
`compression` block and under `drinkme` on `/v1/models`.

## `meta.json`

| Field | Contents |
|---|---|
| `formatVersion` | 1 |
| `hfRepo`, `revision` | the caller's coordinates: the repo and the revision the pack directory is named for (`source`, below, is the resolved identity) |
| `profile`, `profileWidths`, `blockSize` | compression profile (`sip` or `gulp`), tier widths, and block size; the profile also appears in `/v1/models` as `drinkme.compressionProfile`, boot logs as `profile`, and benchmark records as `compression.profile` |
| `dtype`, `sourceDtype` | `bf16`; `sourceDtype` records the dtype read by the packer, which rejects other dtypes for eligible tensors |
| `tensorCount`, `meanBpw`, `weightedBpw` | the packed population, a vision tower's Linears included; `meanBpw` = `meanTensorBitsPerWeight` and `weightedBpw` = `bitsPerWeight` in the record and on `/v1/models` |
| `radixTensorCount`, `radixBytes` | coded tensors and their npz bytes |
| `rawFallbackTensorCount`, `rawFallbackBytes` | the raw fallbacks |
| `rawTensorCount`, `rawResidentBytes` | tensors the loader streams raw, from the embedded checkpoint |
| `packedBytes`, `residentBytes` | the npz sum; the measured device footprint plus the raw remainder |
| `radixEncoder`, `radixEncodeSeconds`, `packSeconds` | `native` (the C++ encoder, JIT-built) or `numpy`; both write the same bytes |
| `tensors`, `sha256`, `tensorInfo`, `manifestSha256`, `source` | the file hashes and manifest, below |
| `embedded` | the embedded checkpoint: `files` (file name in `checkpoint/` → SHA-256), `tensorCount`, `tensorBytes`, `bytes`; required — a pack missing it is refused at load ([below](#the-embedded-checkpoint)) |
| `mtp*` | the MTP head sub-pack's headline numbers, when the checkpoint has a head |
| `vision` | the vision tower's share of the pack, when it carries one: `tower`, `paths`, `imageTokenId`, `tensorCount`, `packedTensorCount`, `residentBytes` ([below](#the-vision-tower)) |

The loader ignores unknown keys.

## The launch schedule

At load time, the decoder selects blocks per program and warps for each tensor
from [`codec/radix_schedule.py`](../src/drinkme/codec/radix_schedule.py).
The table is keyed by box class (`box_class()`), compression profile family, and
tensor shape class. The box class is `cuda` on NVIDIA and an AMD target on
ROCm, identified through `gcnArchName`: `gfx1151` or `gfx1102`, which have
their own rows, and `gfx1151` for any other AMD target.
The schedule is part of the code, so changing it does not require re-packing.

| Class | Rule | Qwen3-8B tensors | sip on gfx1151: M=1 (blocks/program, warps) | sip on cuda: M=1 | sip on gfx1102: M=1 | sip: mc (every box class) | scheduled (gulp) on gfx1151 and cuda: M=1 | scheduled on gfx1102: M=1 | scheduled: mc |
|---|---|---|---|---|---|---|---|---|---|
| head | R ≥ 32768 | `lm_head` 151,936×4,096 | whole row, 1 | whole row, 4 | 1, 1 | 1, 1 | 1, 1 | 1, 2 | 1, 1 |
| narrow | R ≤ 1024 | k/v_proj 1,024×4,096 | whole row, 1 | whole row, 4 | whole row, 2 | 1, 1 | 1, 1 | 1, 2 | 1, 1 |
| long | ⌈C/1024⌉ ≥ 8 | down_proj 4,096×12,288 | whole row, 1 | whole row, 4 | 1, 1 | 1, 1 | 1, 1 | 1, 2 | 1, 1 |
| wide | R ≥ 8192 | gate/up_proj 12,288×4,096 | whole row, 1 | whole row, 4 | 1, 1 | 1, 1 | 1, 1 | 1, 2 | 1, 1 |
| square | else | q/o_proj 4,096×4,096 | whole row, 1 | whole row, 4 | 1, 1 | 1, 1 | 1, 1 | 1, 2 | 1, 1 |

The gfx1151 rows come from rotation sweeps on gfx1151. CUDA sip rows use the
same sweep on L4, A10, A10G, L40S, and H100. RX 7600 XT sweeps cover both sip
and gulp. These rows come from maintainer rotation sweeps
(`bench/parity_rotation.py`); the ratios below explain the choices and are not
published results.

On gfx1102, sip is fastest with one block per program and one warp in four
shape classes. The narrow class is fastest with a whole row and two warps.
The gfx1151 configuration takes 1.04–1.08× as long per class. For gulp, one
block and two warps wins every class; the shared scheduled configuration is
5% slower. Other AMD targets use the gfx1151 rows until measured separately.

On gfx1151, sip's whole-row program with one warp takes 0.90–0.95× the time
of the `spike` configuration (one block per program, two warps). On the tested
NVIDIA cards, the whole-row program is fastest with four warps (on the L4, two,
with four within 1–3%) and slowest with one. The reason for this difference is not established. Triton uses 32-lane
warps on gfx10+, including gfx1151, so these configurations assign different
numbers of lanes per row; the choices are based on measured minima.

For compression profiles with a per-block schedule, whole-row programs are 1.3–2.3× slower
than one-block programs on gfx1151 under the scheduled decoder. Under the lean
gulp decoder the whole row at one warp no longer spills and is within 1–3% of
one block per program (2% faster on the head class), so gulp's row stays one
block at one warp. This is why the table has two compression
profile families. On NVIDIA the scheduled family uses the gfx1151 row. A
graph-mode sweep of Qwen3-8B gulp under `_decode_gulp_lean` on an L4 and an
L40S put one block at two or four warps, the whole row at one, two or four
warps, and two blocks at two warps between 0.93× (one block at two warps,
L40S; 0.99× on the L4) and 1.09× of that row's step time. The row stays
until the scheduled family is swept under `_decode_gulp_lean_cuda`.

For A/B runs, `DRINKME_RADIX_SCHEDULE` accepts `table`, `table=gfx1151`,
`table=gfx1102`, `table=cuda`, `spike`, `tiles=…,warps=…`, a per-class JSON
object, or `@file`. The boot log reports the row and box class used for
each family.

On a CUDA build, sip tensors whose rows are whole 1024-weight blocks decode
through a leaner decoder (`radix_kernel_gpu._decode_sip_lean`), on by
default: the same bits as the scheduled decoder from fewer instructions per
weight. It decodes each thread's eight lanes together, ranks their escapes
with integer arithmetic on 3-bit fields rather than per lane, places palette
entries and terminal bytes with shifts, and builds the FP32 weight directly.
`bench/radix_sass_count.py` counts 25.25 instructions per weight per thread
in the sm_89 GEMV loop, against 43.5 for the first lean decoder and 49.6 for
the scheduled one. On NVIDIA the sip decode is bound by instruction issue
rather than memory: under the first lean decoder an L4 held its 72 W cap at
about 1.1 GHz, and an H100 read at 0.20 of its bandwidth. That first lean
decoder took 0.93× the scheduled decoder's GEMV time per shape class on an
L4, and `drinkme bench` on Qwen3-8B went from 13.51 to 14.55 compressed
tok/s. Ragged rows keep the scheduled decoder, because the twin is bitwise
the scheduled kernel there and not the lean one. ROCm builds keep the
scheduled decoder for sip, which is not yet measured against the lean one on AMD.

On a ROCm build, gulp tensors whose rows are whole 1024-weight blocks decode
through a lean gulp decoder (`radix_kernel_gpu._decode_gulp_lean`), on by
default: the same bits as the scheduled decoder. It decodes each thread's
eight lanes together. Per tier, it reads the eight lanes' codes from one
32-bit window of the stream, keeps their escape flags as one bit per nibble,
gets each lane's rank and the group's total from one multiply, and runs one
prefix sum per block over one element per group. On gfx1151 the scheduled
decoder's gulp GEMV at one block per program and one warp used all 256
VGPRs and spilled to scratch, about 150 AMDGCN instructions per weight per
thread; the lean one compiles to about 65 with no spills (sip's scheduled
GEMV: 58). On Qwen3-8B it took 0.67–0.73× the scheduled decoder's GEMV time
per shape class, and `drinkme bench` went from 10.80 to 14.91 compressed
tok/s (MiMo-9B 10.28 to 14.20, Qwen3-1.7B 38.8 to 48.3). Sip stayed faster
(Qwen3-8B 16.04 tok/s). Ragged rows keep the scheduled decoder for the same
reason as sip's on CUDA, measured on gfx1151.

On a CUDA build the same rows decode through `radix_kernel_gpu._decode_gulp_lean_cuda`,
on by default: `_decode_gulp_lean`'s bits with fewer SASS instructions. Its
nibble prefixes are 32-bit, each lane's rank is pre-scaled so one funnel
shift takes its field, tiers 0 and 1's palettes are byte-reversed so one
shift places the entry at the exponent's bits, and a lane that stops at a
tier reads a forced escape code and a zero terminal byte at every later
tier, so the tiers OR together without selects. At the CUDA table's
scheduled row (one block per program, one warp) the sm_89 GEMV compiles to
51.0 instructions per weight per thread, against 68.25 for
`_decode_gulp_lean` and 122.0 for the scheduled decoder. In graph mode,
`drinkme bench --pack-dir` on Qwen3-8B gulp went from 6.86 to 14.46
compressed tok/s on an L4 (stock 15.93), from 25.9 to 51.9 on an L40S
(stock 42.0) and from 18.1 to 51.0 on an H100 (stock 96.6), and
Qwen3.8-27B gulp on an L40S, where only the compressed arm fits, from 7.86
to 16.31. Ragged rows keep the scheduled decoder.
`DRINKME_RADIX_DECODER=scheduled` or `=lean` selects one for A/B runs.

The bench's order-matched twin (the same kernels reading raw BF16 in place of
the decode, [bench.md](bench.md#torch-benchmark-scope)) has its own rows
where they have been measured, `radix_schedule.TWIN_TABLES`. They affect only the twin's M=1 GEMV;
its multi-column rows are the compressed tensor's. On gfx1151 the whole row
at eight warps was fastest for raw BF16 in every shape class of Qwen3-0.6B, 1.7B,
8B and the 27B's shapes, where sip's row (one warp) reads raw BF16 1.05–1.6×
slower: a raw load has no decode to bind the program on compute. The same
sweep (`parity_rotation.py --radix-arms sip twin`) left sip's own rows
unchanged. Box classes without twin rows run the twin at the compressed row,
and so does `DRINKME_TWIN_SCHEDULE=served` for A/B runs. The twin gate
(`bench/radix_twin_bitpin.py`) checks the compressed kernels at the twin's
rows too, so at any row the twin runs, it is bitwise the compressed kernel
at that row.

## Verification

Verification has five distinct parts:

1. **Source equality:** at pack time, the encoder decodes each tensor back
   and compares its bits with the source (the native encoder with its own
   decoder, the NumPy encoder with `radix.decode`). Failed tensors are not
   written.
2. **File integrity:** record each file's SHA-256 and a canonical manifest
   binding tensor name, file, hash, shape, dtype, codec descriptors (compression profile,
   widths, block size; `codec: raw`), and source identity. `manifestSha256` hashes that manifest. Loading rechecks hashes
   and rejects mismatches; it does not repeat the source comparison.
3. **Raw-source binding:** raw weights come from the checkpoint identified by
   `source` — a Hub commit, or a local directory's digest, covering headers,
   config, and tokenizer, not its weight payload bytes. At pack time they are
   copied from it byte for byte into the embedded checkpoint, read back and
   compared, and then covered by the file hashes; the embedded config,
   tokenizer and shard headers must recompute `source.digest`.
4. **Kernel correctness on each platform:** device acceptance gates compare decoded weights
   and kernel computations against their references —
   `bench/radix_gemv_bitpin.py` (every BF16 bit pattern, every compression profile, the
   fused epilogue), `bench/radix_mc_bitpin.py` (the multi-column arm against
   a float64 oracle), `bench/radix_schedule_bitpin.py` (every launch-table
   row). CPU tests do not execute the GPU gates.
5. **Upstream equality:** for a pack of a Hub commit, every source weight
   file rebuilt from the pack hashes to the SHA-256 the Hub publishes for
   it ([upstream verification](#upstream-verification)). `drinkme serve`
   runs it on a downloaded pack; `drinkme verify --upstream` on any pack.

All loaders call `iter_pack_dir` to validate tensor descriptors. Before reading
arrays, it checks stream sizes against `R × C`. For coded tensors it then
checks the palette, widths, and directory: offsets start at zero, are
monotone, end at the payload boundary, and describe one entry per block
within the size bounds below. Serving requires 1024-weight blocks, at most
four tiers, and nonterminal widths of 1–4 bits. Raw tensors are shape-checked.

### Hash and validation limits

The pack stores its own hashes and manifest in `meta.json`, alongside its
tensor files. Matching hashes establish file integrity, but an externally
generated pack can have valid hashes over malformed data. Descriptor validation is still required.

Without reading payload words, `radix.validate` checks each block's extent
against `radix.block_word_bounds`. A block must hold at least its fixed streams:
`ceil(n / 4)` literal words and `ceil(n · w₀ / 32)` first-tier words for `n`
weights. Its maximum size allows every stream to be as wide as possible.
For 1024 weights, sip permits 352–608 words and gulp 320–768. Invalid extents,
directory entries into neighboring blocks, and blocks past the payload are
rejected at load.

These checks cannot establish whether data-dependent streams fit their block.
The lengths of later tiers and terminal exponents depend on how many weights
escaped, which requires decoding. Startup does not perform a whole-pack CPU
decode. The CPU decoders report `truncated radix stream` for these malformed
blocks; device readers can return invalid weights without a fault.

To bound those reads, device allocations include zero words after the streams,
enough for the widest block. Readers can reach at most that many words from a
validated block start. `swap.to_device_radix` and `metal/gemv_radix.pad_words`
allocate 4 × 608 bytes per sip tensor or 4 × 768 per gulp tensor. This padding
is included in `residentBytes` and adds about 0.005% (sip) to 0.007% (gulp)
to a Qwen3-8B pack.
Kernels do not check bounds on each load: adding a block-end comparison to
just the terminal read increased sip GEMV time by 2% on an RX 7600 XT.
`tests/test_radix_bounds.py` covers the adversarial cases.

Every pack records its hashes when written. Missing per-file hashes or a manifest digest
indicate an externally written or modified pack. Loading rejects it with
`not verifiable` or `no manifest` and a re-pack command. `drinkme verify` rechecks
file hashes and the manifest against the current metadata without loading a
model or rewriting `meta.json`.

## Upstream verification

Source equality is established on the packer's machine, and a pack's file
hashes prove only that it is the pack its packer wrote. Upstream
verification checks a pack against the checkpoint's publisher instead. The
Hugging Face Hub publishes the SHA-256 of every LFS file at every commit,
and every safetensors shard is an LFS file. A pack holds everything a shard
is made of: its header bytes (`checkpoint/source-identity.json`), its
packed Linears, and every other tensor
(`checkpoint/remainder.safetensors`). `upstream.verify_upstream` rebuilds
each shard the source identity names: the 8-byte header length, the header,
then each tensor's bytes in header order, a packed tensor decoded to its
BF16 bits and any other copied. It streams the result into SHA-256
without writing it out, and compares the digest with the one the Hub lists
for that file at `source.revision`. The embedded small files are compared
too: an LFS file by its SHA-256, any other by its git blob id.

Each file gets one verdict:

| Verdict | Meaning |
|---|---|
| `MATCH` | the rebuilt file hashes to the digest the Hub lists for it |
| `MISMATCH` | it does not; or a pack tensor has another shape or dtype than the header gives it; or an embedded file differs from the Hub's copy or is not in the repository at that commit. The file is named |
| `NOT COVERED` | the pack cannot rebuild the file: a shard holds a tensor the pack does not carry (every one is named), the header leaves bytes outside every tensor, the Hub lists a top-level weight file the pack records no header for, or the Hub lists no LFS hash for it |

A pass is MATCH for every file. `drinkme verify --upstream` exits 0 only
then, 3 for any MISMATCH or NOT COVERED, and 4 when the Hub cannot be asked.

A tied output embedding is the one tensor a pack leaves out that a
checkpoint may still ship: Qwen3-0.6B and Qwen3-1.7B carry a 311 MB copy of
`lm_head.weight` ([the embedded checkpoint](#the-embedded-checkpoint)).
The rebuild fills it from the input embedding it is tied to, and the file's
hash then shows whether the checkpoint's copy is byte-identical; the MATCH
line says the tensor was rebuilt that way. A vision tower drinkme does not
serve for an architecture is not carried, and a shard holding one is NOT
COVERED, naming its tensors.

Decoding runs on the CPU: the native decoder on up to eight threads when it
is built, NumPy's otherwise. The next tensor decodes while the current one
hashes. Measured 09-29 on a Ryzen AI Max+ 395 (8 threads, sip packs in the
page cache), every file MATCH:

| Model | Weight files | BF16 bytes | Wall time |
|---|---|---|---|
| Qwen3-0.6B | 1 | 1.50 GB | 1.6 s |
| Qwen3-1.7B | 2 | 4.06 GB | 2.5 s |
| Qwen3-8B | 5 | 16.38 GB | 10.9 s |
| Qwen3.8-27B | 18 | 55.56 GB | 47.1 s |

At the 27B's 1.18 GB/s, Qwen2.5-72B-Instruct's 145.4 GB would take about
two minutes; that one is an estimate, not a run. NumPy's decoder, on a
machine that cannot build the native one, took 21.9 s on Qwen3-0.6B, 14
times as long.

### The receipt

A run writes `upstream-verified.json` into the pack directory: the repo and
commit, each file's verdict with the Hub's digest and the rebuilt one, the
overall verdict, the drinkme version, the time and decoder, and the pack's
manifest digests (the trunk's and `mtp/`'s), recomputed from `meta.json`.
A receipt is current while it records MATCH and those digests are
unchanged. They cover every file hash, and the load-time hash check ties
the files on disk to them. `drinkme serve` reads the receipt so a
downloaded pack is checked once rather than at every serve. The receipt is
not in the manifest or the file hashes, and neither is `hub-pack.json`,
which records where a downloaded pack came from. A published pack's own
receipt stays in the pack repo: `serve` does not download it and writes
its own ([published packs](serve.md#published-packs)).

## Source identity

`meta.json` records a source descriptor:

```json
{
  "identityVersion": 1,
  "kind": "hub",
  "repo": "Qwen/Qwen3-8B",
  "revision": "b968826d9c46dd6066d109eabc6255188de91218",
  "digest": "1151b8c0…"
}
```

For `kind: "hub"`, revision is the resolved snapshot commit. For `kind: "local"`,
revision is null. In both cases, the digest covers safetensors headers (names,
dtypes, shapes, byte ranges), config, tokenizer files and the image processor
configs (`preprocessor_config.json`, `processor_config.json`) — it is not a checksum of the weight bytes. A local
weight file edited in place without header changes is not covered by this
identity check; a hub checkpoint's identity rests on its resolved commit.
In benchmark records produced with `--pack-dir`, and in every MLX record,
`raw.verified_tensors` counts tensors whose hashes matched at load.
`raw.verification: "pack"` means the run relies on the pack-time source
comparison. Only the in-memory torch arm repeats that comparison and records
`"roundtrip"`.

A pack loads config, tokenizer, raw weights, and the MTP head's raw tensors
from its embedded checkpoint, which is checked against this identity.
An explicit model or revision resolving to another identity is refused before
weight allocation. A moving Hub branch does not change the bound snapshot.
`drinkme pack` always records `source`, and a pack without it also has no
embedded checkpoint, so a normal load refuses it outright ([below](#the-embedded-checkpoint)).
`drinkme bench`, which always needs the real checkpoint, resolves such a
tool-written pack's raw tensors from the caller's coordinates instead,
unpinned, with a warning.

## The MTP head sub-pack

A checkpoint with a trained MTP draft head gets that head's eligible Linears
packed into `<pack>/mtp/` using the trunk's compression profile, raw fallback rules, writer,
and hashing procedure. It has its own `meta.json` (`profile`, `profileWidths`,
`radixTensorCount`, `rawFallbackTensorCount`, the measured `residentBytes`).
The trunk's `meta.json` echoes the head's headline numbers. See
[speculative decoding](serve-speculation.md).

## The vision tower

A vision model's pack carries its vision tower beside the text model: every
subtree the tower spans, at the path the checkpoint names it by. The tower's
Linears pass the same eligibility rule as the text model's and are coded by
the same codec at the same compression profile; the rest of it (the patch
embedding where it is not an eligible Linear, position tables, norms,
biases) goes into the [embedded checkpoint](#the-embedded-checkpoint) raw.
Its coded tensors are in the tensor map, the hashes and the manifest like
any other, and the tower counts in the pack's `tensorCount`, bits per weight
and `residentBytes`. `meta.json`'s `vision` block records its share:

```json
"vision": {
  "tower": "model.visual",
  "paths": ["model.visual"],
  "imageTokenId": 248056,
  "tensorCount": 333,
  "packedTensorCount": 110,
  "residentBytes": 654194412
}
```

| Field | Contents |
|---|---|
| `tower` | the path of the ViT itself |
| `paths` | every subtree the tower spans, the ViT first: `model.visual` (Qwen3.5); `model.vision_tower`, `model.embed_vision` (gemma-4); `model.vision_tower`, `model.vision_adapter`, `model.vision_projection` (Muse-Glimmer). A path can name a single Linear |
| `imageTokenId` | the chat template's image placeholder id, for reference; the loader reads `config.json`'s |
| `tensorCount`, `packedTensorCount` | the tower's tensors, and how many of them are coded (Qwen3.5 333 and 110, gemma-4 356 and 190, Muse-Glimmer 809 and 304) |
| `residentBytes` | the tower's share of the pack's `residentBytes`: what the server holds for it, and what `DRINKME_VISION=0` leaves out of the fit |

The block is what makes a pack serve images. A pack of a model with a tower
and no `vision` block is refused at load, naming the re-pack command and
`DRINKME_VISION=0` ([image input](serve.md#when-images-are-refused)). The manifest binds the
tower's tensors through the tensor map and `embedded`; the block itself is a
summary of them and is not in the manifest. With `DRINKME_VISION=0` the loader
builds no tower and skips its pack tensors by name, never opening their
files; they are hash-checked like the rest.

The image processor configs (`preprocessor_config.json`,
`processor_config.json`) are part of the [source identity](#source-identity)
and are embedded with the rest of the checkpoint's small files, so an edited
processor config changes the digest, and an embedded copy that does not
match is refused (`embedded checkpoint mismatch`). A checkpoint that ships
none (gemma-4-31B-it) is preprocessed with its processor's class defaults.

## The embedded checkpoint

`drinkme pack` writes a self-contained pack: `<pack>/checkpoint/` is a small
checkpoint directory holding everything the loaders read besides the packed
Linears, so the pack serves with `HF_HUB_OFFLINE=1` and an empty Hugging Face
cache.

| File | Contents |
|---|---|
| `config.json`, `generation_config.json`, tokenizer files, `chat_template.jinja`, image processor configs | copied verbatim: every top-level `*.json`, `*.jinja`, `*.txt` and `*.model` file of the snapshot (the files `snapshot_dir` fetches), except the shard index |
| `LICENSE*`, `NOTICE*`, `*.md` | the license and model card, when the checkpoint ships them |
| `remainder.safetensors` | every tensor the loaders read raw, each copied as the byte range its shard header names, with its dtype and shape. Nothing is decoded or cast; the loaders apply the same BF16 rule to these bytes as to the checkpoint's |
| `source-identity.json` | each source shard's name, size and safetensors header, and the shard index: the inputs of `source.digest` that the directory does not otherwise hold |

The tensors in `remainder.safetensors` are the trunk's (every tensor with a
place in the served tree that the pack does not carry, a vision tower's
included, and the biases of packed Linears) and the MTP head's (every
`mtp.*` tensor the `mtp/` sub-pack does not carry: its norms, or the whole
raw head for a pack without a sub-pack). Two kinds are left out because no
loader reads them: a vision tower drinkme does not serve for that
architecture, and a tied output embedding, which the loaders re-tie to the
input embedding (Qwen3-0.6B ships a redundant 311 MB copy).
`DRINKME_MTP_DIET=0` serves the head's packed Linears raw from the source
checkpoint, so it needs the checkpoint.

The tensors are stored as safetensors rather than as `raw` pack tensors: the
remainder holds every dtype and rank a checkpoint ships (1-D norms, F32
biases, integer buffers) where the `raw` kind is a uint16 `[R, C]` plane; a
byte-range copy keeps the source's own encoding, so the served weights are
identical by construction; and both runtimes already read the format.

**Hashes and identity.** `meta.json`'s `embedded.files` records each file's
SHA-256, and the manifest includes the `embedded` block, so
`manifestSha256` binds it like the tensor map. Loading and `drinkme verify`
check every embedded file against its hash, then recompute the source digest
from the embedded `config.json`, tokenizer files and `source-identity.json`
and refuse the pack if it differs from `source.digest`
(`embedded checkpoint mismatch`). At pack time each embedded tensor is read
back and compared with its source bytes, and the recomputed digest must equal
the source's, before the pack is published. `generation_config.json` is
hash-checked but, as for a snapshot, not part of the identity digest.

**Loading.** The serving loaders (torch and MLX) resolve a pack's source to
its `checkpoint/` — never a snapshot, never the network. They make the same
refusals a snapshot load would: a pack cut from another Hub repo, an explicit
commit other than the pack's, a branch or tag that resolves elsewhere in the
local cache (or cannot be resolved without the network), and a `--model`
naming an existing local directory whose digest differs. The prefix-slot
store keys on `source.revision`, or on `local-` and the first 16 hex digits
of `source.digest` for a pack cut from a local directory. `drinkme bench`
resolves the real source snapshot even for a self-contained pack: its stock
and twin arms read the weights the pack compresses, on purpose (measuring
against the uncompressed original), so deleting a checkpoint that packs
still reference only affects `bench`, never `serve` or `verify`.

### Format version

`embedded` is a required field, and the manifest binds it, so it
cannot be stripped or added silently. `vision` is optional: a pack without
it serves text.

A pack contains the model's original tensors and tokenizer, so sharing one
is redistributing the model, and the model's license governs it.
Packs are published only for models whose licence allows it
([published packs](serve.md#published-packs)).

## Transactional writes

`PackWriter` stages files in `<dir>.staging-<pid>`, writes `meta.json` atomically,
and publishes the completed pack by rename. Verification failures and incomplete
writes leave the destination unchanged. A non-empty destination requires
`--replace`; the previous pack is retained until the replacement succeeds.
`<dir>.lock` serializes writers. An `mtp/` sub-pack and the embedded
`checkpoint/` are staged and published with the trunk. Never replace a pack
used by a running server.

## Cache paths

Default: `~/.cache/drinkme/packs/<org>--<model>@<revision>`, using the first
12 characters of the revision. A menu model uses its pinned revision. An
off-menu repository is pinned to the commit its default branch resolves to on
this machine: the local cache first, then the Hub. A later push to the branch
is therefore a different pack. `main` is used only when that commit cannot be
resolved, for example offline with nothing cached.
`DRINKME_HOME` changes the root; `serve --pack-dir` loads another location.
Persisted prefix slots use the manifest digest as part of their cache identity,
so re-packing a model starts its slot store cold.
