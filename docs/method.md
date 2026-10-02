# Compression method

## The method

Drinkme uses exponent-radix coding for BF16 weights and decodes them inside
the matrix-vector multiplication (GEMV) kernel. A BF16 weight has one sign bit,
an 8-bit exponent, and a 7-bit mantissa. In a trained tensor, the mantissa is
incompressible, but a few exponent values account for most weights.

The codec stores the sign and mantissa as one literal byte per weight. It
encodes the exponent using a per-tensor palette ordered by frequency. A first
tier of width *w*₁ represents the 2^*w*₁ − 1 most common exponents and escapes
the rest to a second tier. Subsequent tiers follow the same rule; the last
stores the exponent itself. Every BF16 bit pattern round-trips, including
signed zeros, subnormals, infinities, and NaN payloads. Blocks of 1024 weights
along a row decode independently using a word-offset directory, allowing a
kernel program to process one block or a whole row.

The **compression profile** sets the tier widths for a pack. The default, `sip` (3, 8),
uses about 11.4 bits per weight on Qwen3-8B and decodes faster. `gulp`
(2, 2, 4, 8) uses about 10.9 bits per weight. Tensors that would grow under
compression are stored raw. The [pack format](pack-format.md) specifies the
layout, measured sizes, and launch schedule.

The compression approach follows DFloat11: encode exponents according to their
frequency and preserve mantissas, yielding roughly 11 bits per weight. Drinkme
uses fixed-width tiers with escapes where DFloat11 uses entropy coding. As in
ZipServ, decoding runs inside the multiplication kernel: the M=1 GEMV and
multi-column speculative verification paths do not materialize BF16 weights.
The correctness checks cover every BF16 bit pattern, compression profile, and
launch-table row, and compare GEMV results against a float64 reference.

A vision model's image encoder is packed the same way: its eligible Linears
are coded with the text model's. On gfx1151 the compressed tower's features
are byte-equal to the uncompressed tower's for all four vision models on the
menu ([checks](checks.md#image-input)).

### Related work

- [DFloat11](https://arxiv.org/abs/2504.11651) uses variable-length entropy
  coding of the exponent and GPU decompression of transformer blocks before
  the matmul. This codec is the same compression class — the exponent coded, the
  mantissa literal, ~11 bits per weight — with fixed-width tiers in place of
  the entropy code and the decode fused into the GEMV instead of a separate
  decompression pass.
- [ZipServ](https://arxiv.org/abs/2603.17435) uses a fixed-length bitmap format
  decoded inside fused GEMM.
- Split12, from Brian Bell's
  [weight-compression](https://github.com/brianbell-x/weight-compression)
  project: BF16 weights split into
  an unchanged low byte and a high byte encoded in four bits through a
  per-tensor exponent window, with exceptions stored separately, decoded
  inside the GEMV at 12 bits per weight plus exceptions.

## Numerical behavior

Packed weights must reconstruct bit for bit. Kernel outputs can differ because
floating-point sums occur in different orders. Greedy decoding can then choose
different tokens when the leading logits are nearly tied. The same issue
applies to speculative batched verification versus serial decoding.

Use tensor-level decode checks for weight correctness and each platform's
acceptance gates for computation. A divergent transcript alone does not identify a codec
bug; unexplained differences still require investigation. See
[pack verification](pack-format.md#verification) and
[speculative decoding](serve-speculation.md).

## Measurements and records

Measured throughput, bandwidth, and fit numbers belong to the published
network records ([Bench and publish](bench.md)), which carry each run's
device, configuration, and result under a verifiable source identity;
`site/data/points.json` commits a snapshot of them. When reading them:

- Bandwidth is measured on the device during the run.
- Every compressed tensor is verified before a record is written:
  `raw.verification` says whether by a fresh round trip (`roundtrip`) or by
  the pack's hashes, which rest on the round trip at pack time (`pack`).
- Fit estimates are distinct from attempted loads and measured outcomes.
- A vision model's weights, resident bytes and bits per weight include its
  vision tower, which a server holds whenever image input is on
  ([bench](bench.md#loaders)).
- Records are self-reported and attributed to a DID when published. Site
  plausibility checks do not independently reproduce them.
- Compare matching versions, precisions, baselines, and measurement scopes;
  [versioning](versioning.md) defines the compatibility policy.
