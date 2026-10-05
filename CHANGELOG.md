# Changelog

Notable changes to drinkme, newest first. Versions follow
[docs/versioning.md](docs/versioning.md): the major version is the pack format,
so packs and measurement records are comparable across every minor and patch
release of one major version.

## Unreleased

## 1.0.1 — 2026-10-04

### Added

- Apple silicon: `drinkme bench` on the MLX lane records `stock_decode_read_gb`
  and `compressed_decode_read_gb`, the bytes one decode step reads, as the
  CUDA and ROCm lanes already did.

### Changed

- Results site: the aggregator checks a record that carries no decode-read
  figure against the arm's resident weight size instead, which is the stricter
  bound, so earlier Apple silicon records stay on the chart.
- Apple M4 figures in the docs now come from re-measured records
  (Qwen3-1.7B 1.21×, Qwen3-4B 1.25× MLX-LM's BF16 decode speed).
- The measurement lexicon is documented as published, as a
  `com.atproto.lexicon.schema` record named by `_lexicon.drinkme.petrichor.wtf`.
- The results site links the GitHub mirror beside Tangled.

## 1.0.0 — 2026-10-01

First public release.

- Lossless BF16 weight compression with two profiles, `sip` (the default) and
  `gulp`, in pack format 1. Every packed tensor is checked bit for bit against
  the source checkpoint.
- `drinkme serve`: OpenAI Chat Completions, Responses and Anthropic Messages
  APIs, image input on models with a vision tower, tool calling, a prefix
  cache with slots persisted to disk, speculative decoding and sleep/wake.
- Runs on AMD (ROCm) and NVIDIA (CUDA) GPUs; Apple silicon support is in
  progress.
- Published packs on Hugging Face, fetched automatically by `serve`.
- `drinkme bench` and `drinkme publish`: measurements published as
  `wtf.petrichor.drinkme.measurement` records to your own atproto account,
  charted on the [results site](https://drinkme.petrichor.wtf).
