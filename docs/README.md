# Documentation

The [project README](../README.md) covers installation and client setup.
Commands here assume an activated environment; alternatively prefix `drinkme`
with `uv run --no-sync`. Development starts at [AGENTS.md](../AGENTS.md).

## Running drinkme

- [CLI reference](cli.md): commands, flags, and environment variables.
- [Hardware](hardware.md): platform status, and the first-run probe and self-test.
- [Models](models.md): tested models, what each vision model makes of an image, automatic selection, and `drinkme check`.
- [Serving](serve.md): API capabilities, image and video input, chunked prefill, metrics, context limits, token budgets, and YaRN.
- [Responses API](serve-responses.md): request mapping, supported fields, streaming, and clients.
- [Clients](clients.md): connecting a client (base URLs, keys, model ids), a first request, Codex, turning thinking off, and a pi recipe.
- [Tool formats](serve-tool-formats.md): dialect detection, parsers, and adding support.
- [Speculative decoding](serve-speculation.md): MTP and prompt lookup.
- [Prefix cache](serve-prefix-slots.md): memory and disk slots, sizing, and template interactions.
- [Sleep/wake](serve-sleep.md): memory release, idle timers, and service integration.
- [ROCm](rocm.md): the two AMD install lanes and each gfx target's status, installing a TheRock line by hand, the attention setup (the AOTriton flag, systemd, SDPA backend limits), and the vision towers on gfx1151.
- [Apple silicon](metal.md): MLX/Metal implementation, limitations, and hardware checks.

## Changing drinkme (start at AGENTS.md)

- [AGENTS.md](../AGENTS.md): the entry point for development — invariants, setup, the CPU test command, and a map of the tree.
- [Architecture and vocabulary](architecture.md): terminology, serving and packing flows, benchmark flow, and module responsibilities.
- [Checks by change type](checks.md): the CPU tests, hardware acceptance, and gates each kind of change needs, as one table.
- [Development environment](dev-environment.md): why every uv run takes `--no-sync`, recovery, running servers beside a live one, and known warnings.
- [Serving kernels](serve-kernels.md): attention paths over the live cache and MTP suffixes, segmented attention, the DeltaNet recurrence and convolution kernels, and narrow GEMV — how each is chosen and A/B-tested.
- [Testing](testing.md): the CPU suite's settings, expected warnings and skips, and tests that need cached checkpoints.

## Compression and measurement

- [Pack format](pack-format.md): BF16 encoding, the sip/gulp compression profiles, launch schedules, integrity checks, source identity, the vision tower, and format compatibility.
- [Method](method.md): codec design, related work, and numerical behavior.
- [Bench and publish](bench.md): benchmark arms, fit checks, records, and atproto publishing.
- [Measurement lexicon](../lexicons/README.md): record fields and protocol development.
- [Versioning](versioning.md): what a patch, minor or major means for packs and published records, the comparison identity, and what each change re-runs.
- [Developer instruments](../bench/README.md): GPU acceptance and performance tools.
- [Site](../site/README.md): static preview, chart data, and frontend checks.
