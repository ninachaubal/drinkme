# Contributing

Issues and pull requests are welcome: bug reports, features, documentation,
and measurements on your hardware. You don't need to open an issue before
sending a pull request, though discussing a large change first can save work.

## Reporting a bug

Include enough to reproduce it:

- the model, as passed to `--model` (a menu name or `org/repo`), and the
  compression profile if not the default;
- the device, and the output of `uv run --no-sync drinkme bootstrap --detect-only`,
  which prints what drinkme detected and the install lane it chose;
- the drinkme commit (`git rev-parse --short HEAD`);
- the exact command, and any `DRINKME_*` variables you set;
- the relevant log output, from the command line through the error.

For a wrong answer rather than a crash, say whether `serve --stock` (the same
model, uncompressed) gives the right one.

## Submitting measurements

`drinkme bench` measures your hardware and writes a record; `drinkme publish`
puts it on your own atproto PDS. [Bench and publish](docs/bench.md) covers both,
including what a record contains and what publishing asks permission for.
The results site's [aggregator](site/aggregator/README.md) reads the
measurement records published on atproto and decides which ones the page
shows, so a record you publish reaches the page with no further step.

## Sending a pull request

Point your coding agent at [AGENTS.md](AGENTS.md): the invariants, setup, the
test command, and a map of the code. To run the CPU test suite yourself:

```sh
DRINKME_NO_AUTO_DEPS=1 HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=4 DRINKME_HOME=$(mktemp -d) \
  .venv/bin/python -m pytest -q
```

It takes four to five minutes. Read pytest's printed verdict rather than the
exit status: ROCm builds of torch can exit 0 after a failure. The suite hides
accelerators, so passing it does not validate GPU kernel dispatch, dtype
handling, or device placement. [Checks by change type](docs/checks.md) lists
the hardware checks each kind of change needs, and
[versioning](docs/versioning.md) says what a change to the codec, a kernel, a
metric or the engine string means for packs and published records.

A good pull request here:

- makes one change, or one concern per commit;
- adds or updates tests for what the change does;
- says which checks from [checks by change type](docs/checks.md) you ran, on
  which device, with their printed verdicts;
- keeps facts in code: docs are prose, and anything that must hold belongs in
  code with a test on it.
