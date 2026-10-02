# AGENTS.md — drinkme

This file is for coding agents working on drinkme on someone's behalf. People
start with the [README](README.md); how to report a bug, submit measurements or
send a pull request is in [CONTRIBUTING.md](CONTRIBUTING.md). This file gives
the invariants, the setup and CPU test commands, and a map of where everything
else lives.

Read this before running commands. Drinkme serves losslessly compressed model
weights: any change to what the accelerator reads requires correctness checks,
and packed weights must reconstruct bit for bit.

## Ask your user first

These have costs that are your user's to accept. Say what you are about to do
and wait for a yes:

- **Anything that downloads model weights or accelerator packages**:
  `drinkme bootstrap`, `serve`, `pack` and `bench` can each fetch several to
  tens of gigabytes, and the connection may be slow or metered.
- **`drinkme publish`**, which posts a public measurement record to the user's
  atproto repository under their identity. Never run it as a side effect of
  other work ([bench and publish](docs/bench.md)).
- **Long GPU work**, such as `drinkme bench` or the gates in `bench/`: the GPU
  may be in use, and a measurement wants an otherwise quiet machine.
- **Environment changes**, below.

## Invariants

- **Packed weights reconstruct bit for bit**, checked against the source at
  pack time and by the device gates. Greedy transcripts can still differ at
  near-ties because kernels accumulate in different orders; investigate
  unexplained differences with those checks
  ([numerical behavior](docs/method.md#numerical-behavior)).
- **Code owns facts and tests test code.** Docs are prose, reviewed by people
  and agents, not pinned by tests: no test reads a Markdown doc, so anything
  that must hold belongs in code, with a test on the code. A doc table
  rendered from a function carries a
  `<!-- generated from module.fn(); edit the source, not this table -->` line;
  change the source and paste its new output.
- **Tests never download and never touch a production server.**
- **Every uv invocation takes `--no-sync`.** A bare `uv sync` or `uv run` can
  remove or replace the accelerator packages under a live server. Beyond the
  one-time [fresh environment](#fresh-environment), an environment change is
  for a human in a separate session: report it, and do not combine it with
  other work ([why, and recovery](docs/dev-environment.md)).
- **Port 3215 may be your user's running server.** Test on another port, such
  as `--port 3299`; do not restart it or pack into its active pack directory
  ([running servers](docs/dev-environment.md#running-servers)).
- **Read the printed verdict, not the exit status.** TheRock ROCm torch can
  exit 0 after a failure ([known behavior](docs/dev-environment.md#known-behavior)).
- **Code comments link to an in-tree document or state the relevant fact
  directly.**

## Fresh environment

On a fresh clone with no `.venv`, once, in its own session:

```sh
git clone https://tangled.org/ninachaubal.com/drinkme
cd drinkme
uv sync                                        # base dependencies: no torch, no pytest
uv run --no-sync drinkme bootstrap --dry-run   # the lane it would install; installs nothing
uv run --no-sync drinkme bootstrap             # install the lane (several GB) into .venv and self-test it
uv pip install pytest jsonschema               # the dev extra's packages, after the lane
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

A `+cuXXX` torch on AMD or `False` on a GPU machine is an environment problem;
on Apple silicon use the [Metal runtime check](docs/metal.md#setup) instead.
[Development environment](docs/dev-environment.md) explains each step.

## CPU tests

Accelerators hidden, automatic dependency installation off, and a temporary
pack cache, so the run stays away from a live GPU server and your packs:

```sh
DRINKME_NO_AUTO_DEPS=1 HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=4 DRINKME_HOME=$(mktemp -d) \
  .venv/bin/python -m pytest -q
```

In a worktree, borrow the main checkout's interpreter rather than creating an
environment: `PYTHONPATH=$PWD/src /path/to/drinkme/.venv/bin/python -m pytest tests/ -q`,
with the same variables. It takes a few minutes on a desktop CPU; read
pytest's printed verdict. A passing suite does not validate GPU kernel dispatch, dtype handling,
or device placement: each change type's hardware checks are in
[checks](docs/checks.md).

## Where things live

Development docs:

- [docs/checks.md](docs/checks.md): the checks each change type needs, as one table.
- [docs/versioning.md](docs/versioning.md): what a patch, minor or major means for packs and published records, and what each change re-runs.
- [docs/dev-environment.md](docs/dev-environment.md): `--no-sync`, recovery, running servers, known warnings.
- [docs/testing.md](docs/testing.md): the CPU suite's settings, expected warnings and skips.
- [docs/architecture.md](docs/architecture.md): vocabulary, flows, and the module map.
- [docs/serve-kernels.md](docs/serve-kernels.md): how each serving kernel is chosen and A/B-tested.
- [docs/README.md](docs/README.md): every other doc, for running drinkme and for changing it.

Code, under `src/drinkme/` ([module map](docs/architecture.md#where-the-rest-lives)):

- [`codec/`](src/drinkme/codec): the compression codec, pack writer, GPU kernels and launch schedules ([pack format](docs/pack-format.md), [method](docs/method.md)).
- [`serving/`](src/drinkme/serving): the engines, cache, speculation, and the HTTP dialects ([serving](docs/serve.md)).
- [`metal/`](src/drinkme/metal): the MLX kernels for Apple silicon ([Metal](docs/metal.md)).
- [`publish/`](src/drinkme/publish): atproto publishing of measurement records ([bench and publish](docs/bench.md)).
- The top level: [`cli.py`](src/drinkme/cli.py), [`bootstrap.py`](src/drinkme/bootstrap.py) and its self-test, [`bench.py`](src/drinkme/bench.py) and the arms, fit, detection and `drinkme check` ([CLI](docs/cli.md), [hardware](docs/hardware.md)).

Elsewhere:

- [`tests/`](tests): the CPU suite ([testing](docs/testing.md)).
- [`bench/`](bench): GPU acceptance gates and performance instruments ([bench/README.md](bench/README.md)).
- [`lexicons/`](lexicons): the published measurement record's schema ([lexicons/README.md](lexicons/README.md)).
- [`site/`](site): the static results page ([site/README.md](site/README.md)).
- [`pyproject.toml`](pyproject.toml): the dependency groups per accelerator; its `[tool.uv]` comment explains the shared lock.
