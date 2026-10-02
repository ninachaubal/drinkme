# Development environment

The rules in [AGENTS.md](../AGENTS.md#invariants) in full: why every uv
command takes `--no-sync`, what the fresh-environment commands do, how a human
repairs an environment, how to share a machine with a running server, and the
warnings and exit codes that are normal here.

## Every uv run requires `--no-sync`

**Never let uv resolve this project's environment during development work.**
**Every uv invocation here takes `--no-sync`:**

```sh
uv run --no-sync drinkme serve --port 3216
```

Do not run a bare `uv sync` or `uv run` in an existing working environment.
Torch is deliberately outside the base dependencies, and the `cuda`, `rocm`,
`rocm-official` and `metal` groups conflict and share one lock. `uv sync` is
exact, so a bare one removes the accelerator packages; a bare `uv run` re-locks
when `pyproject.toml` has changed, which can move them under a live server
(pyproject.toml's `[tool.uv]` comment).

A resync can silently replace the accelerator wheel. A pipe such as `| tail`
hides uv's install output, and the next server start then runs on the wrong
runtime while CPU tests still pass. Check the actual device before trusting
performance results.

On a fresh clone with no `.venv`, follow
[Fresh environment](../AGENTS.md#fresh-environment) once, in its own session;
`--no-sync` applies from then on. Any other environment change is for a human
in a separate setup session: report it, and do not combine it with other work.

## Fresh environment

The commands are in [AGENTS.md](../AGENTS.md#fresh-environment).

On a fresh checkout, `uv sync` installs the base dependencies without an
accelerator group (`default-groups = []` in `pyproject.toml`). Bootstrap selects
and installs the accelerator runtime. `serve`, `pack`, `bench`, and `verify`
can also trigger bootstrap; running it explicitly lets you inspect the plan
first ([hardware setup](hardware.md)).

Bootstrap runs `uv sync --no-default-groups --group <group>` for the chosen
lane (for example, lane `therock` installs group `rocm`), without extras.
`pytest` and `jsonschema` belong to the optional `dev` extra, so install them
after the runtime with `uv pip install pytest jsonschema`.

Check the installed runtime before testing. A `+cuXXX` torch on AMD or `False`
on a GPU machine indicates an environment problem. On Apple silicon, the
Metal group installs MLX, and torch's CPU wheel for packing; use the
[Metal runtime check](metal.md#setup) in place of the torch command.

## Recovery

Symptoms include `torch.cuda.is_available()` returning false on a GPU machine,
a `+cuXXX` torch version on AMD, Triton's `0 active drivers ([])`, or `serve`
refusing at startup (exit 4 — [exit codes](cli.md#exit-codes)) with a broken-install
message naming the torch build.

For a human repairing an environment in a separate session:
`uv run --no-sync drinkme bootstrap --dry-run` shows the lane for this machine,
and `uv run --no-sync drinkme bootstrap` installs and self-tests it
([hardware setup](hardware.md)). On Strix Halo (gfx1151) that lane is
`uv sync --no-default-groups --group rocm`. Then
`uv pip install pytest jsonschema`. Verify the runtime before testing:

```sh
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## Running servers

3215 is `drinkme serve`'s default port, so a server there may be someone's
working server. Leave it alone:

- Do not restart it to test changes; start a test server on another port.
- Do not pack into its active pack directory.
- Announce any authorized restart. Afterward, check `/health`'s device and the
  boot log's `[drinkme] device:` line. If either differs from the expected
  accelerator, stop and report it.

## Known behavior

- `causal-conv1d unavailable — expected on ROCm` is expected. FLA supplies the
  delta-rule fast path; causal-conv1d is a CUDA-only dependency not shipped here.
  The boot log's `[drinkme] deltanet recurrence:` line says which kernel the
  decode recurrence runs (fla's fused kernel, or the torch reference;
  [serving kernels](serve-kernels.md)).
- Greedy A/B transcripts can diverge at near-ties because accumulation order
  differs. The invariant is bit-identical weights.
- TheRock ROCm torch can mask failures with an atexit `_exit(0)` after HIP
  initializes. Never trust exit status alone; require a verdict in the output.
- Under `DRINKME_STOCK_GEMV=mv`, `Warning: torch.backends.cuda.preferred_blas_library is an
  experimental feature` appears once per process, from the
  [stock GEMV](serve-kernels.md#stock-gemv) route's first one-row call: it
  prefers hipBLASLt for that call and restores the default after.
  The `linear` and `triton` modes (gfx1151's default) do not set it.
