"""`drinkme serve` — the artifact: an OpenAI-, Responses- and Anthropic-
compatible endpoint over a resident-compressed model.

Quickstart is one word and packs on demand — no --model detects the machine,
scans $DRINKME_HOME/packs for what's already there, and offers the largest
fit that comfortably holds its weights plus a KV charge at the assumed ctx/
slots (cli.py's no-model branch, packs.py): confirm, cancel, or type another
model on a TTY; --yes or a non-TTY stdin serves the pick straight away.

    drinkme serve

`--stock` serves the same model UNCOMPRESSED through the identical server,
tokenizer, sampling loop and KV. That is the A/B control for the endpoint
A/Bs under bench/ (serve_timing_ab.py and friends: stock vs compressed
through the same client). `drinkme bench` itself does NOT drive this file:
its three arms are module trees through the same HF forward, timed by its
own greedy loop (arms.py; docs/bench.md#torch-benchmark-scope), and
no record claims an endpoint number.

What the request path actually is (serving/http.py): ONE global generation
lock — requests are served one at a time, in order, on a batch-1 kernel;
there is no scheduler, no prefill quantum and no concurrent decoding.
Prefix slots (below) are reusable cache STATE, not concurrency. Streaming
aborts the generation on client disconnect (backpressure is a signal, not
something to buffer), and the hot loop allocates nothing.

Prefix/KV reuse (serving/engines.py module docstring — persistent
StaticCache + longest-common-prefix; --prefix-slots 0 gives
per-request caches). That cache has N SLOTS (DRINKME_PREFIX_SLOTS, default
1), so interleaved conversations do not evict each other; the per-slot cost
is announced at load and N slots must fit with the fit check's headroom or
the server refuses. The slots have a COLD TIER on disk (serving/slotstore.py,
docs/serve-prefix-slots.md): an evicted slot is written to DRINKME_SLOT_DIR
and a matching prompt is restored from it instead of prefilled, across a
bounce as well as across requests. This file
owns exactly one part of that — SIGTERM writes the live slots out before the
process exits, which is why a systemd unit's TimeoutStopSec must cover a
few GiB of writes (120s covers two 27B slots at ctx 32768 with room to
spare — docs/serve-prefix-slots.md).

SLEEP / WAKE (serving/sleep.py, docs/serve-sleep.md): `POST /sleep?level=1`
parks every model tensor in host RAM, persists the live prefix slots through
the SAME cold-tier path SIGTERM uses, and gives the accelerator back without
stopping the process; `POST /wake_up` reverses it. Level 2 also frees the
host copies and wakes by REPLAYING this file's own load — which is why the
swap between drinkme and another model server on one machine can become a
resume instead of a stop. Two things about it live in THIS file: --sleep-on-idle /
DRINKME_SLEEP_ON_IDLE_S (opt-in, default 0 = never), and the boot line that
says so; everything else is serving/http.py's routes and the engine's pass.
"""

from __future__ import annotations

import os
import sys
import time

from . import exitcodes

# `drinkme serve`'s listen port unless --port says otherwise (3215 upside-down
# on a calculator spells SIZE). A working server is most likely here, so the
# bench scripts that start or drive a server refuse this port by name.
DEFAULT_PORT = 3215


def check_pack_dir(repo: str, revision: str | None, path: str, verb: str = "drinkme serve") -> dict:
    """The front door's torch-free gate over an EXISTING pack directory,
    shared by `drinkme serve` (ensure_pack) and `drinkme bench --pack-dir`
    (bench.run — so a pack this build refuses is refused before the stock
    and twin arms have run). Returns the pack's meta.json. In order:

      1. a pack of a format version this build does not read: the loader's
         own refusal line, naming the version seen and ending in the exact
         re-pack command — `drinkme pack --model <its repo> --replace`
      2. a pack cut from one checkpoint served with another
         is refused with the two identities named (check_pack_repo, then
         resolve_pack_source when the pack carries `source` — which also
         catches a pack without an embedded checkpoint, by name, since
         `drinkme pack` always records both together; a pack without
         either gets the name check only here — the loader itself refuses
         it the same way once it actually loads)

    Every message starts with `verb` so the user sees which command refused.
    Nothing here imports torch: the refusal is one line before any load."""
    import json

    from .codec.pack import _check_format_version
    from .serving.checkpoint import (PackSourceMismatch, check_pack_repo,
                                     resolve_pack_source)

    with open(os.path.join(path, "meta.json")) as f:
        pack_meta = json.load(f)
    try:
        _check_format_version(pack_meta, path)
    except ValueError as e:
        # a definite verdict about THIS pack (a format this build does not
        # read) — exitcodes.Refused, the same bucket `check` REFUSED and
        # `verify`'s refusal are in
        raise exitcodes.Refused(f"{verb}: {e}") from None
    try:
        check_pack_repo(repo, pack_meta, path)
        if pack_meta.get("source"):
            resolve_pack_source(repo, revision, pack_meta, path)
    except PackSourceMismatch as e:
        raise exitcodes.Refused(f"{verb}: {e}") from None
    return pack_meta


def _embedded_config(pack_dir: str | None) -> dict | None:
    """A self-contained pack's own config.json, for the runtime's family
    door (runtimes.refusal_for_repo), so serving one needs no Hub lookup;
    None for no pack or a pack without an embedded checkpoint."""
    import json

    if not pack_dir:
        return None
    from .codec.pack import embedded_dir

    try:
        with open(os.path.join(embedded_dir(pack_dir), "config.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def ensure_pack(repo: str, revision: str | None, pack_dir: str | None,
                auto_pack: bool, hub_pack: bool = True) -> str:
    """Cache hit -> path, after check_pack_dir (the torch-free front door:
    format, then identity). Cache miss -> the published pack for this
    commit and profile when the pack hub has one and it verifies against
    the upstream hashes (hub_packs.fetch; `hub_pack` False is
    --no-hub-pack), else pack it now (that is the whole point of the
    one-liner), unless the caller asked us not to. A cached pack that came
    from the pack hub without a current upstream receipt is checked first
    (hub_packs.recheck)."""
    from . import hub_packs
    from .codec.pack import UnsupportedCheckpoint, default_pack_dir, pack_model

    path = pack_dir or default_pack_dir(repo, revision)
    hub_pack = hub_pack and not hub_packs.disabled_by_env()
    if os.path.exists(os.path.join(path, "meta.json")):
        check_pack_dir(repo, revision, path)
        if not hub_packs.is_hub_pack(path) or _receipt_current(path) \
                or hub_packs.recheck(repo, path):
            return path
        hub_pack = False  # that published pack did not verify, and is gone
    if not auto_pack:
        # --no-auto-pack was given and there is nothing cached: the same
        # bucket as `pack` onto an existing pack without --replace — the
        # fix is a different command (drop the flag, or pack first)
        raise exitcodes.Usage(
            f"drinkme serve: no pack at {path} and --no-auto-pack was given.\n"
            f"  run: drinkme pack --model {repo}"
        )
    if hub_pack:
        # before torch and the compiler: a published pack needs neither
        from .codec.radix_pack import resolve_compression_profile

        got = hub_packs.fetch(repo, revision, path, resolve_compression_profile(None))
        if got is not None:
            return got
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        # pack_model reads the checkpoint through torch. Every lane installs
        # it (the metal lane too: pyproject), so this is an environment made
        # before the metal lane carried torch, or one managed by hand. Say
        # what is missing and the two ways out, rather than an ImportError
        # from three frames down — an environment problem, not a usage one.
        raise exitcodes.CantRunHere(
            f"drinkme serve: no pack at {path}, and packing needs torch, which "
            "this environment does not have.\n"
            "  either: drinkme bootstrap      (installs this machine's lane, "
            "torch included), then run this command again\n"
            "  or pack on another machine and copy the pack directory here "
            "(docs/metal.md).")
    # THE FRONT DOOR, before any Hub traffic: a machine with no C++
    # compiler (or one that fails to build the native encoder) refuses
    # here, not partway through the download below.
    from .codec.radix_pack import refuse_unless_encoder_available

    refuse_unless_encoder_available(lead_in="drinkme serve")
    # About to download + pack a repo we may never have seen: check it from
    # kilobytes first, so an ineligible one is refused with its NAMED
    # condition before the 16GB download. Two planes (check.py): a
    # REFUSAL blocks; a broken CHECKER (offline, HF down) warns and proceeds —
    # cached weights must keep packing on an offline machine.
    from .check import CheckUnavailable, check

    try:
        verdict = check(repo, revision)
    except CheckUnavailable as e:
        print(f"[drinkme] check unavailable ({e}) — proceeding; "
              "pack verification is the hard gate regardless")
    else:
        print(f"[drinkme] {verdict.line()}")
        if not verdict.ok:
            raise exitcodes.Refused(f"drinkme serve: {verdict.line()}")
    print(f"[drinkme] no pack for {repo} yet — compressing once, then serving.")
    print("[drinkme] (one-time; the pack is cached and reused from here on)")
    t0 = time.time()
    try:
        out = pack_model(repo, revision, path, progress=lambda *_: None)
    except UnsupportedCheckpoint as e:
        raise exitcodes.Refused(f"drinkme serve: {e}") from None  # the one line (an FP8 checkpoint)
    except FileExistsError as e:
        # A directory with files but no meta.json is not a pack (meta.json
        # is the commit point) and is never written over silently.
        raise exitcodes.Usage(f"drinkme serve: {e}") from None
    print(f"[drinkme] packed in {time.time() - t0:.0f}s -> {out}")
    return out


def _receipt_current(path: str) -> bool:
    from .upstream import receipt_current

    return receipt_current(path) is not None


def resolve_device(torch) -> str:
    """Decide the device, SAY IT OUT LOUD, and refuse a broken accelerator install.

    Takes `torch` as an argument rather than importing it so this is testable
    without a GPU (or a 20s import): neither failure mode below reproduces
    on the happy path.

    A `uv sync` can swap a pinned TheRock ROCm wheel for a PyPI +cuXXX one,
    which reports zero devices; a server that fell back to CPU would then
    report every perf number as a CPU number wearing a GPU label. A warning
    alone does not stop that: `print()` to stdout is BLOCK-buffered when
    stdout is a pipe (the systemd journal), so an unflushed `[drinkme]
    device:` line can stay out of the log for a whole boot, and a stderr
    warning reads like the routine upstream warnings beside it while the
    machine serves CPU at ~30x slow.

    So: flush every announce, and REFUSE rather than warn. An accelerator-
    flavoured wheel that sees no device is a broken environment, not a CPU machine.
    A genuine CPU wheel (torch.version.cuda and .hip both None) is a choice a
    stranger may have made on purpose — warn there, and serve.
    """
    forced = os.environ.get("DRINKME_DEVICE")
    device = forced or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and not forced:
        built_for = ("CUDA" if torch.version.cuda
                     else "ROCm/HIP" if torch.version.hip else None)
        if built_for:
            # no usable device: exitcodes.CantRunHere, the same bucket as
            # resolve_device's other name in the pinned table (docs/cli.md)
            raise exitcodes.CantRunHere(
                f"drinkme serve: refused — torch {torch.__version__} is a "
                f"{built_for} build but sees zero devices — that is a broken "
                "install, not a CPU machine. On Strix Halo this usually means a bare "
                "`uv sync` or `uv run` (without `--no-sync`) replaced the pinned "
                "ROCm wheel with a PyPI +cuXXX one; fix with "
                "`uv sync --no-default-groups --group rocm`. "
                "To serve on CPU deliberately, set DRINKME_DEVICE=cpu.")
        print(f"drinkme: warning: torch {torch.__version__} sees no GPU — "
              "serving on CPU. On a GPU machine this usually means the wrong "
              "torch build (e.g. a +cuXXX wheel on ROCm). Set "
              "DRINKME_DEVICE=cpu to say this is intentional.",
              file=sys.stderr, flush=True)
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"[drinkme] device: {device} ({name}), torch {torch.__version__}",
          flush=True)
    return device


APPLE_SILICON_NOTICE = ("[drinkme] warning: Apple silicon support is a work in progress "
                        "(docs/metal.md).")


def apple_silicon() -> bool:
    """This host is a Mac on Apple silicon."""
    import platform

    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def resolve_runtime(explicit: str | None = None) -> str:
    """Which RUNTIME serves: `torch` (serving/engines.py, CUDA/ROCm/CPU) or
    `mlx` (serving/engine_mlx.py, Apple silicon — docs/metal.md). --runtime
    wins, then DRINKME_RUNTIME, then the host: Darwin on arm64 is mlx,
    everything else torch. Forcing `mlx` on a Linux machine is how the MLX
    engine's reference path gets tested here on CPU; forcing `torch` on a
    Mac is the PyPI MPS/CPU wheel, if one is installed."""
    b = (explicit or os.environ.get("DRINKME_RUNTIME", "")).strip().lower()
    if b:
        if b not in ("torch", "mlx"):
            raise exitcodes.Usage(f"drinkme serve: --runtime {b!r} is not 'torch' or 'mlx'")
        return b
    if apple_silicon():
        return "mlx"
    return "torch"


def build_engine(repo: str, revision: str | None, pack_path: str | None, stock: bool,
                 ctx: int | None = None, prefix_slots: int | None = None,
                 rope_scaling: dict | None = None, runtime: str | None = None):
    """Both arms -> ONE engine class per runtime, so the A/B never diverges:
    same server, same seam, only the weight-read differs. On the torch
    runtime compressed loads via engines.load_compressed (the meta-skeleton
    streaming fit path) and --stock loads bf16 through the identical class
    minus the codec; on the mlx runtime (resolve_runtime) the same two arms
    are engine_mlx.load_compressed_mlx / load_stock_mlx."""
    from .detect import force_ipv4

    force_ipv4()  # idempotent; the HF-CDN IPv6 blackhole guard must cover the
    # serve path too, not only check (the hot-loop audit)
    if resolve_runtime(runtime) == "mlx":
        return _build_engine_mlx(repo, revision, pack_path, stock, ctx=ctx,
                                 prefix_slots=prefix_slots, rope_scaling=rope_scaling)
    import torch

    from .serving import engines

    # a unified-memory OOM should take inference, not the desktop
    # (ds4.c:47930). Best-effort: not our call
    # to make on platforms without the knob.
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    device = resolve_device(torch)
    # Honest kernel-path announce for hybrid (DeltaNet) models. transformers
    # prints a misleading "falling back" warning whenever causal-conv1d is
    # absent, even though fla alone carries the delta-rule fast path (fla is
    # a base dep, pure triton — works on CUDA and ROCm;
    # causal-conv1d is a CUDA-only build, deliberately not shipped). Say
    # what is actually running so nobody has to decode upstream's warning.
    try:
        import fla  # noqa: F401
        _fla = f"fla {getattr(fla, '__version__', '?')} active"
    except Exception:
        _fla = "fla missing (delta-rule fast path off — was it dropped by a sync?)"
    try:
        import causal_conv1d  # noqa: F401
        _conv = "causal-conv1d active"
    except Exception:
        _conv = "conv on torch fallback (causal-conv1d unavailable — expected on ROCm)"
    print(f"[drinkme] hybrid kernels: {_fla}; {_conv}", flush=True)
    if stock:
        engine = engines.load_stock(repo, revision, device, ctx=ctx,
                                    prefix_slots=prefix_slots,
                                    rope_scaling=rope_scaling)
    else:
        engine = engines.load_compressed(repo, revision, pack_path, device, ctx=ctx,
                                         prefix_slots=prefix_slots,
                                         rope_scaling=rope_scaling)
    _announce_attention(engine, device)
    return engine


def _build_engine_mlx(repo: str, revision: str | None, pack_path: str | None,
                      stock: bool, ctx: int | None = None,
                      prefix_slots: int | None = None,
                      rope_scaling: dict | None = None):
    """The mlx runtime's half of build_engine. The device line is printed by
    the loader itself (engine_mlx names the SoC); no SDPA announce — that
    probe is torch's. Things the MLX engine v0 does not do are refused here
    by name rather than accepted and ignored."""
    from .serving import engine_mlx

    if prefix_slots is not None and prefix_slots not in (0, 1):
        raise exitcodes.Usage("drinkme serve: --prefix-slots is not implemented on the "
                              "MLX engine (v0 has no prefix cache; every request "
                              "re-prefills — docs/metal.md)")
    from .serving import ctx_checkpoints

    if ctx_checkpoints.max_from_env() not in (0, ctx_checkpoints.DEFAULT_MAX):
        raise exitcodes.Usage("drinkme serve: --ctx-checkpoints is not implemented on the "
                              "MLX engine (v0 has no prefix cache to checkpoint — "
                              "docs/metal.md)")
    if stock:
        return engine_mlx.load_stock_mlx(repo, revision, ctx=ctx)
    return engine_mlx.load_compressed_mlx(repo, revision, pack_path, ctx=ctx,
                                          rope_scaling=rope_scaling)


def _announce_attention(engine, device: str) -> None:
    """Say which SDPA backend this machine will actually run, and — when the answer
    is 'only math' — what that costs at THIS model's head count.

    A 25,057-token prompt to Qwen3.8-27B on gfx1151 fails with `Tried to
    allocate 56.13 GiB`, and `[1, 24, 25057, 25057]` fp32 IS 56.13 GiB. The math
    fallback is O(T^2) in MEMORY and the error names neither attention nor the
    env var that fixes it (sdpa.py has the full mechanism). This turns that
    future OOM into an instruction.

    It runs AFTER the load rather than beside the `device:` line because the
    head shape lives in the model config: reading it earlier would mean an
    extra config fetch that can touch the network on a first `--stock` run.
    Announcing here still puts it in the load block, ahead of warmup and ahead
    of the port bind — nothing has been served yet.

    Never fatal. A diagnostic that can take down a load is worse than no
    diagnostic; the probe already degrades internally, and this is the belt.
    """
    from . import sdpa

    try:
        sdpa.announce_for_engine(engine, device)
    except Exception as e:  # noqa: BLE001 — containment is the point
        print(f"[drinkme] SDPA probe failed ({type(e).__name__}: {e}) — "
              "serving anyway", file=sys.stderr, flush=True)


def _warmup(engine) -> None:
    """One tiny generation before the port binds: allocates the StaticCache,
    JITs the triton kernel signatures, compiles the chat template, and (MTP
    on) exercises the draft head — so the first user request is a request,
    not an initialization (the hot-loop audit). The server must never
    claim ready before it is. DRINKME_NO_WARMUP=1 skips (dev)."""
    if os.environ.get("DRINKME_NO_WARMUP") == "1":
        return
    import time

    from .serving.engine import GenerationRequest, SampleParams, complete

    t0 = time.perf_counter()
    try:
        complete(engine, GenerationRequest([{"role": "user", "content": "hi"}],
                                           SampleParams(temperature=0.0, max_tokens=8)))
        print(f"[drinkme] warm in {time.perf_counter() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001 — warmup must never kill the server
        print(f"[drinkme] warmup failed (serving anyway): {e}")


def _advertised_ctx_from_env(explicit: int | None) -> int | None:
    """`explicit` (CLI --advertised-ctx, advertised context) -> the ctx to advertise below
    real; else DRINKME_ADVERTISED_CTX; else None = unset = the default
    (real numbers everywhere). Same precedence and same "garbage warns and
    falls back rather than dying" posture as engines.slots_from_env — a typo
    here must not take the server down, and must not silently start lying
    about token counts either."""
    if explicit is not None:
        return explicit
    raw = os.environ.get("DRINKME_ADVERTISED_CTX", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if n < 1:
        print(f"[drinkme] DRINKME_ADVERTISED_CTX={raw!r} is not a positive "
              "token count — ignoring (real ctx will be reported)",
              file=sys.stderr, flush=True)
        return None
    return n


def _rope_scaling_from_env(explicit: dict | None) -> dict | None:
    """`explicit` (CLI --rope-scaling, rope scaling, already parsed by
    cli.parse_rope_scaling) wins; else DRINKME_ROPE_SCALING, parsed with the
    same grammar; else None = off, the default exactly — same
    precedence as --ctx/DRINKME_CTX (engines._ctx).

    Same "garbage warns and falls back rather than dying" posture as
    _advertised_ctx_from_env next door: a typo in the environment must not
    take the server down, and must not silently start serving a wider window
    nobody asked for either."""
    if explicit is not None:
        return explicit
    raw = os.environ.get("DRINKME_ROPE_SCALING", "").strip()
    if not raw:
        return None
    from .cli import parse_rope_scaling

    try:
        return parse_rope_scaling(raw)
    except ValueError as e:
        print(f"[drinkme] DRINKME_ROPE_SCALING={raw!r}: {e} — rope scaling "
              "stays off", file=sys.stderr, flush=True)
        return None


def _sleep_on_idle_from_env(explicit: float | None) -> float:
    """`explicit` (CLI --sleep-on-idle, sleep/wake) -> seconds of generation quiet
    before an automatic level-1 sleep; else DRINKME_SLEEP_ON_IDLE_S; else
    0 = NEVER, which is the default, unchanged from before this flag existed.

    Same precedence and the same "garbage warns and falls back rather than
    dying" posture as _advertised_ctx_from_env next door and
    engines.slots_from_env: a typo in a timer must not take the server down,
    and must not silently start parking a live server's weights either. The
    feature is opt-in for a reason — the machine this ships on serves a real
    person from a real client, and a bottle that decided on its own to put
    itself to sleep would be indistinguishable from a broken one until the
    wake landed."""
    if explicit is not None:
        if explicit < 0:
            print(f"[drinkme] --sleep-on-idle {explicit} is not a number of "
                  "seconds — idle sleep stays off", file=sys.stderr, flush=True)
            return 0.0
        return float(explicit)
    raw = os.environ.get("DRINKME_SLEEP_ON_IDLE_S", "").strip()
    if not raw:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        val = -1.0
    if val < 0:
        print(f"[drinkme] DRINKME_SLEEP_ON_IDLE_S={raw!r} is not a number of "
              "seconds — idle sleep stays off", file=sys.stderr, flush=True)
        return 0.0
    return val


def _persist_slots(engine) -> None:
    """Write the live prefix slots to the cold tier on the way out (on-disk prefix slots).

    getattr, not a method call: FakeEngine and any future Engine are held to
    engine.py's Protocol, and persisting a KV cache is an HFEngine detail, not
    part of the contract. Never fatal — a shutdown that fails to save a cache
    still has to be a shutdown."""
    persist = getattr(engine, "persist_slots", None)
    if persist is None:
        return
    try:
        t0 = time.perf_counter()
        n = persist()
        if n:
            print(f"[drinkme] persisted {n} prefix slot(s) in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001 — containment is the point
        print(f"[drinkme] persisting prefix slots failed ({type(e).__name__}: "
              f"{e}) — exiting anyway", file=sys.stderr, flush=True)


def _serve(engine, host: str, port: int, auth: str | None = None,
          advertised_ctx: int | None = None,
          served_names: list[str] | None = None,
          generation_profiles: dict | None = None,
          sleep_on_idle: float = 0.0) -> int:
    import signal

    from .serving.http import start_server

    try:
        srv = start_server(engine, host, port, auth_token=auth,
                           advertised_ctx=advertised_ctx,
                           served_names=served_names, generation_profiles=generation_profiles,
                           sleep_on_idle=sleep_on_idle)
    except OSError:
        print(f"drinkme serve: port {port} is already in use\n"
              "  pass --port N, or stop the process holding it.", file=sys.stderr)
        return exitcodes.CANT_RUN_HERE
    hp = "%s:%d" % srv.server_address[:2]
    ids = ", ".join(sorted(srv.known_ids - {engine.model_id}))
    print(f"[drinkme] serving {engine.model_id}"
          + (f" (aliases: {ids})" if ids else "")
          + (f" — profiles: {', '.join(sorted(generation_profiles))}" if generation_profiles else "")
          + f" at http://{hp}/v1/chat/completions"
          + (" (bearer auth ON)" if auth else ""))
    if advertised_ctx:
        real = engine.model_meta().get("contextWindow")
        # Advertised context: say the lie out loud, once, at boot — the one place an
        # operator reading the log will see both numbers side by side.
        print(f"[drinkme] advertised context: {advertised_ctx} (real {real}) — "
              "/v1/messages usage.input_tokens and count_tokens are SCALED "
              f"by {(real / advertised_ctx) if real else '?'}x; the engine's "
              "own accounting, /metrics and the bench stay real")
    if sleep_on_idle:
        # Sleep/wake: an opt-in that changes what a probe sees has to be in the boot
        # log, or the first person to meet a 503 on a healthy machine has nothing
        # to grep for.
        print(f"[drinkme] sleep on idle: level 1 after {sleep_on_idle:g}s with "
              "no generation — POST /wake_up (or /sleep?level=1 by hand); "
              "/health carries the state")

    # SIGTERM is how systemd stops this (on-disk prefix slots): save the slots, then let the
    # serving thread finish. The handler runs on the MAIN thread while
    # srv.thread.join() blocks it, so shutdown() here does not deadlock —
    # serve_forever is on the other thread and shutdown() only asks it to
    # stop. A slot another thread is mid-generation on presents as having no
    # valid ids and is skipped, which is the invariant, not a race.
    def _stop(_signum, _frame):
        print("[drinkme] SIGTERM — persisting prefix slots, then stopping",
              flush=True)
        _persist_slots(engine)
        srv.shutdown()

    try:
        signal.signal(signal.SIGTERM, _stop)
    except ValueError:  # not the main thread (a test, an embedder)
        pass
    try:
        srv.thread.join()
    except KeyboardInterrupt:
        _persist_slots(engine)
        srv.shutdown()
    return exitcodes.OK  # SIGINT/SIGTERM here is a clean shutdown, same as a normal stop


def run(repo: str, revision: str | None, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
        stock: bool = False, pack_dir: str | None = None,
        auto_pack: bool = True, ctx: int | None = None,
        auth: str | None = None, prefix_slots: int | None = None,
        ctx_checkpoints: int | None = None,
        advertised_ctx: int | None = None, spec: str | None = None,
        served_names: list[str] | None = None,
        generation_profile_flags: list[str] | None = None,
        sleep_on_idle: float | None = None,
        rope_scaling: dict | None = None,
        runtime: str | None = None,
        menu_name: str | None = None,
        image_max_pixels: int | None = None,
        tower_cache_gib: float | None = None,
        no_image_urls: bool = False,
        media_path: str | None = None,
        hub_pack: bool = True) -> int:
    """Resolve + pack, build the engine, serve. A load failure propagates —
    a server that cannot load its model has nothing honest to serve.

    `runtime` is --runtime (torch | mlx), resolve_runtime's precedence:
    CLI, then DRINKME_RUNTIME, then the host.

    `hub_pack` False is --no-hub-pack (or DRINKME_NO_HUB_PACK=1): a cache
    miss packs locally without asking the pack hub (hub_packs.py).

    `sleep_on_idle` is --sleep-on-idle (sleep/wake): seconds of generation quiet
    before an automatic level-1 sleep. None falls through to
    DRINKME_SLEEP_ON_IDLE_S and then to 0 = never.

    `prefix_slots` is --prefix-slots (prefix slots); engines.slots_from_env's own
    docstring carries the precedence decision (CLI wins over
    DRINKME_PREFIX_SLOTS when both are set, --ctx/DRINKME_CTX is the
    standing precedent). `advertised_ctx` is --advertised-ctx (advertised context), same
    precedence over DRINKME_ADVERTISED_CTX. `served_names`/`generation_profile_flags`
    are --served-model-name/--profile (aliases and generation profiles);
    generation_profiles.py's own docstring carries the alias/profile precedence
    and the thinking hazard. `menu_name` is --model when it named a menu entry
    (e.g. Qwen3-8B for Qwen/Qwen3-8B): it is always served as an alias, beside
    the flag's or the environment's names, and engine.model_id stays the repo.

    `ctx_checkpoints` is --ctx-checkpoints (serving/ctx_checkpoints.py),
    applied the way `spec` is, by WRITING DRINKME_CTX_CHECKPOINTS: the
    engine reads it at construction, and a level-2 wake constructs the engine
    again through the same loader, which must read the same count.

    `spec` is --spec (n-gram speculation), same precedence again — and it is applied by
    WRITING DRINKME_SPEC, because the mode is read in two places deep inside
    the engine (whether to load the head, and which proposer a request gets)
    and threading a third optional through HFEngine to reach both would buy
    nothing the environment does not already do. Set before build_engine, so
    the head-load decision sees it.

    DRINKME_FAKE_ENGINE=1 (dev-only, not a flag) serves FakeEngine instead:
    the full HTTP layer over canned generation, so the endpoint contract is
    curl-able with no model, no pack, no GPU.

    `image_max_pixels` is --image-max-pixels, the image pixel cap. It is
    applied the way `spec` is, by WRITING DRINKME_IMAGE_MAX_PIXELS, and for
    the same reason: the engine's Vision and the dialect handlers both read
    it (serving/vision.max_pixels_from_env). None leaves the environment
    alone, and then its value or the default 2560x1440 holds.

    `tower_cache_gib` is --tower-cache-gib, the vision tower output cache's
    cap (serving/tower_cache.py), applied the same way: a non-None value
    WRITES DRINKME_TOWER_CACHE_GIB, which the engine reads at construction,
    a level-2 wake's included.

    `no_image_urls` is --no-image-urls (http(s) image URLs are fetched
    by default, llama.cpp parity): False (the default; the flag absent)
    leaves DRINKME_IMAGE_URLS alone, so an operator can still set it
    directly; True (the flag given) WRITES DRINKME_IMAGE_URLS=0, the same
    "flag wins when both are set" precedence as --spec/--image-max-pixels
    (serving/vision.fetch_urls_from_env). `media_path` is --media-path
    (file:// image paths, disabled unless set — llama.cpp semantics),
    applied the same way: a non-None value WRITES DRINKME_MEDIA_PATH
    (serving/vision.media_path_from_env); the CLI's argparse type= already
    validated it names an existing directory.

    `rope_scaling` is --rope-scaling (rope scaling): None (default, unset) = off, the
    engine's native window, unclamped-by-this-feature behaviour; a dict
    (cli.parse_rope_scaling's shape) enables YaRN. Same CLI-over-env
    precedence as --ctx (_rope_scaling_from_env, next to
    _advertised_ctx_from_env), threaded through build_engine into
    engines.load_stock/load_compressed rather than written to an
    environment variable — unlike `spec`, nothing else in the engine needs
    to read it independently.

    On Apple silicon, cli.main prints APPLE_SILICON_NOTICE before the
    bootstrap and the revision pin run, so it is serve's first line."""
    from .serving import generation_profiles

    auth = auth or os.environ.get("DRINKME_AUTH_TOKEN") or None
    advertised_ctx = _advertised_ctx_from_env(advertised_ctx)
    rope_scaling = _rope_scaling_from_env(rope_scaling)
    if spec is not None:
        os.environ["DRINKME_SPEC"] = spec
    if ctx_checkpoints is not None:
        os.environ["DRINKME_CTX_CHECKPOINTS"] = str(ctx_checkpoints)
    if image_max_pixels is not None:
        os.environ["DRINKME_IMAGE_MAX_PIXELS"] = str(image_max_pixels)
    if tower_cache_gib is not None:
        os.environ["DRINKME_TOWER_CACHE_GIB"] = str(tower_cache_gib)
    if no_image_urls:
        os.environ["DRINKME_IMAGE_URLS"] = "0"
    if media_path is not None:
        os.environ["DRINKME_MEDIA_PATH"] = media_path
    names = generation_profiles.served_names_from_env(served_names)
    if menu_name and menu_name != repo and menu_name not in names:
        names = [menu_name, *names]
    idle = _sleep_on_idle_from_env(sleep_on_idle)
    if os.environ.get("DRINKME_FAKE_ENGINE") == "1":
        from .serving.engine import FakeEngine

        print("[drinkme] DRINKME_FAKE_ENGINE=1 — canned generation, no model loaded")
        profs = generation_profiles.generation_profiles_from_sources(generation_profile_flags, None)
        return _serve(FakeEngine(model_id=repo), host, port, auth=auth,
                      advertised_ctx=advertised_ctx, served_names=names,
                      generation_profiles=profs, sleep_on_idle=idle)
    # The runtime's family door, BEFORE a download or a pack (runtimes.
    # refusal_for_repo: the same predicate the no-model picker filtered
    # with, from config.json alone): a Mac asked for the 27B is told here,
    # not after 38 GB of packing.
    from . import runtimes

    refusal = runtimes.refusal_for_repo(resolve_runtime(runtime), repo, revision,
                                        config=_embedded_config(pack_dir))
    if refusal is not None:
        # this runtime (this machine's Apple-silicon MLX engine, usually)
        # cannot serve this model family — a capability gap in the runtime,
        # not a verdict about the model itself: exitcodes.CantRunHere
        raise exitcodes.CantRunHere(f"drinkme serve: {refusal}")
    path = None if stock else ensure_pack(repo, revision, pack_dir, auto_pack, hub_pack)
    profs = generation_profiles.generation_profiles_from_sources(generation_profile_flags, path)
    engine = build_engine(repo, revision, path, stock, ctx=ctx, prefix_slots=prefix_slots,
                          rope_scaling=rope_scaling, runtime=runtime)
    _warmup(engine)
    return _serve(engine, host, port, auth=auth, advertised_ctx=advertised_ctx,
                  served_names=names, generation_profiles=profs, sleep_on_idle=idle)
