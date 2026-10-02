"""Sleep/wake verify: the memory came back, the numbers did not change.

THE GATE, in one sentence: after `POST /sleep?level=1` the accelerator is
holding materially less of this model, and after `POST /wake_up` the model
computes BIT-IDENTICAL logits on a fixed prompt with its prefix slots
restored. Both halves matter and neither implies the other — a sleep that
frees nothing is a no-op wearing a feature's name, and a sleep that frees
everything but comes back subtly different is worse than not sleeping.

Four arms, one process, one loaded model:

  arm `load`    the cold boot, timed. This is the number a wake is measured
                against, taken on THIS box on THIS run rather than quoted
                from elsewhere.
  arm `sleep1`  level 1. Device free bytes before/during, host high-water
                mark (VmHWM) during, and the engine's own inventory: how many
                tensors moved, how many bytes, split into packed planes / raw
                tensors / MTP head. That split IS the report's tensor
                inventory; nothing here estimates it.
  arm `wake1`   wall clock, then the logits comparison. torch.equal, not
                allclose: the claim the whole project rests on is bit-
                identical weights, so a wake that perturbs a single low bit
                is a failure, not a rounding difference.
  arm `level2`  (--level2) free the host copies too and wake by REPLAYING the
                loader. The number to read is wake2_s against load_s: they
                should agree, because they are the same code path.

Prefix slots: the run builds one on a long prompt before sleeping, so
slots_persisted and slots_restored are exercised rather than asserted about
an empty cache. That needs a cold tier — pass --slot-dir (the default points
at a scratch directory under /tmp, NOT the live server's, so a verify run can
never evict a real conversation).

VRAM is read two ways on purpose. torch.cuda.mem_get_info() is what the
process can see and is what the fit story is written in; `rocm-smi
--showmeminfo vram` is what the BOX sees, which is the number that decides
whether the next model server can start. They disagree when the caching allocator holds
blocks it has not returned, and that disagreement is exactly what
sleep.release()'s empty_cache() exists to close — so both are recorded.

THE RESERVED GATE: while asleep, torch.cuda.memory_reserved()
must be at most RESERVED_CEILING — one small-pool segment — at level 1, at
level 2, and after an idempotent level-2 repeat. A direct level-2 sleep on
this very model can report 5.84 GiB freed while 6,352,273,408 B stay
reserved (and 36 MiB after level 1: the BLAS workspace); both wake paths
still restore bit-identical logits then, so a gate
that only checked the wake passed. This one reads the allocator. To make
that reading about the ENGINE, the harness holds nothing on the device
while the model sleeps: the Tap is unwrapped before level 2 and the captured
logits rows live on the host.

Writes verification/sleep_wake_<model>_<host>_<date>.json.
Run: uv run --no-sync python bench/sleep_wake_verify.py [--pack DIR]
     (--no-sync is not optional here — see AGENTS.md)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

import torch

# The 8B re-answers a settled question cheaply. The 27B hybrid is the run
# that matters — it is the model a swap with another server is actually about, and
# its MTP head is the third class in the inventory.
DEFAULT_PACK = os.path.expanduser("~/.cache/drinkme/packs/Qwen--Qwen3-8B@b968826d9c46")

_GIB = 1024 ** 3

# What the allocator may still hold while the model sleeps: one small-pool
# segment. A model's worth is three to four orders of magnitude more, so
# the ceiling separates "nothing of the model" from "the model" without
# pretending torch never keeps a 2 MiB block around.
RESERVED_CEILING = 2 * 1024 ** 2


def _device_bytes() -> dict:
    """(free, total) as the PROCESS sees them, in bytes. None off a GPU."""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"free": free, "total": total, "used": total - free,
            "torch_reserved": torch.cuda.memory_reserved(),
            "torch_allocated": torch.cuda.memory_allocated()}


def _smi_vram() -> dict:
    """`rocm-smi --showmeminfo vram`, parsed loosely — what the BOX sees.
    Absent tool, absent parse, absent number: this is corroboration, never
    the gate, and a missing rocm-smi must not fail a verify."""
    exe = shutil.which("rocm-smi")
    if exe is None:
        return {"available": False}
    try:
        out = subprocess.run([exe, "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=30)
        return {"available": True, "json": json.loads(out.stdout)}
    except Exception as e:  # noqa: BLE001 — corroboration, never the gate
        return {"available": True, "error": f"{type(e).__name__}: {e}"}


def _host_mem() -> dict:
    """Host RSS now and the process high-water mark, from /proc/self/status.
    VmHWM is the number the report's "host-RAM high-water mark during a
    level-1 sleep" asks for, and it is a HIGH-WATER mark, so it survives the
    wake that gives the memory back."""
    out = {}
    try:
        for line in open("/proc/self/status"):
            key, _, val = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                out[key] = int(val.split()[0]) * 1024  # kB -> bytes
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    out["MemAvailable"] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return out


def _snapshot(label: str) -> dict:
    return {"label": label, "t": time.time(), "device": _device_bytes(),
            "host": _host_mem(), "smi": _smi_vram()}


def _gib(n) -> str:
    return "?" if n is None else f"{n / _GIB:.2f} GiB"


def _prompt(n_rules: int) -> list[dict]:
    """A long, agent-shaped prompt — long enough to build a real prefix slot
    (well past slotstore.BLOCK_TOKENS) and to make the prefill worth having
    restored. ~22k tokens at the default, a long agent prompt on the 27B."""
    body = " ".join(
        f"Rule {i}: when the operator mentions topic-{i}, answer with a "
        f"numbered plan citing section {(i * 7) % 101}, tone plain."
        for i in range(n_rules))
    return [{"role": "system", "content": f"Handbook.\n{body}"},
            {"role": "user", "content": "Summarize your rules for topic-3."}]


class Tap:
    """Keeps the FIRST forward of each armed window — the prefill row, which
    is where a weight difference shows before rounding hides it.
    prefix_slots_verify.py's ModelTap, minus the arms it does not need."""

    def __init__(self, model):
        self._m = model
        self.first = None

    def arm(self):
        self.first = None

    def __call__(self, *a, **kw):
        out = self._m(*a, **kw)
        if self.first is None and getattr(out, "logits", None) is not None:
            # to the HOST: a device-resident row is the harness's own
            # allocation, and the reserved gate below must read the engine's
            self.first = out.logits[0, -1].detach().float().cpu()
        return out

    def __getattr__(self, name):
        return getattr(self._m, name)


def _turn(eng, tap, messages, max_tokens):
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    tap.arm()
    t0 = time.perf_counter()
    res = complete(eng, GenerationRequest(
        list(messages), SampleParams(temperature=0.0, max_tokens=max_tokens)))
    return res, tap.first, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=DEFAULT_PACK,
                    help="pack dir to verify; point it at the 27B hybrid for "
                         "the run that matters most (DeltaNet state + KV)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--rules", type=int, default=900,
                    help="length of the system prompt, in rules (~22k tokens "
                         "at the default) — the prefix a slot has to carry")
    ap.add_argument("--slot-dir", default="/tmp/drinkme-sleep-verify-slots",
                    help="cold tier for this run. NOT the live server's: a "
                         "verify must never evict a real conversation")
    ap.add_argument("--level2", action="store_true",
                    help="also measure a level-2 sleep and the reload wake")
    ap.add_argument("--pin", action="store_true",
                    help="stage the host copies in PINNED memory "
                         "(DRINKME_SLEEP_PIN=1). Off by default for the "
                         "reason serving/sleep.py names: on a unified-memory "
                         "box pinned RAM is memory the next model cannot have")
    args = ap.parse_args()
    pack = os.path.expanduser(args.pack)
    if not os.path.isdir(pack):
        sys.exit(f"[sleep] no such pack: {pack}")

    os.environ["DRINKME_SLOT_DIR"] = args.slot_dir
    if args.pin:
        os.environ["DRINKME_SLEEP_PIN"] = "1"

    from drinkme.serve import build_engine

    meta = json.load(open(os.path.join(pack, "meta.json")))
    snaps = [_snapshot("boot")]
    print(f"[sleep] loading {meta['hfRepo']}@{str(meta['revision'])[:12]} "
          f"(meanBpw {meta['meanBpw']}) ...", flush=True)
    t0 = time.perf_counter()
    eng = build_engine(meta["hfRepo"], meta["revision"], pack, stock=False,
                       ctx=args.ctx)
    load_s = time.perf_counter() - t0
    print(f"[sleep] loaded in {load_s:.1f}s", flush=True)
    if str(eng.device) == "cpu" and not os.environ.get("VERIFY_ALLOW_CPU"):
        # a CPU verify already produced a day of mislabeled numbers once
        sys.exit("[sleep] REFUSING to measure on CPU (VERIFY_ALLOW_CPU=1 to override)")
    snaps.append(_snapshot("loaded"))

    tap = Tap(eng.model)
    eng.model = tap
    messages = _prompt(args.rules)
    res, before_row, warm_s = _turn(eng, tap, messages, args.max_tokens)
    print(f"[sleep] prefill+decode: {res.prompt_tokens} prompt tokens in "
          f"{warm_s:.1f}s; slot ids {max(len(s.ids) for s in eng._slots)}",
          flush=True)
    if before_row is None:
        sys.exit("[sleep] no prefill row captured — the tap saw no logits; "
                 "refusing to report a pass")
    snaps.append(_snapshot("after_first_turn"))

    # -- level 1 ------------------------------------------------------------
    t0 = time.perf_counter()
    st1 = eng.sleep(1)
    sleep1_s = time.perf_counter() - t0
    snaps.append(_snapshot("asleep_level1"))
    print(f"[sleep] level 1 in {sleep1_s:.1f}s: {st1['tensors']} tensors, "
          f"{_gib(st1['parked_bytes'])} parked; slots "
          f"{st1['slots_persisted']} persisted / {st1['slots_dropped']} dropped",
          flush=True)

    t0 = time.perf_counter()
    st_wake = eng.wake()
    wake1_s = time.perf_counter() - t0
    snaps.append(_snapshot("awake_level1"))
    print(f"[sleep] wake in {wake1_s:.1f}s ({wake1_s / load_s:.2f}x the cold "
          f"load); {st_wake['slots_restored']} slot(s) restored", flush=True)

    _, after_row, warm2_s = _turn(eng, tap, messages, args.max_tokens)
    identical = bool(after_row is not None and torch.equal(before_row, after_row))
    delta = (0.0 if after_row is None
             else float((before_row - after_row).abs().max()))
    argmax_same = bool(after_row is not None
                       and int(before_row.argmax()) == int(after_row.argmax()))
    print(f"[sleep] logits after wake: identical={identical} "
          f"argmax_same={argmax_same} max|delta|={delta:.3e}", flush=True)

    # -- level 2 (optional) -------------------------------------------------
    level2 = None
    if args.level2:
        # unwrap the tap FIRST: it is this harness's reference to the model
        # object, and a level-2 sleep can only free what nothing else holds.
        # With it in place the run measured its own leak — 6.3 GB allocated
        # after sleep(2), two models resident after the wake.
        eng.model, tap = tap._m, None
        t0 = time.perf_counter()
        st2 = eng.sleep(2)
        sleep2_s = time.perf_counter() - t0
        snaps.append(_snapshot("asleep_level2"))
        eng.sleep(2)  # idempotent repeat: must ALSO leave nothing reserved
        snaps.append(_snapshot("asleep_level2_again"))
        t0 = time.perf_counter()
        eng.wake()
        wake2_s = time.perf_counter() - t0
        snaps.append(_snapshot("awake_level2"))
        # the reload built a new model object; tap that one
        tap = Tap(eng.model)
        eng.model = tap
        _, row2, _ = _turn(eng, tap, messages, args.max_tokens)
        l2_identical = bool(row2 is not None and torch.equal(before_row, row2))
        level2 = {"sleep_s": sleep2_s, "wake_s": wake2_s,
                  "wake_vs_cold_load": wake2_s / load_s,
                  "bytes_freed": st2["parked_bytes"],
                  "logits_identical": l2_identical}
        print(f"[sleep] level 2: slept in {sleep2_s:.1f}s, RELOADED in "
              f"{wake2_s:.1f}s ({wake2_s / load_s:.2f}x the cold load); "
              f"logits identical={l2_identical}", flush=True)

    def _free(label):
        for s in snaps:
            if s["label"] == label:
                return s["device"].get("free")
        return None

    returned = None
    if _free("after_first_turn") is not None and _free("asleep_level1") is not None:
        returned = _free("asleep_level1") - _free("after_first_turn")

    def _reserved(label):
        for s in snaps:
            if s["label"] == label:
                return s["device"].get("torch_reserved")
        return None

    # the allocator while asleep, per level (None where a level was not run
    # or the box cannot report it)
    reserved_asleep = {lbl: _reserved(lbl) for lbl in
                       ("asleep_level1", "asleep_level2", "asleep_level2_again")}
    reserved_asleep = {k: v for k, v in reserved_asleep.items() if v is not None}

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M %Z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "model": {k: meta.get(k) for k in ("hfRepo", "revision", "meanBpw")},
        "device": str(eng.device),
        "ctx": args.ctx,
        "prompt_tokens": res.prompt_tokens,
        "pinned": bool(args.pin),
        "load_s": load_s,
        "sleep1_s": sleep1_s,
        "wake1_s": wake1_s,
        "wake_vs_cold_load": wake1_s / load_s,
        "inventory": {"tensors": st1["tensors"], "bytes": st1["parked_bytes"],
                      "by_class": st1["by_class"]},
        "slots": {"persisted": st1["slots_persisted"],
                  "dropped": st1["slots_dropped"],
                  "restored": st_wake.get("slots_restored")},
        "device_bytes_returned_while_asleep": returned,
        "reserved_while_asleep": reserved_asleep,
        "reserved_ceiling": RESERVED_CEILING,
        "host_high_water_bytes": max(
            (s["host"].get("VmHWM", 0) for s in snaps), default=0),
        "snapshots": snaps,
        "logits": {"identical": identical, "argmax_same": argmax_same,
                   "max_abs_delta": delta, "first_turn_s": warm_s,
                   "after_wake_turn_s": warm2_s},
        "level2": level2,
    }
    # THE GATE. Bit-identity is non-negotiable; the memory claim is asserted
    # only where it is measurable (mem_get_info exists), because a box that
    # cannot report free bytes has not failed the sleep, it has failed to be
    # measured — and saying "pass" on that would be a false pass.
    ok = identical
    if returned is not None:
        ok = ok and returned > 0
    if level2 is not None:
        ok = ok and level2["logits_identical"]
    # and the allocator: nothing of the model may stay RESERVED while it
    # sleeps, at either level, repeat included
    reserved_ok = all(v <= RESERVED_CEILING for v in reserved_asleep.values())
    ok = ok and reserved_ok
    report["pass"] = bool(ok)
    report["memory_measurable"] = returned is not None
    report["reserved_while_asleep_ok"] = bool(reserved_ok)

    out = os.path.join(os.path.dirname(__file__), "..", "verification",
                       f"sleep_wake_{str(meta.get('hfRepo')).split('/')[-1]}"
                       f"_{socket.gethostname()}_{time.strftime('%Y-%m-%d')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[sleep] device returned while asleep: {_gib(returned)}; "
          f"host high-water {_gib(report['host_high_water_bytes'])}")
    print("[sleep] reserved while asleep: " + ", ".join(
        f"{k.removeprefix('asleep_')}={v:,} B" for k, v in reserved_asleep.items())
        + f" (ceiling {RESERVED_CEILING:,} B): {'PASS' if reserved_ok else 'FAIL'}")
    print(f"[sleep] {'PASS' if ok else 'FAIL'} — wrote {os.path.normpath(out)}")
    return 0 if ok else 2


if __name__ == "__main__":
    # os._exit, NOT a bare return: on this box's TheRock ROCm torch an atexit
    # handler calls _exit(0) once HIP has initialised, so a verify that
    # REFUSED to run would otherwise report success (prefix_slots_verify.py
    # carries the same footer).
    import os as _os

    try:
        _code = main() or 0
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()
    _os._exit(_code)
