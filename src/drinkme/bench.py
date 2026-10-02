"""`drinkme bench` — the community loop: detect -> confirm -> menu -> probe ->
arms -> local result. Auth never happens here; publish is a separate, opt-in
verb.

The result dict IS a `wtf.petrichor.drinkme.measurement` record minus
repo bookkeeping — one self-contained datapoint: this device, this model,
these numbers (metrics[] self-describing with all samples), signed by whoever
opts to publish it. Decode arms run 3x; the site aggregates median per
(model+revision, device, memoryBytes, memoryKind) cell.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import re
import sys

from . import MEASUREMENT, __version__, exitcodes, spec_pass
from .codec.identity import is_commit_sha
from .detect import Hardware, detect, force_ipv4
from .fit import (GB, STOCK_LOADER_STREAM, STOCK_LOADERS, stock_arm_fits,
                  stock_transient_bytes, streamed_arm_fits, streamed_transient_bytes)
from .probe import host_load_warning
from .serving.checkpoint import cached_snapshot_dir, largest_tensor_bytes
from .serving.kernel_route import describe as describe_route
from .suggest import (FIT_HEADROOM, RATIO_HEADROOM, Suggestion,
                      gemma_gate_open, suggest)


def _confirm(question: str, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        return default
    yn = "[Y/n]" if default else "[y/N]"
    ans = input(f"{question} {yn} ").strip().lower()
    return default if not ans else ans.startswith("y")


def _menu_suggestion(hw: Hardware, runtime: str, gemma_ok: bool) -> Suggestion:
    """The menu bench offers. On the mlx runtime the budget is
    packs.hardware_budget()'s (the smaller of MLX's working set and what macOS
    would hand a process now, the serve picker's), not detect's sysctl x 0.75,
    and there is no fit point: that lane has no stock-does-not-fit case, every
    arm loads through its own fit check (arms_mlx.run_arms)."""
    budget_gb = hw.budget_gb
    if runtime == "mlx":
        from .packs import GIB, hardware_budget

        gib, _ = hardware_budget()
        if gib is not None:
            budget_gb = gib * GIB / GB  # GiB -> decimal GB, suggest()'s unit
    # CUDA graphs run only on NVIDIA through torch (serving/cudagraph.py);
    # suggest charges them on the fit point's compressed arm there
    graphs = runtime != "mlx" and any(e.startswith("nvidia-smi:") for e in hw.evidence)
    s = suggest(budget_gb, hw.memory_gb, hw.memory_kind, gemma_ok, graphs=graphs)
    if runtime == "mlx":
        s.fit, s.fit_knife_edge, s.fit_honest_negative = None, False, False
    return s


def _stock_ceiling_gb(hw: Hardware) -> float | None:
    """The ceiling drinkme.fit.stock_arm_fits should charge the stock arm's
    transient against: physical memory on unified (the OOM killer's
    own limit — CPU RAM and GTT are the same pool there), VRAM budget on
    discrete (a separate pool the CPU staging copy never touches)."""
    if hw.memory_kind == "unified":
        return hw.memory_gb
    return hw.budget_gb


def _stock_ceiling_bytes(hw: Hardware) -> int | None:
    """_stock_ceiling_gb's exact figure — the detector's own byte count for
    the same pool — for the record's stock.budgetBytes."""
    if hw.memory_kind == "unified":
        return hw.memory_bytes
    return hw.budget_bytes


def _stock_fit_ok(hw: Hardware, bf16_gb: float, stock_loader: str = STOCK_LOADER_STREAM,
                        tensor_bytes: float | None = None) -> bool:
    """Would the stock arm's load fit THIS machine, honoring the real transient
    of the loader that will run (drinkme.fit.stock_arm_fits)? `stock_loader`:
    "stream" charges resident + one tensor
    (`tensor_bytes`, the checkpoint's largest — None = the 5 GiB shard
    bound); "from_pretrained" charges 2x resident on unified memory
    (measured). False on an unreadable ceiling — an unknown budget is not a
    green light to gamble a whole-model load on."""
    if stock_loader not in STOCK_LOADERS:
        raise ValueError(f"unknown stock loader {stock_loader!r}; one of {STOCK_LOADERS}")
    ceiling_gb = _stock_ceiling_gb(hw)
    if ceiling_gb is None or hw.memory_kind is None:
        return False
    return stock_arm_fits(bf16_gb * GB, 0.0, ceiling_gb * GB, hw.memory_kind, RATIO_HEADROOM,
                          direct=stock_loader == STOCK_LOADER_STREAM, tensor_bytes=tensor_bytes)


def _largest_tensor_bytes(m) -> int | None:
    """The streaming loaders' one-tensor term for every arm's fit check
    (stock, twin and compressed all stream — fit.streamed_transient_bytes):
    the largest tensor in the checkpoint's shard headers IF the snapshot is
    already on this machine (never the network — bench downloads at load time,
    and --dry-run downloads nothing); None otherwise, which the arithmetic
    charges as fit.STAGING_SHARD_BYTES."""
    snap = cached_snapshot_dir(m.hf_repo, m.revision)
    return largest_tensor_bytes(snap) if snap else None


def _twin_fit_ok(hw: Hardware, bf16_gb: float, tensor_bytes: float | None) -> bool:
    """Would the twin's load fit THIS machine? The twin streams too
    (arms.load_twin_streaming): one bf16 copy resident + one tensor of
    host transient on unified
    memory, judged with the stock arm's RATIO_HEADROOM against the same
    ceiling (physical memory on unified, the VRAM budget on discrete).
    False on an unreadable ceiling."""
    ceiling_gb = _stock_ceiling_gb(hw)
    if ceiling_gb is None or hw.memory_kind is None:
        return False
    return streamed_arm_fits(bf16_gb * GB, 0.0, ceiling_gb * GB, hw.memory_kind,
                             RATIO_HEADROOM, tensor_bytes)


def _compressed_fit(hw: Hardware, comp_gb: float,
                          tensor_bytes: float | None) -> tuple[bool, bool]:
    """(fits with FIT_HEADROOM, fits at all) for the compressed arm's
    streaming load (arms.load_compressed_streaming): the pack's resident
    bytes (suggest.Sizes.comp_gb: a pack's own, or the estimate) + one
    tensor on unified memory, against the
    same ceiling the other arms are judged against. The second verdict is
    the hard line — below it bench does not attempt the point at all (no
    fallback arm exists, so no record could come of it); between the two
    the point is a knife-edge attempt, suggest()'s own stance ("attempting,
    may OOM honestly"). (False, False) on an unreadable ceiling."""
    ceiling_gb = _stock_ceiling_gb(hw)
    if ceiling_gb is None or hw.memory_kind is None:
        return False, False
    args = (comp_gb * GB, 0.0, ceiling_gb * GB, hw.memory_kind)
    return (streamed_arm_fits(*args, FIT_HEADROOM, tensor_bytes),
            streamed_arm_fits(*args, 1.0, tensor_bytes))


def _streamed_charge_line(hw: Hardware, resident_gb: float, what: str, headroom: float,
                          tensor_bytes: float | None) -> str:
    """One human line naming what a streamed arm's fit check charged and
    against what: `resident_gb` (`what` says whose — 'bf16' for the twin,
    'compressed' for the pack) + one tensor on unified, x headroom, against
    the ceiling."""
    if hw.memory_kind != "unified":
        term = "x1 (VRAM is its own pool)"
    elif tensor_bytes:
        term = f"+ one tensor ({_gb(tensor_bytes)} GB, streaming loader)"
    else:
        term = "+ one tensor (5 GiB shard bound: snapshot not cached, headers unread; streaming loader)"
    charge = streamed_transient_bytes(resident_gb * GB, 0.0, hw.memory_kind or "vram", tensor_bytes)
    ceiling = _stock_ceiling_gb(hw)
    return (f"{resident_gb:.2f} GB {what} {term} = {_gb(charge)} GB x{headroom} headroom "
            f"against {ceiling} {'GB' if ceiling is not None else ''} {hw.memory_kind or '?'}")


def _comp_what(sz) -> str:
    """How a charge line names the compressed arm's resident bytes: an
    estimate says so (suggest.Sizes)."""
    return "compressed (estimate)" if sz.estimated else "compressed (the pack's own size)"


def _stock_charge_line(hw: Hardware, bf16_gb: float, stock_loader: str,
                       tensor_bytes: float | None) -> str:
    """One human line naming what the fit check charged and against what."""
    if hw.memory_kind != "unified":
        term = "x1 (VRAM is its own pool)"
    elif stock_loader == STOCK_LOADER_STREAM:
        term = (f"+ one tensor ({_gb(tensor_bytes)} GB, streaming loader)" if tensor_bytes
                else "+ one tensor (5 GiB shard bound: snapshot not cached, headers unread; "
                     "streaming loader)")
    else:
        term = "x2 (from_pretrained's transient on unified memory, measured)"
    charge = stock_transient_bytes(bf16_gb * GB, 0.0, hw.memory_kind or "vram",
                                   direct=stock_loader == STOCK_LOADER_STREAM,
                                   tensor_bytes=tensor_bytes)
    ceiling = _stock_ceiling_gb(hw)
    return (f"{bf16_gb:.2f} GB bf16 {term} = {_gb(charge)} GB x{RATIO_HEADROOM} headroom "
            f"against {ceiling} {'GB' if ceiling is not None else ''} {hw.memory_kind or '?'}")


def _device_fallback_refusal(hw: Hardware) -> str | None:
    """A discrete AMD card `detect()` could only name via the pci-id
    fallback (see detect.Hardware.device_source) must not silently write
    `<model>_unknown-device_<date>.json`.
    None when there is nothing to refuse."""
    if hw.device_source != "fallback":
        return None
    return (f"device name is only the pci-id fallback ({hw.device_class!r}) — seen: "
           f"{hw.gpu_info}. Add an entry to detect._AMD_PCI_NAMES "
           "(src/drinkme/detect.py) for this pci id — see docs/bench.md — "
           "or pass --allow-unknown-device to write the record anyway (the "
           "slug carries the pci id, never the word 'unknown').")


def run(publishable_only_detect: bool = False, model_name: str | None = None,
        no_gemma: bool = False, runtime: str | None = None,
        allow_unknown_device: bool = False, on_point=None,
        stock_loader: str = STOCK_LOADER_STREAM,
        pack_dir: str | None = None, no_spec: bool = False) -> dict | list[dict]:
    """`runtime` is --runtime (torch | mlx): which arms module runs —
    arms.py on CUDA/ROCm, arms_mlx.py on Apple silicon (docs/metal.md).
    serve.resolve_runtime's precedence: CLI, DRINKME_RUNTIME, host.
    `stock_loader` is --stock-loader (stream | from_pretrained, torch runtime
    only): how the stock arm loads and therefore what its fit check charges
    (see _stock_fit_ok). `no_spec` is --no-spec: skip the speculation pass
    (spec_pass.py)."""
    from .serve import resolve_runtime

    runtime = resolve_runtime(runtime)
    force_ipv4()

    hw = detect()
    print("── hardware ──")
    print(f"  device      {hw.device_class or '?(could not normalize)'}")
    print(f"  memory      {hw.memory_gb} GB {hw.memory_kind or '?'}")
    print(f"  budget      {hw.budget_gb} GB")
    for e in hw.evidence:
        print(f"  · {e}")
    if hw.heuristic and hw.memory_kind:
        if not _confirm(f"memory looks {hw.memory_kind.upper()} — correct?"):
            hw.memory_kind = "unified" if hw.memory_kind == "vram" else "vram"
            print(f"  -> recorded as {hw.memory_kind} (your call, kept verbatim)")
    if hw.budget_gb is None:
        # this machine (or its environment) is unreadable — exitcodes.CantRunHere
        raise exitcodes.CantRunHere(
            "drinkme bench: no memory budget detected — cannot pick a model; run on a "
            "machine with a CUDA/ROCm GPU or Apple silicon, or file an issue with your setup")

    # Refuse to WRITE a record off a device name that is only the
    # pci-id fallback — before the bandwidth probe or any arm runs, not
    # after. --detect-only never writes a record, so it is exempt (its own
    # "no torch, no probe" promise already lets an operator see this exact
    # refusal reason without spending any GPU time on a run that would only
    # be refused at the end anyway).
    if not publishable_only_detect:
        refusal = _device_fallback_refusal(hw)
        if refusal and not allow_unknown_device:
            # fixable by a flag (--allow-unknown-device): exitcodes.Usage
            raise exitcodes.Usage(f"drinkme bench: {refusal}")

    # --no-gemma: the gate probe sends the user's HF token (see suggest.py's
    # gemma_gate_open docstring) — skip it entirely rather than merely
    # discarding the result. Gemma is simply not offered. --detect-only makes
    # the same promise in its help string ("no torch, no probe"), so it skips
    # the probe too; its menu shows gemma as not offered rather than lying
    # about having checked.
    gemma_ok = False if (no_gemma or publishable_only_detect) else gemma_gate_open()
    # hw.budget_gb is decimal GB — the same convention as suggest()'s menu
    # sizes (suggest.Sizes); no conversion at this boundary.
    s: Suggestion = _menu_suggestion(hw, runtime, gemma_ok)
    print("── menu ──")
    if s.ratio:
        print(f"  ratio point  {s.ratio.name}  (stock + compressed both run{_bf16_part(s, s.ratio)})")
    for m in s.ratio_extra:
        print(f"  also ratio   {m.name}  ({_bf16_part(s, m).lstrip('; ')})")
    if s.fit:
        tag = " [knife-edge]" if s.fit_knife_edge else (
            " [fit-cliff experiment]" if s.fit_honest_negative else "")
        print(f"  fit point    {s.fit.name}{tag}  (stock does not fit; compressed does)")
        sz = _point_sizes(s, s.fit)
        if sz:
            print(f"               compressed {sz.comp_label()}")
    for n in s.notes:
        print(f"  ! {n}")

    if publishable_only_detect:
        return {"environment": hw.as_record_env(), "detector": hw.as_record_detector(),
                "gemmaOk": gemma_ok}

    result = {
        "$type": MEASUREMENT,
        "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": __version__,
        "environment": hw.as_record_env(),
        "metrics": [],
    }

    if s.ratio is None:
        if runtime == "mlx":
            from .arms_mlx import measure_bandwidth
        else:
            from .probe import measure_bandwidth  # imports torch — after the cheap part

        print("── bandwidth probe (pure read, up to 4 GiB, empty device) ──")
        bw = measure_bandwidth()
        print(f"  read {_gb_s(bw['read_bytes_s'])} GB/s · copy {_gb_s(bw['copy_bytes_s'])} GB/s")
        result["bandwidth"] = bw
        why = "no-ratio-model-fits"
        print("── arms ──")
        print(f"  {why} — no decode arms ran. This record carries the hardware")
        print("  identity + measured bandwidth only, and is NOT publishable as a")
        print("  curve point. (Honesty over theater.)")
        result["incomplete"] = why
        result["raw"] = {"detector": hw.as_record_detector()}
        return result

    # ── the point(s): bare `bench` (no --model) runs BOTH menu points
    # that exist (ratio, then fit), each its own record.
    if model_name:
        from .suggest import MODELS

        by = {x.name: x for x in MODELS} | {x.hf_repo: x for x in MODELS}
        if model_name not in by:
            raise exitcodes.Usage(
                f"drinkme bench: unknown model {model_name!r}; menu models: "
                f"{', '.join(x.name for x in MODELS)}")
        points = [by[model_name]]
        print(f"  (menu overridden: running {points[0].name} by request)")
    else:
        points = [m for m in (s.ratio, s.fit) if m is not None]

    # Each point's record is handed to `on_point` the moment it exists —
    # never held until the whole menu finishes, so a later point that
    # crashes (the 8B fit point on an RX 7600 XT) cannot take an
    # already-measured ratio record with it.
    if pack_dir and (runtime == "mlx" or len(points) != 1):
        raise exitcodes.Usage(
            "drinkme bench: --pack-dir needs the torch runtime and exactly one --model point")
    if pack_dir:
        # the front door serve uses, torch-free, BEFORE any arm runs (so a
        # pack this build refuses is refused before the stock and twin arms
        # have been paid for): the pack exists, this build reads its format,
        # it was cut from this model
        from .serve import check_pack_dir

        if not os.path.exists(os.path.join(pack_dir, "meta.json")):
            # a bad/missing --pack-dir, the same bucket `verify --pack-dir`
            # is in: exitcodes.Usage
            raise exitcodes.Usage(f"drinkme bench: no pack at {pack_dir}")
        check_pack_dir(points[0].hf_repo, points[0].revision, pack_dir, verb="drinkme bench")
    elif runtime != "mlx":
        # THE FRONT DOOR, before any arm downloads a checkpoint: without
        # --pack-dir the torch runtime's compressed arm packs IN MEMORY
        # (arms.load_compressed_streaming -> swap.make_compressed ->
        # pack.pack_weight_served), which needs the native encoder exactly
        # as `drinkme pack` does. The mlx runtime never packs in memory
        # (arms_mlx.run_arms reads an existing pack_dir, or refuses on its
        # own when there is none) so it is exempt here.
        from .codec.radix_pack import refuse_unless_encoder_available

        refuse_unless_encoder_available(lead_in="drinkme bench")
    results = _run_points(points, lambda m: _run_point(hw, m, runtime, base_result=dict(result),
                                                       stock_loader=stock_loader, pack_dir=pack_dir,
                                                       no_spec=no_spec),
                          on_point)
    return results[0] if len(results) == 1 else results


def apple_silicon() -> bool:
    """Darwin on arm64: the host serve.resolve_runtime gives the mlx runtime."""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def record_summary(result: dict) -> list[str]:
    """The lines printed under a written record's path: whether
    `drinkme publish` will send it."""
    if "incomplete" in result:
        return []
    if (result.get("raw") or {}).get("unpublishable"):
        # the operator keeps their numbers; the public network never sees them
        return ["   that file is your local record; `drinkme publish` will refuse it "
                f"(raw.unpublishable: {result['raw']['unpublishable']})"]
    return ["   that file is your record: `drinkme publish` sends exactly those bytes"]


def _run_points(points, run_one, on_point=None) -> list:
    """Run the menu's points in order, handing each record to `on_point` the
    moment it exists (so a later point's crash cannot lose it)."""
    results = []
    for m in points:
        r = run_one(m)
        if on_point is not None:
            on_point(r)
        results.append(r)
    return results


def _point_sizes(s: Suggestion, m):
    """A menu point's sizes: the ones suggest() read, else read now."""
    from .suggest import sizes

    return s.sizes.get(m.name) or sizes(m, vision=True)


def _bf16_part(s: Suggestion, m) -> str:
    sz = _point_sizes(s, m)
    return f"; checkpoint {sz.bf16_gb:.2f} GB bf16" if sz else ""


def _sizes_or_exit(m, pack_dir: str | None = None):
    """suggest.sizes for a point bench is about to run, with its tower
    (every arm loads it, arms._TOWER_BY_DEFAULT): the compressed size is the
    pack's own when `pack_dir` (--pack-dir) or this row's default pack
    holds one, else the estimate. Exits when the checkpoint cannot be sized:
    its metadata is neither cached here nor on a Hub that answers, so its
    weights could not be downloaded either."""
    from .suggest import sizes

    sz = sizes(m, vision=True, pack_dir=pack_dir)
    if sz is None:
        # HF/network unreachable: exitcodes.CantRunHere
        raise exitcodes.CantRunHere(
            f"drinkme bench: cannot size {m.name} ({m.hf_repo}@{(m.revision or 'main')[:8]}): "
            "its checkpoint's metadata is not in the local Hugging Face cache and the "
            "Hub did not answer. Reconnect, or download the checkpoint first.")
    return sz


def _run_point(hw: Hardware, m, runtime: str, base_result: dict,
               stock_loader: str = STOCK_LOADER_STREAM, pack_dir: str | None = None,
               no_spec: bool = False) -> dict:
    """One measurement point (base_result carries $type/createdAt/version/
    environment, already built by run()). A STOCK-ARM
    fit check (drinkme.fit.stock_arm_fits, off the detect()-time snapshot)
    decides whether the stock pass runs at all — a predicted non-fit routes
    straight to the fit-point shape (stock.outcome skipped_predicted_nonfit,
    stock metrics OMITTED, twin+compressed still measured) instead of
    crashing into it. arms.py's own live guard (refuse_if_stock_wont_fit)
    is the second, in-process line of defense right before the allocation;
    its refusal, or an OOM out of the load, lands as stock.outcome
    failed_load with the message in stock.error and the compressed arm
    still runs."""
    if runtime == "mlx":
        from .arms_mlx import run_arms
    else:
        from .arms import run_arms

    result = dict(base_result)
    sz = _sizes_or_exit(m, pack_dir)
    # the streaming loaders' one-tensor term, off the cached snapshot's
    # headers when the checkpoint is already here (None = 5 GiB shard bound)
    tensor_bytes = _largest_tensor_bytes(m) if runtime != "mlx" else None
    stock_ok = runtime != "mlx" and _stock_fit_ok(hw, sz.bf16_gb, stock_loader, tensor_bytes)
    # The twin streams too (arms.load_twin_streaming): one bf16 copy
    # resident + one tensor on unified memory — it fits iff that fits. A
    # fit point whose bf16 does not fit at all is compressed-only (RX 7600 XT:
    # the 8B's twin pass OOMs a 16 GB card where the stock fit check
    # correctly steps aside).
    twin_ok = runtime == "mlx" or stock_ok or _twin_fit_ok(hw, sz.bf16_gb, tensor_bytes)
    # The compressed arm streams (arms.load_compressed_streaming): the
    # pack's resident bytes + one tensor — never the whole bf16 model on
    # the host, which a load_cpu + swap + .cuda() path would hold. Its own
    # fit check, same ceiling.
    comp_ok, comp_at_all = ((True, True) if runtime == "mlx"
                            else _compressed_fit(hw, sz.comp_gb, tensor_bytes))
    print(f"── arms: {m.name} ({m.hf_repo}@{(m.revision or 'main')[:8]}) [{runtime}] ──")
    if runtime == "mlx":
        print("  pass 1 stock bf16 · pass 2 twin (the pack decoded once at load) · pass 3 compressed")
    elif stock_ok:
        print(f"  pass 1 stock bf16 ({stock_loader} loader) · pass 2 compressed · "
              "pass 3 order-matched twin")
        print(f"  stock-arm fit check: {_stock_charge_line(hw, sz.bf16_gb, stock_loader, tensor_bytes)} — fits")
    else:
        print(f"  stock-arm fit check: {_stock_charge_line(hw, sz.bf16_gb, stock_loader, tensor_bytes)} — "
              "predicted NOT to fit; routing to the fit-point record "
              f"({'pass 1 compressed · pass 2 twin only' if twin_ok else 'compressed arm only'})")
    if runtime != "mlx" and not stock_ok:
        print(f"  twin fit check: {_streamed_charge_line(hw, sz.bf16_gb, 'bf16', RATIO_HEADROOM, tensor_bytes)}"
              f" — {'fits' if twin_ok else 'one bf16 copy does not fit either — compressed arm only'}")
    if runtime != "mlx":
        comp_line = _streamed_charge_line(hw, sz.comp_gb, _comp_what(sz), FIT_HEADROOM, tensor_bytes)
        if comp_ok:
            print(f"  compressed-arm fit check: {comp_line} — fits")
        elif comp_at_all:
            print(f"  compressed-arm fit check: {comp_line} — fits without the headroom; "
                  "attempting (knife-edge: it may run out of memory)")
        else:
            # No fallback arm: a compressed arm that cannot fit at all is
            # a point this machine cannot take, and the honest output is that
            # sentence, before any load, not a record. (suggest() never
            # picks such a fit point; a `--model` override can name one.)
            raise exitcodes.CantRunHere(
                f"drinkme bench: refusing to run: {m.name} — compressed-arm fit check: {comp_line} "
                "— does not fit even without headroom. The compressed arm has no fallback, "
                "so no record can come of this point on this machine; nothing was loaded.")
    # stock_memory_kind/stock_bf16_bytes ride along even when stock_ok is
    # True (torch runtime only — see run_stock above): arms.py's own live
    # guards (refuse_if_stock_wont_fit, refuse_if_compressed_wont_fit) are
    # a SECOND check, against live memory right before each allocation,
    # not just this detect()-time snapshot — the two can disagree
    # (Strix Halo: "three browser tabs" between detect and load), and the
    # live one runs regardless of which way this one landed.
    kw = ({} if runtime == "mlx" else
         {"run_stock": stock_ok, "stock_memory_kind": hw.memory_kind or "vram",
          "stock_bf16_bytes": sz.bf16_gb * GB, "run_twin": twin_ok,
          # the loader pass 1 runs, and the same one-tensor term every
          # fit check charged, so arms' live guards charge what bench did
          "stock_loader": stock_loader, "stock_tensor_bytes": tensor_bytes,
          "compressed_bytes": sz.comp_gb * GB,
          # the speculation pass after each measured stock and compressed
          # arm (spec_pass.py), unless --no-spec
          "spec": not no_spec})
    if pack_dir:
        # --pack-dir: the compressed arm is the pack on disk through the
        # serve loader — every profile is one pack through one loader, the
        # stock and twin arms identical
        if not os.path.exists(os.path.join(pack_dir, "meta.json")):
            raise exitcodes.Usage(f"drinkme bench: no pack at {pack_dir}")
        kw["pack_dir"] = pack_dir
        print(f"  compressed arm: the pack at {pack_dir} (serve loader), not an in-memory re-pack")
    raw = run_arms(m.hf_repo, m.revision, PROMPT, **kw)
    if runtime == "mlx":
        # the MLX engine decodes one token per step (serving/engine_mlx.py):
        # there is no speculation to time, whatever --no-spec said
        raw["spec"] = {"skipped": spec_pass.SKIP_RUNTIME}
    # model.revision is the hub commit EVERY arm loaded (arms resolve one
    # snapshot before any arm or the tokenizer loads — the pack's bound
    # commit under --pack-dir, else the coordinate resolved once), never
    # the menu's coordinate: an unpinned row (revision None) would otherwise
    # publish no revision at all, and its stock and compressed arms could
    # read two commits. raw.revision keeps the coordinate asked for. The lexicon
    # requires it, as the 40-hex sha: a local checkpoint directory resolves
    # to a structural digest instead (checkpoint.resolved_revision's
    # `local-…`), which is an identity for this box's KV store but not one
    # anybody else can group or reproduce — so the typed field is OMITTED,
    # the digest stays in raw.resolved_revision, raw.unpublishable says why
    # and `drinkme publish` refuses the file by name (its door). The record
    # is still written: the operator's own numbers are theirs.
    if not raw.get("resolved_revision"):
        raise ValueError("arms reported no resolved_revision: the record cannot name the "
                         "commit its arms loaded")
    resolved = str(raw["resolved_revision"])
    result["model"] = {"name": m.name, "hfRepo": m.hf_repo}
    if is_commit_sha(resolved):
        result["model"]["revision"] = resolved
    else:
        raw["unpublishable"] = UNPUBLISHABLE_NO_REVISION
    # compression.profile comes FROM the loaded arm (arms.arm_compression_profile off
    # each swapped tensor's codec/profile; arms_mlx off the loaded pack's
    # meta.json) — the writer's own string, never a literal here (a literal
    # would label every record with one profile whatever the arm and the
    # writer built). bitsPerWeight is WEIGHT-weighted (total encoded payload bits
    # over total packed weights); meanTensorBitsPerWeight is the unweighted
    # mean over tensors.
    # Both are over the packed population only — the Linears the codec
    # swapped, never the raw remainder or an MTP head.
    result["compression"] = {
        "profile": raw["compression_profile"],
        "bitsPerWeight": str(raw["weighted_bpw"]),
        "meanTensorBitsPerWeight": str(raw["mean_bpw"]),
    }
    # packedBytes / residentBytes: read off the loaded pack's meta.json (the
    # numbers `drinkme pack` wrote), so present iff the compressed arm was a
    # pack on disk — every mlx run, the torch arm under --pack-dir; the
    # in-memory torch re-pack has no meta.json and publishes neither.
    pack = raw.get("pack") or {}
    for key in ("packedBytes", "residentBytes"):
        if pack.get(key) is not None:
            result["compression"][key] = int(pack[key])
    # The per-tensor bitwise round trip is a GATE, not a field (arms.run_arms
    # raises before returning if any tensor fails it), so a record that
    # exists here already passed — a field that can only ever be true is
    # not a metric.
    # stock: the stock arm's OUTCOME as an evidence category — `measured`,
    # `skipped_predicted_nonfit` (a fit-policy decision against a budget,
    # named with that budget in bytes), `failed_load` (an attempted
    # load that the live guard refused or that OOM'd, with the message) —
    # one object, the budget or the error beside it. The twin's outcome
    # stays in raw.twin_outcome; its decode is a metric (below).
    stock_outcome = raw["stock_outcome"]
    stock: dict = {"outcome": stock_outcome}
    if stock_outcome == "skipped_predicted_nonfit":
        stock["budgetBytes"] = _stock_ceiling_bytes(hw)
    elif stock_outcome == "failed_load":
        stock["error"] = str(raw["stock_error"])[:2048]
    result["stock"] = stock
    stock_ran = stock_outcome == "measured"
    if stock_ran and raw.get("stock_decode_tok_s") is None:
        raise ValueError("arms reported stock_outcome 'measured' with no stock decode metric")
    bwd = raw["bandwidth"]
    metrics = []
    if stock_ran:
        metrics += [
            _metric("stock_decode_tok_s", raw["stock_decode_tok_s"], "tok/s",
                    raw["stock_decode_samples"]),
            _metric("stock_prefill_tok_s", raw["stock_prefill_tok_s"], "tok/s",
                    raw["stock_prefill_samples"]),
            _metric("stock_ttft_s", raw["stock_ttft_s"], "s", raw["stock_ttft_samples"]),
            _metric("stock_weights_gb", _gb(raw["vram_bf16_bytes"]), "GB"),
        ]
        metrics += _decode_read_metric(raw, "stock")
    # The twin: on torch, the same bf16 bytes through the compressed arm's
    # own kernels with the decode replaced by a raw load, at the box's twin
    # rows (swap.RadixTwinLinear), so compressed / twin isolates what
    # reading fewer bytes buys: the diagnostic we tune against, kept as
    # data. The results site divides by stock, what a user runs without
    # the compression. Decode only (arms.py's docstring). On mlx "twin" is
    # the reference path's arithmetic over the pack decoded once at load
    # (engine_mlx.RadixTwinLinear: a correctness check), not a baseline,
    # so it stays in raw there.
    twin_ran = runtime != "mlx" and raw.get("twin_outcome") == "measured"
    if twin_ran:
        if raw.get("twin_decode_tok_s") is None or raw.get("vram_twin_bytes") is None:
            raise ValueError("arms reported twin_outcome 'measured' with no twin decode or footprint")
        metrics += [
            _metric("twin_decode_tok_s", raw["twin_decode_tok_s"], "tok/s",
                    raw["twin_decode_samples"]),
            _metric("twin_weights_gb", _gb(raw["vram_twin_bytes"]), "GB"),
        ]
        metrics += _decode_read_metric(raw, "twin")
    metrics += [
        _metric("compressed_decode_tok_s", raw["compressed_decode_tok_s"], "tok/s",
                raw["compressed_decode_samples"]),
        _metric("compressed_prefill_tok_s", raw["compressed_prefill_tok_s"], "tok/s",
                raw["compressed_prefill_samples"]),
        _metric("compressed_ttft_s", raw["compressed_ttft_s"], "s",
                raw["compressed_ttft_samples"]),
        # THE RECORD BOUNDARY for memory and bandwidth: raw carries bytes
        # and bytes/second; the metric carries decimal GB / GB/s, so the
        # unit string is true.
        _metric("read_gb_s", _gb_s(bwd["read_bytes_s"]), "GB/s"),
        _metric("copy_gb_s", _gb_s(bwd["copy_bytes_s"]), "GB/s"),
        _metric("compressed_weights_gb", _gb(raw["vram_compressed_bytes"]), "GB"),
    ]
    metrics += _decode_read_metric(raw, "compressed")
    # decode with speculation on, per arm and prompt (spec_pass.py): torch
    # only, when the engine loaded a head, stock_* only for a measured stock arm
    metrics += [_metric(name, value, "tok/s", samples)
                for name, value, samples in spec_pass.metrics(raw.get("spec"), stock_ran)]
    result["metrics"] = metrics
    # raw is bench's own working detail (the lexicon declares it `unknown`):
    # arms' report as returned, plus how detect() named the device — the
    # input to _device_fallback_refusal, kept beside the record it gated —
    # and the detector's working notes
    result["raw"] = raw
    raw["device_source"] = hw.device_source
    raw["detector"] = hw.as_record_detector()
    result["environment"] = dict(result["environment"])
    # environment.engine names the runtime AND the kernels every arm ran,
    # composed from the routing record the arms agreed on (the MLX arms
    # are not routed: their form names the runtime only)
    result["environment"].update(_engine_env(runtime, None if runtime == "mlx" else record_routing(raw),
                                             None if runtime == "mlx" else record_decode_step(raw)))

    twin_txt = (f"twin {raw['twin_decode_tok_s']}" if raw.get("twin_decode_tok_s") is not None
                else f"twin {raw['twin_outcome']}")
    stock_txt = (f"stock {raw['stock_decode_tok_s']}" if stock_ran
                 else f"stock {stock_outcome} ({raw.get('stock_error')})")
    # compressed / stock first: the site's ratio; compressed / twin beside it
    ratios = [f"{raw['compressed_decode_tok_s'] / raw[f'{arm}_decode_tok_s']:.3f}x over {label}"
              for arm, label, ran in (("stock", "stock", stock_ran), ("twin", "the twin", twin_ran)) if ran]
    print(f"  {stock_txt} · {twin_txt} · compressed {raw['compressed_decode_tok_s']} tok/s"
          + (f" -> {', '.join(ratios)}" if ratios else ""))
    for line in spec_pass.summary_lines(raw.get("spec"), stock_ran):
        print(f"  {line}")
    host = raw.get("host_load")
    if host:
        # the host's 1-minute load before -> after each arm's timed passes (probe.HOST_LOAD_WARN_PER_CPU)
        def fmt(v):
            return "?" if v is None else f"{v:.1f}"
        print(f"  host load (1-minute average) while timed, {host['cpu_count']} CPUs: " + " · ".join(
            f"{arm} {fmt(a)}->{fmt(b)}" for arm, (a, b) in host["load1"].items()))
        warning = host_load_warning(host)
        if warning:
            print(f"  warning: {warning}")
    if runtime != "mlx":
        print("  of each arm's bandwidth bound (read / decode-read bytes): " + " · ".join(
            f"{arm} {frac:.0%}" for arm, frac in bound_fractions(result["metrics"])))
    if runtime == "mlx":
        # the Apple discipline (arms_mlx.py): minima beside medians, and the
        # drift band printed where the ratio is, so a throttled run cannot
        # masquerade as a slow kernel
        print(f"  min-of-n ratio {raw['ratio_min_of_n']}x · drift band "
              f"{raw['drift_band_pct']}% · compressed path: {raw['compressed_path']}")
    else:
        # the kernel routes beside the ratio they condition: every arm on the same DeltaNet recurrence kernel and narrow
        # GEMV, or arms.run_arms refused this record before it got here
        routed = [(arm, raw.get(f"{arm}_routing")) for arm in ("stock", "twin", "compressed")]
        print("  routing: " + " / ".join(f"{arm} {describe_route(rec)}"
                                         for arm, rec in routed if rec is not None))
    evidence = {"roundtrip": "verified by a fresh round trip",
                "pack": "verified (the pack's hashes matched at load)"}.get(raw.get("verification"), "verified")
    print(f"  read {_gb_s(bwd['read_bytes_s'])} GB/s · bit-exact gate: "
          f"{raw['verified_tensors']}/{raw['swapped_linears']} tensors {evidence}"
          f" (a record is never written otherwise) · "
          f"{raw['weighted_bpw']} bpw (weighted; {raw['mean_bpw']} mean over tensors)")
    if "unpublishable" in raw:
        print(f"  source {resolved}: {raw['unpublishable']}")
    return result


# raw.unpublishable on a record whose arms loaded a checkpoint that is not a
# hub snapshot: why the typed model.revision is absent and what a
# publishable record needs. The publish door (publish/validate.
# revision_verdict) says the same thing when it refuses the file.
UNPUBLISHABLE_NO_REVISION = (
    "no resolved revision: the weights came from a local checkpoint directory, which "
    "has a structural digest (raw.resolved_revision) but no hub commit to name in "
    "model.revision, so this record is the operator's own and `drinkme publish` will "
    "refuse it — bench against a hub repo for a publishable record")


PROMPT = "The key idea of lossless weight compression is"

# (arm, decode metric, decode-read metric): the arms whose decode speed has
# a bandwidth bound, read_gb_s / decode_read_gb — the bytes one decode step
# reads (arms.decode_read_bytes: the Linears outside the vision tower, one
# embedding row), not the resident weights_gb, which also counts the
# tower and the embedding rows a step does not read
BOUND_ARMS = (("stock", "stock_decode_tok_s", "stock_decode_read_gb"),
              ("twin", "twin_decode_tok_s", "twin_decode_read_gb"),
              ("compressed", "compressed_decode_tok_s", "compressed_decode_read_gb"))


def bound_fractions(metrics: list) -> list:
    """[(arm, decode tok/s over read_gb_s / decode_read_gb)] for each arm
    the metrics carry both for: how near the arm runs to reading what one
    decode step reads once per token at the machine's measured bandwidth.
    A record without an arm's decode-read metric (an arm that walks no
    module tree, _decode_read_metric) has no bound for that arm here: its
    weights_gb is the resident footprint, and dividing by it states a
    different quantity."""
    m = {x["name"]: float(x["value"]) for x in metrics}
    read = m.get("read_gb_s")
    return [(arm, m[tok] / (read / m[gb])) for arm, tok, gb in BOUND_ARMS
            if read and m.get(tok) is not None and m.get(gb)]


def _decode_read_metric(raw: dict, arm: str) -> list:
    """[`<arm>_decode_read_gb`] off raw.<arm>_bytes_per_token (arms.
    decode_read_bytes' total), or [] when the arm did not measure it (a
    runtime or a stub that walks no module tree)."""
    per = raw.get(f"{arm}_bytes_per_token") or {}
    if per.get("total_bytes") is None:
        return []
    return [_metric(f"{arm}_decode_read_gb", _gb(per["total_bytes"]), "GB")]


def _metric(name: str, value, unit: str, samples: list | None = None) -> dict:
    m = {"name": name, "value": str(value), "unit": unit}
    if samples:
        m["samples"] = [str(x) for x in samples]
    return m


def _gb(nbytes) -> float:
    """Bytes -> DECIMAL gigabytes (10**9), 3 places: the record's "GB". The
    one conversion for memory figures; raw keeps the bytes."""
    return round(nbytes / GB, 3)


def _gb_s(bytes_s) -> float:
    """Bytes/second -> decimal GB/s, 3 places: the record's "GB/s"."""
    return round(bytes_s / GB, 3)


ARMS = ("stock", "twin", "compressed")


def record_routing(raw: dict) -> dict:
    """The ONE kernel routing record a torch measurement has: every arm
    that ran carries serving/kernel_route.route_kernels's record as
    raw.<arm>_routing, and arms.refuse_unless_routed_alike has already
    refused a run whose arms disagree — so the arms present agree, and
    the first is the record's. Held here again at bench's own boundary:
    a report with no route at all, or two routes, makes no record (the
    engine string could not say which kernels ran the numbers)."""
    routes = {arm: raw.get(f"{arm}_routing") for arm in ARMS}
    routes = {arm: r for arm, r in routes.items() if r is not None}
    if not routes:
        raise ValueError("arms reported no routing: the record cannot say which kernels "
                         "ran (raw.<arm>_routing is missing on every arm)")
    if len({json.dumps(r, sort_keys=True) for r in routes.values()}) > 1:
        raise ValueError("arms did not route alike; the record cannot name one route: "
                         + "; ".join(f"{arm} {describe_route(r)}" for arm, r in routes.items()))
    return next(iter(routes.values()))


def record_decode_step(raw: dict) -> str:
    """The step path the record's decode figures ran on (arms.decode_step,
    raw.<arm>_decode_step): "cudagraph" when every arm that was timed
    replayed a CUDA graph, else "eager". Arms that disagree (a capture that
    failed on one arm only) make the record unpublishable, since its ratio
    would compare two step paths; raw keeps which arm ran which."""
    steps = {arm: raw[f"{arm}_decode_step"] for arm in ARMS if raw.get(f"{arm}_decode_step")}
    if len(set(steps.values())) > 1:
        why = ("the arms decoded on different step paths ("
               + ", ".join(f"{a} {v}" for a, v in steps.items())
               + "), so their ratio is not like for like")
        raw["unpublishable"] = f"{raw['unpublishable']}; {why}" if raw.get("unpublishable") else why
    if steps and all(v == "cudagraph" for v in steps.values()):
        return "cudagraph"
    return "eager"


# environment.engine: ONE grammar, slash-separated segments each starting
# with its noun, so the string is greppable by segment
# (lexicons/README.md, docs/bench.md#the-engine-string). After the
# version-carrying segments, every kernel segment is `<noun> <impl>[ <version>]`:
#   drinkme <semver> / torch <version>[+<accel tag>] / deltanet <fla <ver> | torch | none> / conv <fla <ver> | torch | none> / narrow-gemv <triton | torch> / stock-gemv <mv | linear | triton> / raw-gemv <twin | stock>[ / decode <eager | cudagraph>]
#   drinkme <semver> / mlx <version>
# The decode segment (the step path the decode figures ran on,
# serving/cudagraph.py) was appended after the seven-segment form was
# published; a record without it predates graph mode and decoded eager.
ENGINE_SEP = " / "
DECODE_STEPS = ("eager", "cudagraph")
DELTANET_KERNELS = ("fla", "torch", "none")
DELTANET_CONV_KERNELS = ("fla", "torch", "none")
# narrow-gemv's impl: drinkme's own Triton GEMV (swap.NarrowLinear) when the
# route is allowed, torch's F.linear under DRINKME_NARROW_GEMV=0
NARROW_GEMV_IMPLS = {True: "triton", False: "torch"}
# stock-gemv's impl: the one-row call of the raw Linears at or above the
# codec's row threshold (swap.STOCK_GEMV_MODES: torch.mv with the Lt BLAS
# library preferred for the call, F.linear on PyTorch's default library, or
# triton, the twin arm's Triton GEMV over the raw weight)
STOCK_GEMV_IMPLS = ("linear", "mv", "triton")
# raw-gemv's impl: the one-row call of the raw Linears a codec tree holds
# (the compressed and twin arms' tied lm_head, a raw fallback;
# swap.RAW_GEMV_MODES): the twin arm's Triton kernel, or the stock GEMV
RAW_GEMV_IMPLS = ("twin", "stock")


def engine_string(runtime: str, runtime_version: str, routing: dict | None = None,
                  fla_version: str | None = None, decode: str | None = None) -> str:
    """Compose the record's environment.engine. `runtime_version` is the
    runtime's own version string verbatim (torch.__version__ carries the
    accelerator tag: 2.12.0+rocm7.1, 2.12.0+cu128). For torch, `routing`
    is record_routing's dict and the kernel segments are read off it —
    including `deltanet_conv` (the prefill convolution route: kernel_route.
    route_kernels always supplies it; a routing dict without the key is
    refused rather than silently written as `conv torch`); `fla_version`
    names the fla that ran, carried by every segment whose impl is fla
    (`deltanet fla <ver>`, `conv fla <ver>`); `stock_gemv` (the raw
    Linears' one-row call) and `raw_gemv` (a codec tree's raw
    Linears' one-row call) are refused when missing the same way.
    `decode` (record_decode_step's answer) appends the decode segment;
    None leaves the seven-segment form. mlx: the runtime only —
    environment.platform already says metal."""
    head = f"drinkme {__version__}{ENGINE_SEP}{runtime} {runtime_version}"
    if runtime == "mlx":
        return head
    if routing is None:
        raise ValueError("engine_string: the torch runtime's engine string needs the routing record")
    kernel = routing["deltanet_kernel"]
    if kernel not in DELTANET_KERNELS:
        raise ValueError(f"engine_string: deltanet kernel {kernel!r} is not one of {DELTANET_KERNELS}")
    if "deltanet_conv" not in routing:
        raise ValueError("engine_string: the routing record has no deltanet_conv — "
                         "route_kernels must supply it, or the record cannot say which "
                         "prefill convolution kernel ran")
    conv = routing["deltanet_conv"]
    if conv not in DELTANET_CONV_KERNELS:
        raise ValueError(f"engine_string: deltanet_conv {conv!r} is not one of {DELTANET_CONV_KERNELS}")
    if "stock_gemv" not in routing:
        raise ValueError("engine_string: the routing record has no stock_gemv — "
                         "route_kernels must supply it, or the record cannot say which "
                         "call the raw Linears' decode made")
    stock = routing["stock_gemv"]
    if stock not in STOCK_GEMV_IMPLS:
        raise ValueError(f"engine_string: stock_gemv {stock!r} is not one of {STOCK_GEMV_IMPLS}")
    if "raw_gemv" not in routing:
        raise ValueError("engine_string: the routing record has no raw_gemv — "
                         "route_kernels must supply it, or the record cannot say which "
                         "call the codec tree's raw Linears made")
    raw = routing["raw_gemv"]
    if raw not in RAW_GEMV_IMPLS:
        raise ValueError(f"engine_string: raw_gemv {raw!r} is not one of {RAW_GEMV_IMPLS}")

    def impl(name: str) -> str:
        return f"fla {fla_version or '?'}" if name == "fla" else name

    if decode is not None and decode not in DECODE_STEPS:
        raise ValueError(f"engine_string: decode {decode!r} is not one of {DECODE_STEPS}")
    narrow = f"narrow-gemv {NARROW_GEMV_IMPLS[bool(routing['narrow_gemv'])]}"
    segments = (head, f"deltanet {impl(kernel)}", f"conv {impl(conv)}", narrow,
                f"stock-gemv {stock}", f"raw-gemv {raw}")
    if decode is not None:
        segments += (f"decode {decode}",)
    return ENGINE_SEP.join(segments)


def _engine_head(engine: str) -> tuple[list[list[str]], dict]:
    segments = engine.split(ENGINE_SEP)
    words = [seg.split(" ") for seg in segments]
    if words[0][:1] != ["drinkme"] or len(words[0]) != 2:
        raise ValueError(f"engine {engine!r}: the first segment is not 'drinkme <semver>'")
    if len(words) < 2 or len(words[1]) != 2 or not words[1][1]:
        raise ValueError(f"engine {engine!r}: the second segment is not '<runtime> <version>'")
    return words, {"drinkme": words[0][1], "runtime": words[1][0], "runtime_version": words[1][1]}


def _check_deltanet_pair(engine: str, out: dict) -> dict:
    if (out["deltanet_kernel"] == "none") != (out["deltanet_conv"] == "none"):
        raise ValueError(f"engine {engine!r}: deltanet {out['deltanet_kernel']!r} and conv "
                         f"{out['deltanet_conv']!r} disagree on whether the model has "
                         "DeltaNet layers")
    return out


def parse_engine(engine: str) -> dict:
    """engine_string's inverse: the segments back to their fields —
    {drinkme, runtime, runtime_version} plus, for torch, {deltanet_kernel,
    fla_version (when any segment's impl is fla), deltanet_conv,
    narrow_gemv, stock_gemv, raw_gemv}, and `decode` when the string
    carries the decode segment (eager | cudagraph). Raises
    ValueError for anything outside the grammar, for a
    `deltanet`/`conv` pair that disagrees on whether the model has DeltaNet
    layers (`deltanet none` must pair with `conv none`), and for two fla
    segments naming different fla versions, so a consumer grouping on it
    cannot misread one."""
    words, out = _engine_head(engine)
    if out["runtime"] == "mlx":
        if len(words) != 2:
            raise ValueError(f"engine {engine!r}: the mlx form is 'drinkme <semver> / mlx <version>'")
        return out
    if len(words) not in (7, 8):
        raise ValueError(f"engine {engine!r}: the torch form has seven segments, or eight with decode")
    fla_versions = set()
    for seg, noun, key in ((words[2], "deltanet", "deltanet_kernel"), (words[3], "conv", "deltanet_conv")):
        if seg[:1] != [noun]:
            raise ValueError(f"engine {engine!r}: expected the {noun} segment, got {' '.join(seg)!r}")
        if len(seg) == 3 and seg[1] == "fla" and seg[2]:
            fla_versions.add(seg[2])
        elif len(seg) != 2 or seg[1] not in ("torch", "none"):
            raise ValueError(f"engine {engine!r}: the {noun} segment is not "
                             f"'{noun} fla <ver>|torch|none'")
        out[key] = seg[1]
    if len(fla_versions) > 1:
        raise ValueError(f"engine {engine!r}: the fla segments name different fla versions")
    if fla_versions:
        out["fla_version"] = fla_versions.pop()
    impls = {v: k for k, v in NARROW_GEMV_IMPLS.items()}
    narrow = words[4]
    if len(narrow) != 2 or narrow[0] != "narrow-gemv" or narrow[1] not in impls:
        raise ValueError(f"engine {engine!r}: the fifth segment is not 'narrow-gemv triton|torch'")
    out["narrow_gemv"] = impls[narrow[1]]
    stock = words[5]
    if len(stock) != 2 or stock[0] != "stock-gemv" or stock[1] not in STOCK_GEMV_IMPLS:
        raise ValueError(f"engine {engine!r}: the sixth segment is not 'stock-gemv mv|linear|triton'")
    out["stock_gemv"] = stock[1]
    raw = words[6]
    if len(raw) != 2 or raw[0] != "raw-gemv" or raw[1] not in RAW_GEMV_IMPLS:
        raise ValueError(f"engine {engine!r}: the seventh segment is not 'raw-gemv twin|stock'")
    out["raw_gemv"] = raw[1]
    if len(words) == 8:
        decode = words[7]
        if len(decode) != 2 or decode[0] != "decode" or decode[1] not in DECODE_STEPS:
            raise ValueError(f"engine {engine!r}: the eighth segment is not 'decode eager|cudagraph'")
        out["decode"] = decode[1]
    return _check_deltanet_pair(engine, out)


def _engine_env(runtime: str = "torch", routing: dict | None = None,
                decode: str | None = None) -> dict:
    """The lexicon's `engine` and `platform` environment fields — the one
    place the record says which COMPUTE PLATFORM ran the decode (the
    lexicon's word for it is `platform`: cuda | rocm | metal) and, in `engine`, which runtime
    and which kernels (engine_string over `routing`, record_routing's
    answer for the torch arms) — read from the runtime (torch | mlx) that
    just ran the arms (so only ever called after run_arms).

    The mlx runtime's code never imports torch, and a metal environment made before
    the lane carried torch has none (docs/metal.md): asking torch here would
    crash `bench --runtime mlx` AFTER all three arms have run — the whole
    measurement done, then lost assembling this dict."""
    if runtime == "mlx":
        import mlx.core as mx
        return {"engine": engine_string("mlx", mx.__version__), "platform": "metal"}
    import torch

    if getattr(torch.version, "hip", None):
        platform = "rocm"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        platform = "metal"
    else:
        platform = "cuda"
    fla_version = None
    if routing is not None and "fla" in (routing["deltanet_kernel"], routing.get("deltanet_conv")):
        # the arms bound an fla kernel, so the import that route made succeeds here
        import fla
        fla_version = str(getattr(fla, "__version__", "?"))
    return {"engine": engine_string("torch", torch.__version__, routing, fla_version, decode),
            "platform": platform}


MEASUREMENTS_DIR = "measurements"


def _slug(text: str | None, fallback: str, limit: int = 48) -> str:
    """Filesystem-safe lowercase token: runs of anything but [a-z0-9.] become
    one '-', edges trimmed, cut at `limit` characters."""
    t = re.sub(r"[^a-z0-9.]+", "-", (text or "").lower()).strip("-.")
    return t[:limit].rstrip("-.") or fallback


def measurement_path(result: dict, root: str = ".") -> str:
    """Where `drinkme bench` keeps a record it just produced:
    `<root>/measurements/<model-short>_<device-slug>_<YYYY-MM-DD>[_<compression profile>][_<n>].json`
    (root = the current working directory: a clone user is in the checkout,
    a pip user gets it beside them). The default compression profile (sip)
    owns the bare name; any other profile appends itself (`_gulp`), so a
    same-day sip and gulp of one model never look like two runs of one thing
    (sip keeps the current name, gulp gets a suffix). The
    first free name is chosen — an existing file is never overwritten, the
    second run of a day is `_2`, the third `_3`. The directory is created
    here so the caller can open the path directly."""
    from .codec.radix_pack import DEFAULT_PROFILE  # lazy: keeps bench's import light
    model = _slug((result.get("model") or {}).get("name"), "incomplete")
    device = _slug((result.get("environment") or {}).get("deviceClass"), "unknown-device")
    day = (result.get("createdAt") or "")[:10] or datetime.date.today().isoformat()
    profile = (result.get("compression") or {}).get("profile")
    d = os.path.join(root, MEASUREMENTS_DIR)
    os.makedirs(d, exist_ok=True)
    stem = f"{model}_{device}_{day}"
    if profile and profile != DEFAULT_PROFILE:
        stem += f"_{_slug(profile, 'profile')}"
    n = 1
    while True:
        path = os.path.join(d, f"{stem}{'' if n == 1 else f'_{n}'}.json")
        if not os.path.exists(path):
            return path
        n += 1


def lexicon_safe(value):
    """The record as the atproto data model can carry it: floats become
    decimal strings (the data model has no float — a PDS refuses 1.5 outright
    and turns 124.0 into 124, measured against a self-hosted PDS; the lexicon's
    own numbers are decimal strings for the same reason) and None-valued
    keys are dropped (null is only legal where a lexicon says nullable).
    Integers, booleans, strings and nesting are untouched, so `raw` keeps
    every research-harness field, just spelled as strings."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        return {k: lexicon_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [lexicon_safe(v) for v in value]
    return value


def write_record(result: dict, path: str | None = None) -> str:
    """Write the record and return the path it went to. `-o FILE` wins;
    otherwise measurement_path() under the working directory. The bytes on
    disk are the record exactly as `drinkme publish` will send them, which
    is why they go through lexicon_safe() here and not at publish time."""
    path = path or measurement_path(result)
    with open(path, "w") as f:
        f.write(json.dumps(lexicon_safe(result), indent=2))
    return path


def main_json(path: str | None = None, model_name: str | None = None,
             no_gemma: bool = False, runtime: str | None = None,
             allow_unknown_device: bool = False,
             stock_loader: str = STOCK_LOADER_STREAM,
             pack_dir: str | None = None, no_spec: bool = False) -> None:
    """Bare `bench` (no --model) can return TWO records — its ratio
    point and its fit point, each its own file. `-o FILE` names a
    single explicit path, so it only applies when exactly
    one record came back; with two, each gets its usual auto-generated name
    and `-o` is reported as ignored rather than silently overwritten by the
    second record onto the first."""
    written = []

    def _sink(result):
        # write as each point lands (a later point's crash must not lose it);
        # -o applies only when exactly one record can come out of this run
        where = write_record(result, path if model_name else None)
        written.append(where)
        print(f"-> {where}")
        for line in record_summary(result):
            print(line)

    if path and not model_name:
        print(f"!! -o {path} ignored: a menu run can produce two records (ratio + fit "
              "point) — each gets its own auto-generated name")
    run(model_name=model_name, no_gemma=no_gemma, runtime=runtime,
        allow_unknown_device=allow_unknown_device, on_point=_sink,
        stock_loader=stock_loader, pack_dir=pack_dir, no_spec=no_spec)


def dry_run(model_name: str | None = None, runtime: str | None = None,
            stock_loader: str = STOCK_LOADER_STREAM) -> dict:
    """The plan `bench` would execute — arms, predicted bytes, which
    record shape, per point — with NO torch import at all. `detect()` and
    `suggest()` are both torch-free by construction (their own module
    docstrings); this function never imports arms/arms_mlx/probe, which is
    what actually pulls torch in. Also why cli.py checks --dry-run BEFORE
    bootstrap.ensure_accelerator ever runs — a plan must not be able to
    trigger a `uv sync` behind the box's back just by being asked for.
    Never probes the gemma gate, same as --detect-only: no token,
    regardless of any flag; the one network read a plan can make is a
    menu checkpoint's safetensors metadata, asked of the Hub anonymously
    when this machine has not cached it (suggest.checkpoint_bytes).
    `stock_loader` is the plan's
    stock-arm loader: the streaming one by default, whose one-tensor term is
    read off the cached snapshot's headers when the checkpoint is already on
    the machine (serving.checkpoint.largest_tensor_bytes — torch-free, and
    cached_snapshot_dir never touches the network)."""
    from .serve import resolve_runtime

    runtime = resolve_runtime(runtime)
    hw = detect()
    plan: dict = {"environment": hw.as_record_env(), "detector": hw.as_record_detector(),
                  "runtime": runtime, "stockLoader": stock_loader}
    refusal = _device_fallback_refusal(hw)
    if refusal:
        plan["deviceRefusal"] = refusal

    if model_name:
        from .suggest import MODELS

        by = {x.name: x for x in MODELS} | {x.hf_repo: x for x in MODELS}
        if model_name not in by:
            plan["error"] = (f"unknown model {model_name!r}; menu models: "
                             f"{', '.join(x.name for x in MODELS)}")
            plan["points"] = []
            return plan
        plan["points"] = [_point_plan(hw, by[model_name], stock_loader)]
        return plan

    if hw.budget_gb is None:
        plan["note"] = "no memory budget detected — nothing to plan"
        plan["points"] = []
        return plan
    s: Suggestion = _menu_suggestion(hw, runtime, gemma_ok=False)
    if s.ratio is None:
        plan["note"] = "no-ratio-model-fits"
        plan["points"] = []
        return plan
    plan["points"] = [_point_plan(hw, m, stock_loader) for m in (s.ratio, s.fit) if m is not None]
    return plan


def _point_plan(hw: Hardware, m, stock_loader: str = STOCK_LOADER_STREAM) -> dict:
    """One dry-run point: the stock-arm fit check verdict and the record
    shape it implies, plus the filename stem `measurement_path` would
    produce (recomputed here rather than called — `measurement_path` stats
    the filesystem to dodge a same-day collision, a real effect a dry run
    must not have; the stem is enough to show the plan). The plan
    names the stock loader and what its fit check charged — the transient
    (decimal GB, before headroom), the largest tensor it read off the
    cached snapshot (null when the checkpoint is not on the machine yet and the
    5 GiB shard bound was charged), the ceiling and the headroom — so an
    operator can check the arithmetic against their own MemAvailable."""
    from .suggest import sizes

    sz = sizes(m, vision=True)
    if sz is None:
        return {"model": m.name, "hfRepo": m.hf_repo, "revision": m.revision,
                "error": "cannot size this model: its checkpoint's metadata is not in the local "
                         "Hugging Face cache and the Hub did not answer"}
    ceiling_gb = _stock_ceiling_gb(hw)
    day = datetime.date.today().isoformat()
    tensor_bytes = _largest_tensor_bytes(m)
    stock_ok = _stock_fit_ok(hw, sz.bf16_gb, stock_loader, tensor_bytes)
    charge = (stock_transient_bytes(sz.bf16_gb * GB, 0.0, hw.memory_kind,
                                    direct=stock_loader == STOCK_LOADER_STREAM,
                                    tensor_bytes=tensor_bytes)
              if hw.memory_kind else None)
    # the twin and compressed arms stream: resident + one
    # tensor each, the same verdicts _run_point reaches — so the plan for a
    # fit point says which of them will run and what each is charged
    twin_ok = stock_ok or _twin_fit_ok(hw, sz.bf16_gb, tensor_bytes)
    comp_ok, comp_at_all = _compressed_fit(hw, sz.comp_gb, tensor_bytes)
    twin_charge = (streamed_transient_bytes(sz.bf16_gb * GB, 0.0, hw.memory_kind, tensor_bytes)
                   if hw.memory_kind else None)
    comp_charge = (streamed_transient_bytes(sz.comp_gb * GB, 0.0, hw.memory_kind, tensor_bytes)
                   if hw.memory_kind else None)
    if stock_ok:
        shape = "full (3 arms, stock.outcome measured)"
    elif twin_ok:
        shape = ("fit-point (2 arms: twin+compressed, stock.outcome "
                 "skipped_predicted_nonfit, stock metrics omitted)")
    else:
        shape = ("fit-point (1 arm: compressed only, stock.outcome skipped_predicted_nonfit, "
                 "raw.twin_outcome skipped_predicted_nonfit, stock metrics omitted)")
    if not comp_at_all:
        shape = ("none: the compressed arm is predicted not to fit even without headroom — "
                 "bench refuses this point before any load, and no record is written")
    return {
        "model": m.name,
        "hfRepo": m.hf_repo,
        "revision": m.revision,
        # the checkpoint's safetensors bytes; the compressed arm's resident
        # bytes, a pack's own or the estimate (suggest.sizes)
        "bf16GB": _gb(sz.bf16_gb * GB),
        "compressedGB": _gb(sz.comp_gb * GB),
        "compressedGBEstimated": sz.estimated,
        "compressedGBFrom": sz.comp_label(),
        "arms": (["stock"] if stock_ok else []) + (["twin"] if twin_ok else []) + ["compressed"],
        "stockArmPredictedToFit": stock_ok,
        "stockLoader": stock_loader,
        "stockArmChargeGB": _gb(charge) if charge is not None else None,
        "largestTensorGB": _gb(tensor_bytes) if tensor_bytes else None,
        "largestTensorBytes": int(tensor_bytes) if tensor_bytes else None,
        "stockArmCeilingGB": _gb(ceiling_gb * GB) if ceiling_gb is not None else None,
        "stockArmHeadroom": RATIO_HEADROOM,
        # the twin: one bf16 copy + one tensor (unified), the stock headroom
        "twinArmPredictedToFit": twin_ok,
        "twinArmChargeGB": _gb(twin_charge) if twin_charge is not None else None,
        # the compressed arm: the pack's resident bytes + one tensor
        # (unified), suggest's FIT_HEADROOM; fits without it but not with it
        # is the knife-edge case
        "compressedArmPredictedToFit": comp_ok,
        "compressedArmFitsWithoutHeadroom": comp_at_all and not comp_ok,
        "compressedArmChargeGB": _gb(comp_charge) if comp_charge is not None else None,
        "compressedArmHeadroom": FIT_HEADROOM,
        # every arm streams; only the stock arm has a selectable alternative
        "armLoaders": {"stock": stock_loader, "twin": STOCK_LOADER_STREAM,
                       "compressed": STOCK_LOADER_STREAM},
        "recordShape": shape,
        "predictedFilenameStem": f"{_slug(m.name, 'incomplete')}_"
                                 f"{_slug(hw.device_class, 'unknown-device')}_{day}",
    }
