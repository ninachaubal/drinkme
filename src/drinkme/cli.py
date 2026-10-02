"""drinkme — serve · pack · bench · publish · bootstrap.

serve      THE verb. OpenAI-compatible endpoint over a compressed model
           (streaming, chat template), weights resident-compressed. Packs the
           model first if it is not in the cache, and with no --model it detects
           the machine, scans what's already packed, and offers the largest
           comfortable fit — confirm, cancel, or type another model — so the
           quickstart is one word:
               drinkme serve
pack       force the one-time compress step on its own (serve does it for you).
           The pack is self-contained: it carries the checkpoint's tokenizer,
           config and unpacked tensors, and loads with no checkpoint and no
           network.
check      eligibility verdict for any HF repo, from kilobytes (no weight
           download) — the same checks pack/serve run, ahead of time.
verify     verify a pack in place — every file against its recorded
           sha256, the tensor manifest against the live meta.json — and say
           so, without loading a model. Every pack records its hashes at pack
           time; this re-checks them.
bench      detect hardware, pick models, measure bandwidth, run the arms over
           the serving pack format and kernels in its own forward loop (not
           through the HTTP server — endpoint timing is bench/serve_timing_ab.py's),
           show the result locally. Never needs auth.
publish    OPT-IN: write finished bench records to YOUR atproto PDS as
           wtf.petrichor.drinkme.measurement records (loopback OAuth; the
           scope is limited to that one collection wherever the PDS can
           grant one that narrow). A record without a resolved hub commit
           is refused by name; the rest of a set still goes.
bootstrap  the first-run install, on its own and out loud: probe the machine
           (no torch, no ROCm), pick the lane, install it into the project
           venv, prove it launches in a child interpreter. `serve` and the
           other weight-reading verbs do the same thing silently when torch
           is missing; --dry-run prints the plan and installs nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import exitcodes


def parse_rope_scaling(raw: str) -> dict | None:
    """`--rope-scaling` / `DRINKME_ROPE_SCALING` grammar, shared with
    serve._rope_scaling_from_env so the env fallback speaks the same syntax
    as the flag: 'off' -> None (disabled, the default behaviour);
    'yarn:<factor>[:<original>]' (factor > 0; original = the pre-extension
    window the vendor's recipe names, e.g. Qwen3's card: factor 4.0,
    original_max_position_embeddings 32768 -> 131,072) -> the dict
    engines._apply_rope_scaling mutates a config with. Without <original>
    the model's own native max_position_embeddings is used — which for a
    Qwen3-8B (40,960 = 32k context + 8k generation) is NOT the card's 32,768,
    so `yarn:4:32768` is the recipe-exact form. Raises ValueError on
    anything else, which the CLI type= wrapper below turns into an argparse
    error and the env helper turns into a stderr warning."""
    if raw == "off":
        return None
    if raw.startswith("yarn:"):
        parts = raw[len("yarn:"):].split(":")
        try:
            factor = float(parts[0]) if len(parts) in (1, 2) else 0.0
            original = int(parts[1]) if len(parts) == 2 else None
        except ValueError:
            factor, original = 0.0, None
        if factor > 0 and (original is None or original > 0):
            spec = {"rope_type": "yarn", "factor": factor}
            if original is not None:
                spec["original_max_position_embeddings"] = original
            return spec
    raise ValueError(f"{raw!r} is not 'yarn:<factor>[:<original>]' (factor > 0, "
                     "original > 0 — e.g. yarn:4 or yarn:4:32768) or 'off'")


def _rope_scaling_arg(raw: str) -> dict | None:
    try:
        return parse_rope_scaling(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _pixels_arg(raw: str) -> int:
    from .serving.vision import parse_pixels

    try:
        return parse_pixels(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{raw!r} is not a pixel count: give N or WxH (e.g. 2560x1440)") from None


def _media_path_arg(raw: str) -> str:
    """--media-path DIR, validated and resolved to a real path HERE (at
    parse time): a directory a human just typed wrong should fail loud and
    immediately, the same posture --image-max-pixels's argparse type=
    takes for a bad pixel count — not warn-and-fall-back, which is the
    posture for a bad ENVIRONMENT value (serving/vision.media_path_from_env)."""
    p = os.path.realpath(raw)
    if not os.path.isdir(p):
        raise argparse.ArgumentTypeError(f"{raw!r} is not a directory")
    return p


def resolve_model(name: str, verb: str) -> tuple[str, str | None] | None:
    """Menu name or HF repo -> (repo, pinned revision). A bare unknown name is
    refused with the menu; anything with a slash is taken as a repo id.
    `verb` is the calling command (e.g. "pack"), for the error prefix."""
    from .suggest import MODELS

    by = {x.name: x for x in MODELS} | {x.hf_repo: x for x in MODELS}
    m = by.get(name)
    if m:
        return m.hf_repo, m.revision
    if "/" in name:
        return name, None
    print(f"drinkme {verb}: unknown menu model {name!r}; menu models: "
          f"{', '.join(x.name for x in MODELS)} (or pass a full HF repo id)",
          file=sys.stderr)
    return None


def pin_revision(repo: str, rev: str | None) -> tuple[str, str | None]:
    """An off-menu hub repo arrives with no revision: pin it to the commit
    its default branch is on this machine (checkpoint.resolve_commit), so
    the pack is keyed `<repo>@<sha>` and /v1/models reports that sha. A
    menu model keeps its own pin; a local directory has no commit; when the
    commit cannot be told (offline, nothing cached) the pack is keyed
    `@main`, and one line says so."""
    if rev is not None or os.path.isdir(repo):
        return repo, rev
    from .serving.checkpoint import resolve_commit

    sha = resolve_commit(repo)
    if sha is None:
        print(f"[drinkme] {repo}: could not resolve its default branch to a commit "
              "(offline with nothing cached?) — keying the pack by 'main'",
              file=sys.stderr)
        return repo, None
    print(f"[drinkme] {repo}: main is commit {sha[:12]} here; the pack is keyed by it",
          file=sys.stderr)
    return repo, sha


def _verify_upstream(pdir: str) -> int:
    """`drinkme verify --upstream`, after the file-hash check passed: one
    line per file and the verdict. 0 only when every file is MATCH; 3 for a
    MISMATCH, a NOT COVERED, or a pack with no Hub commit to check; 4 when
    the Hub cannot be asked."""
    from . import upstream

    try:
        v = upstream.verify_upstream(pdir)
    except upstream.UpstreamUnavailable as e:
        print(f"drinkme verify: upstream verification unavailable ({e}); the file-hash "
              "check above passed", file=sys.stderr)
        return exitcodes.CANT_RUN_HERE
    except ValueError as e:
        print(f"drinkme verify: no upstream verification — {e}", file=sys.stderr)
        return exitcodes.REFUSED
    for f in v.files:
        print(f"  {f.line()}")
    print(v.summary())
    print(f"receipt: {os.path.join(pdir, upstream.RECEIPT_FILE)}")
    return exitcodes.OK if v.ok else exitcodes.REFUSED


def main(argv: list[str] | None = None) -> int:
    from . import __version__

    ap = argparse.ArgumentParser(prog="drinkme", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"drinkme {__version__}")
    sub = ap.add_subparsers(dest="verb", required=True)

    b = sub.add_parser("bench", help="run the benchmark on this machine")
    b.add_argument("-o", "--out",
                   help="write the record here instead of ./measurements/"
                        "<model>_<device>_<date>[_<compression profile>][_<n>].json "
                        "(with --model only; a menu run ignores it)")
    b.add_argument("--model", help="override the menu with a menu model, by its name or its "
                   "HF repo id; menu models only (serve, pack and check take any HF repo)")
    b.add_argument("--pack-dir", default=None,
                   help="time the compressed arm off THIS pack dir (loaded through the "
                        "serve loader, after serve's own torch-free front door: format, "
                        "then identity) instead of re-packing in memory; needs --model "
                        "and the torch runtime")
    b.add_argument("--detect-only", action="store_true",
                   help="hardware identity + menu only; no torch, no probe")
    b.add_argument("--dry-run", action="store_true",
                   help="print the plan (arms, predicted bytes, which record "
                        "shape) for every point bench would run; no torch, "
                        "no probe, no write")
    b.add_argument("--allow-unknown-device", action="store_true",
                   help="write a record even when the device name is only "
                        "the pci-id fallback (e.g. 'amdgpu 0x7480') — "
                        "normally refused (see docs/bench.md)")
    b.add_argument("--runtime", choices=["torch", "mlx"], default=None,
                   help="which arms run: torch (arms.py) or mlx (arms_mlx.py: mlx-lm "
                        "bf16 stock, MLX engine twin decoded once at load, fused compressed). "
                        "Default: mlx on Darwin/arm64, torch elsewhere")
    b.add_argument("--no-gemma", action="store_true",
                   help="skip the Gemma license check (a HEAD request to "
                        "huggingface.co with your HF token, if one is set); gemma "
                        "is simply left off the menu")
    b.add_argument("--stock-loader", choices=["stream", "from_pretrained"], default="stream",
                   help="how the stock (uncompressed bf16) arm loads, torch runtime "
                        "only. stream (default): the module tree is built empty and "
                        "each tensor is moved from its safetensors shard onto the "
                        "device one at a time — host transient = one tensor, so the "
                        "fit check charges weights + the largest tensor. "
                        "from_pretrained: transformers' own loader, whose transient "
                        "on unified memory is ~2x the model whether or not it loads "
                        "straight to the device (measured) — the fit check charges "
                        "2x there. The twin and compressed arms always stream "
                        "(resident + one tensor each). See docs/bench.md")
    b.add_argument("--no-spec", action="store_true",
                   help="skip the speculation pass: decode timed with the checkpoint's "
                        "MTP head proposing, as `drinkme serve --spec mtp` runs it, on the "
                        "stock and compressed arms of a model with a head (torch runtime); "
                        "the record says raw.spec.skipped = --no-spec")

    p = sub.add_parser("pack", help="compress a model to a serialized pack (CPU-only)")
    p.add_argument("--model", help="menu model name or HF repo")
    p.add_argument("-o", "--out", help="pack directory (default: ~/.cache/drinkme/packs/...)")
    p.add_argument("--replace", action="store_true",
                   help="re-pack over an existing pack directory: the new pack is "
                        "staged whole and swapped in only when complete; the old "
                        "one is kept until then (without this an existing pack "
                        "is refused)")
    # the two compression profiles; one or neither (refused below, on one line, when both)
    p.add_argument("--sip", action="store_true",
                   help="the default compression profile (tiers 3,8): faster; every eligible "
                        "bf16 tensor is encoded at it unless --gulp is given. "
                        "A tensor the codec would expand is stored raw. See "
                        "docs/pack-format.md")
    p.add_argument("--gulp", action="store_true",
                   help="the smaller compression profile (tiers 2,2,4,8): ~4%% fewer "
                        "bytes than sip, slower on bandwidth-bound devices; the compression "
                        "profile for a card where sip's bytes do not fit; not with --sip. "
                        "Without -o the pack lands beside sip's at "
                        "<org>--<model>-gulp@<rev>")

    sl = sub.add_parser("verify",
                        help="verify a pack in place (every file against its "
                             "recorded sha256, the tensor manifest against meta.json) "
                             "and say so; every pack records its hashes at pack time, "
                             "this re-checks them without loading a model")
    sl.add_argument("--model", help="menu model name or HF repo (resolves the cached pack)")
    sl.add_argument("--pack-dir", help="check this pack directory instead of the cache")
    sl.add_argument("--upstream", action="store_true",
                    help="also rebuild every source weight file from the pack and compare "
                         "its sha256 with the one the Hugging Face Hub publishes for it at "
                         "the pack's commit, and write the receipt "
                         "(docs/pack-format.md#upstream-verification)")

    sv = sub.add_parser("serve", help="serve a compressed model, OpenAI-compatible")
    sv.add_argument("--model",
                    help="menu model name or HF repo (default: detect this "
                         "machine and serve the largest comfortable compressed fit)")
    from .serve import DEFAULT_PORT

    sv.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"listen port (default: {DEFAULT_PORT})")
    sv.add_argument("--host", default="127.0.0.1", help="listen host (default: 127.0.0.1)")
    sv.add_argument("--stock", action="store_true",
                    help="serve UNCOMPRESSED bf16 — the A/B control arm for timing by "
                         "bench/serve_timing_ab.py")
    sv.add_argument("--pack-dir", help="use this pack instead of the cache")
    sv.add_argument("--runtime", choices=["torch", "mlx"], default=None,
                    help="which engine serves: torch (CUDA/ROCm/CPU) or mlx (Apple "
                         "silicon). Default: mlx on Darwin/arm64, torch elsewhere; "
                         "DRINKME_RUNTIME is the env spelling (docs/metal.md)")
    sv.add_argument("--no-auto-pack", action="store_true",
                    help="fail instead of packing when the cache misses")
    sv.add_argument("--no-hub-pack", action="store_true",
                    help="on a cache miss, download the BF16 checkpoint and pack it "
                         "here, without looking for a published pack (or set "
                         "DRINKME_NO_HUB_PACK=1; docs/serve.md#published-packs)")
    sv.add_argument("--ctx", type=int, default=None,
                    help="context window to allocate (StaticCache bound; default caps "
                         "at 8192) — memory cost is LINEAR in ctx (e.g. Qwen3-8B at "
                         "its native 40k would preallocate ~6GB)")
    sv.add_argument("--rope-scaling", type=_rope_scaling_arg, default=None,
                    metavar="SPEC",
                    help="opt-in YaRN context extension: 'yarn:<factor>[:<original>]' "
                         "serves factor x original — the vendor recipe's "
                         "pre-extension window (Qwen3's card: yarn:4:32768 -> "
                         "131072); without <original> the model's own "
                         "max_position_embeddings is used; 'off' disables (default; "
                         "or set DRINKME_ROPE_SCALING — this flag wins if "
                         "both are set, same precedence as --ctx/DRINKME_CTX). "
                         "Off by default: nothing else implies this, --ctx alone "
                         "never turns it on. Per the Qwen3 model card: static "
                         "YaRN at a fixed factor can hurt quality on short "
                         "inputs — enable only when you need the longer window")
    sv.add_argument("--auth", default=None, metavar="TOKEN",
                    help="require 'Authorization: Bearer TOKEN' on /v1/* "
                         "(or set DRINKME_AUTH_TOKEN); /health stays open")
    sv.add_argument("--prefix-slots", type=int, default=None, metavar="N",
                    help="how many whole prefix-cache states to keep warm "
                         "(default 1; 0 turns the prefix cache off; or set "
                         "DRINKME_PREFIX_SLOTS — this flag "
                         "wins if both are set, same precedence as --ctx over "
                         "DRINKME_CTX); an explicit N that will not fit is "
                         "refused at load with the arithmetic printed")
    sv.add_argument("--ctx-checkpoints", type=int, default=None, metavar="N",
                    help="context checkpoints kept per prefix slot, llama.cpp's "
                         "--ctx-checkpoints (default 32; or set "
                         "DRINKME_CTX_CHECKPOINTS — this flag wins if both are "
                         "set): a slot is reused up to the prefix a prompt shares "
                         "with it, not only when the prompt extends it; 0 keeps "
                         "the extends-only rule")
    sv.add_argument("--spec", default=None, metavar="MODE",
                    choices=["off", "mtp", "ngram", "ngram+mtp", "auto"],
                    help="which speculative proposer to serve with (or set "
                         "DRINKME_SPEC; this flag wins if both are set): "
                         "off | mtp (the checkpoint's MTP head) | ngram "
                         "(prompt lookup over the context — no draft model, "
                         "no residency) | ngram+mtp (chained) | auto "
                         "(default: mtp if the checkpoint carries a head, "
                         "else ngram)")
    sv.add_argument("--advertised-ctx", type=int, default=None, metavar="N",
                    help="report a SMALLER context window than --ctx on "
                         "/v1/messages (or set DRINKME_ADVERTISED_CTX; this "
                         "flag wins if both are set): usage.input_tokens and "
                         "count_tokens are scaled up by real/advertised, so "
                         "Claude Code's auto-compact fires before the real "
                         "window is exhausted; default is unset = real ctx, no "
                         "scaling. Only that reported number "
                         "is fictional — /metrics and the engine stay real")
    sv.add_argument("--served-model-name", nargs="+", default=None, metavar="NAME",
                    help="extra id(s) this server also answers to, beside the "
                         "resolved model id and, when --model is a menu name, "
                         "that name (or set DRINKME_SERVED_MODEL_NAMES, "
                         "space/comma-separated — this flag wins if both are "
                         "set); listed on /v1/models")
    sv.add_argument("--profile", action="append", default=None,
                    metavar="NAME=KEY:VAL,KEY:VAL",
                    help="a named sampling/chat-template overlay, exposed as "
                         "'<id>:NAME' on /v1/models (repeatable); e.g. "
                         "--profile fast=enable_thinking:false,temperature:0.7 "
                         "— or a profiles.json beside the pack, which this "
                         "flag layers on top of by name")
    sv.add_argument("--sleep-on-idle", type=float, default=None, metavar="SEC",
                    help="park the model in host RAM after SEC seconds with no "
                         "generation (or set DRINKME_SLEEP_ON_IDLE_S — "
                         "this flag wins if both are set). Default 0 = never. "
                         "While asleep /health says so and generate routes "
                         "answer 503; POST /wake_up brings it back")
    sv.add_argument("--image-max-pixels", type=_pixels_arg, default=None, metavar="N|WxH",
                    help="the largest image the model reads: a bigger one is "
                         "resized down to this many pixels (default 2560x1440; "
                         "or set DRINKME_IMAGE_MAX_PIXELS — this flag wins if "
                         "both are set). A request's detail 'low' asks for "
                         "512x512; nothing a request sends can raise the cap")
    sv.add_argument("--no-image-urls", action="store_true",
                    help="do not fetch http(s) image URLs (or set "
                         "DRINKME_IMAGE_URLS=0); default: fetch them (10s "
                         "timeout, 20 MiB cap, no address filtering — turn this "
                         "off if the server is reachable from other machines)")
    sv.add_argument("--media-path", type=_media_path_arg, default=None, metavar="DIR",
                    help="serve file:// image paths relative to this existing "
                         "directory, llama.cpp semantics (or set "
                         "DRINKME_MEDIA_PATH); default: unset, file:// is refused")
    sv.add_argument("--yes", action="store_true",
                    help="skip the confirm prompt; serve the pick (default "
                         "behaviour anyway when stdin is not a TTY)")
    pf = sub.add_parser("check",
                        help="eligibility verdict for any HF repo, from kilobytes "
                             "(no weight download)")
    pf.add_argument("--model", required=True, help="menu model name or HF repo")
    pf.add_argument("--json", action="store_true", help="machine-readable verdict")

    bs = sub.add_parser("bootstrap",
                        help="probe the machine, pick the accelerator lane, install it "
                             "into the project venv and self-test it (docs/hardware.md)")
    bs.add_argument("--dry-run", action="store_true",
                    help="print the probe, the lane and the exact install and self-test "
                         "that would run; install nothing, run nothing")
    bs.add_argument("--detect-only", action="store_true",
                    help="the probe and the lane only (no torch, no ROCm needed); "
                         "install nothing")
    bs.add_argument("--gfx", default=None, metavar="TARGET",
                    help="pretend the machine is this AMD gfx target (e.g. gfx1201) — "
                         "with --dry-run or --detect-only only, so a pretence can "
                         "never install anything")

    pb = sub.add_parser("publish",
                        help="opt-in: put a bench record on YOUR atproto PDS as a "
                             "wtf.petrichor.drinkme.measurement record (loopback OAuth)")
    pb.add_argument("file", nargs="*", default=[], metavar="FILE",
                    help="the record(s) to publish (default: the newest file under "
                         "./measurements/); a record without a resolved hub commit in "
                         "model.revision is refused by name and the others still go")
    pb.add_argument("--handle", help="your atproto handle (remembered after the first "
                                     "successful publish)")
    pb.add_argument("--did", help="your DID, instead of a handle")
    pb.add_argument("--plc", metavar="URL",
                    help="the PLC directory to resolve did:plc through (default "
                         "https://plc.directory; remembered)")
    pb.add_argument("--logout", action="store_true",
                    help="delete the stored session (tokens + DPoP key) under "
                         "~/.config/drinkme/ and stop, unless a record is also given")
    pb.add_argument("--no-browser", action="store_true",
                    help="do not try to open a browser; only print the authorization URL")

    a = ap.parse_args(argv)

    # --dry-run and --detect-only both promise "no torch" in their own help
    # strings — checked and returned BEFORE ensure_accelerator ever runs, not
    # after. Checking them afterwards would break --detect-only's
    # promise: ensure_accelerator imports and classifies
    # torch on every `bench` call regardless of the flag, and on a machine whose
    # accelerator is deliberately not visible to torch (a masked
    # CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES, e.g. a CPU-only test run) it
    # reads that as "broken" and drives a real `uv sync` — the exact silent
    # environment mutation AGENTS.md's `--no-sync` rule exists to prevent,
    # triggered just by asking for a plan.
    if a.verb == "bench" and (a.dry_run or a.detect_only):
        import json

        from . import bench

        if a.dry_run:
            print(json.dumps(bench.dry_run(a.model, runtime=a.runtime,
                                           stock_loader=a.stock_loader), indent=2))
        else:
            out = bench.run(publishable_only_detect=True, no_gemma=a.no_gemma)
            # the pre-install probe (bootstrap.probe_host: Linux, the gfx
            # target from KFD's sysfs topology, amdgpu, /dev/kfd access) on
            # an AMD machine — the same free reads `drinkme bootstrap` gates on
            from .bootstrap import probe_host

            host = probe_host()
            if host.has_amd:
                out["amdHost"] = host.as_dict()
            print(json.dumps(out, indent=2))
        return exitcodes.OK

    if a.verb == "bootstrap":
        # its own verb, never behind ensure_accelerator: the whole point is
        # to see the decision before it is made
        from .bootstrap import run_bootstrap

        return run_bootstrap(dry_run=a.dry_run, detect_only=a.detect_only, gfx=a.gfx)

    if a.verb == "serve":
        from .serve import APPLE_SILICON_NOTICE, apple_silicon

        # serve's first line on a Mac: ahead of the bootstrap's install lines,
        # the no-model picker and the revision pin below
        if apple_silicon():
            print(APPLE_SILICON_NOTICE, flush=True)

    # One command should be enough. The verbs below all end up reading weights,
    # so before any of them runs, make sure the venv holds a torch that matches
    # this machine — installing it if the accelerator fork left the choice open.
    # No-op whenever the environment is already fine, which is the normal case.
    # `detect`/`check`/`publish` stay instant and torch-free.
    if a.verb in ("bench", "pack", "serve", "verify"):
        from .bootstrap import ensure_accelerator

        outcome = ensure_accelerator()
        if not outcome.ok:
            # The runtime was just shown not to be one weights may be read
            # through (its self-test died at a kernel launch, its sync
            # failed, its host was refused...). The diagnostic and the
            # next-lane line are already on stderr; the verdict goes there
            # too, and the verb never starts — an exit code alone can be
            # clobbered to 0 by TheRock torch's atexit (entry()'s docstring),
            # so the line is the record. This is an environment problem
            # (a broken/missing accelerator), not a usage error: exitcodes.
            # CANT_RUN_HERE.
            print(f"drinkme {a.verb}: refused — accelerator bootstrap failed: {outcome.reason}",
                  file=sys.stderr)
            return exitcodes.CANT_RUN_HERE

    if a.verb == "bench":
        from . import bench

        bench.main_json(a.out, a.model, no_gemma=a.no_gemma, runtime=a.runtime,
                        allow_unknown_device=a.allow_unknown_device,
                        stock_loader=a.stock_loader, pack_dir=a.pack_dir,
                        no_spec=a.no_spec)
        return exitcodes.OK

    if a.verb == "serve" and not a.model:
        # The no-args UX (docs/models.md): real discovery and a
        # KV-aware fit — detect the machine, scan what's already packed, charge the
        # memory serve will actually use (weights + KV at the assumed ctx/slots
        # + a known MTP head), and confirm out loud before serving anything —
        # never silently, and refused with the menu when the machine can't be read.
        from . import packs, serve

        budget_gib, source = packs.hardware_budget()
        if budget_gib is None:
            print(f"drinkme serve: {source} — pass --model explicitly", file=sys.stderr)
            return exitcodes.USAGE
        # The RUNTIME first (--runtime / DRINKME_RUNTIME / the host), so
        # the menu is filtered through the one runtime-capability predicate
        # (runtimes.refusal) before anything is ranked: a Mac is never
        # offered the 27B its MLX loader would refuse after the download.
        runtime = serve.resolve_runtime(a.runtime)
        root = packs.packs_root()
        local = packs.local_packs()
        candidates, excluded = packs.filter_for_runtime(packs.build_candidates(local), runtime)
        ctx = packs.ctx_estimate(a.ctx)
        slots = packs.slots_estimate(a.prefix_slots)
        ranked = packs.rank_fits(budget_gib, candidates, ctx, slots,
                                 packs.checkpoints_estimate(a.ctx_checkpoints),
                                 packs.graph_platform(runtime))
        for line in packs.format_summary(budget_gib, source, ranked, local, root, ctx, slots,
                                         runtime=runtime, excluded=excluded):
            print(f"drinkme serve: {line}", file=sys.stderr)

        fits = [f for f in ranked if f.fits]
        if not fits:
            from .suggest import FIT_HEADROOM

            if ranked:
                smallest = ranked[-1]
                print(f"drinkme serve: nothing fits {budget_gib:.1f} GiB with "
                      f"{FIT_HEADROOM}x headroom (smallest candidate: "
                      f"{smallest.candidate.name} at {smallest.total_gib:.2f} GiB) "
                      "— pass --model to override", file=sys.stderr)
            elif excluded:
                print(f"drinkme serve: no candidate the {runtime} runtime serves "
                      f"({len(excluded)} excluded above) — pass --model to override",
                      file=sys.stderr)
            else:
                print("drinkme serve: no candidates (menu and local packs both "
                      "empty) — pass --model to override", file=sys.stderr)
            return exitcodes.USAGE
        pick = fits[0]

        if a.yes or not sys.stdin.isatty():
            a.model = pick.candidate.name
        else:
            try:
                ans = input(f"serve {pick.candidate.name}? [Y/n, or type a "
                           "model name/HF repo] ")
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            low = ans.strip().lower()
            if low in ("", "y"):
                a.model = pick.candidate.name
            elif low in ("n", "q"):
                print("drinkme serve: cancelled", file=sys.stderr)
                return exitcodes.USAGE
            else:
                a.model = ans.strip()

    if a.verb == "pack" and not a.model:
        print("drinkme pack: --model is required", file=sys.stderr)
        return exitcodes.USAGE

    if a.verb in ("pack", "serve", "check"):
        resolved = resolve_model(a.model, a.verb)
        if resolved is None:
            return exitcodes.USAGE
        repo, rev = pin_revision(*resolved)

    if a.verb == "check":
        from dataclasses import asdict

        from .check import CheckUnavailable, check

        try:
            v = check(repo, rev)
        except CheckUnavailable as e:
            print(f"drinkme check: could not check ({e})", file=sys.stderr)
            return exitcodes.CANT_RUN_HERE  # checker failure — distinct from a refusal
        if a.json:
            import json

            print(json.dumps(asdict(v), indent=2))
        else:
            print(v.line())
            if v.encoder_problem is not None:
                # the verdict (above) is the RESULT and stays on stdout; the
                # can't-pack-it-HERE half is a separate machine fact, on
                # stderr, same as every other verb's refusal
                print(v.encoder_problem, file=sys.stderr)
        if not v.ok:
            return exitcodes.REFUSED
        if v.encoder_problem is not None:
            # eligible, but THIS machine cannot currently pack it — distinct
            # from a refusal about the repo itself (v.ok), and from
            # CheckUnavailable above (that's the checker failing to reach
            # HF; this is a diagnosed local encoder problem, carried in the
            # JSON verdict too as encoder_problem so a script can tell them apart)
            return exitcodes.CANT_RUN_HERE
        return exitcodes.OK

    if a.verb == "pack":
        from .codec.pack import UnsupportedCheckpoint, pack_model

        try:
            if a.sip and a.gulp:
                print("drinkme pack: --sip and --gulp are two compression profiles; give one or neither", file=sys.stderr)
                return exitcodes.USAGE
            # neither flag = None: resolve_compression_profile's default (and its env door)
            compression_profile = "gulp" if a.gulp else ("sip" if a.sip else None)
            pack_model(repo, rev, a.out, replace=a.replace, compression_profile=compression_profile)
        except UnsupportedCheckpoint as e:
            # an FP8 (or otherwise quantized) checkpoint: the one line, from
            # config.json and the shard headers, before torch was imported —
            # a definite verdict about THIS checkpoint, not a usage error
            print(f"drinkme pack: {e}", file=sys.stderr)
            return exitcodes.REFUSED
        except FileExistsError as e:
            print(f"drinkme pack: {e}", file=sys.stderr)
            return exitcodes.USAGE
        return exitcodes.OK

    if a.verb == "verify":
        from .codec.pack import default_pack_dir, verify_pack

        if a.pack_dir:
            pdir = a.pack_dir
        elif a.model:
            resolved = resolve_model(a.model, "verify")
            if resolved is None:
                return exitcodes.USAGE
            repo, rev = pin_revision(*resolved)
            pdir = default_pack_dir(repo, rev)
        else:
            print("drinkme verify: pass --model or --pack-dir", file=sys.stderr)
            return exitcodes.USAGE
        if not os.path.exists(os.path.join(pdir, "meta.json")):
            print(f"drinkme verify: no pack at {pdir}", file=sys.stderr)
            return exitcodes.USAGE
        try:
            verify_pack(pdir)
        except ValueError as e:
            # a definite verdict about THIS pack (a bad hash, a truncated
            # file, a format this build does not read) — the same bucket as
            # `check` REFUSED and pack's UnsupportedCheckpoint, not the
            # uncaught-exception code (1) this used to share with a bug
            print(f"drinkme verify: refused — {e}", file=sys.stderr)
            return exitcodes.REFUSED
        if a.upstream:
            return _verify_upstream(pdir)
        return exitcodes.OK

    if a.verb == "serve":
        from . import serve

        return serve.run(repo, rev, host=a.host, port=a.port, stock=a.stock,
                         pack_dir=a.pack_dir, auto_pack=not a.no_auto_pack, ctx=a.ctx,
                         auth=a.auth, prefix_slots=a.prefix_slots,
                         ctx_checkpoints=a.ctx_checkpoints,
                         advertised_ctx=a.advertised_ctx, spec=a.spec,
                         served_names=a.served_model_name, generation_profile_flags=a.profile,
                         sleep_on_idle=a.sleep_on_idle, rope_scaling=a.rope_scaling,
                         runtime=a.runtime, image_max_pixels=a.image_max_pixels,
                         no_image_urls=a.no_image_urls, media_path=a.media_path,
                         hub_pack=not a.no_hub_pack,
                         # resolve_model maps a menu name to its repo and returns a
                         # repo id or local path unchanged, so a changed name is a
                         # menu name, and the server also answers to it.
                         menu_name=a.model if a.model != repo else None)

    if a.verb == "publish":
        from . import publish

        return publish.run(a.file, handle=a.handle, did=a.did, plc=a.plc,
                           logout=a.logout, no_browser=a.no_browser)

    print(f"drinkme {a.verb}: unknown verb", file=sys.stderr)
    return exitcodes.USAGE


def entry() -> None:
    """Console entry: exit via os._exit so the verdict SURVIVES.

    TheRock ROCm torch ships an atexit handler — `(anonymous
    namespace)::earlyExit()` in libtorch_cpu.so, their fix for HIP teardown
    hangs (pytorch#160759 class) — that calls `_exit(0)` once HIP has
    initialized. libc runs it mid-`exit(N)`, so ANY exit code after
    `torch.cuda.is_available()` is clobbered to 0: an uncaught traceback
    exits "success" (measured on a Strix Halo (gfx1151, 128 GB unified): a
    pack refusal exited 0). os._exit skips atexit entirely;
    flush first, since it also skips buffered-IO teardown. Exit codes carry
    verdicts, and this keeps them immune to library shutdown hacks.
    """
    import os
    import traceback

    try:
        code = main()
    except SystemExit as e:
        # PRINT THE PAYLOAD. sys.exit("some message") carries its message in
        # e.code, and Python's DEFAULT handler writes that to stderr — but
        # this entry replaces the default handler, so catching it here
        # without printing would emit zero bytes with exit 1.
        #
        # That would silence resolve_device's refusing to serve guard, whose
        # job is to make a silent failure loud. Through this console entry
        # point — how a systemd unit starts the server — a swallowed payload
        # would make that guard itself silent.
        if e.code is not None and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
        if isinstance(e, exitcodes.DrinkmeExit):
            # one of exitcodes.py's Usage/Refused/CantRunHere — the message
            # is already printed above (same branch, same as any other
            # string-payload SystemExit); this is the one place that reads
            # off `pinned` instead of falling through to the "a message with
            # no code means 1" rule below, which is for everything else
            # (argparse's own SystemExit(2), a bare sys.exit(N), an
            # uncaught SystemExit(str) that was never classified).
            code = e.pinned
        else:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
        code = exitcodes.BUG
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code if isinstance(code, int) else 0)


if __name__ == "__main__":
    entry()
