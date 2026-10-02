"""Interleave masked and mask-free single-token attention on one loaded model.

Uses the real HTTP serving loop and serve_drive's SSE timing, with resident
prefix slots enabled. The reference restores LiveStaticLayer's inherited
is_compileable=True flag; both arms return the same live KV window and use
the very same weight tensors. Cold disk slots are disabled for isolation.
No pack is written. The server binds a private, OS-assigned localhost port.

Run from the checkout, using the existing interpreter (AGENTS.md):
    PYTHONPATH=src /path/to/drinkme/.venv/bin/python \
        bench/live_cache_attention_ab.py --pack-dir /path/to/pack \
        --out verification/live-cache.json

On experimental ROCm targets, set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
as for the server. Repeat with --spec auto for the shipped speculative path.
Timings are end-to-end, so run without competing GPU work. A transcript
change is reported separately: attention backends can round differently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import serve_drive


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt-repeats", type=int, nargs="+", default=[0, 64, 256])
    ap.add_argument("--prompt-file", type=Path, help="task text; default is serve_drive's repetition task")
    ap.add_argument("--spec", choices=["off", "auto"], default="off")
    ap.add_argument("--profile", action="store_true", help="print a decode-step profile after timing")
    args = ap.parse_args()
    if args.runs < 1 or args.tokens < 2 or any(n < 0 for n in args.prompt_repeats):
        ap.error("runs must be positive, tokens >= 2, prompt-repeats nonnegative")

    os.environ["DRINKME_SLOT_DIR"] = "off"
    os.environ["DRINKME_SPEC"] = args.spec

    import torch
    from drinkme.serve import build_engine
    from drinkme.serving.http import start_server
    from drinkme.serving.kvcache import LiveStaticLayer

    assert torch.cuda.is_available(), "REFUSING a CPU measurement"
    torch.set_num_threads(4)
    pack = Path(args.pack_dir).expanduser().resolve()
    meta = json.loads((pack / "meta.json").read_text())
    eng = build_engine(meta["hfRepo"], meta.get("revision"), str(pack),
                       stock=False, ctx=args.ctx)
    assert str(eng.device) != "cpu"
    srv = start_server(eng, "127.0.0.1", 0)
    port = srv.server_address[1]
    print(f"[cache-ab] test port {port}; device {eng.device}; spec {args.spec}", flush=True)
    original_prompt = serve_drive.PROMPT
    base_prompt = args.prompt_file.read_text() if args.prompt_file else original_prompt
    original_flag = LiveStaticLayer.is_compileable
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cpu_threads": torch.get_num_threads(),
              "model": meta["hfRepo"], "revision": meta.get("revision"),
              "pack_dir": str(pack), "ctx": args.ctx, "spec": args.spec,
              "prefix_slots": len(eng._slots), "disk_slots": False,
              "aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
              "tokens": args.tokens, "runs": args.runs, "cases": []}
    report["task_prompt"] = base_prompt
    try:
        for repeats in args.prompt_repeats:
            serve_drive.PROMPT = (
                "Background notes for this task:\n" +
                ("The river flows past the old oak tree. The boat is red and the owl is grey.\n" * repeats) +
                "\nTask:\n" + base_prompt if repeats else base_prompt)
            # Prime both attention paths and all kernel signatures outside timing.
            for flag in [True, False]:
                LiveStaticLayer.is_compileable = flag
                eng.reset_prefix_cache()
                serve_drive.request(port, 8)
            rows = {"masked": [], "unmasked": []}
            for i in range(args.runs):
                order = ["masked", "unmasked"] if i % 2 == 0 else ["unmasked", "masked"]
                for arm in order:
                    LiveStaticLayer.is_compileable = arm == "masked"
                    eng.reset_prefix_cache()
                    before = serve_drive.metrics(port)
                    result = serve_drive.request(port, args.tokens)
                    after = serve_drive.metrics(port)
                    assert result["completion_tokens"] > 1 and result["decode_tok_s"] is not None, result
                    result["text_sha256"] = hashlib.sha256(result["text"].encode()).hexdigest()
                    result["spec_metrics"] = {
                        key: after.get(key, 0) - before.get(key, 0)
                        for key in ["drinkme_spec_proposed_tokens_total", "drinkme_spec_accepted_tokens_total"]}
                    rows[arm].append(result)
                    print(f"[cache-ab] repeats={repeats} run={i} {arm}: "
                          f"{result['prompt_tokens']} prompt tokens, "
                          f"{result['decode_tok_s']:.3f} tok/s, TTFT {result['ttft_s']:.3f}s", flush=True)
            medians = {arm: statistics.median(r["decode_tok_s"] for r in runs)
                       for arm, runs in rows.items()}
            case = {"prompt_repeats": repeats, "rows": rows, "median_tok_s": medians,
                    "speedup": medians["unmasked"] / medians["masked"],
                    "all_text_identical": len({r["text_sha256"] for runs in rows.values() for r in runs}) == 1}
            report["cases"].append(case)
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[cache-ab] summary {medians}; {case['speedup']:.3f}x; "
                  f"text identical={case['all_text_identical']}", flush=True)
        if args.profile:
            LiveStaticLayer.is_compileable = False
            slot = next(s for s in eng._slots if s.cache is not None)
            token = torch.tensor([[slot.ids[-1]]], device=eng.device)
            with torch.inference_mode():
                eng.model(token, past_key_values=slot.cache, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        eng.model(token, past_key_values=slot.cache, use_cache=True, logits_to_keep=1)
                    torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30), flush=True)
    finally:
        LiveStaticLayer.is_compileable = original_flag
        serve_drive.PROMPT = original_prompt
        srv.shutdown()
        srv.server_close()
    print(f"VERDICT: PASS completed serving A/B; transcript agreement is recorded separately in {args.out}", flush=True)


if __name__ == "__main__":
    main()
