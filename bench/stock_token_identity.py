"""Do the arms produce the same greedy tokens on the bench prompt? One process,
one arm resident at a time, each loaded and routed as `drinkme bench` loads
and routes it (decode_step_profile.load_arm), under each stock GEMV:

  stock-mv             the stock arm, DRINKME_STOCK_GEMV=mv (torch.mv under
                       hipBLASLt: gfx1151's stock GEMV before "triton")
  stock-triton         the stock arm, DRINKME_STOCK_GEMV=triton (the twin's
                       kernel over the raw weight, at the twin's rows)
  stock-triton-served  the same at the compressed tensor's own launch row
                       (DRINKME_TWIN_SCHEDULE=served): the order-matched
                       stock arm, the compressed kernel with the decode
                       replaced by a load of the raw weight
  twin                 the twin arm (its own rows)
  compressed           the compressed arm off the pack

Each runs arms.greedy's loop (eager, the DynamicCache) for --n-new tokens
over bench.PROMPT and records, per generated token, the argmax, the top-1
minus top-2 logit, and a hash of the whole logits row, so a pair of
variants is compared three ways: tokens (where they first differ, and both
margins there), logits rows bitwise (the first step where they differ),
and the prompt's own length (M of the prefill, which decides whether the
compressed arm's prefill is the dense F.linear over the decoded weight
(M >= swap.CompressedLinear.GEMM_MIN_ROWS) or the multi-column kernel).

    PYTHONPATH=src python bench/stock_token_identity.py --model Qwen3-8B \\
        --pack-dir ~/.cache/drinkme/packs/Qwen--Qwen3-8B@<rev> --json out.json

Prints one VERDICT line per pair against compressed and against
stock-triton; exits through os._exit (ROCm torch can exit 0 after a
failure: read the printed verdicts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from drinkme import arms  # noqa: E402
from drinkme.bench import PROMPT  # noqa: E402
from drinkme.codec.swap import CompressedLinear  # noqa: E402
from drinkme.serving.checkpoint import resolve_source, tokenizer as load_tokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_step_profile import load_arm, menu_model  # noqa: E402

VARIANTS = {
    "stock-mv": ("stock", {"DRINKME_STOCK_GEMV": "mv"}),
    "stock-triton": ("stock", {"DRINKME_STOCK_GEMV": "triton"}),
    "stock-triton-served": ("stock", {"DRINKME_STOCK_GEMV": "triton", "DRINKME_TWIN_SCHEDULE": "served"}),
    "twin": ("twin", {}),
    "compressed": ("compressed", {}),
}
ENV_KEYS = ("DRINKME_STOCK_GEMV", "DRINKME_TWIN_SCHEDULE")


@torch.inference_mode()
def greedy_trace(model, ids, n_new: int) -> dict:
    """arms.greedy's loop, recording each step's pick, margin and logits hash."""
    out, past = ids, None
    toks, margins, hashes = [], [], []
    for _ in range(n_new):
        res = model(out if past is None else out[:, -1:], past_key_values=past, use_cache=True)
        past = res.past_key_values
        row = res.logits[:, -1, :]
        nxt = row.argmax(-1, keepdim=True)
        top = row.float().topk(2, dim=-1).values[0]
        toks.append(int(nxt))
        margins.append(float(top[0] - top[1]))
        hashes.append(hashlib.sha256(row.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16])
        out = torch.cat([out, nxt], -1)
    return dict(tokens=toks, margins=margins, logits_sha=hashes)


def compare(a: dict, b: dict) -> dict:
    n = min(len(a["tokens"]), len(b["tokens"]))
    tok = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
    lg = next((i for i in range(n) if a["logits_sha"][i] != b["logits_sha"][i]), None)
    rec = dict(tokens_identical=tok is None, first_token_diff=tok, logits_identical=lg is None, first_logits_diff=lg)
    if tok is not None:
        rec.update(token_a=a["tokens"][tok], token_b=b["tokens"][tok],
                   margin_a=a["margins"][tok], margin_b=b["margins"][tok])
    return rec


def verdict(name_a: str, name_b: str, c: dict) -> str:
    logits = ("logits bitwise identical at every step" if c["logits_identical"]
              else f"logits first differ at generated token {c['first_logits_diff']}")
    if c["tokens_identical"]:
        return f"VERDICT {name_a} == {name_b}: tokens IDENTICAL; {logits}"
    return (f"VERDICT {name_a} vs {name_b}: tokens DIFFER at generated token {c['first_token_diff']} "
            f"({c['token_a']} vs {c['token_b']}); top-2 logit margin there: {name_a} {c['margin_a']}, "
            f"{name_b} {c['margin_b']}; {logits}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--n-new", type=int, default=arms.N_NEW)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    m = menu_model(a.model)
    pack_dir = os.path.expanduser(a.pack_dir)
    report = dict(instrument="stock_token_identity", model=m.hf_repo, prompt=PROMPT, n_new=a.n_new,
                  started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), runs={}, pairs={}, verdict="INCOMPLETE")
    status = 1
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")
        snap, _ = resolve_source(m.hf_repo, m.revision, pack_dir)
        tok = load_tokenizer(snap, None)
        ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
        report["prompt_tokens"] = int(ids.shape[1])
        report["compressed_prefill_route"] = ("dense" if ids.shape[1] >= CompressedLinear.GEMM_MIN_ROWS else "mc")
        print(f"prompt: {ids.shape[1]} tokens (the compressed arm's prefill: "
              f"{report['compressed_prefill_route']})", flush=True)
        for name in a.variants:
            arm, env = VARIANTS[name]
            saved = {k: os.environ.get(k) for k in ENV_KEYS}
            for k in ENV_KEYS:
                os.environ.pop(k, None)
            os.environ.update(env)
            try:
                model, route = load_arm(arm, m.hf_repo, m.revision, pack_dir if arm != "stock" else None, snap)
                report["runs"][name] = dict(arm=arm, env=env, route=route, **greedy_trace(model, ids, a.n_new))
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                model = None
                arms.free_all()
            r = report["runs"][name]
            print(f"  {name:20s} {len(r['tokens'])} tokens, min margin {min(r['margins']):.4f} "
                  f"at token {r['margins'].index(min(r['margins']))}", flush=True)
        for ref in ("compressed", "stock-triton"):
            if ref not in report["runs"]:
                continue
            for name in report["runs"]:
                if name == ref or f"{name}|{ref}" in report["pairs"]:
                    continue
                c = compare(report["runs"][name], report["runs"][ref])
                report["pairs"][f"{ref}|{name}"] = c
                print(verdict(name, ref, c), flush=True)
        report["verdict"] = "DONE"
        print("STOCK_TOKEN_IDENTITY DONE", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        report["verdict"] = "FAIL"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"STOCK_TOKEN_IDENTITY FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {a.json}", flush=True)
    return status


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
