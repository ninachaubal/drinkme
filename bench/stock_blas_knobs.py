"""The stock arm's M=1 decode Linears under every BLAS knob PyTorch gives
without a kernel of ours: which library F.linear dispatches to
(torch.backends.cuda.preferred_blas_library: "cublas" is rocBLAS on a ROCm
build, "cublaslt" hipBLASLt), torch.mv in place of F.linear, and PyTorch
TunableOp (torch.cuda.tunable: tunes each GEMM signature over the rocBLAS
and hipBLASLt solutions on first use and caches the pick in a CSV); and
beside them the one kernel of ours the stock GEMV may be, "triton": the
twin arm's Triton GEMV over the raw weight (swap.stock_linear with
raw_gemv_dict's row). This is the instrument behind the bench's stock-arm
GEMV choice per box (codec/swap.py's STOCK_GEMV, applied by
serving/kernel_route.route_kernels).

    PYTHONPATH=src flock /tmp/drinkme-gpu.lock .venv/bin/python bench/stock_blas_knobs.py \
        --models Qwen/Qwen3-0.6B Qwen/Qwen3-8B --json out.json [--tunableop-dir DIR]

For each model: every decode Linear of the checkpoint's config (every
layer, and lm_head), as random bf16 weights on the device (a GEMM's time
does not depend on the values), x one bf16 row. Each knob is timed two
ways over the whole set:
  device  the rotation protocol (bench/parity_common.rotation, with the
          primer): back-to-back passes, an event pair around each call,
          the per-call medians summed — the GPU time of one token's
          Linears
  wall    back-to-back passes with no events and one synchronize per
          pass, wall-clock per pass — the same calls with their host
          cost, which is what an eager decode pays when the host is
          slower than the kernels
The kernel each knob launches per shape is recorded (torch.profiler),
and every knob's output is checked against a float64 reference.
TunableOp's first-use tuning is timed per shape, and its results file is
kept (--tunableop-dir; a fresh one per run unless it already has rows).

--shapes-from-pack reads a pack's tensor shapes instead of the config
(the hybrid 27B's DeltaNet projections), over --layers of it, and scales
the per-token sums by the model's layer count over the layers taken.

Prints STOCK_BLAS_KNOBS PASS/FAIL; exits through os._exit (ROCm torch can
exit 0 after a failure: read the verdict line).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "bench")
import parity_common as pc  # noqa: E402

# knob -> (preferred BLAS library, op, TunableOp)
KNOBS = {
    "rocblas": ("cublas", "linear", False),
    "hipblaslt": ("cublaslt", "linear", False),
    "rocblas-mv": ("cublas", "mv", False),
    "hipblaslt-mv": ("cublaslt", "mv", False),
    "tunableop": (None, "linear", True),
    "tunableop-mv": ("cublaslt", "mv", True),
    # the twin arm's kernel over the raw weight: stock GEMV "triton"
    "triton": (None, "triton", False),
    # what the stock arm runs: swap.stock_linear's one-row call at this
    # box's stock GEMV (swap.stock_gemv_mode), the default library around it
    "route": (None, "route", False),
}
PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def config_shapes(model_id: str) -> tuple[list[tuple[str, int, int]], int, int]:
    """[(name, R, C)] for every decode Linear of a Qwen3-shaped checkpoint's
    config (the HF cache's config.json), lm_head last; (list, layers
    taken, layers in the model)."""
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(model_id, "config.json", local_files_only=True)).read_text())
    cfg = cfg.get("text_config", cfg)
    H, I, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    hd = cfg.get("head_dim") or H // cfg["num_attention_heads"]
    q, kv = cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd
    dims = dict(q_proj=(q, H), k_proj=(kv, H), v_proj=(kv, H), o_proj=(H, q),
                gate_proj=(I, H), up_proj=(I, H), down_proj=(H, I))
    out = [(f"model.layers.{i}.{p}", *dims[p]) for i in range(L) for p in PROJS]
    out.append(("lm_head", cfg["vocab_size"], H))
    return out, L, L


def pack_shapes(pack_dir: str, layers: list[int], n_layers: int) -> tuple[list, int, int]:
    """[(name, R, C)] off a pack's npz headers for `layers` and lm_head."""
    meta = json.loads((Path(pack_dir) / "meta.json").read_text())
    want = [n for n in meta["tensors"] if n == "lm_head"
            or any(n.startswith(f"model.layers.{i}.") for i in layers)]
    out = []
    for n in want:
        with np.load(Path(pack_dir) / meta["tensors"][n]) as z:
            sc = json.loads(str(z["scalars"][0])) if "scalars" in z.files else {k: z[k] for k in ("R", "C")}
            out.append((n, int(sc["R"]), int(sc["C"])))
    out.sort(key=lambda t: t[0] == "lm_head")
    return out, len(layers), n_layers


_TWIN_DICTS: dict = {}  # (R, C) -> swap.raw_gemv_dict, built once per shape as install_stock_gemv does
_ROUTE: list = []  # [this box's stock GEMV mode], read once


def _twin_dict(W: torch.Tensor) -> dict:
    from drinkme.codec.swap import raw_gemv_dict

    key = tuple(W.shape)
    if key not in _TWIN_DICTS:
        _TWIN_DICTS[key] = raw_gemv_dict(*key)
    return _TWIN_DICTS[key]


def _call(op: str, W: torch.Tensor, x: torch.Tensor):
    if op in ("route", "triton"):
        from drinkme.codec.swap import stock_gemv_mode, stock_linear

        if not _ROUTE:
            _ROUTE.append(stock_gemv_mode("cuda"))
        mode = _ROUTE[0] if op == "route" else "triton"
        return stock_linear(x, W, None, mode, _twin_dict(W) if mode == "triton" else None)
    return F.linear(x, W) if op == "linear" else torch.mv(W, x.reshape(-1))


def _set_blas(lib):
    if lib is not None:
        torch.backends.cuda.preferred_blas_library(lib)


def kernel_names(fn) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})


def wall_passes(fns, passes: int) -> list[float]:
    """Back-to-back passes, no events, one synchronize per pass: ms per pass."""
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for fn in fns:
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def write_tunableop_file(path: str) -> None:
    """TunableOp's results CSV (its own format: Validator rows, then
    op,params,solution,ms). TunableOp writes it at interpreter exit, which
    os._exit skips, and this build has no torch.cuda.tunable.write_file."""
    import torch.cuda.tunable as tn

    lines = [",".join(["Validator", *map(str, v)]) for v in tn.get_validators()]
    lines += [",".join(map(str, r)) for r in tn.get_results()]
    Path(path).write_text("\n".join(lines) + "\n")


def gpu_busy() -> int | None:
    p = Path("/sys/class/drm/card1/device/gpu_busy_percent")
    try:
        return int(p.read_text())
    except (OSError, ValueError):
        return None


@torch.inference_mode()
def run_model(label: str, shapes, taken: int, n_layers: int, a, wall_bytes_s: float, report: dict):
    g = torch.Generator(device="cuda").manual_seed(2026)
    weights, xs = [], {}
    for name, R, C in shapes:
        weights.append(torch.randn((R, C), device="cuda", dtype=torch.bfloat16, generator=g))
        if C not in xs:
            xs[C] = torch.randn((1, 1, C), device="cuda", dtype=torch.bfloat16, generator=g)
    byts = sum(R * C * 2 for _, R, C in shapes)
    scale = n_layers / taken
    lm = [i for i, (n, _, _) in enumerate(shapes) if n == "lm_head"]
    uniq = {}
    for i, (n, R, C) in enumerate(shapes):
        uniq.setdefault((R, C), i)
    # the float64 reference, one tensor per distinct shape
    ref = {}
    for (R, C), i in uniq.items():
        ref[(R, C)] = (weights[i].double() @ xs[C].reshape(-1).double(),
                       2e-3 * (weights[i].double().abs() @ xs[C].reshape(-1).double().abs()) + 1e-6)
    rec = dict(model=label, tensors=len(shapes), layers_taken=taken, layers=n_layers, bytes=byts,
               per_token_scale=scale, shapes=sorted({(R, C) for _, R, C in shapes}), knobs={})
    for knob in a.knobs:
        lib, op, tunable = KNOBS[knob]
        cell = dict(library=lib, op=op, tunableop=tunable)
        _set_blas(lib if lib is not None else a.tunableop_blas)
        if tunable:
            import torch.cuda.tunable as tn

            d = Path(a.tunableop_dir or os.path.expanduser("~/.cache/drinkme/tunableop"))
            d.mkdir(parents=True, exist_ok=True)
            fname = str(d / f"tunableop_results{torch.cuda.current_device()}.csv")
            tn.set_filename(fname, insert_device_ordinal=False)
            tn.enable(True)
            tn.tuning_enable(True)
            tn.set_max_tuning_duration(a.tunableop_ms)
            tn.set_max_tuning_iterations(a.tunableop_iters)
            rows_before = len(tn.get_results())
            tune_s = {}
            for (R, C), i in uniq.items():
                t0 = time.perf_counter()
                _call(op, weights[i], xs[C])
                torch.cuda.synchronize()
                tune_s[f"{R}x{C}"] = time.perf_counter() - t0
            write_tunableop_file(fname)
            tn.tuning_enable(False)
            results = tn.get_results()
            cell.update(tuning_s=tune_s, tuning_total_s=sum(tune_s.values()), results_file=fname,
                        results_rows_before=rows_before, results=[list(r) for r in results],
                        validators=[list(v) for v in tn.get_validators()])
        try:
            errs = []
            for (R, C), i in uniq.items():
                y, bound = ref[(R, C)]
                v = _call(op, weights[i], xs[C]).reshape(-1).double()
                errs.append(float(((v - y).abs() / bound).max()))
            cell["err_over_bound"] = max(errs)
            if not cell["err_over_bound"] <= 1:
                raise AssertionError(f"{label} {knob}: output exceeds the bound ({cell['err_over_bound']:.3g})")
            cell["kernels"] = {f"{R}x{C}": kernel_names(lambda i=i, C=C: _call(op, weights[i], xs[C]))
                               for (R, C), i in uniq.items()}
            fns = [(lambda W=W, x=xs[W.shape[1]]: _call(op, W, x)) for W in weights]
            busy0 = gpu_busy()
            dev, wall = [], []
            for r in range(a.repeats):
                per, totals = pc.rotation(fns, a.passes, prime=True)
                dev.append(sum(float(np.median(s)) for s in per))
                wall.append(float(np.median(wall_passes(fns, a.passes))))
                if r == 0:
                    cell["per_shape_ms"] = {}
                    for (R, C) in uniq:
                        idx = [k for k, (_, r_, c_) in enumerate(shapes) if (r_, c_) == (R, C)]
                        cell["per_shape_ms"][f"{R}x{C}"] = float(np.median([np.median(per[k]) for k in idx]))
                    if lm:
                        cell["lm_head_ms"] = float(np.median(per[lm[0]]))
            busy1 = gpu_busy()
            dev_ms, wall_ms = float(np.median(dev)), float(np.median(wall))
            head = cell.get("lm_head_ms", 0.0)
            cell.update(device_ms_per_pass=dev_ms, device_ms_per_repeat=dev, wall_ms_per_pass=wall_ms,
                        wall_ms_per_repeat=wall, gpu_busy_before=busy0, gpu_busy_after=busy1,
                        device_efficiency=byts / wall_bytes_s * 1e3 / dev_ms,
                        # one token's Linears: the taken layers scaled to the model, lm_head once
                        device_ms_per_token=(dev_ms - head) * scale + head,
                        wall_ms_per_token=(wall_ms - head) * scale + head)
            print(f"  {label:18s} {knob:13s} device {dev_ms:8.3f} ms/pass (eff {cell['device_efficiency']:.3f})"
                  f"  wall {wall_ms:8.3f} ms/pass  per token dev {cell['device_ms_per_token']:.2f} / wall "
                  f"{cell['wall_ms_per_token']:.2f} ms" + (f"  tuning {cell['tuning_total_s']:.1f}s" if tunable else ""),
                  flush=True)
        finally:
            if tunable:
                import torch.cuda.tunable as tn

                tn.enable(False)
        rec["knobs"][knob] = cell
    _set_blas(a.restore_blas)
    best = min(rec["knobs"], key=lambda k: rec["knobs"][k]["wall_ms_per_token"])
    rec["best_wall"] = best
    rec["best_device"] = min(rec["knobs"], key=lambda k: rec["knobs"][k]["device_ms_per_token"])
    report["models"].append(rec)
    del weights, xs, ref
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"])
    ap.add_argument("--shapes-from-pack", nargs="*", default=[],
                    help="LABEL=PACK_DIR:LAYERS:N_LAYERS, e.g. 27B=~/.cache/drinkme/packs/X:0,1,2,3:64")
    ap.add_argument("--knobs", nargs="*", default=list(KNOBS), choices=list(KNOBS))
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tunableop-dir", default=None, help="TunableOp results directory (default ~/.cache/drinkme/tunableop)")
    ap.add_argument("--tunableop-ms", type=int, default=30, help="TunableOp max tuning duration per solution set, ms")
    ap.add_argument("--tunableop-iters", type=int, default=100)
    ap.add_argument("--tunableop-blas", default=None, help="the preferred library while TunableOp runs (default: unchanged)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("STOCK_BLAS_KNOBS FAIL: no GPU")
        return 2
    from drinkme.probe import measure_bandwidth

    a.restore_blas = {"Cublas": "cublas", "Cublaslt": "cublaslt"}.get(
        str(torch.backends.cuda.preferred_blas_library()).split(".")[-1], None)
    facts = pc.machine_facts()
    from drinkme.codec.swap import stock_gemv_mode

    report = dict(instrument="stock_blas_knobs", verdict="INCOMPLETE", machine=facts,
                  default_blas=str(torch.backends.cuda.preferred_blas_library()),
                  route=stock_gemv_mode("cuda"),
                  args={k: v for k, v in vars(a).items()}, models=[],
                  started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    status = 1
    try:
        bw = measure_bandwidth()
        report["bandwidth"] = bw
        wall = float(bw["read_bytes_s"])
        print(f"device {facts['device']} {facts.get('arch')} torch {facts['torch']} default BLAS "
              f"{report['default_blas']} read wall {wall / 1e9:.1f} GB/s", flush=True)
        for model in a.models:
            shapes, taken, n = config_shapes(model)
            run_model(model.split("/")[-1], shapes, taken, n, a, wall, report)
        for spec in a.shapes_from_pack:
            label, _, rest = spec.partition("=")
            pack, layers, n = rest.rsplit(":", 2)
            shapes, taken, n = pack_shapes(os.path.expanduser(pack), [int(x) for x in layers.split(",")], int(n))
            run_model(label, shapes, taken, n, a, wall, report)
        report["verdict"] = "PASS"
        print("PER MODEL (ms per token's Linears; best by wall):", flush=True)
        for m in report["models"]:
            print(f"  {m['model']:18s} " + "  ".join(
                f"{k} {c['wall_ms_per_token']:.2f}/{c['device_ms_per_token']:.2f}" for k, c in m["knobs"].items())
                + f"  -> {m['best_wall']}", flush=True)
        print("STOCK_BLAS_KNOBS PASS", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        report["verdict"] = "FAIL"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"STOCK_BLAS_KNOBS FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    if a.json:
        pc.write_json(a.json, report)
        print(f"wrote {a.json}", flush=True)
    return status


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
