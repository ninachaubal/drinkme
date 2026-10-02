"""The launch-config sweep under ROTATION: for every Qwen3-8B tensor shape class — the
4096^2 q/o projections, the 1024-row k/v, the 12288-row gate/up, the
4096x12288 down_proj, the 151,936-row lm_head — a profile's M=1 GEMV
at every launch config of the grid (blocks-per-program in {1, 2, 4, whole
row} x warps in {1, 2, 4}; split-K partials + _finish when the program holds
less than a row), with the other profiles at fixed configs and the `spike`
schedule (t1/w2) as the controls in the SAME session; medians and every rep;
efficiency against the wall (verification/parity/wall.json:
drinkme.probe.measure_bandwidth x3 via bench/roofline_wall.py, run first).
This is the instrument that fills codec/radix_schedule.py's table for a box
class.

    PYTHONPATH=src .venv/bin/python bench/parity_rotation.py --layers 0 18 --lm-head \
        --json verification/parity/rotation.json

Protocol: a pass runs every tensor of
the set back to back with an event pair around each call and no explicit
flush (the set is >1 GB per arm: the other tensors evict each one, the
model's own regime); `--passes` passes per config per repeat, `--repeats`
separated repeats (every config is timed again after every other config,
so drift shows as a repeat-to-repeat delta, not as a config effect). Before
timing, every arm's decoded weights are compared bit-for-bit with the
checkpoint's safetensors bits and every config's GEMV against the float64
oracle (the schedule gate's checks, repeated in-session on the timed
buffers). --mc-Ms adds the multi-column arm (radix_ops.gemv_mc) at those M
for a small mc config grid, so the mc schedule's stay-or-move is measured too.

`twin` is an arm too (--radix-arms sip twin): the bench's order-matched
twin (swap.to_device_twin over the checkpoint's bf16 bits, the radix
kernels with RAW=True) at the widths of the first other radix arm, timed
over the same grid beside it. Its weight is checked bit for bit against
the checkpoint and its GEMV against the oracle; at every config its M=1
output is also compared bitwise with that arm's (recorded as
`bitwise_vs_<arm>`; bench/radix_twin_bitpin.py is the gate). Its picks
fill codec/radix_schedule.TWIN_TABLES.

Prints PARITY_ROTATION PASS/FAIL, exits through os._exit.

Off the Strix Halo (gfx1151, 128 GB unified) — e.g. in
bench/modal_nvidia2.py's containers — the packs come from
DRINKME_PARITY_PACK_{SIP,BALANCED,GULP}, the checkpoint from the HF cache
(parity_common.snapshot_dir), the sensors from nvidia-smi; --tensors names
any checkpoint's tensors (the 27B's DeltaNet projections), --gulp-configs
times gulp at more than the `spike` schedule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("DRINKME_PREFILL_DENSE_MIN", "9")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "bench")
from drinkme.codec import radix_ops  # noqa: E402
from drinkme.codec.radix_schedule import select, shape_class  # noqa: E402
import parity_common as pc  # noqa: E402
from parity_common import Sensors  # noqa: E402

SPIKE = "t1/w2"
TWIN = "twin"  # swap.to_device_twin's codec: the order-matched twin as an arm
# the mc grid: (tiles, warps), the `spike` schedule first. Whole-row mc programs hold
# eight [1024] fp32 accumulators — at 1-2 warps that spills to scratch
# (parity/rotation_session1.json: 56-113 ms per set-pass vs 8-12), and the
# scratch pressure disturbed the M=1 arms timed after it, so trow is not in
# the default mc grid; --mc-configs trow/w1 puts it back.
MC_DEFAULT = ("t1/w2", "t1/w1", "t2/w1", "t1/w4")


class NvSensors:
    """parity_common.Sensors for an NVIDIA container (no sysfs hwmon):
    nvidia-smi's SM clock / temperature / power draw, sampled in a thread."""

    def __init__(self, interval=1.0):
        import threading
        self.interval = interval
        self.samples = {'sclk_mhz': [], 'temp_c': [], 'power_w': []}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import subprocess
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=5).stdout.strip().split(",")
                for key, v in zip(('sclk_mhz', 'temp_c', 'power_w'), out):
                    self.samples[key].append(float(v))
            except Exception:  # noqa: BLE001 — a missed sample is not a failure
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        out = {"source": "nvidia-smi"}
        for key, values in self.samples.items():
            if values:
                out[key] = dict(min=float(min(values)), median=float(np.median(values)), max=float(max(values)), samples=len(values))
        return out


def pick_sensors():
    s = Sensors()
    return s if s.files else NvSensors()


def parse_config(label: str) -> tuple[int, int]:
    t, w = label.split("/")
    return (0 if t == "trow" else int(t[1:])), int(w[1:])


def _u16(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.int16).cpu().numpy().view(np.uint16)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="verification/parity/rotation.json")
    ap.add_argument("--wall", default="verification/parity/wall.json")
    ap.add_argument("--layers", type=int, nargs="*", default=[0])
    ap.add_argument("--lm-head", action="store_true")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tiles", type=int, nargs="+", default=[1, 2, 4, 0], help="0 = whole row")
    ap.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--mc-Ms", type=int, nargs="*", default=[])
    ap.add_argument("--mc-configs", nargs="*", default=list(MC_DEFAULT), help="mc launch configs, e.g. t1/w2 trow/w1")
    ap.add_argument("--gulp", action="store_true", help="add gulp at the `spike` schedule (t1/w2) as a control")
    # any profile can be the SWEPT arm
    # (the full tiles x warps grid) or a CONTROL at fixed configs, e.g.
    #   --radix-arms sip balanced gulp --grid-arms balanced
    #   --fixed-configs sip=trow/w1 gulp=t1/w2,trow/w1
    # keys are "<arm>:<label>"; picks are computed per class for every grid arm.
    ap.add_argument("--radix-arms", nargs="*", default=["sip"], help="profiles to load (sip balanced gulp)")
    ap.add_argument("--grid-arms", nargs="*", default=None, help="which of --radix-arms get the full grid (default: all)")
    ap.add_argument("--fixed-configs", nargs="*", default=[],
                    help="arm=cfg[,cfg...] for a control arm not in --grid-arms, e.g. sip=trow/w1 gulp=t1/w2")
    # the nvidia2 harness's spelling of the same (bench/modal_nvidia2.py): with
    # --gulp, the configs gulp is timed at — `grid` makes it a grid
    # arm, a list makes it a control at those configs (--fixed-configs gulp=...)
    ap.add_argument("--gulp-configs", nargs="*", default=[SPIKE],
                    help="with --gulp: the launch configs gulp is timed at (e.g. t1/w2 trow/w1, or `grid` = "
                         "the sip grid), the `spike` schedule alone by default")
    ap.add_argument("--tensors", nargs="*", default=[],
                    help="explicit tensor names instead of --layers/--lm-head (any checkpoint the packs hold)")
    ap.add_argument("--primer", action="store_true", help="a ~1 ms GPU backlog at the head of every pass (parity_common.primer)")
    ap.add_argument("--no-sensors", action="store_true", help="do not sample sysfs during timing")
    ap.add_argument("--fill-vram", type=int, default=0,
                    help="MB of a dummy device buffer allocated FIRST and held: on this APU the first ~0.5 GB of "
                         "allocations land in the VRAM carve-out (mem_info_vram_total) and the buffers at its "
                         "eviction boundary time erratically (parity/rotation_session{1,2}: layer-0 q/k/v/o, sip "
                         "arms only, 2x pass-to-pass); with the carve-out pre-filled every timed buffer is in GTT, "
                         "where a 12 GB model's tensors live anyway")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.set_num_threads(1)
    wall = pc.wall_bytes_s(a.wall)
    names = tuple(a.tensors) if a.tensors else (
        tuple(n for layer in a.layers for n in pc.layer_names(layer)) + ((pc.LM_HEAD,) if a.lm_head else ()))
    facts = pc.machine_facts()
    print(f"device: {facts['device']} torch {facts['torch']} triton {facts['triton']} wall {wall / 1e9:.1f} GB/s "
          f"units={facts['watched_units']} gtt_used={facts['gtt_used_bytes']}", flush=True)
    report = dict(verdict="INCOMPLETE", wall_read_bytes_s=wall, machine=facts, args=vars(a),
                  kernel_sha256=hashlib.sha256(Path("src/drinkme/codec/radix_kernel_gpu.py").read_bytes()).hexdigest(),
                  ops_sha256=hashlib.sha256(Path("src/drinkme/codec/radix_ops.py").read_bytes()).hexdigest(),
                  protocol=dict(passes=a.passes, repeats=a.repeats,
                                rotation="back-to-back passes over the tensor set, event pair per call, no flush; "
                                         "repeats separated by every other config"),
                  arms=dict(radix="radix_ops.gemv_fused (the served M=1 call) at each launch config; x bf16, out bf16"),
                  tensors=[], started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    g = torch.Generator(device="cuda").manual_seed(2026)
    vram = Path("/sys/class/drm/card1/device/mem_info_vram_used")
    report["vram_used_bytes_before"] = int(vram.read_text()) if vram.exists() else None
    filler = torch.empty(a.fill_vram * 2**20, dtype=torch.uint8, device="cuda") if a.fill_vram else None
    if filler is not None:
        filler.fill_(1)
        torch.cuda.synchronize()
    report["vram_used_bytes_after_fill"] = int(vram.read_text()) if vram.exists() else None
    print(f"vram_used before fill {report['vram_used_bytes_before']} after {report['vram_used_bytes_after_fill']} (fill {a.fill_vram} MB)", flush=True)
    # ---- load + check ------------------------------------------------------------
    radix_arms = list(a.radix_arms)
    if a.gulp and "gulp" not in radix_arms:
        radix_arms.append("gulp")
    grid_arms = list(a.grid_arms) if a.grid_arms is not None else [arm for arm in radix_arms if arm != "gulp" or not a.gulp]
    fixed = {}
    for spec in a.fixed_configs:
        arm, _, cfgs = spec.partition("=")
        fixed[arm] = [c for c in cfgs.split(",") if c]
    if a.gulp and "gulp" not in grid_arms:
        if a.gulp_configs == ["grid"]:
            grid_arms.append("gulp")
        else:
            fixed.setdefault("gulp", list(a.gulp_configs))
    for arm in radix_arms:
        if arm not in grid_arms and arm not in fixed:
            raise ValueError(f"--radix-arms {arm}: neither a grid arm nor given --fixed-configs")
    report["radix_arms"] = radix_arms
    report["grid_arms"] = grid_arms
    report["fixed_configs"] = fixed
    packed = [arm for arm in radix_arms if arm != TWIN]
    if TWIN in radix_arms and not packed:
        raise ValueError("--radix-arms twin needs a radix arm beside it (the widths it runs at)")
    packs = {arm: pc.load_pack_dicts(pc.PACKS[arm], names) for arm in packed}
    items = []
    for name in names:
        t0 = time.perf_counter()
        src = pc.source_bits(name)
        R, C = src.shape
        nb = triton.cdiv(C, 1024)
        x16 = torch.randn((C,), device="cuda", dtype=torch.bfloat16, generator=g)
        src_gpu = torch.from_numpy(src.view(np.int16).copy()).cuda().view(torch.uint16)
        y, bound = pc.oracle(x16, src_gpu)
        item = dict(name=name, short=pc.short(name), shape=[R, C], nb=nb, shape_class=shape_class(R, C), weights=R * C,
                    arms={}, fns={})
        # every radix arm: the grid arms at every config, the controls at their fixed configs
        rts = {}
        for arm in sorted(radix_arms, key=lambda arm: arm == TWIN):  # the twin after its widths arm
            if arm == TWIN:
                from drinkme.codec.swap import to_device_twin

                rt = to_device_twin(torch.from_numpy(src.view(np.int16).copy()).view(torch.bfloat16),
                                    packs[packed[0]][name]["widths"], "cuda")
                rt["rx_launch"] = select(R, C, int(rt["block_size"]), rt["widths"], override="spike").as_dict()
                if not np.array_equal(_u16(rt["weight"]), src):
                    raise AssertionError(f"{name} twin weight differs from the checkpoint")
            else:
                rt = pc.to_runtime(packs[arm][name])
                for w in sorted(set(a.warps)):
                    if not np.array_equal(_u16(radix_ops.decode_bits(rt, num_warps=w)), src):
                        raise AssertionError(f"{name} {arm} dense bits differ from the checkpoint at {w} warps")
            rts[arm] = rt
            if arm in grid_arms:
                configs = pc.configs_for(nb, tuple(a.tiles), tuple(a.warps))
            else:
                configs = []
                for lab in fixed[arm]:
                    t, w = parse_config(lab)
                    eff = 0 if (t == 0 or t >= nb) else t  # a program holding the whole row IS the row program
                    lab = f"t{'row' if eff == 0 else eff}/w{w}"
                    if lab not in [x[0] for x in configs]:
                        configs.append((lab, eff, w))
            for label, tiles, warps in configs:
                rt2 = pc.with_launch(rt, tiles, warps)
                err = pc.gemv_error(radix_ops.gemv(rt2, x16), y, bound)
                if not err <= 1:
                    raise AssertionError(f"{name} {arm} {label} GEMV exceeds the bound: {err}")
                counts = pc.radix_bytes(rt2, tiles)
                key = f"{arm}:{label}"
                item["arms"][key] = dict(arm=arm, label=label, tiles=tiles, warps=warps, bytes=counts,
                                         floor_ms=counts["total"] / wall * 1e3, err_over_bound=err,
                                         launches=1 if counts["splits"] == 1 else 2, samples={})
                if arm == TWIN:
                    other = pc.with_launch(rts[packed[0]], tiles, warps)
                    item["arms"][key][f"bitwise_vs_{packed[0]}"] = bool(torch.equal(
                        radix_ops.gemv_fused(rt2, x16, None, torch.bfloat16).view(torch.int16),
                        radix_ops.gemv_fused(other, x16, None, torch.bfloat16).view(torch.int16)))
                item["fns"][key] = (lambda rt2, x: (lambda: radix_ops.gemv_fused(rt2, x, None, torch.bfloat16)))(rt2, x16)
        rt = rts[grid_arms[0]] if grid_arms else rts[radix_arms[0]]  # the mc arm's pack (the first grid arm)
        # mc arms
        if a.mc_Ms:
            item["mc"] = {}
            item["mc_fns"] = {}
            for M in a.mc_Ms:
                xm = torch.randn((M, C), device="cuda", dtype=torch.bfloat16, generator=g)
                ym = [pc.oracle(xm[i], src_gpu) for i in range(M)]
                mc_arm = grid_arms[0] if grid_arms else radix_arms[0]
                for tiles, warps in (parse_config(c) for c in a.mc_configs):
                    rt2 = pc.with_launch(rt, 1, 2, mc_tiles=tiles, mc_warps=warps)
                    out = radix_ops.gemv_mc(rt2, xm)
                    errs = [pc.gemv_error(out[i], ym[i][0], ym[i][1]) for i in range(M)]
                    if not max(errs) <= 1:
                        raise AssertionError(f"{name} {mc_arm} mc t{tiles or 'row'}/w{warps} M={M} exceeds the bound: {max(errs)}")
                    label = f"{mc_arm}:t{'row' if tiles == 0 else tiles}/w{warps}:M{M}"
                    item["mc"][label] = dict(arm=mc_arm, M=M, tiles=tiles, warps=warps, err_over_bound=max(errs), samples={})
                    item["mc_fns"][label] = (lambda rt2, x: (lambda: radix_ops.gemv_mc(rt2, x)))(rt2, xm)
                # the other radix arms' mc at the table's row (t1/w1) as controls
                for arm in radix_arms:
                    if arm == mc_arm:
                        continue
                    rt2 = pc.with_launch(rts[arm], 1, 2, mc_tiles=1, mc_warps=1)
                    out = radix_ops.gemv_mc(rt2, xm)
                    errs = [pc.gemv_error(out[i], ym[i][0], ym[i][1]) for i in range(M)]
                    if not max(errs) <= 1:
                        raise AssertionError(f"{name} {arm} mc t1/w1 M={M} exceeds the bound: {max(errs)}")
                    label = f"{arm}:t1/w1:M{M}"
                    item["mc"][label] = dict(arm=arm, M=M, tiles=1, warps=1, err_over_bound=max(errs), samples={})
                    item["mc_fns"][label] = (lambda rt2, x: (lambda: radix_ops.gemv_mc(rt2, x)))(rt2, xm)
        del src_gpu, y, bound
        items.append(item)
        print(f"LOADED_AND_CHECKED {name} {R}x{C} class={item['shape_class']} arms={len(item['arms'])} {time.perf_counter() - t0:.1f}s", flush=True)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    report["mem_free_after_load_bytes"] = free
    print(f"loaded {len(items)} tensors; device free {free / 2**30:.1f} GiB of {total / 2**30:.1f}", flush=True)
    # ---- time ----------------------------------------------------------------------
    arm_keys = list(items[0]["arms"].keys())
    for it in items:  # the union, in first-seen order (down_proj has t4 that others fold into row)
        for k in it["arms"]:
            if k not in arm_keys:
                arm_keys.append(k)
    # config by config, every arm at it back to back: drift lands on a
    # config's arms together, not on one arm's whole grid
    labels = list(dict.fromkeys(k.split(":", 1)[1] for k in arm_keys))
    arm_keys.sort(key=lambda k: (labels.index(k.split(":", 1)[1]), radix_arms.index(k.split(":", 1)[0])))
    report["pass_totals_ms"] = {}
    with (Sensors(interval=1e9) if a.no_sensors else pick_sensors()) as sensors:
        for repeat in range(a.repeats):
            for key in arm_keys:
                have = [it for it in items if key in it["fns"]]
                fns = [it["fns"][key] for it in have]
                per, totals = pc.rotation(fns, a.passes, prime=a.primer)
                for it, samples in zip(have, per):
                    it["arms"][key]["samples"][f"r{repeat}"] = samples
                report["pass_totals_ms"].setdefault(key, {})[f"r{repeat}"] = totals
                total_med = sum(float(np.median(s)) for s in per)
                print(f"ROTATION r{repeat} {key:16s} set-sum {total_med:8.4f} ms  pass-median {float(np.median(totals)):8.4f} ms", flush=True)
            if a.mc_Ms:
                mc_keys = list(items[0]["mc"].keys())
                for key in mc_keys:
                    fns = [it["mc_fns"][key] for it in items]
                    per, totals = pc.rotation(fns, a.passes, prime=a.primer)
                    for it, samples in zip(items, per):
                        it["mc"][key]["samples"][f"r{repeat}"] = samples
                    print(f"ROTATION-MC r{repeat} {key:22s} set-sum {sum(float(np.median(s)) for s in per):8.4f} ms", flush=True)
    report["sensors"] = sensors.summary() if not a.no_sensors else "off"
    # ---- reduce --------------------------------------------------------------------
    for it in items:
        it.pop("fns")
        it.pop("mc_fns", None)
        for key, cell in it["arms"].items():
            pooled = [s for rep in cell["samples"].values() for s in rep]
            cell["median_ms"] = float(np.median(pooled))
            cell["median_per_repeat_ms"] = {r: float(np.median(s)) for r, s in cell["samples"].items()}
            cell["min_ms"] = float(min(pooled))
            cell["max_ms"] = float(max(pooled))
            cell["efficiency"] = cell["floor_ms"] / cell["median_ms"]
            cell["achieved_GB_s"] = cell["bytes"]["total"] / cell["median_ms"] / 1e6
        if "mc" in it:
            for key, cell in it["mc"].items():
                pooled = [s for rep in cell["samples"].values() for s in rep]
                cell["median_ms"] = float(np.median(pooled))
                cell["median_per_repeat_ms"] = {r: float(np.median(s)) for r, s in cell["samples"].items()}
        report["tensors"].append(it)
    # per-class sums (over every tensor of the class in the set) and the pick
    classes = {}
    for it in items:
        classes.setdefault(it["shape_class"], []).append(it)
    def served_as(it, key):
        # the config a class row names, as this tensor runs it: blocks
        # per program >= its row's blocks IS its whole-row program (a
        # 1024-wide tensor has no t1 of its own), so a class mixing widths
        # (Qwen3-0.6B's narrow: 1024 and 2048/3072 wide) sums each tensor
        # at its own effective config
        arm, label = key.split(":", 1)
        t, w = parse_config(label)
        eff = 0 if (t == 0 or t >= it["nb"]) else t
        return f"{arm}:t{'row' if eff == 0 else eff}/w{w}"

    picks = {}
    for cls, its in classes.items():
        sums = {}
        for key in arm_keys:
            cells = [it["arms"].get(served_as(it, key)) for it in its]
            if all(cells):
                sums[key] = dict(median_ms=sum(c["median_ms"] for c in cells),
                                 floor_ms=sum(c["floor_ms"] for c in cells),
                                 per_repeat_ms={r: sum(c["median_per_repeat_ms"][r] for c in cells)
                                                for r in cells[0]["median_per_repeat_ms"]})
                sums[key]["efficiency"] = sums[key]["floor_ms"] / sums[key]["median_ms"]
        picks[cls] = dict(tensors=[it["name"] for it in its], sums=sums, picks={})
        for arm in grid_arms:
            arm_keys_ = [k for k in sums if k.startswith(arm + ":")]
            best = min(arm_keys_, key=lambda k: sums[k]["median_ms"])
            spike = f"{arm}:{SPIKE}"
            pk = dict(best=best, tiles=parse_config(best.split(":", 1)[1])[0],
                      warps=parse_config(best.split(":", 1)[1])[1],
                      best_ms=sums[best]["median_ms"],
                      # every config within 1% of the best: the pick is inside noise among these
                      within_1pct=[k for k in arm_keys_ if sums[k]["median_ms"] <= 1.01 * sums[best]["median_ms"]])
            if spike in sums:
                pk.update(spike_ms=sums[spike]["median_ms"], best_vs_spike=sums[best]["median_ms"] / sums[spike]["median_ms"])
            picks[cls]["picks"][arm] = pk
        # the first grid arm's pick at the top level too (the parity report's shape)
        if grid_arms:
            picks[cls].update(picks[cls]["picks"][grid_arms[0]])
    report["classes"] = picks
    # the mc arm per class: per M the config sums; the pick = the mc arm's
    # (the first grid arm's) config with the least time summed over the
    # measured Ms; the other arms' t1/w1 controls sit in per_M beside it
    if a.mc_Ms:
        mc_arm = grid_arms[0] if grid_arms else radix_arms[0]
        mc_picks = {}
        for cls, its in classes.items():
            keys = list(its[0]["mc"].keys())
            per_M = {}
            for key in keys:
                cfg, M = key.rsplit(":M", 1)
                per_M.setdefault(int(M), {})[cfg] = sum(it["mc"][key]["median_ms"] for it in its)
            cfgs = [k.rsplit(":M", 1)[0] for k in keys if k.startswith(mc_arm + ":")]
            cfgs = list(dict.fromkeys(cfgs))
            total = {c: sum(per_M[M][c] for M in per_M) for c in cfgs}
            best = min(cfgs, key=total.get)
            t, w = parse_config(best.split(":", 1)[1])
            mc_picks[cls] = dict(arm=mc_arm, per_M={str(M): v for M, v in sorted(per_M.items())}, total_ms=total,
                                 best=best, mc_tiles=t, mc_warps=w)
            if f"{mc_arm}:{SPIKE}" in total:
                mc_picks[cls]["best_vs_spike"] = total[best] / total[f"{mc_arm}:{SPIKE}"]
        report["mc_classes"] = mc_picks
    # layer sums (the seven projections) per layer, and lm_head
    layer_sums = {}
    layers_present = sorted({int(it["name"].split(".")[2]) for it in items if it["name"].startswith("model.layers.")})
    for layer in layers_present:
        its = [it for it in items if it["name"].startswith(f"model.layers.{layer}.")]
        layer_sums[str(layer)] = {key: dict(median_ms=sum(it["arms"][served_as(it, key)]["median_ms"] for it in its),
                                            floor_ms=sum(it["arms"][served_as(it, key)]["floor_ms"] for it in its))
                                  for key in arm_keys if all(served_as(it, key) in it["arms"] for it in its)}
        for v in layer_sums[str(layer)].values():
            v["efficiency"] = v["floor_ms"] / v["median_ms"]
    report["layer_sums"] = layer_sums
    have_head = any(it["name"] == pc.LM_HEAD for it in items)
    if have_head:
        head = next(it for it in items if it["name"] == pc.LM_HEAD)
        report["lm_head"] = {key: dict(median_ms=c["median_ms"], floor_ms=c["floor_ms"], efficiency=c["efficiency"],
                                       per_repeat=c["median_per_repeat_ms"]) for key, c in head["arms"].items()}
    report["vram_used_bytes_end"] = int(vram.read_text()) if vram.exists() else None
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    del filler
    report["verdict"] = "PASS"
    pc.write_json(a.json, report)
    # ---- print -------------------------------------------------------------------
    print("\nPER CLASS (sum over the class's tensors in the set; ms, pooled median; eff = floor/median):")
    for cls, pk in picks.items():
        print(f"  {cls:7s} {pk['tensors']}")
        bests = {p["best"] for p in pk["picks"].values()}
        for key, s in sorted(pk["sums"].items(), key=lambda kv: kv[1]["median_ms"]):
            reps = " ".join(f"{v:.4f}" for v in s["per_repeat_ms"].values())
            mark = " <- best" if key in bests else (" (spike)" if key.endswith(":" + SPIKE) else "")
            print(f"    {key:20s} {s['median_ms']:8.4f} ms  eff {s['efficiency']:.3f}  reps [{reps}]{mark}")
        for arm, p in pk["picks"].items():
            spike = f" ({p['best_vs_spike']:.3f}x spike)" if "spike_ms" in p else ""
            print(f"    {arm}: best {p['best']} = {p['best_ms']:.4f} ms{spike}; within 1%: {p['within_1pct']}")
    if a.mc_Ms:
        print("MC PICKS per class (sum over the measured Ms):")
        for cls, mp in report["mc_classes"].items():
            spike = f" = {mp['best_vs_spike']:.3f}x spike" if "best_vs_spike" in mp else ""
            print(f"    {cls:7s} best {mp['best']}{spike}; "
                  + "; ".join(f"M={M}: " + ", ".join(f"{c} {v:.3f}" for c, v in row.items()) for M, row in mp["per_M"].items()))
    for layer, sums in layer_sums.items():
        print(f"LAYER {layer} seven-tensor sums:")
        for key, s in sorted(sums.items(), key=lambda kv: kv[1]["median_ms"]):
            print(f"    {key:20s} {s['median_ms']:8.4f} ms  eff {s['efficiency']:.3f}")
    if have_head:
        print("LM_HEAD:")
        for key, s in sorted(report["lm_head"].items(), key=lambda kv: kv[1]["median_ms"]):
            print(f"    {key:20s} {s['median_ms']:8.4f} ms  eff {s['efficiency']:.3f}  reps {list(s['per_repeat'].values())}")
    if a.mc_Ms:
        print("MC (sum over the set):")
        for key in items[0]["mc"]:
            print(f"    {key:24s} {sum(it['mc'][key]['median_ms'] for it in items):8.4f} ms")
    print(f"sensors: {json.dumps(report['sensors'])}")
    print("PARITY_ROTATION PASS", flush=True)


if __name__ == "__main__":
    status = 1
    try:
        main()
        status = 0
    except BaseException:
        print("PARITY_ROTATION FAIL", flush=True)
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
