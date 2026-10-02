"""Shared loaders, checks and the rotation protocol for the launch-schedule
instruments: bench/radix_schedule_bitpin.py (the gate),
bench/parity_rotation.py (the config sweep), and the Sensors sampler
(sysfs sclk/temp/power).

Every tensor comes off the REAL packs on disk through the serve loader's own
path (pack.iter_pack_dir -> swap.to_device_radix / swap.make_module), so what
is timed and gated is what is served; the launch schedule of a radix runtime
dict is then set per config by `with_launch`. Source bits come from the HF
snapshot's safetensors (the checkpoint the packs were cut from) and
from the CPU reference decoder (radix_pack.decode_back_radix). The
instruments compare the profiles and schedules against each other and
against the wall (the original parity measurements also carried the
prior codec's kernel as an in-session control; that arm is not on this tree).

The rotation protocol: back-to-back passes over a
tensor set with an event pair around each call and no explicit cache flush
— the other tensors in the pass (hundreds of MB) evict each one, which is
the model's own regime. do_bench's 256 MiB memset before every timed call
makes bandwidth-bound kernels look 1.4-1.6x slower on this APU, so it is
not used for any conclusion here.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import triton

# The Qwen3-8B packs the instruments load, one per profile: the sip pack
# at the default pack directory (`drinkme pack --model Qwen/Qwen3-8B
# --revision b968826d9c46…`, no -o — the directory names the model and
# revision only), the other two beside it at their own defaults
# `<root>/packs/Qwen--Qwen3-8B-<profile>@b968826d9c46` (`drinkme pack
# --gulp`; DRINKME_COMPRESSION_PROFILE=balanced). DRINKME_PARITY_PACK_{SIP,
# BALANCED,GULP} point them elsewhere (the Modal containers of
# bench/modal_nvidia2.py: packs built in-container or on the 27B volume).
_PACKS_ROOT = os.path.expanduser("~/.cache/drinkme/packs")
PACKS = {
    "sip": os.environ.get("DRINKME_PARITY_PACK_SIP",
                          os.path.join(_PACKS_ROOT, "Qwen--Qwen3-8B@b968826d9c46")),
    "balanced": os.environ.get("DRINKME_PARITY_PACK_BALANCED",
                               os.path.join(_PACKS_ROOT, "Qwen--Qwen3-8B-balanced@b968826d9c46")),
    "gulp": os.environ.get("DRINKME_PARITY_PACK_GULP",
                           os.path.join(_PACKS_ROOT, "Qwen--Qwen3-8B-gulp@b968826d9c46")),
}
# The checkpoint the packs were cut from: DRINKME_PARITY_SNAPSHOT, else
# the HF cache's snapshot for the sip pack's meta (hfRepo @ revision) under
# HF_HOME, else the default HF cache's Qwen3-8B snapshot.
_SNAPSHOT_ENV = os.environ.get("DRINKME_PARITY_SNAPSHOT")
SNAPSHOT = Path(_SNAPSHOT_ENV) if _SNAPSHOT_ENV else None
_SNAPSHOT_DEFAULT = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"))
PROJS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
LM_HEAD = "lm_head"


def pack_meta(arm: str = "sip") -> dict:
    return json.loads((Path(PACKS[arm]) / "meta.json").read_text())


def snapshot_dir() -> Path:
    """The checkpoint snapshot directory (module comment on SNAPSHOT)."""
    global SNAPSHOT
    if SNAPSHOT is None:
        try:
            meta = pack_meta("sip")
            hub = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
            cand = hub / f"models--{meta['hfRepo'].replace('/', '--')}" / "snapshots" / meta["revision"]
            SNAPSHOT = cand if cand.exists() else _SNAPSHOT_DEFAULT
        except (OSError, KeyError, ValueError):
            SNAPSHOT = _SNAPSHOT_DEFAULT
    return SNAPSHOT


def layer_names(layer: int) -> tuple[str, ...]:
    """The projections of one decoder layer: Qwen3's seven by name, or —
    for a checkpoint whose layers are not shaped like that (the hybrid
    27B's DeltaNet layers: in_proj_qkv / in_proj_z / out_proj + the MLP)
    — every coded tensor the sip pack's meta lists under that
    layer, in the meta's order."""
    try:
        meta = pack_meta("sip")
    except (OSError, KeyError, ValueError):
        return tuple(f"model.layers.{layer}.{p}" for p in PROJS)
    prefix = f"model.layers.{layer}."
    names = tuple(n for n in meta["tensors"] if n.startswith(prefix))
    qwen3 = tuple(prefix + p for p in PROJS)
    return qwen3 if set(qwen3) <= set(names) else names


def short(name: str) -> str:
    return name.split(".", 2)[-1] if name.startswith("model.layers.") else name


def load_pack_dicts(pack_dir: str, names) -> dict:
    """name -> pack dict (numpy arrays), one npz at a time, for the named tensors."""
    from drinkme.codec.pack import iter_pack_dir
    want = set(names)
    out = {}
    for name, p in iter_pack_dir(pack_dir):
        if name in want:
            out[name] = p
            if len(out) == len(want):
                break
    missing = want - set(out)
    if missing:
        raise KeyError(f"not in {pack_dir}: {sorted(missing)}")
    return out


def to_runtime(p: dict, device="cuda") -> dict:
    """The served runtime dict (swap.to_device_radix) at the `spike` launch
    schedule; the instruments set rx_launch per config afterwards."""
    from drinkme.codec.swap import to_device_radix
    from drinkme.codec.radix_schedule import select
    rt = to_device_radix(p, device)
    rt["rx_launch"] = select(int(p["R"]), int(p["C"]), int(p["block_size"]), p["widths"], override="spike").as_dict()
    return rt


def with_launch(rt: dict, tiles: int, warps: int, mc_tiles: int | None = None, mc_warps: int | None = None) -> dict:
    """A shallow copy of the runtime dict at another launch schedule (the
    device buffers are shared; only rx_launch differs)."""
    out = dict(rt)
    base = dict(rt["rx_launch"])
    base.update(gemv_tiles=int(tiles), gemv_warps=int(warps))
    if mc_tiles is not None:
        base["mc_tiles"] = int(mc_tiles)
    if mc_warps is not None:
        base["mc_warps"] = int(mc_warps)
    out["rx_launch"] = base
    return out




_SAFE_INDEX = None


def checkpoint_key(name: str) -> str:
    """The checkpoint key for a pack tensor name: the name itself, else the
    wrapped spelling arms._TEXT_WRAP undoes at load (Qwen3.8-27B ships its
    text weights as model.language_model.* and the pack calls them
    model.*), else the one index key that ends in `.name.weight`."""
    from drinkme.arms import _TEXT_WRAP
    key = name + ".weight"
    if key in _SAFE_INDEX:
        return key
    for ckpt_prefix, skel_prefix in _TEXT_WRAP.values():
        if name.startswith(skel_prefix):
            cand = ckpt_prefix + name[len(skel_prefix):] + ".weight"
            if cand in _SAFE_INDEX:
                return cand
    tails = [k for k in _SAFE_INDEX if k.endswith("." + key)]
    if len(tails) == 1:
        return tails[0]
    raise KeyError(f"{name}: not in the checkpoint index (tried {key!r}, the wrapped spellings; suffix matches {tails})")


def source_bits(name: str) -> np.ndarray:
    """The checkpoint's uint16 bf16 bits for `name` (no '.weight' suffix)."""
    global _SAFE_INDEX
    from safetensors import safe_open
    snap = snapshot_dir()
    if _SAFE_INDEX is None:
        index = snap / "model.safetensors.index.json"
        if index.exists():
            _SAFE_INDEX = json.loads(index.read_text())["weight_map"]
        else:  # a one-shard checkpoint (Qwen3-0.6B): every key in model.safetensors
            with safe_open(str(snap / "model.safetensors"), framework="pt") as f:
                _SAFE_INDEX = {k: "model.safetensors" for k in f.keys()}
    key = checkpoint_key(name)
    with safe_open(str(snap / _SAFE_INDEX[key]), framework="pt") as f:
        t = f.get_tensor(key)
    if t.dtype != torch.bfloat16:
        raise TypeError(f"{key}: {t.dtype}, expected bf16")
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def configs_for(nb: int, tiles_grid=(1, 2, 4, 0), warps_grid=(1, 2, 4)):
    """The config grid for a row of nb blocks: (label, tiles, warps), with
    tiles >= nb collapsed onto the whole-row config (tiles=0) so no config
    is timed twice under two names."""
    out = []
    seen = set()
    for t in tiles_grid:
        eff = 0 if (t == 0 or t >= nb) else t
        for w in warps_grid:
            key = (eff, w)
            if key in seen:
                continue
            seen.add(key)
            out.append((f"t{'row' if eff == 0 else eff}/w{w}", eff, w))
    return out


def radix_bytes(rt: dict, tiles: int, x_elem: int = 2, out_elem: int = 2) -> dict:
    """Bytes one M=1 call addresses at this schedule (the roofline's
    radix_bytes's accounting): payload words, the directory, the padded
    palette tables, the schedule (gulp), the activation once, the split-K
    partials written + read back (tiles < row), the output. A twin dict
    (swap.to_device_twin) reads its bf16 weight in place of the streams."""
    R, C, B = int(rt["R"]), int(rt["C"]), int(rt["block_size"])
    nb = triton.cdiv(C, B)
    t = tiles or nb
    splits = triton.cdiv(nb, t)
    if rt.get("codec") == "twin":  # swap.to_device_twin: the raw bf16 weight, no streams
        weight, directory, palette, schedule = R * C * 2, 0, 0, 0
    else:
        weight = int(rt.get("NW", rt["rx_data"].numel())) * 4  # the payload; not the reach pad after it
        directory = rt["rx_offsets"].numel() * 4
        palette = rt["rx_palette"].numel() * 4
        schedule = rt["rx_schedule"].numel() * 2
    activation = C * x_elem
    partial = 0 if splits == 1 else R * splits * 4 * 2
    output = R * out_elem
    return dict(weight_stream=weight, directory=directory, palette=palette, schedule=schedule,
                activation=activation, partial=partial, output=output,
                total=weight + directory + palette + schedule + activation + partial + output,
                resident=weight + directory + palette + schedule, splits=splits, programs=R * splits)




def oracle(x: torch.Tensor, bits_gpu: torch.Tensor, chunk: int = 16384):
    """(y, bound) in float64 for one bf16 activation row against the
    uint16 weight bits, row-chunked so lm_head's 5 GB float64 twin never
    exists at once: y = W.x, bound = 2e-6 * |W|.|x| + 1e-7 (the research
    gate's cancellation-aware bound)."""
    R = bits_gpu.shape[0]
    xd = x.reshape(-1).double()
    xa = xd.abs()
    y = torch.empty(R, dtype=torch.float64, device=x.device)
    bound = torch.empty_like(y)
    for r0 in range(0, R, chunk):
        w = bits_gpu[r0:r0 + chunk].view(torch.bfloat16).double()
        y[r0:r0 + chunk] = w @ xd
        bound[r0:r0 + chunk] = 2e-6 * (w.abs() @ xa) + 1e-7
        del w
    return y, bound


def gemv_error(value: torch.Tensor, y: torch.Tensor, bound: torch.Tensor) -> float:
    """max |value - y| / bound (<= 1 is within the bound)."""
    return float(((value.reshape(-1).double() - y).abs() / bound).max().item())


_PRIMER = None


def primer(mb: int = 256):
    """A ~1 ms GPU backlog: a reduction over `mb` MB enqueued at the head
    of a pass so the GPU is busy while the host enqueues the pass's first
    kernels. Without it the first calls of a pass (the GPU queue is empty
    right after the previous pass's synchronize) time the HOST's launch
    jitter inside their event pair — parity/rotation_session{1,2,3}.json
    show layer-0 q/k/v/o (pass positions 1-4) swinging 2x pass to pass
    while the same shapes at positions 8-11 are steady. The primer's own
    time is outside every event pair; it also evicts L2, which the previous
    pass's lm_head/gate/up already did."""
    global _PRIMER
    if _PRIMER is None or _PRIMER.numel() != mb * 2**18:
        _PRIMER = torch.ones(mb * 2**18, dtype=torch.float32, device="cuda")
    return _PRIMER.sum()


def rotation(fns, passes: int, settle_s: float = 0.3, prime: bool = False):
    """The roofline's rotation protocol: warm every callable for
    `settle_s`, then `passes` back-to-back passes over them with an event
    pair around each call. `prime` enqueues primer() at the head of every
    pass (see there). Returns (per-fn ms lists, pass totals ms)."""
    deadline = time.time() + settle_s
    while time.time() < deadline:
        if prime:
            primer()
        for fn in fns:
            fn()
        torch.cuda.synchronize()
    per = [[] for _ in fns]
    totals = []
    for _ in range(passes):
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in fns]
        whole = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        if prime:
            primer()
        whole[0].record()
        for (s, e), fn in zip(events, fns):
            s.record(); fn(); e.record()
        whole[1].record()
        torch.cuda.synchronize()
        for i, (s, e) in enumerate(events):
            per[i].append(s.elapsed_time(e))
        totals.append(whole[0].elapsed_time(whole[1]))
    return per, totals


def wall_bytes_s(path="verification/parity/wall.json") -> int:
    report = json.loads(Path(path).read_text())
    if report["verdict"] != "PASS":
        raise ValueError(f"{path} is not a PASS record")
    return int(report["read_bytes_s_median"])


def machine_facts() -> dict:
    import subprocess
    free, total = torch.cuda.mem_get_info()
    gtt = Path("/sys/class/drm/card1/device/mem_info_gtt_used")
    props = torch.cuda.get_device_properties(0)
    facts = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, hip=getattr(torch.version, "hip", None),
                 cuda=getattr(torch.version, "cuda", None), triton=triton.__version__,
                 # gcnArchName is the ROCm build's; a CUDA build has the capability
                 arch=getattr(props, "gcnArchName", None) or f"sm_{props.major}{props.minor}",
                 multi_processor_count=props.multi_processor_count,
                 mem_free_bytes=free, mem_total_bytes=total, hostname=os.uname().nodename,
                 gtt_used_bytes=int(gtt.read_text()) if gtt.exists() else None)
    # PARITY_WATCH_UNITS: comma-separated systemd user units whose state is
    # recorded beside the timings (another model server sharing the device,
    # say); mem_free_bytes and gtt_used_bytes record contention either way.
    facts["watched_units"] = {}
    for unit in filter(None, os.environ.get("PARITY_WATCH_UNITS", "").split(",")):
        try:
            facts["watched_units"][unit] = subprocess.run(
                ["systemctl", "--user", "is-active", unit],
                capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception as e:  # noqa: BLE001
            facts["watched_units"][unit] = f"unknown ({e})"
    return facts


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


class Sensors:
    """Samples GPU sclk/temperature/power from sysfs in a thread; reports min/median/max."""
    ROOT = Path('/sys/class/drm/card1/device')

    def __init__(self, interval=0.05):
        import threading
        self.interval = interval
        self.samples = {'sclk_mhz': [], 'temp_c': [], 'power_w': [], 'busy_pct': []}
        self.files = {}
        for key, name in (('sclk_mhz', 'freq1_input'), ('temp_c', 'temp1_input'), ('power_w', 'power1_input')):
            found = list(self.ROOT.glob('hwmon/hwmon*/' + name))
            if found:
                self.files[key] = found[0]
        # the GPU's own busy figure: another process's kernels show here
        if (self.ROOT / 'gpu_busy_percent').exists():
            self.files['busy_pct'] = self.ROOT / 'gpu_busy_percent'
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        scale = {'sclk_mhz': 1e-6, 'temp_c': 1e-3, 'power_w': 1e-6, 'busy_pct': 1}
        while not self._stop.is_set():
            for key, path in self.files.items():
                try:
                    self.samples[key].append(int(path.read_text()) * scale[key])
                except OSError:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        out = {}
        for key, values in self.samples.items():
            if values:
                out[key] = dict(min=float(min(values)), median=float(np.median(values)), max=float(max(values)), samples=len(values))
        return out
