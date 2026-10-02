#!/usr/bin/env python3
"""MTP cycle floor — bench/host_floor.py's method applied to ONE draft/verify
cycle, so the drafter's cost coefficient `c` is measured instead of assumed.

Leviathan et al.'s walltime factor (1-a^(k+1)) / ((1-a)(kc+1)) says draft
DEPTH is flat at our acceptance (~2% across k=4..7 at a=0.70) and that `c`
— the wall of ONE draft step over the wall of ONE trunk step — is the lever:
c=0.05 gives 2.31x at k=4, c->0 gives 2.77x. Nobody had measured c on this
box; this file does, before and after any change to serving/mtp.py's chain.

METHOD (host_floor.py's, on the cycle): run the REAL engine loop
(HFEngine.generate -> mtp.Speculator.cycle -> MTPHead.draft ->
mtp.forward_with_hidden) and time its pieces through wrappers on those very
objects — nothing is re-implemented. Then rebind every compressed Linear's
class to a stub whose forward launches a no-op of identical output shape
(weights stay resident, no bytes read, no math) and time the same loop
again. Three arms:

  real       everything real
  stub_head  the head's OWN Linears (fc, its decoder layer) stubbed; the
             trunk — including lm_head, which the draft borrows — real. The
             chain is then dispatch + k lm_head reads + the embedding gather.
  stub_all   every compressed Linear in trunk and head stubbed: the host
             floor of the whole cycle; the chain is dispatch and sync alone.

  head layer bytes  = chain(real)      - chain(stub_head)
  lm_head reads     = chain(stub_head) - chain(stub_all)      (k of them)
  dispatch + sync   = chain(stub_all)

Each arm runs each path (greedy, sampled) twice: WHOLE — wrappers record
timestamps only, no synchronize, so the cycle's period (cycle start to next
cycle start) is exactly what the server pays; and DISSECTED — the chain and
the verify each bracketed by torch.cuda.synchronize, which is what makes the
chain's wall a number (at the price of the overlap a sync-free chain buys,
so the dissected sum can exceed the whole period after the change). The
trunk's serial M=1 step is timed the same way on a serial generation (the
engine's own loop with MTP depth 0 for that request).

  c              = (chain wall / k) / trunk step wall     (the paper's c)
  chain / trunk  = k * c
  host floor     = stub_all's cycle period; host fraction = floor / real

The stubbed arms produce garbage tokens (all-zero logits: argmax 0, every
draft accepted); the greedy stubbed cycle therefore always rebuilds k head
entries, which is the maximal rebuild and stated in the receipt (m per
cycle is recorded).

Run on the 27B pack (GPU; ~5 min):

    PYTHONPATH=src python bench/mtp_cycle_floor.py \\
        --pack ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 \\
        --repeats 3 --json verification/<dir>/cycle_floor_<when>.json

PHASES WITHOUT SYNCS. The whole run also brackets each cycle's phases with
device events (CUDA/HIP), recorded in the stream and read only after the
generation ends, so they add no sync: draft (the head's k steps), verify
(the M=k+1 trunk forward, the DeltaNet capture inside it), accept (the
host-side pick loop and its device reads), restore (_restore_rows: the
DeltaNet state and conv window copies, the KV counters) and rebuild (the
head's crop and its m-row re-run). Each is the device timeline between two
events, so a host-bound gap shows up in the phase that waited on it. The
serial step gets an event pair too. `--stock` runs the same on the BF16
checkpoint (the head raw, stub arms meaningless there: use --arms real), and
`--prompt-name chat|code|...` takes a prompt from bench/ngram_gpu_ab.py.

    PYTHONPATH=src python bench/mtp_cycle_floor.py --stock --arms real \\
        --prompt-name chat --tokens 256 --repeats 1 --no-sampled --json /tmp/cf_bf16.json

CPU smoke on the toy (no GPU, no download; stubs the toy's nn.Linears since
it has no compressed ones — the harness path, not a speed claim):

    PYTHONPATH=src python bench/mtp_cycle_floor.py --toy --repeats 1

Verdict is in the output, never in $?: TheRock torch _exit(0)s on atexit.
"""

import argparse
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, "src")

PROMPT = "How do I kill a Python process that is hogging my GPU?"


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", default=os.path.expanduser(
        "~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60"))
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=48,
                    help="new tokens per timed generation")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--json", default=None)
    ap.add_argument("--toy", action="store_true",
                    help="the tests' synthetic qwen3_5 on CPU (harness smoke)")
    ap.add_argument("--no-sampled", action="store_true")
    ap.add_argument("--arms", default="real,stub_head,stub_all")
    ap.add_argument("--stock", action="store_true",
                    help="the BF16 checkpoint, not the pack (use --arms real)")
    ap.add_argument("--prompt-name", default=None,
                    help="a prompt from bench/ngram_gpu_ab.py's PROMPTS (overrides --prompt)")
    args = ap.parse_args()
    if args.prompt_name:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ngram_gpu_ab", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngram_gpu_ab.py"))
        ab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ab)
        args.prompt = ab.PROMPTS[args.prompt_name]
    return args


ARGS = parse()
# read at engine build (the head loads iff DRINKME_MTP_DEPTH asks) and at the first
# generation (the depth); the serial arm flips it per request (set_arm)
os.environ["DRINKME_MTP_DEPTH"] = str(ARGS.depth)
os.environ.setdefault("DRINKME_PREFIX_SLOTS", "0")

import torch  # noqa: E402

from drinkme.codec.swap import CompressedLinear  # noqa: E402
from drinkme.serving import mtp  # noqa: E402
from drinkme.serving.engine import GenerationRequest, SampleParams, complete  # noqa: E402


# ------------------------------------------------------------------ stubs --


def _stub_forward(self, x):
    """host_floor.StubbedLinear.forward: identical output shape and dtype,
    the same zeros-then-cast pair of launches, no weight read, no math."""
    shape = x.shape
    R = self.R if hasattr(self, "R") else self.out_features
    C = self.C if hasattr(self, "C") else self.in_features
    xf = x.reshape(-1, C)
    outs = torch.zeros(xf.shape[0], R, device=x.device, dtype=torch.float32)
    out = outs.to(x.dtype).reshape(*shape[:-1], R)
    if self.bias is not None:
        out = out + self.bias
    return out


_STUB_CLASSES: dict = {}


def _stub_class(cls):
    """A stub SUBCLASS per real class, so `__class__` can be rebound in place
    (the module keeps its buffers, its residency, its dict) and put back."""
    if cls not in _STUB_CLASSES:
        _STUB_CLASSES[cls] = type("Stubbed" + cls.__name__, (cls,),
                                  {"forward": _stub_forward, "_drinkme_stub_of": cls})
    return _STUB_CLASSES[cls]


def linears(module, raw: bool):
    """The Linears whose kernel the floor removes: every CompressedLinear,
    plus plain nn.Linear when `raw` (the toy has nothing else)."""
    out = []
    for m in module.modules():
        if isinstance(m, CompressedLinear) or (raw and type(m) is torch.nn.Linear):
            out.append(m)
    return out


class Stubbed:
    """Context: rebind `mods` to their stub classes, restore on exit."""

    def __init__(self, mods):
        self.mods = mods

    def __enter__(self):
        for m in self.mods:
            m.__class__ = _stub_class(type(m))
        return self

    def __exit__(self, *exc):
        for m in self.mods:
            m.__class__ = m.__class__._drinkme_stub_of


# ------------------------------------------------------------- the probe --


class Probe:
    """Timing wrappers over the real objects. `dissect` brackets the chain
    and the verify with device syncs; otherwise only timestamps are taken."""

    def __init__(self, engine, dissect: bool):
        self.engine = engine
        self.dissect = dissect
        self.cuda = torch.device(engine.device).type == "cuda"
        self.serial: list[dict] = []
        self.cycles: list[dict] = []
        self._cur = None
        self._last_cycle_start = None
        self._last_serial_start = None
        # device events per phase, whole runs only (module docstring)
        self.events = self.cuda and not dissect
        self._ev_serial: list[tuple] = []

    def mark(self, name):
        """Record a device event named `name` into the current cycle."""
        if self.events and self._cur is not None:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._cur.setdefault("_ev", []).append((name, e))

    def resolve_events(self):
        """After the generation: turn each cycle's events into phase times,
        the device timeline between consecutive marks."""
        if not self.events:
            return
        torch.cuda.synchronize()
        for rec in self.cycles:
            evs = rec.pop("_ev", None)
            if not evs:
                continue
            phases = {}
            for (a, ea), (b, eb) in zip(evs, evs[1:]):
                phases[b] = phases.get(b, 0.0) + ea.elapsed_time(eb)
            rec["ev_ms"] = {k: round(v, 3) for k, v in phases.items()}
            rec["ev_span_ms"] = round(evs[0][1].elapsed_time(evs[-1][1]), 3)
        for i, (e0, e1) in enumerate(self._ev_serial):
            self.serial[i]["ev_ms"] = round(e0.elapsed_time(e1), 3)

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def _wrap_head(self, head):
        real_draft, real_sampled = head.draft, head.draft_sampled

        def timed(fn):
            def w(*a, **kw):
                if self.dissect:
                    self.sync()
                t0 = time.perf_counter()
                out = fn(*a, **kw)
                self.mark("draft")
                t1 = time.perf_counter()
                if self.dissect:
                    self.sync()
                t2 = time.perf_counter()
                if self._cur is not None:
                    self._cur["chain_enqueue_ms"] = (t1 - t0) * 1e3
                    self._cur["chain_wall_ms"] = (t2 - t0) * 1e3 if self.dissect else None
                return out
            return w

        head.draft = timed(real_draft)
        head.draft_sampled = timed(real_sampled)
        return lambda: (setattr(head, "draft", real_draft),
                        setattr(head, "draft_sampled", real_sampled))

    def _wrap_verify(self):
        real = mtp.forward_with_hidden

        def w(model, input_ids, cache, cache_position, last_row_only=False, **kw):
            if last_row_only or self._cur is None:  # prefill / serial_step
                return real(model, input_ids, cache, cache_position, last_row_only, **kw)
            self.mark("pre_verify")
            t0 = time.perf_counter()
            out = real(model, input_ids, cache, cache_position, last_row_only)
            self.mark("verify")
            self._cur["verified"] = True
            t1 = time.perf_counter()
            if self.dissect:
                self.sync()
            t2 = time.perf_counter()
            self._cur["verify_rows"] = int(input_ids.shape[1])
            self._cur["verify_enqueue_ms"] = (t1 - t0) * 1e3
            self._cur["verify_wall_ms"] = (t2 - t0) * 1e3 if self.dissect else None
            return out

        mtp.forward_with_hidden = w
        return lambda: setattr(mtp, "forward_with_hidden", real)

    def _wrap_restore(self):
        """_restore_rows inside a cycle: its start closes the accept phase,
        its end closes the restore phase."""
        real = mtp._restore_rows

        def w(cache, cap, keep):
            if self._cur is None:  # Speculator.finish, outside any cycle
                return real(cache, cap, keep)
            self.mark("accept")
            if self.dissect:
                self.sync()
            t0 = time.perf_counter()
            out = real(cache, cap, keep)
            self.mark("restore")
            if self.dissect:
                self.sync()
            self._cur["restore_wall_ms"] = (time.perf_counter() - t0) * 1e3 if self.dissect else None
            return out

        mtp._restore_rows = w
        return lambda: setattr(mtp, "_restore_rows", real)

    def _wrap_rebuild(self, head):
        """The head's m-row re-run after a cycle (its KV rebuild)."""
        real = head.run

        def w(*a, **kw):
            # prefill seeding runs outside any cycle, and the draft chain
            # calls run() for each of its own steps: only a run after the
            # verify is the rebuild
            if self._cur is None or not self._cur.get("verified"):
                return real(*a, **kw)
            if self.dissect:
                self.sync()
            t0 = time.perf_counter()
            out = real(*a, **kw)
            if self.dissect:
                self.sync()
                self._cur["rebuild_wall_ms"] = (time.perf_counter() - t0) * 1e3
            return out

        head.run = w
        return lambda: setattr(head, "run", real)

    def _wrap_cycle(self):
        real = mtp.Speculator.cycle
        probe = self

        def w(self_spec, *a, **kw):
            if probe.dissect:
                probe.sync()
            t0 = time.perf_counter()
            probe._cur = {"k": None, "m": None}
            probe.mark("start")
            d0, a0 = self_spec.drafted, self_spec.accepted
            out = real(self_spec, *a, **kw)
            probe.mark("rebuild")
            t1 = time.perf_counter()
            if probe.dissect:
                probe.sync()
            t2 = time.perf_counter()
            rec = probe._cur
            probe._cur = None
            rec["k"], rec["m"] = self_spec.drafted - d0, self_spec.accepted - a0
            rec["emitted"] = len(out)
            rec["cycle_host_ms"] = (t1 - t0) * 1e3
            rec["cycle_wall_ms"] = (t2 - t0) * 1e3 if probe.dissect else None
            rec["period_ms"] = (None if probe._last_cycle_start is None
                                else (t0 - probe._last_cycle_start) * 1e3)
            probe._last_cycle_start = t0
            probe.cycles.append(rec)
            return out

        mtp.Speculator.cycle = w
        return lambda: setattr(mtp.Speculator, "cycle", real)

    def _wrap_serial(self):
        """The engine's own M=1 decode step: `self.model(step_in, ...)`. The
        instance attribute is what nn.Module.__call__ resolves; prefill (T>1)
        passes through untimed."""
        model = self.engine.model
        had = model.__dict__.get("forward")  # an instance-level forward, if any
        real = model.forward

        def w(input_ids=None, *a, **kw):
            ids = input_ids if input_ids is not None else kw.get("input_ids")
            if ids is None or ids.shape[1] != 1:
                return real(input_ids, *a, **kw) if input_ids is not None else real(*a, **kw)
            if self.dissect:
                self.sync()
            if self.events:
                e0 = torch.cuda.Event(enable_timing=True)
                e0.record()
            t0 = time.perf_counter()
            out = real(input_ids, *a, **kw)
            if self.events:
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()
                self._ev_serial.append((e0, e1))
            t1 = time.perf_counter()
            if self.dissect:
                self.sync()
            t2 = time.perf_counter()
            self.serial.append({
                "enqueue_ms": (t1 - t0) * 1e3,
                "wall_ms": (t2 - t0) * 1e3 if self.dissect else None,
                "period_ms": (None if self._last_serial_start is None
                              else (t0 - self._last_serial_start) * 1e3)})
            self._last_serial_start = t0
            return out

        model.forward = w

        def restore():
            if had is None:
                del model.__dict__["forward"]
            else:
                model.forward = had
        return restore

    def __enter__(self):
        self._restore = [self._wrap_cycle(), self._wrap_verify(), self._wrap_serial(),
                         self._wrap_restore()]
        if self.engine.mtp_head is not None:
            self._restore.append(self._wrap_head(self.engine.mtp_head))
            self._restore.append(self._wrap_rebuild(self.engine.mtp_head))
        return self

    def __exit__(self, *exc):
        for r in reversed(self._restore):
            r()
        self.resolve_events()


# ------------------------------------------------------------- driving --


def set_arm(engine, depth: int) -> None:
    """bench/mtp_gpu_acceptance.set_arm: depth 0 is DRINKME_SPEC=off, any
    other depth is DRINKME_MTP_DEPTH; `_mtp_depth = None` drops the
    once-per-engine cache."""
    if depth:
        os.environ["DRINKME_MTP_DEPTH"] = str(depth)
        os.environ.pop("DRINKME_SPEC", None)
    else:
        os.environ["DRINKME_SPEC"] = "off"
    engine._mtp_depth = None
    engine._spec_plan = None


def generate(engine, prompt: str, params: SampleParams):
    out = []
    return complete(engine, GenerationRequest([{"role": "user", "content": prompt}], params),
                    lambda d: out.append(d) or True), out


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


def summarize(probe: Probe, warm: int) -> dict:
    """Medians over the timed cycles (the first `warm` cycles of a generation
    are dropped: the first cycle pays the head's KV seed and allocator
    warm-up)."""
    cyc = probe.cycles[warm:]
    ser = probe.serial[warm:]
    emitted = sum(c["emitted"] for c in cyc)
    period = [c["period_ms"] for c in cyc if c["period_ms"] is not None]
    return {
        "n_cycles": len(cyc),
        "k_median": med([c["k"] for c in cyc]),
        "m_mean": round(sum(c["m"] for c in cyc) / len(cyc), 3) if cyc else None,
        "tok_per_cycle": round(emitted / len(cyc), 3) if cyc else None,
        "cycle_period_ms": med(period),
        "cycle_host_ms": med([c["cycle_host_ms"] for c in cyc]),
        "cycle_wall_ms": med([c["cycle_wall_ms"] for c in cyc]),
        "chain_enqueue_ms": med([c.get("chain_enqueue_ms") for c in cyc]),
        "chain_wall_ms": med([c.get("chain_wall_ms") for c in cyc]),
        "verify_rows": med([c.get("verify_rows") for c in cyc]),
        "verify_enqueue_ms": med([c.get("verify_enqueue_ms") for c in cyc]),
        "verify_wall_ms": med([c.get("verify_wall_ms") for c in cyc]),
        "tok_s_from_periods": (round(1e3 * sum(c["emitted"] for c in cyc if c["period_ms"] is not None)
                                     / sum(period), 3) if period else None),
        "restore_wall_ms": med([c.get("restore_wall_ms") for c in cyc]),
        "rebuild_wall_ms": med([c.get("rebuild_wall_ms") for c in cyc]),
        # device-event phases (whole runs): median and mean per phase, ms
        "ev_phase_median_ms": {ph: med([c["ev_ms"].get(ph, 0.0) for c in cyc if "ev_ms" in c])
                               for ph in ("draft", "pre_verify", "verify", "accept", "restore", "rebuild")}
        if any("ev_ms" in c for c in cyc) else None,
        "ev_phase_mean_ms": {ph: round(st.mean([c["ev_ms"].get(ph, 0.0) for c in cyc if "ev_ms" in c]), 3)
                             for ph in ("draft", "pre_verify", "verify", "accept", "restore", "rebuild")}
        if any("ev_ms" in c for c in cyc) else None,
        "ev_span_mean_ms": (round(st.mean([c["ev_span_ms"] for c in cyc if "ev_span_ms" in c]), 3)
                            if any("ev_span_ms" in c for c in cyc) else None),
        "period_mean_ms": round(st.mean(period), 3) if period else None,
        "serial_ev_median_ms": med([s.get("ev_ms") for s in ser]),
        "serial_period_mean_ms": (round(st.mean([s["period_ms"] for s in ser if s["period_ms"] is not None]), 3)
                                  if any(s["period_ms"] is not None for s in ser) else None),
        "n_serial": len(ser),
        "serial_period_ms": med([s["period_ms"] for s in ser]),
        "serial_enqueue_ms": med([s["enqueue_ms"] for s in ser]),
        "serial_wall_ms": med([s["wall_ms"] for s in ser]),
    }


def run_arm(engine, name: str, mods, args, warm: int = 2) -> dict:
    """One arm: serial steps (dissected), then per path: whole cycles and
    dissected cycles. Returns the per-path summaries."""
    paths = {"greedy": SampleParams(temperature=0.0, max_tokens=args.tokens)}
    if not args.no_sampled:
        paths["sampled"] = SampleParams(temperature=0.7, seed=7, max_tokens=args.tokens)
    out = {"arm": name, "stubbed_linears": len(mods), "paths": {}}
    with Stubbed(mods):
        # the trunk's serial step: the engine's own loop, MTP depth 0 for
        # this request. Greedy (the step's cost does not depend on the
        # sampler; the chain's does, hence two paths below).
        set_arm(engine, 0)
        try:
            with Probe(engine, dissect=True) as p:
                generate(engine, args.prompt, paths["greedy"])
            serial = summarize(p, warm)
            with Probe(engine, dissect=False) as p:
                generate(engine, args.prompt, paths["greedy"])
            whole = summarize(p, warm)
            serial["serial_period_ms_whole"] = whole["serial_period_ms"]
            serial["serial_period_mean_ms_whole"] = whole["serial_period_mean_ms"]
            serial["serial_ev_median_ms"] = whole["serial_ev_median_ms"]
        finally:
            set_arm(engine, args.depth)
        out["serial"] = {k: v for k, v in serial.items() if k.startswith("serial") or k == "n_serial"}
        for path, params in paths.items():
            rec = {}
            for dissect in (False, True):
                with Probe(engine, dissect=dissect) as p:
                    res, _ = generate(engine, args.prompt, params)
                s = summarize(p, warm)
                s["completion_tokens"] = res.completion_tokens
                s["cycles_raw"] = p.cycles
                rec["dissected" if dissect else "whole"] = s
            out["paths"][path] = rec
    return out


def derive(arm: dict, depth: int) -> dict:
    """c and friends, per path, from the dissected medians."""
    t = arm["serial"]["serial_wall_ms"]
    d = {}
    for path, rec in arm["paths"].items():
        ds, wh = rec["dissected"], rec["whole"]
        chain = ds["chain_wall_ms"]
        k = ds["k_median"] or depth
        d[path] = {
            "trunk_step_ms": t,
            "chain_wall_ms": chain,
            "chain_enqueue_ms": wh["chain_enqueue_ms"],
            "verify_wall_ms": ds["verify_wall_ms"],
            "cycle_period_ms": wh["cycle_period_ms"],
            "tok_per_cycle": wh["tok_per_cycle"],
            "tok_s": wh["tok_s_from_periods"],
            "chain_over_trunk": round(chain / t, 4) if (chain and t) else None,
            "c": round(chain / k / t, 4) if (chain and t and k) else None,
            "verify_over_trunk": (round(ds["verify_wall_ms"] / t, 4)
                                  if (ds["verify_wall_ms"] and t) else None),
            "restore_wall_ms": ds["restore_wall_ms"],
            "rebuild_wall_ms": ds["rebuild_wall_ms"],
            "ev_phase_mean_ms": wh["ev_phase_mean_ms"],
            "ev_phase_median_ms": wh["ev_phase_median_ms"],
            "ev_span_mean_ms": wh["ev_span_mean_ms"],
            "cycle_period_mean_ms": wh["period_mean_ms"],
            "serial_period_mean_ms": arm["serial"].get("serial_period_mean_ms_whole"),
            # the cycle's cost in serial-token equivalents: mean period over
            # the serial loop's mean period
            "cycle_over_serial": (round(wh["period_mean_ms"] / arm["serial"]["serial_period_mean_ms_whole"], 4)
                                  if wh["period_mean_ms"] and arm["serial"].get("serial_period_mean_ms_whole")
                                  else None),
        }
    return d


# ---------------------------------------------------------------- build --


def build_toy():
    """tests/test_serving_mtp.py's toy, rebuilt here so the smoke needs no
    pytest: a 4-layer hybrid qwen3_5 with a real MTPHead, float32, CPU."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from drinkme.serving.engines import HFEngine

    # prefill on CPU needs the pure-torch chunk rule (the tests' caveat)
    fn = mod.torch_chunk_gated_delta_rule
    mod.torch_chunk_gated_delta_rule = getattr(fn, "__wrapped__", fn)
    cfg = Qwen3_5TextConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                         "partial_rotary_factor": 0.25},
        tie_word_embeddings=False, eos_token_id=None, pad_token_id=None)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).eval().to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over",
             "lazy", "dog", "user", "assistant", "system", ":"]
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in words:
        vocab[w] = len(vocab)
    while len(vocab) < cfg.vocab_size:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    head = mtp.MTPHead(cfg, model)
    torch.manual_seed(1)
    for name, p in list(head.named_parameters()):
        path, _, attr = name.rpartition(".")
        owner = head.get_submodule(path) if path else head
        owner._parameters[attr] = torch.nn.Parameter(
            torch.randn(p.shape, dtype=torch.float32) * 0.08, requires_grad=False)
    head.eval()
    mtp.install_deltanet_capture(model)
    return HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=512,
                    mtp_head=head)


def build_real(args):
    from drinkme.serve import build_engine

    engine = build_engine(args.model, args.revision, None if args.stock else args.pack,
                          stock=args.stock, ctx=args.ctx)
    if engine.mtp_head is None:
        raise SystemExit("no MTP head loaded — DRINKME_MTP_DEPTH=4 and a pack with a head sub-pack?")
    engine.template_kwargs = {"enable_thinking": False}
    return engine


def main() -> int:
    args = ARGS
    t0 = time.time()
    engine = build_toy() if args.toy else build_real(args)
    dev = torch.device(engine.device)
    print(f"[cycle_floor] engine on {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}); "
          f"head {'present' if engine.mtp_head else 'ABSENT'}; depth {args.depth}; "
          f"prompt {args.prompt!r}", flush=True)
    # the ruler times CYCLES; the adaptive bail (a product behaviour, not a
    # cycle cost) would only shorten the sample on the free-text prompt
    mtp.BAIL_FLOOR = 0.0
    raw = args.toy
    trunk_mods = linears(engine.model, raw)
    head_mods = [m for m in linears(engine.mtp_head, raw)] if engine.mtp_head else []
    # the head borrows embed/lm_head from the trunk: head_mods holds only its
    # OWN Linears (fc + its decoder layer), which is what stub_head means
    arms = {"real": [], "stub_head": head_mods, "stub_all": trunk_mods + head_mods}
    wanted = [a for a in args.arms.split(",") if a]
    receipt = {"model": engine.model_id, "arm": engine.arm,
               "pack": None if (args.toy or args.stock) else args.pack,
               "stock": args.stock, "prompt_name": args.prompt_name,
               "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
               "depth": args.depth, "tokens": args.tokens, "prompt": args.prompt,
               "toy": args.toy, "bail_floor": 0.0, "trunk_linears": len(trunk_mods),
               "head_linears": len(head_mods),
               "date": time.strftime("%Y-%m-%d %H:%M %Z"), "repeats": []}
    # warm the engine once (allocator, triton compile) outside every timing
    generate(engine, args.prompt, SampleParams(temperature=0.0, max_tokens=16))
    for r in range(args.repeats):
        rep = {"arms": {}}
        for name in wanted:
            arm = run_arm(engine, name, arms[name], args)
            arm["derived"] = derive(arm, args.depth)
            rep["arms"][name] = arm
            d = arm["derived"]
            ser = arm["serial"]
            print(f"[cycle_floor] repeat {r + 1} arm {name:9s} trunk step {ser['serial_wall_ms']} ms "
                  f"(period {ser['serial_period_ms_whole']})", flush=True)
            for path, v in d.items():
                print(f"[cycle_floor]   {path:7s} chain {v['chain_wall_ms']} ms (enqueue {v['chain_enqueue_ms']}) "
                      f"verify {v['verify_wall_ms']} ms  period {v['cycle_period_ms']} ms "
                      f"tok/cycle {v['tok_per_cycle']}  tok/s {v['tok_s']}  "
                      f"chain/trunk {v['chain_over_trunk']}  c {v['c']}", flush=True)
                print(f"[cycle_floor]   {path:7s} restore {v['restore_wall_ms']} ms rebuild {v['rebuild_wall_ms']} ms  "
                      f"events (mean ms) {v['ev_phase_mean_ms']} span {v['ev_span_mean_ms']}  "
                      f"cycle/serial {v['cycle_over_serial']}", flush=True)
        receipt["repeats"].append(rep)

    # medians across repeats, per arm/path — the table the report quotes
    table = {}
    for name in wanted:
        row = {"trunk_step_ms": med([rp["arms"][name]["serial"]["serial_wall_ms"] for rp in receipt["repeats"]]),
               "trunk_period_ms": med([rp["arms"][name]["serial"]["serial_period_ms_whole"] for rp in receipt["repeats"]])}
        for path in receipt["repeats"][0]["arms"][name]["derived"]:
            first = receipt["repeats"][0]["arms"][name]["derived"][path]
            row[path] = {key: (med([rp["arms"][name]["derived"][path][key] for rp in receipt["repeats"]])
                               if not isinstance(first[key], dict)
                               # a per-phase dict: the median of each phase across repeats
                               else {ph: med([rp["arms"][name]["derived"][path][key][ph]
                                              for rp in receipt["repeats"]]) for ph in first[key]})
                         for key in first}
        table[name] = row
    if "real" in table and "stub_all" in table:
        for path in table["real"]:
            if isinstance(table["real"][path], dict):
                fl, re_ = table["stub_all"][path]["cycle_period_ms"], table["real"][path]["cycle_period_ms"]
                table["real"][path]["host_floor_ms"] = fl
                table["real"][path]["host_fraction"] = round(fl / re_, 4) if (fl and re_) else None
                ch_all = table["stub_all"][path]["chain_wall_ms"]
                ch_head = table.get("stub_head", {}).get(path, {}).get("chain_wall_ms")
                ch_real = table["real"][path]["chain_wall_ms"]
                table["real"][path]["chain_dispatch_ms"] = ch_all
                if ch_head is not None and ch_real is not None:
                    table["real"][path]["chain_head_bytes_ms"] = round(ch_real - ch_head, 3)
                    table["real"][path]["chain_lm_head_ms"] = round(ch_head - (ch_all or 0), 3)
    receipt["table"] = table
    receipt["elapsed_s"] = round(time.time() - t0, 1)
    print("=== TABLE (medians across repeats; ms) ===")
    for name, row in table.items():
        print(f"{name}: trunk step {row['trunk_step_ms']} (period {row['trunk_period_ms']})")
        for path, v in row.items():
            if isinstance(v, dict):
                print(f"  {path:7s} chain {v['chain_wall_ms']} (enq {v['chain_enqueue_ms']})  "
                      f"verify {v['verify_wall_ms']}  period {v['cycle_period_ms']}  "
                      f"tok/cycle {v['tok_per_cycle']}  tok/s {v['tok_s']}  "
                      f"chain/trunk {v['chain_over_trunk']}  c {v['c']}"
                      + (f"  host floor {v['host_floor_ms']} ({v['host_fraction']})" if "host_floor_ms" in v else "")
                      + (f"  chain split: head {v.get('chain_head_bytes_ms')} / lm_head {v.get('chain_lm_head_ms')} / dispatch {v.get('chain_dispatch_ms')}"
                         if "chain_head_bytes_ms" in v else ""))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(receipt, f, indent=1)
        print(f"wrote {args.json}")
    print("VERDICT: measured" if table else "VERDICT: nothing measured")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # os._exit discipline: TheRock torch _exit(0)s on atexit
