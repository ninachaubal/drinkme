"""The MTP draft chain performs no device->host sync, and the cycle reads the device exactly once outside the
engine's own decision functions.

WHY THIS IS A TEST AND NOT A BENCH NOTE. Every `int(tensor)`, `.item()`,
`.tolist()`, `bool(tensor)` on a CUDA tensor drains the GPU queue before the
host can enqueue anything else. Inside the k-step draft chain that
serialized host dispatch with GPU execution once per drafted token — the
drafter's cost coefficient c was dispatch + sync rather than the head's
bytes (bench/mtp_cycle_floor.py measured 0.058 before the change). The toy
runs on CPU, where none of these calls sync, so the test counts the CALLS:
the conversion methods are monkeypatched on torch.Tensor for the duration of
one generation, and the count is attributed to (a) the draft chain
(MTPHead.draft / draft_sampled), (b) the engine's decision functions
(sampling.sample_next / sample_probs, speculative.speculative_sample — the
accept rule and the per-row picks are host decisions by design), and (c)
the rest of Speculator.cycle. The contract:

    (a) == 0 for every cycle          (was k: one int(argmax) per draft)
    (c) == 1 for every drafting cycle (the drafts read back, once, after
                                       the verify forward is enqueued)
    (c) == 0 for the k == 0 cycle     (nothing was drafted)

Fails on the tree before the change with (a) == k and (c) == k.
"""

import os

import pytest
import torch

from drinkme.serving import engines, mtp, sampling, speculative
from drinkme.serving.engine import SampleParams
from test_serving_mtp import (  # noqa: F401
    PROMPTS, _engine, _msgs, _run, _torch_chunk_on_cpu, mtp_depth, toy,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

CONVERSIONS = ("__int__", "__float__", "__bool__", "__index__", "item", "tolist", "cpu", "numpy")


class _Count:
    """Attributes every host read to the innermost scope that made it."""

    def __init__(self):
        self.scope = "cycle"   # "chain" | "decision" | "cycle" | "outside"
        self.cycles: list[dict] = []
        self.cur = None

    def hit(self, name):
        if self.cur is None:
            return
        self.cur[self.scope] = self.cur.get(self.scope, 0) + 1
        self.cur.setdefault("names", {}).setdefault(f"{self.scope}:{name}", 0)
        self.cur["names"][f"{self.scope}:{name}"] += 1


@pytest.fixture
def counted(monkeypatch, toy):
    """Count Tensor host conversions inside one generation, by scope."""
    c = _Count()
    _, _, head, _ = toy

    for name in CONVERSIONS:
        real = getattr(torch.Tensor, name)

        def make(real, name):
            def w(self, *a, **kw):
                c.hit(name)
                return real(self, *a, **kw)
            return w

        monkeypatch.setattr(torch.Tensor, name, make(real, name))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **kw: c.hit("cuda.synchronize"))

    def scoped(fn, scope):
        def w(*a, **kw):
            prev, c.scope = c.scope, scope
            try:
                return fn(*a, **kw)
            finally:
                c.scope = prev
        return w

    monkeypatch.setattr(head, "draft", scoped(head.draft, "chain"))
    monkeypatch.setattr(head, "draft_sampled", scoped(head.draft_sampled, "chain"))
    # the engine's closures reach these through their module globals
    monkeypatch.setattr(engines, "sample_next", scoped(sampling.sample_next, "decision"))
    monkeypatch.setattr(engines, "sample_probs", scoped(sampling.sample_probs, "decision"))
    monkeypatch.setattr(mtp, "speculative_sample",
                        scoped(speculative.speculative_sample, "decision"))
    real_cycle = mtp.Speculator.cycle

    def cycle(self, *a, **kw):
        c.cur = {"k": None}
        d0 = self.drafted
        try:
            return real_cycle(self, *a, **kw)
        finally:
            c.cur["k"] = self.drafted - d0
            c.cycles.append(c.cur)
            c.cur = None

    monkeypatch.setattr(mtp.Speculator, "cycle", cycle)
    return c


def _check(c: _Count, path: str):
    assert c.cycles, "no cycle ran"
    for i, cyc in enumerate(c.cycles):
        names = cyc.get("names", {})
        assert cyc.get("chain", 0) == 0, \
            f"{path} cycle {i}: {cyc.get('chain')} host reads inside the draft chain: {names}"
        want = 1 if cyc["k"] else 0
        assert cyc.get("cycle", 0) == want, \
            f"{path} cycle {i} (k={cyc['k']}): {cyc.get('cycle')} host reads in the cycle " \
            f"outside the decision functions, want {want}: {names}"


def test_greedy_draft_chain_performs_no_host_sync(toy, counted):
    eng = _engine(toy, with_mtp=True)
    with mtp_depth(4):
        res, _ = _run(eng, _msgs(PROMPTS[2]), SampleParams(temperature=0.0, max_tokens=12))
    assert res.completion_tokens == 12
    _check(counted, "greedy")
    # the chain drafted: k == 4 on every cycle but the budget tail
    assert [cyc["k"] for cyc in counted.cycles][:2] == [4, 4]


def test_sampled_draft_chain_performs_no_host_sync(toy, counted):
    eng = _engine(toy, with_mtp=True)
    with mtp_depth(4):
        res, _ = _run(eng, _msgs(PROMPTS[3]),
                      SampleParams(temperature=0.9, seed=4, max_tokens=12))
    assert res.completion_tokens == 12
    _check(counted, "sampled")
    assert counted.cycles[0]["k"] == 4


def test_the_drafts_are_read_back_after_the_verify_is_enqueued(toy, counted, monkeypatch):
    """The one read happens AFTER forward_with_hidden (the verify) returns:
    the chain and the verify are both in the queue before the host blocks.
    (An ordering guard for the tree after the change — it also held before
    it, when the chain's reads were the chain's own.)"""
    order = []
    real = mtp.forward_with_hidden

    def fwd(*a, **kw):
        order.append(("verify", counted.cur.get("cycle", 0) if counted.cur else None))
        return real(*a, **kw)

    monkeypatch.setattr(mtp, "forward_with_hidden", fwd)
    eng = _engine(toy, with_mtp=True)
    with mtp_depth(4):
        _run(eng, _msgs(PROMPTS[4]), SampleParams(temperature=0.0, max_tokens=8))
    # prefill's call happens outside any cycle (None); every verify inside a
    # cycle saw ZERO cycle-scope reads before it
    inside = [n for tag, n in order if n is not None]
    assert inside and all(n == 0 for n in inside), order


def test_the_cycle_ruler_runs_on_the_toy(tmp_path):
    """bench/mtp_cycle_floor.py --toy: the stub path and the probe on CPU."""
    import json
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = tmp_path / "floor.json"
    env = {**os.environ, "PYTHONPATH": os.path.join(root, "src")}
    r = subprocess.run([sys.executable, os.path.join(root, "bench", "mtp_cycle_floor.py"),
                        "--toy", "--repeats", "1", "--tokens", "32", "--arms", "real,stub_all",
                        "--json", str(out)], cwd=root, env=env, capture_output=True, text=True,
                       timeout=600)
    assert "VERDICT: measured" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads(out.read_text())
    t = rec["table"]
    assert set(t) == {"real", "stub_all"} and rec["toy"] is True
    for arm in t.values():
        assert arm["trunk_step_ms"] > 0
        for path in ("greedy", "sampled"):
            v = arm[path]
            assert v["chain_wall_ms"] > 0 and v["verify_wall_ms"] > 0 and v["c"] > 0
            assert v["cycle_period_ms"] > 0
    raw = rec["repeats"][0]["arms"]
    assert raw["stub_all"]["stubbed_linears"] > 0 and raw["real"]["stubbed_linears"] == 0
    # depth 4 -> M = 5 verify rows, on both arms and both paths
    for arm in raw.values():
        for path in ("greedy", "sampled"):
            assert arm["paths"][path]["dissected"]["verify_rows"] == 5
    assert "host_floor_ms" in t["real"]["greedy"]


def test_the_cycle_ruler_times_restore_and_rebuild_and_takes_a_named_prompt(tmp_path):
    """The cycle ruler's phases on the toy: --prompt-name takes bench/ngram_gpu_ab.py's
    prompt, the dissected run times _restore_rows and the head's rebuild, and
    the rebuild excludes the draft chain's own head.run calls: with every
    Linear stubbed every draft is accepted (m = k, a rebuild each cycle); with
    the real toy nothing is (m = 0, no rebuild at all)."""
    import json
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = tmp_path / "floor.json"
    env = {**os.environ, "PYTHONPATH": os.path.join(root, "src")}
    r = subprocess.run([sys.executable, os.path.join(root, "bench", "mtp_cycle_floor.py"),
                        "--toy", "--repeats", "1", "--tokens", "24", "--arms", "real,stub_all",
                        "--no-sampled", "--prompt-name", "chat", "--json", str(out)],
                       cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert "VERDICT: measured" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads(out.read_text())
    assert rec["prompt"].startswith("Write a detailed, multi-paragraph explanation")
    assert rec["prompt_name"] == "chat" and rec["stock"] is False
    stub, real = rec["table"]["stub_all"]["greedy"], rec["table"]["real"]["greedy"]
    assert stub["restore_wall_ms"] > 0 and stub["rebuild_wall_ms"] > 0
    assert real["rebuild_wall_ms"] is None  # the toy's head never agrees: m == 0
    for v in (stub, real):
        assert v["cycle_over_serial"] > 0
        assert v["ev_phase_mean_ms"] is None  # device events are GPU-only


def test_verify_rows_linear_groups_every_linear_by_shape():
    """bench/verify_rows_linear.py's grouping and byte count, on CPU."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "verify_rows_linear", os.path.join(root, "bench", "verify_rows_linear.py"))
    vrl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vrl)
    m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(8, 16),
                            torch.nn.Linear(16, 4, bias=False))
    g = vrl.groups(m)
    assert {k: len(v) for k, v in g.items()} == {("Linear", 16, 8): 2, ("Linear", 4, 16): 1}
    assert vrl.resident_bytes(m[3]) == 4 * 16 * 4  # fp32 weight, bias not counted
