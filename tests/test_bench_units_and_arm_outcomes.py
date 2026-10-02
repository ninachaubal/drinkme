"""Units, and the stock arm's outcome (a prediction is not an observation),
at the boundaries where each lives.

Units: raw carries BYTES and bytes/second, and bench._gb/_gb_s convert to
decimal at the record boundary, where the unit string is — never a
1024**3 division published under "GB" (one such unit would be 1.073741824
decimal GB). Pinned here: a 4*1024**3-byte probe over 1 s is 4.295 GB/s
(not 4.0); 1024**3 bytes of VRAM is 1.074 GB.

Outcome: a bare boolean would conflate "stock ran? no" — whether bench's
fit check had merely PREDICTED a non-fit or a load had actually FAILED.
The one `stock` object names the evidence category (`outcome`: measured |
skipped_predicted_nonfit | failed_load), a skip carries the budget it was
judged against (`budgetBytes`), a failure carries its message
(`error`), the twin's outcome stays in raw (it is bench's own control, not
a public field), and a failed stock load does not take the compressed
arm's measurement down with it. No GPU anywhere here.
"""

from __future__ import annotations

import pytest
import torch

from drinkme import arms, arms_mlx, bench, exitcodes, probe
from drinkme.detect import Hardware
from drinkme.fit import GB, GIB
from drinkme.publish import validate as V
from drinkme.suggest import Model, Suggestion
from hosts import pin_linux_host

# ------------------------------------------------------------------ units --



def test_probe_reports_bytes_per_second_and_the_boundary_reports_decimal():
    """Exactly 4*1024**3 bytes read once in 1 s."""
    bw = probe.bandwidth_from_timings(4 * GIB, reps=1, copy_seconds=2.0, read_seconds=1.0)
    assert bw["read_bytes_s"] == 4 * GIB == 4_294_967_296
    assert bw["copy_bytes_s"] == 4 * GIB  # 2x the bytes cross the bus, over 2 s
    assert bench._gb_s(bw["read_bytes_s"]) == 4.295
    assert bench._gb_s(bw["read_bytes_s"]) != 4.0


def test_vram_of_one_gib_is_1_074_gb(monkeypatch):
    monkeypatch.setattr(arms.torch.cuda, "memory_allocated", lambda: GIB)
    assert arms.vram_bytes() == GIB
    assert bench._gb(arms.vram_bytes()) == 1.074


def test_measure_bandwidth_runs_on_the_host_and_returns_bytes():
    """The plumbing (probe_bytes -> floats -> bytes/s), on the CPU with a
    small buffer; the number itself is a host number nobody publishes."""
    bw = probe.measure_bandwidth(probe_bytes=8 << 20, device="cpu")
    assert bw["probe_bytes"] == 8 << 20 and bw["device"] == "cpu"
    assert isinstance(bw["read_bytes_s"], int) and bw["read_bytes_s"] > 0
    assert isinstance(bw["copy_bytes_s"], int) and bw["copy_bytes_s"] > 0
    assert "read_gb_s" not in bw and "probe_size_gb" not in bw


def test_probe_size_is_bytes_a_quarter_of_free_memory(monkeypatch):
    monkeypatch.setattr(probe, "pick_device", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * GIB, 16 * GIB))
    assert probe.probe_size_for("cuda") == 2 * GIB
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (32 * GIB, 64 * GIB))
    assert probe.probe_size_for("cuda") == 4 * GIB  # capped at the 4 GiB request
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100 << 20, 16 * GIB))
    assert probe.probe_size_for("cuda") == GIB // 4  # the 256 MiB floor


def test_ceiling_is_the_same_unit_free_ratio():
    assert probe.ceiling_tok_s(222 * GB, 16 * GB) == 13.88
    assert probe.ceiling_tok_s(222 * GIB, 16 * GIB) == 13.88  # the conversion cancels


def _fake_mx(working_set=None, active=0):
    """The mlx.core surface arms_mlx's probe touches, stubbed: a buffer that
    only knows its length, Metal present iff `working_set` is given."""
    import types

    class _Buf:
        def __init__(self, n):
            self.n = n

        def __add__(self, _):
            return self

    calls = []
    fake = types.SimpleNamespace(
        bfloat16="bf16", ones=lambda shape, dtype: calls.append(("ones", shape[0], dtype)) or _Buf(shape[0]),
        eval=lambda *_: None, sum=lambda b: calls.append(("sum", b.n)) or b,
        array=lambda *a, **k: 0, synchronize=lambda: calls.append(("synchronize",)),
        clear_cache=lambda: calls.append(("clear_cache",)),
        get_active_memory=lambda: active,
        metal=types.SimpleNamespace(is_available=lambda: working_set is not None),
        device_info=lambda: {"max_recommended_working_set_size": working_set})
    return fake, calls


def _install_mx(monkeypatch, fake):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake))
    monkeypatch.setitem(sys.modules, "mlx.core", fake)


def test_mlx_bandwidth_shares_the_probes_arithmetic(monkeypatch):
    """arms_mlx.measure_bandwidth: bytes in, bytes/second out, through the
    same bandwidth_from_timings and the torch probe's repeat count — mx
    stubbed, the clock stubbed. The read sums the bf16 buffer itself (no
    fp32 cast), and the probe ends by handing MLX's cache back."""
    fake, calls = _fake_mx()
    _install_mx(monkeypatch, fake)
    monkeypatch.setattr(arms_mlx, "_device", lambda: "test-m")
    ticks = iter([0.0, 1.0, 1.0, 3.0])  # read: 1 s over the reps; copy: 2 s over the reps
    monkeypatch.setattr(arms_mlx.time, "perf_counter", lambda: next(ticks))
    bw = arms_mlx.measure_bandwidth(probe_bytes=2 * GIB)
    reps = arms_mlx.PROBE_REPS
    assert reps == 5  # probe.measure_bandwidth's
    assert bw["probe_bytes"] == 2 * GIB
    assert bw["read_bytes_s"] == reps * 2 * GIB  # reps of 2 GiB in 1 s
    assert bw["copy_bytes_s"] == reps * 2 * 2 * GIB // 2  # reps, 2x bytes, 2 s
    assert bench._gb_s(bw["read_bytes_s"]) == 10.737
    assert calls[0] == ("ones", GIB, "bf16")  # one bf16 buffer: 2 GiB is 1 Gi elements
    assert calls.count(("sum", GIB)) == reps + 1  # the warm-up, then the timed reps
    assert calls[-2:] == [("synchronize",), ("clear_cache",)]  # in flight work done, then the cache back


def test_mlx_probe_size_is_the_torch_probes_rule_on_the_working_set(monkeypatch):
    """probe_size_for_mlx: a quarter of (Metal's recommended working set -
    MLX's active bytes), whole MiB, at most the 2 GiB request, at least
    256 MiB; with no Metal, the request itself."""
    for ws, active, want in ((16 * GIB, 0, 2 * GIB),            # an M4 (24 GB): the request
                             (6 * GIB, 0, 3 * GIB // 2),          # a small Mac: a quarter
                             (6 * GIB + 123, 0, 3 * GIB // 2),    # whole MiB
                             (8 * GIB, 4 * GIB, GIB),             # what MLX already holds is not free
                             (GIB // 2, 0, GIB // 4),             # the 256 MiB floor
                             (None, 0, 2 * GIB)):                 # no Metal: the request
        fake, _ = _fake_mx(working_set=ws, active=active)
        _install_mx(monkeypatch, fake)
        assert arms_mlx.probe_size_for_mlx() == want, (ws, active)
        assert arms_mlx.probe_size_for_mlx() % (1 << 20) == 0
    assert probe.probe_size_from_free(16 * GIB, 4 * GIB) == 4 * GIB  # the torch probe's cuda rule, shared


# ------------------------------------------- the MLX arms' warm-up rep --


class _StubMlxVal:
    """Whatever mx.array/engine.model return on this stub: indexable
    (mx.array[...]) and .astype-able (mx.array(...).astype(...)), like a
    real mx.array, so timed_prefill/timed_ttft's real bodies run unmodified."""

    def __getitem__(self, key):
        return self

    def astype(self, dtype):
        return self


class _StubMlxEngine:
    """A stand-in for the MLX engine timed_prefill/timed_ttft call:
    engine.model(...), counted."""

    def __init__(self):
        self.calls = 0

    def model(self, *args, **kwargs):
        self.calls += 1
        return _StubMlxVal()


def _install_fake_mlx_runtime(monkeypatch):
    """mlx.core + mlx_lm.models.cache stubbed just enough for
    arms_mlx.timed_prefill/timed_ttft to run their real bodies on a box
    with no mlx installed (this test's box, on Linux)."""
    import sys
    import types

    fake_mx = types.SimpleNamespace(
        array=lambda *a, **k: _StubMlxVal(), eval=lambda *a, **k: None,
        argmax=lambda v: _StubMlxVal(), int32="int32", float32="float32")
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    cache_mod = types.SimpleNamespace(make_prompt_cache=lambda model: object())
    models_mod = types.SimpleNamespace(cache=cache_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(models=models_mod))
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    # engine_mlx (the shared last-row function's home) imports real mlx at top
    # level; the stub's last_row_logits is one call of engine.model, counted
    monkeypatch.setitem(sys.modules, "drinkme.serving.engine_mlx", types.SimpleNamespace(
        last_row_logits=lambda model, ids, cache=None: model(ids, cache=cache)[0]))


def test_mlx_timed_decode_runs_one_untimed_warmup_rep_first(monkeypatch):
    """timed_decode: WARMUP_REP on, reps=2 -> greedy() called 3x (the
    untimed warm-up, then the 2 timed reps) and 2 samples come back — the
    warm-up's own tok/s (4.0, from the first tick pair) never appears."""
    calls = []
    monkeypatch.setattr(arms_mlx, "greedy",
                        lambda eng, ids, n: calls.append(n) or list(ids) + [0] * n)
    monkeypatch.setattr(arms_mlx, "WARMUP_REP", True)
    ticks = iter([0.0, 2.0, 3.0, 7.0])  # the warm-up makes no clock calls at all
    monkeypatch.setattr(arms_mlx.time, "perf_counter", lambda: next(ticks))
    _out, samples = arms_mlx.timed_decode(object(), [1, 2, 3], n_new=4, reps=2)
    assert len(calls) == 3
    assert samples == [2.0, 1.0]


def test_mlx_timed_decode_runs_exactly_reps_with_warmup_off(monkeypatch):
    calls = []
    monkeypatch.setattr(arms_mlx, "greedy",
                        lambda eng, ids, n: calls.append(n) or list(ids) + [0] * n)
    monkeypatch.setattr(arms_mlx, "WARMUP_REP", False)
    ticks = iter([0.0, 2.0, 3.0, 7.0])
    monkeypatch.setattr(arms_mlx.time, "perf_counter", lambda: next(ticks))
    _out, samples = arms_mlx.timed_decode(object(), [1, 2, 3], n_new=4, reps=2)
    assert len(calls) == 2
    assert samples == [2.0, 1.0]


def test_mlx_timed_prefill_runs_one_untimed_warmup_rep_first(monkeypatch):
    """timed_prefill: WARMUP_REP on, reps=2 -> engine.model called 3x and 2
    samples come back. The discarded warm-up rep (i==0) hits its `continue`
    before its own second clock read (arms.timed_prefill's own structure),
    so it consumes exactly one tick; the two timed reps consume two each."""
    _install_fake_mlx_runtime(monkeypatch)
    monkeypatch.setattr(arms_mlx, "WARMUP_REP", True)
    ticks = iter([0.0, 1.0, 3.0, 4.0, 8.0])
    monkeypatch.setattr(arms_mlx.time, "perf_counter", lambda: next(ticks))
    eng = _StubMlxEngine()
    samples = arms_mlx.timed_prefill(eng, [1, 2, 3, 4], reps=2)
    assert eng.calls == 3
    assert samples == [2.0, 1.0]


def test_mlx_timed_prefill_runs_exactly_reps_with_warmup_off(monkeypatch):
    _install_fake_mlx_runtime(monkeypatch)
    monkeypatch.setattr(arms_mlx, "WARMUP_REP", False)
    ticks = iter([0.0, 2.0, 2.0, 6.0])
    monkeypatch.setattr(arms_mlx.time, "perf_counter", lambda: next(ticks))
    eng = _StubMlxEngine()
    samples = arms_mlx.timed_prefill(eng, [1, 2, 3, 4], reps=2)
    assert eng.calls == 2
    assert len(samples) == 2


def test_mlx_timed_ttft_runs_one_untimed_warmup_rep_first(monkeypatch):
    """timed_ttft: WARMUP_REP on, reps=2 -> engine.model called 3x and 2
    samples come back. Same one-tick discarded warm-up as timed_prefill."""
    _install_fake_mlx_runtime(monkeypatch)
    monkeypatch.setattr(arms_mlx, "WARMUP_REP", True)
    ticks = iter([0.0, 1.0, 1.5, 2.0, 2.2])
    monkeypatch.setattr(arms_mlx.time, "perf_counter", lambda: next(ticks))
    eng = _StubMlxEngine()
    samples = arms_mlx.timed_ttft(eng, [1, 2, 3, 4], reps=2)
    assert eng.calls == 3
    assert samples == [0.5, 0.2]


def test_mlx_timed_ttft_runs_exactly_reps_with_warmup_off(monkeypatch):
    _install_fake_mlx_runtime(monkeypatch)
    monkeypatch.setattr(arms_mlx, "WARMUP_REP", False)
    ticks = iter([0.0, 0.5, 1.0, 1.2])
    monkeypatch.setattr(arms_mlx.time, "perf_counter", lambda: next(ticks))
    eng = _StubMlxEngine()
    samples = arms_mlx.timed_ttft(eng, [1, 2, 3, 4], reps=2)
    assert eng.calls == 2
    assert len(samples) == 2


def test_record_metrics_are_decimal_from_raw_bytes(monkeypatch):
    from tests.test_bench_record_shape import _build_record, _fake_raw

    raw = _fake_raw()
    raw["vram_bf16_bytes"] = GIB
    raw["vram_compressed_bytes"] = 3 * GIB
    raw["bandwidth"]["read_bytes_s"] = 4 * GIB
    raw["bandwidth"]["copy_bytes_s"] = 2 * GIB
    monkeypatch.setattr("tests.test_bench_record_shape._fake_raw", lambda: raw)
    r = _build_record(monkeypatch)
    by = {m["name"]: m for m in r["metrics"]}
    assert (by["stock_weights_gb"]["value"], by["stock_weights_gb"]["unit"]) == ("1.074", "GB")
    assert (by["compressed_weights_gb"]["value"], by["compressed_weights_gb"]["unit"]) == ("3.221", "GB")
    assert (by["read_gb_s"]["value"], by["read_gb_s"]["unit"]) == ("4.295", "GB/s")
    assert (by["copy_gb_s"]["value"], by["copy_gb_s"]["unit"]) == ("2.147", "GB/s")
    # raw keeps the bytes, untouched
    assert r["raw"]["vram_bf16_bytes"] == GIB
    assert r["raw"]["bandwidth"]["read_bytes_s"] == 4 * GIB


def test_lexicon_says_the_units_are_decimal():
    lex = V.load_lexicon()
    desc = lex["defs"]["main"]["record"]["properties"]["metrics"]["description"]
    assert "decimal" in desc and "10^9 bytes" in desc


def test_no_binary_division_feeds_a_published_gb_figure():
    """The three modules that compute bytes carry no `/ 1024**3` on a value
    that reaches a metric: bytes stay bytes until bench._gb/_gb_s."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "drinkme"
    for name in ("arms.py", "arms_mlx.py", "probe.py"):
        src = (root / name).read_text()
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert not re.search(r"/\s*1024\s*\*\*\s*3", code), name


# ----------------------------------------------------------- arm outcomes --


def test_stock_load_failure_names_only_load_failures():
    assert arms.stock_load_failure(exitcodes.LoadRefused("drinkme: refusing to load: x — ..."))
    assert arms.stock_load_failure(RuntimeError("HIP out of memory. Tried to allocate 2 GiB"))
    assert arms.stock_load_failure(MemoryError())
    oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    assert arms.stock_load_failure(oom("CUDA out of memory"))
    # not load failures: a bug, a missing dependency, an unrelated exit
    assert not arms.stock_load_failure(RuntimeError("--stock needs the `accelerate` package"))
    assert not arms.stock_load_failure(KeyError("lm_head.weight"))
    assert not arms.stock_load_failure(SystemExit("drinkme: refusing to write the record: ..."))
    # decided by type, not text: another CantRunHere (even one whose message says
    # "refusing to load") and a bare SystemExit are not the stock arm failing to load
    assert not arms.stock_load_failure(exitcodes.CantRunHere("drinkme: refusing to load: x — ..."))
    assert not arms.stock_load_failure(exitcodes.Refused("drinkme: refusing to write the record: ..."))
    assert not arms.stock_load_failure(SystemExit("drinkme: refusing to load: x — ..."))
    assert exitcodes.LoadRefused.pinned == exitcodes.CANT_RUN_HERE


def _hw(**kw):
    base = dict(device_class="test-device", memory_gb=16.0, memory_kind="vram", budget_gb=16.0,
                cpu_info="test-cpu", gpu_info="test-gpu", evidence=["fixture"],
                device_source="table")
    base.update(kw)
    # the detectors always carry the exact bytes beside the rounded GB
    base.setdefault("memory_bytes", round(base["memory_gb"] * GB))
    base.setdefault("budget_bytes", round(base["budget_gb"] * GB))
    return Hardware(**base)


def _point(monkeypatch, raw: dict, hw: Hardware, stock_ok: bool) -> dict:
    """One torch point through the real bench.run, on a Linux host (pinned:
    a Mac resolves the mlx runtime and would run the mlx arms instead)."""
    pin_linux_host(monkeypatch)
    m = Model(name="ToyModel", hf_repo="toy/model", revision="deadbeef")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: Suggestion(
        ratio=m, ratio_extra=[], fit=None, fit_knife_edge=False,
        fit_honest_negative=False, notes=[]))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: stock_ok)
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: raw)
    return bench.run(no_gemma=True)


def test_measured_stock_arm(monkeypatch):
    from tests.test_bench_stock_fit_and_device_refusal import _fake_raw

    r = _point(monkeypatch, _fake_raw(stock=True), _hw(), stock_ok=True)
    assert r["stock"] == {"outcome": "measured"}
    assert "twinArm" not in r and r["raw"]["twin_outcome"] == "measured"
    assert not any(k in r for k in ("stockFits", "stockArm", "stockArmBudgetGB", "stockArmError"))
    assert V.validate(bench.lexicon_safe(r)) == []


def test_skipped_stock_arm_carries_the_decimal_budget_it_was_judged_against(monkeypatch):
    from tests.test_bench_stock_fit_and_device_refusal import _fake_raw

    # discrete: the ceiling is the VRAM budget; detect's budget_gb is decimal
    # GB, so this is a straight pass-through.
    r = _point(monkeypatch, _fake_raw(stock=False), _hw(budget_gb=16.0), stock_ok=False)
    assert r["stock"] == {"outcome": "skipped_predicted_nonfit", "budgetBytes": 16_000_000_000}
    assert type(r["stock"]["budgetBytes"]) is int
    assert not any(m["name"].startswith("stock_") or m["name"] == "stock_weights_gb"
                   for m in r["metrics"])
    assert V.validate(bench.lexicon_safe(r)) == []
    # unified: the ceiling is physical memory
    r = _point(monkeypatch, _fake_raw(stock=False),
               _hw(memory_kind="unified", memory_gb=124.0, budget_gb=100.0), stock_ok=False)
    assert r["stock"]["budgetBytes"] == 124_000_000_000


def test_failed_stock_load_is_recorded_and_the_compressed_arm_still_lands(monkeypatch):
    from tests.test_bench_stock_fit_and_device_refusal import _fake_raw

    raw = _fake_raw(stock=False)
    raw["stock_outcome"] = "failed_load"
    raw["stock_error"] = "OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB"
    r = _point(monkeypatch, raw, _hw(), stock_ok=True)
    assert r["stock"]["outcome"] == "failed_load" and set(r["stock"]) == {"outcome", "error"}
    assert r["stock"]["error"].startswith("OutOfMemoryError: HIP out of memory")
    names = {m["name"] for m in r["metrics"]}
    assert {"compressed_decode_tok_s", "twin_decode_tok_s", "twin_weights_gb"} <= names
    assert not any(n.startswith("stock_") or n == "stock_weights_gb" for n in names)
    assert V.validate(bench.lexicon_safe(r)) == []


def test_twin_skip_is_named_as_a_prediction(monkeypatch):
    from tests.test_bench_stock_fit_and_device_refusal import _fake_raw

    raw = _fake_raw(stock=False)
    raw["twin_outcome"] = "skipped_predicted_nonfit"
    raw["twin_decode_tok_s"] = None
    raw["twin_decode_samples"] = []
    r = _point(monkeypatch, raw, _hw(), stock_ok=False)
    assert "twinArm" not in r and r["raw"]["twin_outcome"] == "skipped_predicted_nonfit"
    assert not any(m["name"].startswith("twin_") for m in r["metrics"])
    assert V.validate(bench.lexicon_safe(r)) == []


def test_a_measured_claim_without_a_measurement_is_refused(monkeypatch):
    from tests.test_bench_stock_fit_and_device_refusal import _fake_raw

    raw = _fake_raw(stock=False)
    raw["stock_outcome"] = "measured"  # says measured, carries no stock metric
    with pytest.raises(ValueError, match="stock_outcome 'measured'"):
        _point(monkeypatch, raw, _hw(), stock_ok=True)


def test_lexicon_defines_the_stock_object_precisely(monkeypatch):
    lex = V.load_lexicon()
    props = lex["defs"]["main"]["record"]["properties"]
    assert props["stock"] == {"type": "ref", "ref": "#stock"}
    assert "stock" in lex["defs"]["main"]["record"]["required"]
    stock = lex["defs"]["stock"]
    assert stock["required"] == ["outcome"] and set(stock["properties"]) == {"outcome", "budgetBytes", "error"}
    assert stock["properties"]["outcome"]["enum"] == ["measured", "skipped_predicted_nonfit", "failed_load"]
    assert stock["properties"]["error"]["maxLength"] == 2048
    for gone in ("stockFits", "stockArm", "stockArmBudgetGB", "stockArmError", "twinArm", "stockPath"):
        assert gone not in props, gone
    assert "baselinePrecision" not in lex["defs"]["model"]["properties"]
    # the enum is closed, and the object is required: the validator says so by path
    from tests.test_bench_stock_fit_and_device_refusal import _fake_raw

    r = bench.lexicon_safe(_point(monkeypatch, _fake_raw(stock=True), _hw(), stock_ok=True))
    assert V.validate(r) == []
    r["stock"]["outcome"] = "unsupported_on_backend"
    assert any(e.startswith("stock.outcome:") for e in V.validate(r))
    del r["stock"]
    assert any(e.startswith("stock:") for e in V.validate(r))


def test_run_arms_stock_pass_survives_a_load_failure(monkeypatch):
    """run_arms itself, with the device-touching pieces stubbed: the stock
    load raises an OOM-shaped RuntimeError, and the report that comes back
    says failed_load with the message, carries no stock metric, and still
    carries the twin and compressed passes. Driven through the
    from_pretrained loader (`stock_loader="from_pretrained"`, load_cpu's
    device_map path); the streaming loader's twin of this test is in
    test_stock_stream_loader.py."""
    import types

    class _Ids:
        shape = (1, 4)

        def cuda(self):
            return self

    class _Tok:
        def __call__(self, text, **kw):
            if kw.get("return_tensors"):
                return types.SimpleNamespace(input_ids=_Ids())
            return types.SimpleNamespace(input_ids=[1, 2, 3])

    monkeypatch.setitem(__import__("sys").modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok())))
    # the one source every arm loads from, resolved without the hub
    # (tests/test_bench_one_source.py has the resolution itself)
    monkeypatch.setattr(arms, "resolve_source",
                        lambda repo, rev, pack_dir: (__import__("tempfile").gettempdir(), "deadbeef"))
    monkeypatch.setattr(arms.torch, "tensor", lambda *a, **k: _Ids())
    monkeypatch.setattr(arms.torch.cuda, "get_device_name", lambda i: "test-gpu")
    monkeypatch.setattr(arms, "measure_bandwidth", lambda: {
        "probe_bytes": GIB, "read_bytes_s": 200 * GB, "copy_bytes_s": 180 * GB, "device": "cuda"})
    monkeypatch.setattr(arms, "free_all", lambda: None)
    monkeypatch.setattr(arms, "refuse_if_stock_wont_fit", lambda *a, **k: None)
    monkeypatch.setattr(arms, "live_available_bytes", lambda kind: 100 * GB)

    class _Model:
        def cuda(self):
            return self

    def load_cpu(model_id, revision=None, config=None, device_map=None, snap=None):
        if device_map == "cuda":
            raise RuntimeError("HIP out of memory. Tried to allocate 15.26 GiB")
        return _Model()

    monkeypatch.setattr(arms, "load_cpu", load_cpu)
    stat = {"name": "l", "shape": [1024, 1024], "numel": 1024 * 1024, "bits": 12 * 1024 * 1024,
            "bpw": 12.0, "format_version": 1, "codec": "radix", "profile": "sip", "dtype": None, "verified": True}
    # passes 2/3 stream (tests/test_stream_compressed_arms.py): stubbed loaders
    monkeypatch.setattr(arms, "load_twin_streaming",
                        lambda model_id, revision=None, device="cuda", snap=None, plan=None: (_Model(), []))
    monkeypatch.setattr(arms, "load_compressed_streaming",
                        lambda model_id, revision=None, device="cuda", snap=None: (_Model(), [dict(stat)]))
    monkeypatch.setattr(arms, "vram_bytes", lambda: 500_000_000)
    monkeypatch.setattr(arms, "timed_decode", lambda model, ids, **k: (None, [10.0, 10.0, 10.0]))
    monkeypatch.setattr(arms, "timed_prefill", lambda model, ids, **k: [500.0, 500.0, 500.0])
    monkeypatch.setattr(arms, "timed_ttft", lambda model, ids, **k: [0.05, 0.05, 0.05])

    r = arms.run_arms("toy/model", None, "hi", run_stock=True, stock_memory_kind="vram",
                      stock_bf16_bytes=15.26 * GB, run_twin=True, stock_loader="from_pretrained")
    assert r["stock_outcome"] == "failed_load" and r["stock_loader"] == "from_pretrained"
    assert r["stock_error"] == "RuntimeError: HIP out of memory. Tried to allocate 15.26 GiB"
    assert not any(k.startswith("stock_") for k in r
                   if k not in ("stock_outcome", "stock_error", "stock_loader"))
    assert "vram_bf16_bytes" not in r
    assert r["twin_outcome"] == "measured" and r["twin_decode_tok_s"] == 10.0
    assert r["compressed_decode_tok_s"] == 10.0 and r["vram_compressed_bytes"] == 500_000_000
    assert r["compression_profile"] == "sip"

    # and a NON-load failure still crashes the run
    def load_cpu_bug(model_id, revision=None, config=None, device_map=None, snap=None):
        if device_map == "cuda":
            raise RuntimeError("--stock needs the `accelerate` package")
        return _Model()

    monkeypatch.setattr(arms, "load_cpu", load_cpu_bug)
    with pytest.raises(RuntimeError, match="accelerate"):
        arms.run_arms("toy/model", None, "hi", run_stock=True, stock_memory_kind="vram",
                      stock_bf16_bytes=15.26 * GB, run_twin=True, stock_loader="from_pretrained")
