"""The record's shape:

  A. No fidelity metric — no greedy_token_match_pct, first_divergence,
     text_compressed or logit_delta_bf16_ulps in `raw`. A logits comparison
     taken by re-running each arm's forward on the STOCK arm's decoded
     tokens (a prefill-shaped, M-large call) lands, for the compressed arm,
     on swap.CompressedLinear's "dense" route — decode the verified
     bit-exact weight once and call stock's own F.linear — so it would be
     bitwise stock's own answer by construction, never the codec's GEMV
     kernel. See arms.py's module docstring.
  B. The bitwise round trip is a write-time GATE, not a field: bench
     refuses to write a record at all when any compressed tensor fails its
     round trip (or nothing was eligible).
  C. prefill_tok_s + ttft_s per arm, on a fixed 512-token prompt.
  D. The twin arm is decode-only by construction — the torch twin is the
     bench's M=1 twin kernel over uncompressed bf16 weights, with no dense
     route at M>1, so a prefill/TTFT timing on it would be an artifact (8B
     twin prefill 19.94 tok/s vs stock 1059, TTFT 28s), not a measurement.
     No twin_prefill_tok_s/twin_ttft_s is ever assigned. On torch the
     twin's decode and footprint are published (twin_decode_tok_s,
     twin_weights_gb), bench's diagnostic beside the stock arm the site
     divides by. On mlx the twin is
     the engine's reference/correctness path, so it publishes no metric
     there and its decode stays in raw.

These are pure-Python checks (no CUDA, no MLX) against the real arms.py /
arms_mlx.py / bench.py / lexicon / check_points.py code, using AST
inspection and a stubbed run_arms where a GPU would otherwise be needed.
"""

import ast
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

from drinkme import arms, arms_mlx, bench
from drinkme.detect import Hardware
from drinkme.fit import GIB
from drinkme.publish import validate as V
from drinkme.suggest import Model, Suggestion
from hosts import pin_linux_host

ROOT = pathlib.Path(__file__).resolve().parents[1]

# the commit the stubbed arms "loaded": a 40-hex sha, as checkpoint.resolved_revision
# names a hub snapshot (the record's model.revision is the hub commit or nothing)
TOY_SHA = "deadbeef" * 5
# what kernel_route.route_kernels reports for a dense toy tree: no DeltaNet
# layers, the narrow GEMV allowed, nothing adopted — every measured torch arm
# carries one as raw.<arm>_routing
TOY_ROUTING = {"deltanet_kernel": "none", "deltanet_conv": "none", "narrow_gemv": True, "narrow_count": 0,
               "stock_gemv": "linear", "raw_gemv": "stock"}

DROPPED_RAW_FIELDS = (
    "greedy_token_match_pct", "first_divergence", "text_compressed",
    "logit_delta_bf16_ulps", "all_tensors_bitwise_roundtrip",
)
NEW_METRIC_NAMES = (
    "stock_prefill_tok_s", "compressed_prefill_tok_s",
    "stock_ttft_s", "compressed_ttft_s",
)
# the twin has no dense route at M>1 (torch: the codec's own M=1
# kernel; mlx: the reference/correctness path) — timing it at prefill's
# fixed M=512 measured an artifact, not a number (8B twin prefill 19.94
# tok/s vs stock 1059, TTFT 28s), so these are never assigned or published.
DROPPED_TWIN_PREFILL_TTFT_FIELDS = (
    "twin_prefill_tok_s", "twin_prefill_samples",
    "twin_ttft_s", "twin_ttft_samples",
)


# ------------------------------------------------- A: the dropped fields --



def _assigned_dict_keys(module_path: str, func_name: str) -> set[str]:
    """Every string literal used either as `<name>["key"] = ...` or as a
    `{"key": ...}` dict-literal key inside `func_name` — a static stand-in
    for "what keys does this function's report dict end up with", without
    needing the CUDA/MLX runtime the function itself requires."""
    tree = ast.parse((ROOT / module_path).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == func_name)
    keys = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                keys.add(sl.value)
        elif isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    return keys


@pytest.mark.parametrize("module_path, func_name", [
    ("src/drinkme/arms.py", "run_arms"),
    ("src/drinkme/arms_mlx.py", "run_arms"),
])
def test_dropped_fidelity_fields_are_never_assigned(module_path, func_name):
    keys = _assigned_dict_keys(module_path, func_name)
    for field in DROPPED_RAW_FIELDS:
        assert field not in keys, f"{module_path}:{func_name} still assigns {field!r}"


@pytest.mark.parametrize("module_path, func_name", [
    ("src/drinkme/arms.py", "run_arms"),
    ("src/drinkme/arms_mlx.py", "run_arms"),
])
def test_prefill_and_ttft_fields_are_assigned(module_path, func_name):
    keys = _assigned_dict_keys(module_path, func_name)
    for arm in ("stock", "compressed"):
        for suffix in ("prefill_tok_s", "prefill_samples", "ttft_s", "ttft_samples"):
            field = f"{arm}_{suffix}"
            assert field in keys, f"{module_path}:{func_name} never assigns {field!r}"
    assert "prefill_prompt_len" in keys


@pytest.mark.parametrize("module_path, func_name", [
    ("src/drinkme/arms.py", "run_arms"),
    ("src/drinkme/arms_mlx.py", "run_arms"),
])
def test_twin_prefill_and_ttft_fields_are_never_assigned(module_path, func_name):
    """the twin arm has no dense route at M>1 — no dead computation is
    left timing it at prefill's fixed M=512 shape, and no field for it is
    ever assigned (contrast test_prefill_and_ttft_fields_are_assigned,
    which the stock/compressed arms still satisfy)."""
    keys = _assigned_dict_keys(module_path, func_name)
    for field in DROPPED_TWIN_PREFILL_TTFT_FIELDS:
        assert field not in keys, f"{module_path}:{func_name} still assigns {field!r}"


def test_prefill_prompt_is_deterministic_and_exactly_the_requested_length():
    """build_prefill_ids: same tokenizer, same text in -> same ids out, and
    always exactly `length` ids (repeat-then-truncate), which is what makes
    prefill_tok_s comparable across tokenizers with different piece counts
    for the same passage."""
    class _StubTok:
        def __call__(self, text, add_special_tokens=False):
            class _Enc:
                input_ids = list(range(len(text.split())))
            return _Enc()

    tok = _StubTok()
    ids_a = arms.build_prefill_ids(tok, length=37)
    ids_b = arms.build_prefill_ids(tok, length=37)
    assert ids_a == ids_b
    assert len(ids_a) == 37

    ids_mlx_a = arms_mlx.build_prefill_ids(
        type("T", (), {"encode": staticmethod(lambda text: list(range(len(text.split()))))})(),
        length=23,
    )
    assert len(ids_mlx_a) == 23


# --------------------------------------------------------- B: the gate --


def test_torch_gate_passes_when_every_tensor_verified():
    assert arms.refuse_unless_verified("m", [{"verified": True}, {"verified": True}]) is None


def test_torch_gate_refuses_when_a_tensor_is_unverified():
    with pytest.raises(SystemExit, match="refusing to write the record"):
        arms.refuse_unless_verified("toy/model", [{"verified": True}, {"verified": False}])


def test_torch_gate_refuses_when_nothing_was_eligible():
    with pytest.raises(SystemExit, match="refusing to write the record"):
        arms.refuse_unless_verified("toy/model", [])


def test_the_pack_arms_stat_says_pack_verified_not_verified_and_the_gate_names_the_evidence():
    """A tensor read off a pack dir was
    stat'd `verified: True` on the strength of the pack's hashes — a trust decision,
    not the fresh source comparison make_compressed's `verified` names. The
    field is now `pack_verified`; the gate takes either evidence and stat_evidence
    says which; a stat with neither is refused."""
    st = arms.pack_stat("l", {"R": 4, "C": 8, "bpw": 12.0, "format_version": 1, "codec": "radix",
                              "profile": "sip"})
    assert st["pack_verified"] is True and "verified" not in st
    assert arms.stat_evidence(st) == "pack"
    assert arms.stat_evidence({"verified": True}) == "roundtrip"
    assert arms.stat_evidence({"codec": "radix"}) is None
    assert arms.refuse_unless_verified("m", [st, {"verified": True}]) is None
    with pytest.raises(SystemExit, match="refusing to write the record"):
        arms.refuse_unless_verified("toy/model", [st, {"codec": "radix"}])


def test_mlx_gate_passes_when_something_swapped():
    assert arms_mlx.refuse_unless_verified("m", 5) is None


def test_mlx_gate_refuses_when_zero_swapped():
    with pytest.raises(SystemExit, match="refusing to write the record"):
        arms_mlx.refuse_unless_verified("toy/model", 0)


# ------------------------------------------- the full record, stubbed arms --


def _fake_hw():
    return Hardware(
        device_class="test-device", memory_gb=16.0, memory_kind="vram", budget_gb=16.0,
        memory_bytes=16_000_000_000, budget_bytes=16_000_000_000,
        cpu_info="test-cpu", gpu_info="test-gpu", evidence=["fixture"],
    )


def _fake_suggestion():
    m = Model(name="ToyModel", hf_repo="toy/model", revision=TOY_SHA)
    return Suggestion(ratio=m, ratio_extra=[], fit=None, fit_knife_edge=False,
                      fit_honest_negative=False, notes=[])


def _fake_raw():
    """Shaped exactly as the real arms.run_arms returns: no
    dropped fidelity fields, prefill+ttft on the stock/compressed arms only
    (never the twin), verified_tensors present, no
    all_tensors_bitwise_roundtrip (the gate already passed by the time
    run_arms returns at all)."""
    return {
        "model": "toy/model", "revision": TOY_SHA, "gpu": "test-gpu",
        "resolved_revision": TOY_SHA,  # the commit every arm loaded (arms resolve one source first)
        "prefill_prompt_len": 512,
        "bandwidth": {"probe_bytes": 4 * GIB, "read_bytes_s": 200_000_000_000,
                      "copy_bytes_s": 180_000_000_000, "device": "cuda",
                      "total_param_bytes_bf16": 1_000_000_000, "ceiling_tok_s_bf16": 200.0},
        "vram_bf16_bytes": 1_000_000_000, "vram_compressed_bytes": 500_000_000,
        "stock_decode_tok_s": 10.0, "stock_decode_samples": [9.9, 10.0, 10.1],
        "twin_decode_tok_s": 12.0, "twin_decode_samples": [11.9, 12.0, 12.1],
        "vram_twin_bytes": 1_000_000_000,
        "compressed_decode_tok_s": 15.0, "compressed_decode_samples": [14.9, 15.0, 15.1],
        "stock_prefill_tok_s": 500.0, "stock_prefill_samples": [490.0, 500.0, 510.0],
        "compressed_prefill_tok_s": 460.0, "compressed_prefill_samples": [450.0, 460.0, 470.0],
        "stock_ttft_s": 0.05, "stock_ttft_samples": [0.048, 0.05, 0.052],
        "compressed_ttft_s": 0.07, "compressed_ttft_samples": [0.068, 0.07, 0.072],
        "swapped_linears": 7, "verified_tensors": 7,
        "stock_outcome": "measured", "twin_outcome": "measured",
        "compression_profile": "sip", "mean_bpw": 12.03, "weighted_bpw": 12.364,
        "stock_routing": dict(TOY_ROUTING), "twin_routing": dict(TOY_ROUTING),
        "compressed_routing": dict(TOY_ROUTING),
        # arms.decode_read_bytes per arm: what one decode step reads, the bound's denominator
        "stock_bytes_per_token": _per_token(0, 900_000_000), "twin_bytes_per_token": _per_token(900_000_000, 0),
        "compressed_bytes_per_token": _per_token(400_000_000, 0),
    }


def _per_token(packed: int, raw: int, row: int = 2048) -> dict:
    """arms.decode_read_bytes' shape."""
    return {"packed_bytes": packed, "raw_linear_bytes": raw, "embedding_row_bytes": row, "counts": {},
            "total_bytes": packed + raw + row}


def _build_record(monkeypatch, no_gemma=True) -> dict:
    """A torch record through the real bench.run, on a Linux host (pinned:
    on a Mac bench.run resolves the mlx runtime and runs the mlx arms)."""
    pin_linux_host(monkeypatch)
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _fake_suggestion())
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: _fake_raw())
    return bench.run(no_gemma=no_gemma)


def test_bench_run_drops_roundtripbitexact_and_keeps_the_gate_implicit(monkeypatch):
    r = _build_record(monkeypatch)
    assert "roundtripBitExact" not in r
    for field in DROPPED_RAW_FIELDS:
        assert field not in r["raw"], field


def test_bench_run_writes_prefill_and_ttft_metrics_with_right_units(monkeypatch):
    r = _build_record(monkeypatch)
    by_name = {m["name"]: m for m in r["metrics"]}
    for name in ("stock_prefill_tok_s", "compressed_prefill_tok_s"):
        assert by_name[name]["unit"] == "tok/s"
        assert len(by_name[name]["samples"]) == 3
    for name in ("stock_ttft_s", "compressed_ttft_s"):
        assert by_name[name]["unit"] == "s"
        assert len(by_name[name]["samples"]) == 3
    assert r["raw"]["prefill_prompt_len"] == 512
    # every metric name is unique (build_points/check_points key off it)
    assert len(by_name) == len(r["metrics"])


def test_the_twin_publishes_its_decode_and_footprint_beside_stock(monkeypatch):
    """on torch the twin's decode, with every sample, and its weight
    footprint are metrics (data beside the stock arm the site divides by);
    its outcome stays in raw and there is no twinArm at the top level."""
    r = _build_record(monkeypatch)
    by_name = {m["name"]: m for m in r["metrics"]}
    assert by_name["twin_decode_tok_s"] == {"name": "twin_decode_tok_s", "value": "12.0", "unit": "tok/s",
                                            "samples": ["11.9", "12.0", "12.1"]}
    assert by_name["twin_weights_gb"] == {"name": "twin_weights_gb", "value": "1.0", "unit": "GB"}
    assert "stock_decode_tok_s" in by_name
    assert not any(n in by_name for n in DROPPED_TWIN_PREFILL_TTFT_FIELDS)
    assert "twinArm" not in r and r["raw"]["twin_outcome"] == "measured"
    assert [a for a, _ in bench.bound_fractions(r["metrics"])] == ["stock", "twin", "compressed"]
    # each arm's decode-read bytes beside its footprint, in decimal GB
    assert by_name["twin_decode_read_gb"] == {"name": "twin_decode_read_gb", "value": "0.9", "unit": "GB"}


def test_a_measured_twin_without_its_footprint_is_refused(monkeypatch):
    raw = _fake_raw()
    del raw["vram_twin_bytes"]
    pin_linux_host(monkeypatch)
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _fake_suggestion())
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: raw)
    with pytest.raises(ValueError, match="twin_outcome 'measured'"):
        bench.run(no_gemma=True)


def test_pack_bytes_are_published_iff_the_arm_loaded_a_pack_from_disk(monkeypatch):
    """compression.packedBytes / residentBytes are the loaded pack's own
    meta.json numbers (arms.run_arms copies them into raw.pack under
    --pack-dir; arms_mlx always) — integers, never re-derived; the
    in-memory torch re-pack has no meta.json and publishes neither."""
    from drinkme.codec.pack import pack_record_fields

    r = _build_record(monkeypatch)
    assert "packedBytes" not in r["compression"] and "residentBytes" not in r["compression"]

    raw = _fake_raw()
    raw["pack"] = pack_record_fields({"packedBytes": 6_130_000_000, "residentBytes": 7_500_000_000,
                                      "profile": "sip", "formatVersion": 1})
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _fake_suggestion())
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: raw)
    r = bench.run(no_gemma=True)
    assert r["compression"]["packedBytes"] == 6_130_000_000
    assert r["compression"]["residentBytes"] == 7_500_000_000
    assert type(r["compression"]["packedBytes"]) is int
    assert V.validate(bench.lexicon_safe(r)) == []
    r["compression"]["packedBytes"] = "6130000000"
    assert any(e.startswith("compression.packedBytes:") for e in V.validate(bench.lexicon_safe(r)))


def test_raw_is_declared_and_carries_device_source_and_no_top_level_platform(monkeypatch):
    """`raw` is a declared `unknown` field: bench's working detail, with
    device_source (how detect() named the device — bench's own refusal
    input; arms' snake_case like every other raw key) moved there from the
    environment block; the platform lives in environment.platform only,
    never at the top level."""
    r = _build_record(monkeypatch)
    lex = V.load_lexicon()
    assert lex["defs"]["main"]["record"]["properties"]["raw"]["type"] == "unknown"
    assert "platform" not in r and r["environment"]["platform"] in ("cuda", "rocm", "metal")
    assert "deviceSource" not in r["environment"] and "device_source" not in r["environment"]
    assert "device_source" in r["raw"] and "deviceSource" not in r["raw"]  # None here (the fixture names no source)
    assert "device_source" not in _fake_hw().as_record_env()
    # every environment key is one the lexicon declares; the detector's
    # working notes are in raw.detector
    declared = lex["defs"]["environment"]["properties"]
    assert set(r["environment"]) <= set(declared), set(r["environment"]) - set(declared)
    assert r["raw"]["detector"]["version"] == 2
    assert "device_source" in V.load_lexicon()["defs"]["main"]["record"]["properties"]["raw"]["description"]
    assert V.validate(bench.lexicon_safe(r)) == []
    r["raw"] = "not an object"
    assert any(e.startswith("raw:") for e in V.validate(r))


# --------------------------------------------------------- D: check_points --


def _build_points_module():
    """site/data/build_points.py as a module (it is a script, not a
    package): the real wrap_measurement, so the pointers the test writes
    are the ones the site's own bundle would carry."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_points", ROOT / "site" / "data" / "build_points.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_points_on(tmp_path, record: dict) -> None:
    """Wrap `record` exactly as build_points.py wraps a new-shape record
    file (the real wrap_measurement), then run the ACTUAL
    check_points.py — copied so its ROOT resolves under tmp_path, never
    rewritten — against it as a subprocess. Then run build_points.py itself
    on the same record file and check_points again, so both scripts are
    proven on the new shape. No existing measurements/ record is touched."""
    site_dir = tmp_path / "site" / "data"
    site_dir.mkdir(parents=True, exist_ok=True)
    for name in ("check_points.py", "build_points.py"):
        shutil.copy(ROOT / "site" / "data" / name, site_dir / name)
    (tmp_path / "records").mkdir(exist_ok=True)
    source = tmp_path / "records" / "toy.json"
    source.write_text(json.dumps(bench.lexicon_safe(record)))

    bp = _build_points_module()
    safe = bench.lexicon_safe(record)
    wrapped = bp.wrap_measurement(safe, "records/toy.json")
    # the record verbatim (minus raw) plus `site`: the stock object's fields
    # pointed at themselves, allocatableGB the memory in decimal GB pointed
    # at the bytes, a stock sentence for a stock arm that did not run
    assert {k: v for k, v in wrapped.items() if k != "site"} == {k: v for k, v in safe.items() if k != "raw"}
    for field in safe["stock"]:
        assert wrapped["site"]["at"][f"stock.{field}"] == f"/stock/{field}"
    assert wrapped["site"]["at"]["allocatableGB"] == "/environment/memoryBytes"
    assert wrapped["site"]["allocatableGB"] == safe["environment"]["memoryBytes"] / 1e9
    if safe["stock"]["outcome"] == "measured":
        assert "stockError" not in wrapped["site"] and "stockError" not in wrapped["site"]["at"]
    else:
        assert wrapped["site"]["stockError"] == (safe["stock"].get("error") or safe["raw"]["stock_error"])
    for name in ("stockFits", "stockArm", "twinArm", "stockPath"):
        assert name not in wrapped["site"]["at"] and name not in wrapped
    (site_dir / "points.json").write_text(json.dumps({
        "lexicon": "lexicons/wtf.petrichor.drinkme.measurement.json",
        "records": [wrapped]}))
    proc = subprocess.run([sys.executable, "site/data/check_points.py"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "all clear" in proc.stdout

    # and build_points.py itself, driven the way the operator drives it
    # after a re-run (record files on the command line), then checked
    proc = subprocess.run([sys.executable, "site/data/build_points.py", str(source)],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    built = json.loads((site_dir / "points.json").read_text())
    assert built["records"] == [wrapped]
    proc = subprocess.run([sys.executable, "site/data/check_points.py"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "all clear" in proc.stdout


def test_new_shape_record_passes_check_points(tmp_path, monkeypatch):
    """A record built through the real bench.run() path (stubbed
    run_arms only, everything else is the real code), wrapped by the real
    build_points.wrap_measurement, passes the real check_points.py — and
    build_points.py itself builds the same bundle from the record file.
    A synthetic new-shape point only."""
    _check_points_on(tmp_path, _build_record(monkeypatch))


def test_build_points_reads_measurements_by_default_and_the_empty_set_checks_clean(tmp_path, monkeypatch):
    """`build_points.py` with no arguments builds every record in
    measurements/ (where `drinkme bench` writes them) — the committed
    points.json is that build over the checkout's measurements/, empty
    until the menu is re-run — and check_points.py passes on the empty
    set: no records is not an error."""
    site_dir = tmp_path / "site" / "data"
    site_dir.mkdir(parents=True)
    for name in ("check_points.py", "build_points.py"):
        shutil.copy(ROOT / "site" / "data" / name, site_dir / name)

    def run(script):
        proc = subprocess.run([sys.executable, f"site/data/{script}"],
                              cwd=tmp_path, capture_output=True, text=True)
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        return proc.stdout

    # no measurements/ at all (gitignored; a fresh checkout) -> the empty set
    assert "0 records" in run("build_points.py")
    built = json.loads((site_dir / "points.json").read_text())
    assert built["records"] == [] and built["lexicon"] == "lexicons/wtf.petrichor.drinkme.measurement.json"
    assert "all clear" in run("check_points.py")

    # one record written by the real writer into measurements/ -> one point
    (tmp_path / "measurements").mkdir()
    bench.write_record(_build_record(monkeypatch), str(tmp_path / "measurements" / "toy.json"))
    out = run("build_points.py")
    assert "1 records" in out and "stock=measured" in out
    built = json.loads((site_dir / "points.json").read_text())
    assert len(built["records"]) == 1
    assert built["records"][0]["site"]["source"] == "measurements/toy.json"
    assert built["records"][0]["site"]["at"]["stock.outcome"] == "/stock/outcome"
    assert "all clear" in run("check_points.py")


# ------------------------------------------------ E: the mlx runtime --


def _fake_raw_mlx():
    """Shaped as the real arms_mlx.run_arms returns: the twin's
    decode is still timed (the twin path's own correctness exercise — the
    pack decoded once at load; it must run to check the pack decodes at
    all) but, like the torch twin,
    it has no prefill/ttft. ratio_min_of_n/drift_band_pct/compressed_path
    are this lane's own fields (docs/metal.md's min-of-n-with-drift-band
    discipline)."""
    return {
        "model": "toy/model", "revision": TOY_SHA, "gpu": "test-gpu",
        "resolved_revision": TOY_SHA,  # the commit every arm loaded (arms resolve one source first)
        "prompt": bench.PROMPT, "runtime": "mlx", "compressed_path": "reference",
        "prefill_prompt_len": 512,
        "bandwidth": {"probe_bytes": 2 * GIB, "read_bytes_s": 20_000_000_000,
                      "copy_bytes_s": 18_000_000_000, "device": "test-gpu",
                      "total_param_bytes_bf16": 1_000_000_000, "ceiling_tok_s_bf16": 20.0},
        "vram_bf16_bytes": 1_000_000_000, "vram_compressed_bytes": 500_000_000,
        "stock_decode_tok_s": 10.0, "stock_decode_samples": [9.9, 10.0, 10.1],
        "stock_decode_stats": {"median": 10.0, "min": 9.9, "max": 10.1, "drift_pct": 2.0},
        "stock_outcome": "measured", "twin_outcome": "measured",
        "stock_prefill_tok_s": 500.0, "stock_prefill_samples": [490.0, 500.0, 510.0],
        "stock_ttft_s": 0.05, "stock_ttft_samples": [0.048, 0.05, 0.052],
        "twin_decode_tok_s": 0.3, "twin_decode_samples": [0.29, 0.3, 0.31],
        "twin_decode_stats": {"median": 0.3, "min": 0.29, "max": 0.31, "drift_pct": 6.9},
        "compression_profile": "sip", "mean_bpw": 12.03, "weighted_bpw": 12.364,
        "compressed_decode_tok_s": 15.0, "compressed_decode_samples": [14.9, 15.0, 15.1],
        "compressed_decode_stats": {"median": 15.0, "min": 14.9, "max": 15.1, "drift_pct": 1.3},
        "compressed_prefill_tok_s": 460.0, "compressed_prefill_samples": [450.0, 460.0, 470.0],
        "compressed_ttft_s": 0.07, "compressed_ttft_samples": [0.068, 0.07, 0.072],
        "swapped_linears": 7, "verified_tensors": 7,
        "ratio_min_of_n": round(14.9 / 9.9, 3), "drift_band_pct": 6.9,
        # arms_mlx.decode_read_bytes per arm, the twin's included (it stays in raw)
        "stock_bytes_per_token": _per_token(0, 900_000_000), "twin_bytes_per_token": _per_token(900_000_000, 0),
        "compressed_bytes_per_token": _per_token(400_000_000, 0),
    }


def _build_record_mlx(monkeypatch) -> dict:
    """bench.run(runtime='mlx') through the real orchestration code, with
    only detect/suggest/arms_mlx.run_arms stubbed — same discipline as
    _build_record, plus the mlx.core stub _engine_env needs (see
    test_engine_env_on_the_mlx_lane_never_imports_torch: this lane must
    answer from mlx alone, no torch import)."""
    import sys
    import types

    from drinkme import arms_mlx

    fake_mx = types.SimpleNamespace(__version__="0.32.2")
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _fake_suggestion())
    monkeypatch.setattr(arms_mlx, "run_arms", lambda *a, **k: _fake_raw_mlx())
    return bench.run(no_gemma=True, runtime="mlx")


def test_mlx_records_carry_no_twin_metric(monkeypatch):
    """on mlx, 'twin' names the engine's reference/correctness path, a
    different animal from the torch twin — none of its numbers (decode
    included) are a comparable kernel-quality figure, so no twin_* metric
    is published for this backend at all."""
    r = _build_record_mlx(monkeypatch)
    names = {m["name"] for m in r["metrics"]}
    assert not any(n.startswith("twin_") for n in names), names


def test_an_mlx_record_carries_the_stock_and_compressed_decode_read(monkeypatch):
    """the mlx arms record what one decode step reads (arms_mlx.
    decode_read_bytes), so a metal record carries the canonical metrics and
    stock's and compressed's <arm>_decode_read_gb, the denominators of their
    bandwidth bounds; the twin's read stays in raw with its decode."""
    r = _build_record_mlx(monkeypatch)
    names = [m["name"] for m in r["metrics"]]
    assert sorted(names) == sorted(CANONICAL_METRIC_NAMES + ("stock_decode_read_gb", "compressed_decode_read_gb"))
    by_name = {m["name"]: m for m in r["metrics"]}
    assert by_name["compressed_decode_read_gb"] == {"name": "compressed_decode_read_gb", "value": "0.4", "unit": "GB"}
    assert r["raw"]["twin_bytes_per_token"]["total_bytes"] == 900_002_048
    assert [arm for arm, _ in bench.bound_fractions(r["metrics"])] == ["stock", "compressed"]


def test_new_shape_mlx_record_passes_check_points(tmp_path, monkeypatch):
    """check_points.py never hardcodes a twin metric name (it walks
    whatever `metrics` a record carries and requires only that each has a
    pointer), so an mlx record that is simply missing every twin_* metric
    needs no special case there — this proves it, the same way
    test_new_shape_record_passes_check_points proves it for the torch
    shape."""
    _check_points_on(tmp_path, _build_record_mlx(monkeypatch))


# ------------------------------------------------------- the lexicon file --


def test_lexicon_no_longer_defines_roundtripbitexact():
    lex = V.load_lexicon()
    props = lex["defs"]["main"]["record"]["properties"]
    assert "roundtripBitExact" not in props


def test_lexicon_is_valid_json_and_documents_the_new_metrics():
    lex = json.loads(
        (ROOT / "lexicons" / "wtf.petrichor.drinkme.measurement.json").read_text())
    assert lex["lexicon"] == 1
    assert lex["id"] == "wtf.petrichor.drinkme.measurement"
    metrics_desc = lex["defs"]["main"]["record"]["properties"]["metrics"]["description"]
    for name in NEW_METRIC_NAMES:
        assert name in metrics_desc, name


CANONICAL_METRIC_NAMES = (
    "stock_decode_tok_s", "compressed_decode_tok_s", "stock_prefill_tok_s",
    "compressed_prefill_tok_s", "stock_ttft_s", "compressed_ttft_s", "read_gb_s",
    "copy_gb_s", "stock_weights_gb", "compressed_weights_gb",
)
# The twin's metrics. `metrics` is an open name/value list, so a record
# carrying them validates against the shipped lexicon; the description's
# canonical list is the ten above, and it names these two as the twin's.
TWIN_METRIC_NAMES = ("twin_decode_tok_s", "twin_weights_gb")
# Each arm's decode-read bytes (arms.decode_read_bytes; on mlx arms_mlx's,
# and no twin metric), the denominator of its bandwidth bound; the
# description names them too.
DECODE_READ_METRIC_NAMES = ("stock_decode_read_gb", "twin_decode_read_gb", "compressed_decode_read_gb")


def test_lexicon_names_the_metrics_bench_writes(monkeypatch):
    """the metrics description names the ten canonical metrics and the
    decode-read ones, and a full torch record carries those and the twin's
    two, no more."""
    lex = json.loads(
        (ROOT / "lexicons" / "wtf.petrichor.drinkme.measurement.json").read_text())
    metrics_desc = lex["defs"]["main"]["record"]["properties"]["metrics"]["description"]
    for name in CANONICAL_METRIC_NAMES + DECODE_READ_METRIC_NAMES:
        assert name in metrics_desc, name
    assert "twinArm" not in lex["defs"]["main"]["record"]["properties"]
    written = [m["name"] for m in _build_record(monkeypatch)["metrics"]]
    assert sorted(written) == sorted(CANONICAL_METRIC_NAMES + TWIN_METRIC_NAMES + DECODE_READ_METRIC_NAMES)


def test_a_new_shape_record_validates_against_the_shipped_lexicon(monkeypatch):
    r = _build_record(monkeypatch)
    assert V.validate(bench.lexicon_safe(r)) == []


def test_a_new_shape_mlx_record_validates_against_the_shipped_lexicon(monkeypatch):
    r = _build_record_mlx(monkeypatch)
    assert V.validate(bench.lexicon_safe(r)) == []


def test_engine_env_on_the_mlx_lane_never_imports_torch(monkeypatch):
    """A metal environment may have no torch (one made before the lane carried
    it, docs/metal.md); `bench --runtime mlx`
    would die in _engine_env on a real M4 after all three arms had run. The mlx branch must answer from mlx alone."""
    import builtins
    import sys
    import types
    from drinkme import bench

    fake_mx = types.SimpleNamespace(__version__="0.32.2")
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    env = bench._engine_env("mlx")
    assert env["platform"] == "metal"
    assert "mlx 0.32.2" in env["engine"]


def test_the_decode_prompt_text_is_not_in_the_record():
    """With the token outputs gone the prompt string
    is a leftover of the transcript era — the record keeps the prompt LENGTHS
    (prefill_prompt_len) and nothing of the text."""
    import pathlib
    for f in ("src/drinkme/arms.py", "src/drinkme/arms_mlx.py"):
        src = pathlib.Path(f).read_text()
        assert '"prompt": prompt' not in src, f


# ------------------------------------------------------- the host's load --

def test_the_host_load_rides_in_raw_and_the_summary(monkeypatch, capsys):
    """raw.host_load (the 1-minute load before and after each arm's timed
    passes, and the CPU count) is kept as run_arms wrote it, and the summary
    prints it; a quiet host gets no warning."""
    raw = _fake_raw()
    raw["host_load"] = {"cpu_count": 32, "load1": {"stock": [1.4, 1.6], "compressed": [1.6, 1.8],
                                                    "twin": [1.8, 1.9]}}
    monkeypatch.setattr("test_bench_record_shape._fake_raw", lambda: raw)
    r = _build_record(monkeypatch)
    assert r["raw"]["host_load"] == raw["host_load"]
    out = capsys.readouterr().out
    assert "host load (1-minute average) while timed, 32 CPUs: stock 1.4->1.6 · compressed 1.6->1.8 · " \
           "twin 1.8->1.9" in out
    assert "warning" not in out
    assert "-> 1.500x over stock, 1.250x over the twin" in out  # compressed / stock first


def test_a_busy_host_is_warned_about_never_refused(monkeypatch, capsys):
    raw = _fake_raw()
    raw["host_load"] = {"cpu_count": 32, "load1": {"stock": [2.0, 2.1], "compressed": [5.6, 8.2],
                                                    "twin": [None, None]}}
    monkeypatch.setattr("test_bench_record_shape._fake_raw", lambda: raw)
    r = _build_record(monkeypatch)
    assert r["metrics"]  # the record is written
    out = capsys.readouterr().out
    warning = [line for line in out.splitlines() if "warning" in line]
    assert len(warning) == 1 and "compressed was timed" in warning[0]
    assert "8.2, over 6.4 = 0.2 x 32 CPUs" in warning[0]
    assert "twin ?->?" in out
