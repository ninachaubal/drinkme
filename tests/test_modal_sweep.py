"""bench/modal_sweep.py's local side: the record-name rule it replicates
(bench.measurement_path's `_2`, `_3` — an existing record is never
overwritten) and the log line collapse. The module is imported with a stub
`modal` in its seat: the SDK is a uv tool here, not a project dependency."""

import importlib.util
import json
import os
import sys
import types

import pytest

from drinkme import bench

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "..", "bench", "modal_sweep.py")
DRIVE = os.path.join(HERE, "..", "bench", "modal_sweep_drive.py")


def _fake_modal():
    class _Chain:
        def __getattr__(self, name):
            return lambda *a, **k: self

    class _App:
        def __init__(self, name="", *a, **k):
            self.name = name

        def function(self, *a, **k):
            return lambda f: f

        def local_entrypoint(self, *a, **k):
            return lambda f: f

    fake = types.ModuleType("modal")
    fake.App = _App
    fake.Image = _Chain()
    fake.is_local = lambda: False  # the container's side: no git call at import
    return fake


def _load(path, name, fake, extra_saved=()):
    saved = {k: sys.modules.get(k) for k in ("modal", *extra_saved)}
    sys.modules["modal"] = fake
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


@pytest.fixture(scope="module")
def sweep():
    return _load(SWEEP, "modal_sweep_under_test", _fake_modal())


def test_free_record_path_matches_benchs_own_rule(sweep, tmp_path):
    record = {"model": {"name": "Qwen3-8B"}, "environment": {"deviceClass": "NVIDIA L4"},
              "createdAt": "2026-09-15T12:00:00+00:00"}
    root = str(tmp_path)
    out_dir = os.path.join(root, bench.MEASUREMENTS_DIR)
    os.makedirs(out_dir)
    # the container's name is always the n=1 form (its measurements/ started empty)
    basename = os.path.basename(bench.measurement_path(record, root))
    assert basename == "qwen3-8b_nvidia-l4_2026-09-15.json"
    for _ in range(3):
        # what bench would pick next == what the local side picks next, every time
        expect = bench.measurement_path(record, root)
        got = sweep.free_record_path(out_dir, basename)
        assert got == expect
        with open(got, "w") as f:
            json.dump(record, f)
    assert sorted(os.listdir(out_dir)) == [
        "qwen3-8b_nvidia-l4_2026-09-15.json",
        "qwen3-8b_nvidia-l4_2026-09-15_2.json",
        "qwen3-8b_nvidia-l4_2026-09-15_3.json",
    ]


def test_free_record_path_never_returns_an_existing_file(sweep, tmp_path):
    (tmp_path / "a_b_2026-09-15.json").write_text("{}")
    (tmp_path / "a_b_2026-09-15_2.json").write_text("{}")
    got = sweep.free_record_path(str(tmp_path), "a_b_2026-09-15.json")
    assert os.path.basename(got) == "a_b_2026-09-15_3.json"
    assert not os.path.exists(got)


def test_collapse_cr_keeps_the_final_state(sweep):
    assert sweep.collapse_cr("Fetching 0%\rFetching 50%\rFetching 100%\n") == "Fetching 100%\n"
    assert sweep.collapse_cr("plain line\n") == "plain line\n"
    assert sweep.collapse_cr("no newline") == "no newline"


def test_shapes_respect_the_timeout_ceiling(sweep):
    assert set(sweep.SHAPES) == {"L4", "A100-80GB", "H100", "L40S"}
    assert set(sweep.CARD_WITNESS) == set(sweep.SHAPES)
    for card, shape in sweep.SHAPES.items():
        assert shape["timeout"] <= 3600, card
    assert sweep.SHAPES["L4"] == {"memory": 32 * 1024, "cpu": 8, "timeout": 2700}
    assert sweep.SHAPES["H100"] == sweep.SHAPES["A100-80GB"] == sweep.SHAPES["L40S"] == {"memory": 96 * 1024, "cpu": 16, "timeout": 3600}


def test_the_image_mounts_nothing_git_does_not_track(sweep, tmp_path):
    """_untracked's globs cover every untracked path, ignored or not, and no
    tracked one: private working files beside the checkout never ride to
    the image, whatever they are named."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("no git on this machine")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "kept.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text("ignored/\n")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "log.txt").write_text("")
    (repo / "private").mkdir()
    (repo / "private" / "draft.md").write_text("")
    (repo / "loose.txt").write_text("")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "src/kept.py", ".gitignore"], check=True)
    globs = set(sweep._untracked(str(repo)))
    assert {"ignored", "ignored/**", "private", "private/**", "loose.txt"} <= globs
    assert not any(g.startswith(("src", ".gitignore")) for g in globs)
    with pytest.raises(subprocess.CalledProcessError):  # not a checkout: refused, never uploaded whole
        sweep._untracked(str(tmp_path / "nowhere"))


def test_write_cell_keeps_the_record_bytes_and_names_a_failure(sweep, tmp_path, capsys):
    """The local side's receipts, shared by the entrypoint and the spawn /
    collect driver: bench's bytes verbatim under bench's name, the log and
    the meta beside them, and a cell with no record written as FAILED with
    a nonzero return."""
    out_dir, log_dir = tmp_path / "records", tmp_path / "logs"
    out_dir.mkdir()
    log_dir.mkdir()
    record = {"model": {"name": "Qwen3-8B"}, "environment": {"deviceClass": "NVIDIA L4"}, "metrics": []}
    text = json.dumps(record, indent=2) + "\n"
    report = {"record_text": text, "record_basename": "qwen3-8b_nvidia-l4_2026-09-26.json",
              "log": "the container's log\n", "steps": {"drinkme bench": {"seconds": 1.0, "exit": 0}},
              "step_order": ["drinkme bench"]}
    assert sweep.write_cell(report, "Qwen3-8B", "L4", str(out_dir), str(log_dir), "2026-09-26T000000Z") == 0
    assert (out_dir / "qwen3-8b_nvidia-l4_2026-09-26.json").read_text() == text
    assert (log_dir / "qwen3-8b_nvidia-l4_2026-09-26.log").read_text() == "the container's log\n"
    meta = json.loads((log_dir / "qwen3-8b_nvidia-l4_2026-09-26.meta.json").read_text())
    assert "record_text" not in meta and "log" not in meta
    assert meta["record_path"] == str(out_dir / "qwen3-8b_nvidia-l4_2026-09-26.json")

    failed = {"error": "drinkme bench wrote no record", "log": "tail\n", "steps": {}, "step_order": []}
    assert sweep.write_cell(failed, "Qwen3-8B", "L4", str(out_dir), str(log_dir), "2026-09-26T000000Z") == 1
    assert (log_dir / "FAILED_qwen3-8b_l4_sip_2026-09-26T000000Z.log").read_text() == "tail\n"
    assert sorted(os.listdir(out_dir)) == ["qwen3-8b_nvidia-l4_2026-09-26.json"]


def test_write_cell_keeps_a_gulp_record_under_benchs_gulp_name(sweep, tmp_path):
    """A gulp cell's record keeps bench's own `_gulp` name and bytes, a
    second one takes `_gulp_2`, a same-day sip record keeps the bare name,
    and each record's log and meta take its stem, so the two profiles of one
    model and card never share a file."""
    out_dir, log_dir = tmp_path / "records", tmp_path / "logs"
    out_dir.mkdir()
    log_dir.mkdir()

    def cell(basename, profile, body):
        text = json.dumps({"compression": {"profile": profile}, "n": body}) + "\n"
        report = {"record_text": text, "record_basename": basename, "log": f"{body}\n",
                  "steps": {}, "step_order": []}
        assert sweep.write_cell(report, "Qwen3-8B", "L40S", str(out_dir), str(log_dir),
                                "2026-09-27T000000Z", profile) == 0
        return text

    gulp1 = cell("qwen3-8b_nvidia-l40s_2026-09-27_gulp.json", "gulp", 1)
    sip = cell("qwen3-8b_nvidia-l40s_2026-09-27.json", "sip", 2)
    gulp2 = cell("qwen3-8b_nvidia-l40s_2026-09-27_gulp.json", "gulp", 3)
    assert (out_dir / "qwen3-8b_nvidia-l40s_2026-09-27_gulp.json").read_text() == gulp1
    assert (out_dir / "qwen3-8b_nvidia-l40s_2026-09-27.json").read_text() == sip
    assert (out_dir / "qwen3-8b_nvidia-l40s_2026-09-27_gulp_2.json").read_text() == gulp2
    assert (log_dir / "qwen3-8b_nvidia-l40s_2026-09-27_gulp_2.log").read_text() == "3\n"
    assert (log_dir / "qwen3-8b_nvidia-l40s_2026-09-27.meta.json").exists()


def test_a_failed_cells_names_carry_its_profile(sweep, tmp_path):
    for profile in ("sip", "gulp"):
        failed = {"error": "no record", "log": f"{profile}\n", "steps": {}, "step_order": []}
        assert sweep.write_cell(failed, "Qwen3-8B", "L40S", str(tmp_path), str(tmp_path),
                                "2026-09-27T000000Z", profile) == 1
    assert (tmp_path / "FAILED_qwen3-8b_l40s_sip_2026-09-27T000000Z.log").read_text() == "sip\n"
    assert (tmp_path / "FAILED_qwen3-8b_l40s_gulp_2026-09-27T000000Z.log").read_text() == "gulp\n"
    assert sweep.cell_stem("Qwen3-8B", "L40S", "sip", "s") != sweep.cell_stem("Qwen3-8B", "L40S", "gulp", "s")


def test_profile_mismatch_fails_a_cell_that_timed_another_profile(sweep):
    assert sweep.PROFILES == ("sip", "gulp")
    assert sweep.profile_mismatch({"compression": {"profile": "gulp"}}, "gulp") is None
    assert sweep.profile_mismatch({"compression": {"profile": "sip"}}, "sip") is None
    assert "'sip', not the 'gulp'" in sweep.profile_mismatch({"compression": {"profile": "sip"}}, "gulp")
    assert "None" in sweep.profile_mismatch({}, "sip")


def test_pack_verified_reads_drinkme_verify(sweep):
    ok = "verified, manifest and all: /root/packs/p — 253 files verified, manifest 0123456789abcdef…\n"
    assert sweep.pack_verified(ok)
    assert not sweep.pack_verified("")
    assert not sweep.pack_verified(ok + "drinkme verify: REFUSED — sha256 mismatch on w.npz\n")
    assert not sweep.pack_verified("  " + ok)  # the line must start with it, as a grep ^ would


def test_spawn_names_the_call_by_its_profile(tmp_path, monkeypatch):
    """The driver passes --profile into the function and into the call
    JSON's name and body; a sip spawn and a gulp spawn of one model and
    card at one stamp are two files."""
    calls = []

    class _Fn:
        def spawn(self, *a, **k):
            calls.append((a, k))
            return types.SimpleNamespace(object_id=f"fc-{len(calls)}")

    fake = _fake_modal()
    fake.Function = types.SimpleNamespace(from_name=lambda app, name: _Fn())
    drive = _load(DRIVE, "modal_sweep_drive_under_test", fake, extra_saved=("modal_sweep",))
    monkeypatch.setattr(drive.ms, "local_commit", lambda: "c0ffee")
    monkeypatch.setattr(drive.ms, "local_token", lambda: "")
    monkeypatch.setattr(drive.ms, "utc_stamp", lambda: "2026-09-27T000000Z")
    assert drive.spawn("L40S", "Qwen3-8B", str(tmp_path)) == 0
    assert drive.spawn("L40S", "Qwen3-8B", str(tmp_path), "gulp") == 0
    assert drive.spawn("L40S", "Qwen3-8B", str(tmp_path), "fp8") == 2
    assert [a[-1] for a, _ in calls] == ["sip", "gulp"]
    sip = json.loads((tmp_path / "qwen3-8b_l40s_sip_2026-09-27T000000Z.call.json").read_text())
    gulp = json.loads((tmp_path / "qwen3-8b_l40s_gulp_2026-09-27T000000Z.call.json").read_text())
    assert (sip["profile"], sip["call_id"]) == ("sip", "fc-1")
    assert (gulp["profile"], gulp["call_id"]) == ("gulp", "fc-2")
