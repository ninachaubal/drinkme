"""publish/validate.py: a record that fails the lexicon is refused with the
field named; the shipped lexicon file is found; the atproto data-model
rule (no floats) and the baseline verdict are applied when present."""

import json

import pytest

from drinkme import MEASUREMENT, __version__, bench
from drinkme.detect import Hardware
from drinkme.publish import validate as V


def good():
    """A record as bench writes it: the in-memory result run through
    bench.lexicon_safe (floats -> decimal strings, None dropped)."""
    hw = Hardware(device_class="4090", memory_gb=24.0, memory_kind="vram", budget_gb=24.0,
                  memory_bytes=25_757_220_864, budget_bytes=25_757_220_864,
                  cpu_info="cpu", gpu_info="NVIDIA GeForce RTX 4090, 24564", os_info="Linux 6")
    return bench.lexicon_safe({
        "$type": MEASUREMENT,
        "createdAt": "2026-09-12T20:01:02.345678+00:00",
        "version": __version__,
        "environment": hw.as_record_env(),
        "model": {"name": "Qwen3-8B", "hfRepo": "Qwen/Qwen3-8B",
                  "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
        "compression": {"profile": "sip", "bitsPerWeight": "11.37"},
        "stock": {"outcome": "measured"},
        "metrics": [{"name": "stock_decode_tok_s", "value": "6.92", "unit": "tok/s",
                     "samples": ["6.9", "6.92", "6.95"]},
                    {"name": "read_gb_s", "value": "222.4", "unit": "GB/s"}],
        "raw": {"anything": "goes", "n": 3, "ok": True, "tok_s": 6.92},
    })


def test_lexicon_file_is_found_and_is_ours():
    lex = V.load_lexicon()
    assert lex["id"] == MEASUREMENT and lex["defs"]["main"]["type"] == "record"


def test_a_bench_shaped_record_is_valid():
    assert V.validate(good()) == []


def test_extra_fields_are_ignored_per_the_lexicon_spec():
    r = good()
    r["site"] = {"whatever": 1}
    r["environment"]["evidence"] = ["x"]
    assert V.validate(r) == []


@pytest.mark.parametrize("mutate, field", [
    (lambda r: r.pop("model"), "model"),
    (lambda r: r.pop("version"), "version"),
    (lambda r: r.pop("stock"), "stock"),
    (lambda r: r.pop("compression"), "compression"),
    (lambda r: r["compression"].pop("profile"), "compression.profile"),
    (lambda r: r["compression"].__setitem__("profile", "balanced"), "compression.profile"),
    (lambda r: r["stock"].__setitem__("outcome", "unsupported_on_backend"), "stock.outcome"),
    (lambda r: r["stock"].__setitem__("error", "x" * 2049), "stock.error"),
    (lambda r: r["environment"].pop("memoryBytes"), "environment.memoryBytes"),
    (lambda r: r["environment"].__setitem__("memoryBytes", "25757220864"), "environment.memoryBytes"),
    (lambda r: r["environment"].__setitem__("memoryBytes", 24.0), "environment.memoryBytes"),
    (lambda r: r["environment"].__setitem__("memoryKind", "ram"), "environment.memoryKind"),
    (lambda r: r["environment"].__setitem__("platform", "vulkan"), "environment.platform"),
    (lambda r: r["metrics"][0].__setitem__("value", 6.92), "metrics[0].value"),
    (lambda r: r["metrics"][0]["samples"].append(7.0), "metrics[0].samples[3]"),
    (lambda r: r.__setitem__("metrics", "nope"), "metrics"),
    (lambda r: r.__setitem__("createdAt", "2026-09-12 20:01"), "createdAt"),
    (lambda r: r.__setitem__("createdAt", "2026-09-12T20:01:02"), "createdAt"),
    (lambda r: r["model"].__setitem__("name", "x" * 129), "model.name"),
    (lambda r: r["model"].__setitem__("revision", None), "model.revision"),
    (lambda r: r.__setitem__("$type", "app.bsky.feed.post"), "$type"),
])
def test_refusals_name_the_field(mutate, field):
    r = good()
    mutate(r)
    errors = V.validate(r)
    assert errors, f"expected a refusal for {field}"
    assert any(e.startswith(field + ":") for e in errors), errors


CLOSED_ENUMS = {
    "compression.profile": ["sip", "gulp"],
    "environment.memoryKind": ["vram", "unified"],
    "environment.platform": ["cuda", "rocm", "metal"],
    "stock.outcome": ["measured", "skipped_predicted_nonfit", "failed_load"],
}


def test_the_four_enums_are_closed():
    """Each is an `enum` in the lexicon with exactly these values, and the
    validator refuses any other value by path — no open string anywhere a
    record names a kind."""
    defs = V.load_lexicon()["defs"]
    for path, values in CLOSED_ENUMS.items():
        obj, field = path.split(".")
        assert defs[obj]["properties"][field]["enum"] == values, path
        r = good()
        r[obj][field] = "something-else"
        assert any(e.startswith(f"{path}: 'something-else' not one of") for e in V.validate(r)), path


def test_floats_anywhere_are_refused_with_the_path():
    r = good()
    r["raw"]["tok_s"] = 6.92
    r["raw"]["nested"] = [{"p50": 0.001}]
    errors = V.validate(r)
    assert any(e.startswith("raw.tok_s: float 6.92") for e in errors), errors
    assert any(e.startswith("raw.nested[0].p50: float") for e in errors), errors


def test_maxlength_is_utf8_bytes():
    r = good()
    r["model"]["name"] = "é" * 100  # 100 graphemes, 200 bytes > 128
    assert any(e.startswith("model.name:") for e in V.validate(r))


def test_the_file_bench_writes_passes(tmp_path):
    """bench.write_record -> validate: the two verbs agree on the bytes."""
    r = good()
    r["raw"] = {"tok_s": 6.92, "samples": [6.9, 7.0], "none": None}
    r["environment"]["os"] = None  # an optional field left None is dropped, never written null
    path = bench.write_record(r, str(tmp_path / "r.json"))
    on_disk = json.load(open(path))
    assert V.validate(on_disk) == []


# ---- nothing local leaves the machine ----

def test_drop_local_removes_pack_dir_and_leaves_the_rest():
    r = good()
    r["raw"]["pack_dir"] = "/home/someone/.cache/drinkme/packs-v1/Qwen--Qwen3-8B@b968826d9c46"
    out = V.drop_local(r)
    assert "pack_dir" not in out["raw"] and out["raw"]["anything"] == "goes"
    assert r["raw"]["pack_dir"].startswith("/home/")  # the local file's dict is not mutated
    assert V.drop_local(good()) == good()


@pytest.mark.parametrize("path", [
    "/home/someone/x", "/Users/someone/x", "/root/packs/x", "~/.cache/drinkme/x",
    "C:\\Users\\someone\\x", "loaded from /home/someone/x"])
def test_a_local_path_anywhere_is_refused(path):
    r = good()
    r["raw"]["some_future_key"] = {"nested": [path]}
    verdict = V.local_path_verdict(r)
    assert verdict and verdict.startswith("raw.some_future_key.nested[0]")


def test_ordinary_strings_are_not_paths():
    r = good()
    r["raw"]["gpu"] = "amdgpu 0x1586 vram=0.54GB gtt=133.14GB"
    r["raw"]["note"] = "cache hit; a/b tested 1/2 the time"
    assert V.local_path_verdict(r) is None


def test_publish_refuses_a_record_with_a_path_and_sends_one_without_pack_dir():
    from drinkme import publish
    r = good()
    r["raw"]["pack_dir"] = "/home/someone/.cache/drinkme/packs-v1/x"
    publish.check_record(V.drop_local(r), "r.json")  # stripped: goes
    r["raw"]["other"] = "/home/someone/y"
    with pytest.raises(publish.RecordRefused) as e:
        publish.check_record(V.drop_local(r), "r.json")
    assert "raw.other" in str(e.value)
