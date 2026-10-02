"""serve.run() wiring: build_engine routes each arm to the right loader with
the right arguments (loaders faked — engine loading itself is
test_serving_engines.py's job), and the DRINKME_FAKE_ENGINE=1 dev path serves
FakeEngine without ever packing."""

import os

import pytest

from drinkme import serve
from drinkme.bootstrap import BootstrapOutcome
from drinkme.serving import engines


@pytest.fixture(autouse=True)
def pinned_env(linux_host, monkeypatch):
    # linux_host: build_engine takes the torch runtime (on a Mac it would take mlx)
    monkeypatch.delenv("DRINKME_FAKE_ENGINE", raising=False)
    monkeypatch.setenv("DRINKME_DEVICE", "cpu")  # never auto-pick a device in tests


def test_run_compressed_wires_pack_into_load_compressed(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text('{"formatVersion": 1}')  # ensure_pack sees a cache hit
    seen, eng = {}, object()
    monkeypatch.setattr(engines, "load_compressed",
                        lambda repo, rev, path, device, **kw: seen.update(
                            repo=repo, rev=rev, path=path, device=device) or eng)
    monkeypatch.setattr(serve, "_serve", lambda e, host, port, *a, **kw: 0 if e is eng else 1)
    rc = serve.run("fake/repo", "r1", pack_dir=str(tmp_path), auto_pack=False)
    assert rc == 0  # the loaded engine is exactly what got served
    assert seen == {"repo": "fake/repo", "rev": "r1", "path": str(tmp_path),
                    "device": "cpu"}


def test_run_stock_wires_load_stock_and_never_packs(monkeypatch):
    seen, eng = {}, object()
    monkeypatch.setattr(serve, "ensure_pack",
                        lambda *a, **k: pytest.fail("--stock must not pack"))
    monkeypatch.setattr(engines, "load_stock",
                        lambda repo, rev, device, **kw: seen.update(
                            repo=repo, rev=rev, device=device) or eng)
    monkeypatch.setattr(serve, "_serve", lambda e, host, port, *a, **kw: 0 if e is eng else 1)
    rc = serve.run("fake/repo", None, stock=True)
    assert rc == 0
    assert seen == {"repo": "fake/repo", "rev": None, "device": "cpu"}


def test_prefix_slots_reaches_build_engine(tmp_path, monkeypatch):
    """--prefix-slots (prefix slots): serve.run -> build_engine -> the loader, same
    plumbing depth as --ctx."""
    (tmp_path / "meta.json").write_text('{"formatVersion": 1}')
    seen, eng = {}, object()
    monkeypatch.setattr(engines, "load_compressed",
                        lambda repo, rev, path, device, **kw: seen.update(kw) or eng)
    monkeypatch.setattr(serve, "_serve", lambda e, host, port, *a, **kw: 0)
    serve.run("fake/repo", "r1", pack_dir=str(tmp_path), auto_pack=False, prefix_slots=3)
    assert seen["prefix_slots"] == 3


def test_sleep_on_idle_reaches_the_server(tmp_path, monkeypatch):
    """--sleep-on-idle (sleep/wake): serve.run resolves it (flag over env, garbage
    to 0) and hands _serve a number of seconds — http.py parses no
    environment of its own."""
    (tmp_path / "meta.json").write_text('{"formatVersion": 1}')
    seen, eng = {}, object()
    monkeypatch.setattr(engines, "load_compressed", lambda *a, **k: eng)
    # *a/**kw so this absorbs every OTHER wiring argument _serve now carries
    # (advertised_ctx, served_names/profiles); sleep_on_idle stays
    # named because it is the one this test is about.
    monkeypatch.setattr(serve, "_serve",
                        lambda e, host, port, *a, sleep_on_idle=0.0, **kw:
                        seen.update(idle=sleep_on_idle) or 0)
    serve.run("fake/repo", "r1", pack_dir=str(tmp_path), auto_pack=False)
    assert seen["idle"] == 0.0  # default: never
    serve.run("fake/repo", "r1", pack_dir=str(tmp_path), auto_pack=False,
              sleep_on_idle=900)
    assert seen["idle"] == 900.0
    monkeypatch.setenv("DRINKME_SLEEP_ON_IDLE_S", "300")
    serve.run("fake/repo", "r1", pack_dir=str(tmp_path), auto_pack=False)
    assert seen["idle"] == 300.0


def test_fake_env_serves_fake_engine_and_never_packs(monkeypatch):
    monkeypatch.setenv("DRINKME_FAKE_ENGINE", "1")
    monkeypatch.setattr(serve, "ensure_pack",
                        lambda *a, **k: pytest.fail("fake path must not pack"))
    served = {}
    monkeypatch.setattr(serve, "_serve",
                        lambda eng, host, port, *a, **kw: served.update(eng=eng, port=port) or 0)
    rc = serve.run("Qwen/Qwen3-8B", None, port=0)
    assert rc == 0 and served["eng"].model_id == "Qwen/Qwen3-8B"
    assert type(served["eng"]).__name__ == "FakeEngine"


def test_port_already_in_use_gives_an_actionable_message_not_a_traceback(
    monkeypatch, capsys):
    """The OSError from the socket bind is named, never a bare 12-frame
    traceback naming neither the port nor --port. DRINKME_FAKE_ENGINE
    exercises the real bind through _serve with no model/GPU needed."""
    import socket

    monkeypatch.setenv("DRINKME_FAKE_ENGINE", "1")
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        rc = serve.run("Qwen/Qwen3-8B", None, port=port)
    finally:
        blocker.close()
    assert rc != 0
    err = capsys.readouterr().err
    assert str(port) in err
    assert "--port" in err


def test_cli_prefix_slots_flag_reaches_serve_run(monkeypatch):
    """`drinkme serve --prefix-slots N` (prefix slots), end to end through argparse."""
    from drinkme import cli

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    rc = cli.main(["serve", "--model", "Qwen3-8B", "--prefix-slots", "3"])
    assert rc == 0 and seen["prefix_slots"] == 3


def test_cli_no_prefix_slots_flag_passes_none(monkeypatch):
    """Unset CLI flag must not silently pick a slot count — None means "fall
    through to DRINKME_PREFIX_SLOTS or the default", engines.slots_from_env's
    job, not cli.py's."""
    from drinkme import cli

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    cli.main(["serve", "--model", "Qwen3-8B"])
    assert seen["prefix_slots"] is None


def test_cli_advertised_ctx_flag_reaches_serve_run(monkeypatch):
    """`drinkme serve --advertised-ctx N` (advertised context), end to end through argparse."""
    from drinkme import cli

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    rc = cli.main(["serve", "--model", "Qwen3-8B", "--advertised-ctx", "8192"])
    assert rc == 0 and seen["advertised_ctx"] == 8192


def test_cli_no_advertised_ctx_flag_passes_none(monkeypatch):
    from drinkme import cli

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    cli.main(["serve", "--model", "Qwen3-8B"])
    assert seen["advertised_ctx"] is None


def test_cli_image_max_pixels_flag_reaches_serve_run(monkeypatch, capsys):
    """`drinkme serve --image-max-pixels` takes N or WxH, through argparse;
    garbage is argparse's usage error, not a traceback."""
    from drinkme import cli

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    assert cli.main(["serve", "--model", "Qwen3-8B", "--image-max-pixels", "1920x1080"]) == 0
    assert seen["image_max_pixels"] == 1920 * 1080
    cli.main(["serve", "--model", "Qwen3-8B"])
    assert seen["image_max_pixels"] is None
    with pytest.raises(SystemExit):
        cli.main(["serve", "--model", "Qwen3-8B", "--image-max-pixels", "huge"])
    assert "not a pixel count" in capsys.readouterr().err


def test_image_max_pixels_is_written_for_the_engine_and_the_dialects(monkeypatch):
    """--image-max-pixels is applied the way --spec is, by writing the
    environment serving/vision.max_pixels_from_env reads; no flag leaves
    the environment's value alone."""
    from drinkme.serving import vision

    monkeypatch.setenv("DRINKME_FAKE_ENGINE", "1")
    monkeypatch.setenv("DRINKME_IMAGE_MAX_PIXELS", "100")  # monkeypatch restores it afterwards
    monkeypatch.setattr(serve, "_serve", lambda *a, **kw: 0)
    serve.run("Qwen/Qwen3-8B", None, port=0)
    assert vision.max_pixels_from_env() == 100
    serve.run("Qwen/Qwen3-8B", None, port=0, image_max_pixels=2_073_600)
    assert vision.max_pixels_from_env() == 2_073_600


def test_cli_no_image_urls_and_media_path_flags_reach_serve_run(monkeypatch, capsys, tmp_path):
    """`drinkme serve --no-image-urls --media-path DIR`, through
    argparse; --media-path validates the directory AT PARSE TIME (a human
    just typed it — the same posture --image-max-pixels's garbage-value
    error takes), so a bad one is argparse's usage error, not a traceback."""
    from drinkme import cli

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    assert cli.main(["serve", "--model", "Qwen3-8B", "--no-image-urls",
                     "--media-path", str(tmp_path)]) == 0
    assert seen["no_image_urls"] is True
    assert seen["media_path"] == os.path.realpath(str(tmp_path))
    cli.main(["serve", "--model", "Qwen3-8B"])
    assert seen["no_image_urls"] is False and seen["media_path"] is None
    with pytest.raises(SystemExit):
        cli.main(["serve", "--model", "Qwen3-8B", "--media-path", str(tmp_path / "nope")])
    assert "is not a directory" in capsys.readouterr().err


def test_image_urls_and_media_path_are_written_for_the_engine_and_the_dialects(monkeypatch):
    """--no-image-urls/--media-path are applied the way --image-max-pixels
    is, by writing the environment serving/vision.fetch_urls_from_env /
    media_path_from_env read; no flag leaves the environment's value
    alone."""
    from drinkme.serving import vision

    monkeypatch.setenv("DRINKME_FAKE_ENGINE", "1")
    monkeypatch.delenv("DRINKME_IMAGE_URLS", raising=False)
    monkeypatch.delenv("DRINKME_MEDIA_PATH", raising=False)
    monkeypatch.setattr(serve, "_serve", lambda *a, **kw: 0)
    serve.run("Qwen/Qwen3-8B", None, port=0)
    assert vision.fetch_urls_from_env() is True  # no flag: environment untouched, default on
    assert vision.media_path_from_env() is None
    serve.run("Qwen/Qwen3-8B", None, port=0, no_image_urls=True, media_path="/tmp")
    assert vision.fetch_urls_from_env() is False
    assert vision.media_path_from_env() == os.path.realpath("/tmp")


@pytest.mark.parametrize("raw, expect", [("8192", 8192), ("", None), ("0", None), ("nope", None)])
def test_advertised_ctx_from_env(monkeypatch, raw, expect, capsys):
    """Same posture as engines.slots_from_env: garbage warns and falls back
    to None (the default real-ctx behaviour) rather than dying or silently
    starting to lie about token counts."""
    if raw:
        monkeypatch.setenv("DRINKME_ADVERTISED_CTX", raw)
    else:
        monkeypatch.delenv("DRINKME_ADVERTISED_CTX", raising=False)
    assert serve._advertised_ctx_from_env(None) == expect
    if expect is None and raw not in ("", None):
        assert "DRINKME_ADVERTISED_CTX" in capsys.readouterr().err


def test_advertised_ctx_explicit_cli_wins_over_env(monkeypatch):
    monkeypatch.setenv("DRINKME_ADVERTISED_CTX", "4096")
    assert serve._advertised_ctx_from_env(8192) == 8192


def test_ctx_clamps_to_native_window(capsys):
    """An explicit ctx past max_position_embeddings clamps to native (with a
    stderr line) instead of reaching StaticCache as an unbounded ask (an
    81 GiB OOM on Strix Halo). Within-window asks pass through untouched."""
    from drinkme.serving.engines import _ctx

    class Cfg:
        max_position_embeddings = 262144

    assert _ctx(Cfg(), 1048576) == 262144
    assert "clamped" in capsys.readouterr().err
    assert _ctx(Cfg(), 131072) == 131072
    assert _ctx(Cfg(), None) == 8192  # default cap unchanged


# --------------------------------------------------- rope scaling (YaRN) --


class _RopeCfg:
    max_position_embeddings = 32768
    rope_scaling = None


def test_apply_rope_scaling_off_is_a_no_op():
    from drinkme.serving.engines import _apply_rope_scaling

    cfg = _RopeCfg()
    _apply_rope_scaling(cfg, None)
    assert cfg.max_position_embeddings == 32768
    assert cfg.rope_scaling is None


def test_apply_rope_scaling_sets_fields_and_unclamps_ctx(capsys):
    """The three fields _ctx (and transformers' rotary module) read, and the
    clamp this unblocks: with the window widened BEFORE _ctx runs, an
    explicit --ctx past the ORIGINAL native window passes through untouched
    instead of being clamped back down to it."""
    from drinkme.serving.engines import _apply_rope_scaling, _ctx

    cfg = _RopeCfg()
    _apply_rope_scaling(cfg, {"rope_type": "yarn", "factor": 4})
    assert cfg.rope_scaling == {"rope_type": "yarn", "factor": 4,
                                "original_max_position_embeddings": 32768}
    assert cfg.max_position_embeddings == 131072
    capsys.readouterr()  # drain the "rope scaling: yarn x4" boot line
    assert _ctx(cfg, 131072) == 131072
    assert "clamped" not in capsys.readouterr().err


def test_apply_rope_scaling_announces_replacing_an_existing_config(capsys):
    """Never silently: a checkpoint that already carries a rope_scaling gets
    its old value said out loud before it is overwritten."""
    from drinkme.serving.engines import _apply_rope_scaling

    cfg = _RopeCfg()
    cfg.rope_scaling = {"rope_type": "dynamic", "factor": 2}
    _apply_rope_scaling(cfg, {"rope_type": "yarn", "factor": 4})
    err = capsys.readouterr().err
    assert "dynamic" in err and "replac" in err


def test_apply_rope_scaling_preserves_rope_theta_on_a_real_config():
    """Regression: a bare replace of rope_scaling (this function's first
    draft, and the original rope-scaling spec) wipes rope_theta out of
    rope_parameters, and transformers-5's Llama/Qwen2/Qwen3 configs keep no
    top-level rope_theta field to fall back to — YaRN construction then
    reads back None and crashes on `None ** tensor` (measured against a real
    LlamaConfig, not the stub above, so a future transformers upgrade that
    reintroduces a top-level field can't silently regress this into a false
    pass)."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    from drinkme.serving.engines import _apply_rope_scaling

    cfg = LlamaConfig(hidden_size=32, num_attention_heads=2,
                      max_position_embeddings=64)
    assert cfg.rope_parameters["rope_theta"] == 10000.0  # the value at stake
    _apply_rope_scaling(cfg, {"rope_type": "yarn", "factor": 4})
    assert cfg.rope_scaling["rope_theta"] == 10000.0  # preserved, not wiped
    LlamaRotaryEmbedding(config=cfg)  # must not raise


def test_rope_scaling_flag_parses_yarn_factor():
    from drinkme.cli import parse_rope_scaling

    assert parse_rope_scaling("yarn:4") == {"rope_type": "yarn", "factor": 4.0}
    assert parse_rope_scaling("yarn:2.5") == {"rope_type": "yarn", "factor": 2.5}
    # the recipe-exact form: Qwen3's card names original 32768 (the 8B's
    # config says 40,960 = 32k context + 8k generation, a different number)
    assert parse_rope_scaling("yarn:4:32768") == {
        "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


def test_apply_rope_scaling_honours_the_recipe_original(capsys):
    """With an explicit original the window scales from the RECIPE's number,
    not the checkpoint's max_position_embeddings: Qwen3-8B 40,960 native +
    yarn:4:32768 -> 131,072 (the card's figure), original recorded as 32768."""
    from drinkme.serving.engines import _apply_rope_scaling, _ctx

    cfg = _RopeCfg()
    cfg.max_position_embeddings = 40960
    _apply_rope_scaling(cfg, {"rope_type": "yarn", "factor": 4,
                              "original_max_position_embeddings": 32768})
    assert cfg.max_position_embeddings == 131072
    assert cfg.rope_scaling["original_max_position_embeddings"] == 32768
    err = capsys.readouterr().err
    assert "on original 32768" in err and "40960 -> 131072" in err
    assert _ctx(cfg, 131072) == 131072


def test_rope_scaling_flag_off_is_none():
    from drinkme.cli import parse_rope_scaling

    assert parse_rope_scaling("off") is None


@pytest.mark.parametrize("garbage", ["yarn", "yarn:", "yarn:0", "yarn:-4",
                                     "yarn:nope", "on", "",
                                     "yarn:4:0", "yarn:4:x", "yarn:4:32768:9"])
def test_rope_scaling_flag_garbage_raises(garbage):
    from drinkme.cli import parse_rope_scaling

    with pytest.raises(ValueError):
        parse_rope_scaling(garbage)


def test_cli_rope_scaling_garbage_is_a_clear_argparse_error(monkeypatch, capsys):
    from drinkme import cli

    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--model", "Qwen3-8B", "--rope-scaling", "yarn:nope"])
    assert e.value.code == 2
    assert "--rope-scaling" in capsys.readouterr().err


def test_cli_rope_scaling_flag_reaches_serve_run(monkeypatch):
    """`drinkme serve --rope-scaling yarn:4` (rope scaling), end to end through argparse."""
    from drinkme import cli

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    rc = cli.main(["serve", "--model", "Qwen3-8B", "--rope-scaling", "yarn:4"])
    assert rc == 0 and seen["rope_scaling"] == {"rope_type": "yarn", "factor": 4.0}


def test_cli_no_rope_scaling_flag_passes_none(monkeypatch):
    """Unset CLI flag must not silently pick a spec — None means "fall
    through to DRINKME_ROPE_SCALING or off", serve._rope_scaling_from_env's
    job, not cli.py's. --ctx alone (no --rope-scaling) must never turn this
    on either — this is the same None, exercised the same way."""
    from drinkme import cli

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    cli.main(["serve", "--model", "Qwen3-8B", "--ctx", "131072"])
    assert seen["rope_scaling"] is None


@pytest.mark.parametrize("raw, expect", [
    ("yarn:4", {"rope_type": "yarn", "factor": 4.0}),
    ("off", None), ("", None), ("nope", None),
])
def test_rope_scaling_from_env(monkeypatch, raw, expect, capsys):
    """Same posture as _advertised_ctx_from_env: garbage warns and falls back
    to None (rope scaling stays off) rather than dying or silently widening
    the window nobody asked for."""
    if raw:
        monkeypatch.setenv("DRINKME_ROPE_SCALING", raw)
    else:
        monkeypatch.delenv("DRINKME_ROPE_SCALING", raising=False)
    assert serve._rope_scaling_from_env(None) == expect
    if expect is None and raw not in ("", "off"):
        assert "DRINKME_ROPE_SCALING" in capsys.readouterr().err


def test_rope_scaling_explicit_cli_wins_over_env(monkeypatch):
    monkeypatch.setenv("DRINKME_ROPE_SCALING", "yarn:8")
    explicit = {"rope_type": "yarn", "factor": 4.0}
    assert serve._rope_scaling_from_env(explicit) == explicit


def test_rope_scaling_reaches_build_engine(tmp_path, monkeypatch):
    """--rope-scaling (rope scaling): serve.run -> build_engine -> the loader, same
    plumbing depth as --ctx and --prefix-slots."""
    (tmp_path / "meta.json").write_text('{"formatVersion": 1}')
    seen, eng = {}, object()
    monkeypatch.setattr(engines, "load_compressed",
                        lambda repo, rev, path, device, **kw: seen.update(kw) or eng)
    monkeypatch.setattr(serve, "_serve", lambda e, host, port, *a, **kw: 0)
    spec = {"rope_type": "yarn", "factor": 4.0}
    serve.run("fake/repo", "r1", pack_dir=str(tmp_path), auto_pack=False,
              rope_scaling=spec)
    assert seen["rope_scaling"] == spec


# ------------------------------------------------------------ resolve_device --


class _Torch:
    """Just enough torch to exercise resolve_device's three worlds."""

    def __init__(self, version, cuda=None, hip=None, available=False, name="GPU"):
        self.__version__ = version
        self.version = type("v", (), {"cuda": cuda, "hip": hip})()
        self.cuda = type("c", (), {
            "is_available": staticmethod(lambda: available),
            "get_device_name": staticmethod(lambda i: name)})()


def test_accelerator_wheel_seeing_no_device_refuses_to_serve(monkeypatch):
    """A `uv sync` without `--no-sync` can swap a pinned ROCm torch for a
    PyPI +cu130 wheel; it reports zero devices, and a serve that fell back
    to CPU would keep /health saying ok while clients crawl. A CUDA-or-ROCm
    build that sees no device is a broken install — refuse, loudly, with
    the fix command in the message."""
    monkeypatch.delenv("DRINKME_DEVICE", raising=False)  # beat the autouse pin
    with pytest.raises(SystemExit) as e:
        serve.resolve_device(_Torch("2.13.0+cu130", cuda="13.0"))
    msg = str(e.value)
    assert "drinkme serve: refused" in msg and "CUDA" in msg
    assert "--no-default-groups --group rocm" in msg  # the fix, not just the fault

    with pytest.raises(SystemExit):  # same rule for a ROCm build with no device
        serve.resolve_device(_Torch("2.12.0a0+rocm7.13", hip="7.13"))


def test_a_genuine_cpu_wheel_warns_and_serves(monkeypatch, capsys):
    """A stranger on a real CPU box made a real choice — don't refuse them.
    Both version fields None is the discriminator."""
    monkeypatch.delenv("DRINKME_DEVICE", raising=False)
    assert serve.resolve_device(_Torch("2.13.0")) == "cpu"
    out = capsys.readouterr()
    assert "warning" in out.err and "sees no GPU" in out.err
    assert "device: cpu" in out.out


def test_explicit_cpu_is_never_second_guessed(monkeypatch, capsys):
    monkeypatch.setenv("DRINKME_DEVICE", "cpu")
    # even with an accelerator wheel present: the operator said so
    assert serve.resolve_device(_Torch("2.13.0+cu130", cuda="13.0")) == "cpu"
    assert "refusing" not in capsys.readouterr().out


def test_the_device_announce_is_flushed(monkeypatch):
    """Not decoration: an announce can be CORRECT and still never reach the
    journal, because print() block-buffers into a pipe.
    A diagnostic nobody can read is not a diagnostic."""
    monkeypatch.delenv("DRINKME_DEVICE", raising=False)
    flushed = []

    class Sink:
        def write(self, s): return len(s)
        def flush(self): flushed.append(True)

    monkeypatch.setattr("sys.stdout", Sink())
    serve.resolve_device(_Torch("2.12.0a0+rocm7.13", hip="7.13", available=True))
    assert flushed, "device announce must flush — a buffered log line is no log line"


# ------------------------------------------- a menu name is served as an alias --

def _serve_via_cli(monkeypatch, argv):
    """cli.main(["serve", ...]) over DRINKME_FAKE_ENGINE=1, with _serve swapped
    for a non-blocking start_server so the test can query the real HTTP layer."""
    import http.client
    import json

    from drinkme import cli
    from drinkme.serving.http import start_server

    monkeypatch.setenv("DRINKME_FAKE_ENGINE", "1")
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    started = {}

    def fake_serve(eng, host, port, *a, served_names=None, generation_profiles=None, **kw):
        started["srv"] = start_server(eng, "127.0.0.1", 0, served_names=served_names,
                                      generation_profiles=generation_profiles)
        return 0

    monkeypatch.setattr(serve, "_serve", fake_serve)
    assert cli.main(["serve", *argv]) == 0
    srv = started["srv"]
    port = srv.server_address[1]

    def request(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request(method, path, json.dumps(body) if body else None,
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def ids():
        return [e["id"] for e in json.loads(request("GET", "/v1/models")[1])["data"]]

    def status_for(model):
        return request("POST", "/v1/chat/completions",
                       {"model": model, "messages": [{"role": "user", "content": "hi"}]})[0]

    return srv, ids, status_for


@pytest.fixture
def no_env_names(monkeypatch):
    monkeypatch.delenv("DRINKME_SERVED_MODEL_NAMES", raising=False)


def test_menu_name_is_listed_and_accepted_beside_the_repo_id(monkeypatch, no_env_names):
    srv, ids, status_for = _serve_via_cli(monkeypatch, ["--model", "Qwen3-8B"])
    try:
        assert srv.engine.model_id == "Qwen/Qwen3-8B"  # records and logs keep the repo
        assert ids() == ["Qwen/Qwen3-8B", "Qwen3-8B"]
        assert status_for("Qwen3-8B") == 200
        assert status_for("Qwen/Qwen3-8B") == 200
    finally:
        srv.shutdown()


def test_repo_id_adds_no_alias(monkeypatch, no_env_names):
    srv, ids, status_for = _serve_via_cli(monkeypatch, ["--model", "Qwen/Qwen3-8B"])
    try:
        assert ids() == ["Qwen/Qwen3-8B"]
        assert status_for("Qwen3-8B") == 404
    finally:
        srv.shutdown()


def test_local_path_adds_no_alias(tmp_path, monkeypatch, no_env_names):
    path = str(tmp_path / "Qwen3-8B")
    srv, ids, _ = _serve_via_cli(monkeypatch, ["--model", path])
    try:
        assert ids() == [path]
    finally:
        srv.shutdown()


def test_served_model_name_flag_adds_to_the_menu_name(monkeypatch, no_env_names):
    srv, ids, status_for = _serve_via_cli(
        monkeypatch, ["--model", "Qwen3-8B", "--served-model-name", "my-model"])
    try:
        assert ids() == ["Qwen/Qwen3-8B", "Qwen3-8B", "my-model"]
        assert status_for("my-model") == 200 and status_for("Qwen3-8B") == 200
    finally:
        srv.shutdown()


def test_served_names_env_adds_to_the_menu_name(monkeypatch):
    monkeypatch.setenv("DRINKME_SERVED_MODEL_NAMES", "my-model,Qwen3-8B")
    srv, ids, _ = _serve_via_cli(monkeypatch, ["--model", "Qwen3-8B"])
    try:
        assert ids() == ["Qwen/Qwen3-8B", "Qwen3-8B", "my-model"]  # no duplicate
    finally:
        srv.shutdown()
