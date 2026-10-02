"""`drinkme serve` with no `--model`: cli.main(["serve"]) end to end — picks a
model via drinkme.packs and reaches serve.run, or prompts on a real TTY and
honors confirm/cancel/override. Shape borrowed from test_serving_wiring.py
(monkeypatch serve.run + ensure_accelerator, call cli.main).

DRINKME_HOME always points at an empty tmp_path (never the real
~/.cache/drinkme) so these tests are deterministic regardless of what is
actually packed on the machine that runs them; discovery itself is
test_packs.py's job."""

from __future__ import annotations

import pytest

from drinkme import cli, packs, serve
from drinkme.bootstrap import BootstrapOutcome


@pytest.fixture(autouse=True)
def pinned_env(linux_host, tmp_path, monkeypatch):
    # linux_host: the torch runtime's families (on a Mac the mlx runtime's
    # would leave Qwen3.8-27B out of the pick)
    monkeypatch.setenv("DRINKME_DEVICE", "cpu")  # never auto-pick a device in tests
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))  # no real packs found
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    # A fixed budget that isolates Qwen3.8-27B as the pick: its charge is
    # the compressed estimate 40.73 GB -> ~37.9 GiB x1.1 headroom -> ~41.7
    # GiB (fits 45), while Qwen3-32B (the next size up) charges ~49.2 GiB
    # (does not fit 45).
    # Deterministic across boxes since local packs are always empty here.
    monkeypatch.setattr(packs, "hardware_budget", lambda: (45.0, "test 45.0 GiB"))


def _capture_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(serve, "run", lambda repo, rev, **kw: seen.update(
        repo=repo, rev=rev, kw=kw) or 0)
    return seen


def test_non_tty_picks_and_reaches_serve_run(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    seen = _capture_run(monkeypatch)
    rc = cli.main(["serve"])
    assert rc == 0
    assert seen["repo"] == "Qwen/Qwen3.8-27B"
    err = capsys.readouterr().err
    assert "no --model given" in err and "Qwen3.8-27B" in err


def test_yes_flag_skips_prompt_on_a_fake_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input",
                        lambda *a: pytest.fail("--yes must not prompt"))
    seen = _capture_run(monkeypatch)
    rc = cli.main(["serve", "--yes"])
    assert rc == 0
    assert seen["repo"] == "Qwen/Qwen3.8-27B"


def test_piped_non_tty_serves_with_no_prompt_even_if_it_looks_like_an_answer(monkeypatch):
    """`echo n | drinkme serve`: stdin is a pipe, not a TTY -- no prompt at
    all, the pick is served exactly as the quickstart/systemd path expects."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input",
                        lambda *a: pytest.fail("non-TTY must not prompt"))
    seen = _capture_run(monkeypatch)
    rc = cli.main(["serve"])
    assert rc == 0
    assert seen["repo"] == "Qwen/Qwen3.8-27B"


@pytest.mark.parametrize("answer", ["", "y"])
def test_fake_tty_enter_or_y_confirms_the_pick(monkeypatch, answer):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: answer)
    seen = _capture_run(monkeypatch)
    rc = cli.main(["serve"])
    assert rc == 0
    assert seen["repo"] == "Qwen/Qwen3.8-27B"


@pytest.mark.parametrize("answer", ["n", "q"])
def test_fake_tty_n_or_q_cancels(monkeypatch, capsys, answer):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: answer)
    monkeypatch.setattr(serve, "run", lambda *a, **kw: pytest.fail("must not serve"))
    rc = cli.main(["serve"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "cancelled" in err
    assert "Traceback" not in err


def test_fake_tty_eof_cancels(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input",
                        lambda *a: (_ for _ in ()).throw(EOFError()))
    monkeypatch.setattr(serve, "run", lambda *a, **kw: pytest.fail("must not serve"))
    rc = cli.main(["serve"])
    assert rc == 2
    assert "cancelled" in capsys.readouterr().err


def test_fake_tty_ctrl_c_cancels(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input",
                        lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(serve, "run", lambda *a, **kw: pytest.fail("must not serve"))
    rc = cli.main(["serve"])
    assert rc == 2
    assert "cancelled" in capsys.readouterr().err


def test_fake_tty_menu_name_overrides_the_pick(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "Qwen3-8B")
    seen = _capture_run(monkeypatch)
    rc = cli.main(["serve"])
    assert rc == 0
    assert seen["repo"] == "Qwen/Qwen3-8B"


def test_fake_tty_hf_repo_string_overrides_the_pick(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "some-org/some-model")
    seen = _capture_run(monkeypatch)
    rc = cli.main(["serve"])
    assert rc == 0
    assert seen["repo"] == "some-org/some-model" and seen["rev"] is None


def test_fake_tty_unknown_bare_name_gets_the_menu_refusal(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "not-a-real-model")
    monkeypatch.setattr(serve, "run", lambda *a, **kw: pytest.fail("must not serve"))
    rc = cli.main(["serve"])
    assert rc == 2
    assert "unknown menu model" in capsys.readouterr().err


def test_no_budget_refuses_with_pass_model_hint(monkeypatch, capsys):
    monkeypatch.setattr(packs, "hardware_budget",
                        lambda: (None, "could not detect a memory budget on this machine"))
    monkeypatch.setattr(serve, "run", lambda *a, **kw: pytest.fail("must not serve"))
    rc = cli.main(["serve"])
    assert rc == 2
    assert "--model" in capsys.readouterr().err


def test_nothing_fits_names_the_smallest_and_lists_local_packs(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(packs, "hardware_budget", lambda: (0.01, "test 0.01 GiB"))
    monkeypatch.setattr(serve, "run", lambda *a, **kw: pytest.fail("must not serve"))
    rc = cli.main(["serve"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "nothing fits" in err
    assert "Qwen3-0.6B" in err  # the smallest candidate on the menu
    assert "--model" in err
