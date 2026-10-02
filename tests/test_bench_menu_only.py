"""`drinkme bench --model` takes menu models only: its help and its
refusal say the same thing. CPU only, no torch import (dry_run is torch-free)."""

import pytest

from drinkme import bench, cli


def test_bench_help_says_menu_models_only(capsys):
    with pytest.raises(SystemExit):
        cli.main(["bench", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "menu models only" in out


def test_bench_refuses_an_off_menu_repo_and_accepts_a_menu_models_repo_id():
    from drinkme.suggest import MODELS

    plan = bench.dry_run(model_name="someone/off-menu-model")
    assert plan["points"] == [] and "unknown model" in plan["error"]
    q8 = next(m for m in MODELS if m.name == "Qwen3-8B")
    assert bench.dry_run(model_name=q8.hf_repo)["points"]
