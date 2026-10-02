"""serving/cudagraph_fit.py: what CUDA graphs are charged in the fit checks.

The formula (one pool per cache, plus a DeltaNet hybrid's row buffer at the
widest verify step), charged only where graphs run: CUDA, not
DRINKME_CUDA_GRAPHS=0, a verified family. Then the three places that charge
it: the serve picker (packs.rank_fits and its summary line), suggest's fit
point, and the engine at load, which decodes eager when the graphs would
leave less than the fit check's margin. Last, a cache and its captured
steps are not a reference cycle, so an outgrown slot's graphs go with it.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from drinkme import packs, suggest
from drinkme.serving import cudagraph_fit as cf

MIB = 1024**2

# Qwen3.8-27B's text config, as far as the charge reads it: 48 DeltaNet
# layers (3 of every 4), 48 value heads of 128 x 128, an MTP head
QWEN35 = {"model_type": "qwen3_5", "text_config": {
    "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1, "num_hidden_layers": 64,
    "hidden_size": 5120, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_conv_kernel_dim": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
    "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128}}
QWEN3 = {"model_type": "qwen3", "num_hidden_layers": 36}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in (cf.ENV, "DRINKME_SPEC", "DRINKME_MTP_DEPTH"):
        monkeypatch.delenv(k, raising=False)


def test_a_hybrids_row_is_its_deltanet_states_in_float32():
    assert cf.row_state_bytes(QWEN35) == 48 * 48 * 128 * 128 * 4 == 144 * MIB
    assert cf.row_state_bytes(QWEN3) == 0
    assert cf.family_of(QWEN35) == "qwen3_5_text" and cf.family_of(QWEN3) == "qwen3"


def test_on_cuda_the_charge_is_the_pool_plus_the_rows_per_cache():
    # MTP at depth 4: verify up to 5 rows, the last never stored
    assert cf.charge_bytes(QWEN35, "cuda", 1) == cf.POOL_BYTES + 4 * 144 * MIB
    assert cf.charge_bytes(QWEN35, "cuda", 3) == 3 * (cf.POOL_BYTES + 4 * 144 * MIB)
    # a full-attention model: the pool; --prefix-slots 0 still holds one cache
    assert cf.charge_bytes(QWEN3, "cuda", 2) == 2 * cf.POOL_BYTES
    assert cf.charge_bytes(QWEN3, "cuda", 0) == cf.POOL_BYTES


def test_the_plan_sets_the_widest_row(monkeypatch):
    monkeypatch.setenv("DRINKME_MTP_DEPTH", "2")
    assert cf.charge_bytes(QWEN35, "cuda", 1) == cf.POOL_BYTES + 2 * 144 * MIB
    monkeypatch.setenv("DRINKME_SPEC", "off")
    assert cf.charge_bytes(QWEN35, "cuda", 1) == cf.POOL_BYTES


def test_nothing_is_charged_where_graphs_do_not_run(monkeypatch):
    for platform in ("rocm", "metal", "cpu", None):
        assert cf.charge_bytes(QWEN35, platform, 1) == 0
    assert cf.charge_bytes({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}},
                           "cuda", 1) == 0
    assert cf.charge_bytes(None, "cuda", 1) == 0
    monkeypatch.setenv(cf.ENV, "0")
    assert cf.charge_bytes(QWEN35, "cuda", 1) == 0 and cf.charge_bytes(QWEN3, "cuda", 1) == 0


def candidate(config, resident_gib=38.0):
    return packs.Candidate("Qwen3.8-27B", "Qwen/Qwen3.8-27B", None, resident_gib, True, "/p", True,
                           config, 0.56, model_type=cf.family_of(config))


def test_the_picker_charges_graphs_only_on_cuda():
    c = candidate(QWEN35)
    off = packs.rank_fits(100.0, [c], ctx=4096, slots=1, ckpts=0)[0]
    on = packs.rank_fits(100.0, [c], ctx=4096, slots=1, ckpts=0, platform="cuda")[0]
    assert off.graph_gib == 0.0
    assert on.graph_gib == pytest.approx((cf.POOL_BYTES + 4 * 144 * MIB) / packs.GIB)
    assert on.total_gib == pytest.approx(off.total_gib + on.graph_gib)
    # a budget between the two charges: the graphs are what does not fit
    budget = (off.total_gib + on.graph_gib / 2) * suggest.FIT_HEADROOM
    assert packs.rank_fits(budget, [c], 4096, 1, 0)[0].fits
    assert not packs.rank_fits(budget, [c], 4096, 1, 0, platform="cuda")[0].fits
    lines = packs.format_summary(1000.0, "VRAM", [on], [], "/root", 4096, 1)
    assert any("+ CUDA graphs 0.66 GiB" in ln for ln in lines)
    assert not any("CUDA graphs" in ln for ln in packs.format_summary(1000.0, "VRAM", [off], [], "/r", 4096, 1))


def test_graph_platform_is_cuda_only_where_nvidia_smi_answered(monkeypatch):
    from drinkme import detect
    from drinkme.detect import Hardware

    def hw(evidence):
        return Hardware(device_class="d", memory_gb=48.0, memory_kind="vram", budget_gb=48.0,
                        cpu_info=None, gpu_info=None, evidence=evidence)

    monkeypatch.setattr(detect, "live_free_bytes", lambda _hw: (None, "unread"))
    monkeypatch.setattr(detect, "detect", lambda: hw(["nvidia-smi: NVIDIA L40S, 46068"]))
    assert packs.graph_platform("torch") == "cuda"
    assert packs.graph_platform("mlx") is None
    monkeypatch.setattr(detect, "detect", lambda: hw(["sysfs: amdgpu"]))
    assert packs.graph_platform("torch") is None


def test_suggests_fit_point_charges_its_graphs(monkeypatch):
    """A compressed size that fits with the headroom until the graphs are
    added becomes the knife-edge attempt, said as such."""
    row = suggest.Model("Big", "org/big", "r", auto_eligible=True)
    sz = suggest.Sizes(bf16_gb=60.0, comp_gb=43.0, profile="sip", pack_dir="/p")
    monkeypatch.setattr(suggest, "MODELS", [row])
    monkeypatch.setattr(suggest, "size_menu", lambda menu, vision=True: ([(row, sz)], []))
    monkeypatch.setattr(suggest, "graph_bytes", lambda m: 2 * 10**9)
    plain = suggest.suggest(48.0)
    charged = suggest.suggest(48.0, graphs=True)
    assert plain.fit is row and not plain.fit_knife_edge
    assert charged.fit is row and charged.fit_knife_edge


def test_the_engine_decodes_eager_when_its_graphs_would_not_fit(monkeypatch, capsys):
    from drinkme.serving import cudagraph, mtp
    from tests.hotloop_toys import toy_engine

    monkeypatch.setattr(cudagraph, "decide", lambda model, device=None, mode="decode": cudagraph.Decision(
        True, "decode step: CUDA graphs (experimental; DRINKME_CUDA_GRAPHS=0 for eager)"))
    # 0.1 GiB free beside 10 GiB resident: the pool alone leaves under 10%
    monkeypatch.setattr(mtp, "device_headroom_gib", lambda device: (0.1, 10.0))
    eng = toy_engine()
    assert not eng._graphs and eng._graph_bytes == 0
    out = capsys.readouterr().out
    assert "decode step: eager; CUDA graphs would hold 0.10 GiB" in out and "headroom" in out
    # room to spare: graph mode, and its bytes join the slots' charge
    monkeypatch.setattr(mtp, "device_headroom_gib", lambda device: (40.0, 10.0))
    eng = toy_engine()
    assert eng._graphs and eng._graph_bytes == cf.POOL_BYTES * len(eng._slots)
    assert "they hold 98.0 MiB per cache" in capsys.readouterr().out


def test_a_cache_and_its_steps_are_not_a_cycle(monkeypatch):
    """The cache keeps its captured steps; they reach it weakly, so a slot
    that drops its cache drops the graphs and the KV at once, without
    waiting for the cyclic collector."""
    torch = pytest.importorskip("torch")
    from drinkme.serving import cudagraph
    from drinkme.serving.kvcache import LiveStaticCache
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256)
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).eval()
    monkeypatch.setattr(cudagraph.StaticStep, "capture", lambda self, shared=None: None)
    with torch.inference_mode():
        cache = LiveStaticCache(config=model.config, max_cache_len=32)
        model(torch.tensor([[1, 2, 3]]), past_key_values=cache, use_cache=True)
        step = cudagraph.decode_step(model, cache)
        assert step is not None and step.cache is cache
        assert step.forward(step.ids, torch.tensor([[3]]), None).shape == (128,)
    ref = weakref.ref(cache)
    del step
    gc.disable()
    try:
        del cache
        assert ref() is None
    finally:
        gc.enable()
