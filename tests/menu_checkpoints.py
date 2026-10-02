"""The menu checkpoints' safetensors bytes, for the suite.

suggest.checkpoint_bytes reads a checkpoint's size from its metadata (the
local Hugging Face cache, else the Hub). A test must depend on neither what
this machine has cached nor the network, so conftest.py answers it from
CHECKPOINT_BYTES instead. Each figure is the sum of dtype x shape over the
checkpoint's safetensors headers at the menu row's revision (main for the
unpinned rows), every row a BF16 checkpoint, read 2026-09-26.
"""

CHECKPOINT_BYTES = {
    "Qwen/Qwen3.8-27B": 55_562_855_904,
    "Qwen/Qwen3-8B": 16_381_470_720,
    "google/gemma-4-31B-it": 62_546_177_752,
    "meta-models/Muse-Glimmer-30B": 59_553_253_376,
    "ibm-granite/granite-4.2-8b": 17_583_185_920,
    "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B": 18_819_627_488,
    "Qwen/Qwen3-0.6B": 1_503_264_768,
    "Qwen/Qwen3-1.7B": 4_063_479_808,
    "Qwen/Qwen3-4B": 8_044_936_192,
    "Qwen/Qwen3-14B": 29_536_614_400,
    "Qwen/Qwen3-32B": 65_524_246_528,
    "Qwen/Qwen2.5-72B-Instruct": 145_412_407_296,
}

# The suite's toy row, ToyModel (toy/model), which several bench tests run
# through bench.run: a 1 GB checkpoint.
TOY_CHECKPOINT_BYTES = {"toy/model": 1_000_000_000}


def pin_sizes(monkeypatch, table: dict):
    """Pin suggest.sizes for the rows `table` names: {row name: (bf16_gb,
    comp_gb)}, read at each call, so a test's toy-row helper can add to it
    after this is installed. Each such row answers exactly those sizes, the
    compressed one an estimate, so a test of what bench does with a size can
    name the size. Every other row goes through the real suggest.sizes."""
    from drinkme import suggest

    real = suggest.sizes

    def fake(m, profile=None, vision=True, pack_dir=None):
        if m.name in table:
            return suggest.Sizes(*table[m.name], profile or "sip")
        return real(m, profile, vision, pack_dir)

    monkeypatch.setattr(suggest, "sizes", fake)
