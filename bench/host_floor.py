"""Host-floor diagnostic — the FIRST thing to measure before any kernel work.

ds4 (C host loop) measured CUDA-graph capture at only +1-2% and concluded
"launch overhead is not the dominant serial-decode cost". Our host is Python +
HF transformers: 5-20us dispatch x 200+ ops/token. The same measurement may
invert for us, so measure before optimizing.

Method: run the REAL decode loop, but replace every compressed Linear's kernel
with a no-op of identical shape and launch geometry. Weights stay resident (so
allocation/VRAM behaviour is unchanged) but no bytes are read and no math runs.
What remains IS the host+framework floor: attention, norms, KV, sampling,
python dispatch, transformers plumbing.

  floor_tok_s          = tok/s with the GEMV stubbed out
  measured_tok_s       = real compressed arm
  host_fraction        = (1/floor) / (1/measured)   <- share of per-token time
                         that is NOT our kernel
Interpretation:
  host_fraction > ~0.15  -> host-side is a real target: graph capture /
                            torch.compile(reduce-overhead) before kernel work.
  host_fraction < ~0.05  -> the gap to the bandwidth ceiling is kernel-shaped;
                            go do rows-per-program + epilogue fusion.

Run:  python bench/host_floor.py [model_name_or_repo]
"""

import json
import sys
import time

import torch

sys.path.insert(0, "src")

from drinkme.arms import N_NEW, free_all, load_cpu, timed_decode  # noqa: E402
from drinkme.codec.swap import CompressedLinear, make_compressed, swap_linears  # noqa: E402
from drinkme.probe import measure_bandwidth  # noqa: E402
from drinkme.suggest import MODELS  # noqa: E402

PROMPT = "The key idea of lossless weight compression is"


class StubbedLinear(CompressedLinear):
    """Identical module, identical buffers, identical output shape — but the
    kernel launch is skipped. Output is garbage on
    purpose; we are timing the harness, not the math."""

    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, self.C)
        outs = torch.zeros(xf.shape[0], self.R, device=x.device, dtype=torch.float32)
        out = outs.to(x.dtype).reshape(*shape[:-1], self.R)
        if self.bias is not None:
            out = out + self.bias
        return out


def make_stubbed(w, bias, device: str = "cuda"):
    """Build the REAL module (identical packing, buffers, residency), then
    rebind its class so forward() skips the kernel."""
    mod, stat = make_compressed(w, bias, device)
    mod.__class__ = StubbedLinear
    return mod, stat


def main(which: str = "Qwen3-8B"):
    by = {m.name: m for m in MODELS} | {m.hf_repo: m for m in MODELS}
    m = by[which]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(m.hf_repo, revision=m.revision)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    out = {"model": m.hf_repo, "revision": m.revision,
           "gpu": torch.cuda.get_device_name(0), "n_new": N_NEW}
    out["bandwidth"] = measure_bandwidth()

    with torch.inference_mode():
        # --- real compressed arm ---
        model = load_cpu(m.hf_repo, m.revision)
        stats = swap_linears(model, make_compressed)
        model.cuda()
        free_all()
        _, real = timed_decode(model, ids)
        out["compressed_tok_s"] = real
        out["swapped_linears"] = len(stats)
        model = None
        free_all()

        # --- stubbed arm: same everything, no kernel ---
        model = load_cpu(m.hf_repo, m.revision)
        swap_linears(model, make_stubbed)
        model.cuda()
        free_all()
        _, floor = timed_decode(model, ids)
        out["floor_tok_s"] = floor
        model = None
        free_all()

    import statistics as st

    r, f = st.median(real), st.median(floor)
    out["compressed_median"] = r
    out["floor_median"] = f
    out["host_fraction"] = round((1 / f) / (1 / r), 4) if f else None
    out["host_ms_per_token"] = round(1000 / f, 2) if f else None
    out["total_ms_per_token"] = round(1000 / r, 2)
    out["verdict"] = (
        "host-side is a real target (graph capture / torch.compile first)"
        if out["host_fraction"] and out["host_fraction"] > 0.15
        else "gap is kernel-shaped (rows-per-program, epilogue fusion)"
        if out["host_fraction"] and out["host_fraction"] < 0.05
        else "mixed — both host and kernel work worth doing"
    )
    print(json.dumps(out, indent=2))
    with open("/tmp/drinkme_host_floor.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:])
