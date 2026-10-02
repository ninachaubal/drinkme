#!/usr/bin/env python3
"""Build the toy Qwen3 radix pack the MLX-lane tests load (a torch box's
job; the Mac has no torch): a genuine 2-layer Qwen3ForCausalLM (hidden
1024 so q/o/gate/up/down clear codec eligibility; kv_heads=2 keeps k/v at
512 wide so they stay unpacked) with a WordLevel tokenizer, saved to
`<out>/model/`, packed through pack_model into `<out>/pack/` — a radix
pack of 10 tensors. tests/test_serving_engine_mlx.py builds the same
toy in-process where torch exists and takes this one through
DRINKME_RADIX_MLX_TOY where it does not; tests/test_metal_radix.py's
end-to-end test reads the same variable.

    PYTHONPATH=src python bench/radix_mlx_toy_build.py --out /path/to/radix_toy
    # ship <out>/ to the Mac; there: DRINKME_RADIX_MLX_TOY=<out> pytest tests/test_metal_radix.py ...

`--bias` builds the BIASED twin (attention_bias=True: q/k/v/o carry
nonzero biases — drawn, because HF's _init_weights zeroes every Linear
bias and a zero bias exercises no epilogue). q and o are coded at hidden
1024, so a RadixLinear with a bias runs on both the fused and the
reference path; k and v stay unpacked, so the bias that arrives for a
Linear mlx-lm built without one runs too. The tests look for it under
`<toy>/bias/` (the same three commands, one env var):

    PYTHONPATH=src python bench/radix_mlx_toy_build.py --out <toy> --reference
    PYTHONPATH=src python bench/radix_mlx_toy_build.py --out <toy>/bias --bias --reference
    rsync -ac -e 'ssh -4' <toy> <m4-mac>:workspace/

`--reference` writes `<out>/reference.npz` from the TORCH CPU engine —
the control the mlx tests compare against where torch exists, carried
along to the box where it does not: for PROMPTS through the toy's chat
template, the prompt ids, the prompt-position logits (float32), and the
greedy continuation for MAX_TOKENS tokens (ids + the logits row each was
sampled from), plus the digests of the checkpoint bytes and the pack's
recorded manifest, so a reference built for other bytes is REFUSED by
load_reference rather than silently used. Built with the engine's own
sampling defaults and eos set (the toy's generation_config.json) — what
the mlx engine reads too.

`--identity` writes `<out>/identity/`: the pack-
identity probe (tests/test_pack_identity.py's module docstring) —
two toy Llama checkpoints, same shapes and tokenizer, different weights,
in a fabricated HF-cache layout at `<out>/identity/hubcache/` (`source/A`
at commit IDENTITY_AAA, `other/B` at IDENTITY_BBB — a hub-kind identity,
exactly what a real pack's `source` block records), plus a pack cut from
A into `<out>/identity/pack-A/` whose meta.json binds it to
`source/A@IDENTITY_AAA`. tests/test_pack_identity.py's mlx test loads this
by name to check the Metal loader refuses a pack served against the wrong
checkpoint; the file's torch-backed tests build the identical checkpoints
in-process via build_identity_checkpoint (the same function, so the toy is
never duplicated) under their own richer fake_hub fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")

# The reference's prompts: a plain turn, a system turn, a short one, a
# multi-turn — each through the toy's chat template with the generation
# prompt appended.
PROMPTS = [
    [{"role": "user", "content": "hello world the quick brown fox"}],
    [{"role": "system", "content": "alpha beta gamma"},
     {"role": "user", "content": "the lazy dog jumps over"}],
    [{"role": "user", "content": "delta epsilon"}],
    [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"},
     {"role": "user", "content": "quick brown fox jumps"}],
]
MAX_TOKENS = 16
REFERENCE = "reference.npz"


def build_toy_qwen3(model_dir: str, pack_dir: str, bias: bool = False) -> tuple[str, str]:
    """The recipe. Returns (model_dir, pack_dir); asserts the pack is the
    10-tensor radix pack the tests expect. `bias`: attention_bias=True with
    the biases drawn nonzero (module docstring); False is byte-identical
    to the toy before the flag existed."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from drinkme.codec.pack import pack_model
    from drinkme.codec.swap import eligible_linears

    words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over",
             "lazy", "dog", "alpha", "beta", "gamma", "delta", "epsilon",
             "user", "assistant", "system", ":"]
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in words:
        vocab[w] = len(vocab)
    while len(vocab) < 64:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    tok.save_pretrained(model_dir)

    cfg = Qwen3Config(vocab_size=64, hidden_size=1024, intermediate_size=1024,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=256,
                      max_position_embeddings=256, tie_word_embeddings=False,
                      eos_token_id=2, pad_token_id=1, rope_theta=10000.0,
                      attention_bias=bias)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for _, _, _, child in eligible_linears(model):
            child.weight.data.mul_(1 / 64)  # a real exponent spread
        if bias:
            # HF's _init_weights zeroes every Linear bias; drawn at the scale
            # of the scaled projections' own outputs (~0.01 — the accumulator
            # and the bias both carry bits into the epilogue's rounding; at
            # 0.02 the o_proj bias, a constant on the residual, made every
            # continuation one word)
            n_biased = 0
            for m in model.modules():
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
                    n_biased += 1
            assert n_biased == 8, n_biased  # q/k/v/o, two layers
    model.save_pretrained(model_dir, safe_serialization=True)
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    assert meta["tensorCount"] == 10 and meta["radixTensorCount"] == 10, meta
    return model_dir, pack_dir


# --------------------------------------------------- the identity pair ----

IDENTITY_AAA = "a" * 40
IDENTITY_BBB = "b" * 40


def _identity_snapshot(cache: str, repo: str, sha: str) -> str:
    """The fabricated HF-cache layout tests/test_pack_identity.py's
    checkpoints live in: <cache>/models--org--name/snapshots/<sha>/ — the
    same shape a real pack's identity records (codec/identity.py's hub
    kind, hub_revision_of's regex)."""
    return os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots", sha)


def build_identity_checkpoint(model_dir: str, seed: int) -> None:
    """A toy Llama checkpoint at `model_dir` for the pack-identity
    probe (tests/test_pack_identity.py's module docstring): a genuine
    1-layer LlamaForCausalLM plus a WordLevel tokenizer, saved to disk.
    `seed` is the only thing that varies between two calls — same
    architecture, same tokenizer, different weights, the "compatible
    shapes" a pack could be loaded across. Shared by that
    file's `two_models` fixture and build_identity_pack below, so every
    caller builds the identical toy rather than a second copy of it."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from drinkme.codec.swap import eligible_linears

    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in ["hello", "world", "the", "quick", "brown", "fox", "user", "assistant", ":"]:
        vocab[w] = len(vocab)
    while len(vocab) < 64:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    tok.save_pretrained(model_dir)
    cfg = LlamaConfig(vocab_size=64, hidden_size=1024, intermediate_size=1024,
                      num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=False, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for _, _, _, child in eligible_linears(model):
            child.weight.data.mul_(1 / 64)
    model.save_pretrained(model_dir, safe_serialization=True)


def identity_paths(root: str) -> dict | None:
    """The prebuilt identity fixture under `root` (build_identity_pack's
    layout, at `<root>/identity/`) — {"cache", "a", "b", "pack_a"} if
    `root` carries one (its pack's meta.json exists), else None."""
    d = os.path.join(root, "identity")
    pack_a = os.path.join(d, "pack-A")
    if not os.path.isfile(os.path.join(pack_a, "meta.json")):
        return None
    cache = os.path.join(d, "hubcache")
    return {"cache": cache, "a": _identity_snapshot(cache, "source/A", IDENTITY_AAA),
            "b": _identity_snapshot(cache, "other/B", IDENTITY_BBB), "pack_a": pack_a}


def build_identity_pack(out_dir: str) -> dict:
    """source/A@IDENTITY_AAA and other/B@IDENTITY_BBB under a fabricated hub
    cache at `<out_dir>/hubcache/` (build_identity_checkpoint), plus a pack
    cut from A into `<out_dir>/pack-A/` whose meta.json binds it to
    source/A@IDENTITY_AAA. pack_model resolves the checkpoint through
    checkpoint.snapshot_dir, so it is redirected to the fabricated cache for
    the one call and restored after — there is no pytest monkeypatch here;
    this also runs from the CLI. Returns {"cache", "a", "b", "pack_a"}
    (identity_paths' shape)."""
    from drinkme.codec.pack import pack_model
    from drinkme.serving import checkpoint

    cache = os.path.join(out_dir, "hubcache")
    a = _identity_snapshot(cache, "source/A", IDENTITY_AAA)
    b = _identity_snapshot(cache, "other/B", IDENTITY_BBB)
    build_identity_checkpoint(a, seed=1)
    build_identity_checkpoint(b, seed=2)

    pack_dir = os.path.join(out_dir, "pack-A")
    table = {("source/A", IDENTITY_AAA): a, ("source/A", None): a, ("source/A", "main"): a}

    def resolve(repo, revision=None):
        return repo if os.path.isdir(repo) else table[(repo, revision)]

    orig = checkpoint.snapshot_dir
    checkpoint.snapshot_dir = resolve
    try:
        pack_model("source/A", IDENTITY_AAA, pack_dir, progress=lambda *_: None)
    finally:
        checkpoint.snapshot_dir = orig
    return {"cache": cache, "a": a, "b": b, "pack_a": pack_dir}


# ------------------------------------------------------- the reference ----


def toy_digests(model_dir: str, pack_dir: str) -> dict:
    """What binds a reference to a toy, torch-free: sha256 over every file
    of the checkpoint dir (sorted names, name and bytes — the biases and
    the config are RAW tensors and settings the pack's own source digest
    covers only by header), and the pack's manifest digest
    (codec/identity.manifest_digest, recomputed from the live maps)."""
    from drinkme.codec.identity import manifest_digest

    h = hashlib.sha256()
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            continue
        h.update(name.encode("utf-8") + b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    with open(os.path.join(pack_dir, "meta.json")) as f:
        meta = json.load(f)
    return {"checkpoint_sha256": h.hexdigest(), "pack_manifest_sha256": manifest_digest(meta)}


class Recorder:
    """Wraps an engine module's `sample_next` so a generate() call leaves
    behind the id it picked at every step and the logits row it picked it
    from — the engine's own loop, observed, not re-implemented. Works on
    either engine module (serving.engines binds sample_next by name;
    serving.engine_mlx defines it); `to_row` turns the module's logits
    (a torch tensor or a numpy row) into a float32 numpy copy."""

    def __init__(self, module, to_row):
        self.module, self.to_row = module, to_row
        self.ids: list[int] = []
        self.rows: list[np.ndarray] = []

    def __enter__(self):
        self._orig = self.module.sample_next

        def rec(logits, *a, **kw):
            row = self.to_row(logits)  # BEFORE the pick: the row as sampled
            t = self._orig(logits, *a, **kw)
            self.ids.append(int(t))
            self.rows.append(row)
            return t

        self.module.sample_next = rec
        return self

    def __exit__(self, *exc):
        self.module.sample_next = self._orig
        return False


def torch_transcript(eng, prompts=PROMPTS, max_tokens: int = MAX_TOKENS) -> list[dict]:
    """The torch engine's answer on each prompt: prompt ids, the prompt-
    position logits (the model's forward on the ids, float32), and the
    greedy continuation — ids and rows — through the engine's own
    generate(). Speculation must be OFF (DRINKME_SPEC=off before the
    engine's first generate — the plan is resolved once, there): the
    verify batch would call sample_next on batched rows the serial loop
    never samples. Refused loudly otherwise, not recorded wrong."""
    import torch

    from drinkme.serving import engines, mtp
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    plan = eng._spec_plan or mtp.spec_plan(eng.mtp_head is not None, mtp.depth_from_env())
    if plan.mode != "off":
        raise RuntimeError(f"torch_transcript needs the serial loop and this engine's speculation "
                           f"plan is {plan.mode!r}: set DRINKME_SPEC=off before its first generate")
    out = []
    for msgs in prompts:
        ids = eng.tokenize(messages=msgs)
        with torch.inference_mode():
            prompt_logits = eng.model(torch.tensor([ids])).logits[0, -1].float().cpu().numpy().copy()
        with Recorder(engines, lambda t: t.detach().float().cpu().numpy().copy()) as rec:
            res = complete(eng, GenerationRequest(
                msgs, SampleParams(temperature=0.0, max_tokens=max_tokens)))
        out.append({"ids": np.asarray(ids, dtype=np.int64), "prompt_logits": prompt_logits,
                    "gen_ids": np.asarray(rec.ids, dtype=np.int64),
                    "step_logits": np.stack(rec.rows).astype(np.float32),
                    "text": res.text, "finish": res.finish_reason})
    return out


def write_reference(path: str, model_dir: str, pack_dir: str) -> dict:
    """`path` <- the torch CPU engine's transcript of PROMPTS on this toy
    (the control tests/test_serving_engine_mlx.py loads where torch exists:
    load_compressed on cpu, no prefix cache), bound to the toy's digests.
    Returns what was written (load_reference's shape)."""
    import torch
    import transformers

    from drinkme.serving import gen_config
    from drinkme.serving.engines import load_compressed

    saved = {k: os.environ.get(k) for k in ("DRINKME_PREFIX_SLOTS", "DRINKME_SPEC")}
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    os.environ["DRINKME_SPEC"] = "off"  # the serial loop, one sample_next per token
    try:
        eng = load_compressed(model_dir, None, pack_dir, device="cpu")
        rows = torch_transcript(eng)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    payload = {
        "reference_version": np.int64(1),
        "prompts": json.dumps(PROMPTS),
        "max_tokens": np.int64(MAX_TOKENS),
        "eos_ids": np.asarray(sorted(eng.eos_ids), dtype=np.int64),
        "sampling_defaults": json.dumps(gen_config.effective_defaults(eng.sampling_defaults),
                                        sort_keys=True),
        "built_with": json.dumps({"torch": torch.__version__,
                                  "transformers": transformers.__version__}),
        **toy_digests(model_dir, pack_dir),
    }
    for i, r in enumerate(rows):
        for k, v in r.items():
            payload[f"{k}_{i}"] = v
    np.savez(path, **payload)
    return load_reference(path, model_dir, pack_dir)


def load_reference(path: str, model_dir: str, pack_dir: str) -> dict:
    """The reference at `path`, torch-free, REFUSED (ValueError, naming the
    digest) unless it was built for exactly this checkpoint and this pack.
    Returns {"prompts", "max_tokens", "eos_ids", "sampling_defaults",
    "built_with", "rows": [torch_transcript's dicts]}."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    want = toy_digests(model_dir, pack_dir)
    for k, v in want.items():
        got = str(d[k])
        if got != v:
            raise ValueError(f"{path} is a reference for another toy: {k} {got[:16]}… "
                             f"but this toy's is {v[:16]}… — rebuild it with --reference")
    prompts = json.loads(str(d["prompts"]))
    rows = []
    for i in range(len(prompts)):
        rows.append({"ids": d[f"ids_{i}"], "prompt_logits": d[f"prompt_logits_{i}"],
                     "gen_ids": d[f"gen_ids_{i}"], "step_logits": d[f"step_logits_{i}"],
                     "text": str(d[f"text_{i}"]), "finish": str(d[f"finish_{i}"])})
    return {"prompts": prompts, "max_tokens": int(d["max_tokens"]),
            "eos_ids": [int(e) for e in d["eos_ids"]],
            "sampling_defaults": json.loads(str(d["sampling_defaults"])),
            "built_with": json.loads(str(d["built_with"])), "rows": rows}


def ulp(row: np.ndarray) -> float:
    """One bf16 ulp at the row's scale — tests/test_serving_engine_mlx.py's
    tolerance unit for two engines' logits (both round to bf16 at every
    Linear; two accumulation orders agree to a few of these)."""
    return float(2.0 ** (np.floor(np.log2(max(float(np.abs(row).max()), 1.0))) - 7))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory to hold model/ and pack/")
    ap.add_argument("--bias", action="store_true",
                    help="the biased twin: attention_bias=True, biases drawn nonzero")
    ap.add_argument("--reference", action="store_true",
                    help=f"also write <out>/{REFERENCE} from the torch CPU engine")
    ap.add_argument("--identity", action="store_true",
                    help="also write <out>/identity/: two toy "
                         "checkpoints + a pack cut from one, for test_pack_identity.py's "
                         "mlx test. Combines with a plain --out (no --bias); the bias "
                         "twin is a separate --out <toy>/bias invocation and has no "
                         "identity fixture of its own.")
    a = ap.parse_args()
    model_dir, pack_dir = os.path.join(a.out, "model"), os.path.join(a.out, "pack")
    os.makedirs(model_dir, exist_ok=True)
    build_toy_qwen3(model_dir, pack_dir, bias=a.bias)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    print(f"toy radix pack: {pack_dir}  tensors {meta['tensorCount']}  profile {meta['profile']}  "
          f"meanBpw {meta['meanBpw']}  formatVersion {meta['formatVersion']}"
          f"{'  attention_bias=True' if a.bias else ''}")
    if a.reference:
        ref_path = os.path.join(a.out, REFERENCE)
        ref = write_reference(ref_path, model_dir, pack_dir)
        n = sum(len(r["gen_ids"]) for r in ref["rows"])
        print(f"reference: {ref_path}  prompts {len(ref['rows'])}  greedy tokens {n}  "
              f"eos {ref['eos_ids']}  built with {ref['built_with']}")
    if a.identity:
        paths = build_identity_pack(os.path.join(a.out, "identity"))
        print(f"identity fixture: {paths['pack_a']}  "
              f"(source/A@{IDENTITY_AAA[:8]}… vs other/B@{IDENTITY_BBB[:8]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
