"""`drinkme check`: an eligibility verdict for ANY HF repo, from kilobytes.

The --repo escape hatch (the menu curates, check bounces)
must refuse BEFORE a 16GB download, naming the exact failed condition —
the refuse-with-arithmetic doctrine as a verb. Everything here reads
only `config.json` and the safetensors HEADERS (huggingface_hub's
get_safetensors_metadata — per-tensor dtype/shape, zero weight bytes).

The check PREDICTS, pack PROVES: the estimate below approximates which
tensors the swap will touch (codec/pack.header_eligible: 2D, shape gates,
minus embeddings and a tied head — the SAME predicate refuse_checkpoint
applies at every loader's door), and the source-dtype contract over them is
the same function too (codec/pack.source_dtype_refusal_reason: an eligible
tensor stored as anything but BF16 refuses the checkpoint by name); the
encoder's round trip remains the bit-exact per-tensor gate at pack time.

Two failure planes, deliberately distinct (field-guide lesson: a broken
checker must not imitate a failed check):
  - the CHECK fails  -> Verdict(ok=False, reason=...) — a refusal, block.
  - the CHECKER fails (network down, HF unreachable) -> CheckUnavailable
    — callers warn and proceed, so an offline machine with cached weights still
    packs. An unmeasurable condition must fail toward the working path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .codec.pack import (checkpoint_refusal_reason, header_eligible, source_dtype_refusal_reason,
                         unsupported_format)

# meanBpw of the validated Qwen3-8B pack / raw bf16 — an ESTIMATE for the
# verdict line; every real pack reports its own measured number.
PACKED_RATIO = 12.014 / 16.0
GIB = 1024**3

# dtype strings as they appear in safetensors headers
_BF16 = "BF16"
_RAW_OK = {"F32", "F16", "BF16"}  # floats outside the packed set ride raw; ints mean quantized


class CheckUnavailable(Exception):
    """The checker could not run (network/transient) — warn and proceed."""


@dataclass
class Verdict:
    repo: str
    revision: str | None
    ok: bool
    reason: str | None  # exactly one named condition when not ok
    total_gib: float = 0.0
    bf16_pct: float = 0.0        # of checkpoint mass
    packable_pct: float = 0.0    # est. mass the swap will compress
    est_packed_gib: float = 0.0  # packable*ratio + rest raw
    n_tensors: int = 0
    n_packable: int = 0
    notes: list[str] = field(default_factory=list)
    # THIS MACHINE's encoder, not the repo's — set by check() (radix_native.diagnose()),
    # left unset (None/None) by the pure assess(): whether packing this repo would
    # actually run, distinct from whether the repo itself is eligible (`ok`)
    encoder_compiler: str | None = None   # the compiler path, when available
    encoder_version: str | None = None    # its --version first line, when available
    encoder_problem: str | None = None    # radix_native.refusal_message(), when not

    def line(self) -> str:
        if not self.ok:
            return f"check {self.repo}: refused — {self.reason}" + self._encoder_line()
        # the fields are GiB (the JSON keeps them); the line speaks decimal GB,
        # like every other size drinkme prints for a person
        return (f"check {self.repo}: eligible — {self.packable_pct:.1f}% of "
                f"{self.total_gib * GIB / 1e9:.2f} GB packs, est. "
                f"{self.est_packed_gib * GIB / 1e9:.2f} GB"
                + (f" ({'; '.join(self.notes)})" if self.notes else "")
                + self._encoder_line())

    def _encoder_line(self) -> str:
        """Appended by line() when check() probed this machine's encoder AND
        found it available (assess() alone leaves both fields unset, so the
        pure verdicts in tests/test_check.py print exactly as before). The
        UNavailable case (encoder_problem) is deliberately NOT here: that is
        a can't-pack-it-HERE fact about this machine, not part of the
        verdict about the repo, so cli.py prints it separately, on stderr —
        stdout stays the verdict alone, valid JSON in --json mode too."""
        if self.encoder_problem is not None:
            return ""
        if self.encoder_compiler is not None:
            return f"\nencoder: native ({self.encoder_compiler}, {self.encoder_version})"
        return ""


def _dtype_nbytes(dtype: str) -> int:
    return {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
            "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}.get(dtype, 2)


def assess(repo: str, revision: str | None, config: dict,
           tensors: dict[str, tuple[str, list[int]]]) -> Verdict:
    """Pure verdict from config + {name: (dtype, shape)}. No network."""
    if not tensors:
        return Verdict(repo, revision, False,
                         "no safetensors tensors found — drinkme reads "
                         "safetensors checkpoints (not .bin/GGUF)")
    # THE DOOR, first: an FP8 checkpoint (a float8 plane in the headers, or
    # an fp8 quantization_config) and any other named quantization scheme
    # (compressed-tensors, gptq, awq, bitsandbytes — the int4/fp4 family)
    # are refused BY NAME here, before the byte-share heuristic below gets
    # a chance to say only "quantized" without saying what — the same line
    # `drinkme pack` prints (codec/pack.refusal_for_checkpoint).
    reason = checkpoint_refusal_reason(config, (d for d, _ in tensors.values()))
    if reason is not None:
        return Verdict(repo, revision, False, reason)

    import math

    def by_bytes(d: str, s: list[int]) -> int:
        return _dtype_nbytes(d) * math.prod(s or [0])

    total = sum(by_bytes(d, s) for d, s in tensors.values())
    if total == 0:
        return Verdict(repo, revision, False, "empty checkpoint")

    bf16 = sum(by_bytes(d, s) for d, s in tensors.values() if d == _BF16)
    ints = sum(by_bytes(d, s) for d, s in tensors.values() if d not in _RAW_OK)
    if ints > 0.5 * total:
        return Verdict(repo, revision, False,
                         "quantized checkpoint (majority non-float tensors, no "
                         "quantization_config naming the scheme) — drinkme "
                         f"compresses BF16 weights, not quants. "
                         f"{unsupported_format('an unnamed quantized checkpoint')}")
    # THE SOURCE-DTYPE CONTRACT, the same decision `drinkme pack` and every
    # loader's door make (codec/pack.refuse_checkpoint): a tensor the codec
    # would pack that is stored as F16, F32 — anything but BF16 — refuses
    # the checkpoint by name. Tensors outside the packed set ride raw.
    reason = source_dtype_refusal_reason(((n, d) for n, (d, _s) in tensors.items()),
                                         lambda n: header_eligible(n, tensors[n][1], config))
    if reason is not None:
        return Verdict(repo, revision, False, reason)

    notes: list[str] = []
    packable = 0
    n_pack = 0
    for name, (dtype, shape) in tensors.items():
        if not header_eligible(name, shape, config):
            continue  # a checkpoint that omits a tied lm_head owes nothing
                      # (never assert an export detail as a correctness property)
        packable += 2 * int(shape[0]) * int(shape[1])
        n_pack += 1
    if n_pack == 0:
        return Verdict(repo, revision, False,
                         "no tensors pass the shape gates (2D BF16, cols%4==0, "
                         "min dim >= 1024) — nothing for the codec to hold onto")

    arch = (config.get("architectures") or [None])[0]
    if arch:
        try:
            import transformers

            if getattr(transformers, arch, None) is None:
                notes.append(f"architecture {arch} not in installed "
                             f"transformers {transformers.__version__} — serve "
                             "may need an upgrade (pack is architecture-blind)")
        except Exception:
            pass  # no transformers here (pure-pack machine) — not the check's call

    est = packable * PACKED_RATIO + (total - packable)
    # 3D bf16 tensors are not Linears, so they ride raw. Name what they are:
    # [C, 1, K] is a depthwise conv's weight (a dense hybrid's DeltaNet
    # conv1d), anything else a fused or MoE expert layout.
    three_d = [s for d, s in tensors.values() if d == _BF16 and len(s) == 3]
    conv = sum(1 for s in three_d if s[1] == 1)
    if conv:
        notes.append(f"{conv} conv weights [C, 1, K] ride raw")
    if len(three_d) - conv:
        notes.append(f"{len(three_d) - conv} 3D bf16 tensors (fused/MoE layout) ride raw")
    return Verdict(repo, revision, True, None,
                     total_gib=total / GIB,
                     bf16_pct=100 * bf16 / total,
                     packable_pct=100 * packable / total,
                     est_packed_gib=est / GIB,
                     n_tensors=len(tensors), n_packable=n_pack, notes=notes)


def check(repo: str, revision: str | None = None) -> Verdict:
    """Fetch the kilobytes, then assess (_check), and attach THIS MACHINE's
    encoder status (radix_pack.encoder_status — no network, no torch) to
    whatever verdict comes back: a repo can be eligible on a machine that
    cannot currently pack it, and vice versa, so the two travel together
    but neither is the other. An explicit DRINKME_RADIX_ENCODER=numpy
    reports nothing here — packing will proceed either way."""
    return _attach_encoder_status(_check(repo, revision))


def _attach_encoder_status(v: Verdict) -> Verdict:
    """THIS MACHINE's encoder status (radix_pack.encoder_status — no
    network, no torch), attached to an already-computed verdict. Split out
    from check() so tests can exercise every encoder state (available,
    no compiler, a compiler that fails, numpy opted in) without touching
    the network _check does."""
    from .codec.radix_native import refusal_message
    from .codec.radix_pack import encoder_status

    kind, diag = encoder_status()
    if kind == "native":
        v.encoder_compiler, v.encoder_version = diag.compiler, diag.version
    elif kind == "refuse":
        v.encoder_problem = refusal_message(
            diag, lead_in="drinkme check: this machine can't pack it yet")
    return v


def _check(repo: str, revision: str | None = None) -> Verdict:
    """Fetch the kilobytes, then assess. Network problems raise
    CheckUnavailable; definite conditions return a refusal."""
    from .detect import force_ipv4

    force_ipv4()  # the IPv6/HF-CDN blackhole (detect.force_ipv4's own docstring)
    try:
        from huggingface_hub import get_safetensors_metadata, hf_hub_download
        from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError,
                                            NotASafetensorsRepoError,
                                            RepositoryNotFoundError)
    except ImportError as e:  # pragma: no cover
        raise CheckUnavailable(f"huggingface_hub unavailable: {e}")

    try:
        import json

        cfg_path = hf_hub_download(repo, "config.json", revision=revision)
        config = json.load(open(cfg_path))
    except GatedRepoError:
        return Verdict(repo, revision, False,
                         f"license not accepted — accept it at hf.co/{repo}, then retry "
                         "(drinkme never clicks through for you)")
    except RepositoryNotFoundError:
        return Verdict(repo, revision, False, "repo not found on HF (typo, or "
                         "private without a token)")
    except EntryNotFoundError:
        return Verdict(repo, revision, False, "no config.json — not a "
                         "transformers-loadable checkpoint")
    except Exception as e:
        raise CheckUnavailable(f"could not reach HF for config.json: {e}")

    try:
        md = get_safetensors_metadata(repo, revision=revision)
        tensors = {name: (t.dtype, list(t.shape))
                   for f in md.files_metadata.values()
                   for name, t in f.tensors.items()}
    except NotASafetensorsRepoError:
        return Verdict(repo, revision, False,
                         "no safetensors in this repo — drinkme reads "
                         "safetensors checkpoints (not .bin/GGUF)")
    except Exception as e:
        raise CheckUnavailable(f"could not read safetensors headers: {e}")

    return assess(repo, revision, config, tensors)
