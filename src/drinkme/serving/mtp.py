"""MTP speculative decoding: Qwen3.8-27B's own trained multi-token-prediction
head as a draft model (drafts only propose; the trunk decides).

The head is 15 tensors that ship in the checkpoint (`mtp.*`, ~0.4B params) and
that the text skeleton throws away (arms.ckpt_to_skel returns None for them).
It predicts t+2 from the trunk's hidden state at t and the embedding of t+1.
Draft k tokens with it, then run the MAIN model ONCE over the k+1 positions and
keep the longest prefix where the main model's own greedy pick agrees: every
emitted token is the main model's own pick at its row, for ~1.9x fewer
weight-reads per token. The batched verify forward and the serial
single-token one can round differently in accumulation order, so the stream
agrees with the one-token-at-a-time loop except at near-ties
(tests/spec_agree.py). That composes with the codec: the pack cuts the bytes
each weight-read moves, speculation cuts the weight-reads per token.

SAMPLED REQUESTS: at
temperature > 0 the same cycle runs REJECTION SAMPLING instead of the argmax
compare — the draft SAMPLES each token from its own transformed distribution
q, the verify pass yields the target distribution p for every row through
the SAME transform (sampling.sample_probs: penalties with the history the
serial loop would have had at that position, temperature, top-k, top-p),
and serving/speculative.py accepts draft i with probability min(1, p/q),
resamples the residual at the first rejection, and draws the bonus token
from p_depth when every draft survives. Exact by theorem: every emitted
token is distributed as the serial sampler's at that position, for any
draft. So the sampled path is distribution-identical to serial sampling, the
greedy path takes the trunk's argmax at every row, and the two paths share
everything but the decision.

ON BY DEFAULT when the model allows: unset = AUTO, which
resolves to depth 4 iff the checkpoint carries an MTP head and it loads.
`DRINKME_SPEC=off` (or `--spec off`) is the off switch, for every proposer;
`DRINKME_MTP_DEPTH=k` pins a depth and keeps loud failures.

A SECOND PROPOSER (n-gram speculation): serving/ngram.py drafts by looking the
last few tokens up in the request's own context, with no head and no
weights. It does NOT bring its own verify — it hands drafts to the very
Speculator below, which is why `head` and `lookup` are both optional here
and why the accept/commit/rewind code reads the same whichever one proposed.
DRINKME_SPEC picks (ngram.resolve_mode's table); AUTO resolves to this head
when the checkpoint carries one and to the lookup when it does not, so a
head-less model gets speculation for the first time and a head-ful one is
unchanged. `spec_plan` reads all of it once per engine.

RESIDENCY GUARD (D2, decided guard 1 — FIT OUTRANKS SPEED): AUTO only,
never a forced depth. Before loading the head, `_mtp_head` (engines.py)
checks whether doing so would leave less than the fit check's own margin
(`suggest.FIT_HEADROOM`, read there rather than invented here) of headroom
on the device already holding the model; on a knife-edge fit MTP yields —
not loaded, one stderr line naming the margin, `DRINKME_MTP_DEPTH=4` still forces
it (that person asked). `device_headroom_gib` returns None off a real
accelerator (cpu, mps), so the guard is a no-op there by construction.

ADAPTIVE BAIL (decided guard 2): rolling acceptance over the last
BAIL_WINDOW DRAFTED positions below BAIL_FLOOR sends the request to the
serial loop, with one stderr line naming the rate. The denominator is
drafted positions — every position a rejected cycle did not reach counts
as rejected (engines.py D4) — so a request whose decided acceptance is
~70% reads ~47% in the window. Break-even is 6.8%; 8% is the decided
number. Reported in the per-request accounting (`Speculator.stats`). Both
numbers are DEFAULTS: `DRINKME_MTP_BAIL_WINDOW` / `DRINKME_MTP_BAIL_FLOOR`
(bail_from_env, read once into the SpecPlan like the re-arm below) move
them per engine, so the trigger can be set on evidence
(bench/bail_sweep.sh: the shipped 32/0.15 tripped 70% of ordinary sampled free-text
256-token runs at 6-12% over the window; 64/0.08 tripped 0 of 40 runs
across both swept prompts, at +15% decode tok/s over 32/0.15 — a hostile
prompt was not found among the two swept, so 64/0.08 is the default and
32/0.15 stays reachable through the env knobs).

THE BAIL RE-ARMS. As a one-way latch for the REST of the request a trip
costs too much: on the 27B an ordinary sampled 256-token answer trips it
~30-40 tokens in on 2 of 3 runs — a low-acceptance REGION (the window's
denominator is drafted positions, so a steady 71% decided-acceptance reads
~47% and a bad stretch dips under 15%) — and the remaining ~220 tokens
would decode at spec-off speed: 1.1 tok/step vs 2.8 when it holds. So a
trip means serial for REARM_DEFAULT tokens, then a
FRESH window and drafting again; a window that trips again bails again for
twice as long (cap REARM_CAP), so a genuinely hostile request converges to
~serial at a bounded probe cost, and a re-armed window that survives
REARM_SURVIVE_WINDOWS windows' worth of drafted positions (REARM_SURVIVE at
the default window) resets the backoff. `DRINKME_MTP_REARM=0` is a one-way
latch. While serial with a re-arm ahead the engine's decode step
goes through `Speculator.serial_step` — the same M=1 trunk forward, read
through forward_with_hidden so the trunk hidden of every serial position is
KEPT — and `rearm` hands those rows to the head in one batched pass (entry
p = (h_p, t_{p+1}), as at prefill), so the head's attention history has no
hole and no extra trunk forward is ever paid.

WHAT WAS VERIFIED, AND AGAINST WHAT (do not re-derive by guessing)

* `h` is the trunk's POST-final-norm hidden state — the same tensor lm_head
  consumes. Read off llama.cpp's qwen35 graph (src/models/qwen35.cpp): the
  main graph sets `res->t_h_nextn = cur` AFTER `build_norm(cur,
  model.output_norm)`, and the MTP graph likewise publishes its t_h_nextn
  after the shared-head norm — so the recursive h (draft step i+1's hidden
  input) is `mtp.norm(layer_out)`, not the raw layer output. vLLM's DeepSeek
  MTP does the same (`shared_head` = norm, applied before both the lm_head
  and the feedback edge). This module implements exactly that.
* The head's single decoder layer IS a `Qwen3_5DecoderLayer` with
  block_type "full_attention" — checked shape-for-shape against the
  checkpoint index: mtp.layers.0.self_attn.q_proj is [12288, 5120], byte
  identical in shape to trunk layer 3's (24 heads x 256 head_dim x 2 for the
  output gate), o_proj [5120, 6144], k/v [1024, 5120], mlp 17408 wide. So we
  build the real module and stream the real tensors into it: no vendored
  modeling code, no reimplemented attention.
* `mtp.fc` is [5120, 10240] and the concat order is
  cat[norm_e(embed), norm_h(hidden)] (llama.cpp: `ggml_concat(e_norm, h_norm,
  dim=0)`), i.e. the embedding half comes FIRST.
* config.json carries `mtp_num_hidden_layers: 1` and
  `mtp_use_dedicated_embeddings: false` — the head shares the trunk's
  embed_tokens and lm_head, which is why it costs 0.8GB and not 3.
* transformers 5.15.1 `Qwen3_5ForCausalLM.forward` already returns logits for
  ALL positions (`logits_to_keep=0` -> `slice(0, None)`), so plumbing
  num_logits_to_keep through is a no-op here — the verify pass gets its M rows
  for free. We still take the inner text model directly
  (`model.get_decoder()`), because the wrapper drops the hidden states the
  draft head needs, and then run the wrapper's own lm_head call verbatim
  (every row) rather than a narrower one: see forward_with_hidden.

THE FOUR STATE SURFACES ON REJECTION (the hard part)

1. Full-attention layers: `StaticLayer.update` IGNORES the cache_position it
   is handed and writes at its own `cumulative_length` counter
   (cache_utils.py:455, transformers 5.15.1) — the footgun the prefix cache
   was built around. Rewind is therefore: subtract the rejected row count from
   `cumulative_length` IN PLACE (it is a 0-dim tensor with a static address;
   rebinding it would break cudagraph capture) and leave the tensor contents
   alone. Stale rows beyond the counter are unreachable because the causal
   mask is built from `q_offset = get_seq_length()` (masking_utils
   `_preprocess_mask_arguments`) — row j of the next forward can only see kv
   indices <= its own absolute position. The slot's own layer
   (serving/kvcache.py's LiveStaticLayer) keeps a python int of the same
   count beside the tensor and hands attention only the rows below it, so
   its `rewind(drop)` moves both; the plain-StaticLayer branch below stays
   for a cache built elsewhere. `tests/test_serving_mtp.py` pins that
   behaviour BEHAVIOURALLY (write at a lying cache_position, assert where the
   rows landed), so a transformers bump that renames or re-times the counter
   fails loudly instead of corrupting a transcript.
1b. Sliding-window layers (gemma-4's `sliding_attention`, 50 of its 60
   layers): `StaticSlidingWindowLayer` is
   a StaticLayer subclass that keeps TWO counters — the `cumulative_length`
   tensor above, which only the not-yet-full write path advances, and a
   python `cumulative_length_int`, which every path advances and which is
   what `get_seq_length()` (hence the mask's q_offset), `is_full` and
   `get_mask_sizes` read. Subtracting from the tensor alone left the int
   `drop` too high after every rejection, so every later query in those
   layers was masked as if it sat `drop` positions later than it did and
   SAW THE REJECTED DRAFTS' STALE ROWS. On the real 31B that read as text
   with characters missing — the model had "seen" the token it never
   emitted — and a transcript that fell apart one rejection at a time.
   And once the window is full the write path is a RING: an M-row update
   shifts the M oldest rows out of the storage for good, so a rewind there
   has nothing in the layer to rewind to. `_snapshot_sliding` therefore
   records, per sliding layer and per cycle, the int counter as it stood and
   — only when the coming write will shift — the first M rows of the ring
   (M x heads x head_dim: bytes, not megabytes); `_rewind_sliding` puts the
   ring back so that afterwards it is BYTE-IDENTICAL to a ring that only
   ever saw the accepted rows, in all three regimes (below the window,
   crossing it, full), and sets both counters. No recompute, no forward:
   the cost is that snapshot, and only on cycles that would have shifted.
   `tests/test_serving_gemma_spec.py` pins each regime against a cache that
   only ever stepped serially, and greedy agreement with serial decode end
   to end.
2. DeltaNet layers: the recurrence cannot rewind, so it must be REPLAYED per
   position. We wrap the installed `Qwen3_5GatedDeltaNet.forward` (instance
   level, not class level) for the M-in-2..MC_MAX-with-cache case: the four
   input projections and the causal conv still run BATCHED over the M rows —
   that is the weight-read saving MTP exists for, and those Linears serve
   the whole batch off ONE read of the compressed weights (the codec's
   multi-column kernel) instead of one read per row, which is what makes
   the saving real on the compressed arm — and only the
   delta-rule recurrence is unrolled, one `torch_recurrent_gated_delta_rule` call per position with
   the state threaded through and cloned at each step. That call is the exact
   function the M=1 decode path calls with the exact same shapes, so the
   replayed recurrence is bit-identical to stepping. Which function that NAME
   is bound to is serving/deltanet.py's call: on this
   stack transformers' own lookup misses — it wants
   `fla.ops.gated_delta_rule.recurrent_gated_delta_rule` and fla only exports
   `fused_recurrent_gated_delta_rule`, so `use_kernel_func_from_hub_with_fallback`
   keeps the pure-torch loop (~40 launches per position) — and the loaders
   rebind the modeling module's global to fla's fused kernel on an
   accelerator (one launch per position), the torch reference on CPU or by
   `DRINKME_DELTANET_KERNEL=torch`. The import below reads the module
   attribute at call time, so this replay and the M=1 step always run the
   same one. (The chunk path resolves to fla by itself, which is why prefill
   needs a GPU.)
   The conv state rides along for free: `update_conv_state` returns the full
   [prev_window ++ new] tensor, so the state after accepting j rows is just
   `window[..., j+1 : j+1+kernel]`. No `activate_past_recording` /
   `crop` (transformers has that seam, but turning it on forces every M=1
   decode step off `causal_conv1d_update` and onto a growing-window conv —
   a real decode regression to buy a rewind we get from a slice).
3. The head's own KV: one attention layer we own, plain crop. Between
   requests it stays with the prefix slot (`HeadKV`, engines.py), cropped at
   the reuse point the way the trunk's cache is.

WHAT THIS DOES NOT DO: grammar-constrained output. The constraint walks a
grammar token by token, a per-step decision the batch cannot make ahead of
itself; those requests take the serial loop. Greedy WITH penalties is
supported on the argmax path — the main model's per-row pick goes through
`sampling.sample_next` with the same params and the same running id lists
the serial loop would have had, so the penalties see the history serial
decode would have shown them at every row — and the sampled path gets its p
and q rows from `sampling.sample_probs` with the same id lists, which is the
same fact one level down.
"""

from __future__ import annotations

import os
import sys
import types
from typing import NamedTuple

import torch

from ..codec.ops import MC_MAX
from ..codec.swap import CompressedLinear
from . import metrics, ngram
from .image_prompt import step_rows
from .speculative import (
    AcceptanceWindow,
    SpeculativeAudit,
    point_mass_rows,
    speculative_sample,
)

# The default depth: k=4 drafts -> M=5 verify rows.
#
# FUSED_M_MAX is the widest verify batch whose compressed Linears stay on the
# serial step's kernel numerics: codec/swap.py serves 2..MC_MAX through the
# multi-column kernel, one read of the weights for the whole batch, each
# column bitwise equal to the M=1 kernel's answer (bench/radix_mc_bitpin.py;
# within the oracle bound, bitwise recorded). Above the dense threshold the
# batch switches to decode-once + native GEMM: a transient BF16 copy of every
# weight per verify pass, in another reduction order. Derived, not typed, so
# that moving either constant moves this warning with it — and note the depth
# ceiling is FUSED_M_MAX - 1 drafts.
DEFAULT_DEPTH = 4
FUSED_M_MAX = min(MC_MAX, CompressedLinear.GEMM_MIN_ROWS - 1)

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[drinkme.mtp] {msg}", file=sys.stderr)


DEFAULT_DEPTH = 4
"""The auto-mode draft depth: the shipped/GPU-verified configuration (depth 4
= verify M=5, inside the fused multi-column window with margin)."""

# The adaptive bail (module docstring): the decided numbers are the DEFAULTS.
# DRINKME_MTP_BAIL_WINDOW / DRINKME_MTP_BAIL_FLOOR override them per engine
# (bail_from_env -> SpecPlan -> Speculator), read once with the other spec
# knobs; a test pins them through the env or the Speculator's own arguments.
# The window counts DRAFTED positions (engines.py D4).
BAIL_WINDOW = 64
BAIL_FLOOR = 0.08

# The re-arm (module docstring): serial tokens after a trip before a fresh
# window drafts again, the backoff cap, and how many WINDOWS' worth of
# drafted positions a re-armed window has to survive above the floor to
# reset the backoff (REARM_SURVIVE is that at the default window; a
# Speculator scales it with the window it was given).
REARM_DEFAULT = 64
REARM_CAP = 512
REARM_SURVIVE_WINDOWS = 2
REARM_SURVIVE = REARM_SURVIVE_WINDOWS * BAIL_WINDOW


def bail_from_env() -> tuple[int, float]:
    """DRINKME_MTP_BAIL_WINDOW (drafted positions, an integer >= 1) and
    DRINKME_MTP_BAIL_FLOOR (0..1; 0 = never bail) — the adaptive bail's
    trigger (docs/cli.md). Unset = the decided defaults (BAIL_WINDOW,
    BAIL_FLOOR); anything unparsable or out of range is the default with
    one stderr line, never fatal (depth_from_env's rule: a typo in an env
    var must not take the server down, and must say so)."""
    window, floor = BAIL_WINDOW, BAIL_FLOOR
    raw = os.environ.get("DRINKME_MTP_BAIL_WINDOW")
    if raw is not None and raw.strip():
        try:
            n = int(raw)
            if n < 1:
                raise ValueError(raw)
            window = n
        except ValueError:
            _warn_once("bail_window", f"DRINKME_MTP_BAIL_WINDOW={raw!r} is not an "
                                      f"integer >= 1; using {BAIL_WINDOW}")
    raw = os.environ.get("DRINKME_MTP_BAIL_FLOOR")
    if raw is not None and raw.strip():
        try:
            f = float(raw)
            if not 0.0 <= f <= 1.0:  # also rejects nan
                raise ValueError(raw)
            floor = f
        except ValueError:
            _warn_once("bail_floor", f"DRINKME_MTP_BAIL_FLOOR={raw!r} is not a "
                                     f"number in 0..1; using {BAIL_FLOOR}")
    return window, floor


def rearm_from_env() -> int:
    """DRINKME_MTP_REARM: tokens on the serial loop after a trip before the
    bail re-arms (docs/cli.md). Unset = REARM_DEFAULT; 0 = a one-way latch,
    never re-arm; anything unparsable is the default with one stderr line."""
    raw = os.environ.get("DRINKME_MTP_REARM")
    if raw is None or raw.strip() == "":
        return REARM_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        _warn_once("rearm", f"DRINKME_MTP_REARM={raw!r} is not an integer; "
                            f"using {REARM_DEFAULT}")
        return REARM_DEFAULT
    return max(0, n)

# The acceptance audit (docs/serve-speculation.md): the self-referential
# check that has power under real traffic, where the two-arm first-token
# check does not. On by default — "a check nobody runs is a check that
# doesn't exist" — DRINKME_SPEC_AUDIT=0 turns it off.
_AUDIT_CUMULATIVE = SpeculativeAudit()  # across every request this process serves
_AUDIT_PRINT_EVERY = 50
_audit_cycles = 0


def _audit_enabled() -> bool:
    return os.environ.get("DRINKME_SPEC_AUDIT") != "0"


def _audit_tick() -> None:
    """A3: one line every 50 sampled cycles across every request, so the
    cumulative counter accrues under real traffic with no one driving it."""
    global _audit_cycles
    _audit_cycles += 1
    if _audit_cycles % _AUDIT_PRINT_EVERY:
        return
    a = _AUDIT_CUMULATIVE
    print(f"[drinkme.mtp] audit: {a.decided} decided, actual {a.actual_rate:.0%} "
          f"vs expected {a.expected_rate:.0%}, {a.violations} violations",
          file=sys.stderr)


class _FanOutAudit:
    """Feeds the same observations to two SpeculativeAudit sinks in one call:
    the request's own (reset per request, in Speculator.stats()) and the
    cumulative one across requests (A3). Same three methods as
    SpeculativeAudit, so speculative_sample cannot tell the difference."""

    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b

    def observe(self, expected, actual) -> None:
        self.a.observe(expected, actual)
        self.b.observe(expected, actual)

    def observe_resample(self, pi, qi, t) -> None:
        self.a.observe_resample(pi, qi, t)
        self.b.observe_resample(pi, qi, t)

    def observe_bonus(self, p_depth, t) -> None:
        self.a.observe_bonus(p_depth, t)
        self.b.observe_bonus(p_depth, t)


def depth_from_env() -> int | None:
    """DRINKME_MTP_DEPTH -> draft depth k >= 1, or None meaning AUTO.

    UNSET = AUTO (MTP defaults ON when the model allows): the caller
    resolves auto to
    DEFAULT_DEPTH iff the checkpoint actually carries an MTP head, and to
    no head otherwise. This variable is a depth only: turning speculation
    off is DRINKME_SPEC=off (ngram.resolve_mode). Garbage or a depth below
    1 = AUTO with a warning (never fatal: a typo in an env var must not take
    the server down, and must not silently change decode either — it says
    so and serves the default path)."""
    raw = os.environ.get("DRINKME_MTP_DEPTH", "").strip()
    if not raw:
        return None
    try:
        k = int(raw)
    except ValueError:
        k = 0
    if k < 1:
        _warn_once("badenv", f"DRINKME_MTP_DEPTH={raw!r} is not a draft depth of 1 or "
                             "more — using the default (DRINKME_SPEC=off turns speculation off)")
        return None
    if k > FUSED_M_MAX - 1:
        _warn_once(
            "deepk",
            f"DRINKME_MTP_DEPTH={k} puts the verify batch at M={k + 1} rows, past the "
            f"M<={FUSED_M_MAX} fused window (codec/swap.py MC_MAX={MC_MAX}, "
            f"GEMM_MIN_ROWS={CompressedLinear.GEMM_MIN_ROWS}): on a pack, every "
            "verify pass then decodes each compressed weight to a transient "
            "BF16 copy and runs a dense matmul over it, which moves more bytes "
            "per cycle than the multi-column kernel's one read of the "
            f"compressed weights. Depth {FUSED_M_MAX - 1} is the deepest "
            "inside the window.")
    return k


def is_supported(config) -> bool:
    """MTP is qwen3_5-only in v1 — every other family keeps the default path."""
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    return getattr(text, "model_type", None) in ("qwen3_5", "qwen3_5_text")


# --------------------------------------------------------------- the head --


def _head_config(text_cfg):
    """A one-layer, full-attention copy of the text config: the MTP block has
    the trunk's shapes exactly (verified against the checkpoint index), so the
    real `Qwen3_5DecoderLayer` builds it with no shape surgery."""
    import copy

    cfg = copy.deepcopy(text_cfg)
    cfg.num_hidden_layers = 1
    cfg.layer_types = ["full_attention"]
    # from_config/from_pretrained stamp the trunk's config with its resolved
    # attention implementation, and the deepcopy carries it (measured: "sdpa"
    # on the 27B path). A bare AutoConfig does NOT, and an unset one makes
    # ALL_ATTENTION_FUNCTIONS fall back to eager with a warning — the draft
    # head would then run different attention from the trunk it drafts for,
    # slower and for no reason. Say sdpa rather than inherit a None.
    if getattr(cfg, "_attn_implementation", None) is None:
        try:
            cfg._attn_implementation = "sdpa"
        except (AttributeError, ValueError):  # exotic build without sdpa
            pass
    return cfg


def _skel_dtype(trunk) -> torch.dtype:
    """The trunk's floating dtype — what load_head will stream the head in as.
    Falls back to the process default when the trunk has no float parameter
    left to ask (a fully compressed trunk still has its embedding, but nothing
    in this file may assume that)."""
    for p in trunk.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.get_default_dtype()


class MTPHead(torch.nn.Module):
    """The draft model: fc + two pre-norms + one decoder layer + final norm.

    Submodules are named exactly as the checkpoint names them under `mtp.`
    (fc, pre_fc_norm_embedding, pre_fc_norm_hidden, layers.0.*, norm), so
    loading is a name-for-name stream with no translation table to keep in
    sync. embed_tokens and lm_head are NOT owned — they are the trunk's, held
    in a plain tuple so nn.Module never registers (or moves, or saves) them.
    """

    def __init__(self, text_cfg, trunk):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
        )

        cfg = _head_config(text_cfg)
        h = cfg.hidden_size
        # meta skeleton, zero bytes: the tensors stream in one at a time
        # (engines.load_compressed's pack-diet pattern — peak host memory is
        # one tensor, never one head).
        #
        # AT THE TRUNK'S DTYPE: every parameter here is replaced by
        # a streamed one anyway, so the skeleton's dtype never reaches a
        # matmul — but codec.swap.eligible() TESTS it, and the head diet needs
        # "which Linears are eligible" to mean the same thing when the packer
        # asks a zero-byte skeleton as when the loader asks the real head. A
        # float32 default said "none of them" for a bf16 checkpoint. set_
        # default_dtype rather than a dtype= argument: Qwen3_5DecoderLayer
        # builds its own Linears and takes no dtype.
        _old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_skel_dtype(trunk))
        try:
            with torch.device("meta"):
                self.fc = torch.nn.Linear(2 * h, h, bias=False)
                self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
                self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
                self.layers = torch.nn.ModuleList([Qwen3_5DecoderLayer(cfg, 0)])
                self.norm = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        finally:
            torch.set_default_dtype(_old_dtype)
        self.cfg = cfg
        text = trunk.get_decoder() if hasattr(trunk, "get_decoder") else trunk.model
        self._trunk = (trunk, text)
        self.cache = None
        self.entries = 0  # head-KV entries currently resident
        # the reduced-vocab draft projection, or None for the
        # full vocab. A plain attribute holding a NON-Module, so the head's
        # parameters/state_dict are exactly what they were.
        self.draft_proj = None
        from . import suffix_attention
        suffix_attention.install(self)

    # -- trunk borrowings (never registered as submodules) --
    @property
    def embed(self):
        return self._trunk[1].embed_tokens

    @property
    def rotary(self):
        return self._trunk[1].rotary_emb

    @property
    def lm_head(self):
        return self._trunk[0].lm_head

    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache(config=self.cfg)
        self.entries = 0

    def crop(self, n: int) -> None:
        """Drop the last n entries of the head's own KV (rejected drafts)."""
        if n <= 0:
            return
        self.cache.layers[0].crop(-n)
        self.entries -= n

    def _mask(self, q: int, past: int, dtype, device):
        """Additive causal mask for a q-row pass over a cache holding `past`
        entries. None for q == 1: sdpa's `is_causal` is forced False at
        q_length == 1 (integrations/sdpa_attention.py), which is precisely
        "attend to everything cached" — what a decode step wants. For q > 1
        we need lower-right causality: ordinary sdpa's is_causal aligns
        top-left and would mask the existing cache away. Short GPU passes
        use CausalBias; other passes materialize the additive mask."""
        if q == 1:
            return None
        from . import segmented_attention, suffix_attention
        if segmented_attention.needed(q, past):
            # a chunked prefill's seed over a long head history: bounded
            # dispatches, no [q, past + q] mask (serving/segmented_attention.py)
            segmented_attention.register()
            return segmented_attention.SegmentedCausal(
                q, past, segmented_attention.segment_for(q))
        if (suffix_attention.enabled(q, device)
                and self.layers[0].self_attn.config._attn_implementation == suffix_attention.NAME):
            return suffix_attention.causal_lower_right(q, past + q)
        kv = past + q
        idx = torch.arange(kv, device=device)
        allowed = idx[None, :] <= (past + torch.arange(q, device=device))[:, None]
        mask = torch.zeros(1, 1, q, kv, dtype=dtype, device=device)
        return mask.masked_fill_(~allowed, torch.finfo(dtype).min)

    def run(self, hidden, token_ids, first_entry: int, positions=None):
        """One head pass over T entries: entry p consumes (h_p, t_{p+1}) and
        sits at rope position p. Appends T entries to the head's KV and
        returns the post-`mtp.norm` hidden states [1, T, H] — which is both
        what lm_head consumes AND what the next draft step feeds back as `h`
        (see the module docstring: post-norm, verified against llama.cpp).

        After a prompt with images the trunk's positions are M-RoPE's, and
        the head takes the trunk's position at each entry: `positions`
        ([3, 1, T]) for entries inside the prompt, and past it a
        `first_entry` the Speculator has shifted by the prompt's delta
        (entry p sits at p + delta in all three rows there)."""
        e = self.embed(token_ids)
        x = self.fc(torch.cat(
            (self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(hidden)), dim=-1))
        t = x.shape[1]
        if positions is None:
            pos = torch.arange(first_entry, first_entry + t, device=x.device)
            pos = pos.view(1, 1, -1).expand(3, 1, -1)  # mrope: (3, bs, seq)
        else:
            pos = positions
        cos, sin = self.rotary(x, pos)
        # on CUDA without cuDNN attention, as all of drinkme's attention runs
        # (sdpa.route): its per-length plan build cost 21 ms a head call on
        # a fresh HTTP thread
        out = self.layers[0](
            x,
            position_embeddings=(cos, sin),
            attention_mask=self._mask(t, self.entries, x.dtype, x.device),
            past_key_values=self.cache,
        )
        self.entries += t
        return self.norm(out)

    def _propose(self, hn):
        """The draft's own next-token pick, as a [1] long tensor ON THE
        DEVICE — never a python int (see _draft: the chain must not sync).

        Plain argmax: drafts are PROPOSALS — the verify pass decides what is
        emitted, so the head's sampling policy cannot affect correctness, only
        acceptance rate. That is also what lets `draft_proj` exist: with the
        reduced-vocab lever on the argmax runs over K of V
        rows and the winning position maps back through the selection (the
        projection's device-resident `ids`, a gather), so a
        token outside the subset is simply never proposed. It costs that
        cycle's tail; it cannot cost a token, because verify runs the FULL
        lm_head over every row regardless."""
        if self.draft_proj is None:
            return self.lm_head(hn[:, -1]).argmax(-1)
        pos = self.draft_proj(hn[:, -1]).argmax(-1)
        return self.draft_proj.ids[pos]

    def _draft_row(self, hn):
        """The draft's logits over the FULL vocab, [V], for the sampled path.
        With the reduced-vocab lever on, the K subset logits are scattered
        onto a -inf row: tokens outside the subset get zero probability, so
        the draft can never propose them (exactly as the argmax path never
        does) and the q row the accept rule divides by says so."""
        if self.draft_proj is None:
            return self.lm_head(hn[:, -1])[0]
        sub = self.draft_proj(hn[:, -1])[0]
        row = torch.full((self.draft_proj.R,), float("-inf"), dtype=sub.dtype,
                         device=sub.device)
        row[self.draft_proj.ids] = sub
        return row

    def _draft(self, h, token: int, first_entry: int, k: int, propose):
        """The chain STAYS ON THE DEVICE.

        Ending every draft step in `int(argmax)` would be a device->host
        sync per drafted token, k per cycle, each draining the GPU queue
        before the host could enqueue the next head forward. What the
        chain costs, measured (bench/mtp_cycle_floor.py, 27B sip, k=4): the chain
        is ~49 ms against a 212 ms trunk step (c = 0.057 per draft step,
        Leviathan's cost coefficient), and of those 49 ms the FULL-VOCAB
        lm_head read is ~39 (4 x 1.67 GiB at the box's bandwidth), the
        head's own layer ~7.5, dispatch + sync ~2.5. So the chain is
        bandwidth-bound on lm_head, and keeping it on the device buys the
        sync bubbles (a few ms per cycle), not a smaller c; c moves when
        the draft reads fewer lm_head rows (draft_vocab.py's lever, which
        has no row-subset kernel for a packed lm_head). The verify
        batch's own input row is built here, in place: `rows` is
        [1, k + 1] = [token, d_0, ..., d_{k-1}] on the device, draft step i
        reads its input as the view rows[:, i:i+1] (the embedding lookup
        takes a device index) and writes its pick into rows[0, i + 1] —
        the k head forwards and the verify's input are enqueued with no
        host round trip, and the cycle reads the drafts back ONCE, after
        the verify forward is in the queue too. On the sampled path
        `propose` returns the draw as a device tensor and its q row stays
        a device tensor; the `so_far` it is handed is the drafts so far as
        0-dim device tensors, which the penalty history converts only when
        a penalty is actually set (sampling._penalize's `int(i)` — a sync
        per drafted id on that path alone, and bitwise the same history).

        Returns (rows, q rows)."""
        qs = []
        rows = torch.empty((1, k + 1), dtype=torch.long, device=h.device)
        rows[0, 0] = token
        for i in range(k):
            hn = self.run(h, rows[:, i:i + 1], first_entry + i)
            if propose is None:
                rows[0, i + 1] = self._propose(hn)
            else:
                tok, q = propose(self._draft_row(hn), list(rows[0, 1:i + 1]))
                rows[0, i + 1] = tok
                qs.append(q)
            h = hn
        return rows, qs

    def draft(self, h, token: int, first_entry: int, k: int):
        """k recursive draft steps, greedy: each step proposes its argmax.
        Returns the verify rows [1, k + 1] on the device — `token` first,
        then the k drafts (`_draft`); no host sync."""
        return self._draft(h, token, first_entry, k, None)[0]

    def draft_sampled(self, h, token: int, first_entry: int, k: int, propose):
        """k recursive draft steps, SAMPLED. `propose(logits_row, so_far)`
        returns (token, q): the token it drew — a device tensor, or an int —
        and the [V] distribution it drew it from, given the tokens proposed
        so far this cycle (0-dim device tensors) — so the penalty history
        matches the verify row the token will be judged against. Returns
        (rows [1, k + 1] on the device, q rows). The q rows are what the
        accept rule divides by, so they must be the very distributions the
        tokens came from; the caller builds both from one sample_probs
        call."""
        return self._draft(h, token, first_entry, k, propose)


def _rows_below(spans: list, n: int) -> int:
    """How many of the rows `spans` describes hold an entry below n."""
    rows = 0
    for a, b in spans:
        if a >= n:
            break
        rows += min(b, n) - a
    return rows


def _spans_below(spans: list, n: int) -> list:
    """`spans` without the entries at or past n."""
    return [[a, min(b, n)] for a, b in spans if a < n]


class HeadKV:
    """The head's attention history as a prefix slot keeps it between
    requests (engines.py): the head's DynamicCache (`cache`), how many rows
    it holds (`entries`), which entries those rows are (`spans`: [a, b)
    ranges in row order, increasing and disjoint), and `tail`.

    A hole between two spans is a stretch no request ran the head over: a
    request served without it (grammar-constrained, DRINKME_SPEC=off), a
    serial stretch the request ended in, a slot restored from the cold
    tier, which stores no head KV. The head attends over what is there;
    a hole costs acceptance, never a token.

    `tail` is (p, h): the trunk's hidden state [1, 1, H] at the slot's last
    written position p, cloned so it keeps no larger tensor alive, for
    which the head has no entry yet — entry p = (h_p, t_{p+1}) needs the
    token the next prompt brings. None when the request that stored it
    could not say (Speculator.history).

    Entry p reads tokens 0..p+1 alone, so after a request reuses r positions
    of the slot it stays valid for p <= r - 2, and for p = r - 1 when the
    slot's token at r is the prompt's; h_p for p <= r - 1. The engine
    applies that through `keep`."""

    __slots__ = ("cache", "entries", "spans", "tail")

    def __init__(self, cache, entries: int, spans: list, tail=None):
        self.cache = cache
        self.entries = entries
        self.spans = spans
        self.tail = tail

    def keep(self, n: int, reuse: int) -> None:
        """Drop every entry at or past n, and the tail unless its position
        is below `reuse`, the positions the next request keeps. A negative
        crop removes rows from the end (transformers' DynamicLayer.crop)."""
        rows = _rows_below(self.spans, n)
        drop = self.entries - rows
        layer = None if self.cache is None else self.cache.layers[0]
        if drop > 0 and layer is not None and getattr(layer, "is_initialized", False):
            layer.crop(-drop)
        self.entries = rows
        self.spans = _spans_below(self.spans, n)
        if self.tail is not None and self.tail[0] >= reuse:
            self.tail = None

    @property
    def nbytes(self) -> int:
        """Bytes of keys and values the rows hold."""
        layer = None if self.cache is None else self.cache.layers[0]
        if layer is None or not getattr(layer, "is_initialized", False):
            return 0
        return sum(t.numel() * t.element_size() for t in (layer.keys, layer.values))


# ------------------------------------------- residency guard (D2, guard 1) --
# Decided guard 1: FIT
# OUTRANKS SPEED. suggest.py's fit check already owns the margin
# (suggest.FIT_HEADROOM, 1.10 — 10% of the resident footprint) that decides
# "knife-edge" for the model itself; the AUTO head-load reuses that number
# rather than inventing a second one.

_DTYPE_NBYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
                 "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
_GIB = 1024 ** 3


def diet_enabled() -> bool:
    """The head diet: serve the MTP head's eligible Linears out of
    the pack, compressed, exactly as the trunk is. On whenever the pack
    carries a head; `DRINKME_MTP_DIET=0` is the explicit opt-out and gives
    the default raw bf16 head — including the default residency charge, so the guard
    and the loader must read this from the same place."""
    return os.environ.get("DRINKME_MTP_DIET") != "0"


def head_pack_gib(pack_dir: str | None) -> float | None:
    """GiB the head will occupy when it is served OUT OF THE PACK, or None
    when there is no head sub-pack to serve from (or the diet is off).

    MEASURED, not estimated: pack time ran the loader's own to_device on every
    head tensor and recorded what it put on the device (`residentBytes` — a
    little more than the npz on disk: the padded palette tables and, for 3+
    tiers, the per-block schedule), plus the mtp.* tensors the codec does not
    touch. A
    sub-pack from a build that did not record the measurement falls back to
    the npz file sizes, which under-count the derived arrays by ~0.5%."""
    if pack_dir is None or not diet_enabled():
        return None
    from ..codec.pack import mtp_pack_dir, read_mtp_meta

    meta = read_mtp_meta(pack_dir)
    if meta is None:
        return None
    resident = meta.get("residentBytes")
    if resident is None:  # a sub-pack from a build that did not measure
        import glob

        hp = mtp_pack_dir(pack_dir)
        resident = sum(os.path.getsize(f)
                       for f in glob.glob(os.path.join(hp, "*.npz")))
    return (resident + meta.get("unpackedBytes", 0)) / _GIB


def head_gib(repo: str, revision: str | None,
             pack_dir: str | None = None) -> float | None:
    """GiB of every mtp.* tensor across the checkpoint's safetensors shards,
    read from the JSON header alone (the 8-byte length prefix + header,
    check.py's remote trick done locally) — cheap enough to call before
    deciding whether to load. None when the checkpoint carries no head.

    With `pack_dir` naming a pack that carries a packed head (and the diet on),
    the answer is that head's MEASURED compressed residency instead: the guard
    must charge what the loader will actually put on the device, or the fit
    story is decided on a number nothing serves."""
    packed = head_pack_gib(pack_dir)
    if packed is not None:
        return packed
    import glob
    import json
    import struct

    from ..arms import snapshot_dir

    total = 0
    found = False
    for fpath in sorted(glob.glob(os.path.join(snapshot_dir(repo, revision),
                                               "*.safetensors"))):
        with open(fpath, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for key, meta in header.items():
            if key == "__metadata__" or not key.startswith("mtp."):
                continue
            found = True
            nbytes = _DTYPE_NBYTES.get(meta["dtype"], 2)
            n_elem = 1
            for s in meta["shape"]:
                n_elem *= s
            total += nbytes * n_elem
    return total / _GIB if found else None


def device_headroom_gib(device: str) -> tuple[float, float] | None:
    """(free_gib, used_gib) on a real accelerator that already holds the main
    model, or None when there is nothing to measure (cpu — the toy tests;
    mps — no mem_get_info yet, engine.py's Metal note). `used_gib` stands in
    for the model's own resident footprint — the live equivalent of
    suggest.Sizes.comp_gb, measured rather than looked up, since nothing
    upstream of here threads that number through."""
    # Compare the device TYPE, not the string: "cuda", "cuda:0" and
    # torch.device("cuda") all name the accelerator, and a `!= "cuda"` test
    # made the guard silently inert for two of the three (found by running
    # the known-good case on real hardware).
    try:
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
    except (RuntimeError, TypeError):
        return None  # nothing to measure is not a failure; the guard no-ops
    return free / _GIB, (total - free) / _GIB


def residency_check(free_gib: float, used_gib: float,
                    head_gib: float) -> tuple[bool, float, float]:
    """Would loading the head leave less than the fit check's own margin
    (suggest.FIT_HEADROOM)? Pure arithmetic, so the toy tests drive it with
    fixture numbers with no device involved. Returns (yields, left_gib,
    margin_gib) — left_gib is the headroom the head would leave, margin_gib
    the minimum the fit check requires (10% of the resident footprint)."""
    from ..suggest import FIT_HEADROOM

    margin_gib = used_gib * (FIT_HEADROOM - 1)
    left_gib = free_gib - head_gib
    return left_gib < margin_gib, left_gib, margin_gib


def install_head_pack(head, pack_dir: str | None, device: str) -> tuple[set, int]:
    """THE DIET: swap the head's eligible Linears for
    CompressedLinear straight out of `<pack>/mtp`, one tensor at a time —
    engines.load_compressed's loop, on the head's module tree instead of the
    trunk's, with the same hash gate.

    Returns (the checkpoint keys the pack now owns, measured resident bytes).
    Both are empty/zero when there is nothing to do — no pack dir, no head
    sub-pack (a checkpoint without an MTP head), or DRINKME_MTP_DIET=0 —
    and then the caller streams a raw bf16 head exactly as it always has.
    """
    from ..codec.pack import (has_mtp_pack, iter_pack_dir, mtp_pack_dir, resident_bytes,
                              verify_hashes)
    from ..codec.swap import make_module

    if pack_dir is None or not diet_enabled() or not has_mtp_pack(pack_dir):
        return set(), 0
    hp = mtp_pack_dir(pack_dir)
    verify_hashes(hp)  # raises loudly on a mismatched or missing hash
    owned, resident = set(), 0
    for name, pack in iter_pack_dir(hp):
        if not isinstance(head.get_submodule(name), torch.nn.Linear):
            raise ValueError(f"head pack tensor {name} is not a Linear on this head")
        parent_path, _, child = name.rpartition(".")
        parent = head.get_submodule(parent_path) if parent_path else head
        # the class follows the tensor's dtype (swap.make_module), exactly
        # as the trunk's loader picks it. No sever-the-old-storage line
        # (swap.swap_linears has one): the head this replaces is still the
        # META skeleton, holding zero bytes to free
        mod = make_module(pack, None, device)
        setattr(parent, child, mod)
        resident += resident_bytes(mod.p)
        owned.add(f"mtp.{name}.weight")
    return owned, resident


def load_head(model, repo: str, revision: str | None, device: str = "cuda",
              pack_dir: str | None = None):
    """Stream the 15 `mtp.*` tensors out of the checkpoint shards into a real
    MTPHead. Per-tensor, never the whole shard — the same discipline
    load_compressed uses, for the same reason.

    With `pack_dir` naming a pack that carries a packed head, the eligible
    Linears come from the PACK instead (install_head_pack) and the checkpoint
    supplies only what the codec does not touch — the norms. The head is then
    served exactly as the trunk is: compressed, bit-exact, one decode per
    read. Nothing changes for a pack without a head sub-pack.

    Returns None (never raises) when the checkpoint carries no MTP head: an
    unsupported model must degrade to the default decode path, loudly, not 500.
    """
    import glob

    from safetensors import safe_open

    from ..arms import snapshot_dir

    cfg = model.config
    if not is_supported(cfg):
        _warn_once("family", f"MTP requested but {getattr(cfg, 'model_type', '?')} "
                             "has no MTP head — decoding without it")
        return None
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    head = MTPHead(text_cfg, model)
    dtype = next(model.parameters()).dtype
    packed, resident = install_head_pack(head, pack_dir, device)
    found = 0
    snap = snapshot_dir(repo, revision)
    # the raw-trunk door, again for the head's raw path (the trunk's loader
    # already refused an FP8 checkpoint by name; this keeps the head's own
    # walk honest on its own)
    from ..codec.pack import refuse_checkpoint

    refuse_checkpoint(repo, snap)
    for fpath in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(fpath, framework="pt") as sf:
            for key in sf.keys():
                if not key.startswith("mtp."):
                    continue
                found += 1
                if key in packed:
                    found -= 1  # counted from the pack below, wherever the raw half lives
                    # the pack IS this tensor; attaching it would materialize
                    # the bf16 read the codec exists to avoid (get_tensor is
                    # not called at all — the same hot-loop-audit finding as
                    # load_compressed's own load path)
                    continue
                name = key[len("mtp."):]
                mod_path, _, attr = name.rpartition(".")
                mod = head.get_submodule(mod_path) if mod_path else head
                t = sf.get_tensor(key)
                t = t.to(device=device, dtype=dtype)
                if isinstance(mod, CompressedLinear):
                    # a bias on a packed Linear (the head has none);
                    # arrives after the swap, exactly as the trunk's does
                    if attr != "bias":
                        raise ValueError(f"checkpoint tensor {key} collides with "
                                         "the head pack's own tensor")
                    mod.bias = t
                    continue
                if attr not in mod._parameters:
                    raise ValueError(f"checkpoint tensor {key} has no home on the MTP head")
                mod._parameters[attr] = torch.nn.Parameter(t, requires_grad=False)
                del t
    if not (found or packed):
        _warn_once("nohead", f"{repo} carries no mtp.* tensors — decoding without MTP")
        return None
    left = [n for n, p in head.named_parameters() if p.is_meta]
    if left:  # a partial head would serve garbage drafts and tank acceptance
        raise ValueError(f"MTP head incomplete, meta parameters remain: {left}")
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    install_deltanet_capture(model)
    install_draft_vocab(head, model)
    raw_bytes = sum(p.numel() * p.element_size() for p in head.parameters())
    n_params = sum(p.numel() for p in head.parameters())
    diet = ""
    if packed:
        # the head's real size on the device, measured: the compressed
        # tensors' runtime buffers + whatever stayed raw
        n_params += sum(m.R * m.C for m in head.modules()
                        if isinstance(m, CompressedLinear))
        diet = (f", DIET on: {len(packed)} tensors compressed, "
                f"{(resident + raw_bytes) / 2**20:.0f} MiB resident")
    print(f"[drinkme.mtp] head loaded: {found + len(packed)} tensors, "
          f"{n_params / 1e9:.2f}B params {dtype}{diet}")
    return head


def install_draft_vocab(head, model) -> None:
    """Point the head's draft argmax at a vocab SUBSET if
    DRINKME_MTP_DRAFT_VOCAB asks for one. Off by default, and every failure
    mode below degrades to the full vocab with a line on stderr — the lever
    buys speed, so a broken lever must cost speed, never correctness."""
    from . import draft_vocab

    text = (model.config.get_text_config()
            if hasattr(model.config, "get_text_config") else model.config)
    picked = draft_vocab.ids_from_env(model.config, int(text.vocab_size), _warn_once)
    if picked is None:
        return
    ids, source = picked
    proj, note = draft_vocab.make_projection(head.lm_head, ids)
    if proj is None:
        _warn_once("dvhead", f"DRINKME_MTP_DRAFT_VOCAB is set but {note} — "
                             "drafting over the full vocab")
        return
    head.draft_proj = proj
    print(f"[drinkme.mtp] draft vocab: {note}; source: {source}", file=sys.stderr)


# ------------------------------------------------- DeltaNet state capture --


class _Capture:
    """Per-cycle DeltaNet state, captured position by position so a rejection
    can restore the state AT the last accepted position — and
    the sliding-window layers' pre-cycle counter and ring head (surface 1b in
    the module docstring). Cleared every cycle."""

    def __init__(self):
        self.rec: dict[int, list[torch.Tensor]] = {}
        self.win: dict[int, torch.Tensor] = {}
        # per sliding layer index: the state the verify batch was written
        # over (_SlidingSnap). Absent for a layer = no snapshot was taken,
        # which _restore_rows tolerates only while that layer is below its
        # window (nothing was shifted out, the counters are the whole story).
        self.sliding: dict[int, _SlidingSnap] = {}
        self.rows = 0


class _SlidingSnap(NamedTuple):
    """One sliding-window cache layer as it stood BEFORE a verify batch.

    `length` is `cumulative_length_int` then; `keys`/`values` are the first
    `rows` entries of the ring storage — the only rows an update of `rows`
    tokens can shift out — or None when the update could not shift (the
    layer was at least `rows` short of full)."""

    length: int
    keys: torch.Tensor | None
    values: torch.Tensor | None


def _is_sliding_layer(layer) -> bool:
    """transformers' StaticSlidingWindowLayer, recognized by the surface the
    rewind has to manage (the second counter) rather than by class name."""
    return hasattr(layer, "cumulative_length_int") and hasattr(layer, "cumulative_length")


def _snapshot_sliding(cache, cap: _Capture, rows: int) -> None:
    """Before the verify batch writes `rows` tokens: record what a rewind
    will need from every sliding-window layer (docstring, surface 1b)."""
    for i, layer in enumerate(cache.layers):
        if not _is_sliding_layer(layer):
            continue
        if not getattr(layer, "is_initialized", True):
            cap.sliding[i] = _SlidingSnap(0, None, None)
            continue
        length = int(layer.cumulative_length_int)
        window = int(layer.max_cache_len)
        if rows > window:
            raise RuntimeError(
                f"verify batch of {rows} rows exceeds sliding window {window} on "
                f"cache layer {i}: a rewind could not be reconstructed")
        if length + rows > window:
            # the write will shift rows out of the ring; keep the ones it can
            cap.sliding[i] = _SlidingSnap(length, layer.keys[:, :, :rows].clone(),
                                          layer.values[:, :, :rows].clone())
        else:
            cap.sliding[i] = _SlidingSnap(length, None, None)


def _rewind_sliding(layer, snap: _SlidingSnap | None, rows: int, drop: int, i: int) -> None:
    """Rewind `drop` of the `rows` tokens the last update wrote to one
    StaticSlidingWindowLayer, so the layer is byte-identical to one that only
    ever saw the first `rows - drop` of them.

    With L the int counter before the write, W the window and M = rows:
      * L + M <= W: the not-full write path put the rows at [L, L+M) and
        shifted nothing; both counters go back and the stale rows past the
        new length are unreachable through the mask (surface 1's argument).
      * otherwise the storage now holds absolute rows [L+M-W, L+M) — a ring
        that shifted the oldest rows out. The rows a rewind must bring back
        are in the snapshot: the first M rows of the storage as it stood,
        i.e. absolute rows [a0, a0+M) with a0 = max(L-W, 0). After the
        rewind the logical length is N = L+M-drop, and the storage must read
          N <= W:  absolute [0, N) at [0, N)           (below the window again)
          N >  W:  absolute [N-W, N) at [0, W)         (still full)
        which is a concatenation of a slice of the snapshot and a slice of
        the current storage, written back IN PLACE (static addresses)."""
    window = int(layer.max_cache_len)
    length = None if snap is None else snap.length
    if length is None:
        # the two counters part ways exactly when a write shifted the ring
        # (int(tensor) is a device sync: only on this snapshot-less path,
        # which the cycle never takes)
        if int(layer.cumulative_length_int) != int(layer.cumulative_length):
            raise RuntimeError(
                f"no sliding-window snapshot for cache layer {i} and its ring has "
                "shifted: the verify pass did not go through _snapshot_sliding")
        # below the window, counters only (both were advanced together)
        layer.cumulative_length.sub_(drop)
        layer.cumulative_length_int -= drop
        return
    new = length + rows - drop
    if length + rows <= window:
        layer.cumulative_length.fill_(new)
        layer.cumulative_length_int = new
        return
    if snap.keys is None:
        raise RuntimeError(
            f"sliding-window snapshot for cache layer {i} carries no rows but the "
            f"write shifted ({length} + {rows} > {window})")
    a0 = max(length - window, 0)
    if new <= window:
        front = length + rows - window          # absolute [0, front) — shifted out
        keys = torch.cat((snap.keys[:, :, :front], layer.keys[:, :, :window - drop]), dim=2)
        vals = torch.cat((snap.values[:, :, :front], layer.values[:, :, :window - drop]), dim=2)
        layer.keys[:, :, :new].copy_(keys)
        layer.values[:, :, :new].copy_(vals)
        layer.cumulative_length.fill_(new)
    else:
        j0 = new - window - a0                  # snapshot index of absolute N-W
        keys = torch.cat((snap.keys[:, :, j0:j0 + drop], layer.keys[:, :, :window - drop]), dim=2)
        vals = torch.cat((snap.values[:, :, j0:j0 + drop], layer.values[:, :, :window - drop]), dim=2)
        layer.keys.copy_(keys)
        layer.values.copy_(vals)
        # the tensor counter is not read again while the layer is full
    layer.cumulative_length_int = new


_ACTIVE: _Capture | None = None


def install_deltanet_capture(model) -> None:
    """Shadow `forward` on every GatedDeltaNet INSTANCE (not the class): a
    test that builds two models must not have one patch the other, and an
    instance attribute is what nn.Module's __call__ resolves. The wrapper is a
    no-op unless a capture is active AND the batch is multi-row, so ordinary
    decode keeps the upstream code path exactly. This MTP preparation step
    also installs the suffix attention adapter; only an explicit suffix bias
    from forward_with_hidden takes its native path."""
    from . import suffix_attention
    suffix_attention.install(model)
    for mod in model.modules():
        if type(mod).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        if getattr(mod, "_drinkme_mtp_wrapped", False):
            continue
        orig = mod.forward  # bound, decorators intact

        def fwd(self, hidden_states, cache_params=None, attention_mask=None,
                _orig=orig, **kwargs):
            if (_ACTIVE is None or cache_params is None
                    or hidden_states.shape[1] == 1
                    or not cache_params.has_previous_state(self.layer_idx)):
                return _orig(hidden_states, cache_params=cache_params,
                             attention_mask=attention_mask, **kwargs)
            return _deltanet_forward_capturing(self, _ACTIVE, hidden_states,
                                               cache_params, attention_mask)

        mod.forward = types.MethodType(fwd, mod)
        mod._drinkme_mtp_wrapped = True


def _deltanet_forward_capturing(self, cap, hidden_states, cache_params, attention_mask):
    """Qwen3_5GatedDeltaNet.forward for the verify batch: projections and conv
    BATCHED (one weight read for all M rows — the entire point), recurrence
    UNROLLED (one state per position, so a rejection has somewhere to land).

    Mirrors the installed module's forward step for step; the state-restore
    test pins it against M sequential M=1 calls through the ORIGINAL forward,
    so a transformers change that this stops mirroring fails a test rather
    than quietly serving a different model.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_mask_to_padding_states,
        causal_conv1d_fn,
        torch_recurrent_gated_delta_rule,
    )

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    idx = self.layer_idx

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # window = [resident conv state ++ this batch's columns]; the conv state
    # after accepting j rows is window[..., j+1 : j+1+kernel], so one saved
    # tensor covers every rewind depth (see module docstring).
    window = cache_params.update_conv_state(mixed_qkv, idx,
                                            conv_kernel_size=self.conv_kernel_size)
    cap.win[idx] = window.clone()
    mixed_qkv = causal_conv1d_fn(window, self.conv1d.weight.squeeze(1),
                                 self.conv1d.bias, activation=self.activation)
    mixed_qkv = mixed_qkv[:, :, -seq_len:].transpose(1, 2)

    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        rep = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)

    state = cache_params.layers[idx].recurrent_states[0]
    # while a CUDA graph captures this verify (serving/cudagraph.py, MEMORY)
    # the rows go into the cache's one row buffer that every width shares,
    # not into copies each graph would hold; the last row is not stored
    # (a verify that keeps every row needs no rewind: _restore_rows never
    # reads it, and a None there would fail loudly if it did)
    bufs = (cache_params.__dict__.get("_drinkme_row_states")
            if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing() else None)
    buf = None if bufs is None else bufs.get(idx)
    outs, states = [], []
    for j in range(seq_len):
        core, state = torch_recurrent_gated_delta_rule(
            query[:, j:j + 1], key[:, j:j + 1], value[:, j:j + 1],
            g=g[:, j:j + 1], beta=beta[:, j:j + 1],
            initial_state=state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        outs.append(core)
        if buf is None:
            states.append(state.clone())  # fla may update in place; own our copies
        elif j < seq_len - 1:
            buf[j].copy_(state)
            states.append(buf[j])
        else:
            states.append(None)
    cap.rec[idx] = states
    cap.rows = seq_len
    core_attn_out = torch.cat(outs, dim=1)
    cache_params.update_recurrent_state(state, idx)

    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim),
                              z.reshape(-1, self.head_v_dim))
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


def _restore_rows(cache, cap, keep: int) -> None:
    """Rewind every cache surface to `keep` rows of the captured batch.

    Loud on anything unrecognized: a silent no-op here is a corrupted
    transcript, which is the one failure mode this whole feature must not
    have."""
    from transformers.cache_utils import CacheLayerMixin, LinearAttentionCacheLayerMixin

    from .kvcache import LiveStaticLayer

    drop = cap.rows - keep
    if drop <= 0:
        return
    j = keep - 1
    for i, layer in enumerate(cache.layers):
        known = False
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            known = True
            # an uninitialized linear layer holds no state to rewind (nothing
            # was ever written); an initialized one without a capture means the
            # verify pass did not go through the wrap, which is unrecoverable
            if layer.recurrent_states.get(0) is not None:
                states = cap.rec.get(i)
                if states is None:
                    raise RuntimeError(
                        f"no captured DeltaNet state for layer {i}: the verify pass "
                        "did not go through the capturing forward (was the wrap "
                        "installed?)")
                layer.recurrent_states[0].copy_(states[j])
                k = layer.conv_kernel_size[0]
                layer.conv_states[0].copy_(cap.win[i][..., j + 1:j + 1 + k])
            # on transformers 5.15.1 LinearAttentionCacheLayerMixin is NOT a
            # CacheLayerMixin subclass, so the branch below can't reach it —
            # but it DOES define crop() (which actively shrinks states), so a
            # version bump that changes the hierarchy would double-mutate a
            # layer we just restored. Make the branches exclusive by hand.
            continue
        if isinstance(layer, CacheLayerMixin):
            counter = getattr(layer, "cumulative_length", None)
            if _is_sliding_layer(layer):
                # surface 1b: two counters and a ring, never the plain
                # subtraction below
                _rewind_sliding(layer, cap.sliding.get(i), cap.rows, drop, i)
                known = True
            elif isinstance(layer, LiveStaticLayer):
                # the slot's own layer (serving/kvcache.py): the counter
                # tensor in place AND the live-window int, together
                layer.rewind(drop)
                known = True
            elif isinstance(counter, torch.Tensor):
                # StaticLayer writes at this counter and ignores cache_position
                # entirely; in-place so the static address survives.
                counter.sub_(drop)
                known = True
            elif hasattr(layer, "crop"):
                layer.crop(-drop)
                known = True
        if not known:
            raise RuntimeError(
                f"cannot rewind cache layer {i} ({type(layer).__name__}): no "
                "recurrent state, no cumulative_length tensor, no crop(). A "
                "transformers bump moved the write seam — MTP must not run until "
                "this is re-pinned.")
    cap.rows = keep


# ------------------------------------------------------- the state machine --


def forward_with_hidden(model, input_ids, cache, cache_position,
                        last_row_only=False, attention_mask=None,
                        inputs_embeds=None, position_ids=None):
    """One trunk forward returning (hidden [1, T, H], logits [T, V]).

    Goes through the inner text model rather than the CausalLM wrapper because
    the draft head needs the hidden states and the wrapper drops them
    (transformers 5.15.1 returns `hidden_states` only under
    output_hidden_states, and `last_hidden_state` not at all).

    `last_row_only` narrows the lm_head to the FINAL row and returns logits
    [1, V]. The draft cycle must NOT set it — it reads `logits[m]` for every
    row of the verify window. Prefill must, and here is why.

    An lm_head over EVERY prompt row (the wrapper's own call) would cost, at
    vocab 248320 in bf16, 0.474 MiB PER PROMPT TOKEN with every row but one
    discarded:

        4,096 tokens ->   1.9 GiB
       21,233 tokens ->   9.8 GiB    (a measured 500 on Strix Halo)
      262,144 tokens -> 121.2 GiB    (the context we ADVERTISE, on a 124 GiB machine)

    — the advertised context would be structurally unreachable: the
    discarded logits alone exceed the machine before weights or KV.

    BOTH PREFILLS NARROW. On the compressed arm an M-row lm_head takes
    decode-once + native GEMM while a 1-row one takes the GEMV kernel, and
    those differ in accumulation order. engines.py's non-MTP prefill narrows
    too (`logits_to_keep=1`), so with speculation on or off the first token's
    row comes out of the same 1-row lm_head path. The two are ONE design;
    change both halves together.

    What does move is version-to-version: a 1-row lm_head and a many-row one
    differ in accumulation order, so a near-tie at token one can land the
    other way than it did in the previous release. That is measured by the
    greedy A/B gates rather than assumed, and it is the same class of
    difference the pack already documents between the GEMV and dense arms.

    `inputs_embeds` / `position_ids` carry an image prompt
    (serving/image_prompt.py): its embeddings in place of `input_ids`
    (None then) and its M-RoPE rows. A text forward passes neither, and the
    call below is the one it always was."""
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    from . import suffix_attention
    # `attention_mask`: a mapping the caller already built — serving/
    # prefill.py's segmented one for a long extend (serving/
    # segmented_attention.py); None = the MTP suffix mask or transformers' own.
    # trunk_mask reads only the row count and the device of what it is given.
    rows_of = input_ids if input_ids is not None else inputs_embeds[..., 0]
    mask = (suffix_attention.trunk_mask(base, rows_of, cache)
            if attention_mask is None else attention_mask)
    kwargs = {} if mask is None else {"attention_mask": mask}
    if inputs_embeds is not None:
        kwargs["inputs_embeds"] = inputs_embeds
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    out = base(trunk_ids(model, input_ids), past_key_values=cache, use_cache=True,
               cache_position=cache_position, **kwargs)
    hidden = out.last_hidden_state
    rows = hidden[:, -1:] if last_row_only else hidden
    return hidden, head_logits(model, rows)[0]


def trunk_ids(model, input_ids):
    """`input_ids` as the model's own forward hands them to its text
    model's embedding. drinkme calls the text model itself for every
    prefill span but the last and for every speculative verify and re-arm
    step (this module's forward_with_hidden; serving/prefill.py), and the
    serial step goes through the wrapper. transformers 5.15.1's multimodal
    wrappers embed a modality's placeholder id as ANOTHER id first:

      gemma4        the image, video and audio token ids as the text pad id
                    (Gemma4Model.forward)
      muse_glimmer  the image and video token ids as 0
                    (MuseGlimmerModel.forward)

    so without this such an id, in a prompt's text or generated by the
    model, had two embeddings, and a verify row's logits were not the
    serial step's: a random Muse-Glimmer toy that generates <|patch|>
    diverged under n-gram at the next token
    (tests/test_serving_glimmer_vision.py). The ids go through here, the
    same torch.where the wrapper applies, and every other family's are
    returned as they are (the same tensor). None stays None (an image
    prompt's forward, whose embeddings are serving/image_prompt.py's)."""
    cfg = model.config
    kind = getattr(cfg, "model_type", None)
    if input_ids is None or kind not in ("gemma4", "muse_glimmer"):
        return input_ids
    if kind == "gemma4":
        placeholders = (cfg.image_token_id, cfg.video_token_id, cfg.audio_token_id)
        fill = cfg.get_text_config().pad_token_id
    else:
        placeholders, fill = (cfg.image_token_id, cfg.video_token_id), 0
    mask = None
    for tid in placeholders:
        if tid is not None:
            mask = input_ids == tid if mask is None else mask | (input_ids == tid)
    return input_ids if mask is None else torch.where(mask, fill, input_ids)


def head_logits(model, hidden):
    """lm_head over `hidden`, then what the model's own forward does to
    lm_head's output, so a verify row's logits are the ones the serial
    step reads from the wrapper. transformers 5.15.1, per family:

      gemma4, gemma4_text  tanh softcap by final_logit_softcapping, when set
      muse_glimmer         times output_multiplier, then the softcap
      granite              divided by logits_scaling

    Every other family drinkme serves (Qwen3, Qwen3.5, Qwen2.5) uses
    lm_head's output as it is. Without this a sampled speculative verify
    on gemma-4 drew from logits the serial loop never sees (greedy is
    unaffected: each transform is monotone). The ops and their order are
    the wrapper's (Gemma4ForConditionalGeneration.forward,
    MuseGlimmerForConditionalGeneration.forward, GraniteForCausalLM.forward),
    so the rows match its logits bit for bit
    (tests/test_serving_head_logits.py)."""
    logits = model.lm_head(hidden)
    kind = getattr(model.config, "model_type", None)
    text = model.config.get_text_config() if hasattr(model.config, "get_text_config") \
        else model.config
    if kind == "granite":
        return logits / text.logits_scaling
    if kind == "muse_glimmer":
        logits = logits * text.output_multiplier
    elif kind not in ("gemma4", "gemma4_text"):
        return logits
    cap = text.final_logit_softcapping
    if cap is None:
        return logits
    return torch.tanh(logits / cap) * cap


class Speculator:
    """The accept/commit state machine for ONE generation.

    Invariants, held at every cycle boundary:
      * the main cache holds KV for tokens 0..W-1; the token at position W has
        been emitted but NOT written (same invariant the serial loop keeps);
      * `h_last` is the trunk hidden at position W-1 — the row whose logits
        produced that token;
      * the head's KV holds entries up to W-2 and none past it: every entry
        this generation built, from the prefix-cache LCP on (0 after a cold
        prefill), and below it what the prefix slot kept from earlier
        requests (HeadKV), which may have holes. `_spans` names the entries
        its rows hold, and every one of them was computed from a TRUE trunk
        hidden state (entry p = (h_p, t_{p+1}), rope position p). The
        terminal k == 0 cycle leaves its entry to `history`.
    """

    def __init__(self, head, model, depth: int, sampled: bool = False,
                 lookup=None, rearm: int = REARM_DEFAULT,
                 bail_window: int = BAIL_WINDOW, bail_floor: float = BAIL_FLOOR):
        self.head = head
        self.model = model
        self.depth = depth
        # n-gram speculation: an ngram.NgramProposer, or None. head + lookup is the CHAINED
        # mode (lookup proposes; the head drafts the cycles it misses), lookup
        # alone is ngram-only and no head was even loaded. Whichever proposes,
        # the verify below is the one verify — serving/ngram.py's docstring
        # carries the two-line reason a point-mass proposal is exact here.
        self.lookup = lookup
        # the device the cycle builds its row/position tensors on. Read from
        # the model rather than from `h_last`, which ngram-only never has:
        # engines.py's own idiom (next(model.parameters()).device).
        self.device = next(model.parameters()).device
        # greedy = the argmax accept loop; sampled = rejection sampling.
        # Decided once per request by maybe_speculate, from the serial
        # sampler's own greedy test (temperature == 0).
        self.sampled = sampled
        # decided guard 2: rolling acceptance that sends the rest of the
        # request to the serial loop. engines.py watches `bailed`.
        # The head runs k forwards whether or not they are any good, so it
        # needs a floor to stop paying. A lookup that finds nothing proposes
        # nothing and the cycle degenerates to a plain M=1 step, so ngram-only
        # bails by construction (ngram.NGRAM_BAIL_FLOOR = 0 = no bail). The
        # CHAINED mode keeps the head's floor, and its window counts both
        # proposers' positions: a chained request that trips goes fully
        # serial, head included. `bail_window` / `bail_floor` are the plan's
        # (bail_from_env), the module defaults when built directly.
        self.bail = AcceptanceWindow(
            bail_window, bail_floor if head is not None else ngram.NGRAM_BAIL_FLOOR)
        # the backoff resets once a re-armed window has held this many
        # drafted positions: REARM_SURVIVE_WINDOWS windows, whatever the size
        self._survive = REARM_SURVIVE_WINDOWS * self.bail.size
        # The re-arm (module docstring). `drafting` is the state the engine
        # branches on: True = cycles, False = its own serial step. `rearm_after`
        # 0 is a one-way latch. `serial_left` counts the serial forwards still
        # owed before the next re-arm; `_stretch` is the current backoff
        # (rearm_after, doubling per consecutive trip, 0 once a re-armed
        # window survives); `_survived` the drafted positions the current
        # window has held above the floor since its re-arm. `_serial_h` /
        # `_serial_toks` are what serial_step keeps for the head: the trunk
        # hidden and the token of every serial position, in order.
        self.rearm_after = max(0, int(rearm))
        self.drafting = True
        self.serial_left = 0
        self._stretch = 0
        self._survived = 0
        self._serial_h: list = []
        self._serial_toks: list[int] = []
        self.trips = 0
        self.rearms = 0
        self.last_bail_rate: float | None = None
        self._n0 = 0
        # A2 (docs/serve-speculation.md): the audit only applies to the sampled
        # path — greedy has no distribution to audit, nothing to check
        # against a theorem that only speaks in probabilities.
        self.audit = SpeculativeAudit() if (sampled and _audit_enabled()) else None
        self.cache = None
        self.written = 0
        self.h_last = None
        # an image prompt (serving/image_prompt.py), or None: its M-RoPE
        # positions reach every trunk and head forward below, and `_shift`
        # (its delta; 0 for text) is what a head entry past the prompt adds
        # to its index to get its rope position
        self.image = None
        self._shift = 0
        self._cap = None
        self._rows = 0      # cache rows the last cycle committed (= m + 1)
        self._w_pre = 0     # `written` as of the last cycle's start
        self._hidden = None  # the last cycle's verify hidden states, [1, k + 1, H]
        # the entries the head's rows hold, as HeadKV.spans: [a, b) ranges
        # in row order. `head.run` takes a rope position (entry + `_shift`),
        # so the entry indices are kept here
        self._spans: list = []
        # (p, h_p, t_{p+1} [1, 1] on the device): the entry a k == 0 cycle
        # did not build, for `history`
        self._skipped = None
        self.cycles = 0
        self.drafted = 0
        self.accepted = 0
        # True when the engine replays this request's trunk forwards as CUDA
        # graphs (serving/cudagraph.py): every verify width and the serial
        # step alike, never one without the other
        self.graphs = False

    def _trunk(self, key: str, rows_in, eager):
        """The trunk forward of a cycle or serial step: a replayed graph
        when `graphs`, its outputs copied out of the graph's buffers (the
        next replay rewrites them, and h_last outlives it), else `eager()`.
        A capture that fails turns graphs off for the rest of the request."""
        if self.graphs:
            from . import cudagraph

            st = (cudagraph.hidden_step(self.model, self.cache) if key == "hidden"
                  else cudagraph.verify_step(self.model, self.cache, rows_in.shape[1],
                                             self.depth + 1))
            if st is not None:
                if key == "verify" and getattr(st, "cap", None) is None:
                    # captured just now, inside this cycle's capture: the
                    # DeltaNet windows and per-row states the capturing
                    # forward stored there are the graph's own buffers,
                    # rewritten by every replay of this width
                    st.cap = _ACTIVE
                st.ids.copy_(rows_in)
                hidden, logits = st.replay()
                if key == "verify" and _ACTIVE is not None and st.cap is not _ACTIVE:
                    # hand this cycle's rewind those buffers (read before the
                    # next replay of this width, as _restore_rows and finish are)
                    _ACTIVE.win.update(st.cap.win)
                    _ACTIVE.rec.update(st.cap.rec)
                    _ACTIVE.rows = st.cap.rows
                return hidden.clone(), logits.clone()
            self.graphs = False
        return eager()

    def precapture(self) -> bool:
        """Capture every trunk step this request can replay (the re-arm's
        hidden-keeping step when a head drafts, and each verify width 1 ..
        depth + 1) right after prefill, rather than as each first appears
        mid-stream (serving/cudagraph.py). A no-op for a width already
        captured on this cache. False when a capture failed (said once);
        the caller then decodes the request eager."""
        global _ACTIVE
        from . import cudagraph

        if self.head is not None and cudagraph.hidden_step(self.model, self.cache) is None:
            return False
        for rows in range(1, self.depth + 2):
            cap = _Capture()
            _ACTIVE = cap
            try:
                st = cudagraph.verify_step(self.model, self.cache, rows, self.depth + 1)
            finally:
                _ACTIVE = None
            if st is None:
                return False
            if getattr(st, "cap", None) is None:
                st.cap = cap
        return True

    @property
    def mode(self) -> str:
        """The name serving/ngram.resolve_mode gave this request."""
        if self.head is None:
            return "ngram"
        return "ngram+mtp" if self.lookup is not None else "mtp"

    @property
    def needs_hidden(self) -> bool:
        """Does prefill have to hand `begin` the trunk's hidden states?

        Only a head does — it is seeded from them. An ngram-only request
        therefore leaves prefill as the serial loop runs it (logits_to_keep=1,
        one lm_head row): speculation changes nothing before the decode
        loop."""
        return self.head is not None

    # -- after prefill ------------------------------------------------------
    def begin(self, cache, ids: list[int], hidden, start: int, image=None,
              history: HeadKV | None = None) -> None:
        """`hidden` covers prompt positions start..len(ids)-1 (start = the
        prefix-cache LCP, 0 for a cold prefill). Seed the head's KV over every
        entry the prompt can support — entry p needs t_{p+1}, so the prompt
        gives us entries start..n-2 — and leave entry n-1 for the first draft
        step, where t_n is the token the trunk just picked.

        A warm (prefix-reused) prefill only hands us the suffix's hidden
        states; below `start` the head has what `history` holds (`open`):
        the entries the prefix slot kept from earlier requests. A slot
        restored from the cold tier keeps none, and the head then attends
        over the suffix alone, which costs acceptance, never correctness:
        drafts are proposals."""
        self.open(cache, ids, image=image, history=history, start=start)
        if hidden is not None:
            self.seed(hidden, start, len(ids))

    def open(self, cache, ids: list[int], image=None, history: HeadKV | None = None,
             start: int = 0) -> None:
        """begin() without the hidden states: set up for this prompt, then
        let `seed` hand the head the trunk's hidden states span by span —
        the chunked prefill's order (serving/prefill.py), where no one
        forward holds the whole prompt's hidden states. `image` is the
        prompt's ImagePrompt when it carries images (`ids` is then its
        expanded ids).

        The head starts from `history`, the KV the prefix slot kept for its
        first `start` positions (engines.py has cropped it there), or empty
        without one. When the previous request ended at `start` (its tail
        sits at start - 1: this prompt extends everything the slot held),
        entry start - 1 = (h_{start-1}, t_start) is built here from the
        tail and this prompt's token, so the history has no hole where the
        suffix joins it."""
        self.cache = cache
        n = len(ids)
        self._ids = ids
        self.written = n
        self._n0 = n  # the prompt's length: the log lines say +N generated
        self.h_last = None
        self.image = image if image is not None and image.rope else None
        self._shift = 0 if self.image is None else self.image.delta
        if self.lookup is not None:
            # the proposer's context IS the prompt, whole — including the
            # prefix-cache LCP, which it never recomputed but did see. An
            # image run reads as its prefix key there (image_prompt.py), so
            # a lookup never matches across two images or proposes a row
            # of one.
            self.lookup.reset(ids if image is None else image.key_ids)
        self._cap = None
        self._rows = 0
        self._hidden = None
        self._spans = []
        self._skipped = None
        if self.head is None:
            return
        tail = None
        if history is None or history.cache is None:
            self.head.reset()
        else:
            self.head.cache, self.head.entries = history.cache, history.entries
            self._spans = [list(s) for s in history.spans]
            tail = history.tail
        if (tail is not None and tail[0] == start - 1 and start < n
                and (not self._spans or self._spans[-1][1] <= start - 1)):
            p, h = tail
            toks = torch.tensor([[ids[start]]], dtype=torch.long, device=h.device)
            # inside an image prompt the entry takes the trunk's M-RoPE rows,
            # as `seed` gives the prompt's other entries
            kw = ({} if self.image is None else
                  {"positions": self.image.head_positions(p, start, h.device)})
            self.head.run(h, toks, p, **kw)
            self._held(p, start)

    def _held(self, a: int, b: int) -> None:
        """The head's rows now also hold entries a..b-1, appended after
        every entry they held before."""
        if b <= a:
            return
        if self._spans and self._spans[-1][1] == a:
            self._spans[-1][1] = b
        else:
            self._spans.append([a, b])

    def _head_below(self, n: int) -> None:
        """Drop every head entry at or past n."""
        self.head.crop(self.head.entries - _rows_below(self._spans, n))
        self._spans = _spans_below(self._spans, n)

    def seed(self, hidden, a: int, b: int) -> None:
        """The trunk's hidden states for prompt positions a..b-1. Entry p
        needs t_{p+1}, so a span inside the prompt seeds entries a..b-1
        and the LAST span (b = n) seeds a..n-2 and keeps h_{n-1} for the
        first draft step — together, begin()'s one head.run over
        start..n-2, split where the prefill split."""
        n = len(self._ids)
        if b == n:
            self.h_last = hidden[:, -1:]
        if self.head is None:
            return
        stop = min(b, n - 1)  # the last entry the prompt can support, +1
        if stop > a:
            toks = torch.tensor([self._ids[a + 1:stop + 1]], dtype=torch.long,
                                device=hidden.device)
            # inside an image prompt the entries take the trunk's M-RoPE rows
            kw = ({} if self.image is None else
                  {"positions": self.image.head_positions(a, stop, hidden.device)})
            self.head.run(hidden[:, :stop - a], toks, a, **kw)
            self._held(a, stop)

    @property
    def bailed(self) -> bool:
        """The CURRENT window has tripped. Read by the engine once the last
        cycle's tokens are all appended, to hand the request to its serial
        loop (`trip`); a re-arm installs a fresh window, so this is False
        again the moment drafting resumes."""
        return self.bail.tripped

    # -- the adaptive bail's two transitions -------------------------------
    def trip(self) -> int:
        """Drafting -> serial, once every token the tripping cycle decided has
        been appended and `finish` has run (so `written` is exactly the
        cache and there is nothing left of the cycle to rewind). Returns the
        serial stretch: rearm_after on the first trip, doubling on each trip
        that follows a re-arm without a survived window, capped at REARM_CAP;
        0 = the latch (DRINKME_MTP_REARM=0), the rest of the request serial.
        One stderr line per trip, naming the rate and the stretch."""
        self.trips += 1
        self.drafting = False
        self.last_bail_rate = self.bail.rate_at_trip
        self._cap = None  # finish() ran; a stale capture must never rewind
        self._serial_h, self._serial_toks = [], []
        self._survived = 0
        # the head is not run during the stretch; rearm checks it was not
        self._head_entries = None if self.head is None else self.head.entries
        if self.rearm_after == 0:
            self._stretch = self.serial_left = 0
            tail = "the rest of this request decodes serially"
        else:
            self._stretch = (self.rearm_after if self._stretch == 0
                             else min(self._stretch * 2, REARM_CAP))
            self.serial_left = self._stretch
            tail = f"serial for the next {self._stretch} tokens"
        metrics.record_mtp_trip()
        print(f"[drinkme.mtp] adaptive bail at +{self.written - self._n0}: "
              f"acceptance {self.bail.rate_at_trip:.0%} over the last "
              f"{self.bail.size} draft positions (floor {self.bail.floor:.0%}) "
              f"— {tail}", file=sys.stderr)
        return self.serial_left

    @property
    def rearm_pending(self) -> bool:
        """Serial with a re-arm ahead: the engine's decode step must go
        through `serial_step` so the head's input is kept."""
        return not self.drafting and self.serial_left > 0

    @property
    def rearm_due(self) -> bool:
        """The serial stretch is spent: the token the engine holds is the
        last serially-sampled one, and the next step should be a cycle."""
        return (not self.drafting and self.rearm_after > 0
                and self.serial_left == 0 and bool(self._serial_toks))

    def serial_step(self, step_in, step_pos, position_ids=None):
        """The engine's own M=1 decode step while a re-arm is pending: the
        same trunk forward at the same cache position, read through
        forward_with_hidden so the row's hidden state is kept for the head.
        Returns the logits row the engine's `self.model(...).logits[0, -1]`
        would have returned — bitwise: both are the wrapper's own 1-row
        lm_head over the same last_hidden_state (tests pin it). The head is
        NOT run here — its entries are rebuilt in one batched pass at
        `rearm`, which is what keeps the serial stretch at serial cost.
        `position_ids`: the engine's step rows after an image prompt."""
        hidden, logits = self._trunk(
            "hidden", step_in,
            lambda: forward_with_hidden(self.model, step_in, self.cache, step_pos,
                                        last_row_only=True, position_ids=position_ids))
        self._serial_h.append(hidden[:, -1:])
        self._serial_toks.append(int(step_in[0, 0]))
        self.written += 1
        self.serial_left -= 1
        return logits[-1]

    def rearm(self) -> None:
        """Serial -> drafting. The engine holds t_W (sampled serially, not
        yet written) and the cache holds 0..W-1, exactly the cycle-boundary
        invariant; what is missing is the head's side of it. The R serial
        positions W0..W0+R-1 (W0 = `written` at the trip) were never fed to
        the head, so its KV stops at entry W0-2; entry p = (h_p, t_{p+1})
        needs h_{W0-1} (h_last as it stood at the trip) through h_{W0+R-2}
        (the first R-1 serial rows) against t_{W0}..t_{W0+R-1} (the R
        serial tokens) — one batched head pass, as at prefill — and h_last
        becomes the last serial row, h_{W0+R-1}. A chained lookup gets the
        same R tokens (its invariant: len(lookup) == written at every cycle
        boundary). Then a FRESH window: the old one's evidence was about the
        region that tripped it."""
        toks = self._serial_toks
        rows = self._serial_h
        start = self.written - len(toks) - 1  # W0 - 1: the first missing entry
        if self.head is not None:
            if self.head.entries != self._head_entries:
                # `entries` counts items, not positions (a warm prefill seeds
                # fewer than `written` of them), so the invariant is that the
                # stretch left the head exactly as the trip found it
                raise RuntimeError(
                    f"re-arm: the head's KV holds {self.head.entries} entries, "
                    f"{self._head_entries} at the trip: something ran the head "
                    "during the serial stretch — the re-arm plumbing is wrong")
            hs = torch.cat([self.h_last, *rows[:-1]], dim=1)
            ids = torch.tensor([toks], dtype=torch.long, device=hs.device)
            self.head.run(hs, ids, start + self._shift)
            self._held(start, start + len(toks))
        self.h_last = rows[-1]
        if self.lookup is not None:
            self.lookup.extend(toks)
        self._serial_h, self._serial_toks = [], []
        self.bail = AcceptanceWindow(self.bail.size, self.bail.floor)
        self._survived = 0
        self.drafting = True
        self.rearms += 1
        metrics.record_mtp_rearm()
        print(f"[drinkme.mtp] re-armed at +{self.written - self._n0} after "
              f"{len(toks)} serial tokens (trip {self.trips}): fresh window, "
              "drafting resumes", file=sys.stderr)

    # -- one draft/verify cycle --------------------------------------------
    def cycle(self, last_token: int, pick, budget: int,
              probs=None, generator=None) -> list[int]:
        """Draft, verify, commit. Returns the tokens the serial loop would
        have emitted next (m accepted drafts + one more — the main model's
        own token at the first disagreement on the greedy path, the residual
        resample or the bonus token on the sampled path; m+1 tokens, always
        at least one).

        `pick(row_logits, extra)` is the engine's own next-token choice for a
        row, with `extra` the tokens accepted so far this cycle, so penalties
        see exactly the id history the serial loop would have shown them.
        `probs(row_logits, extra)` is the same thing one level down — the
        distribution the serial sampler would draw from for that row
        (sampling.sample_probs) — and `generator` the request's RNG; both
        are read only when this speculator is `sampled`, and then the draft
        rows and the verify rows go through that one function, which is
        what makes q the thing p is divided by. `budget` caps how many
        tokens this cycle may emit (max_tokens and the cache allocation are
        both finite)."""
        global _ACTIVE
        # budget == 1 (the caller's last permitted token) degenerates to k = 0:
        # a plain M=1 step, no drafting, no head entry appended. That is always
        # a TERMINAL cycle — the caller stops on the token it returns — so the
        # head is one entry short of the invariant only for `history`, which
        # builds that entry (`_skipped`).
        k = max(0, min(self.depth, budget - 1))
        w = self.written
        drafts: list[int] = []
        qs = None
        # WHO DRAFTED, which the head-KV bookkeeping below turns on: "lookup"
        # (serving/ngram.py, no head entries appended), "head" (k entries), or
        # None for the degenerate k == 0 cycle, which drafts nothing.
        by = None
        if self.lookup is not None:
            self.lookup.append(last_token)  # the token this cycle continues from
            drafts = self.lookup.propose(k)
            if drafts:
                by, k = "lookup", len(drafts)
        # `rows` is the verify batch's input, [1, k + 1] on the device:
        # [last_token, d_0, ..., d_{k-1}]. The head builds it in place as it
        # drafts (MTPHead._draft) — no host round trip inside the chain; a
        # lookup proposal (host ints) or a scripted drafter (the equivalence
        # tests force a list) is lifted onto the device here instead.
        rows = None
        if k and not drafts and self.head is not None:
            by = "head"
            if self.sampled:
                def propose(row, so_far):
                    q = probs(row, so_far)
                    return torch.multinomial(q, 1, generator=generator), q

                out, qs = self.head.draft_sampled(self.h_last, last_token,
                                                  w - 1 + self._shift, k, propose)
            else:
                out = self.head.draft(self.h_last, last_token, w - 1 + self._shift, k)
            if torch.is_tensor(out):
                rows = out
            else:
                drafts = list(out)
        if rows is None:
            if not drafts:
                k = 0  # no proposal: an M=1 verify, which IS a plain decode step
            rows = torch.tensor([[last_token, *drafts]], dtype=torch.long,
                                device=self.device)
        pos = torch.arange(w, w + k + 1, device=self.device)
        # after an image prompt the verify rows sit at [p, p+delta x3]
        # (image_prompt.step_rows, built on the device); a text prompt
        # passes no position_ids at all
        rope = {} if self.image is None else {"position_ids": step_rows(pos, self._shift)}

        cap = _Capture()
        _snapshot_sliding(self.cache, cap, k + 1)  # surface 1b, before the write
        _ACTIVE = cap
        try:
            hidden, logits = self._trunk(
                "verify", rows, lambda: forward_with_hidden(self.model, rows, self.cache, pos, **rope))
        finally:
            _ACTIVE = None
        cap.rows = k + 1  # attention layers wrote M rows even if no DeltaNet ran
        if not drafts and k:
            # THE cycle's one device->host read: the drafts, now that the
            # verify forward is in the queue behind the chain. Everything
            # below decides on the host (the accept rule, the emission, the
            # rewind depth, the audit) and needs them as ints.
            drafts = rows[0, 1:].tolist()

        if self.sampled:
            # every verify row through the SAME transform the drafts used,
            # row i conditioned on drafts 0..i-1 (the only case it is read
            # in), then the accept/resample rule (serving/speculative.py)
            p = [probs(logits[i], drafts[:i]) for i in range(k + 1)]
            if qs is None:
                # a lookup proposal is a POINT MASS, and its q rows can only
                # be built once a p row exists to take dtype/device/V from
                # (speculative.point_mass_rows carries the exactness
                # argument). [] when nothing was drafted, which is what
                # speculative_sample wants for depth 0.
                qs = point_mass_rows(p[0], drafts)
            sink = (_FanOutAudit(self.audit, _AUDIT_CUMULATIVE)
                    if self.audit is not None else None)
            emitted, m = speculative_sample(p, qs, drafts, generator, audit=sink)
            if self.audit is not None:
                _audit_tick()
        else:
            emitted: list[int] = []
            m = 0
            while True:
                tok = pick(logits[m], emitted)
                if m < k and tok == drafts[m]:
                    emitted.append(tok)
                    m += 1
                    continue
                emitted.append(tok)
                break

        _restore_rows(self.cache, cap, m + 1)
        # The head's KV: keep ONLY the cycle's first entry (W-1 — it was built
        # from the trunk's own h_last and an already-committed token), drop the
        # k-1 recursive ones, and rebuild the accepted entries W..W+m-1 from
        # the TRUE trunk hidden states the verify pass just produced. That one
        # extra m-row head pass is what keeps the head's attention history the
        # same one it was trained on (and the one vLLM's proposer maintains):
        # a plain crop would leave the accepted entries keyed on the head's own
        # recursive guesses forever, which costs acceptance every cycle after.
        #
        # When the LOOKUP drafted (chained mode), the head appended nothing,
        # so entry W-1 was never built either — and the rebuild simply starts
        # one entry earlier, from the two things the Speculator is already
        # holding: h_last (the trunk hidden at W-1) and last_token (t_W).
        # Entry p = (h_p, t_{p+1}) either way, so the head's history is the
        # same one it would have had if it had drafted the cycle itself.
        # The rebuild's token ids are VIEWS of the verify rows already on the
        # device (rows[0, 1:] are the drafts, rows[0, 0] is last_token): the
        # same values the host list holds, with no second upload.
        if by == "head":
            self.head.crop(max(0, k - 1))
            if m:
                self.head.run(hidden[:, :m], rows[:, 1:1 + m], w + self._shift)
            self._held(w - 1, w + m)  # entry W-1, then the m accepted
        elif by == "lookup" and self.head is not None:
            self.head.run(torch.cat((self.h_last, hidden[:, :m]), dim=1),
                          rows[:, :1 + m], w - 1 + self._shift)
            self._held(w - 1, w + m)
        elif self.head is not None:
            # by is None: k == 0, the terminal cycle, a plain M=1 step. The
            # caller stops on the token this returns, and entry W-1 is left
            # to `history`, which builds it when a prefix slot is to keep
            # the head's KV
            self._skipped = (w - 1, self.h_last, rows[:, :1])
        if self.lookup is not None:
            # the invariant: len(lookup) == self.written at every cycle
            # boundary. `last_token` went in at the top of this cycle, so
            # what is left is everything emitted except the last token,
            # which the NEXT cycle appends when it is handed it.
            self.lookup.extend(emitted[:-1])
        self._cap, self._rows, self._w_pre = cap, m + 1, w
        self._hidden = hidden  # the tail `history` reads if a stop lands mid-cycle
        self.written = w + m + 1
        self.h_last = hidden[:, m:m + 1]
        self.cycles += 1
        self.drafted += k
        self.accepted += m
        metrics.record_spec(k, m)  # server-lifetime totals; s['drafted']/s['accepted']
        # above stay this REQUEST's own accounting, printed in the audit line
        # decided guard 2: the window trips here; the engine reads `bailed`
        # once this cycle's tokens are all appended and calls `trip`. A
        # re-armed window that holds `_survive` drafted positions above the
        # floor (REARM_SURVIVE_WINDOWS windows) has outlived the region that
        # tripped it: backoff reset.
        if not self.bail.observe(m, k) and self._stretch and k:
            self._survived += k
            if self._survived >= self._survive:
                self._stretch = 0
        return emitted

    # -- generation over ----------------------------------------------------
    def finish(self, appended: int) -> int:
        """A generation can stop mid-cycle (EOS, a stop string, max_tokens, a
        disconnected client) with tokens still queued. Their KV IS in the
        cache, so the cache must be rewound to exactly what the caller kept —
        otherwise the slot's ids would describe a cache that holds more than it
        claims, which is the corruption the prefix cache's extends-only rule
        exists to prevent. Returns the true written length.

        The head's entries go back with it: a cycle rebuilt one for every
        accepted draft, and entry p stands on t_{p+1}, which the cache now
        holds only for p < written - 1."""
        if self._cap is not None:
            keep = min(appended, self._rows - 1) + 1
            _restore_rows(self.cache, self._cap, keep)
            self.written = self._w_pre + keep
            self._rows = keep
        if self.head is not None:
            self._head_below(self.written - 1)
        return self.written

    def history(self, written: int) -> HeadKV | None:
        """The head's KV for a cache that holds `written` positions, for the
        prefix slot to keep (HeadKV), and the head detached from it, so a
        slot the engine drops or evicts frees its KV rather than leaving it
        resident on the shared head. None without a head.

        Entries 0..written-2 at most — entry written-2 built here when the
        request ended on a k == 0 cycle, which does not build its entry
        W-1 — and the tail h_{written-1} when this speculator saw it: the
        last serial row of a re-arm stretch, the verify row a stop landed
        on (finish ran), or `h_last`. Not on the latch's serial loop
        (DRINKME_MTP_REARM=0), whose steps are the engine's own and never
        reach here: there `written` is past what this speculator counted."""
        if self.head is None:
            return None
        self._head_below(written - 1)
        if self._skipped is not None:
            p, hp, tok = self._skipped
            self._skipped = None
            if p == written - 2 and (not self._spans or self._spans[-1][1] <= p):
                self.head.run(hp, tok, p + self._shift)
                self._held(p, p + 1)
        h = None
        if written == self.written:
            if not self.drafting and self._serial_h:
                h = self._serial_h[-1]
            elif self.drafting and self._cap is not None:
                j = written - 1 - self._w_pre
                if 0 <= j < self._hidden.shape[1]:
                    h = self._hidden[:, j:j + 1]
            else:
                h = self.h_last
        tail = None if h is None else (written - 1, h.clone())
        kv = HeadKV(self.head.cache, self.head.entries, self._spans, tail)
        self.head.cache, self.head.entries = None, 0
        self._spans = []
        return kv

    def stats(self) -> dict:
        audit = None
        if self.audit is not None:
            a = self.audit
            audit = {"decided": a.decided, "accepted": a.accepted,
                     "expected_sum": a.expected_sum,
                     "resample_violations": a.resample_violations,
                     "zero_mass_emits": a.zero_mass_emits,
                     "actual_rate": a.actual_rate, "expected_rate": a.expected_rate,
                     "violations": a.violations}
        return {"mode": self.mode,
                # n-gram speculation: the lookup's own accounting — how often it was asked,
                # how often it found anything, how many tokens it proposed.
                # None when this request has no lookup proposer.
                "lookup": self.lookup.stats() if self.lookup is not None else None,
                "cycles": self.cycles, "drafted": self.drafted,
                "accepted": self.accepted,
                "acceptance": round(self.accepted / self.drafted, 3) if self.drafted else 0.0,
                "sampled": self.sampled,
                # the adaptive bail: whether the engine ever handed this
                # request to the serial loop, the rolling rate of the LAST
                # trip (None = it never did), how many times it tripped and
                # re-armed, and whether the request ended on the serial loop
                "bailed": self.trips > 0, "bail_rate": self.last_bail_rate,
                "trips": self.trips, "rearms": self.rearms,
                "serial_at_end": not self.drafting,
                # A2 (docs/serve-speculation.md): None when the audit is off (greedy
                # requests, or DRINKME_SPEC_AUDIT=0)
                "audit": audit}


class SpecPlan(NamedTuple):
    """Everything the environment decides about speculation, resolved ONCE
    per engine instead of once per request (eight env reads, otherwise on
    every generation). `mode` is ngram.resolve_mode's
    answer, `depth` the cycle's draft budget, tokens/hi/lo the lookup's
    (num_speculative_tokens, prompt_lookup_max, prompt_lookup_min), and
    rearm / bail_window / bail_floor the adaptive bail's three knobs."""

    mode: str
    depth: int
    tokens: int
    hi: int
    lo: int
    rearm: int = REARM_DEFAULT  # DRINKME_MTP_REARM (rearm_from_env)
    bail_window: int = BAIL_WINDOW  # DRINKME_MTP_BAIL_WINDOW (bail_from_env)
    bail_floor: float = BAIL_FLOOR  # DRINKME_MTP_BAIL_FLOOR (bail_from_env)


def spec_plan(has_head: bool, mtp_depth: int | None = None) -> SpecPlan:
    """Read the environment. `mtp_depth` is depth_from_env()'s answer (None =
    AUTO, k > 0 = a pinned depth).

    The DEPTH stays the head's whenever a head is resident — so `mtp` and
    `ngram+mtp` run the verify batch MTP is gated on, and the lookup is
    capped to it — and is the lookup's own num_speculative_tokens when there
    is no head to cap against."""
    mode = ngram.resolve_mode(has_head)
    tokens = hi = lo = 0
    if mode in ("ngram", "ngram+mtp"):
        tokens, hi, lo = ngram.params_from_env(FUSED_M_MAX - 1)
    depth = DEFAULT_DEPTH if mtp_depth is None else mtp_depth
    window, floor = bail_from_env()
    return SpecPlan(mode, tokens if mode == "ngram" else depth, tokens, hi, lo,
                    rearm_from_env(), window, floor)


def maybe_speculate(head, model, params, depth: int | None = None, plan=None):
    """A Speculator for this request, or None to use the serial loop.

    Greedy requests take the argmax accept loop (the trunk's argmax per
    row); sampled ones — by the serial sampler's own test, temperature != 0 —
    take rejection sampling (distribution-identical; module docstring).
    Constrained output is out: the constraint walks a grammar token by
    token, which is a per-step decision the batch cannot make ahead of
    itself.

    WHICH PROPOSER (n-gram speculation) is serving/ngram.resolve_mode's decision, over
    DRINKME_SPEC, carried here as a SpecPlan; `head` being
    None already reflects it too, because a mode that cannot use a head does
    not load one (ngram.wants_head, engines._mtp_head). `plan` is the
    engine's cached one; `depth` (or neither) makes a fresh one, which is
    what a test or a bench calling this directly wants."""
    if plan is None:
        plan = spec_plan(head is not None,
                         depth if depth is not None else depth_from_env())
    if plan.mode == "off":
        return None
    if params.output_schema is not None:
        _warn_once("constrained", f"speculative decoding is on ({plan.mode}) "
                                  "but this request is grammar-constrained — "
                                  "serving it serially")
        return None
    lookup = None
    if plan.mode in ("ngram", "ngram+mtp"):
        lookup = ngram.NgramProposer(plan.tokens, plan.hi, plan.lo)
    return Speculator(None if plan.mode == "ngram" else head, model, plan.depth,
                      sampled=params.temperature != 0, lookup=lookup,
                      rearm=plan.rearm, bail_window=plan.bail_window,
                      bail_floor=plan.bail_floor)
