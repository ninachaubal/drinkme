# Prefix cache

The torch runtime keeps whole conversation caches in accelerator memory and
persists evicted slots to disk. Set `--prefix-slots N` or
`DRINKME_PREFIX_SLOTS=N` (default 1; the flag wins); `0` turns the prefix
cache off, so every request builds its own cache. An explicit slot count
above 1 that exceeds the memory budget is refused at load; the default single
slot only warns. MLX currently has no prefix reuse.

When a prompt extends a cached conversation, the engine reuses its state and
prefills the new suffix. When a prompt shares a prefix with a slot and then
parts from it, which is what a re-rendered history does, the engine reuses the
slot up to that prefix through [context checkpoints](#context-checkpoints). It
first checks resident slots, then disk. Eviction writes the old slot;
`SIGTERM` persists live slots for restart.
`usage.prompt_tokens_details.cached_tokens` reports reuse.

## Identity and reuse

Persisted slots bind to the pack's manifest digest, checkpoint identity,
Transformers version, and slot format. Mismatches are skipped with a log
message. Re-packing a model changes its key and starts its slot store cold.
See [pack identity](pack-format.md).

## Context checkpoints

A chat template re-renders history differently from what the model wrote.
gemma-4 drops the empty `<|channel>thought\n<channel|>` its thinking-off
generation prompt ends in, Qwen3.5 trims a reply's trailing newline, and
Muse-Glimmer's reply channels come back as another channel. The next prompt
then shares a prefix with the slot and parts from it, a few tokens before or
after the previous prompt's end.

Full-attention KV rewinds to that point by length. Three kinds of state
cannot rewind: a sliding-window layer's ring (gemma-4, Muse-Glimmer), which
shifts rows out once the window is full, and a gated-DeltaNet layer's
recurrent and conv states (Qwen3.5), which have no sequence dimension. A
context checkpoint is a snapshot of only those, taken during a prefill
([`serving/ctx_checkpoints.py`](../src/drinkme/serving/ctx_checkpoints.py)).
drinkme follows llama.cpp's `--ctx-checkpoints`; the module's docstring
compares the two rule by rule.

- **Where:** at the start of the request's last user message, at
  `n_prompt - 4 - 512` and `n_prompt - 4` (llama.cpp's two near the end), and
  at `n_prompt` (drinkme's addition, where a history that keeps the whole
  generation prompt parts). The prefill is split at each one, so a checkpoint
  costs up to three extra forwards of a few tokens. A prompt that fits inside
  every sliding window, on a model with no recurrent layer (gemma-4 up to
  1,024 tokens, Muse-Glimmer up to 2,048), is not split: its rings still hold
  every earlier position after the prefill, and the checkpoints are copied
  out of them.
- **How many:** up to `--ctx-checkpoints N` per slot (`DRINKME_CTX_CHECKPOINTS`,
  default 32, the flag wins). Other requests' checkpoints closer than 8,192
  tokens to the previous one are dropped, so a slot holds at most five at the
  default context of 8,192.
- **Restore:** the newest checkpoint at or below the shared prefix, leaving at
  least one prompt token to compute. A cache of full attention only, or one
  whose rings have not wrapped yet, rewinds by length instead.
- **Slot choice:** llama.cpp's rules. A slot the prompt parts from is used when
  it serves more than a tenth of the prompt; a slot that would keep less than
  half of itself is saved to disk first, and the disk tier is asked for a
  better match. drinkme does not take such a slot while an untouched slot is
  free.
- **Images:** no checkpoint lands inside an image run that a prefill span may
  not cut (gemma-4's bidirectional runs, and causal runs that fit the prefill
  chunk); it moves to the run's start. Images inside the reused prefix skip
  the tower.
- **Disk and sleep:** a slot's checkpoints are written with it
  (`checkpoints.safetensors` beside `cache.safetensors`) and come back with it,
  including after a sleep. A stored slot the prompt parts from is served up to
  what its checkpoints reach.
- **Off:** `--ctx-checkpoints 0` restores the extends-only rule: a prompt that
  parts from a slot prefills in full.

A checkpoint lives in accelerator memory beside its slot, and the engine's
load-time memory check and the no-model picker charge the most a slot can
hold. One checkpoint of each menu model, measured on the CPU from its real
shapes by `bench/ctx_checkpoint_sizes.py` (sliding rings and conv states in
bf16, recurrent states in float32):

| Model | State a rewind cannot reach | One checkpoint | Most per slot at ctx 8,192 (5) | at 32,768 (8) |
|---|---|---|---|---|
| gemma-4-31B-it | 50 sliding layers, window 1,024 | 800.0 MiB | 3.91 GiB | 6.25 GiB |
| Qwen3.8-27B | 48 DeltaNet layers | 147.75 MiB | 0.72 GiB | 1.15 GiB |
| Muse-Glimmer-30B | 39 sliding layers, window 2,048 | 78.0 MiB | 0.38 GiB | 0.61 GiB |
| MiMo-V2.6-Distill-Qwen-9B | 24 DeltaNet layers | 49.5 MiB | 0.24 GiB | 0.39 GiB |
| Qwen3 dense, Qwen2.5-72B | none | none taken | 0 | 0 |

`drinkme_prefix_slot_events_total{event="restore"}` counts requests served this
way, and the log names the point and the checkpoint.

## Storage and sizing

| Variable | Default | Purpose |
|---|---|---|
| `DRINKME_SLOT_DIR` | `~/.cache/drinkme/slots` | Store path; `off` disables persistence |
| `DRINKME_SLOT_DISK_GIB` | `64` | LRU disk cap |
| `DRINKME_SLOT_RAM_GIB` | `0` | Optional unpinned host-memory tier between accelerator and disk |

A slot occupies its allocated width, even for a short conversation:
`fixed state + sliding-window rings + bytes per token × context allocation`.
A sliding-window layer holds `min(window, context)` rows at any context, so
its ring counts at that size. Qwen3.8-27B uses 147.75 MiB of DeltaNet state
(its recurrent state is float32) plus 64 KiB/token: 2.14 GiB at context
32768. The dense Qwen3-8B slot at that context uses 4.50 GiB. gemma-4-31B's
50 rings hold 800 MiB and its 10 full layers take 80 KiB/token: 1.41 GiB at
context 8192, 3.28 GiB at 32768. With an MTP head loaded, a slot also keeps
the head's KV between requests, cropped at the reuse point like the rest of
the slot: 4 KiB/token more on Qwen3.8-27B (2 × 4 KV heads × 256 × 2 bytes),
2.27 GiB at context 32768 in all. The disk tier does not store it, so the
head of a slot restored from disk starts at the reuse point. The boot log's
`prefix cache:` line prints the three terms the engine measured, and the
head's share of the per-token term. On unified memory, a host-memory tier
competes with model residency.

The allocation bounds memory only. Each decode step attends over the tokens
the slot holds, not over the allocation
([`serving/kvcache.py`](../src/drinkme/serving/kvcache.py)): a 204-token
request in a 4096-wide slot decodes at the speed of a 204-wide cache.

For systemd, allow time for persistence on shutdown. `TimeoutStopSec=120`
covered two 27B slots at context 32768 on Strix Halo (maintainer run);
account for your storage speed and slot count.

## Images

A prompt's images are keyed on their pixel content, so a conversation that
sends the same image again extends its slot and skips the vision tower for
it, and two different images of the same size never share a slot. The disk
tier and sleep/wake keep the same keys
([image input](serve.md#prefix-caching-with-images)).

## Qwen3-8B structured-output boundary

Qwen3-8B opens its own thinking tag during generation; Qwen3.8-27B's template
opens it in the prompt. Structured output forces thinking off for that turn
because the JSON grammar cannot accept a thinking preamble.

On the 8B, this inserts an empty `<think></think>` pair which the template
strips when rendering the next turn's history. The prompt parts from the
cached prefix at that pair; the 8B is all full attention, so the next turn
rewinds to it and prefills the rest (with `--ctx-checkpoints 0` it prefills
again in full). The 27B template does not have the same behavior.
`bench/prefix_slots_verify.py` documents the checks.

Changing `enable_thinking` through a profile can likewise re-render history.
Keep that setting consistent within a conversation to preserve prefix reuse.

## gemma-4's history

gemma-4-31B-it's chat template renders a past model turn differently from
what the model wrote, so over the wire no later turn extends a slot: every
request prefills the whole conversation, and every image in it runs the
tower again (about 0.22 s each on Strix Halo, plus the image's share of the
prefill). With thinking off, the generation prompt ends in an empty
`<|channel>thought\n<channel|>` pair, which the model also writes after a
tool response, and the template leaves that pair out of history. With
thinking on and the reasoning sent back, the template adds a newline before
`<channel|>` that the model did not write.

## Muse-Glimmer's reasoning messages

Muse-Glimmer writes its reasoning as a message addressed `to=self`, then its
answer `to=user` ([addressed messages](serve-tool-formats.md#addressed-messages)).
drinkme returns the first as reasoning (`reasoning_content`, a `thinking`
block, a `reasoning` item) and the second as the answer, byte for byte. The
chat template writes a history turn's reasoning message only from
`reasoning_content`, so the next prompt extends the cached slot only when the
client sends the reasoning back unchanged, with the tool calls in a tool loop.

A client that drops the reasoning, or trims its whitespace, re-renders a
turn the model did not write, and the next request prefills the whole
conversation again, re-running the vision tower for every image in it. Two
more shapes cannot be rendered back by the template: an answer message
written before a tool call in the same turn (the template renders no content
beside tool calls), and reasoning and answer messages that alternate within
one turn. `tests/test_serving_glimmer_channel.py` pins where each one leaves
the slot.
