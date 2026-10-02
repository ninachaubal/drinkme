"""The serve scaffold — everything between the socket and the weights:

  http.py      OpenAI-compatible HTTP layer (stdlib http.server, zero new deps)
  engine.py    the seam a model must implement; FakeEngine proves the contract
  engines.py   the real thing: HFEngine, one class for both arms (stock bf16
               and compressed), loaded whole or streamed onto a meta skeleton
  mtp.py       MTP speculative decoding (DRINKME_MTP_DEPTH, on by default when the
               checkpoint carries a head): the checkpoint's own draft head,
               verify-and-accept, same tokens
  ngram.py     n-gram (prompt-lookup) speculation: a proposer made out of the
               context itself, no draft model — DRINKME_SPEC picks between it
               and the head, and both go through mtp.py's one verify
  sampling.py  pure logits -> token sampling + stop-sequence scanning
  constrain.py the exact grammar constraint behind response_format
  template.py  chat templating over tokenizer.apply_chat_template
  detok.py     incremental detokenization: prefix diff + grapheme hold-back
  think.py     the think channel: reasoning split off content, incrementally
  tools.py     tool calling over Qwen's <tool_call> blocks, both directions
  slotstore.py SSD-persisted prefix slots: the cold tier under engines.py's
               live slots
  ctx_checkpoints.py
               context checkpoints (llama.cpp's --ctx-checkpoints): a slot
               is reused up to a prefix a prompt shares with it, from a
               snapshot of the state a rewind by length cannot reach
  sleep.py     sleep/wake: park a loaded model in host RAM and resume it
               instead of tearing the process down
  kernel_route.py
               the runtime's kernel routes over a loaded tree — the DeltaNet
               recurrence (deltanet.py) and the narrow GEMV — in one call
               that serve's loaders and every bench arm make, with a record
               of what it did
  metrics.py   in-process metrics registry, rendered as Prometheus text
               exposition, no client library
  capability.py the off-menu capability announcement: a static probe over a
               model's chat template for reasoning + tool-call dialect
  generation_profiles.py
               alias ids and named sampling/template-kwarg overlays (generation
               profiles) exposed as `<id>:<profile>` on /v1/models
  gen_config.py generation_config.json as the request-defaults source
  speculative.py rejection-sampling speculative decoding: the accept/resample
               rule that makes MTP exact at any temperature
  draft_vocab.py the draft diet: a draft step argmaxes over a subset of the
               vocab instead of the full projection
  messages.py  the Anthropic Messages API as a format adapter over the same
               generation core the OpenAI boundary uses
  tool_formats.py the tool-format table: one row per output dialect, data
               not branches

Everything except engines.py and mtp.py is testable with no model, no GPU, and
no network: HTTP tests drive FakeEngine, sampling runs torch-on-CPU, templating
runs against a fake tokenizer. Those two hold the same bar minus "no model":
their tests build a toy causal LM in-process (a hybrid qwen3_5 with a real MTP
head, for mtp.py) — still CPU-only, still no network.
"""
