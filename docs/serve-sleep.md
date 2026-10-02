# Sleep and wake

The torch runtime can release accelerator memory while keeping the server,
tokenizer, and metadata available. Level 1 parks weights in host RAM; level 2
also frees the host copies and reloads them on wake. MLX returns 501.

## Routes

```sh
curl -X POST http://127.0.0.1:3215/sleep
curl -X POST 'http://127.0.0.1:3215/sleep?level=2'
curl -X POST http://127.0.0.1:3215/wake_up
curl http://127.0.0.1:3215/health
```

Sleep and wake require the configured bearer token and are idempotent.
They acquire the generation lock without waiting; an active generation
produces HTTP 409 with `code: "generation_in_flight"`. A caller can retry
once generation finishes. Sleep can deepen from level 1 to 2; level 2 to 1
requires waking first and otherwise returns 409.

Health reports state, level, sleep duration, bytes parked, tensor counts,
persisted/dropped slots, and transition time. While asleep:

| Route | Behavior |
|---|---|
| `/health`, `/metrics`, `/v1/models`, `/tokenizer_info` | Available |
| `/tokenize`, `/detokenize`, `/v1/messages/count_tokens` | Available |
| Generation endpoints | HTTP 503, `model_asleep` |

The error includes a wake URL based on the request's Host header. Anthropic
errors use `model_asleep` rather than a retryable overload error: a sleeping
server needs an explicit wake.

## Memory transitions

[`serving/sleep.py`](../src/drinkme/serving/sleep.py) moves tensors one at a time,
limiting extra transfer memory to one tensor:

- The packed arrays in each packed Linear's `p` dictionary: a coded tensor's
  streams, directory, palette tables and schedule; a raw fallback's weight.
- Raw parameters and buffers, preserving tied embeddings and heads.
- MTP head tensors, including its sub-pack or reduced-vocabulary projection.
- The vision tower's tensors, on a model with image input, packed and raw
  alike.

The tokenizer, capability probe, pack metadata, disk-slot index, and lazy CPU
reference weights stay on the host. Level-1 wake restores resident buffers by
host-to-device copy. Level-2 wake calls the original compressed or stock
loader, which builds the vision tower again and reruns its boot self-test
([image input](serve.md#boot-lines-and-the-self-test)).

Prefix slots are persisted through `serving/slotstore.py`, then released.
Wake restores the parked slots. With `DRINKME_SLOT_DIR=off`, slots are dropped;
the response reports how many were lost.

`DRINKME_SLEEP_PIN=1` uses pinned host memory for faster transfer. It is off by
default. On unified-memory devices, host copies still occupy the physical pool
needed by other models; pinning further restricts how that memory can be used.

## Idle timer

```sh
drinkme serve --model Qwen3.8-27B --sleep-on-idle 1800
```

The default is zero (disabled). `DRINKME_SLEEP_ON_IDLE_S` is the environment
equivalent; the flag wins. Invalid values warn and fall back to zero.

The timer tracks generation start and completion. Health checks and metrics
scrapes do not reset it. It uses the same non-blocking lock, so it cannot
interrupt generation and retries on a later tick when busy.

## Service integration

An `ExecStartPre` in another model's systemd unit can request sleep before
starting, then fall back to stopping the old service if it is busy or unreachable.
Drinkme does not ship a unit file. Account for host-memory residency when
choosing level 1 versus level 2, especially on an APU.

## Verification

`bench/sleep_wake_verify.py` checks bit-identical prefill logits before and
after wake, released device bytes, host-memory high-water mark, and wake time
against a cold load in the same run.

```sh
uv run --no-sync python bench/sleep_wake_verify.py \
  --pack /path/to/pack --ctx 32768 --rules 900 --level2
```

Run the gate in a separate test process, not against a server someone is
using ([running servers](dev-environment.md#running-servers)).
