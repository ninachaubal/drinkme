# Responses API

`POST /v1/responses` adapts text and function-tool requests to the shared
generation core. Use `http://127.0.0.1:3215/v1` as the client base URL and any
non-empty key, or the configured bearer token when `--auth` is enabled.

The adapter is [`serving/responses.py`](../src/drinkme/serving/responses.py).
Cross-API tests in `tests/test_serving_messages.py` compare engine inputs for
the same conversation across Chat Completions, Messages, and Responses.

## Request mapping

| you send | the model sees |
|---|---|
| `instructions` | one `system` turn, first |
| `input` as a string | one `user` turn |
| `input[]` `message` (`user` / `system` / `developer` / `assistant`) | a turn of that role; `developer` becomes `system`; a `system` item stands where it is. Content is a string or parts: `input_text` / `output_text` / `refusal` carry text and **concatenate with no separator** (the same join `/v1/chat/completions` applies to a parts array); a `user` message may also carry `input_image` parts on a vision model ([image input](serve.md#image-input)), kept in wire order |
| `input[]` `function_call` (`call_id`, `name`, `arguments`) | a `tool_calls` entry on the assistant turn; `arguments` is parsed from its JSON string |
| `input[]` `function_call_output` (`call_id`, `output`) | a `tool` turn; `output` is a string or `input_text` parts |
| `input[]` `reasoning` | the assistant turn's `reasoning_content` — read from `content[]` (`reasoning_text`), else `summary[]` (`summary_text`, where this server puts it on the way out); parts join with a blank line. `encrypted_content` is dropped (nothing readable) |
| `tools[]` `{type: "function", name, description, parameters, strict}` | the tool definitions the model's own chat template renders; `strict` is accepted and ignored |
| `tool_choice` `"auto"` / `"none"` | tools offered / withheld |
| `max_output_tokens` | `max_tokens` (default 4096; not required) |
| `temperature`, `top_p` | sampling; `top_k`, `repetition_penalty`, `presence_penalty`, `frequency_penalty` and `seed` are taken by the same names as the chat dialect's extensions |
| `text.format` `{type: "json_object"}` / `{type: "json_schema", schema}` | grammar-constrained output — the same path as `response_format` on `/v1/chat/completions` (`serving/constrain.py`); forces thinking off; cannot be combined with `tools` |
| `reasoning.effort` | the chat template's `reasoning_effort` (Qwen3.8's takes `xhigh` / `medium` / `low`; anything else is the template's own 400) |
| `chat_template_kwargs` | the chat dialect's extension, same rules (`enable_thinking`, …) |
| `model` | resolves like the other dialects: the served id, any `--served-model-name` alias, `<id>:<profile>` |

Consecutive reasoning, assistant-message, and function-call items merge into
one assistant turn. Repeating an already-filled field starts another turn.
Include reasoning items when sending history back: the 27B prefix cache needs
them to extend across turns.

### Accepted and ignored

`store` (nothing is stored; the response always says `store: false`),
`metadata`, `user`, `truncation`, `parallel_tool_calls`, `include`,
`stream_options`, `service_tier`, `safety_identifier`, `prompt_cache_key`,
`top_logprobs`, `text.verbosity`, `reasoning.summary`. Those the response
object carries are echoed back as sent.

### Unsupported fields (HTTP 400)

- `previous_response_id`, `conversation` — this server keeps no response
  state; send the full input every turn, as clients such as Codex do with
  `store: false`.
- `prompt` (stored prompt templates), `background` — server-side objects
  that do not exist here.
- `tool_choice: "required"` and `{type: "function", name}` — forced tool
  use. All three dialects reject forced calls the same way (the
  [capability matrix](serve.md#api-capability-matrix)).
- Hosted tools (`web_search`, `file_search`, `code_interpreter`, `mcp`,
  `computer_use_preview`, `image_generation`, `local_shell`, `custom`, …)
  are unsupported. Your client executes function tools. A `namespace` container
  is accepted: its functions are offered under flat names, and returned calls
  carry `namespace`. Duplicate names across namespaces or at the top level are
  rejected because the model cannot distinguish them.
- `input_file` and `input_audio` parts, and `input_image` on a model
  without [image input](serve.md#image-input) or with a `file_id` (no Files
  API); the message names the item and part index. On a vision model an
  `input_image` in a user message is accepted (a data URL, an http(s) URL,
  or a `file://` path under `--media-path`), with `detail`.
- Any input item type other than `message`, `function_call`,
  `function_call_output`, `reasoning` — `item_reference` (no state), the
  hosted tool calls and their outputs.
- `tools` on a model whose tool-call dialect is unparsed or untested — the
  same capability gate, same message, same `unsupported_tool_format` code
  as `/v1/chat/completions` ([serve-tool-formats.md](serve-tool-formats.md)).

Errors use the same `error` object as Chat Completions, with `message`, `type`,
`param`, and `code` fields.

## Response

```json
{"id": "resp_…", "object": "response", "created_at": 1789276999,
 "status": "completed", "error": null, "incomplete_details": null,
 "model": "Qwen/Qwen3-8B",
 "output": [
   {"id": "rs_…", "type": "reasoning", "status": "completed",
    "summary": [{"type": "summary_text", "text": "the user wants weather"}]},
   {"id": "msg_…", "type": "message", "role": "assistant", "status": "completed",
    "content": [{"type": "output_text", "text": "I will check.\n", "annotations": []}]},
   {"id": "fc_…", "type": "function_call", "call_id": "call_…", "status": "completed",
    "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}],
 "usage": {"input_tokens": 3, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
           "output_tokens": 14, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 17},
 "instructions": null, "max_output_tokens": null, "temperature": 1.0, "top_p": 1.0,
 "tools": [], "tool_choice": "auto", "text": {"format": {"type": "text"}}, "reasoning": null,
 "parallel_tool_calls": true, "truncation": "disabled", "metadata": {},
 "store": false, "previous_response_id": null}
```

- Output items in production order: a `reasoning` item when the model
  emitted reasoning (included in `summary_text`, without `encrypted_content`), a `message` item when there is
  visible text, one `function_call` item per call, `arguments` a JSON string.
- `status` is `completed`, or `incomplete` with `incomplete_details:
  {reason: "max_output_tokens"}` when the length cap stopped it (including
  the [`max_tokens` clamp](serve.md)).
- `usage.input_tokens` is the real prompt count, as on the chat dialect —
  the `--advertised-ctx` scaling applies to `/v1/messages` only.
  `cached_tokens` is the prefix-cache reuse. `reasoning_tokens` is **0**:
  the engine reports one output-token count and separates reasoning by text,
  so it cannot report a measured reasoning-token subtotal.

## Streaming

`stream: true` sends named SSE frames — `event: <type>` and `data: <json>`,
each with a `sequence_number` from 0 and no `[DONE]` sentinel. Events:

```
response.created
response.in_progress
  reasoning item:  response.output_item.added
                   response.reasoning_summary_part.added
                   response.reasoning_summary_text.delta *
                   response.reasoning_summary_text.done
                   response.reasoning_summary_part.done
                   response.output_item.done
  message item:    response.output_item.added
                   response.content_part.added
                   response.output_text.delta *
                   response.output_text.done
                   response.content_part.done
                   response.output_item.done
  function_call:   response.output_item.added
                   response.function_call_arguments.delta      (ONE, the whole string)
                   response.function_call_arguments.done
                   response.output_item.done
response.completed | response.incomplete        (carrying the full response object)
error, then response.failed                      (on an error after the stream opened)
```

`output_index`, `content_index` / `summary_index` and `item_id` are the
indices and IDs in the final object. The OpenAI SDK accumulator checks that
they agree and rejects mismatches; `bench/responses_sdk_smoke.mjs` exercises
that check. Function-call
arguments arrive parsed and complete from the core, so they stream as one
delta whose text equals the final `arguments` byte for byte. A `:
keep-alive` comment frame goes out every `DRINKME_SSE_KEEPALIVE_S` seconds
(default 10) until the first delta, as on the other dialects; a client that
disconnects mid-stream aborts the generation and is logged as `stream
aborted (responses)`.

## Clients

The official SDKs:

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://127.0.0.1:3215/v1", apiKey: "x" });
const r = await client.responses.create({ model: "Qwen/Qwen3-8B", input: "hello" });
console.log(r.output_text);
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:3215/v1", api_key="x")
r = client.responses.create(model="Qwen/Qwen3-8B", input="hello")
print(r.output_text)
```

Codex CLI (`npm i -g @openai/codex`), in `~/.codex/config.toml`:

```toml
model = "Qwen/Qwen3-8B"                # whatever GET /v1/models announces
model_provider = "drinkme"
model_context_window = 8192            # replace with drinkme.contextWindow from /v1/models
model_reasoning_effort = "medium"      # supported values depend on the model template
web_search = "disabled"                # Codex's default ("cached") may offer a hosted web_search
                                       # tool, which this server rejects

[model_providers.drinkme]
name = "drinkme"
base_url = "http://127.0.0.1:3215/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1800000       # allow time for long prefill on slower hardware
# env_key = "DRINKME_API_KEY"          # only with `drinkme serve --auth`
```

Then `codex` in a git repo, or one non-interactive turn:

```
codex exec --json "Run exactly this shell command and reply with its output: echo ok"
```

`bench/responses_codex_smoke.sh` runs a client check with an isolated Codex
configuration directory and records the result. `bench/responses_sdk_smoke.mjs`
checks non-streaming, streaming, and a function-call/output round trip.
Use `bench/fake_server.py` for CPU protocol checks and a separate real server
for model integration checks.
