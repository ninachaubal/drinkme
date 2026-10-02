# Clients

This page is the one place for connecting clients to drinkme: base URLs and
keys, a first request, and worked recipes for real client harnesses. drinkme
speaks three request dialects (`/v1/chat/completions`, `/v1/responses`,
`/v1/messages`); [the capability matrix](serve.md#api-capability-matrix) is
the source of truth for what each dialect accepts.

## Connecting a client

Configure a client with the appropriate base URL and a served model ID.
Supported API features vary; see the endpoint documentation.

| your client speaks | point it at | key |
|---|---|---|
| OpenAI chat completions (`/v1/chat/completions`, `/v1/models`) | `http://127.0.0.1:3215/v1` | any non-empty string |
| OpenAI Responses (`/v1/responses`) — the Agents SDK, Codex CLI; [serve-responses.md](serve-responses.md) | `http://127.0.0.1:3215/v1` | any non-empty string |
| Anthropic Messages (`/v1/messages`, `/v1/messages/count_tokens`) | `http://127.0.0.1:3215` | any non-empty string |

Use an ID from `GET /v1/models`. A server started with a menu name also answers
to that name; add more aliases with `--served-model-name ID`.
When `--auth TOKEN` is set, clients must use that token instead of the placeholder
key. `/health` remains open. Legacy `/v1/completions`, embeddings, and Ollama's
native `/api/*` are unsupported.

For clients that read OpenAI environment variables:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:3215/v1
export OPENAI_API_KEY=x
```

## First request

Every client needs a model ID. List the served IDs:

```sh
curl http://127.0.0.1:3215/v1/models
```

A model served with `--model Qwen3-8B` is listed as `Qwen/Qwen3-8B` and as its
menu name, `Qwen3-8B`; either ID works. A repo ID or local path adds no alias.
`curl http://127.0.0.1:3215/health` shows server status and the active device.
To check the server end to end:

```sh
curl http://127.0.0.1:3215/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Codex

Codex CLI speaks the Responses API. Its `~/.codex/config.toml` recipe — a
`drinkme` provider with `wire_api = "responses"`, the context window from
`/v1/models`, and hosted web search disabled — is in
[Responses: clients](serve-responses.md#clients), with the official SDK
snippets.

## pi (`github.com/earendil-works/pi`)

pi (`@earendil-works/pi-coding-agent`, verified against `0.84.4`) picks a
request shape (`thinkingFormat`) from the base URL, and an unrecognized base
URL such as drinkme's falls back to `"openai"`, which never sends
`enable_thinking`: `pi --thinking off` then has no effect. Name the format
in `~/.pi/agent/models.json` (or `$PI_CODING_AGENT_DIR/models.json`):

```json
{
  "providers": {
    "drinkme": {
      "baseUrl": "http://127.0.0.1:3215/v1",
      "api": "openai-completions",
      "apiKey": "drinkme",
      "compat": {
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        { "id": "Qwen/Qwen3-8B", "reasoning": true }
      ]
    }
  }
}
```

Use `"qwen-chat-template"`, not `"qwen"`. `"qwen-chat-template"` sends
`chat_template_kwargs: {enable_thinking, preserve_thinking: true}`, the field
drinkme reads; `"qwen"` sends a top-level `enable_thinking`, which drinkme
ignores, so thinking stays on. With it set, `pi --thinking off` sends
`enable_thinking: false`; any other level, or no flag, sends `true`.

The `<id>:nothink` profile route ([turning thinking off](models.md#turning-thinking-off)) needs neither: start drinkme with
the matching `--profile` and set the model id in `models.json` to
`Qwen/Qwen3-8B:nothink`.

To try this without touching a real `~/.pi`, point `PI_CODING_AGENT_DIR` at a
throwaway directory with its own `models.json`:

```sh
PI_CODING_AGENT_DIR=/tmp/pi-test/agent pi --thinking off ...
```

## Turning thinking off

The request fields and the `<id>:nothink` profile route are in
[models](models.md#turning-thinking-off).
