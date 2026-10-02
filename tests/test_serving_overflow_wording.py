"""Context-length 400 wording a dialect's own client recognizes.

A client (pi 0.84.4) that hits a context-length 400 it does not
recognize never re-compacts: it classifies overflow by regex against its
OWN dialect's canonical wording, so drinkme's message has to match one of
them. pi, omp, letta-code and open-claw all classify overflow this
way; Claude Code matches Anthropic's own wording the same way.

OVERFLOW_PATTERNS and NON_OVERFLOW_PATTERNS below are copied verbatim (only
translated from JS regex literals to Python `re` ones) from
`@earendil-works/pi-ai` 0.84.4, `dist/utils/overflow.js` in the npm
package (read-only; not a path in this tree). Every dialect's overflow body must
match at least one OVERFLOW_PATTERNS entry and none of NON_OVERFLOW_PATTERNS,
while still carrying drinkme's own N/M/C numbers.
"""

import http.client
import json
import re

from test_serving_http import fake  # noqa: F401 — the shared fixture

MODEL = "drinkme-fake"

OVERFLOW_PATTERNS = [
    re.compile(r"prompt is too long", re.I),  # Anthropic token overflow
    re.compile(r"request_too_large", re.I),  # Anthropic request byte-size overflow (HTTP 413)
    re.compile(r"input is too long for requested model", re.I),  # Amazon Bedrock
    re.compile(r"exceeds the context window", re.I),  # OpenAI (Completions & Responses API)
    re.compile(r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))", re.I),  # OpenAI-compatible proxies (LiteLLM)
    re.compile(r"input token count.*exceeds the maximum", re.I),  # Google (Gemini)
    re.compile(r"maximum prompt length is \d+", re.I),  # xAI (Grok)
    re.compile(r"reduce the length of the messages", re.I),  # Groq
    re.compile(r"maximum context length is \d+ tokens", re.I),  # OpenRouter (most backends)
    re.compile(r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?", re.I),  # OpenRouter/Poolside
    re.compile(r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)", re.I),  # Together AI
    re.compile(r"exceeds the limit of \d+", re.I),  # GitHub Copilot
    re.compile(r"exceeds the available context size", re.I),  # llama.cpp server
    re.compile(r"greater than the context length", re.I),  # LM Studio
    re.compile(r"context window exceeds limit", re.I),  # MiniMax
    re.compile(r"exceeded model token limit", re.I),  # Kimi For Coding
    re.compile(r"too large for model with \d+ maximum context length", re.I),  # Mistral
    re.compile(r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?", re.I),  # DS4 server
    re.compile(r"model_context_window_exceeded", re.I),  # z.ai non-standard finish_reason surfaced as error text
    re.compile(r"prompt too long; exceeded (?:max )?context length", re.I),  # Ollama explicit overflow error
    re.compile(r"range of input length should be", re.I),  # DashScope / Qwen Token Plan
    re.compile(r"context[_ ]length[_ ]exceeded", re.I),  # Generic fallback
    re.compile(r"too many tokens", re.I),  # Generic fallback
    re.compile(r"token limit exceeded", re.I),  # Generic fallback
    re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.I),  # Cerebras: 400/413 with no body
]

NON_OVERFLOW_PATTERNS = [
    re.compile(r"^(Throttling error|Service unavailable):", re.I),  # AWS Bedrock non-overflow errors
    re.compile(r"rate limit", re.I),  # Generic rate limiting
    re.compile(r"too many requests", re.I),  # Generic HTTP 429 style
]


def _is_overflow(message: str) -> bool:
    if any(p.search(message) for p in NON_OVERFLOW_PATTERNS):
        return False
    return any(p.search(message) for p in OVERFLOW_PATTERNS)


def _post(port, path, body, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    c.request("POST", path, json.dumps(body), h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


def _openai_req(max_tokens=20):
    return {"model": MODEL, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "hello world"}]}


def _anthropic_req(max_tokens=20):
    return {"model": MODEL, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "hello world"}]}


def _responses_req(max_output_tokens=20):
    return {"model": MODEL, "input": "hello world", "max_output_tokens": max_output_tokens}


def openai_error(port, max_tokens=20):
    r, body = _post(port, "/v1/chat/completions", _openai_req(max_tokens))
    assert r.status == 400
    return json.loads(body)["error"]


def anthropic_error(port, max_tokens=20):
    r, body = _post(port, "/v1/messages", _anthropic_req(max_tokens),
                    {"anthropic-version": "2023-06-01"})
    assert r.status == 400
    return json.loads(body)["error"]


def responses_error(port, max_output_tokens=20):
    r, body = _post(port, "/v1/responses", _responses_req(max_output_tokens))
    assert r.status == 400
    return json.loads(body)["error"]


# "hello world" is 2 fake tokens (FakeTok: one word == one token, test_serving_http.py).
# ctx=2 fills the window exactly (prompt_only); ctx=10 with max_tokens=20 and
# clamping off leaves the "prompt fits, prompt+max_tokens doesn't" case.


def test_openai_dialect_prompt_only_matches_overflow_patterns(fake):
    _, port = fake(context_window=2)
    err = openai_error(port)
    assert _is_overflow(err["message"])


def test_openai_dialect_clamp_off_matches_overflow_patterns(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)
    err = openai_error(port, max_tokens=20)
    assert _is_overflow(err["message"])


def test_anthropic_dialect_prompt_only_matches_overflow_patterns(fake):
    _, port = fake(context_window=2)
    err = anthropic_error(port)
    assert _is_overflow(err["message"])


def test_anthropic_dialect_clamp_off_matches_overflow_patterns(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)
    err = anthropic_error(port, max_tokens=20)
    assert _is_overflow(err["message"])


def test_responses_dialect_prompt_only_matches_overflow_patterns(fake):
    _, port = fake(context_window=2)
    err = responses_error(port)
    assert _is_overflow(err["message"])


def test_responses_dialect_clamp_off_matches_overflow_patterns(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)
    err = responses_error(port, max_output_tokens=20)
    assert _is_overflow(err["message"])


# ------------------------------------------------------- envelope + numbers --


def test_openai_dialect_code_type_and_param(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)
    err = openai_error(port, max_tokens=20)
    assert err["code"] == "context_length_exceeded"
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "messages"


def test_responses_dialect_keeps_its_own_envelope(fake, monkeypatch):
    """responses.py's error envelope is unchanged: only message/code move,
    `param` stays whatever it always was (None) — the OpenAI dialect's new
    "param": "messages" is chat.completions' own envelope, not this one's."""
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)
    err = responses_error(port, max_output_tokens=20)
    assert err["code"] == "context_length_exceeded"
    assert err["type"] == "invalid_request_error"
    assert err["param"] is None


def test_anthropic_dialect_message_starts_with_prompt_is_too_long(fake):
    _, port = fake(context_window=2)
    err = anthropic_error(port)
    assert err["message"].startswith("prompt is too long:")


def test_anthropic_dialect_error_type(fake):
    _, port = fake(context_window=2)
    err = anthropic_error(port)
    assert err["type"] == "invalid_request_error"


def test_our_numbers_appear_in_openai_message(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)  # C=10
    err = openai_error(port, max_tokens=20)  # N=2 ("hello world"), M=20
    assert "2" in err["message"] and "20" in err["message"] and "10" in err["message"]


def test_our_numbers_appear_in_anthropic_message(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)
    err = anthropic_error(port, max_tokens=20)
    assert "2" in err["message"] and "20" in err["message"] and "10" in err["message"]


def test_our_numbers_appear_in_responses_message(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)
    err = responses_error(port, max_output_tokens=20)
    assert "2" in err["message"] and "20" in err["message"] and "10" in err["message"]
