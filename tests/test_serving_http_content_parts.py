"""Chat content parts are validated BEFORE flattening (serving/http.py
_validate_content_parts): through real HTTP with
FakeEngine, `text: 7`, an unknown part type, a text part with no `text`,
an image-only turn and a mixed text+image turn are each a JSON 400 naming
`messages[i].content[j]` — never a dropped connection (unvalidated, the
`text: 7` TypeError escapes do_POST before its exception-to-HTTP handler),
and never a 200 whose answer silently ignored an attachment (_content_text
would drop the image and the engine would run on the emptied turn, which
docs/models.md says is refused). The same guard runs on /tokenize (a count
must never be taken over a different conversation than generate() sees),
streaming and non-streaming alike; a following request on the same
connection still succeeds; and the engine is never invoked.
"""

import http.client
import http.server
import json
import sys
import os
import threading

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_serving_http import (  # noqa: E402,F401
    VisionEngine, assert_error_shape, data_url, fake, get, msgs, post, tiny_png, vision_engine)
from drinkme.serving import vision  # noqa: E402

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

CASES = [
    ([{"type": "text", "text": 7}], "messages[0].content[0].text must be a string"),
    ([{"type": "frob", "text": "hi"}], "messages[0].content[0]: unknown content part type 'frob'"),
    ([{"type": "text"}], "messages[0].content[0].text must be a string"),
    ([{"type": "text", "text": None}], "messages[0].content[0].text must be a string"),
    (["just a string"], "messages[0].content[0] must be a content part object"),
    ([{"text": "no type"}], "messages[0].content[0] must be a content part object"),
    ([IMAGE], "messages[0].content[0]: model 'drinkme-fake' cannot read images"),
    ([{"type": "text", "text": "what is this"}, IMAGE],
     "messages[0].content[1]: model 'drinkme-fake' cannot read images"),
    ([{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}],
     "'input_audio' content parts are not supported"),
    ([{"type": "file", "file": {"file_id": "f1"}}], "'file' content parts are not supported"),
]


class _Guarded:
    """A FakeEngine whose generate/count_tokens must not be reached."""

    def __init__(self, eng):
        self.eng, self.calls = eng, []
        eng.generate = self._trap(eng.generate, "generate")
        eng.count_tokens = self._trap(eng.count_tokens, "count_tokens")

    def _trap(self, fn, name):
        def wrapped(*a, **k):
            self.calls.append(name)
            return fn(*a, **k)
        return wrapped


@pytest.mark.parametrize("content, needle", CASES)
@pytest.mark.parametrize("stream", [False, True])
def test_a_bad_content_part_is_a_json_400_on_chat_completions(fake, content, needle, stream):
    eng, port = fake()
    g = _Guarded(eng)
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = {"messages": [{"role": "user", "content": content}], "stream": stream}
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    assert r.status == 400, (content, r.status, data)
    err = assert_error_shape(data)
    assert err["type"] == "invalid_request_error"
    assert needle in err["message"], (needle, err["message"])
    assert g.calls == []  # nothing was generated over an emptied turn
    # the connection survived: a valid request on the SAME connection
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": msgs(), "stream": stream}),
              {"Content-Type": "application/json"})
    r2 = c.getresponse()
    assert r2.status == 200, (content, r2.status, r2.read())
    r2.read()
    c.close()


@pytest.mark.parametrize("content, needle", CASES)
def test_the_same_guard_runs_on_tokenize(fake, content, needle):
    eng, port = fake()
    g = _Guarded(eng)
    r, data = post(port, {"messages": [{"role": "user", "content": content}]}, path="/tokenize")
    assert r.status == 400, (content, r.status, data)
    assert needle in assert_error_shape(data)["message"]
    assert g.calls == []


def test_a_bad_part_in_a_later_turn_is_named_by_index(fake):
    _, port = fake()
    r, data = post(port, {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        {"role": "user", "content": [{"type": "text", "text": "and"}, {"type": "text", "text": 3}]}]})
    assert r.status == 400
    assert "messages[2].content[1].text must be a string" in assert_error_shape(data)["message"]


def test_text_parts_still_flatten_and_reach_the_engine(fake):
    """The valid shape is unchanged: text parts join, the engine sees the
    joined turn, on both routes."""
    _, port = fake()
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello "},
                                                      {"type": "text", "text": "world"}]}]}
    r, data = post(port, body)
    assert r.status == 200
    assert json.loads(data)["choices"][0]["message"]["content"] == "echo: hello world"
    r, data = post(port, body, path="/tokenize")
    assert r.status == 200 and json.loads(data)["count"] == 2


def test_content_text_raises_rather_than_dropping_a_part():
    """The flattener itself: handed a part the validator would refuse, it
    raises (a route without the validator gets a 400 from its handler),
    never returns a text with the part missing."""
    from drinkme.serving.http import _content_text

    assert _content_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _content_text(None) == "" and _content_text("s") == "s"
    for bad in ([{"type": "text", "text": 7}], [IMAGE], [{"type": "text", "text": "a"}, IMAGE], [5]):
        with pytest.raises(ValueError):
            _content_text(bad)


# ------------------------------------------------------------------ vision --
# A vision-capable engine (VisionEngine, a FakeEngine subclass with a real
# Vision attached) accepts image_url data URLs, in template order, and
# refuses http(s) URLs when fetching is off, images in a system turn, over MAX_IMAGES, and the
# injected placeholder literal — by name, before the engine ever runs.


class _Recording(VisionEngine):
    """A VisionEngine that records the GenerationRequest.images it was
    handed, so a test can check decode order and content without a real
    forward pass."""

    def generate(self, req):
        self.seen_images = req.images
        return super().generate(req)


def test_a_data_url_image_is_accepted_and_reaches_generation_request(fake):
    eng, port = fake(engine=_Recording())
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "compare"},
        {"type": "image_url", "image_url": {"url": data_url(tiny_png(1))}},
        {"type": "image_url", "image_url": {"url": data_url(tiny_png(2)), "detail": "low"}}]}]}
    r, data = post(port, body)
    assert r.status == 200, data
    assert len(eng.seen_images) == 2
    a, b = eng.seen_images
    assert a.digest != b.digest  # distinct pixels, distinct digest
    assert b.size <= (512, 512)  # detail: "low" -> the 512x512 budget
    obj = json.loads(data)
    assert obj["usage"]["prompt_tokens"] == 1 + a.tokens + b.tokens  # "compare" + both images


class _ImgHandler(http.server.BaseHTTPRequestHandler):
    """One PNG at any path — the "URLs are fetched by default" test's
    local server. No external host is ever contacted."""

    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        body = tiny_png(99)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def img_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ImgHandler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    t.join(timeout=5)


def test_http_urls_are_fetched_by_default(fake, img_server):
    # llama.cpp parity: fetching is ON unless --no-image-urls turned it
    # off. The download itself (timeout, oversize-abort, redirects,
    # concurrency) is test_serving_vision_urls.py's job at the vision.py
    # level; this is the dialect wiring end to end.
    eng, port = fake(engine=_Recording())
    url = f"http://127.0.0.1:{img_server.server_address[1]}/pic.png"
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": url}}]}]}
    r, data = post(port, body)
    assert r.status == 200, data
    assert len(eng.seen_images) == 1 and eng.seen_images[0].tokens > 0


def test_http_urls_are_refused_when_fetching_is_off(fake):
    eng, port = fake(engine=VisionEngine(veng=vision_engine(fetch_urls=False)))
    for url in ("http://example.com/x.png", "https://example.com/x.png"):
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}}]}]}
        r, data = post(port, body)
        assert r.status == 400, data
        err = assert_error_shape(data)
        assert err["code"] == "image_url_fetch_off"
        assert "base64 data URL" in err["message"]  # says what IS accepted, not a toggle
        assert "messages[0].content[0]" in err["message"]


def test_a_non_fetch_image_error_code_names_the_part(fake):
    # vision.ImageError's codes map to the limit refusals generally, not just
    # the fetch-off one: a malformed data URL surfaces its own code.
    _, port = fake(engine=VisionEngine())
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,not-base64!!"}}]}]}
    r, data = post(port, body)
    assert r.status == 400, data
    err = assert_error_shape(data)
    assert err["code"] == "image_base64"
    assert "messages[0].content[0]" in err["message"]


def test_images_in_a_system_turn_are_refused(fake):
    eng, port = fake(engine=VisionEngine())
    body = {"messages": [
        {"role": "system", "content": [
            {"type": "image_url", "image_url": {"url": data_url(tiny_png(1))}}]},
        {"role": "user", "content": "hi"}]}
    r, data = post(port, body)
    assert r.status == 400
    err = assert_error_shape(data)
    assert "system message" in err["message"] and "messages[0].content[0]" in err["message"]


def test_over_max_images_is_refused_before_any_decode(fake):
    eng, port = fake(engine=VisionEngine())
    parts = [{"type": "image_url", "image_url": {"url": data_url(tiny_png(i))}}
             for i in range(vision.MAX_IMAGES + 1)]
    r, data = post(port, {"messages": [{"role": "user", "content": parts}]})
    assert r.status == 400
    assert "32" in assert_error_shape(data)["message"]
    assert eng.calls == 0


def test_the_injection_guard_refuses_the_placeholder_literal(fake):
    eng, port = fake(engine=VisionEngine())
    for literal in vision.QwenVLPreprocessor(architecture="qwen3_5").reserved_text:
        r, data = post(port, {"messages": [{"role": "user", "content": f"say {literal} back"}]})
        assert r.status == 400, (literal, data)
        err = assert_error_shape(data)
        assert err["code"] == "image_injection" and literal in err["message"]
    assert eng.calls == 0


def test_a_vision_less_engine_still_refuses_by_capability(fake):
    # the pre-existing text-only refusal (CASES above) IS the no-vision-
    # engine path; this only re-confirms /v1/models says so beside it.
    _, port = fake()
    (m,) = json.loads(get(port, "/v1/models")[1])["data"]
    assert m["drinkme"]["capabilities"]["vision"] is False
    assert "imageInput" not in m["drinkme"]["capabilities"]


def test_every_dialect_words_the_no_vision_refusal_alike(fake):
    """An image on an engine without vision is one refusal in all three
    dialects: the part, the model and the engine's own reason
    (capability.vision_refusal_message). Nothing names a dialect version."""
    eng, port = fake()
    eng.vision_reason = "text-only model"
    tail = ("model 'drinkme-fake' cannot read images on this server "
            "(text-only model); send text only, or use a vision-capable model.")
    png = data_url(tiny_png(1))
    cases = [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": png}}]}]},
         "messages[0].content[1]"),
        ("/v1/messages", {"model": "drinkme-fake", "max_tokens": 4, "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "x"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": png.split(",", 1)[1]}}]}]},
         "messages.0.content.1"),
        ("/v1/responses", {"model": "drinkme-fake", "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "x"}, {"type": "input_image", "image_url": png}]}]},
         "input.0.content.1"),
    ]
    for path, body, where in cases:
        r, data = post(port, body, path=path)
        assert r.status == 400, (path, data)
        msg = json.loads(data)["error"]["message"]
        assert msg == f"{where}: {tail}", (path, msg)
        assert "(v1)" not in msg
    assert eng.calls == 0


def test_v1_models_announces_vision_and_image_input(fake):
    _, port = fake(engine=VisionEngine())
    (m,) = json.loads(get(port, "/v1/models")[1])["data"]
    caps = m["drinkme"]["capabilities"]
    assert caps["vision"] is True
    # url: on by default; file: off by default (no --media-path)
    assert caps["imageInput"] == {"maxPixels": vision.DEFAULT_MAX_PIXELS,
                                  "formats": ["png", "jpeg", "webp", "gif"],
                                  "sources": ["data", "url"]}
    _, port2 = fake(engine=VisionEngine(veng=vision_engine(fetch_urls=False, media_path="/tmp")))
    caps2 = json.loads(get(port2, "/v1/models")[1])["data"][0]["drinkme"]["capabilities"]
    assert caps2["imageInput"]["sources"] == ["data", "file"]


def test_tokenizer_info_reports_vision(fake):
    _, port = fake(engine=VisionEngine())
    r, data = get(port, "/tokenizer_info")
    assert json.loads(data)["vision"] is True
    _, port2 = fake()
    r, data = get(port2, "/tokenizer_info")
    assert json.loads(data)["vision"] is False


def test_tokenize_expands_images_from_the_header_only_count(fake):
    """count_tokens / /tokenize expand images from the header-only count:
    no ViT runs for a count, and the number equals what generate()
    actually reports as prompt_tokens."""
    eng, port = fake(engine=_Recording())
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "two words"},
        {"type": "image_url", "image_url": {"url": data_url(tiny_png(3))}}]}]}
    r, data = post(port, body, path="/tokenize")
    assert r.status == 200
    tok = json.loads(data)
    r2, data2 = post(port, body)
    assert r2.status == 200
    gen_prompt_tokens = json.loads(data2)["usage"]["prompt_tokens"]
    assert tok["count"] == gen_prompt_tokens
    # the raw `tokens` array is the UN-expanded render (one id per image
    # part): shorter than `count`, which is the actual generate() cost.
    assert len(tok["tokens"]) < tok["count"]
