"""Aliases and generation profiles: <model>:<profile> exposed model IDs — aliases
(--served-model-name) first, named sampling/chat-template overlays (--profile)
on top of them (serving/generation_profiles.py).

Layering mirrors test_serving_gen_config.py: pure-function tests for
generation_profiles.py, HTTP-layer tests through FakeEngine for alias/profile
resolution and precedence, and one real toy HFEngine (a Qwen3.8-style
enable_thinking-aware chat template) for the thinking-hazard rendering test.
"""

import http.client
import json

import pytest
import torch

from drinkme.serving import gen_config, generation_profiles
from drinkme.serving.engine import FakeEngine, SampleParams
from drinkme.serving.http import start_server
from drinkme.serving.template import build_prompt

MSGS = [{"role": "user", "content": "hi"}]


def post(port, body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


def get(port, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r, body


def _capture(engine):
    seen = {}
    orig = engine.generate

    def wrapped(req):
        seen["params"] = req.sampling
        return orig(req)

    engine.generate = wrapped
    return engine, seen


# ------------------------------------------------------------------ pure functions --


def test_served_names_cli_wins_over_env(monkeypatch):
    monkeypatch.setenv("DRINKME_SERVED_MODEL_NAMES", "env-a env-b")
    assert generation_profiles.served_names_from_env(["cli-a", "cli-b"]) == ["cli-a", "cli-b"]


def test_served_names_env_fallback_comma_or_space(monkeypatch):
    monkeypatch.setenv("DRINKME_SERVED_MODEL_NAMES", "a, b,c")
    assert generation_profiles.served_names_from_env(None) == ["a", "b", "c"]


def test_served_names_neither_set_is_empty(monkeypatch):
    monkeypatch.delenv("DRINKME_SERVED_MODEL_NAMES", raising=False)
    assert generation_profiles.served_names_from_env(None) == []


def test_parse_generation_profile_flag_coerces_types():
    name, overlay = generation_profiles.parse_generation_profile_flag(
        "fast=enable_thinking:false,temperature:0.7,top_k:20,reasoning_effort:low")
    assert name == "fast"
    assert overlay == {"enable_thinking": False, "temperature": 0.7,
                       "top_k": 20, "reasoning_effort": "low"}


def test_parse_generation_profile_flag_bad_shapes_raise():
    with pytest.raises(ValueError):
        generation_profiles.parse_generation_profile_flag("no-equals-sign")
    with pytest.raises(ValueError):
        generation_profiles.parse_generation_profile_flag("name=key-without-colon")


def test_generation_profiles_from_sources_json_then_cli_overlay_by_name(tmp_path):
    (tmp_path / "profiles.json").write_text(json.dumps({
        "thinking": {"enable_thinking": True},
        "fast": {"enable_thinking": False, "temperature": 0.9},
    }))
    out = generation_profiles.generation_profiles_from_sources(["fast=temperature:0.7"], str(tmp_path))
    assert out["thinking"] == {"enable_thinking": True}  # untouched by CLI
    assert out["fast"] == {"temperature": 0.7}  # CLI REPLACES the whole named entry


def test_generation_profiles_from_sources_neither_source_is_empty():
    assert generation_profiles.generation_profiles_from_sources(None, None) == {}


def test_resolve_model_known_id_no_profile():
    assert generation_profiles.resolve_model("m", {"m", "alias"}, {}) == ("m", None)
    assert generation_profiles.resolve_model("alias", {"m", "alias"}, {}) == ("alias", None)


def test_resolve_model_id_colon_profile():
    known = {"m"}
    profs = {"thinking": {}, "fast": {}}
    assert generation_profiles.resolve_model("m:thinking", known, profs) == ("m", "thinking")


def test_resolve_model_unknown_id_or_profile_is_none():
    known, profs = {"m"}, {"thinking": {}}
    assert generation_profiles.resolve_model("nope", known, profs) is None
    assert generation_profiles.resolve_model("m:nope", known, profs) is None
    assert generation_profiles.resolve_model("nope:thinking", known, profs) is None


def test_unknown_model_message_names_known_ids_and_profiles():
    msg = generation_profiles.unknown_model_message("bogus", {"m", "m2"}, {"thinking": {}})
    assert "bogus" in msg and "m" in msg and "m2" in msg and "thinking" in msg


def test_effective_defaults_precedence_profile_over_generation_config():
    base = gen_config.effective_defaults({"temperature": 0.6, "top_p": 0.95})
    out = generation_profiles.effective_defaults(base, {"temperature": 0.1})
    assert out["temperature"] == 0.1   # profile wins over generation_config
    assert out["top_p"] == 0.95        # untouched: generation_config's own
    assert out["top_k"] == SampleParams.top_k  # untouched: OpenAI default


def test_template_kwargs_excludes_sampling_fields():
    tkw = generation_profiles.template_kwargs({"enable_thinking": False, "temperature": 0.7,
                                    "reasoning_effort": "low"})
    assert tkw == {"enable_thinking": False, "reasoning_effort": "low"}


# ------------------------------------------------------------- HTTP layer (Fake) --


@pytest.fixture
def fake():
    servers = []

    def make(engine=None, served_names=None, generation_profiles=None, **kw):
        eng = engine or FakeEngine(**kw)
        srv = start_server(eng, "127.0.0.1", 0, served_names=served_names,
                           generation_profiles=generation_profiles)
        servers.append(srv)
        return eng, srv.server_address[1]

    yield make
    for srv in servers:
        srv.shutdown()


def test_v1_models_lists_every_alias_and_profile_pair(fake):
    eng, port = fake(served_names=["alias-a", "alias-b"],
                     generation_profiles={"thinking": {"enable_thinking": True},
                              "fast": {"enable_thinking": False}})
    _, body = get(port, "/v1/models")
    ids = {e["id"] for e in json.loads(body)["data"]}
    base = {eng.model_id, "alias-a", "alias-b"}
    for b in base:
        assert b in ids
        assert f"{b}:thinking" in ids
        assert f"{b}:fast" in ids
    assert len(ids) == len(base) * 3  # 3 ids each: bare + 2 profiles


def test_v1_models_alias_and_profile_entries_carry_the_same_drinkme_object(fake):
    """The compression profile and the sampling profile never share a
    key: the pack's is `drinkme.compressionProfile` and the overlay names
    itself under `drinkme.sampling.profile`, so a `<model>:<profile>` entry
    never clobbers the pack's sip/gulp. Every alias and every `<id>:<profile>` pair
    carries the one `drinkme` object the engine published — identical but
    for `sampling`."""
    eng = FakeEngine(defaults={"temperature": 0.6})
    eng.model_meta = lambda m=eng.model_meta: dict(m(), compressionProfile="gulp")
    eng, port = fake(engine=eng, served_names=["alias-a"],
                     generation_profiles={"fast": {"temperature": 0.1, "enable_thinking": False}})
    data = {e["id"]: e for e in json.loads(get(port, "/v1/models")[1])["data"]}
    assert set(data) == {eng.model_id, "alias-a",
                         f"{eng.model_id}:fast", "alias-a:fast"}
    for e in data.values():
        assert set(e) == {"id", "object", "created", "owned_by", "drinkme"}
        assert e["drinkme"]["compressionProfile"] == "gulp"  # intact on every entry
    base = data[eng.model_id]["drinkme"]
    assert base["sampling"] == {"profile": None,
                                "defaults": gen_config.effective_defaults({"temperature": 0.6})}
    assert data["alias-a"]["drinkme"] == base  # an alias is the same model
    for pid in (f"{eng.model_id}:fast", "alias-a:fast"):
        d = data[pid]["drinkme"]
        assert d["sampling"]["profile"] == "fast"
        assert d["sampling"]["defaults"]["temperature"] == 0.1  # overlaid
        assert d["sampling"]["defaults"]["top_k"] == SampleParams.top_k
        assert "enable_thinking" not in d["sampling"]["defaults"]  # a template kwarg, not sampling
        assert {k: v for k, v in d.items() if k != "sampling"} == \
               {k: v for k, v in base.items() if k != "sampling"}


def test_request_accepts_an_alias(fake):
    eng, port = fake(served_names=["alias-a"])
    r, _ = post(port, {"model": "alias-a", "messages": MSGS})
    assert r.status == 200


def test_request_still_accepts_the_original_id_with_aliases_present(fake):
    eng, port = fake(served_names=["alias-a"])
    r, _ = post(port, {"model": eng.model_id, "messages": MSGS})
    assert r.status == 200


def test_unknown_model_is_still_404_naming_known_ids(fake):
    eng, port = fake(served_names=["alias-a"])
    r, body = post(port, {"model": "nope", "messages": MSGS})
    assert r.status == 404
    msg = json.loads(body)["error"]["message"]
    assert "nope" in msg and eng.model_id in msg and "alias-a" in msg


def test_unknown_profile_suffix_is_404_naming_known_profiles(fake):
    eng, port = fake(generation_profiles={"thinking": {}})
    r, body = post(port, {"model": f"{eng.model_id}:bogus", "messages": MSGS})
    assert r.status == 404
    msg = json.loads(body)["error"]["message"]
    assert "bogus" in msg and "thinking" in msg


def test_profile_sampling_overlay_reaches_sample_params(fake):
    eng = FakeEngine(reply="ok", defaults={"temperature": 0.6, "top_p": 0.95})
    eng, seen = _capture(eng)
    eng, port = fake(engine=eng, generation_profiles={"fast": {"temperature": 0.2, "top_k": 5}})
    r, _ = post(port, {"model": f"{eng.model_id}:fast", "messages": MSGS})
    assert r.status == 200
    p = seen["params"]
    assert p.temperature == 0.2   # profile wins over generation_config
    assert p.top_k == 5           # profile-only field applies too
    assert p.top_p == 0.95        # untouched: generation_config's own default


def test_request_field_wins_over_profile(fake):
    eng = FakeEngine(reply="ok")
    eng, seen = _capture(eng)
    eng, port = fake(engine=eng, generation_profiles={"fast": {"temperature": 0.2}})
    r, _ = post(port, {"model": f"{eng.model_id}:fast", "messages": MSGS,
                       "temperature": 0.99})
    assert r.status == 200
    assert seen["params"].temperature == 0.99  # the request's own, over the profile


def test_bare_alias_with_no_profile_uses_generation_config_defaults_unmodified(fake):
    eng = FakeEngine(reply="ok", defaults={"temperature": 0.6})
    eng, seen = _capture(eng)
    eng, port = fake(engine=eng, served_names=["a"],
                     generation_profiles={"fast": {"temperature": 0.2}})
    r, _ = post(port, {"model": "a", "messages": MSGS})
    assert r.status == 200
    assert seen["params"].temperature == 0.6  # no ":profile" suffix -> no overlay


def test_profile_template_kwargs_reach_the_engine(fake):
    """Not sampling: a request that carries no chat_template_kwargs of its
    own still ends up with the profile's enable_thinking in the request the
    engine is handed — GenerationRequest.template_kwargs, the only place an
    engine reads template kwargs from."""
    seen = {}
    eng = FakeEngine(reply="ok")
    orig = eng.generate

    def wrapped(req):
        seen["kw"] = req.template_kwargs
        return orig(req)

    eng.generate = wrapped
    eng, port = fake(engine=eng, generation_profiles={"fast": {"enable_thinking": False}})
    r, _ = post(port, {"model": f"{eng.model_id}:fast", "messages": MSGS})
    assert r.status == 200
    assert seen["kw"] == {"enable_thinking": False}


def test_anthropic_dialect_accepts_alias_and_profile(fake):
    eng, port = fake(served_names=["alias-a"], generation_profiles={"fast": {"temperature": 0.2}})
    r, _ = post(port, {"model": f"alias-a:fast", "max_tokens": 8,
                       "messages": [{"role": "user", "content": "hi"}]},
               path="/v1/messages")
    assert r.status == 200


def test_anthropic_dialect_unknown_alias_is_404(fake):
    eng, port = fake()
    r, body = post(port, {"model": "bogus", "max_tokens": 8,
                          "messages": [{"role": "user", "content": "hi"}]},
                  path="/v1/messages")
    assert r.status == 404
    assert "bogus" in json.loads(body)["error"]["message"]


# --------------------------------------------------- the thinking hazard (real) --


@pytest.fixture(scope="module")
def thinking_toy(tmp_path_factory):
    """A toy Llama whose chat template mirrors the two real asymmetries
    documented in bench/prefix_slots_verify.py: enable_thinking on ends the
    generation prompt INSIDE an open <think> block; enable_thinking off
    pre-fills the whole empty pair."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("thinking_toy")
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2, "hi": 3, "user": 4, "assistant": 5,
             ":": 6, "<think>": 7, "</think>": 8}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.add_tokens(["<think>", "</think>"])
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant :"
        "{% if enable_thinking == false %}<think> </think> {% else %}<think> {% endif %}"
        "{% endif %}")
    tok.save_pretrained(str(d))
    cfg = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=64,
                      tie_word_embeddings=True, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(str(d), safe_serialization=True)
    return str(d)


def test_thinking_vs_fast_profile_renders_different_prompts(thinking_toy):
    """build_prompt directly, the same tokenizer, only the profile's
    enable_thinking differs — asserts the ids differ EXACTLY at the point
    the template's own {% if %} says they should (the <think>/</think> tail),
    not merely "differ somewhere."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(thinking_toy)
    ids_thinking = build_prompt(tok, MSGS, template_kwargs={"enable_thinking": True})
    ids_fast = build_prompt(tok, MSGS, template_kwargs={"enable_thinking": False})
    assert ids_thinking != ids_fast
    common = 0
    for a, b in zip(ids_thinking, ids_fast):
        if a != b:
            break
        common += 1
    # both share "user : hi assistant : <think>" (identical prefix — the
    # template opens thinking either way) and diverge EXACTLY where the
    # template's own {% if %} says they should: thinking ends there (open,
    # for the model to close), fast immediately pre-fills the close tag.
    assert tok.decode(ids_thinking[:common]).strip().endswith("<think>")
    assert ids_thinking[common:] == []            # thinking: nothing after the open tag
    assert tok.decode(ids_fast[common:]).strip() == "</think>"  # fast: pre-closed right after


def test_thinking_vs_fast_profile_over_http_count_tokens(thinking_toy):
    """The same divergence, end to end: two profiles differing only in
    enable_thinking, read back through /v1/messages/count_tokens (lockless,
    no generation) against a REAL engine with aliases/profiles wired."""
    from drinkme.serving.engines import load_stock

    eng = load_stock(thinking_toy, None, device="cpu")
    srv = start_server(eng, "127.0.0.1", 0, served_names=["m"],
                       generation_profiles={"thinking": {"enable_thinking": True},
                                "fast": {"enable_thinking": False}})
    try:
        port = srv.server_address[1]
        _, body_t = post(port, {"model": "m:thinking", "max_tokens": 8,
                                "messages": [{"role": "user", "content": "hi"}]},
                        path="/v1/messages/count_tokens")
        _, body_f = post(port, {"model": "m:fast", "max_tokens": 8,
                                "messages": [{"role": "user", "content": "hi"}]},
                        path="/v1/messages/count_tokens")
        n_thinking = json.loads(body_t)["input_tokens"]
        n_fast = json.loads(body_f)["input_tokens"]
        assert n_thinking != n_fast  # the profile reached the real template
    finally:
        srv.shutdown()
