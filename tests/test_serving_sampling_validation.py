"""Sampling values the samplers cannot take are a 400 at the boundary,
before any prefill and before any streaming byte.

The three parsers coerced numbers (float()/int()) and checked nothing
else; SampleParams validates nothing. Reproduced: `temperature: "nan"`
passed the parser and died in the sampler (a probability-tensor
RuntimeError); `top_p: -1` filtered every candidate and the draw raised;
`top_k: 1e309` was an uncaught OverflowError. On a real engine the first
two fail AFTER an expensive prefill and, streaming, after the 200 has
committed. Now ONE validator (engine.check_sampling /
validated_sample_params) runs after precedence resolution (request >
named profile > generation_config.json > SampleParams' default) in every
dialect, each of which renders the failure in ITS 400 envelope; a named
profile is refused at boot with its name, and a checkpoint's
generation_config.json at load with its path.

Over real sockets against FakeEngine (tests/test_serving_http.py's
fixture): `eng.calls` is the proof no generation ran.
"""

import json

import pytest

from drinkme.serving import gen_config, generation_profiles
from drinkme.serving.engine import SampleParams
from test_serving_http import fake, msgs, post, sse_events  # noqa: F401 — shared fixture

MODEL = "drinkme-fake"

# three reproduced cases, then overflow and booleans — each
# with the substring its 400 must carry (the field, then the domain)
BAD = [
    ({"temperature": "nan"}, "temperature: must be a number"),
    ({"temperature": float("nan")}, "temperature: must be a finite number"),  # json's NaN literal
    ({"top_p": -1}, "top_p: must be in (0, 1]"),
    ({"top_p": 0}, "top_p: must be in (0, 1]"),
    ({"top_k": 1e309}, "top_k: must be an integer"),  # json's Infinity: int() overflowed
    ({"top_k": -1}, "top_k: must be >= 0"),
    ({"top_k": 10.5}, "top_k: must be an integer"),
    ({"temperature": -0.5}, "temperature: must be >= 0"),
    ({"temperature": True}, "temperature: must be a number, not a boolean"),
    ({"top_k": False}, "top_k: must be an integer, not a boolean"),
]
# the OpenAI-only fields (Anthropic's wire has none of these)
BAD_OPENAI = BAD + [
    ({"repetition_penalty": 0}, "repetition_penalty: must be > 0"),
    ({"presence_penalty": float("inf")}, "presence_penalty: must be a finite number"),
    ({"frequency_penalty": "2"}, "frequency_penalty: must be a number"),
    ({"seed": 1.5}, "seed: must be an integer"),
    ({"seed": True}, "seed: must be an integer, not a boolean"),
]


def _chat(extra, stream=False):
    return {"model": MODEL, "messages": msgs(), "stream": stream, **extra}


def _responses(extra, stream=False):
    return {"model": MODEL, "input": "hello world", "stream": stream, **extra}


def _messages(extra, stream=False):
    return {"model": MODEL, "max_tokens": 64, "stream": stream,
            "messages": [{"role": "user", "content": "hello world"}], **extra}


def _post_messages(port, body):
    return post(port, body, path="/v1/messages")


# ------------------------------------------------------------ the validator --


def test_check_sampling_names_the_field_and_its_domain():
    from drinkme.serving.engine import SampleParamError, check_sampling

    ok = check_sampling({"temperature": 0, "top_p": 1, "top_k": 50.0, "repetition_penalty": 1.1,
                         "presence_penalty": -2, "frequency_penalty": 0})
    assert ok["top_k"] == 50 and isinstance(ok["top_k"], int) and ok["temperature"] == 0.0
    for extra, needle in BAD_OPENAI:
        if "seed" in extra:
            continue
        with pytest.raises(SampleParamError) as ei:
            check_sampling(extra)
        assert needle in str(ei.value), (extra, str(ei.value))
        assert ei.value.field == next(iter(extra))
    with pytest.raises(SampleParamError, match="is not a sampling field"):
        check_sampling({"stream": True})


def test_validated_sample_params_covers_seed_and_the_output_limit_under_its_wire_name():
    from drinkme.serving.engine import SampleParamError, validated_sample_params

    full = gen_config.effective_defaults({})
    p = validated_sample_params(full, seed=7, max_tokens=12, stop=["x"])
    assert p == SampleParams(seed=7, max_tokens=12, stop=["x"])
    assert validated_sample_params(full).max_tokens == SampleParams.max_tokens
    for mt, needle in [(0, "must be a positive integer"), (-3, "must be a positive integer"),
                       (1e309, "must be an integer"), (True, "must be an integer, not a boolean"),
                       ("64", "must be an integer")]:
        with pytest.raises(SampleParamError, match=f"max_output_tokens: {needle}"):
            validated_sample_params(full, max_tokens=mt, max_tokens_field="max_output_tokens")
    with pytest.raises(SampleParamError, match="seed: must be an integer"):
        validated_sample_params(full, seed="7")


# ------------------------------------------------- the three wire dialects --


def test_chat_completions_refuses_before_generate_in_the_openai_envelope(fake):
    eng, port = fake()
    for extra, needle in BAD_OPENAI + [({"max_tokens": 1e309}, "max_tokens: must be an integer"),
                                       ({"max_tokens": -1}, "max_tokens: must be a positive integer"),
                                       ({"max_tokens": True}, "max_tokens: must be an integer, not a boolean")]:
        r, body = post(port, _chat(extra))
        assert r.status == 400, (extra, r.status, body)
        err = json.loads(body)["error"]
        assert err["type"] == "invalid_request_error" and needle in err["message"], (extra, err)
    assert eng.calls == 0


def test_responses_refuses_before_generate_in_its_envelope(fake):
    eng, port = fake()
    for extra, needle in BAD_OPENAI + [({"max_output_tokens": 1e309}, "max_output_tokens: must be an integer")]:
        r, body = post(port, _responses(extra), path="/v1/responses")
        assert r.status == 400, (extra, r.status, body)
        err = json.loads(body)["error"]
        assert err["type"] == "invalid_request_error" and needle in err["message"], (extra, err)
    assert eng.calls == 0


def test_messages_refuses_before_generate_in_anthropics_envelope(fake):
    eng, port = fake()
    for extra, needle in BAD + [({"max_tokens": 1e309}, "max_tokens: must be a positive integer")]:
        r, body = _post_messages(port, _messages(extra))
        assert r.status == 400, (extra, r.status, body)
        obj = json.loads(body)
        assert obj["type"] == "error" and obj["error"]["type"] == "invalid_request_error"
        assert needle in obj["error"]["message"], (extra, obj)
    assert eng.calls == 0


def test_a_streaming_request_is_refused_with_a_400_not_a_committed_200(fake):
    """The streaming case is the one that bit: a sampler failure after the
    200 and the SSE headers had gone out. With stream: true the refusal is
    still a plain 400 JSON body, no event-stream, no generate."""
    eng, port = fake()
    for path, body in [("/v1/chat/completions", _chat({"top_p": -1}, stream=True)),
                       ("/v1/responses", _responses({"temperature": "nan"}, stream=True)),
                       ("/v1/messages", _messages({"top_k": 1e309}, stream=True))]:
        r, data = post(port, body, path=path) if path != "/v1/messages" else _post_messages(port, body)
        assert r.status == 400, (path, r.status, data)
        assert "text/event-stream" not in (r.getheader("Content-Type") or "")
        assert sse_events(data) == [] and json.loads(data)["error"]
    assert eng.calls == 0


def test_valid_values_still_reach_generate_on_every_dialect(fake):
    eng, port = fake()
    good = {"temperature": 0.7, "top_p": 0.9, "top_k": 40}
    r, _ = post(port, _chat({**good, "seed": 1, "max_tokens": 8}))
    assert r.status == 200
    r, _ = post(port, _responses({**good, "max_output_tokens": 8}), path="/v1/responses")
    assert r.status == 200
    r, _ = _post_messages(port, _messages(good))
    assert r.status == 200
    assert eng.calls == 3


# ----------------------------------------------- the defaults, at boot/load --


def test_a_bad_profile_default_is_refused_through_the_same_validator(fake):
    """A profile's value goes through precedence like a request's and the
    same validator judges it: a profile that would fail is refused at boot
    by name, from a profiles.json or a --profile flag; a good one serves."""
    with pytest.raises(SystemExit, match=r"refusing to serve: profile 'hot': temperature: must be >= 0"):
        generation_profiles.generation_profiles_from_sources(["hot=temperature:-1"], None)
    with pytest.raises(SystemExit, match=r"profile 'nuke': top_p: must be in \(0, 1\]"):
        generation_profiles.generation_profiles_from_sources(["nuke=top_p:0"], None)
    with pytest.raises(SystemExit, match=r"profile 'flag': temperature: must be a number, not a boolean"):
        generation_profiles.generation_profiles_from_sources(["flag=temperature:true"], None)
    assert generation_profiles.generation_profiles_from_sources(["fast=temperature:0.2,top_k:5,enable_thinking:false"], None) == {
        "fast": {"temperature": 0.2, "top_k": 5, "enable_thinking": False}}


def test_a_bad_profiles_json_beside_the_pack_is_refused_at_boot_by_name(tmp_path):
    (tmp_path / "profiles.json").write_text(json.dumps({
        "ok": {"temperature": 0.5}, "broken": {"top_k": -3, "enable_thinking": True}}))
    with pytest.raises(SystemExit, match=r"refusing to serve: profile 'broken': top_k: must be >= 0"):
        generation_profiles.generation_profiles_from_sources(None, str(tmp_path))
    (tmp_path / "profiles.json").write_text(json.dumps({"ok": {"temperature": 0.5}, "shape": 3}))
    with pytest.raises(SystemExit, match=r"profile 'shape' must be an object"):
        generation_profiles.generation_profiles_from_sources(None, str(tmp_path))


def test_a_checkpoints_generation_config_is_refused_at_load_with_the_way_out(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": -1, "eos_token_id": 2}))
    with pytest.raises(ValueError, match=r"generation_config.json: top_k: must be >= 0 — .*"
                                          r"DRINKME_IGNORE_GENERATION_CONFIG=1"):
        gen_config.load(model=None, snapshot_dir=str(tmp_path))
    (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": 20, "eos_token_id": 2}))
    assert gen_config.load(model=None, snapshot_dir=str(tmp_path)).sampling == {"top_k": 20}
