"""Templating adapter against a fake tokenizer — no download, no network.
The fake's ids are just content lengths, so every expectation is legible."""

import pytest

from drinkme.serving.template import (build_prompt, effective_kwargs, fold_system,
                                      render_prompt)


class FakeTok:
    """apply_chat_template stand-in. system_ok=False mimics gemma: the
    template RAISES on any {"role": "system"} message."""

    def __init__(self, system_ok=True):
        self.system_ok = system_ok
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, **kw):
        self.calls.append({"messages": messages, "agp": add_generation_prompt,
                           "tokenize": tokenize, **kw})
        if not self.system_ok and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return [len(m["content"]) for m in messages] + ([0] if add_generation_prompt else [])


SYS = {"role": "system", "content": "Be brief."}
USR = {"role": "user", "content": "hi"}


def test_ids_come_from_the_tokenizers_template():
    tok = FakeTok()
    assert build_prompt(tok, [USR]) == [2, 0]  # len("hi") + generation prompt
    assert tok.calls[0]["agp"] is True and tok.calls[0]["tokenize"] is True


def test_template_kwargs_pass_through():
    tok = FakeTok()
    build_prompt(tok, [USR], template_kwargs={"enable_thinking": False})  # Qwen3 knob
    assert tok.calls[0]["enable_thinking"] is False


def test_tools_pass_through_to_the_template():
    tok = FakeTok()
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    build_prompt(tok, [USR], tools=tools)
    assert tok.calls[0]["tools"] == tools  # the template renders them, not us


def test_absent_tools_is_byte_identical_to_today():
    tok = FakeTok()
    build_prompt(tok, [USR])
    build_prompt(tok, [USR], tools=None)
    build_prompt(tok, [USR], tools=[])
    assert tok.calls[0] == tok.calls[1] == tok.calls[2]
    assert "tools" not in tok.calls[0]  # the kwarg is not even passed


# ------------------------------- the two kwargs layers (effective_kwargs) --
#
# The engine's own generation_config.json defaults under the request's
# (GenerationRequest.template_kwargs); nothing ambient.


def test_request_kwargs_reach_the_template():
    tok = FakeTok()
    build_prompt(tok, [USR], template_kwargs=effective_kwargs(
        None, {"enable_thinking": False, "reasoning_effort": "low"}))
    assert tok.calls[0]["enable_thinking"] is False
    assert tok.calls[0]["reasoning_effort"] == "low"


def test_request_kwargs_win_over_the_engines_own():
    assert effective_kwargs({"enable_thinking": False, "preserve_thinking": True},
                            {"enable_thinking": True}) == {
        "enable_thinking": True,      # the request asked
        "preserve_thinking": True}    # the engine's, untouched


def test_no_request_kwargs_is_byte_identical_to_today():
    tok = FakeTok()
    build_prompt(tok, [USR])
    build_prompt(tok, [USR], template_kwargs=effective_kwargs(None, None))
    build_prompt(tok, [USR], template_kwargs=effective_kwargs({}, {}))
    assert tok.calls[0] == tok.calls[1] == tok.calls[2]


def test_effective_kwargs_copies_both_layers():
    engine, request = {"preserve_thinking": True}, {"reasoning_effort": "xhigh"}
    kw = effective_kwargs(engine, request)
    kw["reasoning_effort"] = "low"
    assert engine == {"preserve_thinking": True} and request == {"reasoning_effort": "xhigh"}


def test_a_constrained_render_turns_thinking_off_unless_told_otherwise():
    # the constraint bans "<", so a thinking model would starve
    assert effective_kwargs(None, None, constrained=True) == {"enable_thinking": False}
    assert effective_kwargs(None, {"enable_thinking": True}, constrained=True) == {
        "enable_thinking": True}   # the request said so explicitly; its call
    assert effective_kwargs({"enable_thinking": True}, None, constrained=True) == {
        "enable_thinking": True}   # so did the engine's own defaults
    assert effective_kwargs(None, None, constrained=False) == {}


def test_assistant_reasoning_content_reaches_the_template_intact():
    """Qwen3's template renders an assistant turn's reasoning_content; a
    history that loses it is a history the model never reasoned from."""
    tok = FakeTok()
    hist = [USR, {"role": "assistant", "content": "42",
                  "reasoning_content": "counted twice"}, USR]
    build_prompt(tok, hist)
    assert tok.calls[0]["messages"][1]["reasoning_content"] == "counted twice"


# ------------------------------------------- did the prompt open a <think>? --


class ThinkTok(FakeTok):
    """Qwen3's two prompt tails, measured: `<think>\\n` at the end
    of the generation prompt when thinking is on, and the whole empty pair
    pre-filled when enable_thinking is false. One id per character."""

    def __init__(self, tail="<think>\n"):
        super().__init__()
        self.tail, self.text = tail, ""

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, **kw):
        super().apply_chat_template(messages, add_generation_prompt, tokenize, **kw)
        self.text = "assistant\n" + ("<think>\n\n</think>\n\n"
                                     if kw.get("enable_thinking") is False
                                     else self.tail)
        return list(range(len(self.text)))

    def decode(self, ids, skip_special_tokens=False):
        return self.text[-len(ids):]


def test_a_prompt_ending_in_an_open_tag_is_reported():
    assert render_prompt(ThinkTok(), [USR]).opens_think is True


def test_a_prompt_that_pre_fills_the_pair_is_not():
    p = render_prompt(ThinkTok(), [USR], template_kwargs={"enable_thinking": False})
    assert p.opens_think is False


def test_an_unreadable_tail_is_not_a_thinking_prompt_and_never_raises():
    assert render_prompt(FakeTok(), [USR]).opens_think is False  # no decode() at all


def test_the_verdict_is_a_value_of_the_prompt_not_of_the_module():
    # two renders, two verdicts, neither able to see the other: nothing is
    # published anywhere
    a = render_prompt(ThinkTok(), [USR])
    b = render_prompt(ThinkTok(tail="answer:"), [USR])
    assert (a.opens_think, b.opens_think) == (True, False)
    assert build_prompt(ThinkTok(), [USR]) == a.ids  # build_prompt is the ids of the same render


def test_the_read_back_survives_a_real_tokenizer_round_trip():
    """The fake above asserts the RULE; this asserts the MECHANISM against
    real transformers, because the whole channel hangs off one decode: the
    tag is an ADDED token in Qwen3's vocab, so the prompt tail must come back
    with it intact (hence skip_special_tokens=False), and a prompt that
    pre-fills the closed pair must NOT read as open."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    def tok_ending(tail):
        backend = Tokenizer(WordLevel({"<unk>": 0, "assistant": 1, ":": 2,
                                       "hi": 3, "user": 4}, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
        tok.add_tokens(["<think>", "</think>"])
        tok.chat_template = ("{% for m in messages %}{{ m['content'] }} {% endfor %}"
                             "{% if add_generation_prompt %}assistant :" + tail
                             + "{% endif %}")
        return tok

    for tail, opened in (("<think>\n", True),            # thinking on
                         ("<think>\n\n</think>\n\n", False),  # enable_thinking false
                         ("", False)):                   # a family with no block
        assert render_prompt(tok_ending(tail), [USR]).opens_think is opened, tail


def test_the_read_back_recognises_gemmas_channel_open():
    """The tag is the ROW's (think.OPENS): gemma's template ends the
    generation prompt with `<|channel>thought\\n` after a tool response
    with thinking on, and with the CLOSED pair `<|channel>thought\\n<channel|>`
    when thinking is off — open and not-open respectively."""
    for tail, opened in (("<|channel>thought\n", True),
                         ("<|channel>thought\n<channel|>", False),
                         ("<|turn>model\n", False)):
        assert render_prompt(ThinkTok(tail=tail), [USR]).opens_think is opened, tail


def test_an_engine_with_no_prompt_says_so_in_its_stream_start():
    from drinkme.serving.engine import FakeEngine, GenerationRequest, SampleParams, StreamStart

    req = GenerationRequest([USR], SampleParams(max_tokens=2))
    for flag in (True, False):
        first = next(FakeEngine(reply="x y", opens_think=flag).generate(req))
        assert first == StreamStart(prompt_tokens=1, opens_think=flag, cached_tokens=0)


def test_system_supported_means_no_fold():
    tok = FakeTok(system_ok=True)
    build_prompt(tok, [SYS, USR])
    assert len(tok.calls) == 1  # one call, messages untouched
    assert tok.calls[0]["messages"] == [SYS, USR]


def test_system_folded_into_first_user_on_template_error():
    tok = FakeTok(system_ok=False)
    ids = build_prompt(tok, [SYS, USR])
    assert len(tok.calls) == 2  # raised, folded, retried
    folded = tok.calls[1]["messages"]
    assert [m["role"] for m in folded] == ["user"]
    assert folded[0]["content"] == "Be brief.\n\nhi"
    assert ids == [len("Be brief.\n\nhi"), 0]


def test_system_with_no_user_turn_becomes_the_user_turn():
    tok = FakeTok(system_ok=False)
    build_prompt(tok, [SYS])
    assert tok.calls[1]["messages"] == [{"role": "user", "content": "Be brief."}]


def test_unrelated_template_failure_propagates():
    class Broken(FakeTok):
        def apply_chat_template(self, *a, **k):
            raise RuntimeError("jinja exploded")

    with pytest.raises(RuntimeError, match="jinja exploded"):
        build_prompt(Broken(), [USR])  # no system to fold -> the failure is real


def test_fold_system_returns_same_object_when_nothing_to_fold():
    msgs = [USR]
    assert fold_system(msgs) is msgs  # identity IS the "unchanged" signal


def test_fold_system_does_not_mutate_the_input():
    msgs = [dict(SYS), dict(USR)]
    fold_system(msgs)
    assert msgs[1]["content"] == "hi"
