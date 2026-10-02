"""serving/capability.py — the static capability probe, pure over a fake template,
then pinned against the real cached tokenizers of the two menu models plus
gemma (off-menu for THINKING; its tool dialect is a table row in serving/tool_formats.py).
No torch, no network: the real-tokenizer
tests load only tokenizer_config.json + the jinja template from the local HF
cache (HF_HUB_OFFLINE=1) and skip cleanly, by name, when a repo isn't cached.
"""

from __future__ import annotations

import os

import pytest

from drinkme.serving import capability
from drinkme.serving import tool_formats as tf

# ------------------------------------------------------------- a fake template --


class ProbeTok:
    """apply_chat_template stand-in whose render is built from exactly the
    levers the probe reads: does thinking force-open in the DEFAULT prompt,
    does the template have an enable_thinking knob that responds to False
    (the measured Qwen3-8B shape) and/or to True (the measured gemma-4
    shape — thinking defaults OFF, so only the True render changes),
    and does a tools list render a recognizable dialect's instruction block.
    No real jinja: what is under test is capability.py's CLASSIFICATION
    logic, over renders shaped like the real ones."""

    def __init__(self, *, opens_think=False, has_thinking_knob=False,
                knob_only_on_true=False, raise_on_true=False,
                tool_dialect=None,  # tool_dialect: None | "json" | "qwen-xml" | "garbage"
                history_dialect=None,  # how an assistant tool_calls turn is written back
                raise_on_history=False):
        self.opens_think = opens_think
        self.history_dialect = history_dialect
        self.raise_on_history = raise_on_history
        self.has_thinking_knob = has_thinking_knob
        self.knob_only_on_true = knob_only_on_true
        self.raise_on_true = raise_on_true
        self.tool_dialect = tool_dialect
        self.prompt = ""

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=True, tools=None, **kw):
        if self.raise_on_true and kw.get("enable_thinking") is True:
            raise RuntimeError("this template does not support enable_thinking=True")
        head = ""
        if tools:
            if self.tool_dialect == "json":
                head = ('For each function call, return a json object with '
                        'function name and arguments within <tool_call></tool_call> '
                        'XML tags:\n<tool_call>\n{"name": <fn>, "arguments": <args>}\n'
                        '</tool_call>\n')
            elif self.tool_dialect == "qwen-xml":
                head = ('<tool_call>\n<function=example_function_name>\n'
                        '<parameter=example_parameter_1>\nvalue_1\n</parameter>\n'
                        '</function>\n</tool_call>\n')
            elif self.tool_dialect == "garbage":
                head = 'respond using the zzz-format for tool calls\n'
        body = ""
        if any(m.get("tool_calls") for m in messages):
            if self.raise_on_history:
                raise RuntimeError("this template cannot render a tool_calls turn")
            body = {"json": '<tool_call>\n{"name": "drinkme_probe", "arguments": {"x": "y"}}\n'
                            '</tool_call>\n',
                    "qwen-xml": '<tool_call><function=drinkme_probe><parameter=x>y'
                                '</parameter></function></tool_call>\n',
                    None: 'called drinkme_probe\n'}[self.history_dialect] + "tool: ok\n"
        tail = "assistant\n"
        knob_exists = self.opens_think or self.has_thinking_knob
        if kw.get("enable_thinking") is False and knob_exists and not self.knob_only_on_true:
            tail = "assistant\n<think>\n\n</think>\n\n"
        elif kw.get("enable_thinking") is True and self.knob_only_on_true:
            tail = "assistant\n<|think|>\n"
        elif self.opens_think:
            tail = "assistant\n<think>\n"
        self.prompt = head + body + tail
        return list(range(len(self.prompt)))

    def decode(self, ids, skip_special_tokens=False):
        return self.prompt[-len(ids):]


# -------------------------------------------------------------------- thinking --


def test_prompt_forced_open_by_default_is_open():
    assert capability.probe(ProbeTok(opens_think=True)).thinking == "open"


def test_knob_present_but_default_not_forced_open_is_closed():
    # the measured Qwen3-8B shape: enable_thinking=False changes the render,
    # but the default prompt does not force <think> open — the model still
    # thinks by self-initiative, which this static probe cannot observe
    assert capability.probe(ProbeTok(has_thinking_knob=True)).thinking == "closed"


def test_no_thinking_knob_at_all_is_none():
    assert capability.probe(ProbeTok()).thinking == "none"


def test_knob_that_only_responds_to_explicit_true_is_closed_not_none():
    # the measured gemma-4 shape: thinking defaults OFF, so
    # enable_thinking=False alone changes nothing and would read "none"; True
    # writes a real <|think|> tag the default lacks, so the probe must
    # render True too or it misses the only working half of the toggle.
    assert capability.probe(ProbeTok(knob_only_on_true=True)).thinking == "closed"


def test_false_that_already_proves_closed_never_renders_true(monkeypatch):
    # `_thinking` must not render True unconditionally even
    # after False alone already proved the toggle exists, so a template
    # that raises on True regressed "closed" to "none". False's answer is
    # sufficient here; True must never even be attempted. probe()'s OWN
    # thinking result is what matters (asserted below); the call count is
    # checked against `_thinking` directly, not `probe`, since probe() also
    # renders tool_format through the same tokenizer.
    tok = ProbeTok(has_thinking_knob=True, raise_on_true=True)
    assert capability.probe(tok).thinking == "closed"
    called = {"n": 0}
    orig = tok.apply_chat_template

    def counting(*a, **kw):
        called["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(tok, "apply_chat_template", counting)
    assert capability._thinking(tok) == "closed"
    assert called["n"] == 2  # default + False only, never True


def test_true_raising_after_an_inconclusive_false_is_none_not_a_bubbled_error():
    # the OTHER half of defect 4's fix: when False was a no-op (inconclusive
    # on its own) and True raises, the probe must still never raise —
    # "none" is the honest answer, not whatever probe()'s catch-all would
    # have produced by accident.
    tok = ProbeTok(knob_only_on_true=True, raise_on_true=True)
    assert capability.probe(tok).thinking == "none"


# ---------------------------------------------------------------- tool_format --


def test_qwen_xml_instruction_block_classifies_qwen_xml():
    assert capability.probe(ProbeTok(tool_dialect="qwen-xml")).tool_format == "qwen-xml"


def test_json_instruction_block_classifies_json():
    assert capability.probe(ProbeTok(tool_dialect="json")).tool_format == "json"


def test_tools_change_the_render_but_match_no_signature_is_unknown():
    assert capability.probe(ProbeTok(tool_dialect="garbage")).tool_format == "unknown"


def test_tools_have_no_effect_on_the_render_is_none():
    assert capability.probe(ProbeTok(tool_dialect=None)).tool_format == "none"


# ---------------------------------------------------- the written-back call --


def test_an_unknown_declaration_takes_the_dialect_its_history_writes_back():
    """MiMo's shape: the tools block teaches no call format, the history turn
    writes qwen-xml."""
    tok = ProbeTok(tool_dialect="garbage", history_dialect="qwen-xml")
    assert capability.probe(tok).tool_format == "qwen-xml"


def test_the_declaration_wins_when_the_history_agrees_or_names_nothing():
    for hist in ("json", None):
        tok = ProbeTok(tool_dialect="json", history_dialect=hist)
        assert capability.probe(tok).tool_format == "json", hist


def test_a_declaration_and_a_history_that_disagree_are_unknown():
    tok = ProbeTok(tool_dialect="json", history_dialect="qwen-xml")
    assert capability.probe(tok).tool_format == "unknown"


def test_a_template_that_ignores_tools_stays_none_whatever_its_history():
    tok = ProbeTok(tool_dialect=None, history_dialect="qwen-xml")
    assert capability.probe(tok).tool_format == "none"


def test_a_history_that_raises_leaves_the_first_answer_standing():
    assert capability.probe(ProbeTok(tool_dialect="json", raise_on_history=True)
                            ).tool_format == "json"
    assert capability.probe(ProbeTok(tool_dialect="garbage", raise_on_history=True)
                            ).tool_format == "unknown"


def test_the_declaration_block_cannot_answer_for_the_history():
    """Only what the history ADDS is matched: a qwen-xml example in the
    declaration must not make an unrecognizable written-back call read as
    qwen-xml."""
    tok = ProbeTok(tool_dialect="qwen-xml", history_dialect=None)
    assert capability._history_format(tok) is None


# ------------------------------------------------------------------ never raises --


def test_probe_never_raises_on_a_template_that_blows_up():
    class Boom:
        def apply_chat_template(self, *a, **kw):
            raise RuntimeError("nope")

    cap = capability.probe(Boom())
    assert cap.thinking == "none" and cap.tool_format == "unknown"


# ------------------------------------------------------------- thinking_switch --


def test_as_dict_names_the_switch_key_when_thinking_is_open_or_closed():
    # A client deciding whether to bother sending
    # enable_thinking should not have to guess the field name from prose
    for thinking in ("open", "closed"):
        d = capability.Capability(thinking, "json").as_dict()
        assert d["thinkingSwitch"] == "chat_template_kwargs.enable_thinking"


def test_as_dict_has_no_switch_key_when_thinking_is_none():
    d = capability.Capability("none", "json").as_dict()
    assert d["thinkingSwitch"] is None


def test_as_dict_has_no_switch_key_when_thinking_is_always():
    # the model reasons in every reply: there is no field that turns it off
    d = capability.Capability("always", "atem").as_dict()
    assert d == {"thinking": "always", "toolFormat": "atem", "thinkingSwitch": None}


def test_only_a_row_with_a_self_addressed_reasoning_channel_reasons_always():
    """"always" comes from the row, not the render: addressed messages
    (message_open) with a reasoning channel (think_channel). gemma's row
    names a channel but has no addressed messages, and its thinking is a
    switch; no other row has either."""
    always = {r.name for r in tf.ROWS if capability._reasons_always(r.name)}
    assert always == {"atem"}
    assert not capability._reasons_always("unknown")
    assert not capability._reasons_always("none")


# --------------------------------------------------------------- refusal_message --


def test_refusal_message_names_the_model_and_says_how_to_proceed():
    msg = capability.refusal_message("Qwen/Qwen3-8B", "unknown")
    assert "Qwen/Qwen3-8B" in msg and "tools" in msg
    none_msg = capability.refusal_message("google/gemma-4-31B-it", "none")
    assert "no tool-calling support" in none_msg


# ---------------------------------------------------- the real cached tokenizers --

QWEN8B = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
QWEN27B = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
GEMMA = ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")


MIMO = ("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "2367e865d009c13ac81713a2878291d33ab28177")
# MiMo's chat_template.jinja at that revision, byte for byte (MIT; the notice is
# the .LICENSE file beside it), so the written-back case runs without the model cached.
MIMO_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates",
                             "mimo-v2.6-distill-qwen-9b.jinja")


class JinjaTok:
    """A tokenizer that is only its chat template: transformers' own jinja
    renderer (the `generation` tag included), one id per character."""

    def __init__(self, template: str):
        self.template = template
        self.text = ""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True,
                            tools=None, **kw):
        from transformers.utils.chat_template_utils import render_jinja_template

        rendered, _ = render_jinja_template(
            conversations=[messages], tools=tools, chat_template=self.template,
            add_generation_prompt=add_generation_prompt, **kw)
        self.text = rendered[0]
        return [ord(c) for c in self.text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_mimo_template_declares_tools_without_a_format_but_writes_back_qwen_xml():
    """The written-back call on MiMo's real template: the declaration alone
    matches no row (which alone would read "unknown" and refuse every tools
    request); the written-back call is qwen-xml, and the probe says so."""
    from drinkme.serving import tool_formats

    with open(MIMO_TEMPLATE) as f:
        tok = JinjaTok(f.read())
    tooled = capability._render(tok, tools=capability._PROBE_TOOLS)
    assert "You are provided with the following tools" in tooled
    assert tool_formats.identify(tooled) is None
    assert capability._history_format(tok) == "qwen-xml"
    assert capability.probe(tok).tool_format == "qwen-xml"


def test_real_mimo_tokenizer_is_qwen_xml():
    cap = capability.probe(_cached_tokenizer(*MIMO))
    assert cap.tool_format == "qwen-xml"


def test_the_history_probe_agrees_with_the_declaration_on_the_menu_models_and_gemma():
    """The second probe must not move a model that already identified:
    Qwen3-8B writes back json, the 27B qwen-xml, and gemma's written-back
    call carries no declaration signature (None, so its first answer
    stands)."""
    for repo_rev, want in ((QWEN8B, "json"), (QWEN27B, "qwen-xml"), (GEMMA, None)):
        assert capability._history_format(_cached_tokenizer(*repo_rev)) == want, repo_rev[0]


def _cached_tokenizer(repo: str, revision: str):
    """The real tokenizer from the local HF cache, offline — or a clean
    pytest.skip naming exactly what's missing. No network, ever: a cache
    miss is not this test's problem to fix."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, "tokenizer_config.json", revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache "
                    "(tokenizer_config.json not found) — nothing to probe")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)


def test_real_qwen3_8b_is_json_and_closed():
    # MEASURED (capability.py's module docstring carries the full
    # finding): the cached Qwen3-8B template does not force <think> open at
    # default/explicit-True kwargs — only enable_thinking=False changes the
    # render. "closed" is the honest static answer; the model still thinks
    # by self-initiative at generation time, which think.py's splitter
    # already handles unflagged (a literal opening tag in the stream).
    cap = capability.probe(_cached_tokenizer(*QWEN8B))
    assert cap.tool_format == "json"
    assert cap.thinking == "closed"


def test_real_qwen38_27b_is_qwen_xml_and_open():
    cap = capability.probe(_cached_tokenizer(*QWEN27B))
    assert cap.tool_format == "qwen-xml"
    assert cap.thinking == "open"


def test_real_gemma_classifies_as_its_own_row_and_thinking_closed():
    # MEASURED: gemma's template speaks its own tool dialect
    # (`<|tool>declaration:` in the tools render,
    # `<|tool_call>call:NAME{...}<tool_call|>` on the wire), which matches
    # neither the json nor the qwen-xml signature. It is a ROW of its own in
    # serving/tool_formats.py, so the probe names it and a tools request is
    # not refused as "unknown".
    # THINKING (MEASURED): the default render ends in an
    # empty `<|channel>thought\n<channel|>` pair, which enable_thinking=False
    # does not change — a probe that stopped there would report "none". But
    # enable_thinking=True rewrites the render entirely (a `<|think|>`
    # system turn ahead of the user turn), so the template DOES have a
    # working toggle; the probe checks both and reports "closed".
    cap = capability.probe(_cached_tokenizer(*GEMMA))
    assert cap.tool_format == "gemma"
    assert cap.thinking == "closed"


def test_the_new_rows_do_not_steal_the_menu_models_classification():
    """The regression gate for the table's ORDERING: adding five rows (six now sit in
    front of `json`) must not change what the two models we actually serve
    are announced as. Both are re-probed above; this asserts the stronger
    fact — no OTHER row's signature is present in their tools render at
    all, so the answer does not depend on the order being right."""
    from drinkme.serving import tool_formats

    for repo_rev, expected in ((QWEN8B, "json"), (QWEN27B, "qwen-xml")):
        tok = _cached_tokenizer(*repo_rev)
        tooled = capability._render(tok, tools=capability._PROBE_TOOLS)
        matched = [r.name for r in tool_formats.ROWS if r.matches(tooled)]
        assert matched == [expected], (repo_rev[0], matched)
