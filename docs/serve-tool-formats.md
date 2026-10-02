# Tool-call formats

A model's chat template describes inputs and often teaches an output syntax,
but the server still needs a parser for generated tool calls.
[`serving/tool_formats.py`](../src/drinkme/serving/tool_formats.py) defines one
row per dialect, including detection signatures, markers, and parsers.

At load, `serving/capability.py` renders the template with and without tools,
then matches the tooled prompt against the table. `/v1/models` reports
the row as `drinkme.capabilities.toolFormat`. A template ignoring tools reports `none`; one with tool support
but no matching row reports `unknown`. Requests containing tools require a
recognized, tested row or return an error before generation.

## Detection order

| row | signature (in the tools-rendered prompt) | start | end | streaming |
|---|---|---|---|---|
| `gemma` | `<\|tool>declaration:` | `<\|tool_call>` | `<tool_call\|>` | start marker flips suppression; everything to the close marker is buffered; calls emit when the block completes |
| `atem` | `<atem:function_calls>` + `<atem:invoke` | `<atem:function_calls>` | `</atem:function_calls>` | start marker flips suppression; everything to the close marker is buffered; calls emit when the block completes; one block may carry several invokes |
| `kimi-k2` | `<\|im_system\|>tool_declare` | `<\|tool_calls_section_begin\|>` | `<\|tool_calls_section_end\|>` | start marker flips suppression; everything to the close marker is buffered; calls emit when the block completes; one block may carry several calls |
| `minimax` | `<minimax:tool_call>` | `<minimax:tool_call>` | `</minimax:tool_call>` | start marker flips suppression; everything to the close marker is buffered; calls emit when the block completes; one block may carry several <invoke>s |
| `mistral` | `[AVAILABLE_TOOLS]` | `[TOOL_CALLS]` | _(none — runs to end of turn)_ | start marker suppresses text through end of turn (no close marker); calls emit at flush |
| `glm` | `<arg_key>` + `<arg_value>` | `<tool_call>` | `</tool_call>` | start marker flips suppression; everything to the close marker is buffered; calls emit when the block completes |
| `qwen-xml` | `<function=` + `<parameter=` | `<tool_call>` | `</tool_call>` | start marker flips suppression; everything to the close marker is buffered; calls emit when the block completes |
| `json` | `<tool_call>` + `arguments` | `<tool_call>` | `</tool_call>` | start marker flips suppression; everything to the close marker is buffered; calls emit when the block completes |


The first match wins. GLM and Qwen XML must precede generic JSON because their
instruction text can also match the JSON signature. JSON and Qwen XML share
markers and a parser chain (JSON first, XML parameters second), so either body
form parses; the detected row records what the template teaches.

## Wire formats

| row | family | wire form |
|---|---|---|
| `json` | Llama, Qwen3, DeepSeek, Hermes lineage | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
| `qwen-xml` | Qwen3.5+ / Qwen3.8 / Coder | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` |
| `gemma` | Gemma 4 | `<\|tool_call>call:NAME{key:<\|"\|>value<\|"\|>,num:42}<tool_call\|>` |
| `atem` | Muse Glimmer | `<\|start\|>assistant to=NAME<\|message\|><atem:function_calls><atem:invoke name="NAME"><atem:parameter name="k">v</atem:parameter></atem:invoke></atem:function_calls>` |
| `glm` | GLM 4.7, 5 | `<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` |
| `minimax` | MiniMax M2 | `<minimax:tool_call><invoke name="NAME"><parameter name="k">v</parameter></invoke></minimax:tool_call>` |
| `mistral` | Mistral, Devstral | `[TOOL_CALLS][{"name": …, "arguments": {…}}]` |
| `kimi-k2` | Kimi K2 | `<\|tool_calls_section_begin\|><\|tool_call_begin\|>functions.NAME:0<\|tool_call_argument_begin\|>{…}<\|tool_call_end\|><\|tool_calls_section_end\|>` |


## Streaming and malformed output

Assistant text streams until a start marker appears. The scanner buffers the
call block, parses it at the closing marker, and emits structured calls after
the turn completes. Mistral has no closing marker; it buffers until turn end.

Unrecognized completed blocks return as literal visible text. Identifier checks
prevent stray prose from becoming a bare-name call. Partial markers and unclosed
blocks flush as text at generation end. Tests split each fixture at every byte
offset and compare visible text and calls with unsplit input
(`tests/test_serving_tool_formats.py`).

## Control tokens and reasoning

Some markers are special tokenizer IDs. Stripping all special tokens would
remove them before parsing. Each row declares IDs to preserve as text and
markers for its reasoning channel:

<!-- generated from tool_formats.control_markdown_table(); edit the source, not this table -->
| row | control tokens (special ids kept as text) | reasoning opens | reasoning closes |
|---|---|---|---|
| `gemma` | `<\|tool_call>`, `<tool_call\|>`, `<\|channel>`, `<channel\|>`, `<\|"\|>` | `<\|channel>thought` | `<channel\|>` |
| `atem` | `<\|start\|>`, `<\|message\|>`, `<\|eom\|>` | `to=self<\|message\|>` | `<\|eom\|>` |
| `kimi-k2` | _(none declared)_ | `<think>` | `</think>` |
| `minimax` | _(none declared)_ | `<think>` | `</think>` |
| `mistral` | _(none declared)_ | `<think>` | `</think>` |
| `glm` | _(none declared)_ | `<think>` | `</think>` |
| `qwen-xml` | _(none declared)_ | `<think>` | `</think>` |
| `json` | _(none declared)_ | `<think>` | `</think>` |


An empty declaration does not establish that a family's markers are ordinary
tokens. Gemma's and Muse Glimmer's special tokens and the cached Qwen tokenizers
have been checked; other families need tokenizer inspection before enabling
their tested flag.

At load, `serving/control.py` resolves each declaration against the tokenizer.
Only markers represented by a single special ID are retained, and the server
logs the resulting set. Other special tokens remain stripped. The tool scanner
consumes call markers; `serving/think.py` consumes reasoning markers. Markers
emitted outside an active parser, such as a tool call when no tools were offered,
remain visible text.

For a special call-close ID, the engine uses one-token lookahead: another call
opening continues the turn; otherwise generation ends and the lookahead token
is not emitted. Reasoning closes do not stop the answer. Plain-text closes,
such as Qwen's, do not add a stop ID. Prompt inspection also recognizes when
the template has already opened a reasoning channel.

`bench/gemma_control_tokens_gate.py` covers Gemma's end-to-end behavior.

### Addressed messages

Muse Glimmer (`atem`) does not open and close one reasoning block. Its turn is
a run of messages, each addressed to a recipient, and its generation prompt
already ends with the first one's `<|start|>assistant`:

```text
 to=self<|message|>REASONING<|eom|><|start|>assistant to=user<|message|>ANSWER<|eot|>
```

The row declares the header's literals (`message_open`, `recipient_open`,
`message_body`), and `serving/think.py` walks messages instead of a tag pair.
Messages to `self` are reasoning. Every other message is content: the answer,
and a message to a tool whose call block the scanner did not consume. The
headers and `<|eom|>` are dropped; `<|eot|>` is an end-of-sequence token and
never reaches the scanners. Nothing else is trimmed, because the chat template
renders `reasoning_content` and `content` back verbatim. A client that returns
both unchanged gets a next prompt that extends the prefix cache
([prefix slots](serve-prefix-slots.md#muse-glimmers-reasoning-messages)). The
checkpoint also ships a transformers `response_template` declaring the same
fields, and `tests/test_serving_glimmer_channel.py` holds the split to it.

## Tool definitions

Every dialect's `tools` reach the chat template in OpenAI's nested shape
(`{"type": "function", "function": {"name", "description", "parameters"}}`);
Messages' `input_schema` becomes `parameters`. A tool sent without a
`description`, or with `"description": null`, gets `""`, and one without
`parameters` (or with null) gets `{}`, before any template renders it:
llama.cpp's defaults (`common/chat.cpp`), in all three dialects. Some official
templates read both fields unguarded, Muse-Glimmer's among them, and would
otherwise fail the render. A tool that has both reaches the template as sent.

## Argument typing

JSON, Gemma, Mistral, and Kimi carry types in their syntax. Gemma delimits strings
with `<|"|>` and leaves numbers, booleans, and null bare.

Qwen XML, GLM, MiniMax, and ATEM use request schemas to type textual values. Declared
integer, number, boolean, object, array, and null parameters are JSON-parsed;
failed parses retain the original text. String parameters, undeclared parameters, and calls without schemas retain the
original text.

## Tested support

`json` (Qwen3-8B), `qwen-xml` (Qwen3.8-27B), `gemma` (gemma-4-31B-it), and
`atem` (Muse-Glimmer-30B) have GPU end-to-end coverage. Other rows have parser
fixtures but are refused for tool requests by default. `DRINKME_TOOLS_UNTESTED=1`
enables them for development runs and is announced at boot. The flag gates tool
requests only: a row's reasoning split and control tokens apply to every
request, tested or not.

## Add a dialect

1. Read the model's actual chat template and tokenizer. Record source URLs.
2. Cross-check an existing implementation, such as the family's vLLM parser
   or LiteLLM transform, for cases the template does not show.
3. Add a row to `ROWS`: name, signature, markers, parser chain, control tokens,
   and any non-default reasoning markers. Put specific signatures before generic
   ones. Read `all_special_ids` to classify markers.
4. Add a sourced fixture to `FIXTURES` in `tests/test_serving_tool_formats.py`.
   The parameterized tests cover chunk boundaries, malformed blocks, and flush.
5. Add a cached tokenizer/template to `CACHED` where available, and test that
   rendered calls parse back correctly.
6. Run an end-to-end hardware smoke with the development override before setting
   `tested=True`; record the gate in the row's comment.

The optional `hint` column is unused because supported templates include
their own output instructions. The shared scanner
is in `serving/tools.py`; adding a row should not require another scanner.
