# Models

## Tested models

The torch runtime has end-to-end tool-calling and thinking checks for
Qwen3-8B, Qwen3.8-27B, and gemma-4-31B-it, and image-input checks for the
four models marked below. Model and runtime support are separate: [MLX
currently supports dense Qwen3 text models](metal.md).

| Model | Notes |
|---|---|
| `Qwen3-8B` | Dense reference model; JSON tool calls |
| `Qwen3.8-27B` | Hybrid DeltaNet model with an MTP head; Qwen XML tool calls; reads [images](#image-input) |
| `gemma-4-31B-it` | Gated HF download; requires license acceptance and a token. Tool calls and the `thought` channel are supported; reads [images](#image-input) |
| `Muse-Glimmer-30B` | Reads [images](#image-input). Reasons in every reply (`thinking: always`): its `to=self` messages leave as reasoning, not `content` ([addressed messages](serve-tool-formats.md#addressed-messages)); send them back to keep prefix reuse. Tool calls use the `atem` row, tested on hardware. Image input needs torch 2.12 or later |
| `granite-4.2-8b` | Fits a 16 GB card only when compressed; not benchmarked. Tool calling not tested |
| `MiMo-V2.6-Distill-Qwen-9B` | Xiaomi's distill on the Qwen3.5 hybrid DeltaNet architecture; reads [images](#image-input). The checkpoint has no MTP tensors, so speculation uses the n-gram proposer. Qwen XML tool calls, identified from how its template writes a call back, since its tools block gives no call format |

```sh
drinkme serve --model Qwen3-8B
drinkme serve --model Qwen3.8-27B --ctx 16384
```

[`suggest.py`](../src/drinkme/suggest.py) defines the full menu, including
Qwen3 0.6B through 32B and Qwen2.5-72B. Every menu model can be requested
with `--model`; automatic selection considers only the ones marked
`auto_eligible`. Muse-Glimmer-30B, granite-4.2-8b, and
MiMo-V2.6-Distill-Qwen-9B are not, so they run only when named. This release
supports BF16 only; `pack`, `check`, `serve`, and `bench` reject FP8
checkpoints with an explicit error.

## Image input

Each of these models reads images through its own vision tower, which the
pack carries beside the text model and compresses with the same codec. What
an image becomes depends on the model's own processor, applied under the
server's [pixel cap](serve.md#the-pixel-cap) (2560x1440 by default):

| Model | Tokens per image at the default cap | `detail: "low"` | Vision tower, resident | Smallest UI text read |
|---|---|---|---|---|
| `Qwen3.8-27B` | one per 32×32 pixels after resizing: 64 to 3,600 (a 1920x1080 image is 2,040); a raised cap goes up to 16,384 | at most 256 | 0.65 GB | 7 pt |
| `MiMo-V2.6-Distill-Qwen-9B` | the same tower and processor as Qwen3.8-27B | at most 256 | 0.65 GB | 8 pt |
| `gemma-4-31B-it` | at most 280, by aspect ratio (264 for 16:9), whatever the image size: the checkpoint's own 645,120-pixel budget binds before the cap | at most 113 | 0.83 GB | 12 pt |
| `Muse-Glimmer-30B` | one per 28×28-pixel cell, at most 4,096 (2560x1440 is 4,080); small images are upscaled | at most 334 | 2.73 GB | 6 pt |

The resident column is the pack's `vision.residentBytes`, the memory
`DRINKME_VISION=0` gives back. The last column is the smallest font size each
model read correctly on a 2880x1800 Retina screenshot at the default cap
(maintainer runs on Strix Halo). On the same machine the tower took about
2.8 s per image at the cap on Qwen3.8-27B, 0.22 s on gemma-4-31B-it and
4.3 s on Muse-Glimmer-30B. [Image input](serve.md#image-input) covers the
wire shapes, sources, limits and refusals.

## Automatic selection

Without `--model`, `serve` scans `$DRINKME_HOME/packs`, combines canonical
cached packs with the menu, and ranks candidates by estimated resident memory
plus KV-cache allocation at the selected context and prefix-slot count. The
menu carries no sizes. A packed model is charged its pack's own
`residentBytes`, less the vision tower's share when `DRINKME_VISION=0` is set.
An unpacked model is charged an estimate, shown as one: its checkpoint's
safetensors bytes times the profile's ratio (`suggest.COMP_RATIO`). The
checkpoint's bytes come from the local Hugging Face cache, or else from the
Hub's metadata. The estimate counts the whole checkpoint, tower included. A
menu model whose checkpoint can't be sized (not cached, and the Hub doesn't
answer) is not offered, and a line names it. If the model config is not
cached, the missing KV estimate is reported. The largest candidate within
`suggest.FIT_HEADROOM` is proposed.

The server selects the runtime before ranking models, using `--runtime`,
then `DRINKME_RUNTIME`, then the host's default (`mlx` on Apple silicon,
`torch` elsewhere). `drinkme.runtimes` filters candidates by model family.
MLX supports dense Qwen3 only, so the menu reports Qwen3.8-27B and Qwen2.5-72B
as unavailable on that runtime. Torch leaves family support to its loader.

Explicit model requests use the same support rules before downloading weights
or packing. The check may fetch `config.json` to identify the model family.
If it cannot read the config, it defers the decision to the loader.

On a terminal, confirm, cancel, or enter another model name or HF repo.
`--yes` and non-interactive stdin accept the proposal. Missing packs are created
before serving. Packs in custom directories require `--pack-dir`; a custom
`pack -o` directory is not automatically selected by model name. Gemma enters
the selection only when its HF access gate is known to be open.

## Other checkpoints

```sh
drinkme check --model org/repo
drinkme check --model org/repo --json
```

`drinkme check` reads metadata without downloading weights. It checks supported
dtypes and tensor shapes using the same eligibility rules as `pack` and
`serve`. An unfamiliar architecture is reported but does not automatically
fail; the check does not validate its chat template or establish end-to-end
serving support. Failures include the model and reason. It also reports this
machine's radix encoder — the compiler `pack` would use, or a problem
(missing or broken compiler, with install lines) if packing would refuse.

## Tools

Tool dialects are detected from the chat template and reported as
`drinkme.capabilities.toolFormat` on `/v1/models`. A `tools` request requires a parsed, tested dialect; see
[tool formats](serve-tool-formats.md).

## Sampling defaults

Sampling defaults come from the model's `generation_config.json`, including
temperature, top-p, top-k, penalties, and chat-template kwargs. Precedence is:
request field > profile > `generation_config.json` > OpenAI defaults.

To change them, set the field on the request, or define a named profile with
`--profile NAME=KEY:VAL,...` or `profiles.json` beside the pack and request the
model as `<id>:NAME`. There are no per-field CLI flags such as `--temperature`.
`DRINKME_IGNORE_GENERATION_CONFIG=1` skips the checkpoint's defaults.
`/v1/models` reports the effective defaults as `drinkme.sampling.defaults`.

`serving/engine.py` validates resolved values before rendering a prompt:

- `temperature` must be finite and ≥ 0; `top_p` must be in (0, 1].
- `top_k` must be an integer ≥ 0, and `repetition_penalty` must be > 0.
- Presence and frequency penalties must be finite.
- `seed` must be an integer, and the output limit must be a positive integer.
- Booleans are not accepted as numbers.

Invalid requests return HTTP 400 in the API's error format, naming the field,
before prefill or a streaming 200 response begins. Invalid `--profile` or
`profiles.json` values fail at boot with the profile name. Invalid checkpoint
defaults fail at load with the `generation_config.json` path and instructions
for using `DRINKME_IGNORE_GENERATION_CONFIG=1`.

## Turning thinking off

`chat_template_kwargs: {"enable_thinking": false}` is the vendor-neutral
switch — the same field vLLM, SGLang, and llama.cpp read on an OpenAI-shaped
request — and it works on both `/v1/chat/completions` and `/v1/responses`.
`/v1/messages` uses Anthropic's own `thinking: {"type": "disabled"}` instead.
A model's `/v1/models` entry says whether the switch does anything for that
model (`drinkme.capabilities.thinking` is `open` or `closed`) and names the
field back (`drinkme.capabilities.thinkingSwitch`).

For a client with no per-request thinking control at all, start the server
with a named profile and point the client at `<id>:PROFILE` instead of
`<id>` — no request field involved:

```sh
drinkme serve --model Qwen3-8B --profile nothink=enable_thinking:false
```

then request model id `Qwen/Qwen3-8B:nothink`. No such profile ships by
default; `--profile` and pack-side `profiles.json` are both operator-defined
([CLI reference](cli.md), [sampling defaults](#sampling-defaults)).

`reasoning_effort` is not a thinking switch. Qwen3.8's template accepts only
`"xhigh"` / `"medium"` / `"low"` and raises on anything else, `"none"`
included (the request gets a 400); Qwen3-8B's and Qwen3-0.6B's templates
ignore the field.

Muse-Glimmer-30B has no off switch (`thinking: always`): it writes a
`to=self` reasoning message in every reply, returned as reasoning on all
three dialects, including a Messages request with `thinking` disabled. Its
template sets how much it reasons, not whether: a `Reasoning strength: low`
line in the system turn, or `chat_template_kwargs: {"reasoning_strength":
"low"}`; without either, the template writes `high`. It does not read
`reasoning_effort`.
