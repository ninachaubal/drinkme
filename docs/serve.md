# Serving operations

The server loads lossless BF16 packs. It rejects FP8 checkpoints with
`FP8 checkpoints are not supported in this release (bf16 only)`.
`/v1/models` reports `sourceDtype`, device, capabilities, and context window
under one `drinkme` object (below).
See the [CLI reference](cli.md) for options, [image input](#image-input) for
the models that read images, [sleep/wake](serve-sleep.md) for memory release,
and [Metal](metal.md) for the MLX runtime's restrictions.
[Serving kernels](serve-kernels.md) covers the attention paths and kernel
choices behind the boot log's `[drinkme]` lines.

## Published packs

`drinkme serve --model M` serves the cached pack of M's commit
([cache paths](pack-format.md#cache-paths)). When there is none, it looks
for a published pack of that commit and compression profile on the pack
hub before it downloads the BF16 checkpoint to pack it locally. A published
pack is about 73% of the checkpoint's bytes for Qwen3-8B and 63% for
Qwen3-0.6B, whose checkpoint ships a copy of its tied embedding that no pack
carries, and it needs no packing time and no C++ compiler.

A downloaded pack is not taken on trust. `serve` checks its file hashes,
then runs [upstream verification](pack-format.md#upstream-verification):
it rebuilds each of the checkpoint's weight files from the pack, byte for
byte, and compares its SHA-256 with the one the Hugging Face Hub publishes
for that file at the pack's commit. The pack enters the cache only when
every file matches:

```text
[drinkme] no pack for Qwen/Qwen3-0.6B yet — downloading the published sip pack from <hub>: 0.95 GB, 63% of the 1.50 GB BF16 checkpoint
[drinkme] checking it against Qwen/Qwen3-0.6B@c1899de289a0's published sha256: rebuilding 1 weight file from the pack
[drinkme] upstream MATCH: 1 of 1 weight files (1.50 GB) rebuilt from the pack hash to Qwen/Qwen3-0.6B@c1899de289a0's published sha256; 6 of 6 embedded files match (1.6s, native decoder)
[drinkme] downloaded and verified in 60s -> ~/.cache/drinkme/packs/Qwen--Qwen3-0.6B@c1899de289a0
```

| What happens | What `serve` does |
|---|---|
| every file matches | moves the pack into the cache with its receipt (`upstream-verified.json`) and `hub-pack.json` (where it came from), and serves it |
| a file does not match (MISMATCH) | refuses the pack on stderr, naming the file, keeps nothing of it, and packs locally |
| a file is NOT COVERED, the pack fails its own file hashes, it is bound to another commit, there is no published pack, or the download fails | one line saying why, then packs locally |
| the Hub cannot be asked for the upstream hashes (offline, `HF_HUB_OFFLINE`) | says so, serves the pack on its file-hash check, and checks it against the upstream hashes at the next serve |

A cached pack is checked against the upstream hashes once: its receipt is
bound to the pack's manifest, so the check runs again only when the pack's
files change. A pack packed on this machine is not checked at serve; the
packer compared every tensor with the source when it wrote the pack.
`drinkme verify --upstream` runs the check on any pack
([CLI](cli.md#drinkme-verify)).

**Where published packs live.** One Hub repo per model,
`<namespace>/<model>-drinkme` (`<model>` is the source repo's name, such as
`Qwen3-8B`), with each pack in its own directory,
`v<version>/<profile>/<source commit>/`, where the version is the pack
format's (a new pack format ships as a new major, so `v1` is drinkme 1.x's). The lookup uses the full
commit this machine resolved, so a moved upstream branch finds a different
pack or none, never a pack cut from another commit. The pack's own
`meta.json` is read first and must name the same repo, commit and profile
before anything else downloads. The namespace is `PACK_HUB_NAMESPACE` in
[`hub_packs.py`](../src/drinkme/hub_packs.py), [`drinkme-packs`](https://huggingface.co/drinkme-packs);
`DRINKME_PACK_HUB` overrides it with another namespace or a whole repo id.
Packs are published only for models whose licence lets us redistribute
them; for any other model there is no published pack, and `serve` packs it
locally from its upstream repo.

`--no-hub-pack` (`DRINKME_NO_HUB_PACK=1`) skips the lookup and takes the
full flow. `--no-auto-pack` fetches nothing either. `drinkme pack` never
uses a published pack, and `drinkme bench` needs the BF16 checkpoint for its
stock and twin arms regardless.

A published pack redistributes the model, so the model's licence governs
it. drinkme's maintainers publish packs only for models whose licence
allows it, only after upstream verification matched every file, and with
the upstream licence and its text in the repo.

## API capability matrix

Three dialects share one engine: `/v1/chat/completions` (OpenAI Chat
Completions), `/v1/responses` ([OpenAI Responses](serve-responses.md)) and
`/v1/messages` (Anthropic Messages). The table lists supported features and
fields that are ignored. Rejected features return HTTP 400 with the field name.

| Feature | Chat Completions | Responses | Messages |
|---|---|---|---|
| Streaming | SSE, `stream: true` | SSE, `stream: true` | event-named SSE, `stream: true` |
| Tools, model-decided (`auto`) | yes — the family's [wire format](serve-tool-formats.md); a model whose format is unknown is refused (`unsupported_tool_format`) | yes | yes |
| Tools withheld (`none`) | yes | yes | `{type: none}` |
| Forced tool use (`required`, a named function; Anthropic `any` / `tool`) | refused (`unsupported_tool_choice`); constrained decoding for forced calls is not implemented | refused | refused |
| Several calls in one turn | when the model emits them (`tool_calls[]`); `parallel_tool_calls` is not read | when the model emits them; `parallel_tool_calls` accepted and echoed, not enforced | when the model emits them |
| Images ([image input](#image-input)), on a vision model; refused by name otherwise | `image_url` parts in any turn but a system one (data URL, http(s) URL, `file://`), `detail` | `input_image` parts in user turns (`image_url`, `detail`); `file_id` refused | `image` blocks (`base64` or `url` source) in user turns and inside `tool_result`; a `file` source is refused |
| Audio, video, files | refused by part type (`input_audio`, `file`, `video_url`) | refused (`input_file`, `input_audio`) | refused (`document`) |
| JSON schema output | `response_format` `json_object` / `json_schema` (grammar-constrained by `serving/constrain.py`, which rejects unsupported schema keywords); not combinable with tools; forces thinking off | `text.format`, the same path | `output_config.format` `json_schema`, the same path |
| Thinking | `reasoning_effort`, `chat_template_kwargs.enable_thinking`; leaves as `reasoning_content` | `reasoning.effort`, `chat_template_kwargs.enable_thinking`; leaves as a `reasoning` item | `thinking` enabled / adaptive / disabled, `output_config.effort` (`budget_tokens` recorded, not enforced); leaves as a `thinking` block |
| Token limit | `max_tokens` or `max_completion_tokens` (positive integer; a supplied `0` / `false` is refused) | `max_output_tokens` | `max_tokens` (required) |
| Server-side state (`previous_response_id`, `conversation`, `store`) | — | `previous_response_id` and `conversation` rejected; `store` ignored, always returned as `false` | — |

### Turning thinking off

`chat_template_kwargs: {"enable_thinking": false}` on Chat Completions and
Responses; `thinking: {type: "disabled"}` on Messages. Details, the
`<id>:nothink` profile route, why `reasoning_effort` is not an off switch,
and the model that has no off switch (Muse-Glimmer-30B, `thinking: always`):
[models](models.md#turning-thinking-off).

## Image input

Four menu models read images: Qwen3.8-27B, MiMo-V2.6-Distill-Qwen-9B,
gemma-4-31B-it and Muse-Glimmer-30B, each through its own vision tower in the
pack ([models](models.md#image-input) lists what each one makes of an image).
The first two share the Qwen3.5 architecture and its image processor, called
Qwen3.5 below.
A model's `/v1/models` entry says whether this server reads images for it
(`capabilities.vision`) and, when it does, what it accepts
(`capabilities.imageInput`: `maxPixels`, `formats`, `sources`). On a model
without image input every image part is refused by name, before a token is
generated, with the reason ([below](#when-images-are-refused)). Image input is
torch-only; the MLX runtime refuses images.

### Formats and wire shapes

PNG, JPEG, WebP and GIF are accepted. The magic bytes decide the format, so a
declared media type only has to name one of the four (`image/jpg` is taken
as `image/jpeg`). An animated GIF, WebP or PNG contributes its first frame.
EXIF orientation is applied, and an alpha channel is dropped, not
composited, as transformers' `load_image` does. SVG, HEIC/AVIF, TIFF and
BMP are refused, naming the format.

Chat Completions, an `image_url` part:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What does this screenshot show?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0…", "detail": "auto"}}]}
```

Messages, an `image` block in a user turn or inside a `tool_result`'s
`content` (the shape Claude Code sends a screenshot in):

```json
{"role": "user", "content": [
  {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0…"}},
  {"type": "text", "text": "What does this screenshot show?"}]}
```

A `url` source (`{"type": "url", "url": "https://…"}`) takes the same
routes as an `image_url`.

Responses, an `input_image` part in a user message:

```json
{"role": "user", "content": [
  {"type": "input_text", "text": "What does this screenshot show?"},
  {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0…", "detail": "low"}]}
```

Images go to the template in wire order, turn by turn. An image in a system
turn is refused. So is text, in any turn, that spells one of the model's
image or other modality markers (`<|image_pad|>`, `<|image|>`, `<|patch|>`
and the rest of the tokenizer's set): `image_injection`.

### Sources

| Source | Default | How |
|---|---|---|
| base64 | on | a `data:image/…;base64,` URL, or Messages' `base64` source |
| http(s) URL | on | fetched by the server: one 10 s deadline covers every redirect and the whole body, at most 5 redirects, and a download is aborted as soon as it passes 20 MiB. The images of one turn download concurrently, up to 8 at a time. `--no-image-urls` (`DRINKME_IMAGE_URLS=0`) turns fetching off |
| `file://` path | off | `--media-path DIR` (`DRINKME_MEDIA_PATH`) serves `file://` paths relative to `DIR`. An absolute path, a `..` or empty segment, a missing file, and a symlink that resolves outside `DIR` are refused by name |
| Files API id | never | Messages' `file` source and Responses' `file_id` name a provider's stored file, which this server does not have |

The server fetches a URL from its own network position, with no address
filtering: any address the server can reach, `localhost` and the LAN
included, is fetched when a client names it. On a server other machines can
reach (`--host 0.0.0.0`), start it with `--no-image-urls`. Every source
refusal names the sources this server does accept, and `imageInput.sources`
lists them (`data`, `url`, `file`).

### The pixel cap

Every image is resized to fit the server's pixel cap before the tower reads
it: `--image-max-pixels N` or `WxH` (`DRINKME_IMAGE_MAX_PIXELS`), default
2560x1440 (3,686,400 pixels). Each architecture applies the cap inside its
own resize rule, so an image's token count is what the model's own
processor gives for that budget, and it never exceeds the checkpoint's own
limit. A request's `detail` picks the budget: absent, `auto` and `high`
take the cap; `low` takes 512x512, and never more than the cap. Nothing a
request sends can raise it. Messages has no `detail` field.

At the default cap Qwen3.8-27B read a font-size ladder from 7 pt up on a
2880x1800 Retina capture (maintainer run on Strix Halo); raising the cap to
the capture's full size read the same lines and took 60% longer per request.

### Limits

| Limit | Value | Refusal `code` |
|---|---|---|
| Images per request | 32, counted before anything is decoded | `too_many_images` |
| Encoded size, per image | 20 MiB once decoded, downloaded or read | `image_too_large` |
| Source pixels, per image | 50,000,000, read from the header before decoding | `image_too_many_pixels` |
| Aspect ratio | an image the architecture's resize rule cannot fit (Qwen3.5: a long side over 200 times the short) | `image_aspect_ratio` |
| Request body | 128 MiB (`DRINKME_MAX_BODY_BYTES`), before any image is parsed | HTTP 413 |

Each refusal is a 400 naming the part (`messages[1].content[0]`,
`messages.1.content.0`, `input.1.content.0`). Chat Completions and Responses
carry the `code` in the error object; Anthropic's error shape has no code
field, so Messages carries only the message. The other codes:
`image_url_fetch_off`, `image_file_off`, `image_file_path`,
`image_files_api`, `image_fetch_unreachable`, `image_fetch_timeout`,
`image_fetch_status` (a non-2xx status, a redirect without `Location`, or too
many redirects), `image_url`, `image_data_url`, `image_media_type`,
`image_base64`, `image_format`, `image_decode`, `image_detail` and
`image_injection` (`serving/vision.py`'s `ImageError` lists each).

Images are decoded and preprocessed outside the generation lock, so a slow
download never holds up a running generation. A prepared image stays in host
memory until the tower has read it: at the default cap a Qwen3.5 image is
88 MB of float32 pixel values, so 32 of them hold about 2.8 GB during the
prefill.

### Token counts

An image costs the tokens its model's processor gives it
([models](models.md#image-input)), plus the markers around them.
`/tokenize` and `/v1/messages/count_tokens` count them without running
the tower, and give the number generation reports in `usage`.

### Prefix caching with images

The prefix cache keys an image on its pixel content, not on its placeholder
tokens. A conversation that sends the same image again extends its slot, and
the tower does not run for an image inside the reused prefix; two different
images of the same size never share a slot. The disk tier and sleep/wake
keep the same keys. On Qwen3.8-27B (maintainer run), the third turn of a
conversation about a 2560x1440 screenshot reused 3,702 of 3,728 prompt
tokens and reached its first token in 1.24 s, against 12.9 s cold.

A history that the chat template renders differently from what the model
wrote does not extend the slot, and every image in it runs the tower again
([prefix cache](serve-prefix-slots.md)).

How an image's tokens may be prefilled depends on how the text model attends
among them. On Qwen3.5 and Muse-Glimmer they are causal like text: chunked
prefill keeps an image that fits the chunk (4,096 tokens by default) in one
forward, cuts a longer one like text, and a restored prefix may end inside
an image. gemma-4's sliding-window layers attend in both directions within
an image, so its image is never split across forwards, and a reusable prefix
that ends inside an image is not reused: the log says `the reusable prefix
(N tokens) ends inside an image; full prefill this turn`.

N-gram speculation never proposes an image position, and text that exists
only as pixels gives it nothing to copy.

### Boot lines and the self-test

A model with image input logs its tower, cap and sources at boot:

```
[drinkme] image input: qwen3_5 tower at model.visual, pixel cap 3,686,400, sources: data, urls on, file:// off; attention bounded at 33,554,432 query-key pairs per dispatch (27 layers)
[drinkme] image input: attention self-test @ 16 heads x 72 (padded to 80), 2,048 queries x 4,096 keys, bidirectional bfloat16, against fp32: unmasked max|diff| 1.56e-02 … — AGREES
```

On an accelerator, the engine then runs the tower's attention once at its
own head width and compares it with an fp32 reference (gemma-4 runs its
masked form too). A kernel that disagrees prints a
`Vision tower attention self-test failed — image input refused.` block: the
server keeps serving text and refuses every image, naming the check. A
level-2 wake rebuilds the engine and checks again. Each request with images
logs one line:

```
[drinkme.engine] images [slot 0]: 1 in the prompt (3600 image tokens); the tower ran 1x, 0 inside the reused prefix
```

The tower's attention runs in dispatches of at most 33,554,432 query-key
pairs, as [chunked prefill](#chunked-prefill) bounds the text model's. At
the default cap its longest dispatch measured 2.7 ms on gemma-4-31B-it,
13–14 ms on Qwen3.8-27B and MiMo-V2.6-Distill-Qwen-9B, and 21 ms on
Muse-Glimmer-30B (maintainer runs on Strix Halo). `DRINKME_VISION_BOUNDED=0`
runs one attention call per image, a reference for the bit-exactness gates,
not a serving setting.

### When images are refused

`DRINKME_VISION=0` builds no vision tower: its resident share
([models](models.md#image-input)) goes to the KV cache, the pack's tower
tensors are never read, and every image is refused naming the switch. The
reason an image is refused is logged at boot as `image input: off — <reason>`
and repeated in every refusal:

| Reason | Meaning |
|---|---|
| `text-only model` | the checkpoint has no vision config |
| `this server has no image input for <model_type>` | drinkme serves no tower for this architecture (also: gemma-4's per-layer-embedding E models) |
| `disabled by DRINKME_VISION=0` | the switch above |
| `the checkpoint carries no image processor config` | nothing to preprocess with |
| `this server's torch has no LANCZOS resize …` | Muse-Glimmer's preprocessing needs torch 2.12 or later |
| `the vision tower's attention failed its boot self-test on this device (…)` | the self-test above |

## SDPA backend on ROCm

If the boot log prints `No efficient attention backend — long prompts will
OOM` on an AMD GPU, set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` before
starting the server; [ROCm](rocm.md#attention-setup) explains why, the
systemd drop-in, and the limits.

## Chunked prefill

On a desktop APU, the display compositor and drinkme share one GPU. The
amdgpu driver resets the graphics ring when a graphics job waits longer than
its `lockup_timeout` (2 seconds by default), and the reset ends the desktop
session along with every open application. A single GPU kernel that runs for
seconds can trigger it.

In a single forward pass, the longest kernel grows with the prompt. Measured on gfx1151 at Qwen3.8-27B shapes,
the causal attention kernel takes about 1.4e-6 ms × T² for T prompt tokens:
0.38 s at 16,384 tokens and, by that fit, about 24 s at 131,072. The largest
matrix multiply grows linearly and takes 0.1 s at 16,384 tokens.

drinkme prefills the uncached part of a prompt in chunks of
`DRINKME_PREFILL_CHUNK` tokens (default 4096). Each chunk extends the cache
the way a prefix-cache hit does; only the last chunk computes logits. This
applies to serial decoding, n-gram and MTP speculation (the MTP head is
seeded chunk by chunk), and prompts that extend a prefix-cache slot. An
image's tokens are chunked like text on Qwen3.5 and Muse-Glimmer; gemma-4
keeps each image in one forward ([image input](#prefix-caching-with-images)).

Attention over a long cached prefix is split further, into segments of
cached keys ([segmented attention](serve-kernels.md#segmented-attention-over-a-long-cache)).
With the defaults, the longest prefill kernel measured at 27B shapes is about
58 ms at every cached length up to 262,144 tokens.

The boot log states the setting:

```
[drinkme] prefill: chunks of 4096 tokens; attention over a long cache split above 8,388,608 query-key pairs into 33,554,432-pair dispatches (DRINKME_PREFILL_CHUNK=0 for one forward)
```

`DRINKME_PREFILL_CHUNK=0` restores the single-pass prefill for comparison,
including its attention path. Chunked prefill adds in a different order than
a single pass, so a greedy transcript can differ at a near-tie, as a
prefix-cache hit can differ from a cold prefill; weights are unchanged. On the
Qwen3-8B pack, prefill throughput at 4096 tokens is unchanged at the default
and 3.5% lower with 2048-token chunks. Smaller chunks cost more because every
pass decodes each compressed weight once.

Decoding is not chunked. A single-token step and an MTP verification pass
each attend over the whole cache in one kernel; at 262,144 cached tokens and
27B shapes those kernels measured about 104 ms and 78 ms.

## `GET /health`

```sh
curl http://127.0.0.1:3215/health
```

Always available, requires no bearer token, and answers even mid-generation
and while asleep ([sleep/wake](serve-sleep.md)). Besides `ok`, `model`, and
`device`, it carries `prefix_cache: {"slots": N}`: `N` is the resident slot
count when the prefix cache is on (the default), 0 when `--prefix-slots 0`
turned it off, and `null` for an engine with no
concept of one. [Bench and publish](bench.md) uses this field to tell a
served number taken against the shipped default from one that switched the
cache off. See [prefix cache](serve-prefix-slots.md) for sizing.

## `GET /v1/models`

```sh
curl http://127.0.0.1:3215/v1/models
```

This route does not acquire the generation lock and requires the same
authentication as generation routes (`--auth`). It returns one entry per
served model ID, with OpenAI's four fields at the top level and additional
metadata under `drinkme`. Example response from Qwen3-1.7B on CPU with
`--ctx 4096`:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-1.7B",
      "object": "model",
      "created": 1789891263,
      "owned_by": "drinkme",
      "drinkme": {
        "arm": "compressed",
        "runtime": "torch",
        "hfRepo": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "compressionProfile": "sip",
        "bitsPerWeight": 11.263,
        "meanTensorBitsPerWeight": 11.273,
        "sourceDtype": "bf16",
        "contextWindow": 4096,
        "device": "cpu",
        "capabilities": {
          "thinking": "closed",
          "toolFormat": "json",
          "thinkingSwitch": "chat_template_kwargs.enable_thinking",
          "vision": false
        },
        "sampling": {
          "profile": null,
          "defaults": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
          }
        }
      }
    }
  ]
}
```

| `drinkme.` | meaning |
|---|---|
| `arm` | `compressed` (a pack), or `stock` under `--stock`, on either runtime |
| `runtime` | `torch` or `mlx` ([Metal](metal.md)); the MLX entry also carries `computePath` |
| `hfRepo`, `revision` | the pack's coordinates (`meta.json`) |
| `compressionProfile` | the pack's `sip` / `gulp` (`meta.json`'s `profile`); `null` on the stock arm |
| `bitsPerWeight`, `meanTensorBitsPerWeight` | the pack's bits per weight as numbers, by the [lexicon](../lexicons/README.md)'s names: weight-weighted (`meta.json`'s `weightedBpw`) and the unweighted mean over tensors (`meta.json`'s `meanBpw`); `null` on the stock arm |
| `sourceDtype` | the released precision the pack serves exactly |
| `contextWindow` | allocated context window; use it to configure client compaction |
| `device` | where the weights sit; the same string as `/health`'s |
| `capabilities` | `thinking`: `open` when the default prompt already ends inside a think block, `closed` when the template reacts to `enable_thinking` but the default prompt does not open a think block (the model may still think; `closed` does not mean off), `always` when the model reasons in every reply and no request field turns it off (Muse-Glimmer-30B's `to=self` messages, returned as reasoning), `none` when the template has no thinking switch; `toolFormat` ([tool formats](serve-tool-formats.md)); `thinkingSwitch` — the request field that flips `thinking` off, or `null` when `thinking` is `always` or `none` and there is nothing to flip; `vision` — whether this server reads images for the model; and, when it does, `imageInput`: `maxPixels` (the [pixel cap](#the-pixel-cap)), `formats` and `sources` (`data`, plus `url` and `file` when [enabled](#sources)) |
| `sampling.defaults` | effective sampling defaults for a request with no overrides ([models](models.md)) |
| `sampling.profile` | `null` here; a `--profile NAME` adds an `<id>:NAME` entry whose `sampling.profile` is `NAME` and whose `defaults` carry the overlay |

The `drinkme` object is camelCase throughout, like the record
[lexicon](../lexicons/README.md). The vLLM-shaped routes (`/health`,
`/tokenizer_info`, `/sleep`, `/wake_up`) keep vLLM's snake_case, with units
as suffixes (`uptime_s`, `parked_bytes`). `/tokenizer_info` repeats the
capabilities as `thinking`, `tool_format`, `thinking_switch` and `vision`.

Aliases are the menu name when `--model` names a menu entry (`Qwen3-8B`
beside `Qwen/Qwen3-8B`) and any `--served-model-name` or
`DRINKME_SERVED_MODEL_NAMES` names. An alias lists its own entry with the same
`drinkme` object. Compression and generation profiles use separate fields; the
latter is reported as `sampling.profile`.

## `GET /metrics`

`/metrics` is always available and returns Prometheus text. Like `/health`, it
requires no bearer token and remains reachable during generation.

```sh
curl http://127.0.0.1:3215/metrics
```

| drinkme | shape | vLLM analog |
|---|---|---|
| `drinkme_requests_total{route,status}` | counter | `vllm:request_success_total` |
| `drinkme_inflight_requests` | gauge | `vllm:num_requests_running` |
| `drinkme_ttft_seconds` | histogram | `vllm:time_to_first_token_seconds` |
| `drinkme_inter_token_seconds` | histogram | `vllm:time_per_output_token_seconds` |
| `drinkme_generated_tokens_total` | counter | `vllm:generation_tokens_total` |
| `drinkme_spec_proposed_tokens_total` | counter | `vllm:spec_decode_num_draft_tokens_total` |
| `drinkme_spec_accepted_tokens_total` | counter | `vllm:spec_decode_num_accepted_tokens_total` |
| `drinkme_mtp_bail_trips_total` | counter | drinkme-specific (the adaptive bail handing a request to serial decode, [serve-speculation.md](serve-speculation.md#adaptive-bail-and-re-arm)) |
| `drinkme_mtp_rearms_total` | counter | drinkme-specific (speculation resuming after a bail) |
| `drinkme_prefix_slot_events_total{event=hit\|restore\|miss\|evict}` | counter | drinkme-specific (N whole-cache slots, not vLLM's block allocator) |
| `drinkme_prefix_slot_occupancy` | gauge | drinkme-specific |
| `drinkme_aborted_streams_total` | counter | drinkme-specific |
| `drinkme_4xx_total{status,reason}` | counter | drinkme-specific |


A prefix-slot `hit` extends a resident conversation, `restore` serves a prompt
that parts from one up to their shared prefix
([context checkpoints](serve-prefix-slots.md#context-checkpoints)), `miss`
fills an empty slot, and `evict` replaces another conversation. TTFT measures time to the
first reasoning or content delta, matching the definition used by
`bench/serve_timing_ab.py`.

[`serving/metrics.py`](../src/drinkme/serving/metrics.py) updates counters under
a small lock and formats text only when scraped. The route does not acquire
the generation lock.

## Advertised context

`--advertised-ctx N` scales Anthropic input-token counts so clients using those
counts for compaction can act before the real context window is full.
It does not change the engine's allocated window.

```sh
drinkme serve --model Qwen3-8B --ctx 32768 --advertised-ctx 8192
```

When `N` is below the real context, `/v1/messages` usage `input_tokens` and
`/v1/messages/count_tokens` are multiplied by `real_ctx / N` and rounded to
the nearest token. The example reports four times the actual input count.
Non-streaming and streaming usage use the same multiplier.

The following remain unscaled:

- OpenAI Chat Completions and Responses usage.
- Anthropic cache-read, cache-creation, and output-token counts.
- `/v1/models`' `drinkme.contextWindow`, engine accounting, metrics, and benchmark counts.

The default is unset. A value at or above the real window also leaves counts
unchanged. `DRINKME_ADVERTISED_CTX` is the environment equivalent; the flag
wins. Boot logs report any scaling and its multiplier.

## Context check and `max_tokens` clamp

Every API checks prompt length against the engine's real window before running
inference. Engines that publish no window skip this check.

| Condition | Result |
|---|---|
| Prompt fills or exceeds the window | HTTP 400, `context_length_exceeded` |
| Prompt fits, requested output exceeds remaining room | Clamp `max_tokens` and generate |
| Prompt and output budget fit | Generate with the requested budget |

A clamped generation uses the normal length stop (`finish_reason: "length"`
or `stop_reason: "max_tokens"`) and reports actual usage, subject to the
Anthropic input scaling above. The server logs the original and clamped budgets.

Set `DRINKME_MAX_TOKENS_CLAMP=0` before starting the server to get HTTP 400
for output-budget overshoot.

## Rope scaling

YaRN is opt-in. Increasing `--ctx` alone does not extend the native window.

```sh
drinkme serve --model Qwen3-8B --rope-scaling yarn:4:32768 --ctx 131072
```

Syntax: `yarn:<factor>[:<original>]`, with positive values. `original` is the
pre-extension window; if omitted, the model's `max_position_embeddings` is used.
For Qwen3-8B, the documented recipe uses 32768 even though its config reports
40960. A factor of four with that original window gives 131072 tokens.
`off` disables scaling. The flag overrides `DRINKME_ROPE_SCALING`.

Transformers applies scaling to rotary embeddings before the context clamp.
It does not alter the pack. The server logs the old and new window. Static
YaRN can affect short-input quality; enable it when the longer window is needed.
