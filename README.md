# drinkme

drinkme is a local LLM server that keeps model weights losslessly compressed
in memory and decodes them during inference. Packed BF16 weights retain every
bit of the original checkpoint. The server supports OpenAI Chat Completions,
Responses, and Anthropic Messages APIs, with [image input](docs/serve.md#image-input)
on models that have a vision tower, and [video input](docs/serve.md#video-input) on
the Qwen3.5 ones. Use it to run a BF16 model that would
not otherwise fit in memory, without quantizing it.

Measured on Qwen3-8B and Qwen3.8-27B, BF16 weights take about 27% less memory
at `sip` (the default compression profile) and about 30% at `gulp`. The KV
cache and runtime buffers are not compressed. Speed depends on the model and hardware; see the
[results site](https://drinkme.petrichor.wtf) and [method](docs/method.md).

**drinkme is experimental.** We test it heavily, and every packed weight is
checked bit for bit against the original checkpoint, but it is early software. It
hasn't had the years of use and hardening behind mature servers such as
llama.cpp and vLLM, so expect rough edges, and please report what breaks.

**drinkme is developed with AI coding agents.** People direct the work, test it
on real hardware, and review every change before it merges.

What changed in each version is in the [changelog](CHANGELOG.md).

## Run the server

drinkme runs on Linux with an NVIDIA or ROCm-supported AMD GPU, with AMD
testing on Strix Halo and RX 7600 XT. The first run detects your hardware,
installs the matching PyTorch packages into the project environment, and tests
them before serving. `uv run --no-sync drinkme bootstrap --dry-run` shows the
installation plan. See [hardware setup](docs/hardware.md) and, for AMD targets,
[ROCm](docs/rocm.md#install-lanes) for device coverage,
or set `DRINKME_NO_AUTO_DEPS=1` to manage accelerator dependencies yourself.

With no supported GPU, the first run installs the standard PyPI PyTorch wheel
without asking, and drinkme can only serve on the CPU, which is slow. On Linux
that wheel is a CUDA build, so `serve` refuses to start until you set
`DRINKME_DEVICE=cpu` to confirm. Other operating systems, including Windows, are
untested.

Apple silicon uses MLX/Metal (install lane `metal`). On an M4 with 24 GB,
drinkme has served and benched Qwen3-1.7B and Qwen3-4B end to end, bit for bit
against the source: 1.21× and 1.25× MLX-LM's BF16 decode speed. Other model
families, the 8B and larger on that machine, and speculation on MLX are still
in progress. See [Apple silicon status](docs/metal.md).

Start with Git, Python 3.10+, uv, and a C++ compiler (`c++`, `g++`, or
`clang++` on PATH — packing needs one; serving an existing pack does not):

- Debian/Ubuntu: `sudo apt install g++`
- Fedora: `sudo dnf install gcc-c++`
- Arch: `sudo pacman -S gcc`
- macOS: `xcode-select --install`

For a **new checkout and environment**:

```sh
git clone https://tangled.org/ninachaubal.com/drinkme
cd drinkme
uv sync
uv run --no-sync drinkme serve
```

The GitHub mirror, `https://github.com/ninachaubal/drinkme`, clones the same tree.

For an existing installation, use `uv run --no-sync drinkme …`. Do not repeat
`uv sync`: it removes the accelerator packages from a working environment.

With no model specified, `serve` suggests the largest model estimated to fit
and asks for confirmation. `--yes` or non-interactive stdin accepts the choice
automatically. It fetches the model's pack as needed, caches packs under
`~/.cache/drinkme/packs`, and listens at `http://127.0.0.1:3215`. When a
published pack of the model exists, `serve` downloads it (about 73% of the
checkpoint's bytes for Qwen3-8B) and checks it against the SHA-256 the
Hugging Face Hub publishes for each of the checkpoint's files
([published packs](docs/serve.md#published-packs)). Otherwise it downloads
the original checkpoint and packs it, which needs disk space for both.

A pack is self-contained: besides the compressed weights it carries the
tensors the codec leaves uncompressed (embeddings, norms, biases), the model
configuration and the tokenizer, and it serves with no checkpoint and no
network. Only `drinkme pack` and the comparison arms of `drinkme bench` need
the original checkpoint, so once a model is packed, you can delete it from
the Hugging Face cache. To use a pack on another machine, copy its
directory. See [the embedded checkpoint](docs/pack-format.md#the-embedded-checkpoint)
and [source identity](docs/pack-format.md#source-identity).

A pack contains the model's original tensors and tokenizer, so sharing one
is redistributing the model, and the model's license governs it.

To choose a model explicitly:

```sh
uv run --no-sync drinkme serve --model Qwen3-8B
uv run --no-sync drinkme serve --model Qwen3.8-27B --ctx 16384
```

`--model` also accepts a Hugging Face repo ID. Check eligibility before
downloading weights with `uv run --no-sync drinkme check --model org/repo`.
See [model support](docs/models.md) for tested models and what `drinkme check` checks.
Gated models require accepting the model's license on Hugging Face and a token
(set `HF_TOKEN` or run `hf auth login`).

On AMD GPUs, long prompts may need `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
set before serving; see [ROCm](docs/rocm.md).

### Connect a client

| Client API | Base URL | Endpoints |
|---|---|---|
| OpenAI-compatible | `http://127.0.0.1:3215/v1` | `/v1/chat/completions`, `/v1/responses`, `/v1/models` |
| Anthropic-compatible | `http://127.0.0.1:3215` | `/v1/messages`, `/v1/messages/count_tokens` |

Model IDs, API keys, a first request, and recipes for Codex, pi, and turning
Qwen3 thinking off: [Clients](docs/clients.md).

### Common options

Commands below assume the environment is activated (`source .venv/bin/activate`);
you can also prefix them with `uv run --no-sync`.

| Option | Purpose |
|---|---|
| `--model MODEL` | Model name or HF repo; omitted to choose by available memory |
| `--yes` | Accept the automatic model choice without prompting |
| `--host HOST`, `--port PORT` | Listen address; defaults to `127.0.0.1:3215`. Use `--host 0.0.0.0 --auth TOKEN --no-image-urls` for LAN access (the server fetches image URLs a client names) |
| `--ctx N` | Context allocation; default capped at 8192 tokens. Larger windows use more memory |
| `--auth TOKEN` | Require a bearer token on `/v1/*`; `/health` remains open |
| `--pack-dir DIR` | Load an existing pack from this directory |
| `--no-auto-pack` | Fail if a pack is missing instead of creating one |
| `--no-hub-pack` | Pack locally from the checkpoint instead of downloading a published pack |
| `--prefix-slots N` | Conversations to keep cached in accelerator memory; default 1 |
| `--spec MODE` | Speculative decoding: `auto` (default), `off`, `mtp`, `ngram`, or `ngram+mtp` |
| `--sleep-on-idle SEC` | Release accelerator memory after an idle period; disabled by default |

Run `drinkme serve --help` or see the [CLI reference](docs/cli.md) for all
options and environment variables, including generation profiles and context
extension. Runtime restrictions are listed in the [Metal docs](docs/metal.md).

## Pack and benchmark

`serve` fetches or packs automatically. To prepare a pack separately or measure performance:

```sh
drinkme pack --model Qwen3-8B
drinkme bench --dry-run            # inspect the plan without downloading weights
drinkme bench --model Qwen3-8B
```

Bench compares uncompressed and compressed inference and writes a record to
`measurements/`. It times its own decode loop, not HTTP requests.

Share a record through **atproto**, under your own identity and on your
own PDS:

```sh
drinkme publish --handle you.example.com
```

The first publish opens your PDS's login and consent page. It validates the
measurement, publishes it as a `wtf.petrichor.drinkme.measurement` record, and
reads it back to verify the write. Publishing is optional and separate from
benchmarking. Developers can use the [lexicon](lexicons/) to build their own
views of the data. The results site shows the records published on atproto in
this lexicon. See the [benchmark and publishing guide](docs/bench.md).

## How compression works

drinkme compresses BF16 weights without changing their values. It stores each
weight's sign and mantissa unchanged and gives common exponents shorter codes.
The GPU decodes weights during inference. Embeddings, norms, and small layers
stay uncompressed.

The default compression profile, `sip`, uses about 11.4 bits per packed weight on Qwen3-8B.
`gulp` uses about 10.9 bits but takes more work to decode. Each packed tensor
is checked bit for bit against its source when written, and its file hash is
checked when loaded. Identical weights can still produce different tokens at
near-ties because kernels accumulate sums in different orders.

Compression can improve speed on bandwidth-limited hardware, and it lets
larger models fit: on NVIDIA, Qwen3.8-27B ran on a 48 GB L40S, where BF16 did
not load. Current numbers are on the [results site](https://drinkme.petrichor.wtf).

See [method](docs/method.md) for measurement conditions and the
codec's relationship to DFloat11, ZipServ, and Brian Bell's Split12 design.
The [pack format](docs/pack-format.md) documents the encoding and verification.

## Developer documentation

Start with [AGENTS.md](AGENTS.md) for development setup, testing, and checks,
[docs/architecture.md](docs/architecture.md) for the vocabulary, flows, and
module map, and [docs/README.md](docs/README.md) for the full documentation index.

| Area | Documentation and code |
|---|---|
| Serving | [Operations](docs/serve.md), [image input](docs/serve.md#image-input), [tool calling](docs/serve-tool-formats.md), [runtime](src/drinkme/serving/) |
| Caching and decoding | [Prefix cache](docs/serve-prefix-slots.md), [speculation](docs/serve-speculation.md), [sleep/wake](docs/serve-sleep.md) |
| Compression | [Pack format](docs/pack-format.md), [codec](src/drinkme/codec/), [Metal kernels](src/drinkme/metal/) |
| Performance work | [Method](docs/method.md), [developer instruments](bench/README.md) |

## Contributing

Bug reports, measurements from your hardware, and pull requests are welcome.
[CONTRIBUTING.md](CONTRIBUTING.md) says what to include and how to run the tests.

## License and attribution

Copyright © 2026 Nina Chaubal. Licensed under [Apache-2.0](LICENSE).
Model weights remain subject to their respective licenses.
