# ROCm

This page covers running drinkme on AMD GPUs under ROCm: which runtime
bootstrap installs for each gfx target, installing a TheRock line by hand,
the attention setup long prompts need, running it under systemd, the limits
of that setup, and what the vision towers run differently on gfx1151. The
probe and self-test bootstrap runs on every platform are in
[hardware support](hardware.md#the-probe-and-the-self-test).

Strix Halo uses AMD's TheRock per-target build (the `rocm` dependency group).
The official PyTorch wheel's bundled ROCm 7.1 runtime imports, detects the
device, and allocates memory, but segfaults on its first kernel launch on
gfx1151. [`bootstrap.py`](../src/drinkme/bootstrap.py) therefore selects TheRock
for this target. Other AMD targets, including gfx1102 (RX 7600 XT), use the
official wheel.

## Install lanes

The AMD gfx target determines which runtime to install. Drinkme reads it from
KFD topology in sysfs, falling back to the PCI-ID table in
[`detect.py`](../src/drinkme/detect.py). The `LANES` table in
[`bootstrap.py`](../src/drinkme/bootstrap.py) defines two installation paths:

- **official** — PyTorch's own ROCm wheel, `torch 2.13.0+rocm7.1` from
  `https://download.pytorch.org/whl/rocm7.1`, with `triton-rocm` 3.7.1
  from the same index (the `rocm-official` dependency group; torch
  2.13.0+rocm7.1 requires that package by that name). The wheel
  bundles the ROCm runtime (about 25 shared libraries under `torch/lib`), so
  it runs without a system ROCm installation. It carries kernels for
  fifteen targets: gfx900, 906, 908, 90a, 942, 950, 1030, 1100, 1101, 1102,
  1103, 1150, 1151, 1200, 1201 (`torch.cuda.get_arch_list()`).
- **therock** — AMD's per-target nightly indexes at
  `https://rocm.nightlies.amd.com/v2/`. gfx1151 has a pinned group (`rocm`);
  three more index families carry both a Linux torch and a Linux triton
  wheel and get an exact install line instead of an automatic install.

Bootstrap selects TheRock for gfx1151 and the official wheel for other
targets in that wheel's architecture list. For remaining targets with both
wheels in a TheRock index, it prints a manual install command. It rejects
other targets with an explanation. Selection does not require a listing in
AMD's support matrix: gfx1102 and gfx1151 passed tests while unlisted.
Drinkme never sets `HSA_OVERRIDE_GFX_VERSION`.

| Target | Part | Lane | Status |
|---|---|---|---|
| gfx1151 | Strix Halo (Ryzen AI Max 395, Radeon 8060S) | therock, the `rocm` group | **verified**: packs served, bit-exact gates. The official wheel imports, detects the device, and allocates, but segfaults at the first kernel launch |
| gfx1102 | RX 7600 XT / 7600 / 7700S | official | **verified**: bit-exact gates and the 8B served on an RX 7600 XT (16 GB), with its own launch-table row. Uncompressed BF16 8B does not fit; the lossless pack serves. Neither 27B compression profile fits. Long prompts need the [attention setup](#attention-setup) |
| gfx1100, gfx1101 | RX 7900 / 7800 / 7700 series, Radeon Pro W7x00 | official | kernels present in the wheel; not tested by this project. The local self-test checks whether the runtime works. TheRock's `gfx110X-dgpu` index has no triton wheel and an outdated torch wheel, so there is no fallback line |
| gfx1103 | Radeon 780M / 760M (Phoenix, Hawk Point) | official | kernels present in the wheel; not run by us; no fallback line |
| gfx1150 | Strix Point (Radeon 880M / 890M) | official | kernels present in the wheel; not run by us. Its sibling APU gfx1151 fails this lane at the first launch and TheRock publishes no torch for gfx1150, so there is no fallback if the self-test fails |
| gfx1200, gfx1201 | RX 9060 / 9070 series (RDNA4) | official; TheRock `gfx120X-all` as the fallback line | kernels present in the wheel; not run by us. If the self-test says the wheel cannot launch, the bootstrap prints the `gfx120X-all` install line |
| gfx942 | Instinct MI300 series | official; TheRock `gfx94X-dcgpu` as the fallback line | kernels present in the wheel; not run by us |
| gfx950 | Instinct MI350 series | official; TheRock `gfx950-dcgpu` as the fallback line | kernels present in the wheel; not run by us |
| gfx1030 | RX 6000 series (RDNA2) | official | kernels present in the wheel; not run by us; no fallback line |
| gfx908, gfx90a | Instinct MI100, MI200 series | official | kernels present in the wheel; not run by us; no fallback line (TheRock publishes no torch for them) |
| gfx900, gfx906 | Instinct MI25, MI50 / Radeon VII | official | kernels present in the wheel; not run by us; AMD has dropped both from its own matrix; no fallback line |
| gfx1152, gfx1153 | Krackan Point, Gorgon Point | — | **refused**: not in the official wheel's list; TheRock's gfx1152 and gfx1153 indexes carry ROCm SDK packages but no torch (read 2026-09-23) |
| gfx101x | RX 5000 series (RDNA1) | — | **refused**: the official wheel does not carry its kernels and TheRock publishes no torch for it |

Targets marked "kernels present in the wheel; not run by us" have no project
hardware validation. The local self-test checks whether the installed runtime
works on that machine; it does not establish full model coverage.

### ROCm index pin

As of 2026-09, pytorch.org's quick-start points its "stable" ROCm install at
`rocm7.14`. Starting there, the wheel no longer bundles the ROCm runtime; it
expects TheRock-built `_rocm_sdk_*` sibling packages and locates them by
relative rpath. Through `rocm7.2` the wheel bundles the runtime, which is
what was validated on an RX 7600 XT with no `/opt/rocm` present. The
`rocm-official` group pins the index to `rocm7.1` so a fresh install lands
on the runtime packaging tested here. The unbundled build has not been tested.

## Manual TheRock installation

If bootstrap prints a manual install command, run it from the project
directory, then set `DRINKME_NO_AUTO_DEPS=1` so drinkme leaves that runtime
alone, and use `uv run --no-sync` for project commands. Do not
synchronize the gfx1151 group onto another AMD device: it pins
`rocm-sdk-libraries-gfx1151`.

Check an installed runtime without changing it:

```sh
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No accelerator")'
```

PyTorch uses the `torch.cuda` API for ROCm too. A ROCm build should report a
HIP version and an available device. After starting your server, compare the
boot log's device line with `/health` and `/v1/models`. The boot log's pack
line also names the launch-table rows in use and says when they were not
measured on this part (the rows were measured on gfx1151, gfx1102, and five
NVIDIA cards; other devices use fallback rows).

[`detect.py`](../src/drinkme/detect.py) names AMD cards from PCI IDs and
revisions, then system tools, with a PCI-ID fallback. Benchmark records with
only a fallback name require `--allow-unknown-device`; see
[device naming](bench.md#naming-a-discrete-amd-card).

## Attention setup

If the boot log prints `No efficient attention backend — long prompts will
OOM`, torch found no efficient SDPA kernel for this GPU and fell back to
MATH. Its fp32 attention score matrix has shape `[batch, heads, T, T]`, so
memory use grows quadratically with prompt length. Long prompts can exhaust
memory even when short prompts work. A 25,057-token Qwen3.8-27B prompt with
24 query heads can request 56.13 GiB for that allocation. On the experimental
architectures listed below, enable AOTriton before starting the process:

```sh
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 drinkme serve
```

This applies to ROCm architectures AOTriton classifies as experimental,
including gfx1151 (Strix Halo, with the tested TheRock ROCm builds) and
gfx1102 (tested on an RX 7600 XT). The flag avoids MATH's quadratic memory
allocation for long prompts. At ordinary context lengths, measured decode
speed differed by less than 5% with the flag on or off.

Drinkme leaves this opt-in to the operator because AOTriton classifies the
kernels as experimental. See the module docstring in `src/drinkme/sdpa.py`.

## systemd

If you run drinkme as a systemd service, add the variable in a drop-in:

```ini
# ~/.config/systemd/user/<your-unit>.service.d/10-aotriton.conf
[Service]
Environment=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Run `systemctl --user daemon-reload`, restart the service, then check that
`/health` and the boot log's `[drinkme] device:` line name your GPU.

## SDPA backend limits

The tested AOTriton build contains gfx1151 kernels but classifies the architecture
as experimental. The environment variable enables that dispatch. The tested
AOTriton 0.11/0.12 gfx11xx path supports head dimensions up to 256.
Qwen3.8-27B is within that limit; head dimension 512 still fell back to MATH
with the flag enabled. gemma-4-31B's layers differ: its 50 sliding layers
are head dimension 256 and its 10 full layers 512. The boot log probes each
width and prints one line per width, so on gfx11xx the warning names the
full layers and the sliding layers report their efficient backend.

On gfx1151/TheRock ROCm 7.13, BF16 causal attention at T=4096 with 24 query
heads, four KV heads, and head dimension 256 measured:

| SDPA backend | Peak allocation | Difference from MATH |
|---|---|---|
| MATH | 3.94 GiB | Reference |
| FLASH | 0.23 GiB | Max absolute 7.8e-3; mean 5.6e-5 |
| MEM_EFFICIENT | Unavailable for grouped KV | — |

With expanded KV heads, MATH used 3.98 GiB, FLASH 0.31, and MEM_EFFICIENT
0.36. Differences were at most 1.6e-2, mean 5.6e-5. These attention changes
apply to stock and compressed arms equally; near-tie token choices may change.

## The vision tower on gfx1151

Two kernels a vision tower would run are unusable on gfx1151 (TheRock ROCm
7.13, torch 2.12.0a0, AOTriton with the experimental flag set), and drinkme
routes around both on ROCm devices. CPU and CUDA run the stock calls.

**Attention at head width 72.** AOTriton's attention returns wrong values at
head width 72, the width of the Qwen3.5 and gemma-4 ViTs (16 heads over a
hidden size of 1152). On Qwen3.8-27B's own tower activations for a 1920x1080
screenshot, flash is off by up to 1,240 on outputs under 13 in magnitude, at
all 27 layers, and mem-efficient returns NaN, so the tower's output is all
NaN. gemma-4's unmasked calls (an image that fills its patch budget) go
wrong the same way, by up to 9,085, with NaN features. On synthetic inputs at
16 heads, flash at 72 is within tolerance at 1,024 keys and wrong from 4,096
keys (432 times the tolerance, then NaN); widths 64, 80, 96, 112 and 128 are
within tolerance at every shape measured, up to 2,048 queries over 16,384
keys. drinkme zero-pads a ViT head to the next multiple of 16 before the
call (72 becomes 80) and drops the padding after. The zero columns add
nothing to q·k, so the result is exact, and flash at 80 is no slower: 12.0 ms
per 2^25-pair dispatch at 2560x1440, against 12.8 ms at 72. Muse-Glimmer's
ViT is 96 wide and is not padded.

The engine checks this at boot. On an accelerator it runs the tower's
attention once at the served width, 2,048 queries over 4,096 keys per head
(gemma-4's masked form as well), and compares it with an fp32 reference; a
kernel that disagrees turns image input off and leaves text serving
([image input](serve.md#boot-lines-and-the-self-test)). Run unpadded on
gfx1151 (`bench/vision_selftest.py --no-pad`), the same check fails at 72:
max |diff| 26.6 against a tolerance of 0.15.

**The patch embedding.** Qwen3.5's patch embedding is a Conv3d whose kernel
equals its stride. On ROCm a Conv3d goes to MIOpen, which searches its
solvers the first time it sees an input shape, and every image size is a new
shape. On a 640x480 image that search runs MIOpen's naive kernel for 4.3 s per
dispatch, eight times in a row, well past amdgpu's 2-second ring timeout
([chunked prefill](serve.md#chunked-prefill) explains why that matters on a
desktop APU). With MIOpen off, PyTorch's own conv loops at about 2.5 ms per
patch. drinkme runs the embedding as one GEMM over the flattened patches
instead: the same arithmetic in another accumulation order, with its longest
dispatch about 1 ms at that size. gemma-4's and Muse-Glimmer's patch
embeddings are Linears and need nothing.

## Check a runtime

Run without the environment variable to determine whether the installed runtime
still requires it:

```sh
uv run --no-sync python -m drinkme.sdpa
uv run --no-sync python -m drinkme.sdpa 128 32 8
```

The first probe uses head dimension 256 and 24 query/four KV heads; the second
supplies another shape. If an efficient SDPA backend works without the flag, the
opt-in is no longer needed for that configuration.

`serve` runs the same probe at load, reports fallback memory costs using the
model's attention shape, and compares an enabled efficient SDPA backend with MATH.
It does not set the environment variable automatically.
