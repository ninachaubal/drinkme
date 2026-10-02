# Hardware support

CUDA and ROCm share the PyTorch/Triton serving path; Apple silicon serves
through MLX. `drinkme bootstrap` (and the first `drinkme serve` on a fresh
checkout) detects hardware, installs the matching accelerator packages into
the project environment, and tests kernel launches before continuing. A detected
device name does not by itself establish that its kernels or model
combinations have been tested; the table below and
[ROCm](rocm.md#install-lanes)'s target table say what has been.

| Hardware | Stack | Install lane / group | Status |
|---|---|---|---|
| NVIDIA | PyTorch/CUDA + Triton | lane and group `cuda` (the PyPI wheel) | first-run install and self-test; launch-table rows measured on L4, A10, A10G, L40S, H100; decode replays CUDA graphs for the verified model families, which is experimental ([CUDA graphs](serve-kernels.md#cuda-graphs)) |
| AMD, Linux | PyTorch/ROCm + Triton | two lanes, chosen per gfx target ([ROCm](rocm.md#install-lanes)) | verified on Strix Halo and RX 7600 XT |
| Apple silicon | MLX/Metal | lane and group `metal` | BF16 kernels checked on toy and real Qwen3-8B tensors; Qwen3-1.7B and Qwen3-4B served and benched end to end on an M4 (24 GB), 1.21× and 1.25× MLX-LM's BF16, bit-exact. Other families, the 8B and larger on 24 GB, and speculation on MLX are in progress. See [Metal](metal.md) for kernel measurements and test coverage |

## AMD

AMD GPUs take one of two install lanes, chosen per gfx target;
[ROCm](rocm.md#install-lanes) has the lanes, each target's status, and the
manual TheRock installation.

## The probe and the self-test

Before downloading packages, bootstrap checks the following without requiring
ROCm or torch:

- the OS (the ROCm lanes are Linux-only);
- the gfx target, from `/sys/class/kfd/kfd/topology/nodes/*/properties`
  (`gfx_target_version`, decoded `gfx{v/10000}{v/100%100}{v%100:x}`) when
  KFD is up, and from the PCI-id table otherwise — the line says which;
- whether the `amdgpu` module is loaded;
- whether this user can open `/dev/kfd` and `/dev/dri/renderD*` for read
  and write. When not, the fix is printed — `sudo usermod -aG render,video
  $USER` (the groups that actually own the nodes) followed by a log out and
  back in on Debian/Ubuntu; Arch and CachyOS ship the nodes world-rw through
  udev — and nothing is downloaded.

After installation, a child interpreter runs each check and prints its result:

| Check | Establishes |
|---|---|
| `import torch` | the wheel is installed and loads |
| `torch.cuda.is_available()` | the driver, the device nodes and the runtime agree on a device |
| `get_arch_list()` carries the target | the wheel has kernels for this part (the list is quoted; an LLVM generic such as `gfx11-generic` counts) |
| `torch.zeros(16, device="cuda").sum()` | kernel launch; the official wheel segfaults here on gfx1151 |
| a `@triton.jit` kernel from a file | the JIT path every radix kernel takes |
| `sdpa.probe_backends` at Qwen3.8-27B's head shape | which SDPA backends run; MATH-only passes with a one-line note pointing at the [ROCm attention setup](rocm.md#attention-setup) |
| a toy gemv bit-pin | bit-exact sip decoding of one ragged BF16 tensor and M=1 GEMV within the float64 error bound; a minimal case from `bench/radix_gemv_bitpin.py` |

Bootstrap reports SIGSEGV or exit 135/139 from the child as
`lane_cannot_launch`. If the official wheel fails and matching TheRock wheels
exist, it prints the failure and a
[manual TheRock install command](rocm.md#manual-therock-installation). Otherwise,
it stops with the failure reason.

A failed installation or self-test, an unsupported host, or a target with no
installation lane stops the command that triggered setup. `serve`, `pack`,
`bench`, and `verify` print `refused — accelerator bootstrap failed: <reason>`
and exit 4 (this machine's environment — [exit codes](cli.md#exit-codes))
before reading weights.

```sh
uv run --no-sync drinkme bootstrap --dry-run      # the probe, the lane, the plan; installs nothing
uv run --no-sync drinkme bootstrap --detect-only  # the probe and the lane only
uv run --no-sync drinkme bootstrap                # install into the project venv and self-test
```

`--gfx TARGET` selects another AMD target for a dry run.

## Development checks

Read [AGENTS.md](../AGENTS.md) before running commands in an existing checkout.
Keep environment changes separate from testing, and use a separate port when a
server is already running. The CPU suite cannot validate device dispatch or
kernel behavior; run the relevant [hardware gates](../bench/README.md) for changes
to weight loading or computation.
