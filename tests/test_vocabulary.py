"""Vocabulary consistency: the public names are fixed as of 1.0.0.

One word per concept, the same word on the CLI, in the environment, in a
measurement record, on /v1/models and in code; docs/architecture.md's
vocabulary table is the reference. This test reads src/ and the lexicon
JSON and fails on the retired spellings, by file and line (the docs are
prose, reviewed rather than pinned):

- `--backend` / `DRINKME_BACKEND`: the runtime (torch | mlx) is selected by
  `--runtime` / `DRINKME_RUNTIME`.
- `pack_format_version`: one spelling in code, `format_version`
  (meta.json's on-disk key stays `formatVersion`).
- `"meanBpw"` as a key anywhere but the pack writer and the two loaders that
  read meta.json: the record and /v1/models say `bitsPerWeight` (weight-
  weighted) and `meanTensorBitsPerWeight` (the unweighted mean over tensors,
  which is what meta.json's `meanBpw` is — codec/pack.py).
- a lexicon `environment` property named `backend` or `device`: the compute
  platform (cuda | rocm | metal) is `platform`, the normalized device class
  is `deviceClass`.
- bare "backend" in the lexicons: the one backend that survives is the
  SDPA backend (sdpa.py, torch's attention kernels), and it is always
  spelled with its adjective.

The allowlist is by path and short on purpose: a growing one is the smell.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEXICON = ROOT / "lexicons" / "wtf.petrichor.drinkme.measurement.json"

# sdpa.py's boot-log line; the SDPA backend keeps its name, so a quote of
# the line may too
SDPA_LOG_LINE = "No efficient attention backend"

# the on-disk key `meanBpw` stays (renaming it means re-packing every pack);
# these are the files that write it and read it off meta.json
MEANBPW_FILES = (
    "src/drinkme/codec/pack.py",          # the writer, and bpw_of_pack_dir
    "src/drinkme/serving/engines.py",     # the torch loader reads meta.json into the engine's meta
    "src/drinkme/serving/engine_mlx.py",  # the mlx loader, the same read
)


def _files(*dirs: str):
    for d in dirs:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".md":
                yield p


def _hits(pattern: re.Pattern, dirs: tuple[str, ...], allow: tuple[str, ...] = ()) -> list[str]:
    out = []
    for p in _files(*dirs):
        rel = p.relative_to(ROOT).as_posix()
        if rel in allow:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                out.append(f"{rel}:{n}: {line.strip()}")
    return out


def _bare_backend(line: str) -> bool:
    """`backend`/`backends` not immediately preceded by `SDPA` (a space or a
    hyphen between, so `SDPA backend` and a `#sdpa-backend-on-rocm` anchor
    both pass); identifiers like `probe_backends` are not words; sdpa.py's
    own boot-log line may be quoted."""
    line = line.replace(SDPA_LOG_LINE, "")
    return any(not m.group(0).lower().startswith("sdpa")
               for m in re.finditer(r"\b(?:sdpa[ -])?backends?\b", line, re.I))


@pytest.mark.parametrize("name, pattern, dirs, allow", [
    ("--backend is --runtime", re.compile(r"--backend\b"), ("src", "lexicons"), ()),
    ("DRINKME_BACKEND is DRINKME_RUNTIME", re.compile(r"\bDRINKME_BACKEND\b"), ("src", "lexicons"), ()),
    ("pack_format_version is format_version", re.compile(r"\bpack_format_version\b"), ("src", "lexicons"), ()),
    ("meanBpw is a meta.json key only", re.compile(r"""["']meanBpw["']"""), ("src", "lexicons"), MEANBPW_FILES),
])
def test_retired_spellings_are_gone(name, pattern, dirs, allow):
    hits = _hits(pattern, dirs, allow)
    assert not hits, f"{name}:\n" + "\n".join(hits)


def test_lexicons_say_sdpa_backend_never_bare_backend():
    hits = []
    for p in _files("lexicons"):
        rel = p.relative_to(ROOT).as_posix()
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _bare_backend(line):
                hits.append(f"{rel}:{n}: {line.strip()[:160]}")
    assert not hits, "bare 'backend' (the runtime is torch|mlx, the platform is cuda|rocm|metal, " \
                     "the SDPA backend keeps its adjective):\n" + "\n".join(hits)


def test_lexicon_environment_names_platform_and_device_class():
    props = json.loads(LEXICON.read_text())["defs"]["environment"]["properties"]
    assert "backend" not in props, "environment.backend is environment.platform"
    assert "device" not in props, "environment.device is environment.deviceClass"
    assert "platform" in props and props["platform"]["enum"] == ["cuda", "rocm", "metal"]
    assert "deviceClass" in props
