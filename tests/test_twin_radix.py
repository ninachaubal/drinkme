"""metal/twin_radix: the MLX twin's radix decoder is BITWISE the CPU
oracle (codec/radix.decode) on every bf16 pattern, ragged blocks, every
profile and both block sizes the gates use, and refuses a block that does
not end where the directory says.

Each test runs twice where it can. `standin` executes the module's own
source over a numpy stand-in for the handful of mlx.core calls it makes,
so the ALGORITHM is pinned in the CPU suite on every box, mlx or not.
`mlx` runs it on the real mlx (mlx[cpu] on Linux, Metal on a Mac), which is
what pins mlx's own integer semantics; it skips by name without mlx.
bench/mlx_twin_bitpin.py is the same claim over every tensor of a real pack.

The engine half (engine_mlx.RadixTwinLinear, the twin path's forward
bitwise the reference path's) needs the real mlx and is at the end.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

from drinkme.codec import radix, radix_pack

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "drinkme", "metal", "twin_radix.py")


def _standin_mx() -> types.ModuleType:
    """The mlx.core calls twin_radix makes, over numpy: arrays are numpy
    arrays, dtypes numpy dtypes. Integer semantics differ from mlx's only
    in width (numpy may widen uint32 to uint64 in a sum or a mixed shift),
    which the decoder's masks make harmless; the `mlx` run pins mlx's own."""
    mx = types.ModuleType("mlx.core")
    mx.uint8, mx.uint16, mx.uint32 = np.uint8, np.uint16, np.uint32
    mx.array = lambda a, dtype=None: np.array(a, dtype=dtype)
    mx.zeros = lambda shape, dtype=None: np.zeros(shape, dtype=dtype)
    mx.arange = lambda n, dtype=None: np.arange(n, dtype=dtype)
    mx.take = lambda a, idx: np.take(a, idx.astype(np.int64))
    mx.minimum = np.minimum
    mx.where = np.where
    mx.right_shift = lambda a, s: np.right_shift(a, np.asarray(s).astype(a.dtype))
    mx.left_shift = lambda a, s: np.left_shift(a, np.asarray(s).astype(a.dtype))
    mx.cumsum = lambda a, axis=None, inclusive=True: (np.cumsum(a, axis=axis, dtype=a.dtype)
                                                      - (0 if inclusive else a))
    mx.sum = lambda a, axis=None, keepdims=False: np.sum(a, axis=axis, keepdims=keepdims,
                                                         dtype=a.dtype)
    mx.any = np.any
    mx.eval = lambda *a: None
    return mx


def _load_standin(monkeypatch):
    """twin_radix's source executed against the stand-in: the module keeps
    its own reference to the stand-in, and sys.modules gets its real
    entries back when the test ends."""
    mx = _standin_mx()
    pkg = types.ModuleType("mlx")
    pkg.core = mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    spec = importlib.util.spec_from_file_location("drinkme.metal._twin_radix_standin", SRC)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "drinkme.metal"
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=["standin", "mlx"])
def tr(request, monkeypatch):
    """(twin_radix module, to_numpy) on the stand-in or on the real mlx."""
    if request.param == "standin":
        return _load_standin(monkeypatch), np.asarray
    _real_mlx()
    from drinkme.metal import twin_radix

    return twin_radix, np.array


def _real_mlx():
    """mlx.core, or a skip by name: none installed, or conftest's
    DRINKME_TEST_HOST stand-in (which only answers the host probes)."""
    mx = pytest.importorskip("mlx.core")
    if getattr(mx, "TEST_HOST_STANDIN", False):
        pytest.skip("mlx.core is conftest's DRINKME_TEST_HOST stand-in, not an mlx")
    return mx


def _dict(pk: radix.RadixPack) -> dict:
    """iter_pack_dir's shape for a research-encoder pack."""
    R, C = pk.shape
    return {"rx_palette": pk.palette, "rx_offsets": pk.offsets.astype(np.uint32),
            "rx_data": pk.data.astype(np.uint32), "R": R, "C": C, "bpw": pk.bpw,
            "codec": "radix", "profile": "x", "widths": list(pk.widths),
            "block_size": pk.block_size, "layout": 0, "format_version": 1}


def _every_pattern() -> np.ndarray:
    """bench/radix_bitpin_mlx.py's exhaustive tensor: all 65,536 bf16 bit
    patterns, then zeros, as [512, 513] — enough zeros that the codec keeps
    it (a tensor of the patterns alone would expand and be stored raw), and
    513 columns, so every row's last block is ragged at both block sizes."""
    bits = np.zeros((512, 513), np.uint16)
    bits.flat[:65536] = np.arange(65536, dtype=np.uint16)
    return bits


def _weights(R: int, C: int, seed: int) -> np.ndarray:
    """bf16 patterns shaped like a real weight (a narrow exponent spread,
    so most lanes resolve in tier 0 and a few escape), plus a sprinkle of
    zeros, subnormals, infinities and NaN payloads."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((R, C)) * 0.02).astype(np.float32)
    bits = (w.view(np.uint32) >> 16).astype(np.uint16)
    odd = rng.integers(0, R * C, size=max(1, R * C // 50))
    bits.reshape(-1)[odd] = rng.choice(np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x7F80,
                                                 0xFF80, 0x7FC1, 0xFFFF], dtype=np.uint16),
                                       size=odd.size)
    return bits


@pytest.mark.parametrize("profile", ["sip", "balanced", "gulp"])
@pytest.mark.parametrize("block", [1024, 32])
def test_every_bf16_pattern_decodes_to_the_oracles_bits(tr, profile, block):
    mod, host = tr
    bits = _every_pattern()
    pk = radix.pack_array(bits, compression_profile=profile, block_size=block)
    assert not pk.raw
    want = radix.decode(pk)
    assert np.array_equal(want, bits)  # the oracle round-trips the construction
    got = host(mod.decode(_dict(pk)))
    assert got.dtype == np.uint16 and got.shape == bits.shape
    assert np.array_equal(got, want)


@pytest.mark.parametrize("profile", ["sip", "gulp"])
@pytest.mark.parametrize("shape,block", [((7, 1124), 1024), ((5, 100), 32), ((3, 2048), 1024)])
def test_ragged_and_weight_shaped_tensors_match_the_oracle(tr, profile, shape, block):
    """A row's last block holding C's remainder (1124 = 1024 + 100; 100 =
    3 x 32 + 4), and full blocks, over weight-shaped data."""
    mod, host = tr
    bits = _weights(*shape, seed=shape[1])
    pk = radix.pack_array(bits, compression_profile=profile, block_size=block)
    assert not pk.raw
    got = host(mod.decode(_dict(pk)))
    assert np.array_equal(got, radix.decode(pk))
    assert np.array_equal(got, bits)


def test_chunked_rows_decode_the_same_bits(tr, monkeypatch):
    """Rows go through in chunks of CHUNK_LANES: one row per chunk, three
    (the last chunk short), and all rows at once agree bitwise."""
    mod, host = tr
    bits = _weights(10, 2048, seed=3)
    pk = radix.pack_array(bits, compression_profile="gulp")
    want = radix.decode(pk)
    for lanes in (1, 3 * 2048, 1 << 21):
        monkeypatch.setattr(mod, "CHUNK_LANES", lanes)
        assert np.array_equal(host(mod.decode(_dict(pk))), want), lanes


def test_the_native_oracle_agrees_where_there_is_one(tr):
    """decode_back_radix's default (the native decoder when a C++ compiler
    is at hand) is the oracle the engine's reference path runs."""
    mod, host = tr
    bits = _weights(8, 3072, seed=11)
    p = radix_pack.pack_array_radix(bits, "sip")
    assert np.array_equal(host(mod.decode(p)), radix_pack.decode_back_radix(p))


def test_a_block_that_does_not_end_where_the_directory_says_is_refused(tr):
    """Move one interior directory entry by a word: block 1 now claims a
    word block 2 needs. The oracle refuses the pack; so does the twin's
    decoder, by name, rather than decoding garbage."""
    mod, _ = tr
    bits = _weights(1, 4096, seed=5)
    pk = radix.pack_array(bits, compression_profile="sip")
    p = _dict(pk)
    p["rx_offsets"] = p["rx_offsets"].copy()
    p["rx_offsets"][2] += 1
    with pytest.raises(ValueError):
        radix.decode(radix_pack.as_radixpack(p))
    with pytest.raises(ValueError, match=r"radix block 1 does not end where the directory says"):
        mod.decode(p, "t")
    with pytest.raises(ValueError, match="directory does not match"):
        mod.decode(dict(p, rx_offsets=p["rx_offsets"][:-1]), "t")
    with pytest.raises(ValueError, match="palette has"):
        mod.decode(dict(p, rx_palette=p["rx_palette"][:-1]), "t")


def test_chunk_transient_is_charged_per_lane_of_one_chunk(tr):
    mod, _ = tr
    assert mod.chunk_transient_bytes(4, 1024) == mod.BYTES_PER_LANE * 4 * 1024
    # a big tensor is charged one chunk of whole rows, never the whole tensor
    rows = mod.CHUNK_LANES // 4096
    assert mod.chunk_transient_bytes(151936, 4096) == mod.BYTES_PER_LANE * rows * 4096


# ------------------------------------------ the engine: real mlx only --


def _engine():
    mx = _real_mlx()
    pytest.importorskip("mlx_lm")
    from drinkme.serving import engine_mlx

    return mx, engine_mlx


def test_the_twin_linear_is_the_reference_linear_bit_for_bit():
    """RadixTwinLinear decodes once and keeps the bf16 plane; its forward is
    RadixLinear(path="reference")'s bitwise, at M = 1 (mlx's gemv) and
    M > 1, with and without a bias (biased_matmul's two arms)."""
    mx, em = _engine()
    bits = _weights(48, 2048, seed=21)
    p = radix_pack.pack_array_radix(bits, "sip")
    p.update(layout=0, format_version=1)
    rng = np.random.default_rng(2)
    bias = mx.array((rng.standard_normal(48) * 0.1).astype(np.float32)).astype(mx.bfloat16)
    for b in (None, bias):
        twin = em.make_module_mlx(p, b, path=em.TWIN, name="t")
        ref = em.RadixLinear(p, b, path="reference", name="r")
        assert isinstance(twin, em.RadixTwinLinear) and twin.path == em.TWIN
        assert twin.tensor_codec == em.TWIN
        assert twin.resident_bytes() == 2 * 48 * 2048
        assert np.array_equal(twin.decode(), bits)
        assert not hasattr(twin, "_res")  # the streams are not kept
        for shape in ((1, 2048), (3, 2048), (2, 3, 2048)):
            x = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
            a, r = twin(x), ref(x)
            mx.eval(a, r)
            assert a.shape == (*shape[:-1], 48)
            assert np.array_equal(np.array(a.view(mx.uint16)), np.array(r.view(mx.uint16)))


def test_the_twin_path_serves_a_raw_fallback_as_the_compressed_arm_does():
    """swap.make_twin's rule: a raw fallback's twin is the RawLinear the
    compressed arm serves it with."""
    _, em = _engine()
    bits = _weights(8, 64, seed=4)
    lin = em.make_module_mlx({"raw_bits": bits, "R": 8, "C": 64, "bpw": 16.0, "codec": "raw"},
                             path=em.TWIN, name="w")
    assert isinstance(lin, em.RawLinear) and lin.path == em.TWIN
    assert np.array_equal(lin.decode(), bits)
    with pytest.raises(ValueError, match="RadixTwinLinear serves the radix codec only"):
        em.RadixTwinLinear({"raw_bits": bits, "R": 8, "C": 64, "bpw": 16.0, "codec": "raw"},
                           name="x")
