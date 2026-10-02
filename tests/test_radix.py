"""Bit-pattern and adversarial coverage for the radix encoder's CPU oracle."""
from dataclasses import replace
import numpy as np
import pytest
from drinkme.codec import radix


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('profile', list(radix.PROFILES))
def test_all_bit_patterns_in_compressed_tensor(dtype, profile):
    # A compressible background forces every possible pattern through the
    # compressed streams, including both signs, zeros, NaNs and infinities.
    source = np.zeros((512, 512), dtype=dtype)
    alphabet = np.arange(np.iinfo(dtype).max + 1, dtype=dtype)
    source.flat[:len(alphabet)] = alphabet
    p = radix.pack_array(source, compression_profile=profile)
    assert not p.raw
    assert np.array_equal(radix.decode(p), source)
    assert p.nbytes < source.nbytes


@pytest.mark.parametrize('shape', [(1, 1), (1, 31), (7, 33), (3, 1023), (3, 1025), (2, 4101)])
@pytest.mark.parametrize('block', [32, 256, 1024, 4096])
def test_partial_rows_noncontiguous_and_zero_escape_streams(shape, block):
    source = np.full((shape[0], shape[1] * 2), 0x3f81, np.uint16)[:, ::2]
    p = radix.pack_array(source, block_size=block)
    assert np.array_equal(radix.decode(p), source)
    assert p.nbytes <= source.nbytes


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_random_bytes_fall_back_without_expansion(dtype):
    rng = np.random.default_rng(19)
    a = rng.integers(0, np.iinfo(dtype).max + 1, (128, 1024), dtype=dtype)
    p = radix.pack_array(a)
    assert p.raw
    assert p.nbytes == a.nbytes
    assert np.array_equal(radix.decode(p), a)


def test_deterministic_palette_and_bytes():
    a = np.tile(np.arange(65536, dtype=np.uint16), 4).reshape(256, 1024)
    a[:192] = 0x3f80
    one, two = radix.pack_array(a), radix.pack_array(a)
    for field in ('palette', 'offsets', 'data'):
        assert np.array_equal(getattr(one, field), getattr(two, field))


def test_invalid_arrays_and_truncated_streams():
    p = radix.pack_array(np.full((4, 1024), 0x3f80, np.uint16))
    with pytest.raises(ValueError, match='offsets'):
        radix.decode(replace(p, offsets=p.offsets[:-1]))
    offsets = p.offsets.copy()
    offsets[1] = 1
    # a block cut inside its literal plane is refused by validate, from the
    # directory alone, before the decoder reads a word
    with pytest.raises(ValueError, match='fewer than the 320 its literal plane'):
        radix.decode(replace(p, offsets=offsets))
    with pytest.raises(ValueError, match='palette'):
        radix.decode(replace(p, palette=np.zeros_like(p.palette)))
    # a block cut after its fixed streams but short of its terminal exponents
    # is structurally sound (validate accepts it) and the decoder refuses it
    bits = np.full((4, 1024), 0x3f80, np.uint16)
    bits[1, 100:140] = np.arange(40, dtype=np.uint16) << 7  # 40 exponents past sip's 7-entry palette
    p = radix.pack_array(bits, compression_profile='sip')
    fewest, _ = radix.block_word_bounds(1024, 7, 8, p.widths)
    terminal = int(p.offsets[2] - p.offsets[1] - fewest)
    assert terminal == 9  # 33 escaped exponents, 8 bits each, in whole words
    offsets = p.offsets.copy()
    offsets[2:] -= terminal
    cut = replace(p, offsets=offsets, data=np.delete(p.data, np.s_[offsets[2]:offsets[2] + terminal]))
    radix.validate(cut)
    with pytest.raises(ValueError, match='truncated radix stream'):
        radix.decode(cut)


@pytest.mark.parametrize('kwargs', [dict(block_size=31), dict(block_size=512.0),
    dict(widths=()), dict(widths=(2,)), dict(widths=(8, 8, 8)),
    dict(widths=(2, 0, 8)), dict(widths=(2, 4)), dict(compression_profile='unknown')])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        radix.pack_array(np.zeros((4, 1024), np.uint16), **kwargs)


@pytest.mark.parametrize('a', [np.zeros((0, 32), np.uint16), np.zeros(32, np.uint16),
                               np.zeros((32, 32), np.float32), np.zeros((32, 32), np.int16)])
def test_invalid_inputs(a):
    with pytest.raises(ValueError):
        radix.pack_array(a)
