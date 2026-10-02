"""A pack that verifies is not a safe-to-execute pack.

The file hashes cover bytes; it says nothing about whether those bytes decode.
What each layer proves about a radix tensor's block streams:

  radix.validate     from the directory alone, no payload word read: every
                     block's extent holds at least its FIXED streams (the
                     literal plane and tier 0 — their lengths follow from
                     the block's weight count) and at most every stream at
                     its widest (radix.block_word_bounds). A block cut
                     inside those, or a directory whose block runs into its
                     neighbour or past the payload, is refused here.
  the CPU decoders   a block short of its DATA-DEPENDENT streams (tiers
                     past the first and the terminal exponents — their
                     lengths follow from how many weights escaped) is a
                     clean ValueError ('truncated radix stream'; native:
                     'terminal stream mismatch' / 'short exponent stream').
  the device readers no load leaves the tensor's allocation: a reader's
                     reach past a block's start is at most the widest
                     block's word count (block_word_bounds(B)[1]), the
                     directory is bounded by validate, and the allocation
                     carries that many zero words after the payload
                     (swap.to_device_radix; metal/gemv_radix.pad_words).
                     Such a block decodes to garbage — its neighbour's
                     words, or the pad — and never faults. The kernels
                     carry no per-load bound: a mask on the block's end
                     was measured at +2% on gfx1102 (radix_gpu._read).

The Triton claim runs on a GPU (guard-buffer pattern: a sentinel tail after
the pad; the kernel's output must equal a CPU model of the reads, so any
load past the pad shows up as sentinel bits; a 16-word pad — the Metal
module's old constant — is shown to fail it). The Metal claim is the chunk
decoder's address arithmetic, modelled statement for statement on the CPU
(no Mac in this lane; its loads are a superset of the Triton reader's);
bench/radix_bitpin_mlx.py on an M4 is the device gate.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fixtures import realistic_bf16_bits  # noqa: E402

from drinkme.codec import radix, radix_native, radix_pack as rp  # noqa: E402
from drinkme.codec.pack import iter_pack_dir, load_pack_dir, save_pack_dir, validate_tensor, verify_hashes  # noqa: E402

MANT, EXP = 7, 8
OLD_PAD_WORDS = 16  # a fixed 16-word pad: the ragged over-read's, not a reach bound


def pad_words(B, widths):
    """The reach bound the two resident builders allocate after the payload
    (swap.to_device_radix: block_word_bounds(B)[1]; metal/gemv_radix
    .pad_words: the same with a 16-word floor — that module imports mlx, so
    the formula is pinned here)."""
    return max(OLD_PAD_WORDS, int(radix.block_word_bounds(B, MANT, EXP, tuple(widths))[1]))


# ------------------------------------------------------------------ toys --


def _probe(profile="sip", R=3, C=2100, seed=11):
    """A ragged tensor (two full blocks and a 52-weight tail per row) whose
    exponents spread over eleven octaves, so every tier of every profile
    carries escapes and every block has a terminal stream."""
    U = realistic_bf16_bits(R, C, seed=seed, spread=True)
    p = rp.pack_array_radix(U, profile, encoder="numpy")
    assert p is not None
    return U, p


def _n_of(p, block):
    nb = (p["C"] + p["block_size"] - 1) // p["block_size"]
    return min(p["block_size"], p["C"] - (block % nb) * p["block_size"])


def _cut(p, block, keep):
    """The tensor with block `block` cut to its first `keep` words: the words
    after are dropped from the payload and every later offset moves down."""
    offsets = p["rx_offsets"].astype(np.int64)
    start, end = int(offsets[block]), int(offsets[block + 1])
    assert 0 <= keep < end - start
    data = np.delete(p["rx_data"], np.s_[start + keep:end])
    offsets[block + 1:] -= (end - start) - keep
    return dict(p, rx_offsets=offsets.astype(np.uint32), rx_data=data)


def _grow(p, block, extra):
    """The tensor with `extra` zero words appended to block `block`: every
    later offset moves up; every other block is untouched."""
    offsets = p["rx_offsets"].astype(np.int64)
    end = int(offsets[block + 1])
    data = np.insert(p["rx_data"], end, np.zeros(extra, dtype=np.uint32))
    offsets[block + 1:] += extra
    return dict(p, rx_offsets=offsets.astype(np.uint32), rx_data=data)


def _fixed(p, block):
    fewest, _ = radix.block_word_bounds(_n_of(p, block), MANT, EXP, tuple(p["widths"]))
    return int(fewest)


def _most(p, block):
    _, most = radix.block_word_bounds(_n_of(p, block), MANT, EXP, tuple(p["widths"]))
    return int(most)


def _written(tmp_path, name, d):
    pack = str(tmp_path / name)
    save_pack_dir(pack, {"a": d}, {"hfRepo": "x/y", "revision": None, "dtype": "bf16"})
    assert verify_hashes(pack)  # the hashes are self-attested: malformed streams hash clean
    return pack


def _refused_everywhere(tmp_path, name, bad, why):
    with pytest.raises(ValueError, match=why):
        radix.validate(rp.as_radixpack(bad))
    with pytest.raises(ValueError, match=r"pack tensor a \(radix\).*" + why):
        validate_tensor("a", bad)
    pack = _written(tmp_path, name, bad)
    with pytest.raises(ValueError, match=r"pack tensor a \(radix\).*" + why):
        list(iter_pack_dir(pack))
    with pytest.raises(ValueError, match=r"pack tensor a \(radix\).*" + why):
        load_pack_dir(pack)


# ------------------------------------------------- the CPU models of reads --


def _padded_get(data, start, count, width):
    """radix._get over the device allocation: a word inside the payload is
    read, a word past it is the zero pad (no bound on the block's end)."""
    bit = np.arange(count, dtype=np.int64) * width
    word = start + bit // 32
    safe = np.minimum(word, max(len(data) - 1, 0))
    value = np.where(word < len(data), data[safe] if len(data) else 0, 0).astype(np.uint32)
    value >>= (bit % 32).astype(np.uint32)
    cross = bit % 32 + width > 32
    if np.any(cross):
        nxt = np.minimum(word + 1, max(len(data) - 1, 0))
        extra = np.where(word + 1 < len(data), data[nxt] if len(data) else 0, 0).astype(np.uint32)
        value[cross] |= extra[cross] << (32 - bit[cross] % 32).astype(np.uint32)
    return value & ((1 << width) - 1), start + (count * width + 31) // 32


def padded_decode(p):
    """What the served Triton decoder (radix_kernel_gpu._decode_scheduled)
    produces over ANY directory: radix.decode's walk with every load taken
    from the padded allocation and no consumption check; the terminal is
    read only when the block's end lies past its start (the kernel's
    `if end > pos`). On a well-formed tensor it is radix.decode (pinned)."""
    R, C, B = int(p["R"]), int(p["C"]), int(p["block_size"])
    widths = tuple(int(w) for w in p["widths"])
    palette = rp._palette_u8(p)
    offsets = rp._np(p["rx_offsets"], np.uint32).astype(np.int64)
    data = rp._payload(p)
    out = np.empty((R, C), dtype=np.uint16)
    block = 0
    for r in range(R):
        for c in range(0, C, B):
            n = min(B, C - c)
            start, end = int(offsets[block]), int(offsets[block + 1])
            literal, pos = _padded_get(data, start, n, MANT + 1)
            exps = np.zeros(n, dtype=np.uint32)
            active = np.arange(n)
            base = 0
            for width in widths[:-1]:
                code, pos = _padded_get(data, pos, len(active), width)
                escape = (1 << width) - 1
                done = code != escape
                exps[active[done]] = palette[base + code[done]]
                active = active[~done]
                base += escape
            if end > pos:
                exps[active], pos = _padded_get(data, pos, len(active), widths[-1])
            out[r, c:c + n] = ((literal & ((1 << MANT) - 1)) | ((literal >> MANT) << (MANT + EXP))
                               | (exps << MANT))
            block += 1
    return out


def metal_chunk_model(p, record=None):
    """metal/gemv_radix's rx_decode_chunk + rx_tier, statement for statement
    in numpy over one tensor dict: per block, per thread (32 consecutive
    lanes), the 8-word literal vector load and the W0-word code load, the
    chained tiers by rank (a popcount within the thread's escape mask plus
    the prefix over the block's earlier threads; a thread's codes read out
    of words loaded once, rx_tier's window, or for gulp's tier 2 four
    groups of 8 lanes from two words each, rx_lean4; a chained profile's terminal
    tier skipped when the block ends where its stream would start — the
    kernel skips per SIMD step, all of its blocks or none, which on a
    well-formed tensor changes no bit), every load taken from the
    padded allocation (zero past the payload). Returns the decoded bits;
    `record`, when given, collects per block the word indices the kernel
    loads (`wanted`) — the Triton reader's loads are a subset (it reads a
    lane's own literal word where a thread here reads its 8; the tier
    expressions are the same)."""
    R, C, B = int(p["R"]), int(p["C"]), int(p["block_size"])
    widths = tuple(int(w) for w in p["widths"])
    W0 = widths[0]
    NT = len(widths)
    TPB = B // 32
    table = rp.padded_palette_words(p).view(np.uint8)  # the kernel's palette words, as bytes
    offsets = rp._np(p["rx_offsets"], np.uint32).astype(np.int64)
    data = rp._payload(p).astype(np.int64)
    NB = (C + B - 1) // B
    out = np.empty((R, C), dtype=np.uint16)

    def word(idx, wanted):
        wanted.append(idx)
        return int(data[idx]) if idx < len(data) else 0

    for gb in range(R * NB):
        r, b = divmod(gb, NB)
        n = min(B, C - b * B)
        pos = int(offsets[gb])
        wanted = []
        lw = np.zeros((TPB, 8), dtype=np.int64)
        ex = np.zeros((TPB, 32), dtype=np.int64)
        m = np.zeros(TPB, dtype=np.int64)
        amask = np.zeros(TPB, dtype=np.int64)
        cpos = pos + ((n * 8 + 31) >> 5)
        for sub in range(TPB):
            lane0 = sub * 32
            nact = min(32, n - lane0) if lane0 < n else 0
            amask[sub] = 0xFFFFFFFF if nact >= 32 else (1 << nact) - 1
            cw = np.zeros(W0 + 1, dtype=np.int64)
            if nact:
                for k in range(8):                        # the packed_uint4 pair
                    lw[sub, k] = word(pos + 8 * sub + k, wanted)
                for k in range(W0):                       # the packed_uint3 (sip) or W0 words
                    cw[k] = word(cpos + W0 * sub + k, wanted)
            for i in range(32):
                bit = W0 * i
                sh = bit & 31
                if sh + W0 > 32:
                    v = ((cw[bit >> 5] >> sh) | (cw[(bit >> 5) + 1] << (32 - sh))) & ((1 << W0) - 1)
                else:
                    v = (cw[bit >> 5] >> sh) & ((1 << W0) - 1)
                ex[sub, i] = int(table[v])                # byte v of tier 0's padded table
                if v == (1 << W0) - 1:
                    m[sub] |= 1 << i
            m[sub] &= amask[sub]
        spos = cpos + ((n * W0 + 31) >> 5)
        tbase = ((1 << W0) + 2) // 4
        tiers = [(widths[k], False) for k in range(1, NT - 1)] + [(EXP, True)]
        for W, term in tiers:
            if term and NT > 2 and int(offsets[gb + 1]) == spos:
                break  # a chained profile's block with no terminal stream: no lane escaped to it
            counts = np.array([bin(int(x)).count("1") for x in m])
            pre = np.concatenate([[0], np.cumsum(counts)[:-1]])
            tot = int(counts.sum())
            nm = np.zeros(TPB, dtype=np.int64)
            for sub in range(TPB):
                if term and W == EXP:
                    # the terminal window: the thread's escapes are consecutive
                    # bytes from rank pre[sub]; two words read once when they
                    # hold them all, else one load per escaped lane
                    r0 = int(pre[sub])
                    k = r0 & 3
                    if k + int(counts[sub]) <= 8:
                        w0 = w1 = 0
                        if counts[sub]:
                            w0 = word(spos + (r0 >> 2), wanted)
                            w1 = word(spos + (r0 >> 2) + 1, wanted)
                        for i in range(32):
                            if (m[sub] >> i) & 1:
                                ex[sub, i] = (w0 >> (8 * k) if k < 4 else w1 >> (8 * (k - 4))) & 0xFF
                                k += 1
                    else:
                        rk = r0
                        for i in range(32):
                            if (m[sub] >> i) & 1:
                                ex[sub, i] = (word(spos + (rk >> 2), wanted) >> (8 * (rk & 3))) & 0xFF
                                rk += 1
                    continue
                if widths[:3] == (2, 2, 4) and NT == 4 and W == 4:
                    # gulp's 4-bit tier 2 (rx_lean4): four groups of 8 lanes, a
                    # group's codes the 32 bits from its first rank (two words,
                    # read whatever the group holds), a lane's code the one at
                    # its rank within its group; its palette index's byte, the
                    # kernel's one byte load after the tier, is table[tbase*4+v]
                    for g in range(4):
                        bit = 4 * (int(pre[sub]) + bin(int(m[sub]) & ((1 << (8 * g)) - 1)).count("1"))
                        gw = word(spos + (bit >> 5), wanted) | (word(spos + (bit >> 5) + 1, wanted) << 32)
                        gw = (gw >> (bit & 31)) & 0xFFFFFFFF
                        for i in range(8 * g, 8 * g + 8):
                            if not (m[sub] >> i) & 1:
                                continue
                            j = bin(int(m[sub]) & ((1 << i) - 1) & ~((1 << (8 * g)) - 1)).count("1")
                            v = (gw >> (4 * j)) & 15
                            ex[sub, i] = int(table[tbase * 4 + v])
                            if v == 15:
                                nm[sub] |= 1 << i
                    continue
                # a chained tier: the thread's codes are the consecutive bits
                # from W * pre[sub], read once — three words for W <= 2
                # (they always hold them), two for a wider W when they hold
                # them, else one load per escaped lane
                b0 = int(pre[sub]) * W
                k = b0 & 31
                nwin = 3 if W <= 2 else (2 if k + W * int(counts[sub]) <= 64 else 0)
                win = 0
                if nwin and counts[sub]:
                    for q in range(nwin):
                        win |= word(spos + (b0 >> 5) + q, wanted) << (32 * q)
                for i in range(32):
                    if not (m[sub] >> i) & 1:
                        continue
                    j = bin(int(m[sub]) & ((1 << i) - 1)).count("1")
                    if nwin:
                        assert k + W * j + W <= 32 * nwin
                        v = (win >> (k + W * j)) & ((1 << W) - 1)
                    else:
                        bit = b0 + W * j
                        sh = bit & 31
                        v = word(spos + (bit >> 5), wanted) >> sh
                        if sh + W > 32:
                            v |= word(spos + (bit >> 5) + 1, wanted) << (32 - sh)
                        v &= (1 << W) - 1
                    if v == (1 << W) - 1:
                        nm[sub] |= 1 << i
                    else:
                        ex[sub, i] = int(table[tbase * 4 + v])
            m = nm
            spos += (tot * W + 31) >> 5
            tbase += ((1 << W) + 2) // 4
        for sub in range(TPB):
            for i in range(32):
                lane = sub * 32 + i
                if lane >= n:
                    break
                lit = (int(lw[sub, i >> 2]) >> (8 * (i & 3))) & 0xFF
                out[r, b * B + lane] = (lit & 0x7F) | ((lit & 0x80) << 8) | ((ex[sub, i] & 0xFF) << 7)
        if record is not None:
            record.append(dict(block=gb, start=pos, end=int(offsets[gb + 1]), wanted=wanted))
    return out


# ----------------------------------------------------- what validate proves --


def test_a_one_word_block_is_refused_by_the_directory_gate(tmp_path):
    """Reproduced: a sip tensor [1, 1024] with a valid seven-
    entry palette, offsets [0, 1] and ONE payload word passed save_pack_dir,
    verify_hashes and iter_pack_dir; the CPU decoder then raised 'truncated
    radix stream' and the GPU reader would have loaded 351 words past the
    payload. The literal plane alone needs 256 words; with tier 0 the block
    needs 352. Refused from the directory, before a payload word is read."""
    bad = {"rx_palette": np.arange(7, dtype=np.uint8), "rx_offsets": np.array([0, 1], dtype=np.uint32),
           "rx_data": np.zeros(1, dtype=np.uint32), "R": 1, "C": 1024, "bpw": 0.1,
           "codec": "radix", "profile": "sip", "widths": [3, 8], "block_size": 1024}
    _refused_everywhere(tmp_path, "one-word", bad,
                        "block 0 holds 1 word.*fewer than the 352 its literal plane and first tier")
    assert radix.block_word_bounds(1024, MANT, EXP, (3, 8)) == (352, 608)


@pytest.mark.parametrize("profile", ["sip", "gulp"])
@pytest.mark.parametrize("block", [0, 2, 5])  # a full block, a row's ragged tail, another row's tail
def test_a_block_cut_inside_its_literal_plane_is_refused(tmp_path, profile, block):
    """Truncated literal plane: the block's extent ends inside the 8-bit
    literals (a fixed stream: ceil(n / 4) words for n weights) — the case
    where a device reader's literal loads run into the next block, or past
    the payload when it is the last block. Structural: refused at load."""
    _, p = _probe(profile)
    n = _n_of(p, block)
    literal_words = (n * (MANT + 1) + 31) // 32
    bad = _cut(p, block, literal_words - 1)
    _refused_everywhere(tmp_path, f"literal-{profile}-{block}", bad,
                        f"block {block} holds {literal_words - 1} word.*fewer than the {_fixed(p, block)}")


@pytest.mark.parametrize("profile", ["sip", "gulp"])
def test_a_block_cut_inside_its_first_tier_is_refused(tmp_path, profile):
    """The first tier is fixed too (one code per weight): a block holding
    its literals and a partial tier 0 is refused; one holding exactly the
    fixed streams (no escapes at all, or its terminal missing) passes."""
    _, p = _probe(profile)
    for block in (1, 4):
        fixed = _fixed(p, block)
        _refused_everywhere(tmp_path, f"tier0-{profile}-{block}", _cut(p, block, fixed - 1),
                            f"block {block} holds {fixed - 1} word.*fewer than the {fixed}")
        radix.validate(rp.as_radixpack(_cut(p, block, fixed)))  # structurally sound


def test_offsets_pointing_into_another_block_are_refused(tmp_path):
    """A directory whose entry for block 1 starts inside block 0's streams
    (block 0 then holds fewer words than its fixed streams, block 1 begins
    with the tail of block 0's literals); one whose block 1 is given more
    words than any 1024-weight block can use (each decoder requires exact
    consumption, so it could never decode); and a ragged tail block (52
    weights: 13 literal + 5 tier-0 words, at most 13 terminal) given a full
    block's worth. All three are the directory contradicting the codec
    parameters: refused from the directory."""
    _, p = _probe("sip")
    into = dict(p, rx_offsets=p["rx_offsets"].copy())
    into["rx_offsets"][1] = into["rx_offsets"][0] + 100
    _refused_everywhere(tmp_path, "into", into, "block 0 holds 100 word.*fewer than the 352")
    over = _grow(p, 1, 609 - int(p["rx_offsets"][2] - p["rx_offsets"][1]))
    _refused_everywhere(tmp_path, "over", over, "block 1 holds 609 words, more than the 608")
    assert (_fixed(p, 2), _most(p, 2)) == (18, 31)
    tail = _grow(p, 2, 352 - int(p["rx_offsets"][3] - p["rx_offsets"][2]))
    _refused_everywhere(tmp_path, "tail", tail, "block 2 holds 352 words, more than the 31 any block of 52")


def test_every_encoder_output_sits_inside_the_bounds():
    """The bounds are exact properties of the encoder: every block of every
    profile at 32- and 1024-weight blocks, ragged shapes included, lies in
    [fewest, most] — the gate never refuses a pack the encoder wrote."""
    for profile in ("sip", "balanced", "gulp"):
        for block in (32, 1024):
            for shape in ((1, 1), (7, 33), (3, 1025), (17, 3079)):
                U = realistic_bf16_bits(*shape, seed=3, spread=True)
                rpk = radix.pack_array(U, compression_profile=profile, block_size=block)
                if rpk.raw:
                    continue
                radix.validate(rpk)
                nb = (shape[1] + block - 1) // block
                n = np.minimum(block, shape[1] - np.arange(shape[0] * nb) % nb * block)
                fewest, most = radix.block_word_bounds(n, MANT, EXP, rpk.widths)
                length = rpk.offsets[1:].astype(np.int64) - rpk.offsets[:-1].astype(np.int64)
                assert np.all(length >= fewest) and np.all(length <= most)


# ------------------------------------------ what the decoders do past that --


@pytest.mark.parametrize("profile,cut_at", [
    ("sip", "terminal"),   # the terminal exponents dropped: exactly the fixed streams remain
    ("gulp", "terminal"),
    ("gulp", "tier1"),     # cut inside tier 1 (data-dependent: its length is the tier-0 escape count)
    ("sip", "half"),       # half the terminal words
])
def test_missing_terminal_data_passes_the_gate_and_the_cpu_decoders_refuse_it_cleanly(profile, cut_at):
    """The data-dependent case: a block short of its later tiers or its
    terminal exponents is structurally sound — validate accepts it, by
    design (telling takes decoding the block, the 49-second boot gate this
    project removed) — and every CPU decoder raises a clean ValueError:
    the numpy oracle, the native decoder, and decode_back_radix over both."""
    U, p = _probe(profile)
    block = 1
    fixed = _fixed(p, block)
    length = int(p["rx_offsets"][block + 1] - p["rx_offsets"][block])
    assert length > fixed + 2, "the toy block must carry data-dependent streams to cut"
    if cut_at == "terminal":
        keep = fixed
    elif cut_at == "half":
        keep = fixed + (length - fixed) // 2
    else:
        keep = fixed + 1
    bad = _cut(p, block, keep)
    radix.validate(rp.as_radixpack(bad))
    validate_tensor("a", bad)
    with pytest.raises(ValueError, match="truncated radix stream"):
        radix.decode(rp.as_radixpack(bad))
    with pytest.raises(ValueError, match="truncated radix stream"):
        rp.decode_back_radix(bad, encoder="numpy")
    if radix_native.available():
        with pytest.raises(ValueError, match=r"native radix: (short exponent stream|terminal stream mismatch)"):
            rp.decode_back_radix(bad, encoder="native")
    # the intact tensor still decodes on both
    assert np.array_equal(rp.decode_back_radix(p, encoder="numpy"), U)
    # the device model decodes the short block to garbage and every other
    # block to the source bits (each block decodes independently)
    got = padded_decode(bad)
    rows, cols = np.nonzero(got != U)
    assert rows.size > 0 and set(rows) == {0} and cols.min() >= 1024 and cols.max() < 2048


def test_the_native_decoder_refuses_a_short_directory_or_palette_by_name():
    """radix_native.decode hands the library pointers, not lengths: the
    directory and palette sizes are checked in Python before the call."""
    if not radix_native.available():
        pytest.skip("no C++ compiler")
    U, p = _probe("sip")
    with pytest.raises(ValueError, match="native radix: invalid radix block offsets"):
        radix_native.decode(p["rx_palette"], p["rx_offsets"][:-1], p["rx_data"], p["R"], p["C"], p["widths"])
    with pytest.raises(ValueError, match="native radix: invalid radix palette entries"):
        radix_native.decode(p["rx_palette"][:3], p["rx_offsets"], p["rx_data"], p["R"], p["C"], p["widths"])
    assert np.array_equal(radix_native.decode(p["rx_palette"], p["rx_offsets"], p["rx_data"],
                                              p["R"], p["C"], p["widths"]), U)


@pytest.mark.parametrize("profile", ["sip", "balanced", "gulp"])
def test_the_read_models_are_the_oracle_on_a_well_formed_tensor(profile):
    """Both CPU models of the device readers (the Triton walk, the Metal
    chunk decoder) reproduce radix.decode bit for bit on well-formed
    tensors — the models are faithful before they are used as the reference
    for what a reader does on a malformed one — and every word the Metal
    model loads lies inside the payload plus the pad; on a full block,
    inside the block itself or its next two words (the terminal window's
    second word, when the block's last escape starts it; a 2-bit tier's
    three-word window, when its last codes start the window and no stream
    follows them in the block; gulp's tier-2 group window, when the group
    starts at the stream's end and no stream follows it)."""
    for shape, spread in (((3, 2100), True), ((2, 1024), True), ((3, 40), False), ((2, 2048), None)):
        U = realistic_bf16_bits(*shape, seed=7, spread=bool(spread))  # the 40-weight rows compress only tight
        if spread is None:
            # one block of uniform bits, the rest real-shaped: a gulp block whose
            # terminal tier carries escapes beside blocks whose terminal is
            # empty (the chunk decoder skips those)
            U[1, 1024:] = np.random.default_rng(5).integers(0, 1 << 16, size=1024, dtype=np.uint16)
        p = rp.pack_array_radix(U, profile, encoder="numpy")
        assert p is not None
        assert np.array_equal(padded_decode(p), U)
        record = []
        assert np.array_equal(metal_chunk_model(p, record), U)
        nb = (shape[1] + 1023) // 1024
        payload = int(p["rx_offsets"][-1])
        for rec in record:
            assert max(rec["wanted"]) < payload + pad_words(1024, p["widths"])
            if _n_of(p, rec["block"]) == 1024:
                assert max(rec["wanted"]) <= rec["end"] + 1  # a full block: two words past its end at most
        assert len(record) == shape[0] * nb


def test_the_runtime_dict_carries_the_pad_and_the_payload_view(tmp_path):
    """swap.to_device_radix allocates block_word_bounds(B)[1] zero words
    after the payload (sip 608, gulp 768 at B = 1024) and says where the
    payload ends (NW): the CPU oracle over the runtime dict reads the
    payload alone (radix_pack._payload), read_bytes counts the payload,
    resident_bytes counts the pad (it is resident), and a pack's
    meta.json residentBytes follows."""
    import torch
    from drinkme.codec.pack import resident_bytes
    from drinkme.codec.swap import to_device_radix
    for profile, pad in (("sip", 608), ("gulp", 768)):
        U, p = _probe(profile)
        rt = to_device_radix(p, "cpu")
        assert rt["rx_pad_words"] == pad == pad_words(1024, p["widths"])
        assert rt["NW"] == p["rx_data"].size
        assert rt["rx_data"].numel() == p["rx_data"].size + pad
        assert not rt["rx_data"][rt["NW"]:].any()
        assert np.array_equal(rp._payload(rt), p["rx_data"])
        assert np.array_equal(rp.decode_back_radix(rt, encoder="numpy"), U)
        radix.validate(rp.as_radixpack(rt))
        assert resident_bytes(rt) - 4 * pad == (p["rx_data"].nbytes + p["rx_offsets"].nbytes
                                                + rt["rx_palette"].numel() * 4 + rt["rx_schedule"].numel() * 2)
        from drinkme.codec import radix_ops  # noqa: F401  (triton: on a box with it)
        assert radix_ops.read_bytes(rt) == (p["rx_data"].nbytes + p["rx_offsets"].nbytes
                                            + rt["rx_palette"].numel() * 4 + rt["rx_schedule"].numel() * 2)
        assert torch.is_tensor(rt["rx_data"])


# ------------------------------------------------ the reach bound (Metal) --


@pytest.mark.parametrize("profile", ["sip", "gulp"])
def test_metal_chunk_reader_addresses_stay_inside_the_padded_buffer(profile):
    """The Metal chunk decoder's index expressions (metal/gemv_radix
    _HEADER), modelled here — no Metal in this lane. On the adversarial
    directories: (a) the reader ASKS for words past the payload — past a
    fixed 16-word pad (OLD_PAD_WORDS) on a one-word block and on a last
    block cut inside its literals — so a fixed pad is not a bound; (b) every word it asks for lies inside the
    payload plus pad_words(B, widths): the pad is the bound. The Triton
    reader's loads are a subset of these expressions."""
    _, p = _probe(profile, R=2)
    last = 2 * 3 - 1
    cases = {
        "terminal-inner": _cut(p, 1, _fixed(p, 1)),
        "terminal-last": _cut(p, last, _fixed(p, last)),
        "literal-last": _cut(p, last, 3),
        "one-word": {"rx_palette": p["rx_palette"], "rx_offsets": np.array([0, 1], dtype=np.uint32),
                     "rx_data": np.zeros(1, dtype=np.uint32), "R": 1, "C": 1024, "bpw": 0.1,
                     "codec": "radix", "profile": profile, "widths": p["widths"], "block_size": 1024},
    }
    pad = pad_words(1024, p["widths"])
    for name, bad in cases.items():
        payload = int(bad["rx_offsets"][-1])
        record = []
        metal_chunk_model(bad, record)
        asked = max(max(rec["wanted"]) for rec in record)
        assert asked < payload + pad, name
        if name == "terminal-inner":
            assert max(record[1]["wanted"]) >= record[1]["end"]      # into block 2's words
        elif name == "one-word":
            assert asked >= payload + OLD_PAD_WORDS                   # past the payload AND the old pad
        elif name == "literal-last":
            assert asked >= payload + OLD_PAD_WORDS or _n_of(p, last) < 64
            assert asked >= payload
        else:
            assert asked >= payload                                   # past the payload
    # a full last block cut inside its literals: the reader asks 255 words
    # past the block's start — past the payload and the old pad by hundreds
    _, q = _probe(profile, R=1, C=2048)
    bad = _cut(q, 1, 3)
    record = []
    metal_chunk_model(bad, record)
    asked = max(record[1]["wanted"])
    assert int(bad["rx_offsets"][-1]) + OLD_PAD_WORDS <= asked < int(bad["rx_offsets"][-1]) + pad


def test_the_reach_bound_holds_on_random_directories():
    """Fuzz: random monotone directories over a real payload (any block
    length from 1 word to the whole payload), the one-word block included
    — the Metal model's loads never leave payload + pad."""
    rng = np.random.default_rng(2026)
    _, p = _probe("gulp", R=2)
    words = int(p["rx_offsets"][-1])
    pad = pad_words(1024, p["widths"])
    for _ in range(40):
        cuts = np.sort(rng.integers(1, words, size=5))
        offsets = np.concatenate([[0], cuts, [words]]).astype(np.uint32)
        bad = dict(p, rx_offsets=offsets)
        record = []
        metal_chunk_model(bad, record)
        assert max(max(rec["wanted"]) for rec in record) < words + pad


# ------------------------------------------------------ the Triton readers --


def _have_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _have_gpu(), reason="the Triton readers need a GPU")
@pytest.mark.parametrize("profile", ["sip", "gulp"])
def test_triton_readers_stay_inside_the_padded_allocation(profile):
    """Guard-buffer pattern: the runtime dict's allocation (payload + the
    pad to_device_radix adds) is extended with a 4096-word sentinel tail
    (0xFFFFFFFF — an exponent of 255 and a literal of 255 if ever read) and
    the dense decode over a directory with a short block must equal the
    CPU model of the reads (zeros past the payload). A load past the pad
    would surface as sentinel bits. The run completing is the no-fault
    half of the claim; the fused GEMV over the same buffers agrees with the
    model's weight; (gulp) the schedule the device prepared over the pad
    equals schedule_np's. The negative control: the same buffers with the
    16-word pad the Metal module used before this review DO reach the
    sentinel on a full block cut inside its literals — the test
    discriminates (a 52-weight tail's reach fits in 16 words; a full
    block's literal plane alone is 256)."""
    import torch
    from drinkme.codec import radix_ops
    from drinkme.codec.swap import to_device_radix

    U, p = _probe(profile, R=2)
    last = 2 * 3 - 1
    cases = {
        "terminal-inner": _cut(p, 1, _fixed(p, 1)),
        "terminal-last": _cut(p, last, _fixed(p, last)),
        "literal-last": _cut(p, last, 3),            # below the gate: the allocation's own promise
        "one-word": {"rx_palette": p["rx_palette"], "rx_offsets": np.array([0, 1], dtype=np.uint32),
                     "rx_data": np.zeros(1, dtype=np.uint32), "R": 1, "C": 1024, "bpw": 0.1,
                     "codec": "radix", "profile": profile, "widths": p["widths"], "block_size": 1024},
    }
    if profile == "gulp":
        cases["tier1-inner"] = _cut(p, 1, _fixed(p, 1) + 1)
    _, q = _probe(profile, R=1, C=2048)
    cases["literal-last-full"] = _cut(q, 1, 3)   # a FULL last block cut to 3 words: 255 words of reach
    sentinel = torch.full((4096,), 0xFFFFFFFF, dtype=torch.uint32, device="cuda")
    for name, bad in cases.items():
        expect = padded_decode(bad)
        rt = to_device_radix(bad, "cuda")
        assert rt["rx_data"].numel() == rt["NW"] + rt["rx_pad_words"]
        rt["rx_data"] = torch.cat([rt["rx_data"], sentinel])       # payload | pad | sentinel
        bits = radix_ops.decode_bits(rt)
        torch.cuda.synchronize()
        got = bits.cpu().view(torch.int16).numpy().view(np.uint16)
        assert np.array_equal(got, expect), f"{name}: the dense decode read past the pad"
        if profile == "gulp":
            assert np.array_equal(rt["rx_schedule"].cpu().numpy(), rp.schedule_np(bad)), name
        x = torch.randn(int(bad["C"]), generator=torch.Generator().manual_seed(1)).to(torch.bfloat16).cuda()
        y = radix_ops.gemv_fused(rt, x, None, torch.float32).reshape(-1).double()
        W = torch.from_numpy(expect.view(np.int16)).view(torch.bfloat16).cuda().double()
        ref = W @ x.double()
        bound = 2e-6 * (W.abs() @ x.double().abs()) + 1e-7
        assert bool(((y - ref).abs() <= bound).all()), f"{name}: the GEMV read past the pad"
        if name == "literal-last-full":
            # the negative control: 16 words of pad, then the sentinel
            short = dict(rt, rx_data=torch.cat([rt["rx_data"][: rt["NW"] + OLD_PAD_WORDS], sentinel]))
            got = radix_ops.decode_bits(short).cpu().view(torch.int16).numpy().view(np.uint16)
            assert not np.array_equal(got, expect), "a 16-word pad must not pass this test"
