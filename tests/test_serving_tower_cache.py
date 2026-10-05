"""serving/tower_cache.py on its own: the key, the LRU under a byte cap,
the 0 = off knob and its parsing. CPU tensors and stand-in media items;
the engine's use of it is tests/test_serving_media_reuse.py's."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from drinkme.serving import tower_cache as tc

GIB = 1024 ** 3


@dataclass(frozen=True)
class Item:
    """What the cache reads off a PreparedImage or PreparedVideo."""

    digest: str
    grid_thw: tuple = (1, 2, 2)


@dataclass(frozen=True)
class Clip(Item):
    pass


def rows(n, fill=1.0):
    return torch.full((n, 4), fill, dtype=torch.float32)  # 16 B per row


def test_a_hit_is_the_stored_output_byte_for_byte_and_a_copy():
    c = tc.TowerCache(1024)
    out = torch.randn(3, 4)
    want = out.clone()
    assert c.get(Item("a"), "cpu") is None and c.misses == 1
    assert c.put(Item("a"), out)
    out.add_(1)  # the caller's tensor is not the cache's
    got = c.get(Item("a"), "cpu")
    assert torch.equal(got, want) and c.hits == 1
    assert c.nbytes == 48 and len(c) == 1


def test_the_key_is_identity_kind_digest_and_grid():
    c = tc.TowerCache(1024, identity=("m", "arm", "qwen3_5", "model.visual", "torch.bfloat16"))
    assert c.key(Item("d", (2, 4, 6))) == ("m", "arm", "qwen3_5", "model.visual",
                                           "torch.bfloat16", "Item", "d", (2, 4, 6))
    c.put(Item("d"), rows(2))
    assert c.get(Item("e"), "cpu") is None          # another digest
    assert c.get(Item("d", (1, 4, 1)), "cpu") is None  # another grid
    assert c.get(Clip("d"), "cpu") is None          # another kind
    other = tc.TowerCache(1024, identity=("m", "other arm", "qwen3_5", "model.visual",
                                          "torch.bfloat16"))
    assert other.key(Item("d")) != c.key(Item("d"))
    assert c.get(Item("d"), "cpu") is not None


def test_least_recently_used_goes_first_under_the_cap():
    c = tc.TowerCache(3 * 32)  # three 2-row entries
    for d in "abc":
        assert c.put(Item(d), rows(2))
    assert c.get(Item("a"), "cpu") is not None      # a is now the newest
    c.put(Item("d"), rows(2))                       # b goes
    assert c.get(Item("b"), "cpu") is None
    assert [c.get(Item(d), "cpu") is not None for d in "acd"] == [True] * 3
    assert c.evictions == 1 and c.nbytes == 96
    # a bigger entry evicts as many as it needs, oldest first (a, then c)
    c.put(Item("e"), rows(4))
    assert c.get(Item("d"), "cpu") is not None and c.get(Item("e"), "cpu") is not None
    assert c.get(Item("a"), "cpu") is None and c.get(Item("c"), "cpu") is None
    assert c.nbytes == 96 and c.evictions == 3


def test_an_entry_larger_than_the_cap_is_not_kept_and_evicts_nothing():
    c = tc.TowerCache(64)
    c.put(Item("a"), rows(2))
    assert not c.put(Item("big"), rows(5))
    assert c.get(Item("a"), "cpu") is not None and c.get(Item("big"), "cpu") is None
    assert c.nbytes == 32 and c.evictions == 0


def test_putting_a_key_again_replaces_it():
    c = tc.TowerCache(1024)
    c.put(Item("a"), rows(2, 1.0))
    c.put(Item("a"), rows(3, 2.0))
    assert len(c) == 1 and c.nbytes == 48
    assert torch.equal(c.get(Item("a"), "cpu"), rows(3, 2.0))


def test_zero_is_off():
    c = tc.TowerCache(0)
    assert not c.on and not c.put(Item("a"), rows(1))
    assert c.get(Item("a"), "cpu") is None and (c.hits, c.misses) == (0, 0)
    assert c.describe() == f"off ({tc.ENV}=0)"


def test_clear():
    c = tc.TowerCache(1024)
    c.put(Item("a"), rows(2))
    c.clear()
    assert len(c) == 0 and c.nbytes == 0 and c.get(Item("a"), "cpu") is None


@pytest.mark.parametrize("raw,want", [("", int(tc.DEFAULT_GIB * GIB)), ("0", 0), ("2", 2 * GIB),
                                      ("0.25", GIB // 4), (" 1 ", GIB)])
def test_the_knob(raw, want, monkeypatch):
    monkeypatch.setenv(tc.ENV, raw)
    assert tc.cap_from_env() == want


@pytest.mark.parametrize("raw", ["lots", "-1", "nan"])
def test_a_bad_knob_warns_and_keeps_the_default(raw, monkeypatch, capsys):
    monkeypatch.setenv(tc.ENV, raw)
    assert tc.cap_from_env() == int(tc.DEFAULT_GIB * GIB)
    assert tc.ENV in capsys.readouterr().err


def test_the_flag_wins_over_the_env(monkeypatch, capsys):
    monkeypatch.setenv(tc.ENV, "2")
    assert tc.cap_from_env(0.0) == 0 and tc.cap_from_env(1.0) == GIB
    assert tc.cap_from_env(-1.0) == int(tc.DEFAULT_GIB * GIB)
    assert "--tower-cache-gib" in capsys.readouterr().err


def test_the_default_holds_the_27b_shapes_it_is_documented_to():
    """The module docstring's arithmetic: 10,240 B per token on Qwen3.8-27B
    (hidden 5,120 in bf16), so 0.5 GiB holds 19 2,640-token clips, 14
    3,600-token images, or 4 videos at the 12,288-token budget."""
    cap = int(tc.DEFAULT_GIB * GIB)
    per_token = 5120 * 2
    assert [cap // (n * per_token) for n in (2640, 3600, 12288)] == [19, 14, 4]
