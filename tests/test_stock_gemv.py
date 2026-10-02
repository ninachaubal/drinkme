"""The stock GEMV route (codec/swap.py's STOCK_GEMV, applied by
serving/kernel_route.route_kernels), CPU only: which mode a box gets,
that the knob is read and refused as documented, that adoption keeps the
checkpoint's own Parameters (a tied head stays tied), that a RawLinear is
told the mode, that "triton" gives every routed raw Linear the twin dict
in any tree, and that off the accelerator the forward is F.linear bit
for bit. What each mode costs on a device is bench/stock_blas_knobs.py's
to measure."""
import pytest
import torch

from drinkme.codec import radix_schedule as rs
from drinkme.codec import swap
from drinkme.codec.swap import (STOCK_GEMV_ENV, RawLinear, StockLinear, install_stock_gemv, stock_gemv_mode,
                                stock_linear)


@pytest.fixture
def arch(monkeypatch):
    """Pretend the device is a given gfx target ("" = not ROCm)."""
    monkeypatch.delenv(STOCK_GEMV_ENV, raising=False)

    def set_arch(name: str):
        monkeypatch.setattr(rs, "_BACKEND_FAMILY", "rocm" if name else "cuda")
        monkeypatch.setattr(rs, "_ARCH", name)

    return set_arch


def test_the_measured_boxes_get_the_twin_kernel_and_every_other_box_f_linear(arch):
    """gfx1151 and gfx1102, where bench/stock_blas_knobs.py timed the twin's
    kernel faster than the library on every shape, run "triton"; other AMD
    targets and CUDA keep F.linear."""
    for measured in ("gfx1151", "gfx1102"):
        arch(measured)
        assert stock_gemv_mode("cuda") == "triton", measured
    for other in ("gfx1201", "gfx1100", ""):
        arch(other)
        assert stock_gemv_mode("cuda") == "linear", other
    assert set(swap.STOCK_GEMV) == {"gfx1151", "gfx1102"}


def test_off_the_accelerator_it_is_always_f_linear(arch, monkeypatch):
    arch("gfx1151")
    assert stock_gemv_mode("cpu") == "linear"
    monkeypatch.setenv(STOCK_GEMV_ENV, "mv")
    assert stock_gemv_mode("cpu") == "linear"


@pytest.mark.parametrize("box", ("gfx1151", "gfx1102", ""))
def test_the_knob_overrides_the_table_and_refuses_a_typo(arch, monkeypatch, box):
    arch(box)
    for mode in ("linear", "mv", "triton"):
        monkeypatch.setenv(STOCK_GEMV_ENV, mode)
        assert stock_gemv_mode("cuda") == mode
        assert stock_gemv_mode("cpu") == "linear"
    monkeypatch.setenv(STOCK_GEMV_ENV, "auto")
    assert stock_gemv_mode("cuda") == ("triton" if box in ("gfx1151", "gfx1102") else "linear")
    monkeypatch.setenv(STOCK_GEMV_ENV, "mvv")
    with pytest.raises(ValueError, match=STOCK_GEMV_ENV):
        stock_gemv_mode("cuda")


@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("rows", (1, 2, 9))
def test_on_the_cpu_the_forward_is_f_linear_bit_for_bit(bias, rows):
    g = torch.Generator().manual_seed(0)
    lin = torch.nn.Linear(64, 48, bias=bias).to(torch.bfloat16)
    x = torch.randn(1, rows, 64, generator=g).to(torch.bfloat16)
    want = torch.nn.functional.linear(x, lin.weight, lin.bias)
    for mode in ("linear", "mv"):
        assert torch.equal(stock_linear(x, lin.weight, lin.bias, mode), want)
    assert torch.equal(stock_linear(x, lin.weight, lin.bias, "triton", swap.raw_gemv_dict(48, 64)), want)
    assert torch.equal(stock_linear(x, lin.weight, lin.bias, "triton"), want)  # builds its own dict
    assert torch.equal(StockLinear.adopt(lin, "mv")(x), want)
    tri = StockLinear.adopt(lin, "triton")
    tri.raw_gemv = swap.raw_gemv_dict(48, 64)
    assert torch.equal(tri(x), want)


class _Tied(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(2048, 1024).to(torch.bfloat16)
        self.up = torch.nn.Linear(1024, 2048, bias=False).to(torch.bfloat16)
        self.small = torch.nn.Linear(1024, 48, bias=False).to(torch.bfloat16)  # under the threshold
        self.head = torch.nn.Linear(1024, 2048, bias=False).to(torch.bfloat16)
        self.head.weight = self.embed.weight


def test_adoption_keeps_the_parameters_and_a_tied_head_tied(monkeypatch):
    """install_stock_gemv on a tree (the device check patched out: the
    tensors are on the CPU): the eligible Linears become StockLinear over
    the SAME Parameter objects, the tied head still shares the embedding's
    weight, a Linear under the threshold is left alone, and "linear" does
    nothing at all."""
    monkeypatch.setattr(StockLinear, "wants", classmethod(
        lambda cls, mod: type(mod) is torch.nn.Linear and min(mod.weight.shape) >= cls.MIN_DIM))
    m = _Tied()
    before = {n: p for n, p in m.named_parameters(remove_duplicate=False)}
    assert install_stock_gemv(m, "cuda", "linear") == 0
    assert type(m.up) is torch.nn.Linear
    assert install_stock_gemv(m, "cuda", "mv") == 2
    assert type(m.up) is StockLinear and type(m.head) is StockLinear and type(m.small) is torch.nn.Linear
    assert m.up.stock_gemv == "mv" and m.head.stock_gemv == "mv"
    assert m.head.weight is m.embed.weight
    after = {n: p for n, p in m.named_parameters(remove_duplicate=False)}
    assert all(after[n] is p for n, p in before.items())
    assert install_stock_gemv(m, "cpu", "mv") == 0  # off the accelerator: nothing


def test_a_raw_fallback_is_told_the_mode():
    """RawLinear (the codec's raw fallback: stock's F.linear) makes its
    one-row call through the mode the route set; F.linear on the CPU."""
    import numpy as np

    from drinkme.codec.radix_pack import raw_dict

    w = (torch.randn(1024, 1024) * 0.02).to(torch.bfloat16)
    U = w.view(torch.int16).numpy().view(np.uint16)
    raw = swap.make_module(raw_dict(U), None, "cpu")
    assert type(raw) is RawLinear and raw.stock_gemv == "linear"
    tree = torch.nn.Sequential(raw)
    assert install_stock_gemv(tree, "cuda", "mv") == 1
    assert raw.stock_gemv == "mv"
    x = torch.randn(1, 1, 1024).to(torch.bfloat16)
    assert torch.equal(raw(x), torch.nn.functional.linear(x, w))


def test_every_arm_records_the_same_mode():
    """route_kernels records the mode, not a count: the stock arm holds
    every Linear raw and the compressed arm only what the codec left, and
    arms.refuse_unless_routed_alike compares the records whole."""
    from drinkme.arms import refuse_unless_routed_alike
    from drinkme.serving.kernel_route import route_kernels

    stock = route_kernels(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)), "cpu")
    comp = route_kernels(torch.nn.Sequential(torch.nn.Linear(4, 4)), "cpu")
    assert stock["stock_gemv"] == comp["stock_gemv"] == "linear"
    refuse_unless_routed_alike("toy", {"stock": stock, "compressed": comp})


# ------------------------------------------------------------ the raw GEMV --
# swap.RAW_GEMV: a codec tree's raw Linears (a tied lm_head in the compressed
# and twin arms and in `drinkme serve` over a pack) make their one-row call
# through the twin's kernel. What it costs on a device, and its bits against
# the twin arm's module and today's route, are bench/raw_head_gemv.py's.


def test_the_measured_box_gets_the_twin_kernel_and_every_other_box_the_stock_gemv(arch, monkeypatch):
    monkeypatch.delenv(swap.RAW_GEMV_ENV, raising=False)
    arch("gfx1151")
    assert swap.raw_gemv_mode("cuda") == "twin"
    assert swap.raw_gemv_mode("cpu") == "stock"
    for other in ("gfx1102", "gfx1201", ""):
        arch(other)
        assert swap.raw_gemv_mode("cuda") == "stock", other
    assert set(swap.RAW_GEMV) == {"gfx1151"}


@pytest.mark.parametrize("box", ("gfx1151", "gfx1102", ""))
def test_the_raw_knob_overrides_the_table_and_refuses_a_typo(arch, monkeypatch, box):
    arch(box)
    for mode in swap.RAW_GEMV_MODES:
        monkeypatch.setenv(swap.RAW_GEMV_ENV, mode)
        assert swap.raw_gemv_mode("cuda") == mode
        assert swap.raw_gemv_mode("cpu") == "stock"
    monkeypatch.setenv(swap.RAW_GEMV_ENV, "auto")
    assert swap.raw_gemv_mode("cuda") == ("twin" if box == "gfx1151" else "stock")
    monkeypatch.setenv(swap.RAW_GEMV_ENV, "triton")
    with pytest.raises(ValueError, match=swap.RAW_GEMV_ENV):
        swap.raw_gemv_mode("cuda")


def _codec_tied(bias: bool = False):
    """_Tied with one codec tensor beside it (a RawLinear, the codec's raw
    fallback): the compressed or twin arm's tree, where the tied head is
    the Linear the codec left raw."""
    import numpy as np

    from drinkme.codec.radix_pack import raw_dict

    m = _Tied()
    w = (torch.randn(1024, 1024) * 0.02).to(torch.bfloat16)
    b = (torch.randn(1024) * 0.02).to(torch.bfloat16) if bias else None
    m.fallback = swap.make_module(raw_dict(w.view(torch.int16).numpy().view(np.uint16)), b, "cpu")
    return m, w, b


@pytest.fixture
def cpu_wants(monkeypatch):
    """StockLinear.wants without its device check: the tensors are on the CPU."""
    monkeypatch.setattr(StockLinear, "wants", classmethod(
        lambda cls, mod: type(mod) is torch.nn.Linear and min(mod.weight.shape) >= cls.MIN_DIM))


def test_a_codec_tree_gives_its_raw_linears_the_twin_dict_and_a_stock_tree_does_not(arch, cpu_wants):
    """install_stock_gemv under raw GEMV "twin": in a tree holding a codec
    tensor every routed raw Linear (the adopted tied head, the RawLinear)
    carries raw_gemv_dict's dict — the twin's launch row for its shape
    (radix_schedule.select_twin), no tensor — over the SAME Parameters;
    in a tree with none (the stock arm, `serve --stock`) nothing does."""
    from drinkme.codec.radix_schedule import select_twin

    arch("gfx1151")
    m, _, _ = _codec_tied()
    assert swap.codec_tree(m) and not swap.codec_tree(_Tied())
    assert install_stock_gemv(m, "cuda", "mv", "twin") == 3
    for mod, (R, C) in ((m.head, (2048, 1024)), (m.up, (2048, 1024)), (m.fallback, (1024, 1024))):
        assert mod.raw_gemv == swap.raw_gemv_dict(R, C)
        assert mod.raw_gemv["rx_launch"] == select_twin(R, C, 1024, (3, 8)).as_dict()
        assert mod.raw_gemv["codec"] == swap.TWIN and not any(torch.is_tensor(v) for v in mod.raw_gemv.values())
        assert mod.stock_gemv == "mv"
    assert m.head.weight is m.embed.weight and type(m.small) is torch.nn.Linear
    # "stock" takes the dict away again (the knob, a second route_kernels call)
    assert install_stock_gemv(m, "cuda", "mv", "stock") == 3
    assert m.head.raw_gemv is None and m.fallback.raw_gemv is None

    stock = _Tied()
    assert install_stock_gemv(stock, "cuda", "mv", "twin") == 2
    assert stock.head.raw_gemv is None and stock.up.raw_gemv is None


def test_the_raw_gemv_applies_where_the_stock_gemv_is_f_linear(arch, cpu_wants):
    """Under stock GEMV "linear" (DRINKME_STOCK_GEMV=linear, or a box whose
    stock GEMV is F.linear) a stock tree is left alone, and a codec tree's
    raw Linears are still adopted for the raw GEMV's one-row call."""
    arch("gfx1151")
    stock = _Tied()
    assert install_stock_gemv(stock, "cuda", "linear", "twin") == 0
    assert type(stock.head) is torch.nn.Linear
    m, _, _ = _codec_tied()
    assert install_stock_gemv(m, "cuda", "linear", "twin") == 3
    assert type(m.head) is StockLinear and m.head.stock_gemv == "linear" and m.head.raw_gemv is not None
    assert install_stock_gemv(m, "cpu", "mv", "twin") == 0  # off the accelerator: nothing


def test_stock_gemv_triton_gives_every_routed_raw_linear_the_twin_dict(arch, cpu_wants):
    """Under stock GEMV "triton" a stock tree (the stock arm, `serve
    --stock`: no codec tensor) has every raw Linear at or above the
    threshold adopted with raw_gemv_dict's dict for its shape, whatever
    the raw GEMV; a codec tree's raw Linears get it too, under raw GEMV
    "stock" as under "twin" (the raw GEMV "stock" is the stock GEMV's
    call). Back to "mv", a stock tree's dicts go."""
    arch("gfx1151")
    for raw in swap.RAW_GEMV_MODES:
        stock = _Tied()
        assert install_stock_gemv(stock, "cuda", "triton", raw) == 2
        for mod in (stock.up, stock.head):
            assert type(mod) is StockLinear and mod.stock_gemv == "triton"
            assert mod.raw_gemv == swap.raw_gemv_dict(2048, 1024)
        assert stock.head.weight is stock.embed.weight and type(stock.small) is torch.nn.Linear
        assert install_stock_gemv(stock, "cuda", "mv", raw) == 2
        assert stock.up.raw_gemv is None and stock.head.raw_gemv is None
    m, _, _ = _codec_tied()
    assert install_stock_gemv(m, "cuda", "triton", "stock") == 3
    assert m.fallback.raw_gemv == swap.raw_gemv_dict(1024, 1024) and m.fallback.stock_gemv == "triton"
    assert m.head.raw_gemv == swap.raw_gemv_dict(2048, 1024)
    assert install_stock_gemv(_Tied(), "cpu", "triton") == 0  # off the accelerator: nothing


@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("rows", (1, 2, 9))
def test_with_the_raw_gemv_set_the_cpu_forward_is_still_f_linear_bit_for_bit(arch, cpu_wants, bias, rows):
    arch("gfx1151")
    m, w, b = _codec_tied(bias)
    install_stock_gemv(m, "cuda", "mv", "twin")
    x = torch.randn(1, rows, 1024).to(torch.bfloat16)
    assert torch.equal(m.head(x), torch.nn.functional.linear(x, m.embed.weight))
    assert torch.equal(m.fallback(x), torch.nn.functional.linear(x, w, b))
    assert torch.equal(stock_linear(x, w, b, "mv", swap.raw_gemv_dict(1024, 1024)),
                       torch.nn.functional.linear(x, w, b))


def test_route_kernels_records_the_raw_gemv_mode_alike_on_every_arm(arch, monkeypatch):
    """The mode, not a count, as for the stock GEMV: the stock arm's tree
    holds no codec tensor, and its record still says the box's raw GEMV,
    so arms.refuse_unless_routed_alike passes a stock arm beside a
    compressed one."""
    from drinkme.arms import refuse_unless_routed_alike
    from drinkme.serving.kernel_route import route_kernels

    monkeypatch.delenv(swap.RAW_GEMV_ENV, raising=False)
    stock = route_kernels(torch.nn.Sequential(torch.nn.Linear(4, 4)), "cpu")
    comp, _, _ = _codec_tied()
    comp = route_kernels(comp, "cpu")
    assert stock["raw_gemv"] == comp["raw_gemv"] == "stock"  # off the accelerator
    refuse_unless_routed_alike("toy", {"stock": stock, "compressed": comp})
