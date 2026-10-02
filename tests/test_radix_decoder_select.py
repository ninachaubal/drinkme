"""radix_ops._decoder: which block decoder a radix tensor's kernels run.
The lean sip decoder (radix_kernel_gpu._decode_sip_lean) on a CUDA build
for sip's (3, 8) whole-block rows, a lean gulp decoder for gulp's
(2, 2, 4, 8) whole-block rows (radix_kernel_gpu._decode_gulp_lean on a ROCm
build, _decode_gulp_lean_cuda on a CUDA one), the scheduled decoder
everywhere else, and DRINKME_RADIX_DECODER as
the A/B override. The kernels themselves are
gated on a GPU (bench/radix_gemv_bitpin.py, radix_lean_gate.py)."""

import pytest

from drinkme.codec import radix_ops, radix_schedule

LEAN, SCHED = radix_ops.DECODER_SIP_LEAN, radix_ops.DECODER_SCHEDULED
GULP = radix_ops.DECODER_GULP_LEAN
GULP_CUDA = radix_ops.DECODER_GULP_LEAN_CUDA


@pytest.fixture
def backend(monkeypatch):
    def use(family, env=None):
        monkeypatch.setattr(radix_schedule, "_BACKEND_FAMILY", family)
        if env is None:
            monkeypatch.delenv(radix_ops.DECODER_ENV, raising=False)
        else:
            monkeypatch.setenv(radix_ops.DECODER_ENV, env)
        radix_ops._decoder.cache_clear()
    yield use
    radix_ops._decoder.cache_clear()


def test_cuda_sip_whole_blocks_run_the_lean_decoder(backend):
    backend("cuda")
    assert radix_ops._decoder((3, 8), True) == LEAN


@pytest.mark.parametrize("widths,whole", [((3, 8), False), ((2, 3, 8), True), ((2, 2, 4, 8), False),
                                          ((4, 8), True), ((2, 8), True)])
def test_everything_else_runs_the_scheduled_decoder_on_cuda(backend, widths, whole):
    backend("cuda")
    assert radix_ops._decoder(widths, whole) == SCHED


def test_rocm_keeps_the_scheduled_decoder_unless_asked(backend):
    backend("rocm")
    assert radix_ops._decoder((3, 8), True) == SCHED
    backend("rocm", "lean")
    assert radix_ops._decoder((3, 8), True) == LEAN
    assert radix_ops._decoder((3, 8), False) == SCHED  # lean only where it applies


def test_rocm_gulp_whole_blocks_run_the_lean_gulp_decoder(backend):
    backend("rocm")
    assert radix_ops._decoder((2, 2, 4, 8), True) == GULP
    assert radix_ops._decoder((2, 2, 4, 8), False) == SCHED  # ragged rows: the twin's order
    for widths in ((2, 3, 8), (2, 2, 4), (2, 2, 3, 8), (3, 8)):
        assert radix_ops._decoder(widths, True) == SCHED
    backend("rocm", "scheduled")
    assert radix_ops._decoder((2, 2, 4, 8), True) == SCHED


def test_cuda_gulp_whole_blocks_run_the_cuda_lean_gulp_decoder(backend):
    backend("cuda")
    assert radix_ops._decoder((2, 2, 4, 8), True) == GULP_CUDA
    assert radix_ops._decoder((2, 2, 4, 8), False) == SCHED  # ragged rows: the twin's order
    backend("cuda", "lean")
    assert radix_ops._decoder((2, 2, 4, 8), True) == GULP_CUDA
    assert radix_ops._decoder((2, 3, 8), True) == SCHED  # lean only where it applies
    backend("cuda", "scheduled")
    assert radix_ops._decoder((2, 2, 4, 8), True) == SCHED


def test_the_override_selects_scheduled_and_refuses_nonsense(backend):
    backend("cuda", "scheduled")
    assert radix_ops._decoder((3, 8), True) == SCHED
    backend("cuda", "fast")
    with pytest.raises(ValueError):
        radix_ops._decoder((3, 8), True)


def test_args_carry_the_decoder_by_row_shape(backend):
    backend("cuda")
    p = {"rx_data": 0, "rx_offsets": 1, "rx_palette": 2, "rx_schedule": 3, "C": 4096, "block_size": 1024,
         "widths": (3, 8)}
    assert radix_ops._args(p)[-1] == LEAN
    assert radix_ops._args({**p, "C": 4097})[-1] == SCHED
    for family, gulp in (("rocm", GULP), ("cuda", GULP_CUDA)):
        backend(family)
        assert radix_ops._args({**p, "widths": (2, 2, 4, 8)})[-1] == gulp
        assert radix_ops._args({**p, "widths": (2, 2, 4, 8), "C": 4097})[-1] == SCHED
