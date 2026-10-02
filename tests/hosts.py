"""The host a test assumes, pinned where drinkme reads it.

drinkme asks platform.system() / platform.machine() which host it is on
(serve.apple_silicon, bench.apple_silicon, bootstrap.detect_lane and
is_metal_box, packs.hardware_budget) and DRINKME_RUNTIME which runtime to
use (serve.resolve_runtime). A test written for a Linux torch box that
reads any of those runs the mlx runtime, the metal lane or MLX's working
set on a Mac instead. pin_linux_host makes the assumption explicit, so the
test is the same test on every host; conftest.py's DRINKME_TEST_HOST is
how a Linux box checks that (docs/testing.md).
"""

from __future__ import annotations

import platform


def pin_linux_host(monkeypatch) -> None:
    """Linux on x86_64, and no DRINKME_RUNTIME: the torch runtime, a
    CUDA/ROCm lane, detect's own budget."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("DRINKME_RUNTIME", raising=False)
