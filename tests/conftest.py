import os

# The CLI's bootstrap runs `uv sync` when it sees an accelerator build with
# zero devices — exactly what a CPU-only suite (HIP_VISIBLE_DEVICES="") looks
# like. Any test that spawns `drinkme <verb>` would re-sync the shared venv.
# The suite never wants auto-deps; set it once, here.
os.environ.setdefault("DRINKME_NO_AUTO_DEPS", "1")

"""Suite-wide defaults that have to hold for every test, not only the ones
that remember to say so.

The cold tier (on-disk prefix slots, serving/slotstore.py) defaults to a real directory under
`~/.cache/drinkme/slots` — the right default for a server, and exactly the
wrong one for a test run, which would write slot files into a developer's
cache and, worse, read a previous run's back and quietly change what a test
measures. Every engine this suite builds serves with the cold tier OFF unless
the test says otherwise; `monkeypatch.setenv` inside a test body runs after
this fixture and wins, which is how tests/test_serving_slotstore.py points it
at a tmp_path.

DRINKME_SPEC (n-gram speculation) is the same shape of problem one layer up. Its shipped
default is AUTO, which resolves to n-gram speculation on any model WITHOUT an
MTP head — including every headless engine this suite builds, and including
`serial_runs`, the MTP tests' own serial reference. A reference run that is
itself speculating is not a reference. So the suite pins DRINKME_SPEC=mtp,
which is the default behaviour exactly (head present -> MTP, head absent -> the
serial loop), and tests/test_serving_ngram.py sets the variable it is
actually testing.

SESSION-scoped, unlike the cold tier above, because `serial_runs` is: a
function-scoped fixture is set up AFTER the session-scoped one it shares a
test with, so a function-scoped setenv would arrive too late for the very run
it exists to protect. Autouse orders before non-autouse at the same scope,
which is what makes this land first.
"""

import pytest


@pytest.fixture(autouse=True)
def _cold_tier_off(monkeypatch):
    monkeypatch.setenv("DRINKME_SLOT_DIR", "off")


@pytest.fixture(scope="session", autouse=True)
def _spec_mode_mtp():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DRINKME_SPEC", "mtp")
        yield



@pytest.fixture(autouse=True)
def _no_hub_commit_lookup(monkeypatch):
    """Tests never touch the hub. cli.pin_revision asks it which commit an
    unpinned repo's main is on (serving/checkpoint.resolve_commit) when the
    local cache cannot say, so a test that reaches `serve`, `pack` or
    `check` for an unpinned repo would make a network call. Here
    that call raises, which resolve_commit treats as "cannot tell"; a test of
    the lookup itself replaces HfApi with its own fake."""
    import huggingface_hub

    def offline(self, *a, **k):
        raise OSError("the test suite never asks the hub for a commit")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", offline)


@pytest.fixture(autouse=True)
def _no_pack_hub(monkeypatch):
    """Tests never download: no pack hub lookup (hub_packs.pack_hub) unless
    a test names its own, whatever the constant or the environment says."""
    monkeypatch.setenv("DRINKME_PACK_HUB", "off")
    monkeypatch.delenv("DRINKME_NO_HUB_PACK", raising=False)


@pytest.fixture(autouse=True)
def checkpoint_table(monkeypatch):
    """suggest.checkpoint_bytes answers from tests/menu_checkpoints.py, keyed by
    repo, and None for any repo not in it: the menu's sizes are read off a
    checkpoint's metadata (the local HF cache, else the Hub), and a test
    must not depend on this machine's cache or touch the Hub. A test adds a
    toy repo with `checkpoint_table["toy/x"] = n`; a test of the real
    lookup holds its own reference to the function, taken at import."""
    from menu_checkpoints import CHECKPOINT_BYTES, TOY_CHECKPOINT_BYTES

    from drinkme import suggest

    table = {**CHECKPOINT_BYTES, **TOY_CHECKPOINT_BYTES}
    monkeypatch.setattr(suggest, "checkpoint_bytes", lambda repo, revision=None: table.get(repo))
    return table


# DRINKME_TEST_HOST=darwin-arm64 runs the suite as if on the Mac a contributor
# has after `drinkme bootstrap`: platform.system() is "Darwin",
# platform.machine() is "arm64", and `import mlx.core` gives a stand-in that
# reports Metal with an M4 (24 GB)'s working set (docs/metal.md's runtime
# check, as that M4 printed it on 09-27). A test that depends on the host pins
# the host it assumes (hosts.pin_linux_host), so it passes both ways; this
# switch is how a Linux box checks that it does (docs/testing.md). The
# stand-in answers the host probes only: tests that need a real mlx skip on
# it by name (TEST_HOST_STANDIN), and the switch is for an interpreter
# without mlx.
TEST_HOST = os.environ.get("DRINKME_TEST_HOST", "").strip().lower()
TEST_HOSTS = ("", "darwin-arm64")
M4_DEVICE_INFO = {"device_name": "Apple M4", "max_recommended_working_set_size": 17179885568,
                  "memory_size": 25769803776, "architecture": "applegpu_g16g"}


def pytest_configure(config):
    if TEST_HOST not in TEST_HOSTS:
        raise pytest.UsageError(f"DRINKME_TEST_HOST={TEST_HOST!r}: expected one of "
                                f"{', '.join(repr(h) for h in TEST_HOSTS if h)} or unset")


@pytest.fixture
def linux_host(monkeypatch):
    """hosts.pin_linux_host for a whole test or module
    (`pytestmark = pytest.mark.usefixtures("linux_host")`)."""
    from hosts import pin_linux_host

    pin_linux_host(monkeypatch)


@pytest.fixture(autouse=True)
def _test_host(monkeypatch):
    if TEST_HOST != "darwin-arm64":
        return
    import importlib.machinery
    import platform
    import sys
    import types

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    core = types.ModuleType("mlx.core")
    core.__version__ = "0.32.2"
    core.metal = types.SimpleNamespace(is_available=lambda: True)
    core.device_info = lambda: dict(M4_DEVICE_INFO)
    pkg = types.ModuleType("mlx")
    pkg.__path__ = []
    pkg.core = core
    # specs, as an installed package has: importlib.util.find_spec("mlx")
    # (transformers' availability check) raises on a module without one
    pkg.__spec__ = importlib.machinery.ModuleSpec("mlx", None, is_package=True)
    core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", None)
    core.TEST_HOST_STANDIN = True  # tests that need the real mlx skip on this one
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
