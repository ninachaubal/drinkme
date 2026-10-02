"""Every `bench/<script>.py` a driver spawns must exist in the tree.

A driver that spawns a bench script the tree no longer carries pays for
the container, the sync and both packs and then reports its gate FAILED on
exit 2. A static check over the string literals catches that at `pytest`
time rather than on a rented card.
"""

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = re.compile(r"""["'](bench/[A-Za-z0-9_]+\.py)["']""")


def test_every_bench_script_a_driver_names_exists():
    missing = {}
    for path in glob.glob(os.path.join(ROOT, "bench", "*.py")) + glob.glob(os.path.join(ROOT, "src", "drinkme", "**", "*.py"), recursive=True):
        with open(path) as f:
            for ref in set(REF.findall(f.read())):
                if not os.path.exists(os.path.join(ROOT, ref)):
                    missing.setdefault(ref, []).append(os.path.relpath(path, ROOT))
    assert not missing, f"scripts named but not in the tree: {missing}"


def test_the_modal_drivers_gate_with_the_radix_bitpins():
    """The Modal served-pair driver runs the radix gates by name."""
    for driver in ("bench/modal_serve_ab.py",):
        with open(os.path.join(ROOT, driver)) as f:
            src = f.read()
        assert "bench/radix_gemv_bitpin.py" in src and "bench/radix_mc_bitpin.py" in src, driver
        assert '"bench/_probe_kernel.py"' not in src, driver
