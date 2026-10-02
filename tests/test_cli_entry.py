"""cli.entry() must let a verdict OUT — both halves of it.

entry() catches SystemExit to derive the exit code, and Python's DEFAULT
SystemExit handler, the one that writes a string payload to stderr, is then
out of the way: unless entry() prints the payload itself,
`sys.exit("refusing to serve: ...")` produces exit 1 and ZERO BYTES.

Through the console entry point, which is how a systemd unit starts the
server, the refusing to serve guard, written to make a silent failure loud,
would itself be silent.

Both halves are asserted here because either alone is a silent failure: a code
with no message tells you nothing, and a message with exit 0 tells a job
runner the wrong thing.
"""

import subprocess
import sys


def _entry_raising(payload: str) -> subprocess.CompletedProcess:
    """Run cli.entry() in a fresh process with main() raising SystemExit."""
    src = (
        "import drinkme.cli as c\n"
        f"c.main = lambda: (_ for _ in ()).throw(SystemExit({payload!r}))\n"
        "c.entry()\n"
    )
    return subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)


def test_string_payload_reaches_stderr_and_exits_nonzero():
    r = _entry_raising("refusing to serve: canary")
    assert r.returncode == 1, f"string payload must exit 1, got {r.returncode}"
    assert "refusing to serve: canary" in r.stderr, (
        "the payload must be VISIBLE — a guard nobody can read is not a guard"
    )


def test_integer_code_is_preserved():
    src = (
        "import drinkme.cli as c\n"
        "c.main = lambda: (_ for _ in ()).throw(SystemExit(3))\n"
        "c.entry()\n"
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert r.returncode == 3


def test_clean_exit_stays_zero_and_quiet():
    src = "import drinkme.cli as c\nc.main = lambda: 0\nc.entry()\n"
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert r.returncode == 0
    assert "refusing" not in r.stderr


def test_a_classified_refusal_carries_its_own_code_and_message():
    """exitcodes.Usage/Refused/CantRunHere are SystemExit subclasses carrying
    a pinned code (docs/cli.md#exit-codes); entry() reads it instead of
    falling through to the generic "a message with no code means 1" rule
    the two tests above exercise."""
    for cls, code in (("Usage", 2), ("Refused", 3), ("CantRunHere", 4)):
        src = (
            "import drinkme.cli as c\n"
            "from drinkme import exitcodes\n"
            f"c.main = lambda: (_ for _ in ()).throw(exitcodes.{cls}('drinkme x: canary'))\n"
            "c.entry()\n"
        )
        r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
        assert r.returncode == code, (cls, r.returncode)
        assert "drinkme x: canary" in r.stderr, (cls, r.stderr)


def test_an_uncaught_exception_exits_1_with_a_traceback():
    """exitcodes.BUG (1) is entry()'s own `except BaseException` branch — a
    genuine bug, distinct from every deliberate refusal (exitcodes.Usage/
    Refused/CantRunHere), which carries its own pinned code instead."""
    src = (
        "import drinkme.cli as c\n"
        "def boom():\n"
        "    raise ValueError('a genuine bug, not a refusal')\n"
        "c.main = boom\n"
        "c.entry()\n"
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert r.returncode == 1
    assert "ValueError: a genuine bug, not a refusal" in r.stderr
    assert "Traceback" in r.stderr
