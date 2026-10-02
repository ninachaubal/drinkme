"""drinkme's exit codes — one meaning each, pinned across every verb
(docs/cli.md#exit-codes):

  OK              0   success
  BUG             1   an uncaught exception, traceback printed — entry()
                      (cli.py) sets this itself; nothing here raises it
                      on purpose
  USAGE           2   the command line is wrong — change the command
  REFUSED         3   a definite verdict about the model, pack or record —
                      use a different model/pack
  CANT_RUN_HERE   4   this machine or its environment — fix the machine,
                      or retry

Rules: a refusal about the model beats an environment problem (a refused
repo on a machine with no compiler is REFUSED, not CANT_RUN_HERE); warnings
never change the code; a clean `serve` shutdown (SIGINT/SIGTERM) is OK.

Usage/Refused/CantRunHere are SystemExit subclasses, not a bare
`raise SystemExit(code)`: every call site keeps raising with its message as
the payload (`raise exitcodes.Refused(f"drinkme pack: {e}")`), exactly the
shape `raise SystemExit(f"...")` always had — so `str(exc)` and
`pytest.raises(SystemExit, match=...)` see the same text as before, and
entry() additionally reads `pinned` off the exception to assign the right
process exit code instead of falling through to its generic "a message with
no code means 1" rule (cli.py's docstring on that rule is unchanged: a
BARE `SystemExit("msg")` — an uncaught one, not one of these — still means 1)."""

from __future__ import annotations

OK = 0
BUG = 1
USAGE = 2
REFUSED = 3
CANT_RUN_HERE = 4


class DrinkmeExit(SystemExit):
    """A SystemExit carrying one of the pinned codes above. Subclasses set
    `pinned`; entry() checks `isinstance(e, DrinkmeExit)` before its
    generic SystemExit handling, so this only ever ADDS a code entry() would
    otherwise have defaulted to 1 — every other SystemExit (argparse's own,
    a bare `sys.exit(2)`, an uncaught one) is untouched."""

    pinned: int

    def __init__(self, message: str):
        super().__init__(message)


class Usage(DrinkmeExit):
    """the command line is wrong"""

    pinned = USAGE


class Refused(DrinkmeExit):
    """a definite verdict about the model, pack or record"""

    pinned = REFUSED


class CantRunHere(DrinkmeExit):
    """this machine or its environment"""

    pinned = CANT_RUN_HERE


class LoadRefused(CantRunHere):
    """a load guard refused: what is live on this machine right now cannot hold
    the weights (or could not be read). Same code as CantRunHere (4); a
    separate type so arms.stock_load_failure can tell a load refusal from
    any other CantRunHere without reading the message."""
