"""What `drinkme publish` remembers between runs, under
`~/.config/drinkme/` ($XDG_CONFIG_HOME honoured; DRINKME_CONFIG_DIR
overrides, which is how the tests keep out of a real home):

- publish.json — the identity that worked last time (handle, did, plc), so
  the next run needs no --handle. Not secret.
- session.json — the token set (access + refresh), the issuer and PDS it
  came from, the client_id and scope it was granted under, and the
  session's DPoP PRIVATE key (a JWK with `d`). Mode 0600, written through a
  temp file and os.replace so a crash mid-write leaves the old file whole.
  `--logout` deletes it.

The directory itself is created 0700. Nothing here ever holds a password:
the OAuth flow gives the PDS's own login page the password, not us.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile

CONFIG_NAME = "publish.json"
SESSION_NAME = "session.json"

CLEAR = object()
"""Pass as a save_config() field value to remove that key from the stored
config. None means 'leave whatever is there alone' — the two are not the
same thing, and conflating them is how a cleared handle stays remembered."""


def config_dir() -> str:
    override = os.environ.get("DRINKME_CONFIG_DIR")
    if override:
        return override
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "drinkme")


def _ensure_dir() -> str:
    d = config_dir()
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def _write_private(path: str, data: dict) -> None:
    """0600 from the first byte: the temp file is created with that mode,
    filled, then renamed over the target."""
    d = _ensure_dir()
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_config() -> dict:
    return _read(os.path.join(config_dir(), CONFIG_NAME)) or {}


def save_config(**fields) -> str:
    """None fields are left as-is; CLEAR fields are removed; anything else
    overwrites."""
    cfg = load_config()
    for k, v in fields.items():
        if v is CLEAR:
            cfg.pop(k, None)
        elif v is not None:
            cfg[k] = v
    path = os.path.join(config_dir(), CONFIG_NAME)
    _write_private(path, cfg)
    return path


def session_path() -> str:
    return os.path.join(config_dir(), SESSION_NAME)


def load_session() -> dict | None:
    """The stored session, or None. A session file readable by others is
    refused (deleted is the caller's call; we just will not use it)."""
    path = session_path()
    data = _read(path)
    if data is None:
        return None
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return None
    if mode & 0o077:
        raise PermissionError(f"{path} is mode {mode:o}; refusing to use a token file "
                              f"readable by others — chmod 600 it or `drinkme publish --logout`")
    return data


def save_session(session: dict) -> str:
    path = session_path()
    _write_private(path, session)
    return path


def delete_session() -> bool:
    try:
        os.unlink(session_path())
        return True
    except FileNotFoundError:
        return False
