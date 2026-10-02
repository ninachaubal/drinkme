"""A record is checked against the lexicon before anything leaves the
machine. This is a small validator for the lexicon subset our schema uses —
object/required/properties, string (maxLength in UTF-8 bytes, enum, const,
format datetime), integer (min/max), boolean, array (items, maxLength),
ref (same-document `#name`), unknown, blob — written to the Lexicon spec's
rules: unexpected fields are ignored ("treated at worst as warnings"), a
`$type` on a record must match the lexicon id. Not a general lexicon
engine; a lexicon feature it does not understand fails loudly rather than
passing silently.

Errors name the field by JSON path (`environment.memoryBytes`) so a refusal
tells the user what to look at.

The lexicon file is found in the package (a wheel carries it under
publish/lexicons/) or in the checkout's lexicons/ directory — one file,
found either way.
"""

from __future__ import annotations

import json
import os
import re

from .. import MEASUREMENT
from ..codec.identity import is_commit_sha

# ISO 8601 / RFC 3339 with a mandatory timezone, as the lexicon `datetime`
# format demands: 2026-09-12T20:01:02.345678+00:00, ...Z, ...+05:30.
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


class ValidationError(ValueError):
    pass


def lexicon_path(nsid: str = MEASUREMENT) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    packaged = os.path.join(here, "lexicons", f"{nsid}.json")
    if os.path.exists(packaged):
        return packaged
    d = here
    for _ in range(6):
        d = os.path.dirname(d)
        candidate = os.path.join(d, "lexicons", f"{nsid}.json")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"lexicon {nsid}.json not found beside the package or in a "
                            f"lexicons/ directory above it")


def load_lexicon(nsid: str = MEASUREMENT) -> dict:
    with open(lexicon_path(nsid)) as f:
        lex = json.load(f)
    if lex.get("id") != nsid:
        raise ValidationError(f"lexicon file carries id {lex.get('id')!r}, expected {nsid}")
    return lex


def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


class _Checker:
    def __init__(self, lexicon: dict):
        self.lex = lexicon
        self.errors: list[str] = []

    def fail(self, path: str, msg: str) -> None:
        self.errors.append(f"{path or '<record>'}: {msg}")

    def resolve_ref(self, ref: str, path: str) -> dict | None:
        if ref.startswith("#"):
            name = ref[1:]
        elif "#" in ref:
            nsid, name = ref.split("#", 1)
            if nsid != self.lex["id"]:
                self.fail(path, f"ref to another lexicon ({ref}) is not supported here")
                return None
        else:
            name = "main"
        d = self.lex.get("defs", {}).get(name)
        if d is None:
            self.fail(path, f"lexicon has no def {ref!r}")
        return d

    def check(self, value, schema: dict, path: str) -> None:
        t = schema.get("type")
        if t == "ref":
            target = self.resolve_ref(schema["ref"], path)
            if target is not None:
                self.check(value, target, path)
        elif t == "object":
            self.check_object(value, schema, path)
        elif t == "string":
            self.check_string(value, schema, path)
        elif t == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                self.fail(path, f"expected integer, got {_kind(value)}")
                return
            if "minimum" in schema and value < schema["minimum"]:
                self.fail(path, f"{value} < minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                self.fail(path, f"{value} > maximum {schema['maximum']}")
        elif t == "boolean":
            if not isinstance(value, bool):
                self.fail(path, f"expected boolean, got {_kind(value)}")
        elif t == "array":
            if not isinstance(value, list):
                self.fail(path, f"expected array, got {_kind(value)}")
                return
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                self.fail(path, f"{len(value)} items > maxLength {schema['maxLength']}")
            if "minLength" in schema and len(value) < schema["minLength"]:
                self.fail(path, f"{len(value)} items < minLength {schema['minLength']}")
            for i, item in enumerate(value):
                self.check(item, schema["items"], f"{path}[{i}]")
        elif t == "unknown":
            if not isinstance(value, dict):
                self.fail(path, f"unknown-typed field must be an object, got {_kind(value)}")
        elif t == "blob":
            if not (isinstance(value, dict) and value.get("$type") == "blob"
                    and "ref" in value and "mimeType" in value and "size" in value):
                self.fail(path, "expected a blob reference ($type blob, ref, mimeType, size)")
        elif t in ("bytes", "cid-link", "union"):
            pass  # not used by this lexicon; accepted rather than misjudged
        else:
            self.fail(path, f"lexicon type {t!r} is not supported by this validator")

    def check_string(self, value, schema: dict, path: str) -> None:
        if not isinstance(value, str):
            self.fail(path, f"expected string, got {_kind(value)}")
            return
        if "const" in schema and value != schema["const"]:
            self.fail(path, f"must be {schema['const']!r}")
        if "enum" in schema and value not in schema["enum"]:
            self.fail(path, f"{value!r} not one of {schema['enum']}")
        if "maxLength" in schema and _utf8_len(value) > schema["maxLength"]:
            self.fail(path, f"{_utf8_len(value)} bytes > maxLength {schema['maxLength']}")
        if "minLength" in schema and _utf8_len(value) < schema["minLength"]:
            self.fail(path, f"{_utf8_len(value)} bytes < minLength {schema['minLength']}")
        if schema.get("format") == "datetime" and not _DATETIME.match(value):
            self.fail(path, f"{value!r} is not an ISO 8601 datetime with a timezone")

    def check_object(self, value, schema: dict, path: str) -> None:
        if not isinstance(value, dict):
            self.fail(path, f"expected object, got {_kind(value)}")
            return
        props = schema.get("properties", {})
        nullable = set(schema.get("nullable", []))
        for name in schema.get("required", []):
            if name not in value or (value[name] is None and name not in nullable):
                self.fail(f"{path}.{name}" if path else name, "required, missing")
        for name, sub in props.items():
            if name not in value:
                continue
            if value[name] is None:
                if name not in nullable:
                    self.fail(f"{path}.{name}" if path else name, "null is not allowed")
                continue
            self.check(value[name], sub, f"{path}.{name}" if path else name)


def _kind(v) -> str:
    return {bool: "boolean", int: "integer", float: "float", str: "string",
            list: "array", dict: "object", type(None): "null"}.get(type(v), type(v).__name__)


def _floats(value, path: str, out: list[str]) -> None:
    """The atproto data model has no float type: a PDS refuses 1.5 and
    rewrites 124.0 as 124 (both measured against a self-hosted PDS), so a
    record
    carrying one either fails to write or fails the read-back compare.
    bench spells floats as decimal strings (bench.lexicon_safe); a record
    from anywhere else is refused here with the path named."""
    if isinstance(value, float):
        out.append(f"{path}: float {value!r} — the atproto data model has no float; "
                   f"write it as a decimal string")
    elif isinstance(value, dict):
        for k, v in value.items():
            _floats(v, f"{path}.{k}" if path else k, out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _floats(v, f"{path}[{i}]", out)


def validate(record, lexicon: dict | None = None) -> list[str]:
    """All the ways `record` fails the lexicon or the data model (empty
    list = valid)."""
    lex = lexicon or load_lexicon()
    c = _Checker(lex)
    if not isinstance(record, dict):
        return [f"<record>: expected object, got {_kind(record)}"]
    main = lex["defs"].get("main", {})
    if main.get("type") != "record":
        return ["<lexicon>: main def is not a record"]
    if record.get("$type") != lex["id"]:
        c.fail("$type", f"expected {lex['id']!r}, got {record.get('$type')!r}")
    c.check_object(record, main["record"], "")
    _floats(record, "", c.errors)
    return c.errors


def revision_verdict(record: dict) -> str | None:
    """None when model.revision is a resolved hub commit (a 40-hex sha —
    what bench writes off the one snapshot every arm loaded); otherwise
    the one-line refusal naming the fix. A record without one has no
    immutable source identity: nobody can group it with another run of
    the same weights or reproduce it. bench writes such a record locally
    (a local checkpoint directory has only a structural digest, kept as
    raw.resolved_revision) and marks it raw.unpublishable; this is the
    door that keeps it off the network."""
    rev = (record.get("model") or {}).get("revision") if isinstance(record.get("model"), dict) else None
    if is_commit_sha(rev):
        return None
    what = "no model.revision" if rev is None else f"model.revision {rev!r} is not a resolved hub commit"
    return (f"{what} — the record cannot name the commit its arms loaded, so nobody can "
            f"group or reproduce it; bench against a hub repo (a --model menu entry, or a "
            f"--pack-dir bound to a hub commit) so the revision resolves to the 40-hex sha — "
            f"a local checkpoint directory has only a structural digest "
            f"(raw.resolved_revision), and its record stays local")


# Keys a record written by an older bench may carry that never go on the
# network: raw.pack_dir was the compressed arm's local pack path, which
# names the operator's home directory. `drop_local` removes them before
# publish; `local_path_verdict` refuses anything that still looks like a
# local path, so a key nobody listed fails closed instead of leaking.
LOCAL_RAW_KEYS = ("pack_dir",)
_LOCAL_PATH = re.compile(r"(?:^|[\s\"'(=:])(?:/home/|/Users/|/root/|~/|[A-Za-z]:\\Users\\)")


def drop_local(record: dict) -> dict:
    """The record as it may be published: a copy without LOCAL_RAW_KEYS."""
    raw = record.get("raw")
    if not isinstance(raw, dict) or not any(k in raw for k in LOCAL_RAW_KEYS):
        return record
    return {**record, "raw": {k: v for k, v in raw.items() if k not in LOCAL_RAW_KEYS}}


def local_path_verdict(record: dict) -> str | None:
    """None, or the first field (JSON path) whose string names a local home
    directory. Records are public and permanent once published."""
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                hit = walk(v, f"{path}.{k}" if path else k)
                if hit:
                    return hit
        elif isinstance(o, list):
            for i, v in enumerate(o):
                hit = walk(v, f"{path}[{i}]")
                if hit:
                    return hit
        elif isinstance(o, str) and _LOCAL_PATH.search(o):
            return path
        return None
    where = walk(record, "")
    if where is None:
        return None
    return (f"{where} holds a local path (a home directory); a published record is public "
            f"and permanent, so it does not leave the machine with one")


def decode_step_verdict(record: dict) -> str | None:
    """None, or why the record's arms are not like for like: bench marks
    `raw.<arm>_decode_step` per arm (eager, or a replayed CUDA graph,
    serving/cudagraph.py), and a record whose arms ran different step paths
    compares two runtimes, not two weight reads. bench writes such a record
    locally and marks it raw.unpublishable; this is the door."""
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    steps = {arm: raw.get(f"{arm}_decode_step") for arm in ("stock", "twin", "compressed")}
    steps = {arm: v for arm, v in steps.items() if v}
    if len(set(steps.values())) <= 1:
        return None
    return ("the arms decoded on different step paths ("
            + ", ".join(f"{a} {v}" for a, v in steps.items())
            + "): a CUDA graph capture failed on some arms only; re-run `drinkme bench`, "
              "or set DRINKME_CUDA_GRAPHS=0 to time every arm eager")
