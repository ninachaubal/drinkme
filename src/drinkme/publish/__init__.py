"""`drinkme publish [FILE ...]` — opt-in: put finished bench records on the
USER'S OWN atproto PDS as wtf.petrichor.drinkme.measurement records.

Five steps, each printed as it happens, any one of which stops the run:

1. VALIDATE each record (validate.py) — the file bench wrote under
   ./measurements/ (newest by default) or the paths given. THE DOOR: a
   record without a resolved hub commit in model.revision is refused by
   name with the fix in the reason (a bench of a local checkpoint directory
   writes its record locally and marks it raw.unpublishable; it never
   reaches the network). A record that fails the lexicon is refused with
   the field named; an incomplete run and a baseline-suspect verdict are
   refused too. Refusals are per record (RecordRefused): the other records
   of a set still go, and the refused are listed at the end; a set with
   nothing publishable stops here, before any network.
2. IDENTITY (identity.py): --handle or --did (remembered in
   ~/.config/drinkme/publish.json after the first success) -> DID -> DID
   document -> the PDS that hosts the repo. Handle verified bidirectionally.
3. OAUTH (oauth.py): the loopback public-client flow — PDS metadata ->
   authorization server, PKCE, PAR, DPoP, the browser (URL always printed,
   --no-browser prints only), a one-shot 127.0.0.1 listener for the
   callback, token exchange. The session (tokens + DPoP key) is stored
   0600 under ~/.config/drinkme/ and refreshed on later runs; --logout
   deletes it.
4. WRITE (pds.py): com.atproto.repo.createRecord into the collection;
   the at:// uri is printed.
5. VERIFY: com.atproto.repo.getRecord for that uri, compared byte-equal to
   what was sent after JSON canonicalisation. That read-back is the
   success line. Steps 4 and 5 run once per publishable record.

The exit status is 0 only when every record given was published; a refused
record makes it exitcodes.REFUSED (3) after the others have gone.

bench never imports this package (a test holds that line): auth cannot
leak into the run path.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import webbrowser

from .. import MEASUREMENT, exitcodes
from . import identity, oauth, pds, store, validate
from .jwt import Es256Key, pkce_pair, random_state

CALLBACK_TIMEOUT = 600.0  # a human reading a consent screen
REFRESH_SLACK = 60  # seconds before expiry at which a token counts as expired


class PublishError(Exception):
    """`code` is the exit code `run()` returns for this error (below): most
    of publish's own failures are about reaching an external system (the
    Hub, the PDS, the OAuth server) — exitcodes.CANT_RUN_HERE is the
    default; a caller that means "the command line is wrong" or "a
    definite verdict about this record" passes the other constant."""

    def __init__(self, message: str, code: int = exitcodes.CANT_RUN_HERE):
        super().__init__(message)
        self.code = code


class RecordRefused(PublishError):
    """One record `drinkme publish` will not send, and why — raised by
    check_record so a set can refuse per record and carry on with the
    rest. `reason` is what the operator reads; str() prefixes the path."""

    def __init__(self, path: str, reason: str):
        super().__init__(f"{path}: {reason}", code=exitcodes.REFUSED)
        self.path = path
        self.reason = reason


def _say(line: str) -> None:
    print(line, flush=True)


def _headline(reason: str) -> str:
    """A refusal's first clause, for the inline mark beside the path."""
    return reason.splitlines()[0].split(" — ")[0].rstrip(":")


# ------------------------------------------------------------ the record ----

def newest_measurement(root: str = ".") -> str | None:
    files = glob.glob(os.path.join(root, "measurements", "*.json"))
    return max(files, key=os.path.getmtime) if files else None


def load_record(path: str | None) -> tuple[str, dict]:
    if path is None:
        path = newest_measurement()
        if path is None:
            raise PublishError("no record given and nothing under ./measurements/ — run "
                               "`drinkme bench` first, or pass the file", code=exitcodes.USAGE)
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise PublishError(f"cannot read {path}: {e}", code=exitcodes.USAGE) from None
    try:
        record = json.loads(raw)
    except ValueError as e:
        raise PublishError(f"{path} is not JSON: {e}", code=exitcodes.USAGE) from None
    return path, record


def check_record(record: dict, path: str) -> None:
    """Every reason this record must not leave the machine, as a
    RecordRefused naming the path. The revision door comes first: a record
    without a resolved hub commit fails the lexicon too (model.revision is
    required), but the door's reason is the one that names the fix. The
    decode-step door is next: arms that decoded on different step paths
    are not one comparison. Then no local path anywhere: callers pass the
    record through validate.drop_local first, and anything left fails
    closed."""
    verdict = (validate.revision_verdict(record) or validate.decode_step_verdict(record)
               or validate.local_path_verdict(record))
    if verdict:
        raise RecordRefused(path, verdict)
    errors = validate.validate(record)
    if errors:
        shown = "\n".join(f"    {e}" for e in errors[:12])
        more = f"\n    ... and {len(errors) - 12} more" if len(errors) > 12 else ""
        raise RecordRefused(path, f"not a valid {MEASUREMENT} record:\n{shown}{more}")
    if record.get("incomplete"):
        raise RecordRefused(path, f"an incomplete run ({record['incomplete']}): hardware "
                                  f"identity and bandwidth only, not a point on the curve")


# ------------------------------------------------------------ the session ----

def _expired(session: dict) -> bool:
    return int(time.time()) >= int(session.get("expires_at", 0)) - REFRESH_SLACK


def _session_for(ident: dict, meta: dict) -> tuple[dict | None, oauth.DpopClient | None]:
    """A stored session for this DID on this issuer, refreshed if its access
    token is stale; (None, None) when there is none or it cannot be
    refreshed (then the browser flow runs)."""
    s = store.load_session()
    if not s or s.get("did") != ident["did"] or s.get("issuer") != meta["issuer"]:
        return None, None
    try:
        key = Es256Key.from_jwk(s["dpop_key"])
    except (KeyError, TypeError, ValueError):
        _say("  stored session has no usable DPoP key; authorizing again")
        store.delete_session()
        return None, None
    s["pds"] = ident["pds"]  # the repo may have moved since the token was issued
    client = oauth.DpopClient(key, s.get("nonces"))
    if not _expired(s):
        return s, client
    if not s.get("refresh_token"):
        return None, None
    _say("  access token expired; refreshing")
    try:
        tokens = oauth.refresh_tokens(client, meta, cid=s["client_id"],
                                      refresh_token=s["refresh_token"])
        oauth.check_token_response(tokens, expected_did=ident["did"], issuer=meta["issuer"],
                                   requested_scope=s.get("scope") or "atproto")
    except oauth.OAuthError as e:
        _say(f"  refresh failed ({e}); authorizing again")
        store.delete_session()
        return None, None
    s = oauth.session_from_tokens(tokens, key=key, did=ident["did"], pds=ident["pds"],
                                  issuer=meta["issuer"], cid=s["client_id"],
                                  nonces=client.nonces)
    store.save_session(s)
    return s, client


def authorize(ident: dict, meta: dict, *, open_browser: bool = True,
              login_hint: str | None = None) -> tuple[dict, oauth.DpopClient]:
    """The browser flow, end to end; returns the stored session."""
    key = Es256Key.generate()
    client = oauth.DpopClient(key)
    verifier, challenge = pkce_pair()
    state = random_state()
    listener = oauth.LoopbackCallback(state).start()
    try:
        scope = oauth.preferred_scope()
        try:
            par = oauth.par_request(client, meta, scope=scope, redirect_uri=listener.redirect_uri,
                                    state=state, code_challenge=challenge, login_hint=login_hint)
        except oauth.OAuthError as e:
            if str(e) != "invalid_scope":
                raise
            _say(f"  {meta['issuer']} refused the per-collection scope; asking for the "
                 f"transitional one instead")
            scope = oauth.fallback_scope()
            par = oauth.par_request(client, meta, scope=scope, redirect_uri=listener.redirect_uri,
                                    state=state, code_challenge=challenge, login_hint=login_hint)
        _say(f"  scope     {scope}")
        _say(f"            {oauth.scope_explained(scope)}")
        _say(f"  client_id {par['client_id']}")
        url = oauth.authorize_url(meta, par["client_id"], par["request_uri"])
        _say("  approve in your browser — if nothing opens, visit this URL:")
        _say(f"  {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 — a headless machine; the URL is printed
                pass
        _say(f"  waiting on {listener.redirect_uri} (up to {CALLBACK_TIMEOUT:.0f}s)")
        result = listener.wait(CALLBACK_TIMEOUT)
    finally:
        listener.close()
    if "error" in result:
        raise PublishError(f"authorization did not complete: {result['error']} "
                           f"{result.get('error_description', '')}".strip())
    if result.get("iss") and result["iss"].rstrip("/") != meta["issuer"]:
        raise PublishError(f"callback iss {result['iss']!r} is not {meta['issuer']}")
    if not result.get("code"):
        raise PublishError("callback carried no code")
    tokens = oauth.exchange_code(client, meta, cid=par["client_id"], code=result["code"],
                                 redirect_uri=listener.redirect_uri, code_verifier=verifier)
    granted = oauth.check_token_response(tokens, expected_did=ident["did"],
                                         issuer=meta["issuer"], requested_scope=scope)
    session = oauth.session_from_tokens(tokens, key=key, did=ident["did"], pds=ident["pds"],
                                        issuer=meta["issuer"], cid=par["client_id"],
                                        nonces=client.nonces)
    where = store.save_session(session)
    _say(f"  ✓ authorized as {ident['did']} (granted: {granted})")
    _say(f"  session stored at {where} (mode 0600); `drinkme publish --logout` removes it")
    return session, client


# --------------------------------------------------------------- the run ----

def run(files: list[str] | None = None, *, handle: str | None = None, did: str | None = None,
        plc: str | None = None, logout: bool = False, no_browser: bool = False) -> int:
    """`files`: the records to publish, in order; none = the newest under
    ./measurements/."""
    files = list(files or [])
    if logout:
        gone = store.delete_session()
        _say(f"drinkme publish: session {'removed' if gone else 'was not present'} "
             f"({store.session_path()})")
        if not files and not handle and not did:
            return exitcodes.OK
    try:
        return _run(files, handle=handle, did=did, plc=plc, no_browser=no_browser)
    except PublishError as e:
        print(f"drinkme publish: {e}", file=sys.stderr)
        return e.code
    except (validate.ValidationError, identity.IdentityError,
            oauth.OAuthError, PermissionError) as e:
        # a lexicon file the package should have shipped, a DNS/HTTP failure
        # resolving a handle or DID, an OAuth protocol failure, a permission
        # error writing the session file — all reaching-an-external-system
        # or reading-the-local-install problems: exitcodes.CANT_RUN_HERE.
        # (identity.IdentityError's "'{handle}' is not a handle" case is a
        # bad --handle value, which would read better as USAGE — noted as a
        # case that doesn't split cleanly by exception type alone.)
        print(f"drinkme publish: {e}", file=sys.stderr)
        return exitcodes.CANT_RUN_HERE


def _run(files: list[str], *, handle, did, plc, no_browser) -> int:
    _say("── record ──" if len(files) <= 1 else f"── records ({len(files)}) ──")
    records: list[tuple[str, dict]] = []
    refused: list[RecordRefused] = []
    for file in files or [None]:
        path, record = load_record(file)
        record = validate.drop_local(record)  # older benches wrote raw.pack_dir
        _say(f"  {path}" + ("" if file else "  (newest under ./measurements/)"))
        try:
            check_record(record, path)
        except RecordRefused as e:
            # per record: the others still go; the reason is listed at the end
            refused.append(e)
            _say(f"  ✗ refused — {_headline(e.reason)}")  # the full reason: stderr, at the end
            continue
        model = (record.get("model") or {}).get("name", "?")
        device = (record.get("environment") or {}).get("deviceClass", "?")
        _say(f"  ✓ valid {MEASUREMENT} — version {record.get('version')}, {model} on {device}")
        records.append((path, record))
    if not records:
        # nothing to send: stop here, before any identity resolution or
        # network — every record was a definite refusal (RecordRefused)
        raise PublishError("\n".join(str(e) for e in refused), code=exitcodes.REFUSED)

    _say("── identity ──")
    cfg = store.load_config()
    handle = handle or (None if did else cfg.get("handle"))
    did = did or (None if handle else cfg.get("did"))
    plc = plc or cfg.get("plc")
    if not handle and not did:
        raise PublishError("who are you? pass --handle <your.handle> (or --did) the first "
                           "time; it is remembered after a successful publish",
                           code=exitcodes.USAGE)
    ident = identity.resolve(handle=handle, did=did, plc=plc)
    _say(f"  {ident['handle'] or '(no handle)'} → {ident['did']} → PDS {ident['pds']}")

    _say("── authorize ──")
    meta = oauth.discover(ident["pds"])
    _say(f"  authorization server {meta['issuer']}")
    session, client = _session_for(ident, meta)
    if session is None:
        session, client = authorize(ident, meta, open_browser=not no_browser,
                                    login_hint=handle or did)
    else:
        _say(f"  using the stored session (granted: {session.get('scope')})")

    for path, record in records:
        _say("── write ──" if len(records) == 1 else f"── write {path} ──")
        created = pds.create_record(client, session, MEASUREMENT, record)
        _say(f"  {created['uri']}")
        if created.get("cid"):
            _say(f"  cid {created['cid']}")

        _say("── verify ──")
        back = pds.get_record(client, session, created["uri"])
        sent, got = pds.canonical(record), pds.canonical(back.get("value"))
        if sent != got:
            # a definite verdict about THIS record (what the PDS holds is
            # not what was sent) — exitcodes.REFUSED
            raise PublishError(f"read-back of {created['uri']} differs from what was sent "
                               f"({len(sent)} vs {len(got)} canonical bytes) — the record on "
                               f"the PDS is not the record on disk", code=exitcodes.REFUSED)
        _say(f"  ✓ read back byte-equal ({len(sent)} canonical bytes)")
        _say(f"  your point is published: {created['uri']}")
    if len(files) > 1:
        _say(f"  published {len(records)} of {len(files)}")

    # Remember what worked. Only a handle that went through the bidirectional
    # check is stored — a DID-only flow (a broken handle is the usual reason)
    # stays DID-only next time too, so any handle left over from a *previous*
    # account is explicitly cleared rather than quietly kept as the default.
    store.save_config(handle=handle or store.CLEAR, did=ident["did"], plc=plc)
    if client.nonces != session.get("nonces"):
        session["nonces"] = client.nonces
        store.save_session(session)
    if refused:
        # the ones that did not go, each with its reason, after the ones that did
        for e in refused:
            print(f"drinkme publish: refused {e}", file=sys.stderr)
        return exitcodes.REFUSED
    return exitcodes.OK
