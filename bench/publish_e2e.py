#!/usr/bin/env python3
"""drinkme publish, end to end, against a real PDS.

Runs the SHIPPED verb (`drinkme publish FILE --handle ... --plc ...
--no-browser`) as a subprocess with an isolated config dir, feeds the
authorization URL it prints to bench/publish_consent.cjs (playwright: the
PDS's own login page + the consent screen + the loopback redirect), waits
for the CLI to finish its createRecord -> getRecord read-back, then from
the outside reads the record back once more, compares it byte-equal to the
fixture, and DELETES it so the test account's repo is left as found
(`--keep` to leave it).

The fixture is a lexicon-shaped record built here (a real bench needs the
GPU); the account password is read at runtime from an env file and handed
to the browser driver in its environment — it is never printed, never an
argument, never on the branch.

    PYTHONPATH=src .venv/bin/python bench/publish_e2e.py \\
        --handle test.example.com \\
        --plc https://plc.example.com \\
        --password-env PDS_PASSWORD --env-file path/to/test-account.env

Exit 0 only when every step passed; the verdict is in the output, never
trust the code alone from a GPU box (AGENTS.md).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from drinkme import MEASUREMENT, __version__  # noqa: E402
from drinkme.publish import oauth, pds, store  # noqa: E402
from drinkme.publish.jwt import Es256Key  # noqa: E402

import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402


def fixture_record(tag: str) -> dict:
    """One record shaped like `drinkme bench` writes it (through
    bench.lexicon_safe: floats as decimal strings, no nulls), with `tag` in
    the raw block so the read-back is unmistakably ours."""
    return {
        "$type": MEASUREMENT,
        "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": __version__,
        "environment": {
            "deviceClass": "e2e fixture device", "memoryBytes": 128_000_000_000, "memoryKind": "unified",
            "cpuInfo": "fixture cpu", "gpuInfo": "fixture gpu", "os": "Linux fixture",
            "engine": f"drinkme {__version__} / torch fixture", "platform": "rocm",
        },
        "model": {"name": "Qwen3-8B", "hfRepo": "Qwen/Qwen3-8B",
                  "revision": "0000000000000000000000000000000000000000"},
        "compression": {"profile": "sip", "bitsPerWeight": "11.374"},
        "stock": {"outcome": "measured"},
        "metrics": [
            {"name": "stock_decode_tok_s", "value": "6.92", "unit": "tok/s",
             "samples": ["6.90", "6.92", "6.95"]},
            {"name": "compressed_decode_tok_s", "value": "14.46", "unit": "tok/s",
             "samples": ["14.4", "14.46", "14.5"]},
            {"name": "read_gb_s", "value": "222.4", "unit": "GB/s"},
            {"name": "copy_gb_s", "value": "192.9", "unit": "GB/s"},
            {"name": "stock_weights_gb", "value": "15.26", "unit": "GB"},
            {"name": "compressed_weights_gb", "value": "11.81", "unit": "GB"},
        ],
        "raw": {"e2e_tag": tag, "mean_bpw": "12.031", "all_tensors_bitwise_roundtrip": True,
                "detector": {"budgetBytes": 124_000_000_000, "heuristic": True,
                             "evidence": ["publish_e2e fixture"], "version": 2},
                "unicode": "ünïcödé — kept verbatim", "n": 3},
    }


class LegacySession:
    """A password (app-password level) session on the PDS, used ONLY for
    the cleanup: the OAuth session drinkme publish holds is granted
    `repo:<collection>?action=create` and nothing else — measured, a
    self-hosted PDS answers deleteRecord with 403 insufficient_scope
    "Missing required scope repo:...?action=delete", which is the granular
    scope doing exactly its job. So the driver deletes with the credential
    it already has for the login form, and closes that session after."""

    def __init__(self, pds_url: str, handle: str, password: str):
        self.pds = pds_url.rstrip("/")
        body = self._call("com.atproto.server.createSession",
                          {"identifier": handle, "password": password})
        self.token, self.did = body["accessJwt"], body["did"]

    def _call(self, nsid: str, body: dict | None = None, params: dict | None = None) -> dict:
        url = f"{self.pds}/xrpc/{nsid}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        headers = {"content-type": "application/json"}
        if getattr(self, "token", None):
            headers["authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{nsid} answered {e.code} {e.read()[:200]!r}") from None

    def sweep(self, collection: str, keep: bool) -> list[str]:
        """Every record in the collection this driver ever wrote (raw.e2e_tag
        starts with 'publish_e2e'); deleted unless keep."""
        found = []
        cursor = None
        while True:
            params = {"repo": self.did, "collection": collection, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            page = self._call("com.atproto.repo.listRecords", params=params)
            for rec in page.get("records", []):
                tag = str((rec.get("value", {}).get("raw") or {}).get("e2e_tag", ""))
                if tag.startswith("publish_e2e"):
                    found.append(rec["uri"])
            cursor = page.get("cursor")
            if not cursor or not page.get("records"):
                break
        for uri in found:
            if not keep:
                _, coll, rkey = pds.parse_at_uri(uri)
                self._call("com.atproto.repo.deleteRecord",
                           {"repo": self.did, "collection": coll, "rkey": rkey})
        return found

    def close(self) -> None:
        try:
            self._call("com.atproto.server.deleteSession", {})
        except RuntimeError:
            pass


def read_password(env_file: str, var: str) -> str:
    with open(env_file) as f:
        for line in f:
            if line.startswith(f"{var}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(f"{var} not found in {env_file}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handle", required=True, help="the test account's handle")
    ap.add_argument("--plc", required=True,
                    help="PLC directory for the DID ('' = let publish default to plc.directory)")
    ap.add_argument("--pds", default=None,
                    help="expected PDS origin; the resolved one must match (default: none)")
    ap.add_argument("--password-env", required=True,
                    help="name of the variable in --env-file that holds the account password")
    ap.add_argument("--env-file", required=True,
                    help="env file holding the test account's password")
    ap.add_argument("--browser-dir", default=os.environ.get("DRINKME_E2E_BROWSER_DIR"),
                    help="where playwright + chromium live (node_modules under it); "
                         "default: $DRINKME_E2E_BROWSER_DIR, required if unset")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep", action="store_true", help="leave the record on the PDS")
    ap.add_argument("--shot", default=None, help="screenshot the consent screen here")
    ap.add_argument("--twice", action="store_true",
                    help="publish a second record with the stored session (no browser), "
                         "its access token first marked expired so the run has to REFRESH "
                         "it against the real authorization server; also deleted")
    a = ap.parse_args()
    if not a.browser_dir:
        ap.error("--browser-dir is required (or set DRINKME_E2E_BROWSER_DIR)")

    password = read_password(a.env_file, a.password_env)
    tag = f"publish_e2e {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
    record = fixture_record(tag)

    ok = True
    created: list[str] = []
    with tempfile.TemporaryDirectory(prefix="drinkme-publish-e2e-") as tmp:
        cfg = os.path.join(tmp, "config")
        rec_path = os.path.join(tmp, "fixture.json")
        with open(rec_path, "w") as f:
            json.dump(record, f, indent=2)
        env = {**os.environ, "DRINKME_CONFIG_DIR": cfg, "PYTHONPATH": os.path.join(ROOT, "src"),
               "PYTHONUNBUFFERED": "1"}
        cmd = [a.python, "-m", "drinkme.cli", "publish", rec_path, "--handle", a.handle,
               "--no-browser"]
        if a.plc:
            cmd += ["--plc", a.plc]

        def run_cli(drive_browser: bool) -> tuple[int, str]:
            print(f"$ {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True)
            out, node = [], None
            for line in proc.stdout:
                print(f"  cli │ {line}", end="")
                out.append(line)
                if drive_browser and node is None and line.strip().startswith("https://") \
                        and "request_uri=" in line:
                    node = subprocess.Popen(
                        ["node", os.path.join(HERE, "publish_consent.cjs"), line.strip(),
                         "--handle", a.handle] + (["--shot", a.shot] if a.shot else []),
                        env={**os.environ, "PUBLISH_PASSWORD": password,
                             "NODE_PATH": os.path.join(a.browser_dir, "node_modules")},
                        cwd=a.browser_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True)
            rc = proc.wait()
            if node is not None:
                nout = node.communicate(timeout=60)[0]
                for l in nout.splitlines():
                    print(f"  browser │ {l}")
                if node.returncode != 0:
                    print("!! consent driver failed")
            return rc, "".join(out)

        rc, out = run_cli(drive_browser=True)
        uris = re.findall(r"at://\S+", out)
        print(f"-- cli exit {rc}; uris seen: {uris}")
        if rc != 0 or not uris:
            print("!! FAIL: publish did not complete")
            ok = False
        else:
            created.append(uris[-1])
            if "✓ read back byte-equal" not in out:
                print("!! FAIL: the CLI did not report the byte-equal read-back")
                ok = False
            if "scope     atproto " in out:
                granted = re.search(r"granted: (.*)\)", out)
                print(f"-- scope granted by the AS: {granted.group(1) if granted else '?'}")

        if ok and a.twice:
            rec2 = fixture_record(tag + " (second, stored session)")
            with open(rec_path, "w") as f:
                json.dump(rec2, f, indent=2)
            # Age the stored access token so the second run must refresh.
            spath = os.path.join(cfg, "session.json")
            with open(spath) as f:
                stored = json.load(f)
            stored["expires_at"] = 0
            with open(spath, "w") as f:
                json.dump(stored, f)
            rc2, out2 = run_cli(drive_browser=False)
            uris2 = re.findall(r"at://\S+", out2)
            if rc2 != 0 or not uris2 or "using the stored session" not in out2 \
                    or "access token expired; refreshing" not in out2:
                print("!! FAIL: second publish (stored session + refresh) did not complete")
                ok = False
            else:
                created.append(uris2[-1])
                record2 = rec2
                with open(spath) as f:
                    after = json.load(f)
                if after["access_token"] == stored["access_token"] \
                        or after["refresh_token"] == stored["refresh_token"]:
                    print("!! FAIL: the refresh did not rotate the token set")
                    ok = False
                else:
                    print("-- refresh rotated both the access and the refresh token")
        else:
            record2 = None

        # From the outside: the session the CLI stored, used to read the
        # record back independently and to delete it.
        sess = None
        try:
            os.environ["DRINKME_CONFIG_DIR"] = cfg
            sess = store.load_session()
        except PermissionError as e:
            print(f"!! FAIL: {e}")
            ok = False
        if sess and created:
            mode = oct(os.stat(store.session_path()).st_mode & 0o777)
            print(f"-- session file mode {mode}; scope {sess.get('scope')!r}; "
                  f"client_id {sess.get('client_id')}")
            if mode != "0o600":
                print("!! FAIL: session file is not 0600")
                ok = False
            client = oauth.DpopClient(Es256Key.from_jwk(sess["dpop_key"]), sess.get("nonces"))
            for uri, want in zip(created, [record, record2]):
                back = pds.get_record(client, sess, uri)
                same = pds.canonical(back["value"]) == pds.canonical(want)
                print(f"-- independent getRecord {uri}: byte-equal={same}, "
                      f"cid {back.get('cid')}")
                if not same:
                    ok = False
                    print("!! FAIL: read-back differs from the fixture")
            # The create-only scope must NOT be able to delete: that refusal
            # is the granular scope working, and part of the acceptance.
            if "transition:generic" not in (sess.get("scope") or ""):
                try:
                    pds.delete_record(client, sess, created[0])
                    ok = False
                    print("!! FAIL: the create-only OAuth session was able to deleteRecord")
                except oauth.OAuthError as e:
                    print(f"-- deleteRecord with the create-only session refused as expected: {e}")
            if a.pds and sess.get("pds", "").rstrip("/") != a.pds.rstrip("/"):
                ok = False
                print(f"!! FAIL: resolved PDS {sess.get('pds')} != --pds {a.pds}")
            pds_url = sess.get("pds")
        elif created:
            ok = False
            print("!! FAIL: no stored session to verify with")
            pds_url = None
        else:
            pds_url = None

        # Cleanup with the password session (see LegacySession): everything
        # this driver ever wrote, so a failed earlier run is swept too.
        if pds_url or a.pds:
            legacy = LegacySession(pds_url or a.pds, a.handle, password)
            try:
                swept = legacy.sweep(MEASUREMENT, keep=a.keep)
                verb = "left in place" if a.keep else "deleted"
                print(f"-- sweep: {len(swept)} publish_e2e record(s) {verb}: {swept}")
                if not a.keep:
                    for uri in created:
                        try:
                            pds.get_record(client, sess, uri)
                            ok = False
                            print(f"!! FAIL: {uri} still readable after delete")
                        except oauth.OAuthError as e:
                            print(f"-- getRecord after delete: {e}")
                    if any(u not in swept for u in created):
                        ok = False
                        print("!! FAIL: a record created this run was not found by the sweep")
            finally:
                legacy.close()

    print("PUBLISH E2E:", "PASS" if ok else "FAIL",
          f"— {len(created)} record(s) created" + ("" if a.keep else " and deleted"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
