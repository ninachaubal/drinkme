"""Fixture records in the shape `drinkme bench` writes and `publish` puts on a PDS for anyone to read."""
import copy

from aggregator.pipeline import CURRENT_VERSION, NSID

DID_A = "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa"
DID_B = "did:web:bench.example.com"
STRIX = "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"


def uri(did=DID_A, rkey="3aaa"):
    return f"at://{did}/{NSID}/{rkey}"


def value(comp="16.05", stock="10.0", read="239.3", comp_gb="12.096", stock_gb="16.384", outcome="measured",
          profile="sip", platform="rocm", device=STRIX, name="Qwen3-8B", mem=133680857088,
          created="2026-09-22T06:00:00+00:00", verified=253, swapped=253, verification="pack",
          twin=None, twin_gb="16.417", decode_read=None):
    """`decode_read` maps an arm to its `<arm>_decode_read_gb`; None (the
    default) writes each arm's weights figure there."""
    if decode_read is None:
        decode_read = {"compressed": comp_gb, "stock": stock_gb, "twin": twin_gb}

    def read_gb(arm):
        return ([{"name": f"{arm}_decode_read_gb", "value": decode_read[arm], "unit": "GB"}]
                if arm in decode_read else [])

    metrics = [{"name": "compressed_decode_tok_s", "value": comp, "unit": "tok/s", "samples": [comp, comp, comp]},
               {"name": "read_gb_s", "value": read, "unit": "GB/s"},
               {"name": "compressed_weights_gb", "value": comp_gb, "unit": "GB"}, *read_gb("compressed")]
    if outcome == "measured":
        metrics += [{"name": "stock_decode_tok_s", "value": stock, "unit": "tok/s"},
                    {"name": "stock_weights_gb", "value": stock_gb, "unit": "GB"}, *read_gb("stock")]
    if twin is not None:
        metrics += [{"name": "twin_decode_tok_s", "value": twin, "unit": "tok/s"},
                    {"name": "twin_weights_gb", "value": twin_gb, "unit": "GB"}, *read_gb("twin")]
    raw = {"verified_tensors": verified, "swapped_linears": swapped, "verification": verification,
           "stock_error": None if outcome == "measured" else "skipped by bench's stock-arm fit check",
           "twin_outcome": "measured" if twin is not None else "skipped_predicted_nonfit"}
    return {"$type": NSID, "createdAt": created, "version": CURRENT_VERSION,
            "model": {"name": name, "hfRepo": f"Qwen/{name}", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
            "compression": {"profile": profile, "bitsPerWeight": "11.374"},
            "environment": {"deviceClass": device, "platform": platform, "memoryBytes": mem, "memoryKind": "unified"},
            "stock": {"outcome": "measured"} if outcome == "measured"
            else {"outcome": outcome, "budgetBytes": 17163091968},
            "metrics": metrics, "raw": raw}


def rec(did=DID_A, rkey="3aaa", **kw):
    return {"uri": uri(did, rkey), "cid": "bafy" + rkey, "value": value(**kw)}


def fitpoint(**kw):
    """A record whose stock arm the fit check skipped: compressed numbers only."""
    return rec(outcome="skipped_predicted_nonfit", **kw)


def edit(r, fn):
    r = copy.deepcopy(r)
    fn(r["value"])
    return r
