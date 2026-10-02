"""GET /metrics — exposition, counter movement, histogram cumulative-
ness, and reachability while a generation holds server.gen_lock.

Everything here goes through FakeEngine + real sockets, same posture as
test_serving_http.py — no toy transformers model, so this file is fast.
`metrics` is a module-level registry (one process, one server); `_isolated`
resets it before every test so absolute-value assertions do not depend on
test order or on what other tests in the same session did.
"""

import http.client
import json
import re
import socket
import threading
import time

import pytest

from drinkme.serving import metrics
from test_serving_http import fake, get, msgs, post  # noqa: F401 — shared fixtures/helpers


@pytest.fixture(autouse=True)
def _isolated():
    metrics.reset()
    yield
    metrics.reset()


# ---------------------------------------------------------------- exposition --


def _parse_exposition(text: str) -> dict:
    """A ~20-line Prometheus text-exposition parser: enough to check the
    shape, not a spec-complete implementation. Returns
    {metric_name: {labels_tuple: value_str}}, HELP/TYPE lines dropped.
    labels_tuple is a sorted tuple of (name, value) pairs, () when unlabeled
    — hashable, so callers can index straight into the dict."""
    out: dict[str, dict] = {}
    line_re = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$')
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        assert m, f"line does not parse as exposition: {line!r}"
        name, labelblob, value = m.groups()
        labels = []
        if labelblob:
            for pair in labelblob[1:-1].split(","):
                k, v = pair.split("=", 1)
                labels.append((k, v.strip('"')))
        out.setdefault(name, {})[tuple(sorted(labels))] = value
    return out


def test_exposition_parses_and_has_help_and_type_per_metric(fake):
    _, port = fake()
    post(port, {"messages": msgs()})
    r, body = get(port, "/metrics")
    text = body.decode()
    assert r.status == 200
    assert r.getheader("Content-Type").startswith("text/plain")
    parsed = _parse_exposition(text)
    assert "drinkme_requests_total" in parsed
    assert "drinkme_ttft_seconds_bucket" in parsed
    assert "drinkme_ttft_seconds_sum" in parsed
    assert "drinkme_ttft_seconds_count" in parsed
    # every metric FAMILY (the base name a # TYPE line declares — histograms
    # then emit _bucket/_sum/_count samples under that one family, per spec)
    # is preceded immediately by its own # HELP line
    lines = text.splitlines()
    type_re = re.compile(r"^# TYPE (\S+) (counter|gauge|histogram)$")
    families = [(i, m.group(1)) for i, line in enumerate(lines) if (m := type_re.match(line))]
    assert families  # the file actually declared some metrics
    for i, name in families:
        assert lines[i - 1].startswith(f"# HELP {name} ")


def test_metrics_route_is_lockless_and_ungated(fake):
    _, port = fake(auth="secret")
    r, _ = get(port, "/metrics")
    assert r.status == 200  # no Authorization header sent, unlike /v1/models


# ------------------------------------------------------------- counters move --


def test_requests_total_counts_by_route_and_status(fake):
    _, port = fake()
    get(port, "/health")
    get(port, "/health")
    get(port, "/nope")
    counts = _parse_exposition(get(port, "/metrics")[1].decode())["drinkme_requests_total"]
    assert counts[(("route", "/health"), ("status", "200"))] == "2"
    assert counts[(("route", "other"), ("status", "404"))] == "1"


def test_generated_tokens_and_ttft_move_after_a_request(fake):
    _, port = fake(reply="one two three")
    before = _parse_exposition(get(port, "/metrics")[1].decode())
    post(port, {"messages": msgs()})
    after = _parse_exposition(get(port, "/metrics")[1].decode())
    b = before["drinkme_generated_tokens_total"][()]
    a = after["drinkme_generated_tokens_total"][()]
    assert float(a) - float(b) == 3  # "one", "two", "three" — one word, one token
    assert after["drinkme_ttft_seconds_count"][()] == "1"


def test_inflight_gauge_rises_during_a_generation_and_falls_after(fake):
    eng, port = fake(reply="w " * 30, delay=0.02)  # ~0.6s
    t = threading.Thread(target=lambda: post(port, {"messages": msgs()}))
    t.start()
    time.sleep(0.1)
    mid = _parse_exposition(get(port, "/metrics")[1].decode())
    assert mid["drinkme_inflight_requests"][()] == "2"  # the POST + this GET
    t.join()
    time.sleep(0.05)
    after = _parse_exposition(get(port, "/metrics")[1].decode())
    assert after["drinkme_inflight_requests"][()] == "1"  # only this GET


def test_metrics_reachable_while_a_generation_holds_the_lock(fake):
    """The gate: a slow generation holds server.gen_lock; GET /metrics must
    answer promptly anyway (same shape as test_models_bypasses_generation_lock
    in test_serving_http.py)."""
    eng, port = fake(reply="w " * 60, delay=0.02)  # ~1.2s generation
    t = threading.Thread(target=lambda: post(port, {"messages": msgs()}))
    t.start()
    time.sleep(0.2)  # let the generation take the lock
    t0 = time.time()
    r, _ = get(port, "/metrics")
    assert r.status == 200 and time.time() - t0 < 1.0
    t.join()


def test_aborted_streams_total_counts_a_disconnect(fake):
    eng, port = fake(reply="w " * 500, delay=0.01)
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = json.dumps({"messages": msgs(), "stream": True}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.recv(1024)
    s.close()
    t0 = time.time()
    while time.time() - t0 < 5 and not eng.aborted:
        time.sleep(0.02)
    assert eng.aborted
    parsed = _parse_exposition(get(port, "/metrics")[1].decode())
    assert parsed["drinkme_aborted_streams_total"][()] == "1"


def test_4xx_total_buckets_by_status_and_reason(fake):
    _, port = fake()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("POST", "/v1/chat/completions", "not json",
              {"Content-Type": "application/json"})
    c.getresponse().read()
    c.close()
    counts = _parse_exposition(get(port, "/metrics")[1].decode())["drinkme_4xx_total"]
    assert counts[(("reason", "invalid_request_error"), ("status", "400"))] == "1"


# --------------------------------------------------------------- histograms --


def test_histogram_buckets_are_cumulative():
    h = metrics.Histogram((1.0, 2.0, 5.0))
    for v in (0.5, 1.5, 1.5, 3.0, 100.0):
        h.observe(v)
    cells, sums, counts = h.snapshot()
    cell = cells[()]
    assert cell == [1, 2, 1]  # le=1: {0.5}; le=2: {1.5,1.5}; le=5: {3.0}; 100.0 overflows
    cumulative = []
    running = 0
    for c in cell:
        running += c
        cumulative.append(running)
    assert cumulative == [1, 3, 4]  # non-decreasing
    assert counts[()] == 5  # the +Inf bucket (total observations, overflow included)
    assert sums[()] == pytest.approx(0.5 + 1.5 + 1.5 + 3.0 + 100.0)


def test_rendered_histogram_bucket_lines_are_cumulative_and_le_inf_equals_count():
    metrics.observe_ttft(0.02)
    metrics.observe_ttft(0.3)
    metrics.observe_ttft(3.0)
    parsed = _parse_exposition(metrics.render())
    rows = parsed["drinkme_ttft_seconds_bucket"]  # {(("le", "0.05"),): "1", ...}

    def le_of(labels_tuple):
        v = dict(labels_tuple)["le"]
        return float("inf") if v == "+Inf" else float(v)

    ordered = sorted(rows.items(), key=lambda kv: le_of(kv[0]))
    values = [int(v) for _, v in ordered]
    assert values == sorted(values)  # cumulative -> non-decreasing
    assert values[-1] == 3  # le="+Inf"
    assert parsed["drinkme_ttft_seconds_count"][()] == "3"
