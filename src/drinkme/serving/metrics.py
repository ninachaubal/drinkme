"""In-process metrics registry, rendered as Prometheus text exposition — no
client library: this file plus the ~40-line render() at the bottom
handle Counter/Gauge/Histogram and the `GET /metrics` body in full.

Zero cost when nobody scrapes: every call from the hot paths below is one
dict increment under one small lock (never `server.gen_lock` — this module
imports nothing from http.py/engines.py/mtp.py, only the other direction),
and `render()`, the only place that does string work, runs exclusively on a
GET /metrics.

Names are `drinkme_`-prefixed and mirror vLLM's (as listed in vLLM's docs/design/metrics.md)
where the concept matches:
  drinkme_requests_total{route,status}      ~ vllm:request_success_total
  drinkme_inflight_requests                 ~ vllm:num_requests_running
  drinkme_ttft_seconds                      ~ vllm:time_to_first_token_seconds
  drinkme_inter_token_seconds               ~ vllm:time_per_output_token_seconds
  drinkme_generated_tokens_total            ~ vllm:generation_tokens_total
  drinkme_spec_proposed_tokens_total        ~ vllm:spec_decode_num_draft_tokens_total
  drinkme_spec_accepted_tokens_total        ~ vllm:spec_decode_num_accepted_tokens_total
drinkme-specific, no vLLM analog (this server's own shape — N whole-cache
prefix SLOTS, not vLLM's block allocator, and the abort/4xx-reason lines
an operator's post-mortem needs):
  drinkme_prefix_slot_events_total{event=hit|restore|miss|evict}
  drinkme_prefix_slot_occupancy
  drinkme_aborted_streams_total
  drinkme_4xx_total{status,reason}
  drinkme_mtp_bail_trips_total   the adaptive bail (serving/mtp.py) handing a
  drinkme_mtp_rearms_total       request to the serial loop, and drafting resuming

TTFT here is the SAME definition bench/serve_timing_ab.py uses client-side
(request() there): elapsed time to the first delta carrying reasoning OR
content, from just before the generation core takes the lock — so a queued
request's wait time is IN the number, same as it is in the bench's t0.
"""

from __future__ import annotations

import threading
from collections import defaultdict

# Label values are tuples of (name, value) pairs, sorted, so two calls that
# mean the same labels always hash to the same key regardless of call order.
Labels = tuple


def _key(labels: dict[str, str] | None) -> Labels:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


class Counter:
    """Monotonic, per label combination. Unlabeled = key ()."""

    def __init__(self):
        self._vals: dict[Labels, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, labels: dict[str, str] | None = None, n: float = 1) -> None:
        with self._lock:
            self._vals[_key(labels)] += n

    def snapshot(self) -> dict[Labels, float]:
        with self._lock:
            return dict(self._vals)


class Gauge:
    """Set/inc/dec, per label combination. Unlabeled = key ()."""

    def __init__(self):
        self._vals: dict[Labels, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, labels: dict[str, str] | None = None, n: float = 1) -> None:
        with self._lock:
            self._vals[_key(labels)] += n

    def dec(self, labels: dict[str, str] | None = None, n: float = 1) -> None:
        self.inc(labels, -n)

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._vals[_key(labels)] = value

    def snapshot(self) -> dict[Labels, float]:
        with self._lock:
            return dict(self._vals)


class Histogram:
    """Fixed buckets (upper bounds, seconds), one cell array per label
    combination. observe() is O(buckets) under the lock — buckets here top
    out at a couple dozen, so that is microseconds, not a hot-loop cost.

    Storage is non-cumulative (each observation lands in exactly one cell,
    the smallest bucket it fits under); render() below does the prefix-sum
    that turns that into the exposition format's cumulative `_bucket{le=}`
    series, per the spec — cumulative is a rendering concern, not a storage
    one."""

    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = tuple(sorted(buckets))
        self._cells: dict[Labels, list[int]] = {}
        self._sum: dict[Labels, float] = defaultdict(float)
        self._count: dict[Labels, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _key(labels)
        with self._lock:
            cell = self._cells.setdefault(key, [0] * len(self.buckets))
            for i, b in enumerate(self.buckets):
                if value <= b:
                    cell[i] += 1
                    break
            self._sum[key] += value
            self._count[key] += 1

    def snapshot(self) -> tuple[dict[Labels, list[int]], dict[Labels, float], dict[Labels, int]]:
        with self._lock:
            return dict(self._cells), dict(self._sum), dict(self._count)


# ------------------------------------------------------------- the metrics --
# vLLM's default histogram families span sub-10ms to multi-minute; ours is
# the same shape, narrowed to this box's actual regime (a cold 27B prefill
# on a long prompt is tens of seconds, not the multi-GPU-cluster minutes
# vLLM's own buckets reach for).
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
INTER_TOKEN_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

REQUESTS = Counter()                 # route, status
INFLIGHT = Gauge()                   # unlabeled
SLOT_EVENTS = Counter()              # event = hit | restore | miss | evict
SLOT_OCCUPANCY = Gauge()             # unlabeled: whole slots currently filled
TTFT = Histogram(TTFT_BUCKETS)       # unlabeled, seconds
INTER_TOKEN = Histogram(INTER_TOKEN_BUCKETS)  # unlabeled, seconds
GENERATED_TOKENS = Counter()         # unlabeled
SPEC_PROPOSED = Counter()             # unlabeled
SPEC_ACCEPTED = Counter()             # unlabeled
MTP_BAIL_TRIPS = Counter()           # unlabeled: adaptive bail -> serial
MTP_REARMS = Counter()               # unlabeled: serial -> drafting again
ABORTED_STREAMS = Counter()          # unlabeled (logging aborted streams distinctly)
ERRORS_4XX = Counter()               # status, reason (4xx-reason logging)

# Metrics a fresh process should still show at 0 on the first scrape — the
# unlabeled server-lifetime counters/gauges, the ones a dashboard would plot
# a line for from t=0. Labeled metrics (REQUESTS, SLOT_EVENTS, ERRORS_4XX)
# only grow the label combinations actually seen, standard Prometheus
# practice — pre-declaring every {event=...} triple nobody has hit yet would
# be the opposite of "zero cost when nobody scrapes".
_ALWAYS_VISIBLE = (
    ("drinkme_inflight_requests", "Requests currently being handled.", "gauge", INFLIGHT),
    ("drinkme_prefix_slot_occupancy", "Whole prefix-cache slots currently filled.",
     "gauge", SLOT_OCCUPANCY),
    ("drinkme_generated_tokens_total", "Tokens delivered to a client, all routes.",
     "counter", GENERATED_TOKENS),
    ("drinkme_spec_proposed_tokens_total", "Draft tokens proposed, by either proposer (MTP or n-gram).",
     "counter", SPEC_PROPOSED),
    ("drinkme_spec_accepted_tokens_total", "Draft tokens the verify pass accepted, by either proposer.",
     "counter", SPEC_ACCEPTED),
    ("drinkme_mtp_bail_trips_total",
     "Adaptive-bail trips: a request handed to the serial loop on low rolling acceptance.",
     "counter", MTP_BAIL_TRIPS),
    ("drinkme_mtp_rearms_total",
     "Adaptive-bail re-arms: a tripped request resuming speculation on a fresh window.",
     "counter", MTP_REARMS),
    ("drinkme_aborted_streams_total", "Streams that ended on client disconnect.",
     "counter", ABORTED_STREAMS),
)


def record_slot_event(event: str) -> None:
    """event in {"hit", "restore", "miss", "evict"} — the outcomes of
    engines.py's pick_slot call site: `hit` = the prompt EXTENDED an existing
    slot (the code's own word for it — engines.py's module docstring and
    pick_slot's), `restore` = the prompt parted from a slot and was served up
    to a shared prefix by a rewind or a context checkpoint
    (serving/ctx_checkpoints.py), `miss` = an empty (never-used) slot was
    filled cold, `evict` = a slot that already held a different conversation
    was reset cold."""
    SLOT_EVENTS.inc({"event": event})


def slot_occupied() -> None:
    """One more whole slot went from empty to filled — paired 1:1 with a
    "miss" event, since that is the only transition (an "evict" reuses a
    slot that was already counted here)."""
    SLOT_OCCUPANCY.inc()


def reset_slot_occupancy() -> None:
    """engines.HFEngine.reset_prefix_cache() clears every slot back to
    empty — occupancy follows."""
    SLOT_OCCUPANCY.set(0)


def observe_ttft(seconds: float) -> None:
    TTFT.observe(seconds)


def observe_inter_token(seconds: float) -> None:
    INTER_TOKEN.observe(seconds)


def add_generated_tokens(n: int) -> None:
    if n:
        GENERATED_TOKENS.inc(n=n)


def record_spec(proposed: int, accepted: int) -> None:
    if proposed:
        SPEC_PROPOSED.inc(n=proposed)
    if accepted:
        SPEC_ACCEPTED.inc(n=accepted)


def record_mtp_trip() -> None:
    MTP_BAIL_TRIPS.inc()


def record_mtp_rearm() -> None:
    MTP_REARMS.inc()


def inc_aborted() -> None:
    ABORTED_STREAMS.inc()


def inc_4xx(status: int, reason: str) -> None:
    ERRORS_4XX.inc({"status": str(status), "reason": reason})


def inc_request(route: str, status: int) -> None:
    REQUESTS.inc({"route": route, "status": str(status)})


def inflight_inc() -> None:
    INFLIGHT.inc()


def inflight_dec() -> None:
    INFLIGHT.dec()


def reset() -> None:
    """Test-only: every counter/gauge/histogram back to empty. The registry
    is module-level (one process, one server), so isolated assertions on
    absolute values need this between tests rather than a fresh import."""
    for m in (REQUESTS, SLOT_EVENTS, GENERATED_TOKENS, SPEC_PROPOSED,
             SPEC_ACCEPTED, MTP_BAIL_TRIPS, MTP_REARMS, ABORTED_STREAMS, ERRORS_4XX):
        m._vals.clear()
    for g in (INFLIGHT, SLOT_OCCUPANCY):
        g._vals.clear()
    for h in (TTFT, INTER_TOKEN):
        h._cells.clear()
        h._sum.clear()
        h._count.clear()


# ------------------------------------------------------------- exposition --


def _fmt_labels(pairs: Labels) -> str:
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in pairs) + "}"


def _fmt_bucket(b: float) -> str:
    return f"{b:g}"


def _render_counter(lines: list[str], name: str, help_text: str, c: Counter) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    vals = c.snapshot()
    if not vals:
        lines.append(f"{name} 0")
        return
    for labels, v in sorted(vals.items()):
        lines.append(f"{name}{_fmt_labels(labels)} {v:g}")


def _render_gauge(lines: list[str], name: str, help_text: str, g: Gauge) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    vals = g.snapshot()
    if not vals:
        lines.append(f"{name} 0")
        return
    for labels, v in sorted(vals.items()):
        lines.append(f"{name}{_fmt_labels(labels)} {v:g}")


def _render_histogram(lines: list[str], name: str, help_text: str, h: Histogram) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} histogram")
    cells, sums, counts = h.snapshot()
    for key in cells or {(): [0] * len(h.buckets)}:
        cell = cells.get(key, [0] * len(h.buckets))
        base = _fmt_labels(key)
        cumulative = 0
        for b, c in zip(h.buckets, cell):
            cumulative += c
            lbl = _fmt_labels(key + (("le", _fmt_bucket(b)),))
            lines.append(f"{name}_bucket{lbl} {cumulative}")
        total = counts.get(key, 0)
        inf_lbl = _fmt_labels(key + (("le", "+Inf"),))
        lines.append(f"{name}_bucket{inf_lbl} {total}")
        lines.append(f"{name}_sum{base} {sums.get(key, 0.0):.6f}")
        lines.append(f"{name}_count{base} {total}")


def render() -> str:
    """The full GET /metrics body. Called with no lock held — every metric
    object owns its own tiny lock, and a scrape reads a snapshot of each in
    turn rather than a single consistent point-in-time image, which is the
    standard Prometheus client contract (not a transaction)."""
    lines: list[str] = []
    _render_counter(lines, "drinkme_requests_total",
                    "HTTP requests by route and response status.", REQUESTS)
    _render_counter(lines, "drinkme_prefix_slot_events_total",
                    "Prefix-cache slot picks by outcome (hit/restore/miss/evict).", SLOT_EVENTS)
    _render_counter(lines, "drinkme_4xx_total",
                    "4xx responses by status and reason code.", ERRORS_4XX)
    _render_histogram(lines, "drinkme_ttft_seconds",
                      "Time to first token, server-observed (queue + prefill), seconds.",
                      TTFT)
    _render_histogram(lines, "drinkme_inter_token_seconds",
                      "Gap between successive emitted deltas, seconds.", INTER_TOKEN)
    for name, help_text, kind, metric in _ALWAYS_VISIBLE:
        if kind == "gauge":
            _render_gauge(lines, name, help_text, metric)
        else:
            _render_counter(lines, name, help_text, metric)
    return "\n".join(lines) + "\n"
