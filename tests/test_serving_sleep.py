"""Sleep/wake — CPU only, no downloads, no GPU.

Five sections, because the feature has five separable claims.

  The pass (serving/sleep.py) is an object walk, so it runs against hand-built
  containers and against the real compressed toy: what gets found (parameters,
  buffers, the codec's `p` dict), what gets found ONCE (a tied weight), and
  what deliberately does not get found (a tensor that was never on the device
  — CompressedLinear's `_w_cpu`, modelled here with a meta tensor because a
  CPU box cannot make a cuda one).

  The engine is HFEngine.sleep/wake. On a CPU test the device IS cpu, so
  "moved off the device" is not observable as memory and is not asserted as
  memory: what is asserted is that every tensor the pass claims to have
  touched is a DIFFERENT OBJECT afterwards (the pass really ran, over
  everything), that the bookkeeping adds up to the real byte count, and that
  the model computes BIT-IDENTICAL logits after a wake. The memory claim is
  the GPU gate.

  The slots are the on-disk prefix slots' cold tier under a sleep: persisted on the way down
  through the same path SIGTERM takes, restored on the way up, counted both
  times — and dropped, loudly, when there is no cold tier to persist into.

  The routes are the HTTP contract over real sockets against FakeEngine:
  /health carries the state, both generate dialects answer 503 while asleep,
  a generation in flight makes /sleep a 409, the lockless routes keep
  answering, and both routes are idempotent (a systemd drop-in that must run
  before every unit start cannot also be a thing that fails the second time).

  The timer and the env reader are DRINKME_SLEEP_ON_IDLE_S, which is off by
  default and has to stay that way.

The toys are test_serving_prefix_slots.py's and test_serving_engines.py's —
the same models, the same tokenizer, the same real toy PACK — because sleep/wake is
"the prefix cache and on-disk prefix slots and the codec, parked", and a second set of toys would let them
drift.
"""

import http.client
import json
import threading
import time

import pytest
import torch

from drinkme import serve
from drinkme.serving import engines, sleep as sleep_mod
from drinkme.serving.engine import FakeEngine, SleepError, SleepState
from drinkme.serving.engines import HFEngine, load_compressed
from drinkme.serving.http import _IdleSleeper, start_server
from test_serving_engines import toy  # noqa: F401 — the real toy pack
from test_serving_prefix_slots import engine, greedy, llama  # noqa: F401
from test_serving_slotstore import LONG, agent_turn, cold_engine  # noqa: F401


# ------------------------------------------------------------- the pass --


class _Packed(torch.nn.Module):
    """CompressedLinear's shape as the walk sees it: a `p` dict of device
    tensors that are NOT registered parameters, a plain-attribute bias, and a
    lazily-built CPU reference weight that must never be dragged onto the
    accelerator. Not the real class — codec/swap.py needs a real pack to
    build one, and what this section is testing is the WALK."""

    def __init__(self, cpu_ref=None):
        super().__init__()
        self.p = {"rx_data": torch.zeros(8, dtype=torch.int32),
                  "rx_palette": torch.zeros(4), "R": 8, "block_size": 1024}
        self.bias = torch.zeros(2)
        self._w_cpu = cpu_ref


def _names(refs):
    return [r.name for r in refs]


def _where(refs):
    """The STORAGE address behind each ref, which is what "did this tensor
    move" actually asks. Not id(): a Parameter's `.data` is a fresh detached
    view on every access, so identity would report a move that never
    happened. Callers keep the tensors themselves alive alongside this, so
    the allocator cannot hand a freed address back and fake a non-move."""
    return [t.data_ptr() for r in refs
            for t in (r.get() if isinstance(r.get(), list) else [r.get()])]


def test_the_walk_finds_parameters_buffers_and_the_codec_dict():
    m = torch.nn.Module()
    m.lin = torch.nn.Linear(4, 4)
    m.lin.register_buffer("scratch", torch.zeros(3))
    m.packed = _Packed()
    refs = sleep_mod.walk(m)
    got = set(_names(refs))
    assert {"lin.weight", "lin.bias", "lin.scratch",
            "packed.p.rx_data", "packed.p.rx_palette", "packed.bias"} <= got
    # the codec dict's scalars (R, C, block_size, the widths) are not tensors,
    # and the walk does not invent refs for them
    assert "packed.p.R" not in got and "packed.p.block_size" not in got


def test_the_walk_buckets_packed_planes_apart_from_raw_tensors():
    """The inventory the report is written from: a packed plane is not a raw
    tensor, and the MTP head is neither."""
    m = torch.nn.Module()
    m.lin = torch.nn.Linear(4, 4)
    m.packed = _Packed()
    by = {r.name: r.cls for r in sleep_mod.walk(m)}
    assert by["lin.weight"] == sleep_mod.RAW
    assert by["packed.p.rx_data"] == sleep_mod.PACKED
    assert by["packed.bias"] == sleep_mod.RAW  # the bias is bf16, not a plane


def test_a_tied_weight_is_moved_once_and_stays_tied():
    """lm_head sharing embed_tokens' Parameter is the real case: two modules,
    one Parameter object. Moving it twice would double the byte count and,
    worse, could untie it."""
    m = torch.nn.Module()
    m.embed = torch.nn.Linear(4, 4, bias=False)
    m.head = torch.nn.Linear(4, 4, bias=False)
    m.head.weight = m.embed.weight
    refs = sleep_mod.walk(m)
    assert _names(refs).count("embed.weight") == 1
    assert "head.weight" not in _names(refs)  # one Parameter, one ref
    assert sleep_mod.inventory(refs)[1] == 4 * 4 * 4  # counted ONCE, not twice
    sleep_mod.move(refs, "cpu")
    assert m.head.weight is m.embed.weight  # and it is still tied


def test_the_walk_skips_a_tensor_that_was_never_on_the_device():
    """CompressedLinear._w_cpu, modelled with a meta tensor (a CPU box cannot
    make a cuda one). Waking it would move a host-side reference weight ONTO
    the accelerator — memory nobody asked for."""
    m = torch.nn.Module()
    m.packed = _Packed(cpu_ref=torch.zeros(4, device="meta"))
    assert "packed._w_cpu" in _names(sleep_mod.walk(m))  # device=None: all of it
    assert "packed._w_cpu" not in _names(sleep_mod.walk(m, device="cpu"))
    # and the ones that ARE on the named device still come along
    assert "packed.p.rx_data" in _names(sleep_mod.walk(m, device="cpu"))


def test_a_tuple_of_tensors_is_rebuilt_around_the_moved_ones():
    """draft_vocab.SubsetProjection.esc is a tuple, which has no slot to
    assign into; the whole sequence is rebuilt."""
    holder = _Packed()
    holder.esc = (torch.zeros(2), torch.ones(3))
    refs = [r for r in sleep_mod.walk(holder) if r.name == "esc"]
    assert len(refs) == 1
    old = holder.esc
    sleep_mod.move(refs, "cpu")
    assert isinstance(holder.esc, tuple) and len(holder.esc) == 2
    assert holder.esc[0] is not old[0] and torch.equal(holder.esc[1], old[1])


def test_inventory_counts_bytes_without_moving_anything():
    m = torch.nn.Module()
    m.lin = torch.nn.Linear(4, 4)
    refs = sleep_mod.walk(m)
    before = [r.get() for r in refs]
    where = _where(refs)
    n, nbytes, by_class = sleep_mod.inventory(refs)
    assert n == len(refs)
    assert nbytes == sum(t.numel() * t.element_size() for t in before)
    assert by_class == {sleep_mod.RAW: nbytes}
    assert _where(refs) == where  # untouched: nothing was copied anywhere


def test_pin_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("DRINKME_SLEEP_PIN", raising=False)
    assert sleep_mod.pin_from_env() is False
    monkeypatch.setenv("DRINKME_SLEEP_PIN", "0")
    assert sleep_mod.pin_from_env() is False
    monkeypatch.setenv("DRINKME_SLEEP_PIN", "1")
    assert sleep_mod.pin_from_env() is True


# ------------------------------------------------------------ the engine --


def _logits(eng, ids=(3, 4, 5, 6)):
    inp = torch.tensor([list(ids)], dtype=torch.long, device=eng.device)
    with torch.inference_mode():
        return eng.model(inp, use_cache=False).logits.clone()


def test_sleep_touches_every_tensor_it_says_it_touched(llama):
    """The CPU stand-in for "the weights left the device": every tensor the
    pass claims is a DIFFERENT OBJECT afterwards, and the byte count is the
    real sum, not an estimate. Old tensors are held (not just their ids) so
    the identity check cannot be fooled by a reused address."""
    eng = engine(llama, ctx=256)
    refs = sleep_mod.walk(eng.model, eng.mtp_head, eng.device)
    before = [r.get() for r in refs]  # held, so no address can be recycled
    where = _where(refs)
    st = eng.sleep(1)
    assert st["state"] == "asleep" and st["level"] == 1
    assert st["tensors"] == len(refs)
    assert st["parked_bytes"] == sum(t.numel() * t.element_size() for t in before)
    assert all(a != b for a, b in zip(_where(refs), where))  # every one moved
    assert all(torch.equal(r.get(), old) for r, old in zip(refs, before))
    eng.wake()
    assert eng.sleep_state()["state"] == "awake"


def test_wake_gives_back_bit_identical_logits(llama):
    """The correctness claim, on a fixed prompt: a parked and resumed model
    computes the same numbers. Bit-identical on CPU, where the round trip is
    a memcpy; the GPU form of this gate is bench/sleep_wake_verify.py."""
    eng = engine(llama, ctx=256)
    before = _logits(eng)
    eng.sleep(1)
    eng.wake()
    assert torch.equal(before, _logits(eng))


def test_wake_gives_back_bit_identical_logits_through_the_codec(toy):
    """The same claim where it actually costs something: a REAL pack, whose
    weights live in CompressedLinear.p as packed planes plus escape, mode and
    CSR sidecars — none of them registered parameters, all of them moved by
    the vars() walk. If the pass missed one plane the decode would not
    reproduce, and torch.equal is what says so."""
    eng = load_compressed(toy[0], None, toy[1], device="cpu", ctx=256)
    before = _logits(eng)
    st = eng.sleep(1)
    assert st["by_class"][sleep_mod.PACKED] > 0  # the planes were counted
    assert st["by_class"][sleep_mod.RAW] > 0  # so were embeddings and norms
    eng.wake()
    assert torch.equal(before, _logits(eng))


def test_sleep_is_idempotent_and_only_ever_deepens(llama):
    eng = engine(llama, ctx=256)
    first = eng.sleep(1)
    again = eng.sleep(1)
    assert again["level"] == 1 and again["asleep_since"] == first["asleep_since"]
    eng.sleep(2)
    with pytest.raises(SleepError, match="only deepens"):
        eng.sleep(1)
    with pytest.raises(SleepError, match="not 1 or 2"):
        eng.sleep(3)


def test_level_two_frees_the_host_copies_and_wakes_through_the_loader(toy):
    """Level 2 IS the current stop, minus the stop: the model object is gone, and
    the wake comes back by replaying load_compressed — the same call serve.py
    booted through, not a second code path — and serves identical logits."""
    eng = load_compressed(toy[0], None, toy[1], device="cpu", ctx=256)
    assert eng._reload is not None
    before = _logits(eng)
    calls = []
    real = eng._reload
    eng._reload = lambda: (calls.append(1), real())[1]

    st = eng.sleep(2)
    assert st["level"] == 2 and eng.model is None and eng.mtp_head is None
    assert st["parked_bytes"] > 0  # it still says what it gave back

    st = eng.wake()
    assert st["state"] == "awake" and calls == [1]
    assert eng.model is not None
    assert torch.equal(before, _logits(eng))


def test_level_two_can_deepen_a_level_one_sleep(llama):
    eng = engine(llama, ctx=256)
    eng.sleep(1)
    st = eng.sleep(2)
    assert st["level"] == 2 and eng.model is None


def test_a_level_two_wake_without_a_loader_refuses_legibly(llama):
    """An engine built directly (every test above, and the benches) has no
    loader closure, so it cannot come back from level 2 — and says so instead
    of leaving a server that answers 200 with no model behind it."""
    eng = engine(llama, ctx=256)
    eng.sleep(2)
    with pytest.raises(SleepError, match="loader that built this engine"):
        eng.wake()


def test_wake_on_an_awake_engine_is_a_no_op(llama):
    eng = engine(llama, ctx=256)
    assert eng.wake() == {"state": "awake", "level": 0}


# ------------------------------------------------------------- the slots --


def test_sleep_persists_the_live_slots_and_wake_restores_them(llama, tmp_path,
                                                              monkeypatch):
    """The on-disk prefix slots' cold tier under a sleep. The slot is written down the SAME path
    SIGTERM takes, the KV is then genuinely dropped (vLLM's level 1), and the
    wake brings back exactly the slot this process parked — proved by the
    next turn reusing it instead of prefilling cold."""
    eng = cold_engine(llama, tmp_path, monkeypatch, ctx=1024)
    hist = []
    agent_turn(eng, hist, LONG)
    assert sum(1 for s in eng._slots if s.ids) == 1

    st = eng.sleep(1)
    assert st["slots_persisted"] == 1 and st["slots_dropped"] == 0
    assert all(s.cache is None and not s.ids for s in eng._slots)  # discarded

    st = eng.wake()
    assert st["slots_restored"] == 1
    assert sum(1 for s in eng._slots if s.ids) == 1
    assert agent_turn(eng, hist, "over the lazy dog").cached_tokens > 0


def test_with_no_cold_tier_the_slots_are_dropped_and_counted(llama):
    """DRINKME_SLOT_DIR=off (the suite default, and a real configuration):
    there is nowhere to persist to, so the slots go, and the count says so
    rather than leaving someone to infer it from a slow first request."""
    eng = engine(llama, ctx=512)
    hist = []
    agent_turn(eng, hist, LONG)
    assert eng._cold is None and sum(1 for s in eng._slots if s.ids) == 1
    st = eng.sleep(1)
    assert st["slots_persisted"] == 0 and st["slots_dropped"] == 1
    assert eng.wake()["slots_restored"] == 0


def test_a_slot_shorter_than_one_block_is_dropped_not_stored(llama, tmp_path,
                                                             monkeypatch):
    """The store refuses a prefix with no chain to be found by (on-disk
    prefix slots' own rule); sleep must count that as dropped, not silently as persisted."""
    eng = cold_engine(llama, tmp_path, monkeypatch, ctx=512)
    hist = []
    agent_turn(eng, hist, "hello world")  # far under BLOCK_TOKENS
    st = eng.sleep(1)
    assert st["slots_persisted"] == 0 and st["slots_dropped"] == 1


# ------------------------------------------------------------ the routes --


@pytest.fixture
def fake():
    """fake(**FakeEngine kwargs) -> (engine, port); servers shut down after.
    tests/test_serving_http.py's fixture, with sleep_on_idle reachable."""
    servers = []

    def make(engine=None, auth=None, sleep_on_idle=0.0, **kw):
        eng = engine or FakeEngine(**kw)
        srv = start_server(eng, "127.0.0.1", 0, auth_token=auth,
                           sleep_on_idle=sleep_on_idle)
        servers.append(srv)
        return eng, srv.server_address[1]

    yield make
    for srv in servers:
        srv.shutdown()


def _req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = None if body is None else json.dumps(body)
    c.request(method, path, payload,
              {"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


def get(port, path):
    return _req(port, "GET", path)


def post(port, path, body=None, headers=None):
    return _req(port, "POST", path, body, headers)


def msgs(text="hello world"):
    return [{"role": "user", "content": text}]


def test_health_carries_the_sleep_state(fake):
    _, port = fake()
    body = json.loads(get(port, "/health")[1])
    assert body["ok"] is True and body["state"] == "awake" and body["level"] == 0

    r, body = post(port, "/sleep?level=1")
    assert r.status == 200
    assert json.loads(body)["state"] == "asleep"

    body = json.loads(get(port, "/health")[1])
    assert body["state"] == "asleep" and body["level"] == 1
    assert "asleep_since" in body and "device" in body  # sleep/wake beside the device

    assert post(port, "/wake_up")[0].status == 200
    assert json.loads(get(port, "/health")[1])["state"] == "awake"


def test_a_generation_while_asleep_is_a_503_naming_the_wake_url(fake):
    _, port = fake()
    post(port, "/sleep")
    r, body = post(port, "/v1/chat/completions", {"messages": msgs()})
    assert r.status == 503
    err = json.loads(body)["error"]
    assert err["code"] == "model_asleep" and err["type"] == "service_unavailable"
    assert "/wake_up" in err["message"]


def test_the_anthropic_dialect_refuses_in_its_own_envelope(fake):
    _, port = fake()
    post(port, "/sleep")
    r, body = post(port, "/v1/messages",
                   {"model": "drinkme-fake", "max_tokens": 16, "messages": msgs()})
    assert r.status == 503
    obj = json.loads(body)
    assert obj["type"] == "error" and obj["error"]["type"] == "model_asleep"
    assert "/wake_up" in obj["error"]["message"]


def test_a_streamed_request_while_asleep_says_so_in_the_stream(fake):
    """Headers are committed before the first byte of content can exist, so a
    stream's refusal is an SSE error frame — the context-length check's precedent, same shape."""
    _, port = fake()
    post(port, "/sleep")
    r, body = post(port, "/v1/chat/completions",
                   {"messages": msgs(), "stream": True})
    assert r.status == 200
    frames = [json.loads(b[len("data: "):]) for b in body.decode().split("\n\n")
              if b.startswith("data: ") and b != "data: [DONE]"]
    assert any(f.get("error", {}).get("code") == "model_asleep" for f in frames)


def test_the_lockless_routes_keep_answering_while_asleep(fake):
    """The tokenizer never leaves (serving/sleep.py's "what does not move"),
    so counting a prompt, listing the model and probing health all work on a
    parked server — which is what makes a sleeping server diagnosable."""
    _, port = fake()
    post(port, "/sleep")
    assert get(port, "/v1/models")[0].status == 200
    assert get(port, "/tokenizer_info")[0].status == 200
    r, body = post(port, "/tokenize", {"prompt": "hello world"})
    assert r.status == 200 and json.loads(body)["count"] == 2
    r, body = post(port, "/v1/messages/count_tokens",
                   {"model": "drinkme-fake", "messages": msgs()})
    assert r.status == 200 and json.loads(body)["input_tokens"] == 2


def test_sleep_during_a_generation_is_a_409(fake):
    """The lock is taken NON-blockingly: a caller that asked a busy server to
    park is told so, rather than blocked while it finds out."""
    eng, port = fake(reply="w " * 60, delay=0.02)  # ~1.2s generation
    t = threading.Thread(target=post, args=(port, "/v1/chat/completions",
                                            {"messages": msgs()}))
    t.start()
    time.sleep(0.25)  # let the generation take the lock
    r, body = post(port, "/sleep")
    assert r.status == 409
    assert json.loads(body)["error"]["code"] == "generation_in_flight"
    t.join()
    assert json.loads(get(port, "/health")[1])["state"] == "awake"
    assert post(port, "/sleep")[0].status == 200  # and it works once free


def test_both_routes_are_idempotent(fake):
    """A systemd ExecStartPre that must run before every unit start cannot
    also be a thing that fails the second time."""
    _, port = fake()
    assert post(port, "/wake_up")[0].status == 200  # already awake
    assert post(port, "/sleep")[0].status == 200
    assert post(port, "/sleep")[0].status == 200
    assert post(port, "/wake_up")[0].status == 200
    assert post(port, "/wake_up")[0].status == 200


def test_a_bad_level_is_a_400_and_un_deepening_is_a_409(fake):
    _, port = fake()
    r, body = post(port, "/sleep?level=nine")
    assert r.status == 400 and json.loads(body)["error"]["code"] == "invalid_sleep_level"
    assert post(port, "/sleep?level=2")[0].status == 200
    r, body = post(port, "/sleep?level=1")
    assert r.status == 409
    assert json.loads(body)["error"]["code"] == "sleep_state_conflict"


def test_sleep_is_gated_by_the_bearer(fake):
    """Parking a live server's weights is the most consequential thing this
    port does; the bearer has to hold most exactly where it matters most."""
    _, port = fake(auth="s3cret")
    assert post(port, "/sleep")[0].status == 401
    assert post(port, "/wake_up")[0].status == 401
    r, _ = post(port, "/sleep", headers={"Authorization": "Bearer s3cret"})
    assert r.status == 200


class _NoSleepEngine(FakeEngine):
    """An Engine that implements the Protocol and nothing optional — the
    shape serve.py's `_persist_slots` already tolerates."""

    sleep = None
    wake = None
    sleep_state = None


def test_an_engine_that_cannot_sleep_says_501(fake):
    _, port = fake(engine=_NoSleepEngine())
    r, body = post(port, "/sleep")
    assert r.status == 501
    assert json.loads(body)["error"]["code"] == "sleep_unsupported"
    # and health still answers, with the awake default
    assert json.loads(get(port, "/health")[1])["state"] == "awake"


def test_a_wake_counts_as_activity(fake):
    """Without this the idle timer would find a stamp from before the sleep
    and park the model again on its very next tick."""
    eng, port = fake()
    post(port, "/sleep")
    time.sleep(0.05)
    before = time.monotonic()
    post(port, "/wake_up")
    # the server's stamp moved forward past the wake
    assert get(port, "/health")[0].status == 200
    r, _ = post(port, "/v1/chat/completions", {"messages": msgs()})
    assert r.status == 200 and before < time.monotonic()


# -------------------------------------------------------------- the timer --


class _Clock:
    """A server stand-in for _IdleSleeper: the four attributes it reads."""

    def __init__(self, engine):
        self.engine = engine
        self.last_generation = time.monotonic()
        self.gen_lock = threading.Lock()

    def sleep_state(self):
        return self.engine.sleep_state()


def test_the_idle_timer_parks_a_quiet_server():
    eng = FakeEngine()
    srv = _Clock(eng)
    t = _IdleSleeper(srv, seconds=0.15, tick=0.02)
    t.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and eng.sleep_state()["state"] == "awake":
            time.sleep(0.02)
        assert eng.sleep_state() == {**eng.sleep_state(), "state": "asleep",
                                     "level": 1}
        assert t.slept == 1  # latched: one attempt per idle period
    finally:
        t.stop()


def test_the_idle_timer_leaves_a_busy_server_alone():
    """The stamp is bumped on the way IN and on the way OUT of a generation,
    so a long generation can never be interrupted by a timer that fired while
    it ran."""
    eng = FakeEngine()
    srv = _Clock(eng)
    t = _IdleSleeper(srv, seconds=0.15, tick=0.02)
    t.start()
    try:
        for _ in range(15):
            srv.last_generation = time.monotonic()
            time.sleep(0.03)
        assert eng.sleep_state()["state"] == "awake" and t.slept == 0
    finally:
        t.stop()


def test_the_idle_timer_never_sleeps_under_a_held_lock():
    eng = FakeEngine()
    srv = _Clock(eng)
    srv.gen_lock.acquire()
    t = _IdleSleeper(srv, seconds=0.05, tick=0.02)
    t.start()
    try:
        time.sleep(0.3)
        assert eng.sleep_state()["state"] == "awake" and t.slept == 0
    finally:
        t.stop()
        srv.gen_lock.release()


def test_a_server_starts_no_timer_by_default(fake):
    _, port = fake()
    time.sleep(0.15)
    assert json.loads(get(port, "/health")[1])["state"] == "awake"


# --------------------------------------------------------- the env reader --


def test_sleep_on_idle_defaults_to_never(monkeypatch):
    monkeypatch.delenv("DRINKME_SLEEP_ON_IDLE_S", raising=False)
    assert serve._sleep_on_idle_from_env(None) == 0.0


def test_the_flag_wins_over_the_env_var(monkeypatch):
    """--ctx over DRINKME_CTX is the standing precedent (engines._ctx), and
    slots_from_env and _advertised_ctx_from_env both follow it."""
    monkeypatch.setenv("DRINKME_SLEEP_ON_IDLE_S", "600")
    assert serve._sleep_on_idle_from_env(30) == 30.0
    assert serve._sleep_on_idle_from_env(None) == 600.0


def test_garbage_warns_and_falls_back_rather_than_dying(monkeypatch, capsys):
    monkeypatch.setenv("DRINKME_SLEEP_ON_IDLE_S", "soon")
    assert serve._sleep_on_idle_from_env(None) == 0.0
    assert "idle sleep stays off" in capsys.readouterr().err
    monkeypatch.setenv("DRINKME_SLEEP_ON_IDLE_S", "-5")
    assert serve._sleep_on_idle_from_env(None) == 0.0
    assert serve._sleep_on_idle_from_env(-1) == 0.0


def test_the_state_object_reports_one_vocabulary():
    """/health, /sleep and /wake_up all render from this, so there is exactly
    one set of field names to learn."""
    st = SleepState()
    assert st.as_dict() == {"state": "awake", "level": 0}
    st.level, st.since, st.bytes_moved, st.tensors = 1, 1000.0, 42, 7
    d = st.as_dict()
    assert d["state"] == "asleep" and d["level"] == 1
    assert d["asleep_since"] == 1000 and d["parked_bytes"] == 42 and d["tensors"] == 7
    st.level, st.since, st.wake_s, st.slots_restored = 0, None, 1.25, 3
    assert st.as_dict() == {"state": "awake", "level": 0, "woke_in_s": 1.25,
                            "slots_restored": 3}


def test_engines_still_exports_persist_slots_unchanged(llama, tmp_path,
                                                       monkeypatch):
    """SIGTERM's path (on-disk prefix slots) is now shared with sleep's; its contract — a
    COUNT of the slots queued — must not have moved."""
    eng = cold_engine(llama, tmp_path, monkeypatch, ctx=1024)
    agent_turn(eng, [], LONG)
    assert eng.persist_slots(30) == 1
    assert isinstance(engines.HFEngine.persist_slots(eng, 30), int)


def test_a_fake_engine_carries_the_same_state_object():
    """The HTTP contract above is exercised against FakeEngine, so FakeEngine
    has to answer with the same vocabulary the real engine does."""
    eng = FakeEngine()
    assert eng.sleep_state() == {"state": "awake", "level": 0}
    assert eng.sleep(1)["state"] == "asleep"
    assert eng.wake()["state"] == "awake"
    assert isinstance(HFEngine.sleep_state, type(FakeEngine.sleep_state))


# ------------------------------------------------- the memory claim (GPU) --
#
# Everything above runs on CPU, where "the weights left the device" has no
# memory reading to assert on. These do: they skip cleanly without an
# accelerator, and they are the ruler for the reserved gate — a direct level-2
# sleep can report 128 MiB freed while torch.cuda.memory_reserved() still
# reads 128 MiB after it returns (5.92 GiB on the real 4B): the reported
# byte count and `model is None` both true, the blocks not handed back,
# because a local ref inventory still owns every tensor when empty_cache()
# runs. The isolation (one 128 MiB parameter, no cold tier, no generation)
# is the first test; the real toy pack through the real loader is the second.

_MIB = 2 ** 20
gpu_gate = pytest.mark.skipif(not torch.cuda.is_available(),
                              reason="the memory claim needs an accelerator")


def _reserved() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved()


def _floor() -> int:
    """Whatever the allocator holds before the engine exists — the number a
    level-2 sleep has to get back DOWN to, so the assertion is about this
    engine's bytes and not about what an earlier test left cached."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.memory_reserved()


def _bare_engine(nbytes: int) -> HFEngine:
    """The smallest engine whose sleep(2) is a real level-2 sleep: one
    parameter of `nbytes` on the device, no slots, no cold tier, no loader.
    Built as small as possible, so the only bytes that can stay reserved
    are the model's."""
    eng = HFEngine.__new__(HFEngine)
    eng.model = torch.nn.Module()
    eng.model.register_parameter("weight", torch.nn.Parameter(
        torch.ones(nbytes // 2, device="cuda", dtype=torch.bfloat16),
        requires_grad=False))
    eng.mtp_head = None
    eng.device = torch.device("cuda:0")
    eng._sleep, eng._sleep_refs, eng._slept_entries = SleepState(), [], []
    eng._reload = eng._cold = None
    eng._slots, eng._slot_clock = [], 0
    return eng


@gpu_gate
def test_a_direct_level_two_sleep_hands_the_blocks_back(capsys):
    """After sleep(2) RETURNS, the allocator holds nothing of the model —
    reserved, not just allocated, is back at the floor. A sleep that frees
    without emptying the cache reads floor + 134,217,728 B (the whole
    parameter) with allocated at 0: freed on paper, resident in fact. The
    idempotent repeat is asserted too, because it is the call an operator
    makes when the device still looks full, and it must release as well."""
    floor = _floor()
    eng = _bare_engine(128 * _MIB)
    loaded = _reserved()
    assert loaded >= floor + 128 * _MIB
    st = eng.sleep(2)
    assert st["level"] == 2 and st["parked_bytes"] == 128 * _MIB
    after = _reserved()
    assert torch.cuda.memory_allocated() <= floor  # true without the empty_cache too
    assert after <= floor, (
        f"level-2 sleep returned with {after - floor:,} B still reserved "
        f"(floor {floor:,}, loaded {loaded:,})")
    again = eng.sleep(2)
    assert again["level"] == 2 and again["asleep_since"] == st["asleep_since"]
    assert _reserved() <= floor
    assert "128.0 MiB (134,217,728 B) freed" in capsys.readouterr().out


@gpu_gate
def test_a_repeated_level_two_sleep_releases_what_arrived_in_between(capsys):
    """The repeat is not a pure no-op: blocks the allocator picked up AFTER
    the first level-2 sleep (a stray device tensor, dropped) are handed
    back by the second call rather than by nobody."""
    floor = _floor()
    eng = _bare_engine(16 * _MIB)
    eng.sleep(2)
    assert _reserved() <= floor
    stray = torch.ones(32 * _MIB // 2, device="cuda", dtype=torch.bfloat16)
    del stray
    assert _reserved() >= floor + 32 * _MIB  # cached, not returned
    eng.sleep(2)
    assert _reserved() <= floor


@gpu_gate
def test_level_two_through_the_real_loader_frees_the_device_and_wakes_bit_identical(toy):
    """The same claim on the real toy pack through load_compressed, on the
    accelerator: the packed planes, the escape sidecars, the raw tensors and
    the biases all go, the allocator is back at the floor, and the wake —
    the loader replayed — computes the same logits. This is the existing
    CPU wake check with the memory reading it could not make there."""
    floor = _floor()
    eng = load_compressed(toy[0], None, toy[1], device="cuda", ctx=256)
    before = _logits(eng).cpu()  # held on the host: the device must read empty
    loaded = _reserved()
    assert loaded > floor
    st = eng.sleep(2)
    assert st["level"] == 2 and eng.model is None
    assert st["by_class"][sleep_mod.PACKED] > 0
    after = _reserved()
    assert after <= floor, (
        f"level-2 sleep left {after - floor:,} B reserved of the "
        f"{loaded - floor:,} B the load took")
    eng.sleep(2)
    assert _reserved() <= floor
    st = eng.wake()
    assert st["state"] == "awake" and eng.model is not None
    assert torch.equal(before, _logits(eng).cpu())
    eng.sleep(2)  # leave the device as it was found
    assert _reserved() <= floor
