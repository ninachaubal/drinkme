"""generation_config.json as the request-defaults source.

Layered like the code is: gen_config.py's pure functions get pure-function
tests (no model, no download); the HTTP-layer merge (request field > engine
default > OpenAI default) is exercised through FakeEngine, exactly like
every other http.py boundary test; one real HFEngine (load_stock on a toy
Llama, no codec/pack machinery needed — the file-reading path is identical
for both arms) proves the load_stock/load_compressed wiring itself.
"""

import http.client
import json
import os

import pytest
import torch

from drinkme.serving import gen_config
from drinkme.serving.engine import FakeEngine, SampleParams
from drinkme.serving.http import start_server

MSGS = [{"role": "user", "content": "hi"}]


def post(port, body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


# --------------------------------------------------------------- gen_config.load --


def test_fields_present_are_applied_present_only(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "bos_token_id": 1, "eos_token_id": [2, 3],  # not sampling fields: ignored there
    }))
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.sampling == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert gd.template_kwargs == {}
    assert str(tmp_path) in gd.source


# ------------------------------------------------------------------- eos_ids --
# the compressed arm's manufactured model.generation_config cannot be
# trusted for EOS; gen_config.load reads it from the FILE instead, and
# engines.py unions it in (test_serving_engines.py's _eos_ids tests).


def test_eos_ids_list_shape_normalised_sorted(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "eos_token_id": [106, 1, 50],
    }))
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.eos_ids == (1, 50, 106)


def test_eos_ids_int_shape_normalised_to_one_tuple(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "eos_token_id": 2,
    }))
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.eos_ids == (2,)


def test_eos_ids_absent_is_empty_tuple(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "temperature": 0.7,
    }))
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.eos_ids == ()


def test_eos_ids_missing_file_is_empty_tuple(tmp_path):
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.eos_ids == ()


def test_repetition_and_penalty_fields_pass_through(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "repetition_penalty": 1.1, "presence_penalty": 1.5, "frequency_penalty": 0.3,
    }))
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.sampling == {"repetition_penalty": 1.1, "presence_penalty": 1.5,
                           "frequency_penalty": 0.3}


def test_chat_template_kwargs_extracted(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False},
    }))
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.sampling == {"temperature": 0.7}
    assert gd.template_kwargs == {"enable_thinking": False}


def test_missing_file_is_empty_no_crash(tmp_path):
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.sampling == {} and gd.template_kwargs == {}
    assert gen_config.effective_defaults(gd.sampling) == gen_config.effective_defaults({})


def test_missing_file_falls_back_to_model_generation_config_diff(tmp_path):
    from transformers import GenerationConfig

    class M:
        generation_config = GenerationConfig(temperature=0.42, top_k=7)

    gd = gen_config.load(model=M(), snapshot_dir=str(tmp_path))
    assert gd.sampling == {"temperature": 0.42, "top_k": 7}
    assert "model.generation_config" in gd.source


def test_missing_file_and_no_generation_config_attr_is_empty(tmp_path):
    class M:
        pass

    gd = gen_config.load(model=M(), snapshot_dir=str(tmp_path))
    assert gd.sampling == {}


def test_unreadable_file_falls_back_without_crashing(tmp_path, capsys):
    (tmp_path / "generation_config.json").write_text("{not json")
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.sampling == {}
    assert "unreadable" in capsys.readouterr().err


def test_ignore_env_skips_sampling_and_template_fields(tmp_path, monkeypatch):
    (tmp_path / "generation_config.json").write_text(json.dumps({"temperature": 0.1}))
    monkeypatch.setenv(gen_config.IGNORE_ENV, "1")
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.sampling == {} and "ignored" in gd.source


def test_ignore_env_does_not_skip_eos_ids(tmp_path, monkeypatch):
    # DRINKME_IGNORE_GENERATION_CONFIG is documented (and
    # tested above) as restoring OpenAI SAMPLING defaults; it must not also
    # blind the compressed arm to the file's own eos_token_id, or setting a
    # sampling-only knob silently reopens the EOS bug.
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "temperature": 0.1, "eos_token_id": [2, 99],
    }))
    monkeypatch.setenv(gen_config.IGNORE_ENV, "1")
    gd = gen_config.load(model=None, snapshot_dir=str(tmp_path))
    assert gd.sampling == {}
    assert gd.eos_ids == (2, 99)


# ------------------------------------------------------ effective_defaults/resolve --


def test_effective_defaults_is_the_full_openai_table_when_no_overrides():
    d = gen_config.effective_defaults({})
    assert d == {f: getattr(SampleParams, f) for f in gen_config.SAMPLING_FIELDS}


def test_resolve_sampling_precedence_request_over_defaults_over_openai():
    defaults = {"temperature": 0.6, "top_p": 0.95}
    # request explicit -> wins
    out = gen_config.resolve_sampling({"temperature": 0.9}, defaults)
    assert out["temperature"] == 0.9
    # request absent (None) -> falls to defaults
    assert out["top_p"] == 0.95
    # neither -> SampleParams' own OpenAI default
    assert out["top_k"] == SampleParams.top_k
    assert out["repetition_penalty"] == SampleParams.repetition_penalty


# --------------------------------------------------------- HTTP layer (FakeEngine) --


@pytest.fixture
def fake():
    servers = []

    def make(engine=None, **kw):
        eng = engine or FakeEngine(**kw)
        srv = start_server(eng, "127.0.0.1", 0)
        servers.append(srv)
        return eng, srv.server_address[1]

    yield make
    for srv in servers:
        srv.shutdown()


def _capture(engine):
    """Wrap `engine.generate` to record the SampleParams it receives, the
    same intent as test_serving_http.py's Capture(FakeEngine) subclass, done
    by rebinding the instance method so it works for any Engine instance."""
    seen = {}
    orig = engine.generate

    def wrapped(req):
        seen["params"] = req.sampling
        return orig(req)

    engine.generate = wrapped
    return engine, seen


def test_no_sampling_fields_samples_with_generation_config_defaults(fake):
    eng = FakeEngine(reply="ok", defaults={"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    eng, seen = _capture(eng)
    _, port = fake(engine=eng)
    r, _ = post(port, {"messages": MSGS})
    assert r.status == 200
    p = seen["params"]
    assert (p.temperature, p.top_p, p.top_k) == (0.6, 0.95, 20)


def test_request_field_overrides_generation_config_default(fake):
    eng = FakeEngine(reply="ok", defaults={"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    eng, seen = _capture(eng)
    _, port = fake(engine=eng)
    r, _ = post(port, {"messages": MSGS, "temperature": 0.05})
    assert r.status == 200
    p = seen["params"]
    assert p.temperature == 0.05          # the request's own value
    assert (p.top_p, p.top_k) == (0.95, 20)  # untouched fields still default


def test_no_generation_config_is_openai_defaults_no_crash(fake):
    eng = FakeEngine(reply="ok")  # defaults={} — no generation_config.json found
    eng, seen = _capture(eng)
    _, port = fake(engine=eng)
    r, _ = post(port, {"messages": MSGS})
    assert r.status == 200
    p = seen["params"]
    assert (p.temperature, p.top_p, p.top_k) == (1.0, 1.0, 0)


def test_repetition_penalty_now_readable_from_the_wire(fake):
    eng = FakeEngine(reply="ok")
    eng, seen = _capture(eng)
    _, port = fake(engine=eng)
    r, _ = post(port, {"messages": MSGS, "repetition_penalty": 1.2})
    assert r.status == 200 and seen["params"].repetition_penalty == 1.2


def test_v1_models_reports_effective_defaults(fake):
    _, port = fake(engine=FakeEngine(defaults={"temperature": 0.6, "top_p": 0.95}))
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/v1/models")
    body = json.loads(c.getresponse().read())
    c.close()
    d = body["data"][0]["drinkme"]["sampling"]["defaults"]
    assert d["temperature"] == 0.6 and d["top_p"] == 0.95
    assert d["top_k"] == SampleParams.top_k  # untouched field still OpenAI default


def test_anthropic_dialect_applies_defaults_including_fields_it_has_no_wire_for(fake):
    """Anthropic's wire has no presence_penalty field at all; the effective
    default must still reach the engine when the checkpoint recommends one."""
    eng = FakeEngine(reply="ok", defaults={"presence_penalty": 1.5, "temperature": 0.7})
    eng, seen = _capture(eng)
    _, port = fake(engine=eng)
    r, _ = post(port, {"model": eng.model_id, "max_tokens": 8,
                       "messages": [{"role": "user", "content": "hi"}]},
               path="/v1/messages")
    assert r.status == 200
    p = seen["params"]
    assert p.presence_penalty == 1.5 and p.temperature == 0.7


# -------------------------------------------------- real engine (load_stock, CPU) --


@pytest.fixture(scope="module")
def toy_dir(tmp_path_factory):
    """A tiny real Llama + tokenizer on disk — load_stock only (no codec/pack
    machinery: the generation_config.json file-reading path is identical for both arms, so
    this is enough to prove the load_stock/load_compressed wiring)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("gen_config_toy")
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2, "hi": 3, "user": 4, "assistant": 5, ":": 6}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    tok.save_pretrained(str(d))
    cfg = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=64,
                      tie_word_embeddings=True, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(str(d), safe_serialization=True)
    return str(d)


def _write_gen_config(toy_dir, obj):
    with open(os.path.join(toy_dir, "generation_config.json"), "w") as f:
        json.dump(obj, f)


def test_load_stock_reads_the_checkpoints_generation_config(toy_dir, capsys):
    from drinkme.serving.engines import load_stock

    _write_gen_config(toy_dir, {"temperature": 0.42, "top_p": 0.31, "top_k": 7})
    eng = load_stock(toy_dir, None, device="cpu")
    d = eng.model_meta()["sampling"]["defaults"]
    assert d["temperature"] == 0.42 and d["top_p"] == 0.31 and d["top_k"] == 7
    out = capsys.readouterr().out
    assert "sampling defaults from generation_config.json" in out
    assert "0.42" in out


def test_load_stock_end_to_end_over_http(toy_dir, monkeypatch):
    from drinkme.serving.engines import load_stock

    _write_gen_config(toy_dir, {"temperature": 0.42, "top_p": 0.31, "top_k": 7})
    eng = load_stock(toy_dir, None, device="cpu")
    seen = {}
    orig = eng.generate

    def capture(req):
        seen["params"] = req.sampling
        return orig(req)

    monkeypatch.setattr(eng, "generate", capture)
    srv = start_server(eng, "127.0.0.1", 0)
    try:
        r, _ = post(srv.server_address[1],
                   {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
        assert r.status == 200
        p = seen["params"]
        assert (p.temperature, p.top_p, p.top_k) == (0.42, 0.31, 7)
    finally:
        srv.shutdown()


def test_missing_generation_config_file_on_a_real_load_is_openai_defaults(toy_dir):
    from drinkme.serving.engines import load_stock

    os.remove(os.path.join(toy_dir, "generation_config.json"))
    eng = load_stock(toy_dir, None, device="cpu")
    d = eng.model_meta()["sampling"]["defaults"]
    assert (d["temperature"], d["top_p"], d["top_k"]) == (1.0, 1.0, 0)


def test_ignore_env_restores_openai_defaults_on_a_real_load(toy_dir, monkeypatch):
    from drinkme.serving.engines import load_stock

    _write_gen_config(toy_dir, {"temperature": 0.1, "top_p": 0.2, "top_k": 3})
    monkeypatch.setenv(gen_config.IGNORE_ENV, "1")
    eng = load_stock(toy_dir, None, device="cpu")
    d = eng.model_meta()["sampling"]["defaults"]
    assert (d["temperature"], d["top_p"], d["top_k"]) == (1.0, 1.0, 0)


# --------------------------- real cache: the compressed EOS set --
# No weights loaded — only the real generation_config.json/config.json/
# tokenizer_config.json already in the local HF cache, offline. The
# compressed arm never builds a real model.generation_config (engines.py's
# module docstring); this reproduces the manufactured object it DOES build
# (GenerationConfig.from_model_config over AutoConfig alone) and feeds it,
# the real tokenizer, and gen_config.load's own read of the file into
# engines._eos_ids — the exact union HFEngine.__init__ runs.

GEMMA = ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")
QWEN8B = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
QWEN27B = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")


def _cached_tokenizer(repo: str, revision: str):
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, "tokenizer_config.json", revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache "
                    "(tokenizer_config.json not found)")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)


def _cached_snapshot_dir(repo: str, revision: str) -> str:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, "generation_config.json", revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache "
                    "(generation_config.json not found)")
    return os.path.dirname(hit)


def _cached_manufactured_generation_config(repo: str, revision: str):
    """What load_compressed's meta skeleton actually carries on
    model.generation_config: GenerationConfig.from_model_config over
    config.json alone — skeleton() builds from AutoConfig, never
    from_pretrained, so the repo's own generation_config.json is never
    seen through this object (gen_config.py's module docstring)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoConfig, GenerationConfig

    cfg = AutoConfig.from_pretrained(repo, revision=revision, local_files_only=True)
    return GenerationConfig.from_model_config(cfg)


def _pre_fix_eos(tokenizer, manufactured) -> set:
    """The OLD two-source union (tokenizer + model.generation_config only),
    for the compressed arm where the second source is the manufactured
    object above — reproduces the wrong EOS set exactly."""
    out: set = set()
    for src in (tokenizer.eos_token_id, manufactured.eos_token_id):
        if isinstance(src, int):
            out.add(src)
        elif isinstance(src, (list, tuple)):
            out.update(int(e) for e in src)
    return out


def test_real_gemma4_compressed_eos_set_equals_the_file_not_config_json():
    from drinkme.serving.engines import _eos_ids

    tok = _cached_tokenizer(*GEMMA)
    manufactured = _cached_manufactured_generation_config(*GEMMA)
    gd = gen_config.load(None, _cached_snapshot_dir(*GEMMA))
    # the bug, reproduced: config.json alone is missing 50 (<|tool_response>,
    # its end-of-calls token) that the real generation_config.json carries
    assert _pre_fix_eos(tok, manufactured) == {1, 106}
    assert gd.eos_ids == (1, 50, 106)
    assert _eos_ids(tok, manufactured, gd.eos_ids) == frozenset({1, 106, 50})


def test_real_qwen3_8b_compressed_eos_set_contains_endoftext():
    from drinkme.serving.engines import _eos_ids

    tok = _cached_tokenizer(*QWEN8B)
    manufactured = _cached_manufactured_generation_config(*QWEN8B)
    gd = gen_config.load(None, _cached_snapshot_dir(*QWEN8B))
    endoftext = tok.convert_tokens_to_ids("<|endoftext|>")
    # the bug, reproduced: config.json's eos_token_id is the bare int
    # <|im_end|>; <|endoftext|> lives only in the real file
    assert endoftext not in _pre_fix_eos(tok, manufactured)
    assert endoftext in gd.eos_ids
    assert endoftext in _eos_ids(tok, manufactured, gd.eos_ids)


def _cached_real_generation_config(repo: str, revision: str):
    """The stock arm's actual model.generation_config: from_pretrained reads
    the repo's own generation_config.json (no weights needed for this call —
    it is a small JSON read, same as _cached_manufactured_generation_config's
    AutoConfig call, just through the class that goes looking for the file
    instead of manufacturing a stand-in from config.json alone)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import GenerationConfig

    return GenerationConfig.from_pretrained(repo, revision=revision,
                                            local_files_only=True)


# The general test: for every pack/model row the suite already exercises,
# compressed_eos_set must
# equal stock_eos_set. The three MENU_ROWS below are the only rows this can
# run for on CPU without weights — GenerationConfig.from_pretrained and
# AutoConfig.from_pretrained both read small JSON files, so the REAL stock
# object and the REAL manufactured compressed stand-in can both be built
# from the local cache alone. Rows this general check does NOT cover, and
# why:
#   - the toy Llama in test_serving_engines.py: real weights, both arms
#     genuinely loaded — covered there by its own general/toy-row test.
#   - the MTP Qwen3_5 hybrid toy (test_serving_mtp.py) and the MLX Qwen3 toy
#     (test_serving_engine_mlx.py): the suite only ever builds ONE arm
#     (load_compressed) for either today, so there is no stock_eos_set to
#     compare against without adding a fixture.
#   - the in-memory gemma toy (test_serving_gemma_spec.py and friends):
#     built straight from a model object via HFEngine(...), never through a
#     snapshot_dir — there is no generation_config.json file for it, so this
#     module's machinery is not in play for it at all.
MENU_ROWS = (GEMMA, QWEN8B, QWEN27B)


@pytest.mark.parametrize("repo,revision", MENU_ROWS,
                        ids=["gemma-4-31B-it", "Qwen3-8B", "Qwen3.8-27B"])
def test_real_compressed_eos_set_equals_stock_for_every_cached_pack_model_row(
        repo, revision):
    from drinkme.serving.engines import _eos_ids

    tok = _cached_tokenizer(repo, revision)
    manufactured = _cached_manufactured_generation_config(repo, revision)
    real = _cached_real_generation_config(repo, revision)
    gd = gen_config.load(None, _cached_snapshot_dir(repo, revision))
    compressed_set = _eos_ids(tok, manufactured, gd.eos_ids)
    stock_set = _eos_ids(tok, real, gd.eos_ids)
    assert compressed_set == stock_set
