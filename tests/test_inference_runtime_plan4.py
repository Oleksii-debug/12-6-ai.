from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization.byte import ByteTokenizer
from tools.inference_runtime import (
    GenerationConfig,
    InferenceError,
    ReferenceInference,
)


@pytest.fixture
def runtime():
    spec = ModelSpec(schema_version=1, vocab_size=256, max_seq_len=32,
                     d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                     head_dim=8, d_ff=32, rope_rotary_dim=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = TwelveSixDecoder(spec)
    model.eval()
    return ReferenceInference(model, ByteTokenizer())


def test_stream_nonstream_greedy_same_final_identity_tokens_and_text(runtime):
    cfg = GenerationConfig(max_new_tokens=4)
    result = runtime.generate("A", cfg)
    events = list(runtime.stream("A", cfg))
    assert len(events) == 5
    assert all(event.kind == "TOKEN" for event in events[:-1])
    assert [event.sequence for event in events] == [1, 2, 3, 4, 4]
    assert events[-1].kind == "COMPLETED"
    assert events[-1].result == result
    assert result.status == "COMPLETED"
    assert result.prompt_token_count == 1
    assert len(result.output_token_ids) == 4
    assert all(event.result.request_id == result.request_id for event in events)
    assert all(event.result.output_token_ids == result.output_token_ids[:i+1]
               for i, event in enumerate(events[:-1]))
    assert result.model_weights_sha256 and result.tokenizer_sha256


def test_sampler_uses_decoder_semantics_exact_rng_and_sampling_config(runtime):
    cfg = GenerationConfig(max_new_tokens=3, strategy="sample", temperature=0.7,
                           top_k=7, seed=717)
    a = runtime.generate("x", cfg)
    b = list(runtime.stream("x", cfg))[-1].result
    assert a == b
    ids = torch.tensor([[ord("x")]], dtype=torch.long)
    generator = torch.Generator().manual_seed(717)
    direct = runtime.model.generate(ids, max_new_tokens=3, do_sample=True,
                                    temperature=0.7, top_k=7, generator=generator)
    assert a.output_token_ids == tuple(direct[0, 1:].tolist())
    changed = runtime.generate("x", replace(cfg, seed=718))
    assert changed.request_id != a.request_id
    assert changed.config_sha256 != a.config_sha256


def test_greedy_has_bitwise_reference_decoder_parity(runtime):
    cfg = GenerationConfig(max_new_tokens=4)
    ids = torch.tensor([[65]], dtype=torch.long)
    expected = runtime.model.generate(ids, max_new_tokens=4)
    observed = runtime.generate("A", cfg)
    assert observed.output_token_ids == tuple(expected[0, 1:].tolist())


def test_cancel_stops_token_generation_without_falsely_completing(runtime):
    cfg = GenerationConfig(max_new_tokens=4)
    session = runtime.stream("cancel", cfg)
    it = iter(session)
    first = next(it)
    assert first.kind == "TOKEN"
    with pytest.raises(InferenceError, match="incomplete"):
        session.result()
    session.cancel()
    remaining = list(it)
    assert len(remaining) == 1 and remaining[0].kind == "CANCELLED"
    assert remaining[0].token_id is None
    cancelled = session.result()
    assert cancelled.status == "CANCELLED"
    assert cancelled.output_token_ids == (first.token_id,)
    fresh = runtime.generate("cancel", cfg)
    assert fresh.status == "COMPLETED"
    assert fresh.request_id == cancelled.request_id
    assert fresh.result_sha256 != cancelled.result_sha256
    with pytest.raises(InferenceError, match="terminal"):
        session.cancel()
    with pytest.raises(InferenceError, match="single-use"):
        iter(session)


def test_cancel_before_first_token_and_zero_length_terminal(runtime):
    cfg = GenerationConfig(max_new_tokens=3)
    session = runtime.start("p", cfg)
    session.cancel()
    events = list(session)
    assert len(events) == 1 and events[0].kind == "CANCELLED"
    assert events[0].result.output_token_ids == ()
    zero = GenerationConfig(max_new_tokens=0)
    terminal = list(runtime.stream("p", zero))
    assert len(terminal) == 1 and terminal[0].kind == "COMPLETED"
    assert terminal[0].result == runtime.generate("p", zero)


def test_restart_parity_with_same_state_dict_and_changed_state_rejected(runtime):
    spec = runtime.model.spec
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        restored = TwelveSixDecoder(spec)
    restored.load_state_dict(runtime.model.state_dict())
    restored.eval()
    second = ReferenceInference(restored, ByteTokenizer())
    cfg = GenerationConfig(max_new_tokens=2, strategy="sample", seed=3, top_k=4)
    assert second.generate("same", cfg) == runtime.generate("same", cfg)
    with torch.no_grad():
        next(restored.parameters()).reshape(-1)[0].add_(0.01)
    with pytest.raises(InferenceError, match="identity drift"):
        second.start("same", cfg)


def test_nonfinite_model_state_fails_closed(runtime):
    with torch.no_grad():
        next(runtime.model.parameters()).reshape(-1)[0] = float("nan")
    with pytest.raises(InferenceError, match="nonfinite"):
        runtime.start("hello", GenerationConfig(1))


def test_reference_model_training_mode_rejected_and_recoverable(runtime):
    runtime.model.train()
    with pytest.raises(InferenceError, match="eval mode"):
        runtime.start("x", GenerationConfig(1))
    runtime.model.eval()
    assert runtime.generate("x", GenerationConfig(1)).status == "COMPLETED"


@pytest.mark.parametrize("config", [
    {"max_new_tokens": -1}, {"max_new_tokens": True},
    {"max_new_tokens": 1, "strategy": "unknown"},
    {"max_new_tokens": 1, "strategy": "sample"},
    {"max_new_tokens": 1, "strategy": "sample", "seed": True},
    {"max_new_tokens": 1, "strategy": "sample", "seed": -1},
    {"max_new_tokens": 1, "strategy": "sample", "seed": 1, "top_k": 0},
    {"max_new_tokens": 1, "strategy": "sample", "seed": 1, "temperature": float("nan")},
    {"max_new_tokens": 1, "strategy": "greedy", "temperature": 0.5},
    {"max_new_tokens": 1, "strategy": "greedy", "seed": 1},
    {"max_new_tokens": 1, "decode_errors": "ignore"},
])
def test_invalid_generation_config_rejected(config):
    with pytest.raises(InferenceError):
        GenerationConfig(**config)


def test_context_and_vocab_incompatibility_never_silently_truncate(runtime):
    with pytest.raises(InferenceError, match="context capacity"):
        runtime.generate("a", GenerationConfig(max_new_tokens=32))
    with pytest.raises(InferenceError, match="nonempty"):
        runtime.generate("", GenerationConfig(1))
    with pytest.raises(InferenceError, match="top_k exceeds"):
        runtime.generate("a", GenerationConfig(1, strategy="sample", seed=1, top_k=257))
    class WrongVocab(ByteTokenizer):
        vocab_size = 257
    with pytest.raises(InferenceError, match="vocabulary incompatible"):
        ReferenceInference(runtime.model, WrongVocab())


def test_unicode_prompt_token_count_and_identity_binding(runtime):
    r = runtime.generate("Україна", GenerationConfig(0))
    assert r.prompt_token_count == len("Україна".encode("utf-8"))
    assert r.text == ""
    assert r.request_id != runtime.generate("УкраїнА", GenerationConfig(0)).request_id


def test_strict_bad_utf8_fails_and_replace_is_explicit(runtime, monkeypatch):
    def emit_invalid(ids, **kwargs):
        return torch.cat((ids, torch.full((1, 1), 255, dtype=ids.dtype)), dim=1)
    monkeypatch.setattr(runtime.model, "generate", emit_invalid)
    with pytest.raises(InferenceError, match="decode failed"):
        runtime.generate("a", GenerationConfig(1, decode_errors="strict"))
    assert runtime.generate("a", GenerationConfig(1, decode_errors="replace")).text == "�"


def test_prefix_violation_is_rejected(runtime, monkeypatch):
    def wrong_prefix(ids, **kwargs):
        bad = torch.cat((ids, torch.zeros((1, 1), dtype=ids.dtype)), dim=1)
        bad[0, 0] = (bad[0, 0] + 1) % 256
        return bad
    monkeypatch.setattr(runtime.model, "generate", wrong_prefix)
    with pytest.raises(InferenceError, match="prefix"):
        runtime.generate("a", GenerationConfig(1))


def test_midstream_weight_drift_fails_closed(runtime):
    session = runtime.stream("drift", GenerationConfig(3))
    events = iter(session)
    assert next(events).kind == "TOKEN"
    with torch.no_grad():
        next(runtime.model.parameters()).reshape(-1)[0].add_(0.25)
    with pytest.raises(InferenceError, match="drift during streaming"):
        next(events)
