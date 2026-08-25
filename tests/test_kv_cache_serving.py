from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from twelve_six.inference.sampling import greedy_token
from twelve_six.integration.s0_runtime import (
    S0TorchInferenceBackend,
    kv_cache_payload_bytes,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _s0_model(seed: int = 20260825) -> TwelveSixDecoder:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    torch.manual_seed(seed)
    return TwelveSixDecoder(stage.model, stage.init)


def test_model_cache_is_batch_safe_for_equal_length_prefill_and_decode() -> None:
    model = _s0_model()
    model.eval()
    prompt = torch.tensor([[1, 2, 3], [10, 11, 12]], dtype=torch.long)

    cached_prompt, cache = model.prefill_kv_cache(prompt)
    full_prompt = model(prompt).logits
    torch.testing.assert_close(cached_prompt.logits, full_prompt, rtol=0.0, atol=0.0)
    assert cache.batch_size == 2

    next_tokens = torch.tensor([[4], [13]], dtype=torch.long)
    cached_next, next_cache = model.decode_one_with_kv_cache(next_tokens, cache)
    full_next = model(torch.cat((prompt, next_tokens), dim=1)).logits[:, -1:, :]
    torch.testing.assert_close(cached_next.logits, full_next, rtol=1e-6, atol=1e-6)
    assert next_cache.batch_size == 2
    assert next_cache.sequence_length == 4


def test_overlapping_generation_sessions_share_eval_lifecycle_safely() -> None:
    model = _s0_model()
    backend = S0TorchInferenceBackend(model, ByteTokenizer())
    assert model.training is True

    first = backend.begin_generation(backend.encode("first"))
    second = backend.begin_generation(backend.encode("second"))
    assert backend.active_generation_sessions == 2
    assert model.training is False

    first.close()
    assert backend.active_generation_sessions == 1
    assert model.training is False

    token_id = greedy_token(second.next_token_logits())
    second.append(token_id)
    assert second.sequence_length == len(backend.encode("second")) + 1

    second.close()
    assert backend.active_generation_sessions == 0
    assert model.training is True

    first.close()
    second.close()
    assert backend.active_generation_sessions == 0


def test_cache_memory_estimator_matches_payload_and_gqa_geometry() -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    model = _s0_model()
    backend = S0TorchInferenceBackend(model, ByteTokenizer())
    prompt_ids = backend.encode("cache")

    session = backend.begin_generation(prompt_ids)
    try:
        assert session.cache_bytes == backend.estimate_cache_bytes(len(prompt_ids))
    finally:
        session.close()

    mha_bytes = kv_cache_payload_bytes(
        stage.model,
        sequence_length=stage.model.max_seq_len,
        element_size_bytes=2,
    )
    gqa_spec = replace(stage.model, n_kv_heads=1)
    gqa_bytes = kv_cache_payload_bytes(
        gqa_spec,
        sequence_length=gqa_spec.max_seq_len,
        element_size_bytes=2,
    )
    assert mha_bytes == 2 * gqa_bytes
    assert kv_cache_payload_bytes(
        gqa_spec,
        sequence_length=gqa_spec.max_seq_len,
        batch_size=3,
        element_size_bytes=2,
    ) == 3 * gqa_bytes
