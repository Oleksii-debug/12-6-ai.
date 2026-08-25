from __future__ import annotations

import pytest

from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, count_trainable_parameters
from twelve_six.s3_engineering import (
    S3_CURRENT_EXPECTED_PARAMETERS,
    S3_CURRENT_MODEL_SHA256,
    S3_D11_EXPECTED_PARAMETERS,
    S4_D11_EXPECTED_PARAMETERS,
    kv_cache_bytes,
    s3_current_model_spec,
    s3_d11_model_spec,
    s3_init_spec,
    s4_d11_model_spec,
)
from twelve_six.tokenization import ByteTokenizer


def test_s3_current_exact_algebra_real_construction_and_byte_backend() -> None:
    spec = s3_current_model_spec()
    assert spec.identity_sha256() == S3_CURRENT_MODEL_SHA256
    assert spec.parameter_breakdown() == {
        "token_embedding": 65_536,
        "attention_weights_per_layer": 163_840,
        "attention_biases_per_layer": 0,
        "attention_per_layer": 163_840,
        "mlp_weights_per_layer": 663_552,
        "mlp_biases_per_layer": 0,
        "mlp_per_layer": 663_552,
        "norms_per_layer": 512,
        "block_per_layer": 827_904,
        "blocks_total": 9_934_848,
        "final_norm": 256,
        "lm_head_extra": 0,
        "total": S3_CURRENT_EXPECTED_PARAMETERS,
    }
    model = TwelveSixDecoder(spec, s3_init_spec())
    assert count_trainable_parameters(model) == S3_CURRENT_EXPECTED_PARAMETERS
    tokenizer = ByteTokenizer()
    backend = S0TorchInferenceBackend(model, tokenizer)
    assert backend.max_context_tokens == 1024


def test_s3_d11_future_tokenizer_alternative_still_fails_current_byte_binding() -> None:
    model = TwelveSixDecoder(s3_d11_model_spec(), s3_init_spec())
    tokenizer = ByteTokenizer()
    assert model.spec.parameter_count() == S3_D11_EXPECTED_PARAMETERS
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        S0TorchInferenceBackend(model, tokenizer)


def test_s3_and_s4_gqa_cache_payload_algebra() -> None:
    current = s3_current_model_spec()
    assert kv_cache_bytes(
        current,
        batch_size=1,
        sequence_length=current.max_seq_len,
        bytes_per_element=2,
    ) == 3_145_728

    future = s3_d11_model_spec()
    assert kv_cache_bytes(
        future,
        batch_size=1,
        sequence_length=future.max_seq_len,
        bytes_per_element=2,
    ) == 6_291_456

    s4 = s4_d11_model_spec()
    assert s4.parameter_count() == S4_D11_EXPECTED_PARAMETERS
    assert kv_cache_bytes(
        s4,
        batch_size=1,
        sequence_length=s4.max_seq_len,
        bytes_per_element=2,
    ) == 50_331_648
