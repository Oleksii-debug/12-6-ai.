from __future__ import annotations

import pytest

from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, count_trainable_parameters
from twelve_six.s3_engineering import (
    S3_D11_EXPECTED_PARAMETERS,
    S3_D11_MODEL_SHA256,
    S4_D11_EXPECTED_PARAMETERS,
    kv_cache_bytes,
    s3_d11_init_spec,
    s3_d11_model_spec,
    s4_d11_model_spec,
)
from twelve_six.tokenization import ByteTokenizer


def test_s3_d11_exact_algebra_and_real_construction() -> None:
    spec = s3_d11_model_spec()
    assert spec.identity_sha256() == S3_D11_MODEL_SHA256
    assert spec.parameter_breakdown() == {
        "token_embedding": 2_621_440,
        "attention_weights_per_layer": 245_760,
        "attention_biases_per_layer": 0,
        "attention_per_layer": 245_760,
        "mlp_weights_per_layer": 675_840,
        "mlp_biases_per_layer": 0,
        "mlp_per_layer": 675_840,
        "norms_per_layer": 640,
        "block_per_layer": 922_240,
        "blocks_total": 7_377_920,
        "final_norm": 320,
        "lm_head_extra": 0,
        "total": S3_D11_EXPECTED_PARAMETERS,
    }
    model = TwelveSixDecoder(spec, s3_d11_init_spec())
    assert count_trainable_parameters(model) == S3_D11_EXPECTED_PARAMETERS


def test_s3_current_byte_tokenizer_mismatch_fails_closed() -> None:
    model = TwelveSixDecoder(s3_d11_model_spec(), s3_d11_init_spec())
    tokenizer = ByteTokenizer()
    assert tokenizer.vocab_size == 256
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        S0TorchInferenceBackend(model, tokenizer)


def test_s3_and_s4_gqa_cache_payload_algebra() -> None:
    s3 = s3_d11_model_spec()
    assert kv_cache_bytes(
        s3,
        batch_size=1,
        sequence_length=s3.max_seq_len,
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
