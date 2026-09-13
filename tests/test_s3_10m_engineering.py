from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, count_trainable_parameters, load_stage_config
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


def test_repository_stage_and_d11_are_not_current_byte_bound() -> None:
    tokenizer = ByteTokenizer()
    repository_stage = load_stage_config(Path("configs/stages/s3_10m.json"))
    assert repository_stage.expected_parameters == 10_059_840
    assert (
        repository_stage.model.identity_sha256()
        == "3b6fc1b397e6fea69c2f249ce8ab8eedaad8ca1b13b88b8d2328a6abcf34791a"
    )
    assert repository_stage.model.vocab_size == 8192
    assert repository_stage.model.vocab_size != tokenizer.vocab_size

    model = TwelveSixDecoder(s3_d11_model_spec(), s3_init_spec())
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
