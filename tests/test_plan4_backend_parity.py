"""Plan 4 S5 tiny LOCAL_FREE quantization parity + negative acceptance tests."""
from __future__ import annotations

import pytest
import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization.byte import ByteTokenizer
from tools.inference_runtime import GenerationConfig, ReferenceInference
from tools.plan4_backend_parity import (
    CANDIDATE,
    BackendQualificationError,
    ParityPolicy,
    feature_matrix,
    qualify_backend,
    require_backend,
)


@pytest.fixture
def runtime():
    spec = ModelSpec(schema_version=1, vocab_size=256, max_seq_len=24,
                     d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                     head_dim=8, d_ff=32, rope_rotary_dim=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = TwelveSixDecoder(spec)
    model.eval()
    return ReferenceInference(model, ByteTokenizer())


def test_backend_matrix_has_explicit_prohibitions_and_no_vendor_approximation():
    matrix = feature_matrix()
    assert matrix[CANDIDATE]["supported"] is True
    assert matrix[CANDIDATE]["quantized"] is True
    assert matrix[CANDIDATE]["sample"] is False
    for name in ("gguf", "llama.cpp", "vllm", "onnx", "cuda", "unknown"):
        assert matrix[name]["supported"] is False
        with pytest.raises(BackendQualificationError, match="not qualified"):
            require_backend(name, GenerationConfig(1))
    with pytest.raises(BackendQualificationError, match="not qualified"):
        require_backend("totally-unrecognized", GenerationConfig(1))


def test_int8_real_execution_evidence_has_ids_metrics_and_resource_deltas(runtime):
    evidence = qualify_backend(runtime, "A", GenerationConfig(2))
    assert evidence.accepted
    assert evidence.schema.endswith(".v1")
    assert evidence.backend == CANDIDATE
    assert evidence.model_spec_sha256 == runtime.model_spec_sha256
    assert evidence.reference_model_sha256 == runtime.model_weights_sha256
    assert evidence.tokenizer_sha256 == runtime.tokenizer_sha256
    assert evidence.backend_identity_sha256
    assert evidence.reference_token_ids == evidence.candidate_token_ids
    assert evidence.logit_max_abs_delta >= 0
    assert evidence.logit_mean_abs_delta >= 0
    assert evidence.reference_seconds > 0 and evidence.candidate_seconds > 0
    assert evidence.reference_state_bytes > 0
    assert evidence.candidate_state_bytes_estimate > 0
    assert evidence.to_dict()["accepted"] is True


def test_repeated_conversions_preserve_identity_and_token_parity(runtime):
    a = qualify_backend(runtime, "hello", GenerationConfig(1))
    b = qualify_backend(runtime, "hello", GenerationConfig(1))
    assert a.backend_identity_sha256 == b.backend_identity_sha256
    assert a.reference_token_ids == b.reference_token_ids
    assert a.candidate_token_ids == b.candidate_token_ids
    assert a.policy_sha256 == b.policy_sha256
    assert a.prompt_sha256 == b.prompt_sha256


def test_strict_zero_tolerance_refuses_candidate_without_faking_pass(runtime):
    result = qualify_backend(runtime, "A", GenerationConfig(1), policy=ParityPolicy(0, 0))
    assert not result.accepted
    assert result.logit_max_abs_delta > 0
    assert result.reference_model_sha256 == runtime.model_weights_sha256


@pytest.mark.parametrize("data", [
    {"max_logit_abs_delta": float("nan")},
    {"max_logit_abs_delta": float("inf")},
    {"max_logit_mean_delta": -1},
    {"max_logit_abs_delta": True},
    {"require_greedy_token_parity": "no"},
])
def test_policy_rejects_invalid_or_nonfinite_tolerance(data):
    with pytest.raises(BackendQualificationError):
        ParityPolicy(**data)


def test_sampling_cannot_be_approximated_as_greedy(runtime):
    with pytest.raises(BackendQualificationError, match="sampling unsupported"):
        qualify_backend(runtime, "A", GenerationConfig(1, strategy="sample", seed=10))


def test_training_mode_and_reference_weights_drift_refused(runtime):
    runtime.model.train()
    with pytest.raises(ValueError, match="eval mode"):
        qualify_backend(runtime, "A", GenerationConfig(1))
    runtime.model.eval()
    with torch.no_grad():
        next(runtime.model.parameters()).reshape(-1)[0].add_(0.1)
    with pytest.raises(ValueError, match="drift"):
        qualify_backend(runtime, "A", GenerationConfig(1))


def test_nonfinite_model_state_not_masked_by_quantization(runtime):
    with torch.no_grad():
        next(runtime.model.parameters()).reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        qualify_backend(runtime, "A", GenerationConfig(1))


def test_bounded_prompt_and_context_reuse_canonical_validation(runtime):
    with pytest.raises(ValueError, match="nonempty"):
        qualify_backend(runtime, "", GenerationConfig(1))
    with pytest.raises(ValueError, match="context"):
        qualify_backend(runtime, "A", GenerationConfig(24))


def test_restart_with_same_weights_preserves_provenance(runtime):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        restored = TwelveSixDecoder(runtime.model.spec)
    restored.load_state_dict(runtime.model.state_dict())
    restored.eval()
    second = ReferenceInference(restored, ByteTokenizer())
    a = qualify_backend(runtime, "x", GenerationConfig(1))
    b = qualify_backend(second, "x", GenerationConfig(1))
    assert a.backend_identity_sha256 == b.backend_identity_sha256
    assert a.reference_token_ids == b.reference_token_ids
    assert a.candidate_token_ids == b.candidate_token_ids


def test_candidate_never_mutates_canonical_weights(runtime):
    original = runtime.model_weights_sha256
    qualify_backend(runtime, "x", GenerationConfig(1))
    assert ReferenceInference(runtime.model, ByteTokenizer()).model_weights_sha256 == original
