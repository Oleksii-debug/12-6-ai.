"""Plan 4 S5: real INT8 backend and fail-closed adversarial qualification."""
from __future__ import annotations

import copy
import json

import pytest
import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder
from tools.inference_runtime import _model_weights_sha
from tools.plan4_backend_parity import (
    BACKEND,
    DENIAL,
    FEATURE_MATRIX,
    STATUS,
    BackendParityError,
    qualify_cpu_dynamic_int8,
    verify_report,
)


@pytest.fixture
def model():
    spec = ModelSpec(
        schema_version=1, vocab_size=256, max_seq_len=32, d_model=16,
        n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8, d_ff=32,
        rope_rotary_dim=8, tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model = TwelveSixDecoder(spec)
    return model.eval()


def test_real_dynamic_int8_quality_performance_resources_and_provenance(model):
    before = _model_weights_sha(model)
    report = qualify_cpu_dynamic_int8(model)
    verify_report(report)
    assert before == _model_weights_sha(model) == report["reference_weights_sha256"]
    assert report["backend"] == BACKEND and report["status"] == STATUS
    assert report["production_promotion"] == DENIAL
    assert report["max_abs_logit_error"] < 0.05
    assert report["max_cross_entropy_increase"] <= 0.05
    assert report["greedy_match_rate"] == 1.0
    assert report["converted_linear_count"] == 8
    assert report["optimized_tensor_bytes"] < report["reference_tensor_bytes"]
    assert report["reference_median_ns"] > 0 and report["optimized_median_ns"] > 0
    assert report["resource_unit"] == "TENSOR_PAYLOAD_BYTES_NOT_RSS"
    assert report["feature_matrix"] == FEATURE_MATRIX
    for backend in ("gguf-llama.cpp", "vllm"):
        assert report["feature_matrix"][backend]["availability"] == "UNSUPPORTED"


def test_reconstructed_restart_preserves_exact_semantics_not_wall_clock(model):
    first = qualify_cpu_dynamic_int8(model)
    second = TwelveSixDecoder(model.spec)
    second.load_state_dict(model.state_dict())
    second.eval()
    restored = qualify_cpu_dynamic_int8(second)
    for key in (
        "model_spec_sha256", "reference_weights_sha256",
        "candidate_runtime_sha256", "probe_sha256", "max_abs_logit_error",
        "max_cross_entropy_increase", "greedy_match_rate", "feature_matrix",
    ):
        assert first[key] == restored[key]
    verify_report(json.loads(json.dumps(restored)))


@pytest.mark.parametrize("backend", ["gguf-llama.cpp", "vllm", "unknown", ""])
def test_external_unverified_backend_is_never_promoted(model, backend):
    with pytest.raises(BackendParityError, match="unsupported"):
        qualify_cpu_dynamic_int8(model, backend=backend)


def test_tied_embedding_requires_proof_not_approximation():
    spec = ModelSpec(
        schema_version=1, vocab_size=256, max_seq_len=32, d_model=16,
        n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8, d_ff=32,
        rope_rotary_dim=8, tie_word_embeddings=True,
    )
    source = TwelveSixDecoder(spec).eval()
    with pytest.raises(BackendParityError, match="tied embeddings"):
        qualify_cpu_dynamic_int8(source)


def test_training_mode_and_nonfinite_state_fail_closed(model):
    model.train()
    with pytest.raises(BackendParityError, match="eval mode"):
        qualify_cpu_dynamic_int8(model)
    model.eval()
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = float("nan")
    with pytest.raises(BackendParityError, match="nonfinite"):
        qualify_cpu_dynamic_int8(model)


@pytest.mark.parametrize("arguments", [
    {"max_abs_logit_error": 0.5},
    {"max_abs_logit_error": float("nan")},
    {"max_abs_logit_error": True},
    {"max_cross_entropy_increase": 0.5},
    {"max_cross_entropy_increase": -0.01},
    {"min_greedy_match": 0.0},
    {"min_greedy_match": True},
    {"probes": ((256,),)},
    {"probes": ((1, True),)},
    {"probes": ((1,) * 40,)},
    {"probes": ("untyped",)},
    {"probes": ()},
])
def test_invalid_tolerance_or_probes_cannot_widen_policy(model, arguments):
    with pytest.raises(BackendParityError):
        qualify_cpu_dynamic_int8(model, **arguments)


def test_real_int8_exceeds_impossibly_strict_bound_and_is_rejected(model):
    with pytest.raises(BackendParityError, match="parity tolerance"):
        qualify_cpu_dynamic_int8(model, max_abs_logit_error=1e-9)


@pytest.mark.parametrize("key,value", [
    ("status", "PRODUCTION_READY"),
    ("production_promotion", "PROMOTED"),
    ("backend", "vllm"),
    ("probe_kind", "FINAL_TEST"),
    ("greedy_match_rate", 0.9),
    ("reference_tensor_bytes", True),
    ("optimized_median_ns", -1),
    ("feature_matrix", {"vllm": {"availability": "PASS"}}),
    ("unknown_field", "forged"),
])
def test_even_rehashed_forged_production_report_is_rejected(model, key, value):
    from tools.plan4_backend_parity import _digest

    report = qualify_cpu_dynamic_int8(model)
    forged = copy.deepcopy(report)
    forged[key] = value
    forged["report_sha256"] = _digest({
        k: v for k, v in forged.items() if k != "report_sha256"
    })
    with pytest.raises(BackendParityError):
        verify_report(forged)


def test_report_digest_integrity_and_type(model):
    report = qualify_cpu_dynamic_int8(model)
    report["report_sha256"] = "0" * 64
    with pytest.raises(BackendParityError, match="tampered"):
        verify_report(report)
    with pytest.raises(BackendParityError, match="mapping"):
        verify_report(None)
