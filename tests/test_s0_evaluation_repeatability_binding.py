from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import twelve_six.s0_evaluation_bundle as bundle_module
from twelve_six.s0_evaluation_bundle import (
    S0EvaluationBundleError,
    build_s0_evaluation_bundle,
)
from twelve_six.training.s0_evidence_contract import (
    DATASET_MANIFEST_SHA256,
    INIT_SPEC_SHA256,
    LOCK_INDEX_FILE_SHA256,
    MODEL_SPEC_SHA256,
    PACKING_CONFIG_SHA256,
    PARAMETER_COUNT,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "s0_complete_evidence.json"
_CANDIDATE_SHA = "a" * 40
_OTHER_SHA = "b" * 40


def _candidate_evidence() -> dict[str, object]:
    evidence = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    evidence["schema_version"] = "12-6.s0-real-candidate-evidence.v2"
    evidence["candidate"].update(
        {
            "modelspec_sha256": MODEL_SPEC_SHA256,
            "initspec_sha256": INIT_SPEC_SHA256,
            "parameter_count": PARAMETER_COUNT,
        }
    )
    evidence["tokenizer"].update(
        {
            "config_sha256": TOKENIZER_CONFIG_SHA256,
            "vocab_sha256": TOKENIZER_VOCAB_SHA256,
        }
    )
    evidence["dataset"]["manifest_sha256"] = DATASET_MANIFEST_SHA256
    evidence["checkpoint"].update(
        {
            "packing_sha256": PACKING_CONFIG_SHA256,
            "environment_lock_sha256": LOCK_INDEX_FILE_SHA256,
        }
    )
    evidence["provenance"] = {
        "repository": "Oleksii-debug/12-6-ai.",
        "checkout_head_sha": _CANDIDATE_SHA,
    }
    return evidence


def _repeatability_evidence() -> dict[str, object]:
    return {
        "schema_version": "12-6.s0-repeatability-evidence.v1",
        "identity": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": _CANDIDATE_SHA,
            "modelspec_sha256": MODEL_SPEC_SHA256,
            "initspec_sha256": INIT_SPEC_SHA256,
            "parameter_count": PARAMETER_COUNT,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "packing_config_sha256": PACKING_CONFIG_SHA256,
            "environment": {
                "lock_index_file_sha256": LOCK_INDEX_FILE_SHA256,
                "environment_evidence_sha256": "c" * 64,
            },
        },
        "proof": {
            "same_seed_exact_equivalence": True,
            "different_seed_initialization_diverges": True,
            "different_seed_training_diverges": True,
            "validation_optimized_tokens": 0,
        },
        "evidence_sha256": "d" * 64,
    }


@pytest.fixture(autouse=True)
def _trust_upstream_repeatability_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bundle_module,
        "validate_s0_repeatability_evidence",
        lambda _evidence: None,
    )


def test_bundle_binds_complete_quality_and_repeatability_without_promotion() -> None:
    bundle = build_s0_evaluation_bundle(
        _candidate_evidence(), _repeatability_evidence()
    )

    assert bundle["identity"]["candidate_sha"] == _CANDIDATE_SHA
    assert bundle["quality"]["overall_status"] == "PASS"
    assert bundle["quality"]["evaluation_complete"] is True
    assert bundle["quality"]["counts"] == {
        "PASS": 15,
        "FAIL": 0,
        "NOT_TESTED": 0,
    }
    assert bundle["repeatability"]["same_seed_exact_equivalence"] is True
    assert bundle["promotion_boundary"]["bundle_grants_promotion"] is False
    assert bundle["promotion_boundary"]["source_promotion_eligible"] is False
    assert len(bundle["bundle_sha256"]) == 64


def test_bundle_rejects_repeatability_from_stale_candidate_sha() -> None:
    repeatability = _repeatability_evidence()
    repeatability["identity"]["source_sha"] = _OTHER_SHA

    with pytest.raises(S0EvaluationBundleError, match="stale"):
        build_s0_evaluation_bundle(_candidate_evidence(), repeatability)


def test_bundle_rejects_models_contract_drift() -> None:
    repeatability = _repeatability_evidence()
    repeatability["identity"]["modelspec_sha256"] = "e" * 64

    with pytest.raises(S0EvaluationBundleError, match="modelspec_sha256"):
        build_s0_evaluation_bundle(_candidate_evidence(), repeatability)


def test_bundle_rejects_tokenizer_contract_drift() -> None:
    repeatability = _repeatability_evidence()
    repeatability["identity"]["tokenizer_config_sha256"] = "e" * 64

    with pytest.raises(S0EvaluationBundleError, match="tokenizer_config_sha256"):
        build_s0_evaluation_bundle(_candidate_evidence(), repeatability)


def test_bundle_rejects_environment_lock_drift() -> None:
    repeatability = _repeatability_evidence()
    repeatability["identity"]["environment"]["lock_index_file_sha256"] = "e" * 64

    with pytest.raises(S0EvaluationBundleError, match="environment_lock_file_sha256"):
        build_s0_evaluation_bundle(_candidate_evidence(), repeatability)


def test_bundle_rejects_incomplete_quality_evaluation() -> None:
    candidate = _candidate_evidence()
    candidate["generation_probes"] = []

    with pytest.raises(S0EvaluationBundleError, match="incomplete"):
        build_s0_evaluation_bundle(candidate, _repeatability_evidence())


def test_bundle_rejects_unproven_seed_causality() -> None:
    repeatability = _repeatability_evidence()
    repeatability["proof"]["different_seed_initialization_diverges"] = False

    with pytest.raises(S0EvaluationBundleError, match="initialization"):
        build_s0_evaluation_bundle(_candidate_evidence(), repeatability)


def test_bundle_hash_is_deterministic_and_candidate_bound() -> None:
    first = build_s0_evaluation_bundle(
        _candidate_evidence(), _repeatability_evidence()
    )
    second = build_s0_evaluation_bundle(
        copy.deepcopy(_candidate_evidence()),
        copy.deepcopy(_repeatability_evidence()),
    )
    assert first == second

    changed = _candidate_evidence()
    changed["metrics"]["train_loss_after"] = 2.4
    third = build_s0_evaluation_bundle(changed, _repeatability_evidence())
    assert third["bundle_sha256"] != first["bundle_sha256"]
