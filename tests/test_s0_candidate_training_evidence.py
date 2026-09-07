from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from twelve_six.training.s0_candidate_binding import bind_candidate_training_evidence
from twelve_six.training.s0_evidence import run_s0_training_evidence
from twelve_six.training.s0_evidence_contract import (
    LOCK_INDEX_FILE_SHA256,
    LOCK_INDEX_PATH,
    LOCK_INDEX_SEMANTIC_SHA256,
    LOCK_PROFILE_FILE_SHA256,
    LOCK_PROFILE_ID,
    LOCK_PROFILE_MANIFEST_SHA256,
    PYTHON_VERSION,
    SCHEMA_VERSION,
    S0EvidenceContractError,
    validate_locked_environment_evidence,
    validate_s0_training_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "1caa729c8efafc84e7a5c4b1f7295eb8dcdb5a8d"


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _locked_environment(source_sha: str = SOURCE_SHA) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "12-6.locked-environment-evidence.v1",
        "source_sha": source_sha,
        "profile_id": LOCK_PROFILE_ID,
        "python": {
            "implementation": "cpython",
            "requires_python": ">=3.11,<3.12",
            "version": PYTHON_VERSION,
        },
        "lock_index": {
            "path": LOCK_INDEX_PATH,
            "file_sha256": LOCK_INDEX_FILE_SHA256,
            "index_sha256": LOCK_INDEX_SEMANTIC_SHA256,
        },
        "lock_profile": {
            "manifest_sha256": LOCK_PROFILE_MANIFEST_SHA256,
            "file_sha256": LOCK_PROFILE_FILE_SHA256,
        },
        "wheel": {"filename": "test.whl", "sha256": "0" * 64},
        "installed_distributions": [],
        "installed_distributions_sha256": "0" * 64,
        "verification": {
            "committed_lock_validation": "PASS",
            "editable_install_import_cli": "PASS",
            "wheel_install_import_cli": "PASS",
            "repo_checks": "PASS",
        },
    }
    payload["evidence_sha256"] = _canonical_hash(payload)
    return payload


@pytest.fixture(scope="module")
def bound_evidence() -> dict[str, Any]:
    raw = run_s0_training_evidence(
        ROOT,
        source_sha=SOURCE_SHA,
        max_steps=3,
        batch_size=3,
    )
    return bind_candidate_training_evidence(raw, _locked_environment())


def test_candidate_binding_is_lock_and_source_bound(bound_evidence: dict[str, Any]) -> None:
    validate_s0_training_evidence(bound_evidence, require_locked_environment=True)
    assert bound_evidence["schema_version"] == SCHEMA_VERSION
    assert bound_evidence["identity"]["source_sha"] == SOURCE_SHA
    environment = bound_evidence["identity"]["environment"]
    assert environment["profile_id"] == LOCK_PROFILE_ID
    assert environment["lock_index_file_sha256"] == LOCK_INDEX_FILE_SHA256
    assert environment["lock_index_sha256"] == LOCK_INDEX_SEMANTIC_SHA256
    assert environment["environment_evidence_sha256"] == _locked_environment()["evidence_sha256"]
    assert bound_evidence["split_isolation"]["validation_optimized_tokens"] == 0
    assert bound_evidence["training"]["optimized_tokens"] > 0


def test_candidate_contract_rejects_validation_optimization(
    bound_evidence: dict[str, Any],
) -> None:
    tampered = copy.deepcopy(bound_evidence)
    tampered["split_isolation"]["validation_optimized_tokens"] = 1
    tampered.pop("evidence_sha256")
    tampered["evidence_sha256"] = _canonical_hash(tampered)
    with pytest.raises(S0EvidenceContractError, match="validation tokens were optimized"):
        validate_s0_training_evidence(tampered)


def test_candidate_contract_rejects_metric_tamper_even_with_old_hash(
    bound_evidence: dict[str, Any],
) -> None:
    tampered = copy.deepcopy(bound_evidence)
    tampered["training"]["gradient_norm_max"] += 1.0
    with pytest.raises(S0EvidenceContractError, match="self-hash mismatch"):
        validate_s0_training_evidence(tampered)


def test_locked_environment_rejects_stale_source_and_failed_repo_checks() -> None:
    with pytest.raises(S0EvidenceContractError, match="source SHA mismatch"):
        validate_locked_environment_evidence(_locked_environment(), source_sha="0" * 40)

    failed = _locked_environment()
    failed["verification"]["repo_checks"] = "NOT_RUN"
    failed.pop("evidence_sha256")
    failed["evidence_sha256"] = _canonical_hash(failed)
    with pytest.raises(S0EvidenceContractError, match="repo_checks is not PASS"):
        validate_locked_environment_evidence(failed, source_sha=SOURCE_SHA)


def test_candidate_contract_rejects_environment_binding_tamper(
    bound_evidence: dict[str, Any],
) -> None:
    tampered = copy.deepcopy(bound_evidence)
    tampered["identity"]["environment"]["lock_index_sha256"] = "f" * 64
    tampered["identity_sha256"] = _canonical_hash(tampered["identity"])
    tampered.pop("evidence_sha256")
    tampered["evidence_sha256"] = _canonical_hash(tampered)
    with pytest.raises(S0EvidenceContractError, match="training lock_index_sha256 mismatch"):
        validate_s0_training_evidence(tampered)
