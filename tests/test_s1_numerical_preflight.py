from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from twelve_six.training.s0_evidence_contract import (
    LOCK_INDEX_FILE_SHA256,
    LOCK_INDEX_PATH,
    LOCK_INDEX_SEMANTIC_SHA256,
    LOCK_PROFILE_FILE_SHA256,
    LOCK_PROFILE_ID,
    LOCK_PROFILE_MANIFEST_SHA256,
    PYTHON_VERSION,
)
from twelve_six.training.s1_preflight import (
    S1PreflightError,
    run_s1_numerical_preflight,
    validate_s1_numerical_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "a" * 40


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _locked_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "12-6.locked-environment-evidence.v1",
        "source_sha": SOURCE_SHA,
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
def evidence() -> dict[str, Any]:
    return run_s1_numerical_preflight(
        ROOT,
        source_sha=SOURCE_SHA,
        locked_environment_evidence=_locked_environment(),
        seed=1337,
        max_steps=1,
        batch_size=3,
    )


def _rehash(evidence: dict[str, Any]) -> None:
    evidence["identity_sha256"] = _canonical_hash(evidence["identity"])
    evidence.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = _canonical_hash(evidence)


def test_real_s1_engineering_candidate_executes_fp32_and_bf16(
    evidence: dict[str, Any],
) -> None:
    validate_s1_numerical_preflight(evidence)
    assert evidence["identity"]["parameter_count"] == 107_856
    assert evidence["identity"]["model_vocab_size"] == 512
    assert evidence["identity"]["fixture"]["tokenizer_vocab_size"] == 256
    assert evidence["identity"]["fixture"]["unused_model_vocab_rows"] == 256
    assert evidence["profiles"]["fp32"]["optimizer_steps"] == 1
    assert evidence["profiles"]["bf16"]["optimizer_steps"] == 1
    assert evidence["profiles"]["fp32"]["validation_optimized_tokens"] == 0
    assert evidence["profiles"]["bf16"]["validation_optimized_tokens"] == 0
    assert evidence["fp16_cpu_probe"]["status"] == "FAIL_CLOSED_AS_DESIGNED"


def test_preflight_rejects_validation_optimization(evidence: dict[str, Any]) -> None:
    tampered = copy.deepcopy(evidence)
    tampered["profiles"]["bf16"]["validation_optimized_tokens"] = 1
    _rehash(tampered)
    with pytest.raises(S1PreflightError, match="optimized validation"):
        validate_s1_numerical_preflight(tampered)


def test_preflight_rejects_model_identity_drift(evidence: dict[str, Any]) -> None:
    tampered = copy.deepcopy(evidence)
    tampered["identity"]["modelspec_sha256"] = "f" * 64
    _rehash(tampered)
    with pytest.raises(S1PreflightError, match="ModelSpec identity mismatch"):
        validate_s1_numerical_preflight(tampered)


def test_preflight_rejects_s1_freeze_or_promotion_claim(evidence: dict[str, Any]) -> None:
    tampered = copy.deepcopy(evidence)
    tampered["claims"]["s1_architecture_frozen"] = True
    tampered.pop("evidence_sha256", None)
    tampered["evidence_sha256"] = _canonical_hash(tampered)
    with pytest.raises(S1PreflightError, match="prohibited claim enabled"):
        validate_s1_numerical_preflight(tampered)


def test_preflight_rejects_stale_environment_source() -> None:
    locked = _locked_environment()
    locked["source_sha"] = "b" * 40
    locked.pop("evidence_sha256")
    locked["evidence_sha256"] = _canonical_hash(locked)
    with pytest.raises(ValueError, match="locked environment source SHA mismatch"):
        run_s1_numerical_preflight(
            ROOT,
            source_sha=SOURCE_SHA,
            locked_environment_evidence=locked,
            max_steps=1,
        )
