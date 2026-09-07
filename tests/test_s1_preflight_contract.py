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
from twelve_six.training.s1_preflight import run_s1_numerical_preflight
from twelve_six.training.s1_preflight_contract import validate_s1_preflight_bundle

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "c" * 40


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
def bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    locked = _locked_environment()
    evidence = run_s1_numerical_preflight(
        ROOT,
        source_sha=SOURCE_SHA,
        locked_environment_evidence=locked,
        max_steps=1,
    )
    return evidence, locked


def test_bundle_requires_exact_same_source_lock(
    bundle: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    evidence, locked = bundle
    validate_s1_preflight_bundle(evidence, locked)
    stale = _locked_environment("d" * 40)
    with pytest.raises(ValueError, match="source SHA mismatch"):
        validate_s1_preflight_bundle(evidence, stale)


def test_bundle_rejects_controlled_fixture_identity_drift(
    bundle: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    evidence, locked = bundle
    tampered = copy.deepcopy(evidence)
    tampered["identity"]["tokenizer_vocab_sha256"] = "f" * 64
    tampered["identity"]["fixture"]["tokenizer_vocab_sha256"] = "f" * 64
    tampered["identity_sha256"] = _canonical_hash(tampered["identity"])
    tampered.pop("evidence_sha256")
    tampered["evidence_sha256"] = _canonical_hash(tampered)
    with pytest.raises(ValueError, match="controlled fixture identity mismatch"):
        validate_s1_preflight_bundle(tampered, locked)


def test_bundle_rejects_precision_label_substitution(
    bundle: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    evidence, locked = bundle
    tampered = copy.deepcopy(evidence)
    tampered["profiles"]["bf16"]["precision"] = "fp32"
    tampered.pop("evidence_sha256")
    tampered["evidence_sha256"] = _canonical_hash(tampered)
    with pytest.raises(ValueError, match="bf16 profile label mismatch"):
        validate_s1_preflight_bundle(tampered, locked)
