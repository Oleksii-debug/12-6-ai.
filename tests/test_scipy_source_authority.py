from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.permissive_repo_source_authority import (
    SourceAuthorityError,
    authority_identity,
    load_and_validate_source_authority,
    validate_source_authority,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_PATH = ROOT / "configs" / "data" / "scipy_v118_source_authority_v1.json"


def _document() -> dict:
    return json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))


def _resign(document: dict) -> dict:
    document["authority_sha256"] = authority_identity(document)
    return document


def test_authority_is_deterministic_and_candidate_only() -> None:
    summary = load_and_validate_source_authority(AUTHORITY_PATH)
    assert summary == {
        "authority_id": "scipy-v1.18.0-bounded-first-party-v1",
        "authority_sha256": "a9ddc66c826f32299cb9e69aa64d3e7e7526869391e3e2ab3162abf43cf886a9",
        "files": 2,
        "candidate_raw_bytes": 78307,
        "canonical_credit_bytes": 0,
        "ready_for_corpus_credit": False,
    }
    assert authority_identity(_document()) == summary["authority_sha256"]


def test_tampered_manifest_identity_fails() -> None:
    document = _document()
    document["capacity"]["candidate_raw_bytes"] += 1
    with pytest.raises(SourceAuthorityError, match="authority_sha256 mismatch"):
        validate_source_authority(document)


def test_eval_permission_cannot_be_enabled_even_when_resigned() -> None:
    document = _resign(copy.deepcopy(_document()))
    document["purpose"]["evaluation_allowed"] = True
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="purpose boundary drift"):
        validate_source_authority(document)


def test_canonical_credit_cannot_be_claimed_pre_materialization() -> None:
    document = copy.deepcopy(_document())
    document["capacity"]["canonical_credit_bytes"] = 1
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="canonical credit must remain zero"):
        validate_source_authority(document)


def test_upstream_pin_drift_fails_closed() -> None:
    document = copy.deepcopy(_document())
    document["upstream"]["commit_sha"] = "0" * 40
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="upstream pin drift"):
        validate_source_authority(document)


def test_third_party_or_test_path_is_rejected() -> None:
    document = copy.deepcopy(_document())
    document["allowlist"][0]["path"] = "scipy/optimize/tests/example.py"
    commit = document["upstream"]["commit_sha"]
    document["allowlist"][0]["raw_url"] = (
        f"https://raw.githubusercontent.com/scipy/scipy/{commit}/scipy/optimize/tests/example.py"
    )
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="forbidden provenance path"):
        validate_source_authority(document)


def test_duplicate_allowlist_path_is_rejected() -> None:
    document = copy.deepcopy(_document())
    document["allowlist"][1] = copy.deepcopy(document["allowlist"][0])
    document["capacity"]["candidate_raw_bytes"] = 2 * document["allowlist"][0]["raw_bytes"]
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="duplicate allowlist path"):
        validate_source_authority(document)


def test_candidate_byte_arithmetic_is_bound() -> None:
    document = copy.deepcopy(_document())
    document["capacity"]["candidate_raw_bytes"] += 7
    _resign(document)
    with pytest.raises(SourceAuthorityError, match="candidate byte arithmetic drift"):
        validate_source_authority(document)
