from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "tools/verify_d03_nomis_provenance_root_no_reentry_v1.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("_d03_nomis_root", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_root(verifier):
    root = verifier.prohibited_root()
    root.update(
        {
            "source_pr": 9001,
            "authority_identity_sha256": "1" * 64,
            "source_head_sha": "2" * 40,
            "upstream_repo": "example/clean-source",
            "upstream_commit": "3" * 40,
            "upstream_data_object_git_blob_sha1": "4" * 40,
            "upstream_data_object_path": "data/clean.json",
            "source_filter_exact": ["CleanSource"],
        }
    )
    return root


def test_historical_root_binds_exact_physical_clean_successor() -> None:
    verifier = _load_verifier()
    result = verifier.verify_provenance_root_authority(ROOT)
    assert result["source_authority_identity_sha256"] == verifier.PR462_AUTHORITY
    assert result["source_pr"] == 462
    assert result["blocked_normalized_sha256"] == verifier.BLOCKED_NORMALIZED_SHA256
    assert result["blocked_normalized_bytes"] == 1659
    assert result["physical_execution_head_sha"] == verifier.PHYSICAL_EXECUTION_HEAD
    assert result["release_head_sha"] == verifier.RELEASE_HEAD
    assert result["physical_removed_object_bound_to_prohibited_root"] is True
    assert result["changed_aliases_cannot_bypass_root"] is True
    assert result["changed_payload_cannot_bypass_root"] is True
    assert result["truth_boundary"] == verifier.TRUTH_BOUNDARY


def test_changed_aliases_and_payload_do_not_bypass_same_root() -> None:
    verifier = _load_verifier()
    candidate = {
        "source_id": "totally-renamed-source",
        "record_id": "totally-renamed-record",
        "family": "totally.renamed.family",
        "payload_sha256": "f" * 64,
        "payload_bytes": 999999,
        "provenance_roots": [verifier.prohibited_root()],
    }
    with pytest.raises(
        verifier.ProhibitedProvenanceRoot,
        match="prohibited PR462/Verba provenance root",
    ):
        verifier.candidate_is_admissible(candidate)


def test_self_resealed_authority_cannot_hide_same_verba_object() -> None:
    verifier = _load_verifier()
    root = verifier.prohibited_root()
    root["authority_identity_sha256"] = "a" * 64
    root["source_pr"] = 9999
    root["source_head_sha"] = "b" * 40
    with pytest.raises(verifier.ProhibitedProvenanceRoot, match="upstream_object"):
        verifier.assert_root_admissible(root)


def test_conflicting_pr462_claim_fails_closed() -> None:
    verifier = _load_verifier()
    root = _clean_root(verifier)
    root["source_pr"] = 462
    with pytest.raises(verifier.ProhibitedProvenanceRoot, match="source_pr"):
        verifier.assert_root_admissible(root)


def test_verba_filter_anchor_survives_other_reseals() -> None:
    verifier = _load_verifier()
    root = _clean_root(verifier)
    root["upstream_repo"] = verifier.UPSTREAM_REPO
    root["source_filter_exact"] = ["Nomis1864"]
    with pytest.raises(
        verifier.ProhibitedProvenanceRoot,
        match="source_filter_on_verba",
    ):
        verifier.assert_root_admissible(root)


def test_missing_or_duplicate_provenance_fails_closed() -> None:
    verifier = _load_verifier()
    with pytest.raises(
        verifier.ProvenanceAuthorityError,
        match="exactly one source provenance root",
    ):
        verifier.require_single_candidate_provenance([])
    clean = _clean_root(verifier)
    with pytest.raises(
        verifier.ProvenanceAuthorityError,
        match="exactly one source provenance root",
    ):
        verifier.require_single_candidate_provenance([clean, copy.deepcopy(clean)])


def test_missing_root_field_and_bool_pr_alias_fail_closed() -> None:
    verifier = _load_verifier()
    root = _clean_root(verifier)
    root.pop("upstream_commit")
    with pytest.raises(verifier.ProvenanceAuthorityError, match="key-set drift"):
        verifier.assert_root_admissible(root)

    root = _clean_root(verifier)
    root["source_pr"] = True
    with pytest.raises(verifier.ProvenanceAuthorityError, match="exact int"):
        verifier.assert_root_admissible(root)


def test_unrelated_clean_root_remains_admissible() -> None:
    verifier = _load_verifier()
    clean = _clean_root(verifier)
    assert verifier.require_single_candidate_provenance([clean]) == clean
    assert verifier.candidate_is_admissible({"provenance_roots": [clean]}) is True
