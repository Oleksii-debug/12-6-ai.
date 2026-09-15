from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "tools/verify_d03_nomis_free_release_authority_v1.py"
RELEASE_HEAD = "07754c5a1d61669061e608323ea35ddc093bb946"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("_d03_release_authority", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_free_release_binds_exact_physical_science() -> None:
    verifier = _load_verifier()
    result = verifier.verify_release_checkout(ROOT, RELEASE_HEAD)
    assert result["physical_execution_head_sha"] == verifier.PHYSICAL_EXECUTION_HEAD
    assert result["release_head_sha"] == RELEASE_HEAD
    assert result["temporary_execution_workflow_removed"] is True
    assert result["physical_science_bytes_unchanged"] is True
    assert result["physical_science_blob_count"] == 4
    assert result["truth_boundary"] == verifier.TRUTH_BOUNDARY


def test_release_delta_rejects_extra_product_mutation() -> None:
    verifier = _load_verifier()
    mutated = dict(verifier.EXPECTED_RELEASE_DELTA)
    mutated["tools/run_d03_nomis_free_clean_successor_v1.py"] = "M"
    with pytest.raises(verifier.ReleaseAuthorityError, match="release delta drift"):
        verifier.require_release_delta(mutated)


def test_release_delta_rejects_restored_temporary_workflow() -> None:
    verifier = _load_verifier()
    mutated = dict(verifier.EXPECTED_RELEASE_DELTA)
    mutated.pop(verifier.TEMP_WORKFLOW)
    with pytest.raises(verifier.ReleaseAuthorityError, match="release delta drift"):
        verifier.require_release_delta(mutated)


def test_release_verifier_rejects_coherently_changed_science_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier()
    path = "tools/run_d03_nomis_free_clean_successor_v1.py"
    mutated = dict(verifier.PHYSICAL_SCIENCE_BLOBS)
    mutated[path] = "0" * 40
    monkeypatch.setattr(verifier, "PHYSICAL_SCIENCE_BLOBS", mutated)
    with pytest.raises(verifier.ReleaseAuthorityError, match="physical science blob drift"):
        verifier.verify_release_checkout(ROOT, RELEASE_HEAD)


def test_physical_execution_head_is_not_mislabeled_as_release() -> None:
    verifier = _load_verifier()
    with pytest.raises(verifier.ReleaseAuthorityError, match="release delta drift"):
        verifier.verify_release_checkout(ROOT, verifier.PHYSICAL_EXECUTION_HEAD)
