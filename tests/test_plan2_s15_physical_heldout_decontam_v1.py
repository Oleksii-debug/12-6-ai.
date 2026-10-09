"""Physical source-level heldouts must be independently scanned and never promoted."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_physical_heldout_decontam_v1 as heldout

ROOT = Path(__file__).resolve().parents[1]


def test_real_three_document_holdout_metadata_only(tmp_path: Path) -> None:
    report = heldout.stage(ROOT, tmp_path / "proof")
    assert report["decision"] == "REAL_HELDOUT_SCANNED_COMPONENT_ONLY_NOT_RELEASE"
    assert report["training_span_count"] == 2
    assert report["heldout_span_count"] == 4
    assert report["document_family_count"] == 3
    assert report["canonical_source_family_count"] == 1
    assert report["reserved_role_identities"]["selection_validation"] != (
        report["reserved_role_identities"]["final_test"]
    )
    assert report["final_test_outcomes_read"] is False
    assert report["physical_s9_admitted"] is False
    assert report["training_corpus_authorized"] is False
    assert report["production_release_authorized"] is False
    assert report["terminal_done"] is False
    assert json.loads((tmp_path / "proof" / heldout.OUTPUT).read_bytes()) == report
    assert heldout.fingerprint({k: v for k, v in report.items()
                                if k != "manifest_sha256"}) == report["manifest_sha256"]


def _fake_sources() -> dict:
    return {
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "g06_rejected_record_ids": [],
        "data232_excluded_record_count": 0,
        "data232_quarantined_source_family_count": 0,
        "books": [{"source_id": x} for x in ("a", "b", "c")],
    }


def _fake_splits() -> dict:
    return {
        "production_release_authorized": False,
        "physical_s9_admitted": False,
        "assigned_document_splits": {"a": "train", "b": "validation",
                                     "c": "test"},
    }


@pytest.mark.parametrize("defect", [
    "source_release", "privacy", "data232", "quarantine", "split_release",
    "split_s9_admission", "partition_identity",
])
def test_forged_upstream_source_or_split_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    source, split = _fake_sources(), _fake_splits()
    if defect == "source_release":
        source["production_release_authorized"] = True
    elif defect == "privacy":
        source["g06_rejected_record_ids"] = ["a:r00000000"]
    elif defect == "data232":
        source["data232_excluded_record_count"] = 1
    elif defect == "quarantine":
        source["data232_quarantined_source_family_count"] = 1
    elif defect == "split_release":
        split["production_release_authorized"] = True
    elif defect == "split_s9_admission":
        split["physical_s9_admitted"] = True
    elif defect == "partition_identity":
        split["assigned_document_splits"]["c"] = "train"
    monkeypatch.setattr(heldout.books, "inspect", lambda _root: source)
    monkeypatch.setattr(heldout.split_probe, "inspect", lambda _root: split)
    with pytest.raises(heldout.PhysicalHeldoutDenied):
        heldout.inspect(tmp_path)


def test_publication_rejects_changed_receipt_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {"decision": "REAL_HELDOUT_SCANNED_COMPONENT_ONLY_NOT_RELEASE",
               "production_release_authorized": False}
    monkeypatch.setattr(heldout, "inspect", lambda _: receipt)
    out = tmp_path / "out"
    assert heldout.stage(tmp_path, out) == receipt
    (out / heldout.OUTPUT).write_bytes(b'{"production_release_authorized":true}\n')
    with pytest.raises(heldout.PhysicalHeldoutDenied, match="immutable"):
        heldout.stage(tmp_path, out)


def test_symlink_publication_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest = tmp_path / "real"
    dest.mkdir()
    link = tmp_path / "link"
    link.symlink_to(dest, target_is_directory=True)
    monkeypatch.setattr(heldout, "inspect", lambda _: pytest.fail("unsafe audit"))
    with pytest.raises(heldout.PhysicalHeldoutDenied, match="symlink"):
        heldout.stage(tmp_path, link)
