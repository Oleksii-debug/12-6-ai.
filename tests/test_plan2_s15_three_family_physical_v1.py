"""S15 combined real-book/UA-source audit is exact and never a release."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_three_family_physical_v1 as three

ROOT = Path(__file__).resolve().parents[1]


def test_three_family_real_physical_source_reconstruction(tmp_path: Path) -> None:
    receipt = three.stage(ROOT, tmp_path / "real-three")
    assert receipt["source_family_count"] == 3
    assert receipt["document_identity_count"] == 15
    assert receipt["total_physical_normalized_bytes"] == 1_314_156
    assert len(receipt["families"]) == 3
    assert len({r["record_id"] for r in receipt["source_members"]}) == 15
    assert len({r["document_family"] for r in receipt["source_members"]}) == 15
    assert receipt["global_exact_duplicates"] == 0
    assert receipt["same_modality_cross_family_near_pairs"] == 0
    assert receipt["cross_language_semantic_dedup_established"] is False
    assert receipt["physical_s3_s9_multi_family_admitted"] is False
    assert receipt["physical_production_split_established"] is False
    assert receipt["real_evaluation_custody_established"] is False
    assert receipt["production_tokenizer_fit_authorized"] is False
    assert receipt["training_corpus_authorized"] is False
    assert receipt["production_release_authorized"] is False
    assert receipt["terminal_done"] is False
    assert receipt["physical_candidate_fixture_eval_clean"] == (
        not receipt["g06_rejected_record_ids"]
        and receipt["data232_excluded_record_count"] == 0
        and receipt["data232_quarantined_source_family_count"] == 0
    )
    assert json.loads((tmp_path / "real-three" / three.OUTPUT).read_bytes()) == receipt
    assert three.stage(ROOT, tmp_path / "real-three") == receipt


@pytest.mark.parametrize("tamper", [
    "gutenberg_release", "gutenberg_source_count", "ua_release",
    "ua_source_count", "ua_member_count",
])
def test_authority_promotion_or_family_count_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    en = {"physical_source_families": 1,
          "physical_document_families": 3,
          "g06_rejected_record_ids": [],
          "production_release_authorized": False}
    ua = {"physical_family_count": 2, "physical_record_count": 12,
          "training_corpus_authorized": False,
          "production_release_authorized": False}
    if tamper == "gutenberg_release":
        en["production_release_authorized"] = True
    elif tamper == "gutenberg_source_count":
        en["physical_source_families"] = 3
    elif tamper == "ua_release":
        ua["production_release_authorized"] = True
    elif tamper == "ua_source_count":
        ua["physical_family_count"] = 3
    elif tamper == "ua_member_count":
        ua["physical_record_count"] = 11
    monkeypatch.setattr(three.books, "inspect", lambda _: en)
    monkeypatch.setattr(three.d03, "inspect", lambda _: ua)
    with pytest.raises(three.ThreeFamilyDenied, match="family source evidence"):
        three.inspect(tmp_path)


def test_restarted_publication_refuses_changed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"schema_version": three.SCHEMA,
              "production_release_authorized": False}
    monkeypatch.setattr(three, "inspect", lambda _: report)
    destination = tmp_path / "report"
    assert three.stage(tmp_path, destination) == report
    (destination / three.OUTPUT).write_bytes(
        b'{"production_release_authorized":true}\n'
    )
    with pytest.raises(three.ThreeFamilyDenied, match="immutable"):
        three.stage(tmp_path, destination)


def test_publication_symlink_denied(tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(three, "inspect", lambda _: pytest.fail("unsafe source read"))
    with pytest.raises(three.ThreeFamilyDenied, match="symlink"):
        three.stage(tmp_path, link)
