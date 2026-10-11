"""Three physical source-family split is whole-source and nonrelease."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_three_family_split_probe_v1 as probe

ROOT = Path(__file__).resolve().parents[1]


def test_five_real_source_clusters_cannot_cross_holdout(tmp_path: Path) -> None:
    result = probe.stage(ROOT, tmp_path / "real-split")
    assert result["decision"] == (
        "THREE_REAL_FAMILY_SOURCE_SPLIT_MECHANICS_NOT_ADMISSION"
    )
    assert result["canonical_source_family_count"] == 3
    assert result["physical_document_count"] == 15
    assert result["whole_source_cluster_count"] == 5
    assert len(result["physical_source_cluster_assignments"]) == 5
    assert set(result["physical_source_cluster_assignments"].values()) == {
        "train", "validation", "test",
    }
    assert result["cluster_leakage_count"] == 0
    assert result["train_source_family_count"] == 3
    assert result["train_record_count"] == 13
    assert result["validation_record_count"] == 1
    assert result["test_record_count"] == 1
    assert result["physical_s10_train_all_three_families"] is True
    assert result["validation_test_language_balanced"] is False
    assert len(result["incumbent_s10_split_manifest_sha256"]) == 64
    assert len(result["s10_split_manifest_sha256"]) == 64
    assert len(result["s15_split_policy_sha256"]) == 64
    assert result["s15_split_policy_sha256"] != result["s10_split_manifest_sha256"]
    assert result["s10_split_manifest_sha256"] != (
        result["incumbent_s10_split_manifest_sha256"]
    )
    assert result["physical_source_cluster_assignments"] == {
        "en.public-domain.pride-and-prejudice": "train",
        "en.public-domain.frankenstein": "validation",
        "en.public-domain.alice-in-wonderland": "test",
        "next100-028-php-doc-uk": "train",
        "next100-030-rustbook-ua-oer": "train",
    }
    assert result["validation_source_family_count"] >= 1
    assert result["test_source_family_count"] >= 1
    assert result["production_train_mixture_admitted"] is False
    assert set(result["source_family_roles"]) == {
        "train", "validation", "test",
    }
    assert result["train_record_count"] > 0
    assert result["validation_record_count"] > 0
    assert result["test_record_count"] > 0
    assert sum(result[k] for k in ("train_record_count",
                                  "validation_record_count",
                                  "test_record_count")) == 15
    assert result["physical_s9_admitted"] is False
    assert result["training_corpus_authorized"] is False
    assert result["tokenizer_fit_authorized"] is False
    assert result["production_release_authorized"] is False
    assert result["terminal_done"] is False
    assert json.loads((tmp_path / "real-split" / probe.OUTPUT).read_bytes()) == result
    assert probe.stage(ROOT, tmp_path / "real-split") == result


@pytest.mark.parametrize("change", [
    "missing_family", "missing_doc", "leaked", "source_promotion",
])
def test_bad_three_family_boundary_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    mock = {
        "source_family_count": 3,
        "document_identity_count": 15,
        "global_exact_duplicates": 0,
        "same_modality_cross_family_near_pairs": 0,
        "g06_rejected_record_ids": [],
        "data232_excluded_record_count": 0,
        "data232_quarantined_source_family_count": 0,
        "training_corpus_authorized": False,
        "physical_s3_s9_multi_family_admitted": False,
    }
    if change == "missing_family":
        mock["source_family_count"] = 2
    elif change == "missing_doc":
        mock["document_identity_count"] = 14
    elif change == "leaked":
        mock["g06_rejected_record_ids"] = ["bad-record"]
    else:
        mock["training_corpus_authorized"] = True
    monkeypatch.setattr(probe.combined, "inspect", lambda _: mock)
    with pytest.raises(probe.ThreeFamilySplitDenied, match="privacy/fixture-clean"):
        probe.inspect(tmp_path)


def test_changed_split_receipt_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = {"decision": "THREE_REAL_FAMILY_SOURCE_SPLIT_MECHANICS_NOT_ADMISSION",
            "production_release_authorized": False}
    monkeypatch.setattr(probe, "inspect", lambda _: fake)
    path = tmp_path / "candidate"
    probe.stage(tmp_path, path)
    (path / probe.OUTPUT).write_bytes(b'{"production_release_authorized":true}\n')
    with pytest.raises(probe.ThreeFamilySplitDenied, match="immutable"):
        probe.stage(tmp_path, path)


def test_split_publication_refuses_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dst = tmp_path / "real"
    dst.mkdir()
    link = tmp_path / "link"
    link.symlink_to(dst, target_is_directory=True)
    monkeypatch.setattr(probe, "inspect", lambda _: pytest.fail("source read"))
    with pytest.raises(probe.ThreeFamilySplitDenied, match="symlink"):
        probe.stage(tmp_path, link)
