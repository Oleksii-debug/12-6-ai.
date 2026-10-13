"""Physical Plan2 S15 train partition must never persist holdout text."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_three_family_train_materialization_v1 as physical

ROOT = Path(__file__).resolve().parents[1]


def test_real_train_partition_persists_only_train(tmp_path: Path) -> None:
    report = physical.stage(ROOT, tmp_path / "partition")
    assert report["source_families_total"] == 3
    assert report["source_clusters_total"] == 5
    assert report["train_source_family_count"] == 3
    assert report["train_document_count"] == 13
    assert report["physical_train_bytes"] == 740_518
    assert report["heldout_document_count"] == 2
    assert report["train_document_count"] > 0
    assert report["heldout_document_count"] > 0
    assert report["train_document_count"] + report["heldout_document_count"] == 15
    assert report["total_accounted_source_bytes"] == 1_314_156
    assert report["physical_train_bytes"] > 0
    assert report["physical_train_bytes"] < report["total_accounted_source_bytes"]
    assert report["heldout_plaintext_materialized"] is False
    assert report["physical_s9_admitted"] is False
    assert report["tokenizer_fit_authorized"] is False
    assert report["training_corpus_authorized"] is False
    assert report["production_release_authorized"] is False
    assert report["terminal_done"] is False
    assert len(report["s10_split_manifest_sha256"]) == 64
    assert all("text" not in entry and "path" not in entry
               and "record_id" not in entry
               for entry in report["heldout_members_hash_only"])
    saved = tmp_path / "partition"
    members = sorted(saved.rglob("*.utf8"))
    assert len(members) == report["train_document_count"]
    assert all(x.parent == saved / "train" for x in members)
    assert sum(x.stat().st_size for x in members) == report["physical_train_bytes"]
    assert all(x.read_bytes() for x in members)
    assert json.loads((saved / physical.OUTPUT).read_bytes()) == report
    assert physical.stage(ROOT, saved) == report


@pytest.mark.parametrize("key", [
    "source_family_count", "document_identity_count", "g06_rejected_record_ids",
    "data232_excluded_record_count", "physical_s3_s9_multi_family_admitted",
])
def test_upstream_promotions_or_dirty_input_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str,
) -> None:
    proof = {
        "source_family_count": 3,
        "document_identity_count": 15,
        "global_exact_duplicates": 0,
        "g06_rejected_record_ids": [],
        "data232_excluded_record_count": 0,
        "data232_quarantined_source_family_count": 0,
        "physical_s3_s9_multi_family_admitted": False,
        "production_release_authorized": False,
    }
    proof[key] = (
        ["excluded"] if key == "g06_rejected_record_ids"
        else True if key == "physical_s3_s9_multi_family_admitted"
        else 1 if key == "data232_excluded_record_count"
        else 2 if key == "source_family_count" else 14
    )
    monkeypatch.setattr(physical.combined, "inspect", lambda _: proof)
    with pytest.raises(physical.TrainPartitionDenied, match="privacy"):
        physical.build(tmp_path)


def test_immutable_train_partition_rejects_tampered_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"physical-train-only\n"
    proof = {
        "schema_version": physical.SCHEMA,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    name = "train/one.utf8"
    monkeypatch.setattr(physical, "build", lambda _: (proof, {name: source}))
    destination = tmp_path / "partition"
    physical.stage(tmp_path, destination)
    (destination / name).write_bytes(b"forged\n")
    with pytest.raises(physical.TrainPartitionDenied, match="immutable"):
        physical.stage(tmp_path, destination)


def test_heldout_extra_member_denied_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = {"schema_version": physical.SCHEMA,
             "production_release_authorized": False}
    monkeypatch.setattr(physical, "build",
                        lambda _: (proof, {"train/one.utf8": b"allowed\n"}))
    destination = tmp_path / "partition"
    physical.stage(tmp_path, destination)
    (destination / "final-test.txt").write_bytes(b"NEVER PUBLISH")
    with pytest.raises(physical.TrainPartitionDenied, match="unexpected"):
        physical.stage(tmp_path, destination)


def test_symlink_output_refuses_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = tmp_path / "real"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(physical, "build", lambda _: pytest.fail("unsafe read"))
    with pytest.raises(physical.TrainPartitionDenied, match="symlink"):
        physical.stage(tmp_path, link)
