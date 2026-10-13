"""Physically bound S16 language split refusals, integrity and replay."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools import plan2_s15_three_family_split_probe_v1 as splits
from tools.plan2_s16_physical_split_gate_v1 import (
    OUTPUT, PhysicalSplitDenied, analyze_members, inspect, stage,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def physical_audit() -> dict:
    return inspect(ROOT)


def test_real_physical_splits_are_not_multilingual_holdouts(physical_audit: dict):
    x = physical_audit
    assert x["physical_sources"] == 5
    assert x["physical_families"] == 3
    assert x["physical_records"] == 15
    assert x["record_counts_by_role_language"]["train"] == {"uk": 12, "en": 1}
    assert x["record_counts_by_role_language"]["validation"] == {"uk": 0, "en": 1}
    assert x["record_counts_by_role_language"]["test"] == {"uk": 0, "en": 1}
    assert x["every_language_has_all_roles"] is False
    assert x["strict_source_family_isolation"] is False
    assert "MISSING_UK_VALIDATION" in x["blocking_gates"]
    assert "MISSING_UK_TEST" in x["blocking_gates"]
    assert "SOURCE_FAMILY_CROSSES_TRAIN_VALIDATION_TEST" in x["blocking_gates"]
    assert x["all_release_gates_passed"] is False
    for key in ("s3_s9_corpus_admitted", "training_authorized",
                "evaluation_authorized", "tokenizer_fit_authorized",
                "production_release_authorized", "plan9_binding_authorized",
                "terminal_done"):
        assert x[key] is False


def test_verified_physical_receipt_restart_and_clean_rebuild(tmp_path: Path, physical_audit: dict):
    a = stage(ROOT, tmp_path / "same")
    b = stage(ROOT, tmp_path / "same")
    c = stage(ROOT, tmp_path / "clean")
    assert a == b == c == physical_audit
    assert (tmp_path / "same" / OUTPUT).read_bytes() == (
        tmp_path / "clean" / OUTPUT
    ).read_bytes()


def test_immutable_receipt_tamper_denied(tmp_path: Path):
    location = tmp_path / "receipt"
    stage(ROOT, location)
    (location / OUTPUT).write_bytes(b'{"terminal_done":true}\n')
    with pytest.raises(PhysicalSplitDenied, match="immutable"):
        stage(ROOT, location)


def test_symlinked_publication_denied(tmp_path: Path):
    original = tmp_path / "source"
    original.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(original, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlink unavailable")
    with pytest.raises(PhysicalSplitDenied, match="unsafe"):
        stage(ROOT, alias)


def test_source_role_forgery_denied(monkeypatch: pytest.MonkeyPatch):
    original = splits.inspect(ROOT)
    forged = copy.deepcopy(original)
    source = next(iter(forged["physical_source_cluster_assignments"]))
    forged["physical_source_cluster_assignments"][source] = "test"
    monkeypatch.setattr(splits, "inspect", lambda _root: forged)
    with pytest.raises(PhysicalSplitDenied, match="split lineage checksum drift"):
        inspect(ROOT)


def test_positive_looking_forged_uk_holdout_still_denied():
    members = [
        {"record_id": "r1", "source_id": "ua-train", "source_family": "ua1",
         "modality": "uk", "normalized_bytes": 10},
        {"record_id": "r2", "source_id": "en-train", "source_family": "en1",
         "modality": "en", "normalized_bytes": 10},
        {"record_id": "r3", "source_id": "ua-val", "source_family": "ua1",
         "modality": "uk", "normalized_bytes": 10},
        {"record_id": "r4", "source_id": "en-val", "source_family": "en2",
         "modality": "en", "normalized_bytes": 10},
        {"record_id": "r5", "source_id": "ua-test", "source_family": "ua1",
         "modality": "uk", "normalized_bytes": 10},
        {"record_id": "r6", "source_id": "en-test", "source_family": "en3",
         "modality": "en", "normalized_bytes": 10},
    ]
    roles = {
        "ua-train": "train", "en-train": "train", "ua-val": "validation",
        "en-val": "validation", "ua-test": "test", "en-test": "test",
    }
    report = analyze_members(members, roles)
    assert report["every_language_has_all_roles"] is True
    assert report["strict_source_family_isolation"] is False
    assert report["cross_role_family_names"] == ["ua1"]


@pytest.mark.parametrize("mutation", ["bad_source", "unknown_source", "bad_language",
                                       "duplicate_record", "bad_size", "family_drift"])
def test_malformed_physical_member_refused(mutation: str):
    members = [
        {"record_id": "a", "source_id": "s1", "source_family": "fam1",
         "modality": "uk", "normalized_bytes": 40},
        {"record_id": "b", "source_id": "s1", "source_family": "fam1",
         "modality": "uk", "normalized_bytes": 40},
        {"record_id": "c", "source_id": "s2", "source_family": "fam2",
         "modality": "en", "normalized_bytes": 40},
        {"record_id": "d", "source_id": "s3", "source_family": "fam3",
         "modality": "en", "normalized_bytes": 40},
    ]
    roles = {"s1": "train", "s2": "validation", "s3": "test"}
    if mutation == "bad_source":
        members[1]["source_id"] = "unknown"
    elif mutation == "unknown_source":
        roles["forged"] = "train"
    elif mutation == "bad_language":
        members[0]["modality"] = "de"
    elif mutation == "duplicate_record":
        members[1]["record_id"] = "a"
    elif mutation == "bad_size":
        members[0]["normalized_bytes"] = 0
    else:
        members[1]["source_family"] = "forged"
    with pytest.raises(PhysicalSplitDenied):
        analyze_members(members, roles)
