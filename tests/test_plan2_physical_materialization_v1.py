from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.plan2_physical_materialization_v1 import (
    Plan2MaterializationError,
    stage_candidate_cohort,
)

ROOT = Path(__file__).resolve().parents[1]


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def test_current_candidate_physically_materializes_with_two_distinct_exact_hashes(tmp_path):
    dest = tmp_path / "cohort"
    receipt = stage_candidate_cohort(ROOT, dest)
    assert receipt["source_member_count"] == 1
    assert receipt["source_level_candidate_only"] is True
    assert receipt["training_corpus_authorized"] is False
    assert receipt["tokenizer_fit_authorized"] is False
    assert receipt["evaluation_authorized"] is False
    assert receipt["raw"]["sha256"] != receipt["normalized"]["sha256"]
    assert receipt["raw"]["sha256"] == _digest((dest / "raw.snapshot").read_bytes())
    assert receipt["normalized"]["sha256"] == _digest((dest / "normalized.utf8").read_bytes())
    assert set(p.name for p in dest.iterdir()) == {"raw.snapshot", "normalized.utf8", "manifest.json"}
    assert json.loads((dest / "manifest.json").read_text()) == receipt


def test_exact_restart_is_idempotent(tmp_path):
    dest = tmp_path / "cohort"
    first = stage_candidate_cohort(ROOT, dest)
    manifest = (dest / "manifest.json").read_bytes()
    second = stage_candidate_cohort(ROOT, dest)
    assert first == second
    assert manifest == (dest / "manifest.json").read_bytes()
    assert len(list(dest.iterdir())) == 3


def test_interrupted_before_manifest_resumes_without_duplicate_members(tmp_path):
    dest = tmp_path / "cohort"
    receipt = stage_candidate_cohort(ROOT, dest)
    (dest / "manifest.json").unlink()
    (dest / "normalized.utf8").unlink()
    (dest / ".plan2-partial-crash").write_bytes(b"aborted transaction")
    repaired = stage_candidate_cohort(ROOT, dest)
    assert repaired == receipt
    assert not (dest / ".plan2-partial-crash").exists()
    assert set(p.name for p in dest.iterdir()) == {"raw.snapshot", "normalized.utf8", "manifest.json"}


@pytest.mark.parametrize("member", ["raw.snapshot", "normalized.utf8", "manifest.json"])
def test_postpublication_bit_flip_fails_closed(tmp_path, member):
    dest = tmp_path / "cohort"
    stage_candidate_cohort(ROOT, dest)
    (dest / member).write_bytes((dest / member).read_bytes() + b"malicious")
    with pytest.raises(Plan2MaterializationError):
        stage_candidate_cohort(ROOT, dest)


def test_missing_member_after_publication_is_not_silently_repaired(tmp_path):
    dest = tmp_path / "cohort"
    stage_candidate_cohort(ROOT, dest)
    (dest / "raw.snapshot").unlink()
    with pytest.raises(Plan2MaterializationError, match="membership gap"):
        stage_candidate_cohort(ROOT, dest)


def test_corrupt_partial_member_is_not_silently_overwritten(tmp_path):
    dest = tmp_path / "cohort"
    stage_candidate_cohort(ROOT, dest)
    (dest / "manifest.json").unlink()
    (dest / "raw.snapshot").write_bytes(b"bad existing partial")
    with pytest.raises(Plan2MaterializationError, match="mismatch"):
        stage_candidate_cohort(ROOT, dest)


def test_destination_symlink_and_member_symlink_are_rejected(tmp_path):
    dest = tmp_path / "cohort"
    link = tmp_path / "alias"
    stage_candidate_cohort(ROOT, dest)
    link.symlink_to(dest, target_is_directory=True)
    with pytest.raises(Plan2MaterializationError, match="symlink"):
        stage_candidate_cohort(ROOT, link)
    (dest / "normalized.utf8").unlink()
    (dest / "normalized.utf8").symlink_to(ROOT / "configs/data/plan2_source_inventory_v1.json")
    with pytest.raises(Plan2MaterializationError):
        stage_candidate_cohort(ROOT, dest)


def test_unexpected_cohort_membership_fails_closed(tmp_path):
    dest = tmp_path / "cohort"
    stage_candidate_cohort(ROOT, dest)
    (dest / "unexpected-record.txt").write_text("extra record")
    with pytest.raises(Plan2MaterializationError, match="unexpected"):
        stage_candidate_cohort(ROOT, dest)


def test_training_mode_never_creates_staged_bytes(tmp_path):
    dest = tmp_path / "cohort"
    with pytest.raises(Plan2MaterializationError, match="authority"):
        stage_candidate_cohort(ROOT, dest, purpose="training")
    assert not dest.exists()


def test_rights_promotion_and_revocation_fail_before_materialization(tmp_path):
    rights = json.loads((ROOT / "configs/data/plan2_rights_admissibility_v1.json").read_text())
    for mutation in (
        lambda x: x["grants"][0].update(revoked=True),
        lambda x: x["grants"][0].update(allowed_uses=["training"]),
        lambda x: x["grants"][0].update(license_id="UNKNOWN"),
        lambda x: x["grants"][0].update(evidence_sha256="0" * 64),
    ):
        altered = copy.deepcopy(rights)
        mutation(altered)
        with pytest.raises((Plan2MaterializationError, ValueError)):
            stage_candidate_cohort(ROOT, tmp_path / "cohort", rights_seed=altered)
        assert not (tmp_path / "cohort").exists()


def test_unknown_source_or_duplicate_in_inventory_fails_closed(tmp_path):
    cfg = json.loads((ROOT / "configs/data/plan2_source_inventory_v1.json").read_text())
    duplicated = copy.deepcopy(cfg)
    duplicated["sources"].append(copy.deepcopy(duplicated["sources"][0]))
    with pytest.raises((Plan2MaterializationError, ValueError)):
        stage_candidate_cohort(ROOT, tmp_path / "cohort", inventory_seed=duplicated)
    foreign = copy.deepcopy(cfg)
    foreign["sources"][0]["source_id"] = "unreviewed.corpus.source"
    with pytest.raises((Plan2MaterializationError, ValueError)):
        stage_candidate_cohort(ROOT, tmp_path / "cohort", inventory_seed=foreign)


def test_restart_result_does_not_depend_on_destination_path(tmp_path):
    a = stage_candidate_cohort(ROOT, tmp_path / "a")
    b = stage_candidate_cohort(ROOT, tmp_path / "b")
    assert a == b
    assert a["manifest_sha256"] == b["manifest_sha256"]
