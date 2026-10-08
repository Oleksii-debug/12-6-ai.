"""Plan 2 Section 6: exact global dedup, provenance, fail-closed/restart tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import plan2_exact_dedup_v1 as exact
from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy

ROOT = Path(__file__).resolve().parents[1]


def _fixture(source: str, *lines: str):
    payload = "".join(line + "\n" for line in lines).encode("utf-8")
    parts = []
    start = 0
    for i, chunk in enumerate(payload.splitlines(keepends=True)):
        parts.append({"index": i, "start_byte": start,
                      "end_byte": start + len(chunk), "sha256": exact._sha(chunk)})
        start += len(chunk)
    core = {
        "schema_version": norm.MANIFEST_SCHEMA,
        "source_id": source,
        "record_count": len(parts),
        "records": parts,
        "normalized_bytes": len(payload),
        "normalized_sha256": exact._sha(payload),
        "classification": {"language": "uk", "modality": "text"},
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    normalization = {**core, "manifest_sha256": exact._sha(exact._canonical(core))}
    receipt = privacy.inspect_normalized(
        normalization, payload, policy_sha256="f" * 64)
    return normalization, payload, receipt


TEXT_A = "Це відкритий публічний опис української технічної документації."
TEXT_B = "Документація докладно описує використання відкритого програмного забезпечення."


def test_global_equivalence_is_deterministic_and_preserves_family_provenance():
    a = _fixture("source.a", TEXT_A, TEXT_B, TEXT_A)
    b = _fixture("source.b", TEXT_A, TEXT_B)
    one = exact.inspect_exact((a, b))
    two = exact.inspect_exact((b, a))
    assert one == two
    assert one["input_candidate_record_count"] == 5
    assert one["retained_record_count"] == 2
    assert one["exact_duplicate_record_count"] == 3
    assert len(one["duplicate_families"]) == 2
    assert one["retained_record_ids"] == [
        "source.a:r00000000", "source.a:r00000001"]
    assert {r["source_id"] for r in one["sources"]} == {"source.a", "source.b"}
    assert one["training_corpus_authorized"] is False
    assert TEXT_A not in json.dumps(one, ensure_ascii=False)


def test_identical_whole_members_are_reported_without_erasing_lineage():
    a = _fixture("source.a", TEXT_A)
    b = _fixture("source.b", TEXT_A)
    receipt = exact.inspect_exact((a, b))
    assert receipt["duplicate_member_families"] == [{
        "normalized_member_sha256": exact._sha(a[1]),
        "source_ids": ["source.a", "source.b"],
    }]


def test_nonidentical_records_are_kept_and_altered_input_changes_identity():
    a = exact.inspect_exact((_fixture("source.a", TEXT_A, TEXT_B),))
    b = exact.inspect_exact((_fixture("source.a", TEXT_A, TEXT_A),))
    assert a["retained_record_count"] == 2
    assert b["retained_record_count"] == 1
    assert a["manifest_sha256"] != b["manifest_sha256"]


def test_tombstones_are_not_resurrected():
    normalized, payload, original = _fixture("source.a", TEXT_A, TEXT_B)
    removed = privacy.inspect_normalized(
        normalized, payload, policy_sha256="f" * 64,
        tombstones=("source.a:r00000001",))
    result = exact.inspect_exact(((normalized, payload, removed),))
    assert result["retained_record_ids"] == ["source.a:r00000000"]
    assert result["input_candidate_record_count"] == 1
    with pytest.raises(exact.ExactDedupError, match="forged"):
        exact.inspect_exact(((normalized, payload, original | {
            "candidate_record_count": 1,
        }),))


@pytest.mark.parametrize("mutation", ["payload", "sha", "policy", "promote"])
def test_tamper_and_escalation_fail_closed(mutation):
    normalized, payload, receipt = _fixture("source.a", TEXT_A)
    changed = copy.deepcopy(receipt)
    if mutation == "payload":
        payload += b"x"
    elif mutation == "sha":
        changed["manifest_sha256"] = "0" * 64
    elif mutation == "policy":
        changed["policy_sha256"] = "bad"
    else:
        changed["training_corpus_authorized"] = True
    with pytest.raises(exact.ExactDedupError):
        exact.inspect_exact(((normalized, payload, changed),))


def test_duplicate_source_identity_is_rejected():
    row = _fixture("source.a", TEXT_A)
    with pytest.raises(exact.ExactDedupError, match="duplicate source"):
        exact.inspect_exact((row, row))


def test_physical_current_cohort_restart_and_immutable_no_clobber(tmp_path):
    first = exact.stage_exact(ROOT, tmp_path / "first")
    second = exact.stage_exact(ROOT, tmp_path / "first")
    independent = exact.stage_exact(ROOT, tmp_path / "rebuild")
    assert first == second == independent
    assert first["input_candidate_record_count"] == 50
    assert first["retained_record_count"] + first["exact_duplicate_record_count"] == 50
    artifact = tmp_path / "first" / "exact-dedup-manifest.json"
    assert json.loads(artifact.read_text()) == first
    artifact.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(exact.ExactDedupError, match="immutable"):
        exact.stage_exact(ROOT, tmp_path / "first")


def test_symlink_destination_denied(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(
        tmp_path / "real", target_is_directory=True)
    with pytest.raises(exact.ExactDedupError, match="symlink"):
        exact.stage_exact(ROOT, tmp_path / "link" / "candidate")
