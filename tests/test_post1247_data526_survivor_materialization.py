import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

materializer = importlib.import_module("materialize_post1247_data526_survivors_v1")

PRE_EVIDENCE = ROOT / "evidence" / "data526" / "v8" / "materialization_evidence.json"
PRE_INVENTORY = ROOT / "evidence" / "data526" / "v8" / "record_inventory.json"
DECONTAM_EVIDENCE = (
    ROOT / "evidence" / "data232" / "current_reserved_decontamination_v1_terminal.json"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(record_id: str, source_id: str, payload: str, *, family: str = "family", modality: str = "en") -> dict:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "family": family,
        "modality": modality,
        "normalized_payload": payload,
    }


def test_checked_in_authority_blobs_are_exactly_pinned():
    assert materializer._git_blob_sha1(PRE_EVIDENCE.read_bytes()) == materializer.PRE_EVIDENCE_GIT_BLOB_SHA1
    assert materializer._git_blob_sha1(DECONTAM_EVIDENCE.read_bytes()) == materializer.DECONTAM_EVIDENCE_GIT_BLOB_SHA1


def test_checked_in_pre_materialization_authority_is_exact_and_zero_credit():
    evidence = json.loads(PRE_EVIDENCE.read_text(encoding="utf-8"))
    materializer.validate_pre_authority(evidence)
    assert evidence["record_count"] == 275
    assert evidence["source_object_count"] == 262
    assert evidence["total_payload_bytes"] == 6_095_321
    assert evidence["authorized_unique_optimized_targets"] == 0
    assert evidence["training_executed"] is False


def test_terminal_decontamination_authority_yields_exact_three_hashes():
    evidence = json.loads(DECONTAM_EVIDENCE.read_text(encoding="utf-8"))
    excluded = materializer.validate_decontam_authority(evidence)
    assert excluded == {
        "f8bad2517ce5cda8b8fa66367291500c7c078a4a4a9ba5b9a308cce98ef30e8e",
        "8e663a497e34c237ae0d7281f8da3cc6a1a53aee6e37b5a0a5ee30ed1aa80e6d",
        "7ac52851e9729463b45205d1e1fef15cd60fe115e44909bf5e2feb9f32dfdc84",
    }


def test_current_text_free_inventory_recomputes_exact_post1247_projection():
    inventory = json.loads(PRE_INVENTORY.read_text(encoding="utf-8"))
    decontam = json.loads(DECONTAM_EVIDENCE.read_text(encoding="utf-8"))
    excluded_rows = {
        row["source_id_sha256"]: row for row in decontam["execution"]["excluded_records"]
    }
    matched = []
    survivors = []
    for row in inventory["records"]:
        source_hash = _sha(row["source_id"])
        if source_hash in excluded_rows:
            matched.append(row)
            authority = excluded_rows[source_hash]
            assert _sha(row["family"]) == authority["source_family_sha256"]
            assert row["modality"] == authority["modality"]
        else:
            survivors.append(row)

    assert { _sha(row["source_id"]) for row in matched } == set(excluded_rows)
    assert len(matched) == 3
    assert sum(row["payload_bytes"] for row in matched) == 173_358
    assert len(survivors) == 272
    assert len({row["source_id"] for row in survivors}) == 259
    assert sum(row["payload_bytes"] for row in survivors) == 5_921_963


def test_derive_survivors_removes_every_child_of_an_excluded_source():
    records = [
        _record("r1", "source-a", "one"),
        _record("r2", "source-a", "two"),
        _record("r3", "source-b", "three"),
    ]
    survivors, excluded = materializer.derive_survivors(records, {_sha("source-a")})
    assert [row["record_id"] for row in survivors] == ["r3"]
    assert [row["record_id"] for row in excluded] == ["r1", "r2"]


def test_derive_survivors_fails_closed_if_exclusion_hash_has_no_child():
    with pytest.raises(materializer.Post1247MaterializationError, match="not every decontamination source hash"):
        materializer.derive_survivors([_record("r1", "source-a", "one")], {_sha("missing")})


def test_record_schema_rejects_extra_or_non_text_fields():
    extra = _record("r1", "source-a", "one") | {"unexpected": "x"}
    with pytest.raises(materializer.Post1247MaterializationError, match="schema drift"):
        materializer._validate_record_shape([extra])

    bad = _record("r1", "source-a", "one")
    bad["modality"] = 1
    with pytest.raises(materializer.Post1247MaterializationError, match="non-empty string modality"):
        materializer._validate_record_shape([bad])


def test_strict_integer_validation_rejects_boolean_alias():
    with pytest.raises(materializer.Post1247MaterializationError, match="must be an integer"):
        materializer._strict_int(True, name="count")
