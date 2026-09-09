from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.postdecontam_balance_projection_v1 import (
    BINDING_SCHEMA,
    FAMILY_MAP_SCHEMA,
    ProjectionError,
    build_family_vector,
    read_records_jsonl,
    verify_family_vector,
)


SHA = "a" * 64
GIT_SHA = "b" * 40


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _with_hash(document: dict, field: str) -> dict:
    result = copy.deepcopy(document)
    result.pop(field, None)
    result[field] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _records(tmp_path: Path) -> tuple[list, str]:
    rows = [
        {
            "record_id": "ua-1",
            "source_id": "src-ua-1",
            "family": "ua.family.a",
            "modality": "text",
            "normalized_payload": "Привіт світе",
            "training_eligible": False,
            "evaluation_eligible": False,
        },
        {
            "record_id": "ua-2",
            "source_id": "src-ua-2",
            "family": "ua.family.b",
            "modality": "text",
            "normalized_payload": "Ще український текст",
        },
        {
            "record_id": "en-1",
            "source_id": "src-en-1",
            "family": "en.family.a",
            "modality": "text",
            "normalized_payload": "Hello world",
        },
        {
            "record_id": "code-1",
            "source_id": "src-code-1",
            "family": "code.family.a",
            "modality": "code",
            "normalized_payload": "def f():\n    return 1\n",
        },
    ]
    path = tmp_path / "records.jsonl"
    payload = b"".join(_canonical(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return read_records_jsonl(path)


def _family_map() -> dict:
    return _with_hash(
        {
            "schema": FAMILY_MAP_SCHEMA,
            "families": [
                {"family": "ua.family.a", "stratum": "uk"},
                {"family": "ua.family.b", "stratum": "uk"},
                {"family": "en.family.a", "stratum": "en"},
                {"family": "code.family.a", "stratum": "code"},
            ],
            "training_authorized_by_this_mapping": False,
        },
        "family_map_identity_sha256",
    )


def _binding(records: list, records_sha: str, excluded_ids: list[str]) -> dict:
    excluded_hashes = [
        hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        for record_id in excluded_ids
    ]
    survivor_bytes = sum(
        record.payload_bytes for record in records if record.record_id not in excluded_ids
    )
    return _with_hash(
        {
            "schema": BINDING_SCHEMA,
            "verdict": "PASS_WITH_EXCLUSIONS" if excluded_ids else "PASS_CLEAN",
            "retained_inventory_identity_sha256": SHA,
            "records_jsonl_sha256": records_sha,
            "input_record_count": len(records),
            "input_payload_bytes": sum(record.payload_bytes for record in records),
            "excluded_record_id_sha256": excluded_hashes,
            "survivor_record_count": len(records) - len(excluded_ids),
            "survivor_payload_bytes": survivor_bytes,
            "final_test_outcomes_read": False,
            "model_selection_performed": False,
            "authorized_optimized_target_exposure": 0,
            "training_authorized_by_this_report": False,
        },
        "decontamination_authority_sha256",
    )


def _build(tmp_path: Path, excluded_ids: list[str] | None = None) -> dict:
    records, records_sha = _records(tmp_path)
    return build_family_vector(
        records=records,
        records_jsonl_sha256=records_sha,
        decontamination_binding=_binding(records, records_sha, excluded_ids or []),
        family_map=_family_map(),
        source_git_sha=GIT_SHA,
    )


def test_builds_deterministic_text_free_projection(tmp_path: Path) -> None:
    first = _build(tmp_path, ["ua-1"])
    second = _build(tmp_path, ["ua-1"])
    assert first == second
    assert verify_family_vector(first) == first["family_vector_identity_sha256"]
    assert first["excluded_record_count"] == 1
    assert first["survivor_record_count"] == 3
    assert first["authorized_optimized_target_exposure"] == 0
    serialized = _canonical(first).decode("utf-8")
    assert "Привіт" not in serialized
    assert "Hello world" not in serialized
    assert "def f" not in serialized


def test_unknown_decontamination_exclusion_fails_closed(tmp_path: Path) -> None:
    records, records_sha = _records(tmp_path)
    binding = _binding(records, records_sha, [])
    binding["excluded_record_id_sha256"] = [
        hashlib.sha256(b"not-a-record").hexdigest()
    ]
    binding["survivor_record_count"] = len(records) - 1
    binding = _with_hash(binding, "decontamination_authority_sha256")
    with pytest.raises(ProjectionError, match="unknown records"):
        build_family_vector(
            records=records,
            records_jsonl_sha256=records_sha,
            decontamination_binding=binding,
            family_map=_family_map(),
            source_git_sha=GIT_SHA,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("verdict", "RUNNING", "not terminal PASS"),
        ("final_test_outcomes_read", True, "must be false"),
        ("model_selection_performed", True, "must be false"),
        ("training_authorized_by_this_report", True, "must be false"),
        ("authorized_optimized_target_exposure", 1, "must keep"),
    ],
)
def test_nonterminal_or_widened_decontamination_fails_closed(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    records, records_sha = _records(tmp_path)
    binding = _binding(records, records_sha, [])
    binding[field] = value
    binding = _with_hash(binding, "decontamination_authority_sha256")
    with pytest.raises(ProjectionError, match=match):
        build_family_vector(
            records=records,
            records_jsonl_sha256=records_sha,
            decontamination_binding=binding,
            family_map=_family_map(),
            source_git_sha=GIT_SHA,
        )


def test_records_file_substitution_fails_closed(tmp_path: Path) -> None:
    records, records_sha = _records(tmp_path)
    binding = _binding(records, records_sha, [])
    binding["records_jsonl_sha256"] = "c" * 64
    binding = _with_hash(binding, "decontamination_authority_sha256")
    with pytest.raises(ProjectionError, match="different records"):
        build_family_vector(
            records=records,
            records_jsonl_sha256=records_sha,
            decontamination_binding=binding,
            family_map=_family_map(),
            source_git_sha=GIT_SHA,
        )


def test_duplicate_record_id_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    row = {
        "record_id": "dup",
        "source_id": "src",
        "family": "ua.family.a",
        "modality": "text",
        "normalized_payload": "payload",
    }
    path.write_bytes(_canonical(row) + b"\n" + _canonical(row) + b"\n")
    with pytest.raises(ProjectionError, match="duplicate record_id"):
        read_records_jsonl(path)


def test_payload_byte_or_hash_drift_fails_closed(tmp_path: Path) -> None:
    bad_bytes = {
        "record_id": "r",
        "source_id": "s",
        "family": "f",
        "modality": "text",
        "normalized_payload": "abc",
        "normalized_payload_bytes": 4,
    }
    path = tmp_path / "bad.jsonl"
    path.write_bytes(_canonical(bad_bytes) + b"\n")
    with pytest.raises(ProjectionError, match="bytes drift"):
        read_records_jsonl(path)

    bad_hash = dict(bad_bytes)
    bad_hash["normalized_payload_bytes"] = 3
    bad_hash["normalized_payload_sha256"] = "d" * 64
    path.write_bytes(_canonical(bad_hash) + b"\n")
    with pytest.raises(ProjectionError, match="sha256 drift"):
        read_records_jsonl(path)


def test_unmapped_surviving_family_fails_closed(tmp_path: Path) -> None:
    records, records_sha = _records(tmp_path)
    family_map = _family_map()
    family_map["families"] = [
        row for row in family_map["families"] if row["family"] != "en.family.a"
    ]
    family_map = _with_hash(family_map, "family_map_identity_sha256")
    with pytest.raises(ProjectionError, match="unmapped surviving family"):
        build_family_vector(
            records=records,
            records_jsonl_sha256=records_sha,
            decontamination_binding=_binding(records, records_sha, []),
            family_map=family_map,
            source_git_sha=GIT_SHA,
        )


def test_decontamination_survivor_bytes_must_match_record_reconstruction(
    tmp_path: Path,
) -> None:
    records, records_sha = _records(tmp_path)
    binding = _binding(records, records_sha, ["ua-1"])
    binding["survivor_payload_bytes"] += 1
    binding = _with_hash(binding, "decontamination_authority_sha256")
    with pytest.raises(ProjectionError, match="survivor payload bytes mismatch"):
        build_family_vector(
            records=records,
            records_jsonl_sha256=records_sha,
            decontamination_binding=binding,
            family_map=_family_map(),
            source_git_sha=GIT_SHA,
        )


def test_family_vector_self_hash_and_arithmetic_are_fail_closed(tmp_path: Path) -> None:
    result = _build(tmp_path)
    tampered = copy.deepcopy(result)
    tampered["families"][0]["capacity_bytes"] += 1
    with pytest.raises(ProjectionError, match="self-hash mismatch"):
        verify_family_vector(tampered)

    tampered = copy.deepcopy(result)
    tampered["families"][0]["capacity_bytes"] += 1
    tampered = _with_hash(tampered, "family_vector_identity_sha256")
    with pytest.raises(ProjectionError, match="total bytes mismatch"):
        verify_family_vector(tampered)


def test_input_cannot_preclaim_training_or_evaluation(tmp_path: Path) -> None:
    row = {
        "record_id": "r",
        "source_id": "s",
        "family": "f",
        "modality": "text",
        "normalized_payload": "abc",
        "training_eligible": True,
        "evaluation_eligible": False,
    }
    path = tmp_path / "eligible.jsonl"
    path.write_bytes(_canonical(row) + b"\n")
    with pytest.raises(ProjectionError, match="training eligibility"):
        read_records_jsonl(path)
