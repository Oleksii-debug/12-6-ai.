from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import tools.next100_106_balance_gate as next100_gate
from twelve_six.data.postdecontam_balance_projection_v1 import (
    BINDING_SCHEMA,
    FAMILY_MAP_SCHEMA,
    FAMILY_PROVENANCE_SCHEMA,
    G05_G06_COVERAGE_SCHEMA,
    QUALITY_GRANULARITY_IDENTITY_SHA256,
    QUALITY_POLICY_IDENTITY_SHA256,
    ProjectionError,
    build_family_vector,
    read_records_jsonl,
    verify_family_vector,
)
from twelve_six.data.postdecontam_next100_adapter_v1 import (
    adapt_family_vector_to_next100_106,
)


SHA = "a" * 64
PRIVACY_SHA = "e" * 64
DEDUP_EVIDENCE_SHA = "d" * 64
GIT_SHA = "b" * 40
DEDUP_HEAD_SHA = "c" * 40
DEDUP_WORKER_ID = "D03-GLOBAL-DEDUP"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _with_hash(document: dict, field: str) -> dict:
    result = copy.deepcopy(document)
    result.pop(field, None)
    result[field] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _record_hash(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()


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


def _binding(records: list, records_sha: str, excluded_ids: list[str]) -> dict:
    excluded_hashes = [_record_hash(record_id) for record_id in excluded_ids]
    survivor_bytes = sum(
        record.payload_bytes for record in records if record.record_id not in excluded_ids
    )
    return _with_hash(
        {
            "schema": BINDING_SCHEMA,
            "verdict": "PASS_WITH_EXCLUSIONS" if excluded_ids else "PASS_CLEAN",
            "retained_inventory_identity_sha256": SHA,
            "dedup_evidence_identity_sha256": DEDUP_EVIDENCE_SHA,
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


def _survivors(records: list, excluded_ids: list[str]) -> list:
    return [record for record in records if record.record_id not in excluded_ids]


def _family_map(survivors: list) -> dict:
    stratum_by_family = {
        "ua.family.a": "uk",
        "ua.family.b": "uk",
        "en.family.a": "en",
        "code.family.a": "code",
    }
    families = sorted({record.family for record in survivors})
    return _with_hash(
        {
            "schema": FAMILY_MAP_SCHEMA,
            "families": [
                {"family": family, "stratum": stratum_by_family[family]}
                for family in families
            ],
            "training_authorized_by_this_mapping": False,
        },
        "family_map_identity_sha256",
    )


def _family_provenance(survivors: list) -> dict:
    semantics = {
        "ua.family.a": ("uk", "uk", ["text"]),
        "ua.family.b": ("uk", "uk", ["text"]),
        "en.family.a": ("en", "en", ["text"]),
        "code.family.a": ("und", "code", ["code"]),
    }
    families = sorted({record.family for record in survivors})
    return _with_hash(
        {
            "schema": FAMILY_PROVENANCE_SCHEMA,
            "families": [
                {
                    "family": family,
                    "source_family_identity_sha256": hashlib.sha256(
                        f"authority:{family}".encode()
                    ).hexdigest(),
                    "language": semantics[family][0],
                    "modalities": semantics[family][2],
                    "stratum": semantics[family][1],
                }
                for family in families
            ],
            "training_authorized_by_this_provenance": False,
        },
        "family_provenance_identity_sha256",
    )


def _coverage(records: list, records_sha: str, binding: dict, excluded_ids: list[str]) -> dict:
    survivors = _survivors(records, excluded_ids)
    covered_rows = sorted(
        [
            {
                "record_id_sha256": _record_hash(record.record_id),
                "payload_sha256": record.payload_sha256,
                "payload_bytes": record.payload_bytes,
            }
            for record in survivors
        ],
        key=lambda row: row["record_id_sha256"],
    )
    return _with_hash(
        {
            "schema": G05_G06_COVERAGE_SCHEMA,
            "status": "PASS",
            "records_jsonl_sha256": records_sha,
            "retained_inventory_identity_sha256": SHA,
            "decontamination_authority_sha256": binding[
                "decontamination_authority_sha256"
            ],
            "quality_policy_identity_sha256": QUALITY_POLICY_IDENTITY_SHA256,
            "quality_granularity_identity_sha256": (
                QUALITY_GRANULARITY_IDENTITY_SHA256
            ),
            "privacy_policy_identity_sha256": PRIVACY_SHA,
            "covered_records": covered_rows,
            "covered_record_count": len(covered_rows),
            "covered_payload_bytes": sum(record.payload_bytes for record in survivors),
            "final_test_outcomes_read": False,
            "authorized_optimized_target_exposure": 0,
            "training_authorized_by_this_coverage": False,
        },
        "g05_g06_coverage_identity_sha256",
    )


def _inputs(tmp_path: Path, excluded_ids: list[str] | None = None) -> dict:
    excluded = excluded_ids or []
    records, records_sha = _records(tmp_path)
    binding = _binding(records, records_sha, excluded)
    survivors = _survivors(records, excluded)
    family_map = _family_map(survivors)
    provenance = _family_provenance(survivors)
    coverage = _coverage(records, records_sha, binding, excluded)
    return {
        "records": records,
        "records_jsonl_sha256": records_sha,
        "decontamination_binding": binding,
        "g05_g06_coverage": coverage,
        "expected_g05_g06_coverage_identity_sha256": coverage[
            "g05_g06_coverage_identity_sha256"
        ],
        "expected_privacy_policy_identity_sha256": PRIVACY_SHA,
        "family_map": family_map,
        "expected_family_map_identity_sha256": family_map[
            "family_map_identity_sha256"
        ],
        "family_provenance": provenance,
        "expected_family_provenance_identity_sha256": provenance[
            "family_provenance_identity_sha256"
        ],
        "source_git_sha": GIT_SHA,
    }


def _build(tmp_path: Path, excluded_ids: list[str] | None = None) -> dict:
    return build_family_vector(**_inputs(tmp_path, excluded_ids))


def test_builds_deterministic_text_free_projection(tmp_path: Path) -> None:
    first = _build(tmp_path, ["ua-1"])
    second = _build(tmp_path, ["ua-1"])
    assert first == second
    assert verify_family_vector(first) == first["family_vector_identity_sha256"]
    assert first["excluded_record_count"] == 1
    assert first["survivor_record_count"] == 3
    assert first["authorized_optimized_target_exposure"] == 0
    assert first["g05_g06_coverage_identity_sha256"]
    assert first["family_provenance_identity_sha256"]
    serialized = _canonical(first).decode("utf-8")
    assert "Привіт" not in serialized
    assert "Hello world" not in serialized
    assert "def f" not in serialized


def test_unknown_decontamination_exclusion_fails_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    binding = inputs["decontamination_binding"]
    binding["excluded_record_id_sha256"] = [_record_hash("not-a-record")]
    binding["survivor_record_count"] = len(inputs["records"]) - 1
    inputs["decontamination_binding"] = _with_hash(
        binding,
        "decontamination_authority_sha256",
    )
    with pytest.raises(ProjectionError, match="unknown records"):
        build_family_vector(**inputs)


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
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    inputs = _inputs(tmp_path)
    binding = inputs["decontamination_binding"]
    binding[field] = value
    inputs["decontamination_binding"] = _with_hash(
        binding,
        "decontamination_authority_sha256",
    )
    with pytest.raises(ProjectionError, match=match):
        build_family_vector(**inputs)


def test_records_file_substitution_fails_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    binding = inputs["decontamination_binding"]
    binding["records_jsonl_sha256"] = "c" * 64
    inputs["decontamination_binding"] = _with_hash(
        binding,
        "decontamination_authority_sha256",
    )
    with pytest.raises(ProjectionError, match="different records"):
        build_family_vector(**inputs)


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


def test_decontamination_survivor_bytes_must_match_record_reconstruction(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path, ["ua-1"])
    binding = inputs["decontamination_binding"]
    binding["survivor_payload_bytes"] += 1
    inputs["decontamination_binding"] = _with_hash(
        binding,
        "decontamination_authority_sha256",
    )
    with pytest.raises(ProjectionError, match="survivor payload bytes mismatch"):
        build_family_vector(**inputs)


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


@pytest.mark.parametrize("field", ["training_eligible", "evaluation_eligible"])
@pytest.mark.parametrize("value", [0, 0.0, None, "false", True])
def test_input_eligibility_markers_require_exact_false(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    row = {
        "record_id": "r",
        "source_id": "s",
        "family": "f",
        "modality": "text",
        "normalized_payload": "abc",
    }
    row[field] = value
    path = tmp_path / "eligible.jsonl"
    path.write_bytes(_canonical(row) + b"\n")
    match = "training eligibility" if field == "training_eligible" else "evaluation eligibility"
    with pytest.raises(ProjectionError, match=match):
        read_records_jsonl(path)


def test_self_consistent_coverage_substitution_rejected_by_external_identity(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    original_expected = inputs["expected_g05_g06_coverage_identity_sha256"]
    coverage = inputs["g05_g06_coverage"]
    coverage["covered_payload_bytes"] += 1
    coverage = _with_hash(coverage, "g05_g06_coverage_identity_sha256")
    inputs["g05_g06_coverage"] = coverage
    inputs["expected_g05_g06_coverage_identity_sha256"] = original_expected
    with pytest.raises(ProjectionError, match="independently expected identity"):
        build_family_vector(**inputs)


def test_coverage_must_exactly_cover_survivor_payload_pairs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    coverage = inputs["g05_g06_coverage"]
    coverage["covered_records"] = coverage["covered_records"][:-1]
    coverage["covered_record_count"] -= 1
    coverage = _with_hash(coverage, "g05_g06_coverage_identity_sha256")
    inputs["g05_g06_coverage"] = coverage
    inputs["expected_g05_g06_coverage_identity_sha256"] = coverage[
        "g05_g06_coverage_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="does not exactly cover"):
        build_family_vector(**inputs)


def test_coverage_lineage_is_bound_to_decontam_and_inventory(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    coverage = inputs["g05_g06_coverage"]
    coverage["retained_inventory_identity_sha256"] = "f" * 64
    coverage = _with_hash(coverage, "g05_g06_coverage_identity_sha256")
    inputs["g05_g06_coverage"] = coverage
    inputs["expected_g05_g06_coverage_identity_sha256"] = coverage[
        "g05_g06_coverage_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="retained-inventory lineage mismatch"):
        build_family_vector(**inputs)


def test_coverage_rejects_noncanonical_quality_or_privacy_policy(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    coverage = inputs["g05_g06_coverage"]
    coverage["quality_policy_identity_sha256"] = "f" * 64
    coverage = _with_hash(coverage, "g05_g06_coverage_identity_sha256")
    inputs["g05_g06_coverage"] = coverage
    inputs["expected_g05_g06_coverage_identity_sha256"] = coverage[
        "g05_g06_coverage_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="noncanonical quality policy"):
        build_family_vector(**inputs)

    inputs = _inputs(tmp_path)
    inputs["expected_privacy_policy_identity_sha256"] = "f" * 64
    with pytest.raises(ProjectionError, match="privacy policy identity mismatch"):
        build_family_vector(**inputs)


def test_self_consistent_family_map_reclassification_rejected_by_expected_identity(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    original_expected = inputs["expected_family_map_identity_sha256"]
    family_map = inputs["family_map"]
    target = next(row for row in family_map["families"] if row["family"] == "en.family.a")
    target["stratum"] = "uk"
    family_map = _with_hash(family_map, "family_map_identity_sha256")
    inputs["family_map"] = family_map
    inputs["expected_family_map_identity_sha256"] = original_expected
    with pytest.raises(ProjectionError, match="independently expected identity"):
        build_family_vector(**inputs)


def test_map_cannot_reclassify_bound_family_provenance(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    family_map = inputs["family_map"]
    target = next(row for row in family_map["families"] if row["family"] == "en.family.a")
    target["stratum"] = "uk"
    family_map = _with_hash(family_map, "family_map_identity_sha256")
    inputs["family_map"] = family_map
    inputs["expected_family_map_identity_sha256"] = family_map[
        "family_map_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="reclassifies canonical provenance"):
        build_family_vector(**inputs)


def test_family_map_rejects_missing_or_extra_survivor_families(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    family_map = inputs["family_map"]
    family_map["families"] = family_map["families"][:-1]
    family_map = _with_hash(family_map, "family_map_identity_sha256")
    inputs["family_map"] = family_map
    inputs["expected_family_map_identity_sha256"] = family_map[
        "family_map_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="exactly cover surviving families"):
        build_family_vector(**inputs)

    inputs = _inputs(tmp_path)
    family_map = inputs["family_map"]
    family_map["families"].append({"family": "extra.family", "stratum": "en"})
    family_map = _with_hash(family_map, "family_map_identity_sha256")
    inputs["family_map"] = family_map
    inputs["expected_family_map_identity_sha256"] = family_map[
        "family_map_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="exactly cover surviving families"):
        build_family_vector(**inputs)


def test_family_provenance_is_externally_bound_and_modality_checked(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    original_expected = inputs["expected_family_provenance_identity_sha256"]
    provenance = inputs["family_provenance"]
    target = next(
        row for row in provenance["families"] if row["family"] == "en.family.a"
    )
    target["language"] = "uk"
    provenance = _with_hash(provenance, "family_provenance_identity_sha256")
    inputs["family_provenance"] = provenance
    inputs["expected_family_provenance_identity_sha256"] = original_expected
    with pytest.raises(ProjectionError, match="independently expected identity"):
        build_family_vector(**inputs)

    inputs = _inputs(tmp_path)
    provenance = inputs["family_provenance"]
    target = next(
        row for row in provenance["families"] if row["family"] == "en.family.a"
    )
    target["modalities"] = ["code"]
    provenance = _with_hash(provenance, "family_provenance_identity_sha256")
    inputs["family_provenance"] = provenance
    inputs["expected_family_provenance_identity_sha256"] = provenance[
        "family_provenance_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="modality is not admitted"):
        build_family_vector(**inputs)


def test_coverage_truth_boundary_cannot_widen(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    coverage = inputs["g05_g06_coverage"]
    coverage["training_authorized_by_this_coverage"] = True
    coverage = _with_hash(coverage, "g05_g06_coverage_identity_sha256")
    inputs["g05_g06_coverage"] = coverage
    inputs["expected_g05_g06_coverage_identity_sha256"] = coverage[
        "g05_g06_coverage_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="must be false"):
        build_family_vector(**inputs)


def _one_byte_inputs(tmp_path: Path) -> dict:
    row = {
        "record_id": "en-one",
        "source_id": "src-en-one",
        "family": "en.family.a",
        "modality": "text",
        "normalized_payload": "x",
    }
    path = tmp_path / "one-byte.jsonl"
    path.write_bytes(_canonical(row) + b"\n")
    records, records_sha = read_records_jsonl(path)
    binding = _binding(records, records_sha, [])
    survivors = _survivors(records, [])
    family_map = _family_map(survivors)
    provenance = _family_provenance(survivors)
    coverage = _coverage(records, records_sha, binding, [])
    return {
        "records": records,
        "records_jsonl_sha256": records_sha,
        "decontamination_binding": binding,
        "g05_g06_coverage": coverage,
        "expected_g05_g06_coverage_identity_sha256": coverage[
            "g05_g06_coverage_identity_sha256"
        ],
        "expected_privacy_policy_identity_sha256": PRIVACY_SHA,
        "family_map": family_map,
        "expected_family_map_identity_sha256": family_map[
            "family_map_identity_sha256"
        ],
        "family_provenance": provenance,
        "expected_family_provenance_identity_sha256": provenance[
            "family_provenance_identity_sha256"
        ],
        "source_git_sha": GIT_SHA,
    }


def _dedup_authority(evidence_identity_sha256: str = DEDUP_EVIDENCE_SHA) -> dict:
    return {
        "worker_id": DEDUP_WORKER_ID,
        "head_sha": DEDUP_HEAD_SHA,
        "evidence_identity_sha256": evidence_identity_sha256,
        "terminal_verdict": "PASS",
    }


def test_declared_one_byte_bool_alias_fails_closed(tmp_path: Path) -> None:
    row = {
        "record_id": "r",
        "source_id": "s",
        "family": "en.family.a",
        "modality": "text",
        "normalized_payload": "x",
        "normalized_payload_bytes": True,
    }
    path = tmp_path / "bool-byte.jsonl"
    path.write_bytes(_canonical(row) + b"\n")
    with pytest.raises(ProjectionError, match="non-negative integer"):
        read_records_jsonl(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_record_count", True),
        ("input_payload_bytes", True),
        ("survivor_record_count", True),
        ("survivor_payload_bytes", True),
        ("authorized_optimized_target_exposure", False),
    ],
)
def test_decontamination_bool_integer_aliases_fail_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    inputs = _one_byte_inputs(tmp_path)
    binding = inputs["decontamination_binding"]
    binding[field] = value
    inputs["decontamination_binding"] = _with_hash(
        binding,
        "decontamination_authority_sha256",
    )
    with pytest.raises(ProjectionError, match="non-negative integer"):
        build_family_vector(**inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("covered_record_count", True),
        ("covered_payload_bytes", True),
        ("authorized_optimized_target_exposure", False),
    ],
)
def test_coverage_bool_integer_aliases_fail_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    inputs = _one_byte_inputs(tmp_path)
    coverage = inputs["g05_g06_coverage"]
    coverage[field] = value
    coverage = _with_hash(coverage, "g05_g06_coverage_identity_sha256")
    inputs["g05_g06_coverage"] = coverage
    inputs["expected_g05_g06_coverage_identity_sha256"] = coverage[
        "g05_g06_coverage_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="non-negative integer"):
        build_family_vector(**inputs)


def test_adapter_binds_dedup_evidence_to_family_vector_lineage(
    tmp_path: Path,
) -> None:
    family_vector = _build(tmp_path)
    mismatched_evidence = "f" * 64
    with pytest.raises(ProjectionError, match="family-vector lineage"):
        adapt_family_vector_to_next100_106(
            family_vector,
            expected_family_vector_identity_sha256=family_vector[
                "family_vector_identity_sha256"
            ],
            dedup_authority=_dedup_authority(mismatched_evidence),
            expected_dedup_worker_id=DEDUP_WORKER_ID,
            expected_dedup_head_sha=DEDUP_HEAD_SHA,
            expected_dedup_evidence_identity_sha256=mismatched_evidence,
        )


def test_adapter_output_executes_through_canonical_next100_gate(
    tmp_path: Path,
) -> None:
    family_vector = _build(tmp_path)
    adapted = adapt_family_vector_to_next100_106(
        family_vector,
        expected_family_vector_identity_sha256=family_vector[
            "family_vector_identity_sha256"
        ],
        dedup_authority=_dedup_authority(),
        expected_dedup_worker_id=DEDUP_WORKER_ID,
        expected_dedup_head_sha=DEDUP_HEAD_SHA,
        expected_dedup_evidence_identity_sha256=DEDUP_EVIDENCE_SHA,
    )

    normalized = next100_gate.validate_vector(adapted)
    assert {row["family_id"] for row in normalized} == {
        "ua.family.a",
        "ua.family.b",
        "en.family.a",
        "code.family.a",
    }

    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    result = next100_gate.evaluate(policy, adapted)
    assert result["dedup_authority"] == _dedup_authority()
    assert (
        result["claim_boundary"]["authorized_training_exposure_loss_positions"]
        == 0
    )
    assert result["claim_boundary"]["tokenizer_fit_authorized"] is False
    assert result["claim_boundary"]["model_training_authorized"] is False
