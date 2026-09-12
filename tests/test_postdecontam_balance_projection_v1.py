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
    ProjectionError,
    QUALITY_GRANULARITY_IDENTITY_SHA256,
    QUALITY_POLICY_IDENTITY_SHA256,
    build_family_vector,
    read_records_jsonl,
    verify_family_vector,
)
from twelve_six.data.postdecontam_next100_adapter_v1 import (
    adapt_family_vector_to_next100_106,
)
from twelve_six.data.trusted_family_authority_v1 import (
    TRUSTED_FAMILY_SEMANTICS,
    trusted_family_authority_root_sha256,
)


SHA = "a" * 64
PRIVACY_SHA = "e" * 64
DEDUP_EVIDENCE_SHA = "d" * 64
GIT_SHA = "b" * 40
DEDUP_HEAD_SHA = "c" * 40
DEDUP_WORKER_ID = "D03-GLOBAL-DEDUP"

UK_A = "ua.rada.open-data.laws-texts"
UK_B = "ua.kmu.portal.secretariat-news"
EN_A = "en.standardebooks.manual"
CODE_A = "github:encode/httpx"


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
            "record_id": "uk-1",
            "source_id": "src-uk-1",
            "family": UK_A,
            "modality": "text",
            "normalized_payload": "Привіт світе",
            "training_eligible": False,
            "evaluation_eligible": False,
        },
        {
            "record_id": "uk-2",
            "source_id": "src-uk-2",
            "family": UK_B,
            "modality": "text",
            "normalized_payload": "Ще український текст",
        },
        {
            "record_id": "en-1",
            "source_id": "src-en-1",
            "family": EN_A,
            "modality": "text",
            "normalized_payload": "Hello world",
        },
        {
            "record_id": "code-1",
            "source_id": "src-code-1",
            "family": CODE_A,
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
        record.payload_bytes
        for record in records
        if record.record_id not in excluded_ids
    )
    return _with_hash(
        {
            "schema": BINDING_SCHEMA,
            "verdict": "PASS_WITH_EXCLUSIONS" if excluded_ids else "PASS_CLEAN",
            "retained_inventory_identity_sha256": SHA,
            "dedup_evidence_identity_sha256": DEDUP_EVIDENCE_SHA,
            "records_jsonl_sha256": records_sha,
            "input_record_count": len(records),
            "input_payload_bytes": sum(
                record.payload_bytes for record in records
            ),
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
    return [
        record for record in records if record.record_id not in excluded_ids
    ]


def _family_map(survivors: list) -> dict:
    families = sorted({record.family for record in survivors})
    return _with_hash(
        {
            "schema": FAMILY_MAP_SCHEMA,
            "families": [
                {
                    "family": family,
                    "stratum": TRUSTED_FAMILY_SEMANTICS[family]["stratum"],
                }
                for family in families
            ],
            "training_authorized_by_this_mapping": False,
        },
        "family_map_identity_sha256",
    )


def _family_provenance(survivors: list) -> dict:
    families = sorted({record.family for record in survivors})
    return _with_hash(
        {
            "schema": FAMILY_PROVENANCE_SCHEMA,
            "families": [
                copy.deepcopy(TRUSTED_FAMILY_SEMANTICS[family])
                for family in families
            ],
            "training_authorized_by_this_provenance": False,
        },
        "family_provenance_identity_sha256",
    )


def _coverage(
    records: list,
    records_sha: str,
    binding: dict,
    excluded_ids: list[str],
) -> dict:
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
            "covered_payload_bytes": sum(
                record.payload_bytes for record in survivors
            ),
            "final_test_outcomes_read": False,
            "authorized_optimized_target_exposure": 0,
            "training_authorized_by_this_coverage": False,
        },
        "g05_g06_coverage_identity_sha256",
    )


def _inputs(
    tmp_path: Path,
    excluded_ids: list[str] | None = None,
) -> dict:
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


def _build(
    tmp_path: Path,
    excluded_ids: list[str] | None = None,
) -> dict:
    return build_family_vector(**_inputs(tmp_path, excluded_ids))


def _dedup_authority(
    evidence_identity_sha256: str = DEDUP_EVIDENCE_SHA,
) -> dict:
    return {
        "worker_id": DEDUP_WORKER_ID,
        "head_sha": DEDUP_HEAD_SHA,
        "evidence_identity_sha256": evidence_identity_sha256,
        "terminal_verdict": "PASS",
    }


def test_builds_deterministic_text_free_trusted_projection(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path, ["uk-1"])
    second = _build(tmp_path, ["uk-1"])
    assert first == second
    assert verify_family_vector(first) == first["family_vector_identity_sha256"]
    assert first["excluded_record_count"] == 1
    assert first["survivor_record_count"] == 3
    assert first["authorized_optimized_target_exposure"] == 0
    expected_families = {UK_B, EN_A, CODE_A}
    assert first["trusted_family_authority_root_sha256"] == (
        trusted_family_authority_root_sha256(expected_families)
    )
    serialized = _canonical(first).decode("utf-8")
    assert "Привіт" not in serialized
    assert "Hello world" not in serialized
    assert "def f" not in serialized


def test_coherent_dual_reseal_cannot_reclassify_family(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    for document_key, expected_key in (
        ("family_map", "expected_family_map_identity_sha256"),
        (
            "family_provenance",
            "expected_family_provenance_identity_sha256",
        ),
    ):
        document = inputs[document_key]
        row = next(
            entry for entry in document["families"] if entry["family"] == EN_A
        )
        row["stratum"] = "uk"
        identity_field = (
            "family_map_identity_sha256"
            if document_key == "family_map"
            else "family_provenance_identity_sha256"
        )
        resealed = _with_hash(document, identity_field)
        inputs[document_key] = resealed
        inputs[expected_key] = resealed[identity_field]

    with pytest.raises(
        ProjectionError,
        match="trusted source-family authority",
    ):
        build_family_vector(**inputs)


def test_same_stratum_family_identity_swap_fails_closed(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    provenance = inputs["family_provenance"]
    first = next(
        row for row in provenance["families"] if row["family"] == UK_A
    )
    second = next(
        row for row in provenance["families"] if row["family"] == UK_B
    )
    (
        first["source_family_identity_sha256"],
        second["source_family_identity_sha256"],
    ) = (
        second["source_family_identity_sha256"],
        first["source_family_identity_sha256"],
    )
    provenance = _with_hash(
        provenance,
        "family_provenance_identity_sha256",
    )
    inputs["family_provenance"] = provenance
    inputs["expected_family_provenance_identity_sha256"] = provenance[
        "family_provenance_identity_sha256"
    ]
    with pytest.raises(
        ProjectionError,
        match="trusted source-family authority",
    ):
        build_family_vector(**inputs)


def test_cross_stratum_family_identity_swap_fails_closed(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    provenance = inputs["family_provenance"]
    en_row = next(
        row for row in provenance["families"] if row["family"] == EN_A
    )
    code_row = next(
        row for row in provenance["families"] if row["family"] == CODE_A
    )
    (
        en_row["source_family_identity_sha256"],
        code_row["source_family_identity_sha256"],
    ) = (
        code_row["source_family_identity_sha256"],
        en_row["source_family_identity_sha256"],
    )
    provenance = _with_hash(
        provenance,
        "family_provenance_identity_sha256",
    )
    inputs["family_provenance"] = provenance
    inputs["expected_family_provenance_identity_sha256"] = provenance[
        "family_provenance_identity_sha256"
    ]
    with pytest.raises(
        ProjectionError,
        match="trusted source-family authority",
    ):
        build_family_vector(**inputs)


def test_unknown_survivor_family_fails_before_vector(
    tmp_path: Path,
) -> None:
    row = {
        "record_id": "unknown-1",
        "source_id": "unknown-source",
        "family": "unknown.family",
        "modality": "text",
        "normalized_payload": "unknown",
    }
    path = tmp_path / "unknown.jsonl"
    path.write_bytes(_canonical(row) + b"\n")
    records, records_sha = read_records_jsonl(path)
    binding = _binding(records, records_sha, [])
    coverage = _coverage(records, records_sha, binding, [])
    family_map = _with_hash(
        {
            "schema": FAMILY_MAP_SCHEMA,
            "families": [
                {"family": "unknown.family", "stratum": "en"}
            ],
            "training_authorized_by_this_mapping": False,
        },
        "family_map_identity_sha256",
    )
    provenance = _with_hash(
        {
            "schema": FAMILY_PROVENANCE_SCHEMA,
            "families": [
                {
                    "family": "unknown.family",
                    "source_family_identity_sha256": "f" * 64,
                    "language": "en",
                    "modalities": ["text"],
                    "stratum": "en",
                }
            ],
            "training_authorized_by_this_provenance": False,
        },
        "family_provenance_identity_sha256",
    )
    with pytest.raises(
        ProjectionError,
        match="absent from trusted DATA526/V8 authority",
    ):
        build_family_vector(
            records=records,
            records_jsonl_sha256=records_sha,
            decontamination_binding=binding,
            g05_g06_coverage=coverage,
            expected_g05_g06_coverage_identity_sha256=coverage[
                "g05_g06_coverage_identity_sha256"
            ],
            expected_privacy_policy_identity_sha256=PRIVACY_SHA,
            family_map=family_map,
            expected_family_map_identity_sha256=family_map[
                "family_map_identity_sha256"
            ],
            family_provenance=provenance,
            expected_family_provenance_identity_sha256=provenance[
                "family_provenance_identity_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_family_vector_recomputes_trusted_root(tmp_path: Path) -> None:
    result = _build(tmp_path)
    tampered = copy.deepcopy(result)
    tampered["trusted_family_authority_root_sha256"] = "f" * 64
    tampered = _with_hash(tampered, "family_vector_identity_sha256")
    with pytest.raises(
        ProjectionError,
        match="trusted family authority root mismatch",
    ):
        verify_family_vector(tampered)


def test_unknown_decontamination_exclusion_fails_closed(
    tmp_path: Path,
) -> None:
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
def test_decontamination_truth_boundary_fails_closed(
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


def test_coverage_must_exactly_cover_survivors(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    coverage = inputs["g05_g06_coverage"]
    coverage["covered_records"] = coverage["covered_records"][:-1]
    coverage["covered_record_count"] -= 1
    coverage = _with_hash(
        coverage,
        "g05_g06_coverage_identity_sha256",
    )
    inputs["g05_g06_coverage"] = coverage
    inputs["expected_g05_g06_coverage_identity_sha256"] = coverage[
        "g05_g06_coverage_identity_sha256"
    ]
    with pytest.raises(ProjectionError, match="does not exactly cover"):
        build_family_vector(**inputs)


def test_candidate_provenance_modality_cannot_drift(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    provenance = inputs["family_provenance"]
    target = next(
        row for row in provenance["families"] if row["family"] == EN_A
    )
    target["modalities"] = ["code"]
    provenance = _with_hash(
        provenance,
        "family_provenance_identity_sha256",
    )
    inputs["family_provenance"] = provenance
    inputs["expected_family_provenance_identity_sha256"] = provenance[
        "family_provenance_identity_sha256"
    ]
    with pytest.raises(
        ProjectionError,
        match="trusted source-family authority",
    ):
        build_family_vector(**inputs)


def test_family_vector_self_hash_and_arithmetic_fail_closed(
    tmp_path: Path,
) -> None:
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


@pytest.mark.parametrize(
    "field",
    ["training_eligible", "evaluation_eligible"],
)
@pytest.mark.parametrize("value", [0, 0.0, None, "false", True])
def test_input_eligibility_markers_require_exact_false(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    row = {
        "record_id": "r",
        "source_id": "s",
        "family": EN_A,
        "modality": "text",
        "normalized_payload": "abc",
    }
    row[field] = value
    path = tmp_path / "eligible.jsonl"
    path.write_bytes(_canonical(row) + b"\n")
    match = (
        "training eligibility"
        if field == "training_eligible"
        else "evaluation eligibility"
    )
    with pytest.raises(ProjectionError, match=match):
        read_records_jsonl(path)


def test_payload_byte_and_hash_drift_fail_closed(tmp_path: Path) -> None:
    row = {
        "record_id": "r",
        "source_id": "s",
        "family": EN_A,
        "modality": "text",
        "normalized_payload": "abc",
        "normalized_payload_bytes": 4,
    }
    path = tmp_path / "bad.jsonl"
    path.write_bytes(_canonical(row) + b"\n")
    with pytest.raises(ProjectionError, match="bytes drift"):
        read_records_jsonl(path)

    row["normalized_payload_bytes"] = 3
    row["normalized_payload_sha256"] = "f" * 64
    path.write_bytes(_canonical(row) + b"\n")
    with pytest.raises(ProjectionError, match="sha256 drift"):
        read_records_jsonl(path)


def test_adapter_binds_dedup_lineage(tmp_path: Path) -> None:
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


def test_adapter_executes_through_canonical_balance_gate(
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
        UK_A,
        UK_B,
        EN_A,
        CODE_A,
    }

    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    result = next100_gate.evaluate(policy, adapted)
    assert result["dedup_authority"] == _dedup_authority()
    assert (
        result["claim_boundary"][
            "authorized_training_exposure_loss_positions"
        ]
        == 0
    )
    assert result["claim_boundary"]["tokenizer_fit_authorized"] is False
    assert result["claim_boundary"]["model_training_authorized"] is False
