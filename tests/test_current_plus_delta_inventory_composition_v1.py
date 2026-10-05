from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data import current_plus_delta_inventory_composition_v1 as compose
from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _inventory(rows: list[dict]) -> dict:
    rows = sorted(copy.deepcopy(rows), key=lambda row: row["record_id"])
    payload = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    return {
        "schema_version": compose.INVENTORY_SCHEMA,
        "record_count": len(rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "record_inventory_digest_sha256": _sha256(rows),
        "payload_inventory_digest_sha256": _sha256(payload),
        "records": rows,
    }


def _row(
    record_id: str,
    source_id: str,
    family: str,
    payload: bytes,
    modality: str = "code",
) -> dict:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "family": family,
        "modality": modality,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
    }


def _delta_evidence(rows: list[dict], *, head: str = "1" * 40) -> dict:
    inventory = _inventory(rows)
    survivor_bytes = inventory["total_payload_bytes"]
    survivor_records = inventory["record_count"]
    survivor_sources = len({row["source_id"] for row in inventory["records"]})
    loss = compose.EXPECTED_DELTA_PRE_GATE_BYTES - survivor_bytes
    assert loss >= 0
    core = {
        "schema_version": compose.DELTA_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": head,
        "parent": {
            "execution_head_sha": compose.EXPECTED_DELTA_PARENT_HEAD,
            "workflow_run_id": 123,
            "artifact_id": 456,
            "artifact_zip_sha256": "a" * 64,
            "combined_matcher_report_sha256": (
                compose.EXPECTED_DELTA_PARENT_MATCHER_REPORT_SHA256
            ),
            "survivor_authority_sha256": (
                compose.EXPECTED_DELTA_PARENT_SURVIVOR_AUTHORITY_SHA256
            ),
            "two_clean_proof_identity_sha256": (
                compose.EXPECTED_DELTA_PARENT_PROOF_IDENTITY_SHA256
            ),
            "marginal_global_unique_reserve_bytes": (
                compose.EXPECTED_DELTA_PRE_GATE_BYTES
            ),
            "projected_late_gate_headroom_before_execution": 82_734,
        },
        "delta_authority": {
            "source_object_count": compose.EXPECTED_DELTA_PRE_GATE_OBJECTS,
            "pre_gate_payload_bytes": compose.EXPECTED_DELTA_PRE_GATE_BYTES,
            "inventory_identity_sha256": "b" * 64,
            "subset_authority_sha256": "c" * 64,
            "training_records_sha256": "d" * 64,
            "training_handoff_raw_sha256": "e" * 64,
            "training_handoff_identity_sha256": "f" * 64,
            "source_ids_sha256": "0" * 64,
            "source_authority": {
                "code4_observed_summary_sha256": "1" * 64,
                "flask_physical_report_sha256": "2" * 64,
                "flask_snapshot_manifest_sha256": "3" * 64,
            },
        },
        "gate_execution": {
            "data232_report_sha256": "4" * 64,
            "decontamination_execution_identity_sha256": "5" * 64,
            "eval647_execution_receipt_identity_sha256": "6" * 64,
            "quality_execution_identity_sha256": "7" * 64,
            "privacy_execution_identity_sha256": "8" * 64,
            "composition_receipt_identity_sha256": "9" * 64,
            "input_records": compose.EXPECTED_DELTA_PRE_GATE_OBJECTS,
            "post_decontamination_records": survivor_records,
            "post_quality_records": survivor_records,
            "survivor_records": survivor_records,
            "survivor_source_objects": survivor_sources,
            "survivor_payload_bytes": survivor_bytes,
            "survivor_jsonl_sha256": "a" * 64,
            "survivor_record_inventory_digest_sha256": inventory[
                "record_inventory_digest_sha256"
            ],
            "survivor_payload_inventory_digest_sha256": inventory[
                "payload_inventory_digest_sha256"
            ],
            "rejection_counts": {
                "data232_excluded_records": (
                    compose.EXPECTED_DELTA_PRE_GATE_OBJECTS - survivor_records
                ),
                "g05_reject_documents": 0,
                "g05_partial_documents": 0,
                "g05_rejected_units": 0,
                "g05_rejected_utf8_bytes": 0,
                "g06_redacted_records": 0,
                "g06_quarantine_records": 0,
                "g06_exclude_records": 0,
                "g06_dropped_utf8_bytes": 0,
            },
            "privacy_detector_counts": {"synthetic_test_detector": 0},
            "later_gate_loss_bytes": loss,
        },
        "survivor_inventory": inventory,
        "capacity_projection": {
            "current_post_qp_code_bytes_reference": compose.CURRENT_POST_QP_CODE_BYTES,
            "code_target_bytes": compose.CODE_TARGET_BYTES,
            "pre_gate_gap_bytes": (
                compose.CODE_TARGET_BYTES - compose.CURRENT_POST_QP_CODE_BYTES
            ),
            "qualified_delta_survivor_bytes": survivor_bytes,
            "projected_post_qp_code_bytes": (
                compose.CURRENT_POST_QP_CODE_BYTES + survivor_bytes
            ),
            "projected_headroom_after_decontam_g05_g06": (
                compose.CURRENT_POST_QP_CODE_BYTES
                + survivor_bytes
                - compose.CODE_TARGET_BYTES
            ),
            "still_possible_to_reach_code_target_after_balance_split_pack": (
                compose.CURRENT_POST_QP_CODE_BYTES + survivor_bytes
                >= compose.CODE_TARGET_BYTES
            ),
        },
        "authority_blobs": {"synthetic_test_dependency.py": "a" * 40},
        "content_boundary": copy.deepcopy(compose._EXPECTED_CONTENT_BOUNDARY),
        "truth_boundary": copy.deepcopy(compose._EXPECTED_TRUTH_BOUNDARY),
    }
    return {**core, "evidence_identity_sha256": _sha256(core)}


def _base() -> dict:
    return _inventory(
        [
            _row(
                "base-01",
                "source-base-01",
                "github:pallets/flask",
                b"base flask physical bytes",
            ),
            _row(
                "base-02",
                "source-base-02",
                "github:agronholm/anyio",
                b"base anyio physical bytes",
            ),
        ]
    )


def _delta(
    *,
    record_id: str = "delta-01",
    source_id: str = "source-delta-01",
    family: str = "github:pydantic/pydantic",
    payload: bytes = b"new pydantic physical bytes",
) -> dict:
    return _delta_evidence([_row(record_id, source_id, family, payload)])


def _compose(base: dict, delta: dict) -> dict:
    return compose.compose_current_clean_and_delta_inventory(
        base_inventory=base,
        delta_evidence=delta,
        expected_base_physical_authority_identity_sha256="b" * 64,
        expected_base_record_count=base["record_count"],
        expected_base_total_payload_bytes=base["total_payload_bytes"],
        expected_base_source_object_count=len(
            {row["source_id"] for row in base["records"]}
        ),
        expected_base_record_inventory_digest_sha256=base[
            "record_inventory_digest_sha256"
        ],
        expected_base_payload_inventory_digest_sha256=base[
            "payload_inventory_digest_sha256"
        ],
        expected_delta_evidence_identity_sha256=delta["evidence_identity_sha256"],
        expected_delta_execution_head_sha=delta["execution_head_sha"],
        source_git_sha="2" * 40,
    )


def test_compose_binds_two_physical_inventories_and_family_capacity() -> None:
    base = _base()
    delta = _delta()
    result = _compose(base, delta)

    assert result["status"] == "COMPOSED_ZERO_CREDIT_PENDING_BALANCE"
    assert result["combined_inventory"]["record_count"] == 3
    assert result["combined_inventory"]["total_payload_bytes"] == (
        base["total_payload_bytes"]
        + delta["survivor_inventory"]["total_payload_bytes"]
    )
    assert result["stratum_capacity_bytes"] == {
        "code": result["combined_inventory"]["total_payload_bytes"],
        "en": 0,
        "uk": 0,
    }
    assert {row["family"] for row in result["families"]} == {
        "github:agronholm/anyio",
        "github:pallets/flask",
        "github:pydantic/pydantic",
    }
    assert result["canonical_capacity_credited"] == 0
    assert result["training_authorized_bytes"] == 0
    assert result["tokenizer_fit_authorized"] is False
    assert result["training_executed"] is False
    assert (
        compose.verify_current_clean_and_delta_composition(
            result,
            expected_composition_identity_sha256=result[
                "composition_identity_sha256"
            ],
        )
        == result["composition_identity_sha256"]
    )


def test_coherently_resealed_delta_cannot_replace_external_identity() -> None:
    base = _base()
    delta = _delta()
    expected = delta["evidence_identity_sha256"]
    delta["truth_boundary"]["family_caps_complete"] = True
    delta["evidence_identity_sha256"] = _sha256(
        {
            key: value
            for key, value in delta.items()
            if key != "evidence_identity_sha256"
        }
    )
    with pytest.raises(ProjectionError, match="not independently expected"):
        compose.compose_current_clean_and_delta_inventory(
            base_inventory=base,
            delta_evidence=delta,
            expected_base_physical_authority_identity_sha256="b" * 64,
            expected_base_record_count=base["record_count"],
            expected_base_total_payload_bytes=base["total_payload_bytes"],
            expected_base_source_object_count=2,
            expected_base_record_inventory_digest_sha256=base[
                "record_inventory_digest_sha256"
            ],
            expected_base_payload_inventory_digest_sha256=base[
                "payload_inventory_digest_sha256"
            ],
            expected_delta_evidence_identity_sha256=expected,
            expected_delta_execution_head_sha=delta["execution_head_sha"],
            source_git_sha="2" * 40,
        )


def test_resealed_delta_cannot_drop_pre_gate_input_record() -> None:
    base = _base()
    delta = _delta()
    delta["gate_execution"]["input_records"] = (
        compose.EXPECTED_DELTA_PRE_GATE_OBJECTS - 1
    )
    delta["evidence_identity_sha256"] = _sha256(
        {
            key: value
            for key, value in delta.items()
            if key != "evidence_identity_sha256"
        }
    )

    with pytest.raises(ProjectionError, match="input record count drift"):
        _compose(base, delta)


def test_resealed_delta_cannot_break_clean_gate_rejection_arithmetic() -> None:
    base = _base()
    delta = _delta()
    delta["gate_execution"]["rejection_counts"]["data232_excluded_records"] -= 1
    delta["evidence_identity_sha256"] = _sha256(
        {
            key: value
            for key, value in delta.items()
            if key != "evidence_identity_sha256"
        }
    )

    with pytest.raises(ProjectionError, match="DATA-232 rejection/count accounting drift"):
        _compose(base, delta)


def test_resealed_delta_rejection_counts_remain_closed_world() -> None:
    base = _base()
    delta = _delta()
    del delta["gate_execution"]["rejection_counts"]["g05_partial_documents"]
    delta["evidence_identity_sha256"] = _sha256(
        {
            key: value
            for key, value in delta.items()
            if key != "evidence_identity_sha256"
        }
    )

    with pytest.raises(ProjectionError, match="rejection-count schema drift"):
        _compose(base, delta)


def test_record_id_collision_fails_closed() -> None:
    base = _base()
    delta = _delta(record_id="base-01")
    with pytest.raises(ProjectionError, match="record_id collision"):
        _compose(base, delta)


def test_source_id_collision_fails_closed() -> None:
    base = _base()
    delta = _delta(source_id="source-base-01")
    with pytest.raises(ProjectionError, match="source_id collision"):
        _compose(base, delta)


def test_post_gate_exact_payload_collision_fails_closed() -> None:
    base = _base()
    delta = _delta(payload=b"base flask physical bytes")
    with pytest.raises(ProjectionError, match="exact payload collision"):
        _compose(base, delta)


def test_unknown_delta_family_fails_at_family_projection() -> None:
    base = _base()
    delta = _delta(family="github:example/not-authorized")
    with pytest.raises(ProjectionError, match="absent from current authority"):
        _compose(base, delta)


def test_composition_verifier_rejects_membership_origin_swap() -> None:
    result = _compose(_base(), _delta())
    delta_entry = next(
        row for row in result["inventory_membership"] if row["origin"] == "delta"
    )
    delta_entry["origin"] = "base"
    result["composition_identity_sha256"] = _sha256(
        {
            key: value
            for key, value in result.items()
            if key != "composition_identity_sha256"
        }
    )
    with pytest.raises(ProjectionError, match="base membership count drift"):
        compose.verify_current_clean_and_delta_composition(result)


def test_composition_verifier_rejects_capacity_tamper() -> None:
    result = _compose(_base(), _delta())
    result["families"][0]["capacity_bytes"] += 1
    result["composition_identity_sha256"] = _sha256(
        {
            key: value
            for key, value in result.items()
            if key != "composition_identity_sha256"
        }
    )
    with pytest.raises(ProjectionError, match="family rows drift"):
        compose.verify_current_clean_and_delta_composition(result)
