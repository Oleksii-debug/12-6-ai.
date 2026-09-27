from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data import rada_two_clean_dedup_execution as carrier


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    payload = _canonical(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _authority() -> dict[str, object]:
    return {
        "schema_version": carrier.AUTHORITY_SCHEMA,
        "local_free_only": True,
        "current_main_consumable_rada_intake": True,
        "rada_intake_pr": 1787,
        "rada_intake_head_sha": "1" * 40,
        "rada_intake_audit_ref": "#2062 PASS",
        "rada_adapter_module": "twelve_six.data.rada_laws_qp_dedup_adapter",
        "rada_adapter_git_blob_sha1": "2" * 40,
        "matcher_pr": 1459,
        "matcher_head_sha": "3" * 40,
        "matcher_audit_ref": "#2018 PASS",
        "matcher_different_worker_pass": True,
        "indexed_module": "twelve_six.data.incumbent_dedup_indexed_execution",
        "indexed_module_git_blob_sha1": "4" * 40,
        "v3_module": "authority_runtime.cross_source_capacity_audit_v3",
        "canonical_capacity_credit": 0,
        "authorized_optimized_target_exposure": 0,
        "training_executed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }


def _source(source_id: str, capacity: int) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_family": "ua.rada.open-data.laws-texts",
        "modality": "uk",
        "declared_capacity_bytes": capacity,
        "verified_raw_sha256": "a" * 64,
        "normalized_sha256": "b" * 64,
        "stable_origin_id_sha256": "c" * 64,
        "stable_object_id_sha256": "d" * 64,
    }


def _report() -> dict[str, object]:
    return {
        "report_sha256": "e" * 64,
        "sources": [
            _source("a", 10),
            _source("b", 20),
            _source("c", 20),
        ],
        "terminal_candidates": {
            "duplicate_clusters": [["b", "a"]],
            "declared_capacity_bytes_before": 50,
            "conservative_unique_capacity_bytes_after": 40,
            "duplicate_discount_bytes": 10,
        },
    }


def _receipt(run_id: str, *, report_sha: str = "e" * 64) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": carrier.RECEIPT_SCHEMA,
        "run_id": run_id,
        "local_free_only": True,
        "completed": True,
        "dependency_authority_raw_sha256": "1" * 64,
        "inventory_raw_sha256": "2" * 64,
        "source_object_count": 3,
        "source_payload_utf8_bytes": 50,
        "v3_report_sha256": report_sha,
        "survivor_authority_sha256": "3" * 64,
        "survivor_source_object_count": 2,
        "survivor_declared_capacity_bytes": 40,
        "match_wall_clock_seconds": 1.0,
        "total_wall_clock_seconds": 2.0,
        "source_objects_per_match_second": 3.0,
        "payload_bytes_per_match_second": 50.0,
        "process_max_rss_kib": 100,
        "resource_measurement_complete": True,
        "raw_text_emitted": False,
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    return {
        **core,
        "receipt_identity_sha256": hashlib.sha256(_canonical(core)).hexdigest(),
    }


def test_dependency_authority_requires_external_raw_hash_and_terminal_gates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    value = _authority()
    identity = _write_json(path, value)

    observed = carrier.validate_dependency_authority(
        path,
        expected_raw_sha256=identity,
    )
    assert observed == value

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="raw SHA-256 drift"):
        carrier.validate_dependency_authority(path, expected_raw_sha256="f" * 64)

    value["matcher_different_worker_pass"] = False
    changed = _write_json(path, value)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="different-worker PASS"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=changed)


def test_dependency_authority_rejects_boolean_integer_and_extra_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    value = _authority()
    value["rada_intake_pr"] = True
    identity = _write_json(path, value)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="positive exact int"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)

    value = _authority()
    value["unexpected"] = "not allowed"
    identity = _write_json(path, value)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="schema drift"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)


def test_dependency_authority_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    path.write_text(
        '{"schema_version":"x","schema_version":"y"}',
        encoding="utf-8",
    )
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="duplicate JSON key"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)


def test_survivor_authority_reuses_incumbent_capacity_representative_rule() -> None:
    authority = carrier.derive_survivor_authority(_report())

    assert authority["selection_rule"] == carrier.SELECTION_RULE
    assert authority["post_dedup_survivor_source_object_count"] == 2
    assert authority["post_dedup_declared_capacity_bytes"] == 40
    assert [row["source_id"] for row in authority["survivors"]] == ["b", "c"]
    assert authority["duplicate_clusters"] == [
        {
            "member_source_ids": ["a", "b"],
            "selected_source_id": "b",
            "selected_declared_capacity_bytes": 20,
        }
    ]
    assert authority["raw_text_emitted"] is False
    assert authority["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert "text" not in json.dumps(authority, sort_keys=True)


def test_survivor_authority_rejects_overlapping_clusters() -> None:
    report = _report()
    terminal = report["terminal_candidates"]
    assert isinstance(terminal, dict)
    terminal["duplicate_clusters"] = [["a", "b"], ["b", "c"]]

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="overlap"):
        carrier.derive_survivor_authority(report)


def test_survivor_authority_rejects_capacity_summary_drift() -> None:
    report = _report()
    terminal = report["terminal_candidates"]
    assert isinstance(terminal, dict)
    terminal["conservative_unique_capacity_bytes_after"] = 41

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="capacity"):
        carrier.derive_survivor_authority(report)


def test_two_clean_authority_requires_two_distinct_reproducible_receipts() -> None:
    first = _receipt("clean-a")
    second = _receipt("clean-b")

    authority = carrier.build_two_clean_authority(first, second)
    assert authority["run_receipt_identities"] == [
        first["receipt_identity_sha256"],
        second["receipt_identity_sha256"],
    ]
    assert authority["v3_report_sha256"] == "e" * 64
    assert authority["survivor_source_object_count"] == 2
    assert authority["canonical_capacity_credited"] == 0
    assert authority["authorized_optimized_target_exposure"] == 0
    assert authority["training_executed"] is False

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="run ids"):
        carrier.build_two_clean_authority(first, _receipt("clean-a"))


def test_two_clean_authority_rejects_report_drift_and_receipt_tampering() -> None:
    first = _receipt("clean-a")
    second = _receipt("clean-b", report_sha="f" * 64)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="v3_report_sha256"):
        carrier.build_two_clean_authority(first, second)

    tampered = _receipt("clean-b")
    tampered["source_object_count"] = 4
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="self-hash"):
        carrier.build_two_clean_authority(first, tampered)
