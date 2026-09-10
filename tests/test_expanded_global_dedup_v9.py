from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.expanded_global_dedup_v9 import (
    DATA526_BYTES,
    DATA526_SOURCES,
    ExpandedDedupError,
    RADA_DATASET,
    RADA_FAMILY,
    RADA_INPUT_RECORDS,
    RADA_QP_SAFE_RESULT,
    RADA_QP_SCHEMA,
    RADA_REVISION,
    RADA_SOURCE_BYTES,
    _derive_survivors,
    build_rada_matcher_inputs,
    filter_v8_survivor_inputs,
    validate_data526_authority,
    validate_language_authority,
    validate_rada_quality_privacy_report,
    validate_rada_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return raw + (b"\n" if newline else b"")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def row(
    record_id: str,
    text: str,
    *,
    status: str = "RETAIN_ALL",
    parent_id: str | None = None,
    index: int | None = None,
) -> dict[str, object]:
    raw = text.encode()
    digest = sha(raw)
    parent = parent_id or record_id
    return {
        "record_id": record_id,
        "quality_parent_record_id": parent,
        "source_dataset": RADA_DATASET,
        "source_revision": RADA_REVISION,
        "source_family": RADA_FAMILY,
        "source_path": f"texts/{parent}.txt",
        "source_payload_sha256": "b" * 64,
        "source_payload_bytes": 100,
        "quality_privacy_complete": True,
        "language_quality_privacy_complete": False,
        "privacy_gate_action": "ALLOW",
        "training_eligible": False,
        "evaluation_eligible": False,
        "global_dedup_complete": False,
        "reserved_evaluation_decontamination_complete": False,
        "quality_gate_status": status,
        "quality_unit_kind": (
            "SOURCE_NATIVE_DOCUMENT"
            if status == "RETAIN_ALL"
            else "BOUNDED_NATURAL_LANGUAGE_WINDOW"
        ),
        "quality_window_index": index,
        "quality_unit_utf8_sha256": digest,
        "decoded_text_utf8_sha256": digest,
        "privacy_scan_input_sha256": digest,
        "quality_unit_utf8_bytes": len(raw),
        "decoded_text_utf8_bytes": len(raw),
        "privacy_scan_input_bytes": len(raw),
        "text": text,
    }


def qp_fixture() -> tuple[list[dict[str, object]], bytes, dict[str, object], str]:
    rows = [
        row("parent-a", "alpha"),
        row(
            "parent-b#quality-window-0000:fixture",
            "beta",
            status="RETAIN_PARTIAL",
            parent_id="parent-b",
            index=0,
        ),
    ]
    raw = b"".join(canonical(item, newline=True) for item in rows)
    output_bytes = sum(len(str(item["text"]).encode()) for item in rows)
    core: dict[str, object] = {
        "schema_version": RADA_QP_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "safe_result": RADA_QP_SAFE_RESULT,
        "authority_binding": {
            "candidate_record_count": RADA_INPUT_RECORDS,
            "candidate_source_payload_bytes": RADA_SOURCE_BYTES,
            "candidate_exact_keyset_enforced": True,
            "unknown_candidate_fields_rejected": True,
            "executed_privacy_mechanics_pinned": True,
            "privacy_filter_v3_git_blob_sha": "a" * 40,
        },
        "input_records": RADA_INPUT_RECORDS,
        "output_records": len(rows),
        "output_jsonl_sha256": sha(raw),
        "output_text_utf8_bytes": output_bytes,
        "quality": {
            "statuses": {
                "RETAIN_ALL": 1,
                "RETAIN_PARTIAL": 1,
                "REJECT_DOCUMENT": RADA_INPUT_RECORDS - 2,
            },
            "retained_units_before_privacy": 3,
            "retained_utf8_bytes": output_bytes + 7,
            "rejected_utf8_bytes": 10,
            "partial_records_materialized": 1,
            "partial_units_materialized_before_privacy": 1,
        },
        "privacy": {
            "allowed_units": len(rows),
            "allowed_utf8_bytes": output_bytes,
            "held_units": 1,
            "held_utf8_bytes": 7,
            "quality_retained_bytes_reconciled": True,
            "quality_retained_units_reconciled": True,
            "non_allow_payload_mutation_attempted": False,
        },
        "claim_boundary": {
            "upstream_handoff_authority_bound": True,
            "candidate_schema_exact": True,
            "canonical_privacy_repair_bound": True,
            "global_dedup_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
    }
    report_sha = sha(canonical(core, newline=True))
    return rows, raw, {**core, "report_sha256": report_sha}, report_sha


def test_committed_data526_authority_reproduces_exact_digests() -> None:
    evidence = json.loads(
        (ROOT / "evidence/data526/v8/materialization_evidence.json").read_text()
    )
    inventory = json.loads(
        (ROOT / "evidence/data526/v8/record_inventory.json").read_text()
    )
    validate_data526_authority(evidence, inventory)
    assert evidence["source_object_count"] == DATA526_SOURCES
    assert evidence["total_payload_bytes"] == DATA526_BYTES


def test_data526_inventory_tamper_fails_closed() -> None:
    evidence = json.loads(
        (ROOT / "evidence/data526/v8/materialization_evidence.json").read_text()
    )
    inventory = json.loads(
        (ROOT / "evidence/data526/v8/record_inventory.json").read_text()
    )
    inventory["records"][0]["payload_bytes"] += 1
    with pytest.raises(ExpandedDedupError):
        validate_data526_authority(evidence, inventory)


def test_committed_rada_language_authority_is_exact_terminal_pass() -> None:
    report = json.loads(
        (
            ROOT
            / "evidence/d03-rada-trees/secondary-plaintext-language-gate-v1.json"
        ).read_text()
    )
    validate_language_authority(report)


def test_rada_language_authority_tamper_fails_closed() -> None:
    report = json.loads(
        (
            ROOT
            / "evidence/d03-rada-trees/secondary-plaintext-language-gate-v1.json"
        ).read_text()
    )
    report["language_result"]["language_reject_members"] = 1
    with pytest.raises(ExpandedDedupError):
        validate_language_authority(report)


def test_rada_quality_privacy_report_and_rows_validate() -> None:
    rows, raw, report, report_sha = qp_fixture()
    assert (
        validate_rada_quality_privacy_report(
            report,
            expected_report_sha256=report_sha,
        )
        == report_sha
    )
    validate_rada_rows(rows, raw, report)


def test_rada_report_requires_independent_expected_identity() -> None:
    _, _, report, _ = qp_fixture()
    with pytest.raises(ExpandedDedupError, match="identity mismatch"):
        validate_rada_quality_privacy_report(
            report,
            expected_report_sha256="f" * 64,
        )


def test_partial_raw_parent_substitution_fails_closed() -> None:
    rows, raw, report, report_sha = qp_fixture()
    validate_rada_quality_privacy_report(report, expected_report_sha256=report_sha)
    bad = copy.deepcopy(rows)
    bad[1]["record_id"] = bad[1]["quality_parent_record_id"]
    bad_raw = b"".join(canonical(item, newline=True) for item in bad)
    report["output_jsonl_sha256"] = sha(bad_raw)
    core = dict(report)
    core.pop("report_sha256")
    report["report_sha256"] = sha(canonical(core, newline=True))
    with pytest.raises(ExpandedDedupError, match="raw parent substituted"):
        validate_rada_rows(bad, bad_raw, report)


def test_non_allow_rada_unit_cannot_enter_matcher_input() -> None:
    rows, _, report, _ = qp_fixture()
    bad = copy.deepcopy(rows)
    bad[0]["privacy_gate_action"] = "REDACT"
    bad_raw = b"".join(canonical(item, newline=True) for item in bad)
    report["output_jsonl_sha256"] = sha(bad_raw)
    with pytest.raises(ExpandedDedupError, match="non-ALLOW"):
        validate_rada_rows(bad, bad_raw, report)


def test_rada_matcher_inputs_use_unit_scoped_origins() -> None:
    rows, _, _, report_sha = qp_fixture()
    inventory, payloads = build_rada_matcher_inputs(
        rows,
        authority_report_sha256=report_sha,
    )
    assert len(inventory) == len(payloads) == 2
    assert inventory[0]["stable_origin_id"] != inventory[1]["stable_origin_id"]
    assert all(item["source_family"] == RADA_FAMILY for item in inventory)
    assert all(item["modality"] == "uk" for item in inventory)


def test_filter_v8_survivors_rejects_hash_drift() -> None:
    payload = b"base"
    digest = sha(payload)
    inventory = {
        "sources": [
            {
                "source_id": "base-1",
                "source_family": "family",
                "modality": "en",
                "declared_capacity_bytes": len(payload),
            }
        ]
    }
    authority = {
        "survivors": [
            {
                "source_id": "base-1",
                "source_family": "family",
                "modality": "en",
                "declared_capacity_bytes": len(payload),
                "verified_raw_sha256": digest,
            }
        ],
        "post_dedup_declared_capacity_bytes": len(payload),
    }
    selected, selected_payloads = filter_v8_survivor_inputs(
        inventory,
        {"base-1": payload},
        authority,
    )
    assert selected[0]["source_id"] == "base-1"
    assert selected_payloads == {"base-1": payload}
    authority["survivors"][0]["verified_raw_sha256"] = "0" * 64
    with pytest.raises(ExpandedDedupError, match="raw hash drift"):
        filter_v8_survivor_inputs(inventory, {"base-1": payload}, authority)


def test_survivor_derivation_matches_incumbent_selection_rule() -> None:
    def source(source_id: str, capacity: int) -> dict[str, object]:
        return {"source_id": source_id, "declared_capacity_bytes": capacity}

    core = {
        "sources": [source("a", 10), source("b", 20), source("c", 20), source("d", 5)],
        "terminal_candidates": {
            "duplicate_clusters": [["a", "b"], ["c", "d"]],
            "declared_capacity_bytes_before": 55,
            "conservative_unique_capacity_bytes_after": 40,
            "duplicate_discount_bytes": 15,
            "duplicate_cluster_count": 2,
        },
    }
    matcher_sha = sha(canonical(core))
    authority = _derive_survivors({**core, "report_sha256": matcher_sha})
    assert authority["survivor_source_ids"] == ["b", "c"]
    assert authority["post_dedup_declared_capacity_bytes"] == 40
    assert authority["truth_boundary"]["training_authorized_bytes"] == 0
