from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_d03_rada_current_postdedup_handoff_v1.py"
SPEC = importlib.util.spec_from_file_location("rada_current_handoff_v1", TOOL)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def _canonical(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _seal(value: dict, field: str, *, newline: bool = False) -> dict:
    core = copy.deepcopy(value)
    core.pop(field, None)
    value[field] = _sha(_canonical(core, newline=newline))
    return value


def _source(source_id: str, payload: bytes, *, declared: int) -> dict:
    digest = _sha(payload)
    return {
        "source_id": source_id,
        "source_family": f"family.{source_id}",
        "modality": "uk",
        "declared_capacity_bytes": declared,
        "comparison_policy": "DATA232_GENERIC_FROM_RAW",
        "comparison_payload_bytes": len(payload),
        "comparison_payload_sha256": digest,
        "verified_raw_sha256": digest,
        "stable_origin_id_sha256": _sha(f"origin:{source_id}".encode()),
        "stable_object_id_sha256": _sha(f"object:{source_id}".encode()),
    }


def _fixture() -> tuple[dict, dict, dict, dict[str, bytes]]:
    payloads = {
        "base": "base text".encode(),
        "rada": "Рада".encode(),
    }
    sources = [
        _source("base", payloads["base"], declared=11),
        _source("rada", payloads["rada"], declared=19),
    ]
    report = {
        "schema_version": m.REPORT_SCHEMA,
        "algorithm": "fixture",
        "source_count": 2,
        "sources": sources,
        "raw_text_emitted": False,
        "model_training_executed": False,
        "source_admission_authority": False,
        "terminal_candidates": {
            "conservative_unique_capacity_bytes_after": 30,
        },
    }
    _seal(report, "report_sha256", newline=True)

    selection = {
        "schema_version": "fixture-full-selection.v1",
        "pre_dedup_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": 30,
        "post_dedup_survivor_source_object_count": 2,
        "post_dedup_declared_capacity_bytes": 30,
        "duplicate_discount_bytes": 0,
        "duplicate_cluster_count": 0,
        "survivor_source_ids": ["base", "rada"],
    }
    _seal(selection, "survivor_authority_sha256")

    authority = {
        "schema_version": m.CURRENT_AUTHORITY_SCHEMA,
        "matcher_report_sha256": report["report_sha256"],
        "selection_projection_schema": selection["schema_version"],
        "selection_projection_sha256": selection["survivor_authority_sha256"],
        "pre_dedup_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": 30,
        "post_dedup_survivor_source_object_count": 2,
        "post_dedup_declared_capacity_bytes": 30,
        "duplicate_discount_bytes": 0,
        "duplicate_cluster_count": 0,
        "current_rada_survivor_source_object_count": 1,
        "current_rada_survivor_declared_capacity_bytes": 19,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
    }
    _seal(authority, "survivor_authority_sha256")
    return report, selection, authority, payloads


def _build() -> tuple[dict, dict[str, bytes], dict, dict, dict]:
    report, selection, authority, payloads = _fixture()
    inventory = m.build_retained_inventory(
        report,
        authority,
        selection,
        expected_report_sha256=report["report_sha256"],
        expected_current_authority_sha256=authority["survivor_authority_sha256"],
        expected_full_selection_sha256=selection["survivor_authority_sha256"],
    )
    return inventory, payloads, report, selection, authority


def _rehash_inventory(inventory: dict) -> None:
    _seal(inventory, "inventory_identity_sha256")


def test_builds_exact_full_survivor_inventory_and_handoff() -> None:
    inventory, payloads, _, selection, authority = _build()

    assert inventory["schema_version"] == m.INVENTORY_SCHEMA
    assert inventory["retained_source_count"] == 2
    assert inventory["retained_declared_capacity_bytes"] == 30
    assert inventory["current_rada_survivor_source_object_count"] == 1
    assert inventory["current_rada_survivor_declared_capacity_bytes"] == 19
    assert inventory["authorized_training_exposure"] == 0
    assert inventory["tokenizer_fit_authorized"] is False
    assert inventory["final_test_payload_read"] is False
    assert "text" not in json.dumps(inventory, ensure_ascii=False)

    rows, handoff = m.prepare_ephemeral_data232_rows(
        inventory,
        payloads,
        expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        expected_full_selection_sha256=selection["survivor_authority_sha256"],
        expected_current_authority_sha256=authority["survivor_authority_sha256"],
    )
    assert [row["record_id"] for row in rows] == ["base", "rada"]
    assert rows[1]["text"] == "Рада"
    assert handoff["schema_version"] == m.HANDOFF_SCHEMA
    assert (
        handoff["input_survivor_authority_sha256"]
        == selection["survivor_authority_sha256"]
    )
    assert handoff["retained_source_count"] == 2
    assert handoff["authorized_training_exposure"] == 0
    assert handoff["raw_text_persisted_in_evidence"] is False
    assert "Рада" not in json.dumps(handoff, ensure_ascii=False)


def test_rejects_self_consistent_report_substitution_against_external_identity() -> None:
    report, selection, authority, _ = _fixture()
    expected = report["report_sha256"]
    report["algorithm"] = "substituted"
    _seal(report, "report_sha256", newline=True)
    authority["matcher_report_sha256"] = report["report_sha256"]
    _seal(authority, "survivor_authority_sha256")

    with pytest.raises(
        m.CurrentRadaPostDedupHandoffError,
        match="independently expected identity",
    ):
        m.build_retained_inventory(
            report,
            authority,
            selection,
            expected_report_sha256=expected,
            expected_current_authority_sha256=authority["survivor_authority_sha256"],
            expected_full_selection_sha256=selection["survivor_authority_sha256"],
        )


def test_rejects_self_consistent_selection_substitution_against_external_identity() -> None:
    report, selection, authority, _ = _fixture()
    expected = selection["survivor_authority_sha256"]
    selection["schema_version"] = "substituted-selection.v1"
    _seal(selection, "survivor_authority_sha256")
    authority["selection_projection_schema"] = selection["schema_version"]
    authority["selection_projection_sha256"] = selection["survivor_authority_sha256"]
    _seal(authority, "survivor_authority_sha256")

    with pytest.raises(
        m.CurrentRadaPostDedupHandoffError,
        match="independently expected identity",
    ):
        m.build_retained_inventory(
            report,
            authority,
            selection,
            expected_report_sha256=report["report_sha256"],
            expected_current_authority_sha256=authority["survivor_authority_sha256"],
            expected_full_selection_sha256=expected,
        )


def test_rejects_current_authority_substitution_against_external_identity() -> None:
    report, selection, authority, _ = _fixture()
    expected = authority["survivor_authority_sha256"]
    authority["current_rada_survivor_declared_capacity_bytes"] = 18
    _seal(authority, "survivor_authority_sha256")

    with pytest.raises(
        m.CurrentRadaPostDedupHandoffError,
        match="independently expected identity",
    ):
        m.build_retained_inventory(
            report,
            authority,
            selection,
            expected_report_sha256=report["report_sha256"],
            expected_current_authority_sha256=expected,
            expected_full_selection_sha256=selection["survivor_authority_sha256"],
        )


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_rejects_payload_coverage_drift(mode: str) -> None:
    inventory, payloads, _, selection, authority = _build()
    changed = dict(payloads)
    if mode == "missing":
        changed.pop("base")
    else:
        changed["extra"] = b"extra"

    with pytest.raises(
        m.CurrentRadaPostDedupHandoffError,
        match="coverage must equal retained inventory",
    ):
        m.prepare_ephemeral_data232_rows(
            inventory,
            changed,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
            expected_full_selection_sha256=selection["survivor_authority_sha256"],
            expected_current_authority_sha256=authority["survivor_authority_sha256"],
        )


def test_rejects_payload_hash_drift() -> None:
    inventory, payloads, _, selection, authority = _build()
    payloads["base"] = b"base xxxx"

    with pytest.raises(
        m.CurrentRadaPostDedupHandoffError,
        match="payload identity drift",
    ):
        m.prepare_ephemeral_data232_rows(
            inventory,
            payloads,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
            expected_full_selection_sha256=selection["survivor_authority_sha256"],
            expected_current_authority_sha256=authority["survivor_authority_sha256"],
        )


def test_rejects_non_utf8_comparison_payload_after_exact_hash_binding() -> None:
    inventory, payloads, _, selection, authority = _build()
    payloads["base"] = b"\xff"
    first = next(row for row in inventory["retained_sources"] if row["source_id"] == "base")
    first["comparison_payload_bytes"] = 1
    first["comparison_payload_sha256"] = _sha(b"\xff")
    _rehash_inventory(inventory)

    with pytest.raises(
        m.CurrentRadaPostDedupHandoffError,
        match="not strict UTF-8",
    ):
        m.prepare_ephemeral_data232_rows(
            inventory,
            payloads,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
            expected_full_selection_sha256=selection["survivor_authority_sha256"],
            expected_current_authority_sha256=authority["survivor_authority_sha256"],
        )


def test_rejects_boolean_zero_credit_alias_in_resealed_inventory() -> None:
    inventory, payloads, _, selection, authority = _build()
    inventory["authorized_training_exposure"] = False
    _rehash_inventory(inventory)

    with pytest.raises(
        m.CurrentRadaPostDedupHandoffError,
        match="exact non-negative integer",
    ):
        m.prepare_ephemeral_data232_rows(
            inventory,
            payloads,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
            expected_full_selection_sha256=selection["survivor_authority_sha256"],
            expected_current_authority_sha256=authority["survivor_authority_sha256"],
        )


def test_rejects_resealed_inventory_substitution_against_external_identity() -> None:
    inventory, payloads, _, selection, authority = _build()
    expected = inventory["inventory_identity_sha256"]
    inventory["retained_sources"][0]["source_family"] = "family.substituted"
    _rehash_inventory(inventory)

    with pytest.raises(
        m.CurrentRadaPostDedupHandoffError,
        match="independently expected identity",
    ):
        m.prepare_ephemeral_data232_rows(
            inventory,
            payloads,
            expected_inventory_identity_sha256=expected,
            expected_full_selection_sha256=selection["survivor_authority_sha256"],
            expected_current_authority_sha256=authority["survivor_authority_sha256"],
        )
