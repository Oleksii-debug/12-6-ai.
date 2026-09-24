from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from twelve_six.data import clean_retained_handoff_v1 as bridge
from twelve_six.data import current_reserved_decontamination_v1 as decontam


def _cjson(value: object, *, newline: bool = False) -> bytes:
    return bridge._canonical_bytes(value, newline=newline)


def _proof_truth() -> dict[str, object]:
    return {
        "current_corpus_eligible": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "whole_corpus_external_llm_cleanliness_claimed": False,
    }


def _evidence_truth() -> dict[str, object]:
    return {
        "current_corpus_eligible": False,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }


def _records() -> list[dict[str, str]]:
    return [
        {
            "record_id": "record-a",
            "source_id": "source-a",
            "family": "family.clean.a",
            "modality": "uk",
            "normalized_payload": "альфа",
        },
        {
            "record_id": "record-b",
            "source_id": "source-b",
            "family": "family.clean.b",
            "modality": "code",
            "normalized_payload": "beta",
        },
    ]


def _record_bytes(records: list[dict[str, str]]) -> bytes:
    return b"".join(_cjson(row, newline=True) for row in records)


def _inventory(records: list[dict[str, str]]) -> dict[str, object]:
    rows = []
    for record in records:
        raw = record["normalized_payload"].encode("utf-8")
        rows.append(
            {
                "record_id": record["record_id"],
                "source_id": record["source_id"],
                "family": record["family"],
                "modality": record["modality"],
                "payload_sha256": bridge._sha(raw),
                "payload_bytes": len(raw),
            }
        )
    rows.sort(key=lambda row: str(row["record_id"]))
    projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    return {
        "schema_version": bridge.INVENTORY_SCHEMA,
        "record_count": len(rows),
        "total_payload_bytes": sum(int(row["payload_bytes"]) for row in rows),
        "record_inventory_digest_sha256": bridge._sha(_cjson(rows)),
        "payload_inventory_digest_sha256": bridge._sha(_cjson(projection)),
        "records": rows,
    }


def _evidence() -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": bridge.EVIDENCE_SCHEMA,
        "status": "MATERIALIZED_ZERO_CREDIT",
        "repeat_materialization_byte_identical": True,
        "provenance_guard": {
            "known_external_llm_contamination_absent": True,
            "quarantine_identity_sha256": "a" * 64,
            "whole_corpus_external_llm_cleanliness_claimed": False,
            "successor_authority_rebuild_required_for_invalidated_v5_v6_v8": True,
        },
        "remaining_materialization_blockers": [
            "SUCCESSOR_CORPUS_AUTHORITY_REBUILD_REQUIRED"
        ],
        "truth_boundary": _evidence_truth(),
    }
    result = copy.deepcopy(core)
    result["materialization_identity_sha256"] = bridge._sha(
        _cjson(core, newline=True)
    )
    return result


def _release_and_files() -> tuple[
    bridge._PhysicalRelease,
    bytes,
    bytes,
    bytes,
    bytes,
]:
    records = _records()
    records_raw = _record_bytes(records)
    inventory = _inventory(records)
    inventory_raw = _cjson(inventory, newline=True)
    evidence = _evidence()
    evidence_raw = _cjson(evidence, newline=True)

    release = bridge._PhysicalRelease(
        run_id=101,
        job_id=102,
        artifact_id=103,
        artifact_zip_sha256="1" * 64,
        authoritative_main_sha="2" * 40,
        execution_carrier_head_sha="3" * 40,
        record_count=len(records),
        payload_bytes=int(inventory["total_payload_bytes"]),
        jsonl_sha256=bridge._sha(records_raw),
        inventory_file_sha256=bridge._sha(inventory_raw),
        evidence_file_sha256=bridge._sha(evidence_raw),
        proof_file_sha256="4" * 64,
        record_inventory_digest_sha256=str(
            inventory["record_inventory_digest_sha256"]
        ),
        payload_inventory_digest_sha256=str(
            inventory["payload_inventory_digest_sha256"]
        ),
        materialization_identity_sha256=str(
            evidence["materialization_identity_sha256"]
        ),
        clean_data526_evidence_identity_sha256="5" * 64,
        g05_execution_identity_sha256="6" * 64,
        g06_execution_identity_sha256="7" * 64,
        g06_envelope_identity_sha256="8" * 64,
        g06_terminal_qualification_identity_sha256="9" * 64,
        composition_preflight_identity_sha256="a" * 64,
    )
    proof: dict[str, object] = {
        "schema": bridge.PHYSICAL_PROOF_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "authoritative_main_sha": release.authoritative_main_sha,
        "execution_carrier_head_sha": release.execution_carrier_head_sha,
        "input_record_count": 3,
        "input_payload_bytes": 100,
        "input_jsonl_sha256": "b" * 64,
        "input_rows_sha256": "c" * 64,
        "clean_data526_evidence_identity_sha256": (
            release.clean_data526_evidence_identity_sha256
        ),
        "g05_execution_identity_sha256": release.g05_execution_identity_sha256,
        "g06_execution_identity_sha256": release.g06_execution_identity_sha256,
        "g06_envelope_identity_sha256": release.g06_envelope_identity_sha256,
        "g06_terminal_qualification_identity_sha256": (
            release.g06_terminal_qualification_identity_sha256
        ),
        "composition_preflight_identity_sha256": (
            release.composition_preflight_identity_sha256
        ),
        "materializer_v1_git_blob_sha1": "d" * 40,
        "materializer_v2_git_blob_sha1": "e" * 40,
        "privacy_implementation_git_blob_sha1": "f" * 40,
        "privacy_policy_sha256": "d" * 64,
        "two_fresh_cli_processes": True,
        "payload_byte_identical": True,
        "inventory_byte_identical": True,
        "evidence_byte_identical": True,
        "output_record_count": release.record_count,
        "output_payload_bytes": release.payload_bytes,
        "output_jsonl_sha256": release.jsonl_sha256,
        "output_inventory_file_sha256": release.inventory_file_sha256,
        "output_evidence_file_sha256": release.evidence_file_sha256,
        "output_record_inventory_digest_sha256": (
            release.record_inventory_digest_sha256
        ),
        "output_payload_inventory_digest_sha256": (
            release.payload_inventory_digest_sha256
        ),
        "materialization_identity_sha256": release.materialization_identity_sha256,
        "known_nomis_pr462_payload_absent": True,
        "privacy_redactions_rescanned_allow": True,
        "truth_boundary": _proof_truth(),
    }
    proof_raw = _cjson(proof, newline=True)
    release = replace(release, proof_file_sha256=bridge._sha(proof_raw))
    return release, records_raw, inventory_raw, evidence_raw, proof_raw


def _reseal_evidence(evidence: dict[str, object]) -> bytes:
    core = copy.deepcopy(evidence)
    core.pop("materialization_identity_sha256", None)
    evidence["materialization_identity_sha256"] = bridge._sha(
        _cjson(core, newline=True)
    )
    return _cjson(evidence, newline=True)


def test_exact_clean_release_reuses_incumbent_data232_handoff_contract() -> None:
    release, records_raw, inventory_raw, evidence_raw, proof_raw = (
        _release_and_files()
    )
    rows, handoff, receipt = bridge._prepare_with_release(
        records_raw,
        inventory_raw,
        evidence_raw,
        proof_raw,
        release=release,
    )
    assert handoff["schema_version"] == bridge.HANDOFF_SCHEMA
    assert [row["record_id"] for row in rows] == ["record-a", "record-b"]
    assert set(rows[0]) == {
        "record_id",
        "source_id",
        "source_family",
        "modality",
        "text",
    }
    assert receipt["authorized_optimized_target_exposure"] == 0
    assert receipt["training_executed"] is False

    inventory_id, survivor_id, handoff_id = decontam._verify_training_handoff(
        rows,
        handoff,
        expected_inventory_identity_sha256=release.inventory_file_sha256,
        expected_survivor_authority_sha256=(
            release.clean_data526_evidence_identity_sha256
        ),
        expected_handoff_identity_sha256=handoff["handoff_identity_sha256"],
    )
    assert inventory_id == release.inventory_file_sha256
    assert survivor_id == release.clean_data526_evidence_identity_sha256
    assert handoff_id == handoff["handoff_identity_sha256"]


def test_production_release_constants_are_the_physical_success_result() -> None:
    assert bridge.PHYSICAL_RUN_ID == 36026689718
    assert bridge.PHYSICAL_JOB_ID == 107724922341
    assert bridge.PHYSICAL_ARTIFACT_ID == 10820342689
    assert bridge.OUTPUT_RECORD_COUNT == 257
    assert bridge.OUTPUT_PAYLOAD_BYTES == 5601716
    assert bridge.OUTPUT_JSONL_SHA256 == (
        "bbeb43b3b8e3e4b0e2631c16895d700896f3733d6cd6fe3833fe678594bc86d8"
    )
    assert bridge.OUTPUT_RECORD_INVENTORY_DIGEST_SHA256 == (
        "dbdf741884ec1f147827647908b17a846584b145454e3f82fdb63120422c6059"
    )
    assert bridge.OUTPUT_PAYLOAD_INVENTORY_DIGEST_SHA256 == (
        "2384480c89c19b14d188aa57130ee2967463512bb54241f90b6f17a525653a1e"
    )


def test_same_count_payload_substitution_fails_closed() -> None:
    release, _, inventory_raw, _, _ = _release_and_files()
    inventory_rows = bridge._validate_inventory(inventory_raw, release=release)
    changed = _records()
    changed[0]["normalized_payload"] = "омега"
    changed_raw = _record_bytes(changed)
    changed_release = replace(release, jsonl_sha256=bridge._sha(changed_raw))
    with pytest.raises(bridge.CleanRetainedHandoffError, match="payload SHA drift"):
        bridge._bind_records(changed_raw, inventory_rows, release=changed_release)


def test_source_provenance_aliasing_fails_closed() -> None:
    release, _, inventory_raw, _, _ = _release_and_files()
    inventory_rows = bridge._validate_inventory(inventory_raw, release=release)
    changed = _records()
    changed[0]["source_id"] = "forged-source"
    changed_raw = _record_bytes(changed)
    changed_release = replace(release, jsonl_sha256=bridge._sha(changed_raw))
    with pytest.raises(bridge.CleanRetainedHandoffError, match="provenance drift"):
        bridge._bind_records(changed_raw, inventory_rows, release=changed_release)


def test_duplicate_logical_id_fails_closed() -> None:
    release, records_raw, inventory_raw, _, _ = _release_and_files()
    inventory_rows = bridge._validate_inventory(inventory_raw, release=release)
    duplicate_rows = [inventory_rows[0], inventory_rows[0]]
    with pytest.raises(bridge.CleanRetainedHandoffError, match="logical id collision"):
        bridge._bind_records(records_raw, duplicate_rows, release=release)


def test_unknown_inventory_field_fails_closed() -> None:
    release, _, inventory_raw, _, _ = _release_and_files()
    inventory = bridge._strict_json(inventory_raw, "inventory")
    inventory["unexpected"] = True
    changed_raw = _cjson(inventory, newline=True)
    changed_release = replace(
        release,
        inventory_file_sha256=bridge._sha(changed_raw),
    )
    with pytest.raises(bridge.CleanRetainedHandoffError, match="key set drift"):
        bridge._validate_inventory(changed_raw, release=changed_release)


def test_float_record_count_is_not_integer_alias() -> None:
    release, _, inventory_raw, _, _ = _release_and_files()
    inventory = bridge._strict_json(inventory_raw, "inventory")
    inventory["record_count"] = 2.0
    changed_raw = _cjson(inventory, newline=True)
    changed_release = replace(
        release,
        inventory_file_sha256=bridge._sha(changed_raw),
    )
    with pytest.raises(bridge.CleanRetainedHandoffError, match="exact integer"):
        bridge._validate_inventory(changed_raw, release=changed_release)


def test_stale_pr1836_record_root_is_explicitly_rejected() -> None:
    release, _, inventory_raw, _, _ = _release_and_files()
    inventory = bridge._strict_json(inventory_raw, "inventory")
    inventory["record_inventory_digest_sha256"] = (
        bridge.STALE_PR1836_RECORD_INVENTORY_SHA256
    )
    changed_raw = _cjson(inventory, newline=True)
    changed_release = replace(
        release,
        inventory_file_sha256=bridge._sha(changed_raw),
        record_inventory_digest_sha256=bridge.STALE_PR1836_RECORD_INVENTORY_SHA256,
    )
    with pytest.raises(
        bridge.CleanRetainedHandoffError,
        match="historical contaminated record inventory root",
    ):
        bridge._validate_inventory(changed_raw, release=changed_release)


def test_mixed_clean_and_stale_payload_root_is_rejected() -> None:
    release, _, inventory_raw, _, _ = _release_and_files()
    inventory = bridge._strict_json(inventory_raw, "inventory")
    inventory["payload_inventory_digest_sha256"] = (
        bridge.STALE_PR1836_PAYLOAD_INVENTORY_SHA256
    )
    changed_raw = _cjson(inventory, newline=True)
    changed_release = replace(
        release,
        inventory_file_sha256=bridge._sha(changed_raw),
        payload_inventory_digest_sha256=bridge.STALE_PR1836_PAYLOAD_INVENTORY_SHA256,
    )
    with pytest.raises(
        bridge.CleanRetainedHandoffError,
        match="historical contaminated payload inventory root",
    ):
        bridge._validate_inventory(changed_raw, release=changed_release)


def test_evidence_truth_widening_is_rejected_even_if_resealed() -> None:
    release, _, _, evidence_raw, _ = _release_and_files()
    evidence = bridge._strict_json(evidence_raw, "evidence")
    evidence["truth_boundary"]["training_executed"] = True
    changed_raw = _reseal_evidence(evidence)
    changed_release = replace(
        release,
        evidence_file_sha256=bridge._sha(changed_raw),
        materialization_identity_sha256=str(
            evidence["materialization_identity_sha256"]
        ),
    )
    with pytest.raises(bridge.CleanRetainedHandoffError, match="training_executed"):
        bridge._validate_evidence(changed_raw, release=changed_release)


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(bridge.CleanRetainedHandoffError, match="duplicate JSON key"):
        bridge._strict_json(b'{"a":1,"a":2}\n', "duplicate")


def test_receipt_requires_independently_expected_handoff_roots() -> None:
    handoff = bridge._build_handoff([], release=bridge._RELEASE)
    receipt = bridge._build_receipt(handoff, release=bridge._RELEASE)
    bridge.verify_clean_retained_receipt(
        receipt,
        expected_training_handoff_identity_sha256=handoff[
            "handoff_identity_sha256"
        ],
        expected_matcher_input_projection_sha256=handoff[
            "matcher_input_projection_sha256"
        ],
    )

    tampered = copy.deepcopy(receipt)
    tampered["training_handoff_identity_sha256"] = "f" * 64
    body = copy.deepcopy(tampered)
    body.pop("receipt_identity_sha256", None)
    tampered["receipt_identity_sha256"] = bridge._sha(_cjson(body))
    with pytest.raises(
        bridge.CleanRetainedHandoffError,
        match="not independently expected",
    ):
        bridge.verify_clean_retained_receipt(
            tampered,
            expected_training_handoff_identity_sha256=handoff[
                "handoff_identity_sha256"
            ],
            expected_matcher_input_projection_sha256=handoff[
                "matcher_input_projection_sha256"
            ],
        )


def test_receipt_input_hash_map_is_closed_world() -> None:
    handoff = bridge._build_handoff([], release=bridge._RELEASE)
    receipt = bridge._build_receipt(handoff, release=bridge._RELEASE)
    receipt["input_files_sha256"]["unexpected.json"] = "f" * 64
    body = copy.deepcopy(receipt)
    body.pop("receipt_identity_sha256", None)
    receipt["receipt_identity_sha256"] = bridge._sha(_cjson(body))
    with pytest.raises(bridge.CleanRetainedHandoffError, match="closed-world"):
        bridge.verify_clean_retained_receipt(
            receipt,
            expected_training_handoff_identity_sha256=handoff[
                "handoff_identity_sha256"
            ],
            expected_matcher_input_projection_sha256=handoff[
                "matcher_input_projection_sha256"
            ],
        )
