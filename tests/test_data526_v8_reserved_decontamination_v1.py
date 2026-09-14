from __future__ import annotations

import copy
import hashlib
import json

import pytest

import twelve_six.data.data526_v8_reserved_decontamination_v1 as adapter
from twelve_six.data.data526_v8_reserved_decontamination_v1 import (
    Data526V8DecontaminationBindingError,
    bind_data526_v8_training_records,
    execute_data526_v8_reserved_decontamination,
    verify_data526_v8_execution_evidence,
)

SURVIVOR = "a" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_records() -> list[dict[str, str]]:
    return [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "family-uk",
            "modality": "uk",
            "normalized_payload": "Привіт, світе",
        },
        {
            "record_id": "r2",
            "source_id": "s2",
            "family": "family-code",
            "modality": "code",
            "normalized_payload": "print('hello')\n",
        },
    ]


def _authorities(
    raw_records: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, object]]:
    rows = []
    for raw in raw_records:
        payload = raw["normalized_payload"].encode("utf-8")
        rows.append(
            {
                "record_id": raw["record_id"],
                "source_id": raw["source_id"],
                "family": raw["family"],
                "modality": raw["modality"],
                "payload_sha256": _sha(payload),
                "payload_bytes": len(payload),
            }
        )
    rows.sort(key=lambda row: str(row["record_id"]))
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    inventory: dict[str, object] = {
        "schema_version": adapter.RECORD_INVENTORY_SCHEMA,
        "record_count": len(rows),
        "total_payload_bytes": sum(int(row["payload_bytes"]) for row in rows),
        "record_inventory_digest_sha256": _sha(_canonical(rows)),
        "payload_inventory_digest_sha256": _sha(_canonical(payload_projection)),
        "records": rows,
    }
    canonical_raw = sorted(raw_records, key=lambda row: row["record_id"])
    core: dict[str, object] = {
        "schema_version": adapter.MATERIALIZATION_EVIDENCE_SCHEMA,
        "record_count": len(rows),
        "source_object_count": len({row["source_id"] for row in raw_records}),
        "total_payload_bytes": inventory["total_payload_bytes"],
        "record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "payload_inventory_digest_sha256": inventory[
            "payload_inventory_digest_sha256"
        ],
        "record_payload_jsonl_sha256": _sha(
            b"".join(_canonical(row) + b"\n" for row in canonical_raw)
        ),
        "v8_survivor_authority_sha256": SURVIVOR,
        "authorized_unique_optimized_targets": 0,
        "optimizer_updates": 0,
        "decontamination_executed": False,
        "tokenizer_fit_executed": False,
        "training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
        "raw_payloads_emitted_to_public_evidence": False,
    }
    evidence = dict(core)
    evidence["evidence_identity_sha256"] = _sha(_canonical(core))
    return inventory, evidence


def _bind_args(
    inventory: dict[str, object],
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "record_inventory": inventory,
        "materialization_evidence": evidence,
        "expected_record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "expected_payload_inventory_digest_sha256": inventory[
            "payload_inventory_digest_sha256"
        ],
        "expected_materialization_evidence_identity_sha256": evidence[
            "evidence_identity_sha256"
        ],
        "expected_survivor_authority_sha256": SURVIVOR,
    }


def test_exact_record_level_authority_translates_to_incumbent_matcher_shape():
    raw = _raw_records()
    inventory, evidence = _authorities(raw)
    matcher_rows, handoff = bind_data526_v8_training_records(
        raw,
        **_bind_args(inventory, evidence),
    )
    assert [row["record_id"] for row in matcher_rows] == ["r1", "r2"]
    assert matcher_rows[0]["source_family"] == "family-uk"
    assert matcher_rows[0]["text"] == "Привіт, світе"
    assert handoff["postdedup_inventory_identity_sha256"] == inventory[
        "record_inventory_digest_sha256"
    ]
    assert handoff["input_survivor_authority_sha256"] == SURVIVOR
    assert handoff["retained_source_count"] == 2
    assert handoff["authorized_training_exposure"] == 0


def test_payload_mutation_fails_closed_against_text_free_inventory():
    raw = _raw_records()
    inventory, evidence = _authorities(raw)
    mutated = copy.deepcopy(raw)
    mutated[0]["normalized_payload"] = "tampered"
    with pytest.raises(
        Data526V8DecontaminationBindingError,
        match="payload (byte|SHA-256) drift",
    ):
        bind_data526_v8_training_records(
            mutated,
            **_bind_args(inventory, evidence),
        )


def test_record_metadata_mutation_fails_closed():
    raw = _raw_records()
    inventory, evidence = _authorities(raw)
    mutated = copy.deepcopy(raw)
    mutated[0]["family"] = "substituted-family"
    with pytest.raises(
        Data526V8DecontaminationBindingError,
        match="raw record family drift",
    ):
        bind_data526_v8_training_records(
            mutated,
            **_bind_args(inventory, evidence),
        )


def test_self_consistent_inventory_substitution_still_requires_external_identity():
    raw = _raw_records()
    inventory, evidence = _authorities(raw)
    substituted = copy.deepcopy(inventory)
    substituted_rows = substituted["records"]
    assert isinstance(substituted_rows, list)
    substituted_rows[0]["family"] = "self-consistent-substitute"
    substituted["record_inventory_digest_sha256"] = _sha(
        _canonical(substituted_rows)
    )
    with pytest.raises(
        Data526V8DecontaminationBindingError,
        match="record inventory is not independently expected",
    ):
        bind_data526_v8_training_records(
            raw,
            record_inventory=substituted,
            materialization_evidence=evidence,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            expected_materialization_evidence_identity_sha256=evidence[
                "evidence_identity_sha256"
            ],
            expected_survivor_authority_sha256=SURVIVOR,
        )


def test_materialization_boundary_tamper_fails_external_binding():
    raw = _raw_records()
    inventory, evidence = _authorities(raw)
    tampered = copy.deepcopy(evidence)
    tampered["training_executed"] = True
    tampered.pop("evidence_identity_sha256")
    tampered["evidence_identity_sha256"] = _sha(_canonical(tampered))
    with pytest.raises(
        Data526V8DecontaminationBindingError,
        match="materialization evidence is not independently expected",
    ):
        bind_data526_v8_training_records(
            raw,
            record_inventory=inventory,
            materialization_evidence=tampered,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            expected_materialization_evidence_identity_sha256=evidence[
                "evidence_identity_sha256"
            ],
            expected_survivor_authority_sha256=SURVIVOR,
        )


def test_record_jsonl_identity_blocks_reordering_or_schema_aliasing():
    raw = _raw_records()
    inventory, evidence = _authorities(raw)
    aliased = copy.deepcopy(raw)
    aliased[0]["extra"] = "forbidden"
    with pytest.raises(
        Data526V8DecontaminationBindingError,
        match="schema drift",
    ):
        bind_data526_v8_training_records(
            aliased,
            **_bind_args(inventory, evidence),
        )


def test_wrapper_delegates_only_after_exact_record_binding(monkeypatch: pytest.MonkeyPatch):
    raw = _raw_records()
    inventory, materialization = _authorities(raw)
    delegated: dict[str, object] = {}

    def fake_execute(training_records, evaluation_records, **kwargs):
        delegated["training_records"] = training_records
        delegated["evaluation_records"] = evaluation_records
        delegated["kwargs"] = kwargs
        report = {
            "status": "PASS_CLEAN",
            "report_sha256": "b" * 64,
            "counts": {"input_training_records": len(training_records)},
        }
        inner = {
            "execution_identity_sha256": "c" * 64,
        }
        return report, inner

    monkeypatch.setattr(adapter, "execute_reserved_decontamination", fake_execute)
    monkeypatch.setattr(adapter, "verify_execution_evidence", lambda *_: None)
    report, durable = execute_data526_v8_reserved_decontamination(
        raw,
        [],
        record_inventory=inventory,
        materialization_evidence=materialization,
        reserved_payload_binding={"binding_identity_sha256": "d" * 64},
        expected_record_inventory_digest_sha256=inventory[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=inventory[
            "payload_inventory_digest_sha256"
        ],
        expected_materialization_evidence_identity_sha256=materialization[
            "evidence_identity_sha256"
        ],
        expected_survivor_authority_sha256=SURVIVOR,
        expected_reserved_binding_identity_sha256="d" * 64,
        expected_selection_validation_identity_sha256="e" * 64,
        expected_final_test_identity_sha256="f" * 64,
    )
    assert report["status"] == "PASS_CLEAN"
    matcher_rows = delegated["training_records"]
    assert isinstance(matcher_rows, list)
    assert matcher_rows[0]["source_family"] == "family-uk"
    assert durable["data526_record_inventory_digest_sha256"] == inventory[
        "record_inventory_digest_sha256"
    ]
    assert durable["data526_record_payload_jsonl_sha256"] == materialization[
        "record_payload_jsonl_sha256"
    ]
    assert durable["authorized_training_exposure"] == 0
    assert "Привіт, світе" not in json.dumps(durable, ensure_ascii=False)
    verify_data526_v8_execution_evidence(durable)
