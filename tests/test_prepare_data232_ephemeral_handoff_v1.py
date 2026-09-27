from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "prepare_data232_ephemeral_handoff_v1.py"
spec = importlib.util.spec_from_file_location("data232_clean_runner", MODULE_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _canonical(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _synthetic_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payloads = [
        ("r-a", "s-a", "alpha"),
        ("r-b", "s-a", "beta"),
        ("r-c", "s-b", "gamma"),
    ]
    current_rows = []
    inventory_rows = []
    for record_id, source_id, text in payloads:
        raw = text.encode("utf-8")
        current_rows.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": "fixture",
                "modality": "text",
                "normalized_payload": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": "fixture",
                "modality": "text",
                "payload_sha256": _sha(raw),
                "payload_bytes": len(raw),
            }
        )
    records_raw = b"".join(_canonical(row, newline=True) for row in current_rows)
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in inventory_rows
    ]
    inventory = {
        "schema_version": runner.INVENTORY_SCHEMA,
        "record_count": 3,
        "total_payload_bytes": sum(row["payload_bytes"] for row in inventory_rows),
        "record_inventory_digest_sha256": _sha(_canonical(inventory_rows)),
        "payload_inventory_digest_sha256": _sha(_canonical(payload_projection)),
        "records": inventory_rows,
    }

    authority = deepcopy(runner._RELEASE_AUTHORITY)
    authority.update(
        {
            "retained_source_count": 3,
            "distinct_physical_source_count": 2,
            "retained_payload_bytes": inventory["total_payload_bytes"],
            "records_jsonl_sha256": _sha(records_raw),
            "record_inventory_digest_sha256": inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": inventory[
                "payload_inventory_digest_sha256"
            ],
            "composition_preflight_identity_sha256": "1" * 64,
            "physical_head_git_sha": "2" * 40,
        }
    )
    monkeypatch.setattr(runner, "_RELEASE_AUTHORITY", authority)

    evidence = {
        "schema_version": runner.MATERIALIZATION_SCHEMA,
        "status": "MATERIALIZED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": authority["physical_head_git_sha"],
        "materializer_implementation_git_blob_sha1": "3" * 40,
        "input": {
            "composition_preflight_identity_sha256": authority[
                "composition_preflight_identity_sha256"
            ],
            "g05_execution_identity_sha256": "4" * 64,
            "g06_envelope_identity_sha256": "5" * 64,
            "g06_execution_identity_sha256": "6" * 64,
            "g06_terminal_qualification_identity_sha256": "7" * 64,
            "input_rows_sha256": "8" * 64,
            "privacy_binding": {},
            "privacy_implementation_git_blob_sha1": "9" * 40,
            "record_payload_jsonl_sha256": "a" * 64,
            "record_inventory_digest_sha256": "b" * 64,
            "payload_inventory_digest_sha256": "c" * 64,
            "record_count": 4,
            "source_object_count": 3,
            "total_payload_bytes": 20,
            "external_llm_provenance_quarantine_identity_sha256": "d" * 64,
        },
        "transform": {},
        "result": {
            "record_payload_jsonl_sha256": authority["records_jsonl_sha256"],
            "record_inventory_digest_sha256": authority[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": authority[
                "payload_inventory_digest_sha256"
            ],
            "record_count": authority["retained_source_count"],
            "source_object_count": authority["distinct_physical_source_count"],
            "total_payload_bytes": authority["retained_payload_bytes"],
        },
        "consumed_blockers": [],
        "remaining_materialization_blockers": [
            "SUCCESSOR_CORPUS_AUTHORITY_REBUILD_REQUIRED"
        ],
        "truth_boundary": dict(runner._TRUTH),
        "materializer_v2_implementation_git_blob_sha1": "e" * 40,
        "base_v1_materialization_identity_sha256": "f" * 64,
        "provenance_guard": {
            "known_external_llm_contamination_absent": True,
            "quarantine_identity_sha256": "0" * 64,
            "whole_corpus_external_llm_cleanliness_claimed": False,
            "successor_authority_rebuild_required_for_invalidated_v5_v6_v8": True,
        },
        "repeat_materialization_byte_identical": True,
    }
    evidence["materialization_identity_sha256"] = _sha(
        _canonical(evidence, newline=True)
    )
    authority["materialization_identity_sha256"] = evidence[
        "materialization_identity_sha256"
    ]

    records_path = tmp_path / "records.jsonl"
    inventory_path = tmp_path / "inventory.json"
    evidence_path = tmp_path / "evidence.json"
    inventory_raw = _canonical(inventory, newline=True)
    evidence_raw = _canonical(evidence, newline=True)
    authority["inventory_json_sha256"] = _sha(inventory_raw)
    authority["evidence_json_sha256"] = _sha(evidence_raw)
    records_path.write_bytes(records_raw)
    inventory_path.write_bytes(inventory_raw)
    evidence_path.write_bytes(evidence_raw)
    return records_path, inventory_path, evidence_path, inventory, evidence


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    records_path, inventory_path, evidence_path, inventory, evidence = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    output = tmp_path / "out"
    receipt = runner.prepare_and_publish(
        records_path=records_path,
        inventory_path=inventory_path,
        evidence_path=evidence_path,
        output_dir=output,
        carrier_git_sha="a" * 40,
    )
    return receipt, output, inventory, evidence


def test_release_authority_uses_clean_2174_roots_and_separates_counts() -> None:
    authority = runner._RELEASE_AUTHORITY
    assert authority["materialization_schema_version"] == (
        "12-6.d03-post-g05-g06-materialization.v2"
    )
    assert authority["retained_source_count"] == 257
    assert authority["distinct_physical_source_count"] == 244
    assert authority["retained_source_count"] != authority[
        "distinct_physical_source_count"
    ]
    assert authority["retained_payload_bytes"] == 5_601_716
    assert authority["inventory_json_sha256"] == (
        "3804a43eba5e0bfa6ce2568782cb03bf681e53a68808742874cf89139488c69e"
    )
    assert authority["evidence_json_sha256"] == (
        "877b6233738e0ca6593acbbf9922f5f178ec8456bf101b95f2f4a300805509e4"
    )
    assert authority["materialization_identity_sha256"] == (
        "7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade"
    )
    assert authority["physical_pr_number"] == 2153
    assert authority["physical_job_id"] == 107724922341
    assert authority["independent_audit_issue_number"] == 2174
    assert authority["independent_audit_terminal_comment_id"] == 5818780583


def test_end_to_end_preserves_existing_data232_handoff_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, output, _, _ = _run(tmp_path, monkeypatch)
    records = [
        json.loads(line)
        for line in (output / runner.TRAINING_RECORDS_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    handoff = json.loads(
        (output / runner.TRAINING_HANDOFF_NAME).read_text(encoding="utf-8")
    )
    stored = json.loads((output / runner.RECEIPT_NAME).read_text(encoding="utf-8"))

    assert [row["record_id"] for row in records] == ["r-a", "r-b", "r-c"]
    assert set(records[0]) == {
        "record_id",
        "source_id",
        "source_family",
        "modality",
        "text",
    }
    assert set(handoff) == {
        "schema_version",
        "postdedup_inventory_identity_sha256",
        "input_survivor_authority_sha256",
        "retained_source_count",
        "matcher_input_projection",
        "matcher_input_projection_sha256",
        "raw_text_persisted_in_evidence",
        "final_test_payload_accessed",
        "final_test_outcomes_accessed",
        "authorized_training_exposure",
        "handoff_identity_sha256",
    }
    assert handoff["schema_version"] == "12-6.postdedup-decontam-handoff.v1"
    assert handoff["retained_source_count"] == 3
    assert handoff["postdedup_inventory_identity_sha256"] == runner._RELEASE_AUTHORITY[
        "materialization_identity_sha256"
    ]
    assert handoff["input_survivor_authority_sha256"] == runner._RELEASE_AUTHORITY[
        "composition_preflight_identity_sha256"
    ]
    assert handoff["authorized_training_exposure"] == 0
    assert stored == receipt
    runner.verify_receipt(stored)
    durable = _canonical(stored)
    assert b"r-a" not in durable
    assert b"alpha" not in durable
    assert stored["raw_text_persisted_in_receipt"] is False
    assert stored["record_ids_persisted_in_receipt"] is False
    assert stored["authorized_optimized_target_exposure"] == 0
    assert stored["training_executed"] is False


def test_exact_physical_and_output_hashes_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, inventory_path, evidence_path, _, _ = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    output = tmp_path / "out"
    receipt = runner.prepare_and_publish(
        records_path=records_path,
        inventory_path=inventory_path,
        evidence_path=evidence_path,
        output_dir=output,
        carrier_git_sha="a" * 40,
    )
    assert receipt["input_files_sha256"] == {
        "records_jsonl": _sha(records_path.read_bytes()),
        "record_inventory_json": _sha(inventory_path.read_bytes()),
        "materialization_evidence_json": _sha(evidence_path.read_bytes()),
    }
    assert receipt["output_files_sha256"] == {
        runner.TRAINING_RECORDS_NAME: _sha(
            (output / runner.TRAINING_RECORDS_NAME).read_bytes()
        ),
        runner.TRAINING_HANDOFF_NAME: _sha(
            (output / runner.TRAINING_HANDOFF_NAME).read_bytes()
        ),
    }


@pytest.mark.parametrize(
    ("target", "mutation"),
    [
        (
            "evidence",
            lambda value: value.__setitem__(
                "schema_version",
                "12-6.d03-post-g05-g06-materialization.v1",
            ),
        ),
        ("inventory", lambda value: value.__setitem__("record_count", 2)),
        ("inventory", lambda value: value.__setitem__("record_count", True)),
        ("inventory", lambda value: value.__setitem__("record_count", 3.0)),
        ("inventory", lambda value: value.__setitem__("total_payload_bytes", 999)),
        (
            "inventory",
            lambda value: value.__setitem__(
                "record_inventory_digest_sha256",
                "9" * 64,
            ),
        ),
        (
            "inventory",
            lambda value: value.__setitem__(
                "payload_inventory_digest_sha256",
                "8" * 64,
            ),
        ),
        (
            "evidence",
            lambda value: value["result"].__setitem__(
                "record_payload_jsonl_sha256",
                "7" * 64,
            ),
        ),
        (
            "evidence",
            lambda value: value.__setitem__(
                "materialization_identity_sha256",
                "6" * 64,
            ),
        ),
    ],
)
def test_authority_substitutions_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mutation,
) -> None:
    records_path, inventory_path, evidence_path, inventory, evidence = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    value = inventory if target == "inventory" else evidence
    mutation(value)
    path = inventory_path if target == "inventory" else evidence_path
    path.write_bytes(_canonical(value, newline=True))
    with pytest.raises((ValueError, TypeError)):
        runner.prepare_and_publish(
            records_path=records_path,
            inventory_path=inventory_path,
            evidence_path=evidence_path,
            output_dir=tmp_path / "out",
            carrier_git_sha="a" * 40,
        )
    assert not (tmp_path / "out").exists()


def test_244_distinct_sources_cannot_alias_retained_record_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, inventory_path, evidence_path, inventory, _ = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    assert runner._RELEASE_AUTHORITY["retained_source_count"] == 3
    assert runner._RELEASE_AUTHORITY["distinct_physical_source_count"] == 2
    inventory["record_count"] = 2
    inventory_path.write_bytes(_canonical(inventory, newline=True))
    with pytest.raises(ValueError):
        runner.prepare_and_publish(
            records_path=records_path,
            inventory_path=inventory_path,
            evidence_path=evidence_path,
            output_dir=tmp_path / "out",
            carrier_git_sha="a" * 40,
        )


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    [
        ("tampered", "payload identity mismatch"),
        ("missing", "physical records JSONL root"),
        ("extra", "unexpected payload record_id"),
        ("duplicate", "duplicate payload record_id"),
        ("unknown-key", "schema drift"),
    ],
)
def test_payload_mutation_or_coverage_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    failure_pattern: str,
) -> None:
    records_path, inventory_path, evidence_path, _, _ = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    if kind == "tampered":
        rows[0]["normalized_payload"] += "!"
    elif kind == "missing":
        rows.pop()
    elif kind == "extra":
        rows.append(
            {
                "record_id": "r-extra",
                "source_id": "s-x",
                "family": "fixture",
                "modality": "text",
                "normalized_payload": "x",
            }
        )
    elif kind == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["unexpected"] = "no"
    records_path.write_bytes(b"".join(_canonical(row, newline=True) for row in rows))
    with pytest.raises(ValueError, match="physical records JSONL root"):
        runner.prepare_and_publish(
            records_path=records_path,
            inventory_path=inventory_path,
            evidence_path=evidence_path,
            output_dir=tmp_path / "out",
            carrier_git_sha="a" * 40,
        )
    assert not (tmp_path / "out").exists()


def test_duplicate_json_keys_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, inventory_path, evidence_path, _, _ = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    duplicate_raw = b'{"schema_version":"x","schema_version":"y"}\n'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner._strict_loads(duplicate_raw, "duplicate-fixture")
    inventory_path.write_bytes(duplicate_raw)
    with pytest.raises(ValueError):
        runner.prepare_and_publish(
            records_path=records_path,
            inventory_path=inventory_path,
            evidence_path=evidence_path,
            output_dir=tmp_path / "out",
            carrier_git_sha="a" * 40,
        )
    assert not (tmp_path / "out").exists()


def test_unknown_inventory_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, inventory_path, evidence_path, inventory, _ = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    inventory["unexpected"] = "no"
    inventory_path.write_bytes(_canonical(inventory, newline=True))
    with pytest.raises(ValueError):
        runner.prepare_and_publish(
            records_path=records_path,
            inventory_path=inventory_path,
            evidence_path=evidence_path,
            output_dir=tmp_path / "out",
            carrier_git_sha="a" * 40,
        )


def test_self_reseal_after_external_root_substitution_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, inventory_path, evidence_path, _, evidence = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    independently_expected = runner._RELEASE_AUTHORITY[
        "materialization_identity_sha256"
    ]
    evidence["result"]["record_payload_jsonl_sha256"] = "7" * 64
    evidence.pop("materialization_identity_sha256")
    evidence["materialization_identity_sha256"] = _sha(
        _canonical(evidence, newline=True)
    )
    assert evidence["materialization_identity_sha256"] != independently_expected
    evidence_path.write_bytes(_canonical(evidence, newline=True))
    with pytest.raises(ValueError):
        runner.prepare_and_publish(
            records_path=records_path,
            inventory_path=inventory_path,
            evidence_path=evidence_path,
            output_dir=tmp_path / "out",
            carrier_git_sha="a" * 40,
        )


def test_existing_workspace_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, inventory_path, evidence_path, _, _ = _synthetic_inputs(
        tmp_path, monkeypatch
    )
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        runner.prepare_and_publish(
            records_path=records_path,
            inventory_path=inventory_path,
            evidence_path=evidence_path,
            output_dir=output,
            carrier_git_sha="a" * 40,
        )
    assert marker.read_text() == "keep"


def test_receipt_tamper_and_boolean_count_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _, _, _ = _run(tmp_path, monkeypatch)
    tampered = deepcopy(receipt)
    tampered["training_executed"] = True
    with pytest.raises(ValueError):
        runner.verify_receipt(tampered)

    tampered = deepcopy(receipt)
    tampered["retained_source_count"] = True
    tampered["receipt_identity_sha256"] = _sha(
        _canonical(
            {
                key: value
                for key, value in tampered.items()
                if key != "receipt_identity_sha256"
            }
        )
    )
    with pytest.raises(ValueError, match="exact positive integer"):
        runner.verify_receipt(tampered)
