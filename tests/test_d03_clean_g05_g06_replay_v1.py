from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_d03_clean_g05_g06_replay_v1.py"
SPEC = importlib.util.spec_from_file_location("clean_g05_g06_replay_v1", TOOL)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _valid_data526_evidence() -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema_version": replay._DATA526_SCHEMA,
        "worker_id": "test",
        "execution_profile": replay.EXECUTION_PROFILE,
        "execution_head_sha": "1" * 40,
        "source_report_sha256": replay.EXPECTED_SOURCE_REPORT_SHA256,
        "survivor_authority_sha256": replay.EXPECTED_SURVIVOR_AUTHORITY_SHA256,
        "historical_materializer": {},
        "clean_historical": {},
        "bulk_report_identity_sha256": "2" * 64,
        "clean_data526": {
            "record_count": replay.EXPECTED_RECORDS,
            "source_object_count": replay.EXPECTED_SOURCE_OBJECTS,
            "total_payload_bytes": replay.EXPECTED_PAYLOAD_BYTES,
            "record_payload_jsonl_sha256": replay.EXPECTED_RECORDS_JSONL_SHA256,
            "record_inventory_digest_sha256": replay.EXPECTED_RECORD_INVENTORY_SHA256,
            "payload_inventory_digest_sha256": replay.EXPECTED_PAYLOAD_INVENTORY_SHA256,
        },
        "raw_text_emitted_to_durable_evidence": False,
        "truth_boundary": {
            "clean_data526_record_graph_materialized": True,
            "corpus_released": False,
            "decontamination_executed_for_successor": False,
            "post_composition_quality_privacy_passed": False,
            "balance_release_claimed": False,
            "split_pack_complete": False,
            "tokenizer_fit_authorized": False,
            "authorized_training_exposure": 0,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "learned_weights_created": False,
            "final_test_payload_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
            "raw_payloads_committed_to_repository": False,
            "raw_payloads_uploaded_as_public_evidence": False,
        },
        "remaining_blockers": [],
    }
    evidence["evidence_identity_sha256"] = replay._sha256(replay._canonical(evidence))
    return evidence


def test_retained_clean_roots_are_exactly_pinned() -> None:
    clean = replay._validate_data526_evidence(_valid_data526_evidence())
    assert clean == {
        "record_count": 274,
        "source_object_count": 261,
        "total_payload_bytes": 6_093_662,
        "record_payload_jsonl_sha256": (
            "dc22d829921890ea8c5b51cbedaae099688c37c3cb624a597973305c7fa5b2c3"
        ),
        "record_inventory_digest_sha256": (
            "7d6782e91243505c01b0f2f6d6f85b5bbe3a6abf628d73721b1e0c1c77e4f352"
        ),
        "payload_inventory_digest_sha256": (
            "59f9c5a7b5db9e45fcde2e3bab10a4adc97cb6832f906fc241edc3e35248b576"
        ),
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("record_count", True),
        ("source_object_count", 260),
        ("total_payload_bytes", 6_093_661),
        ("record_payload_jsonl_sha256", "0" * 64),
        ("record_inventory_digest_sha256", "0" * 64),
        ("payload_inventory_digest_sha256", "0" * 64),
    ],
)
def test_clean_root_drift_fails_closed(field: str, bad_value: object) -> None:
    evidence = _valid_data526_evidence()
    clean = evidence["clean_data526"]
    assert isinstance(clean, dict)
    clean[field] = bad_value
    with pytest.raises(replay.CleanG05G06ReplayError):
        replay._validate_data526_evidence(evidence)


def test_positive_scientific_flags_cannot_be_inherited_from_reconstruction() -> None:
    evidence = _valid_data526_evidence()
    boundary = evidence["truth_boundary"]
    assert isinstance(boundary, dict)
    boundary["tokenizer_fit_authorized"] = True
    with pytest.raises(replay.CleanG05G06ReplayError):
        replay._validate_data526_evidence(evidence)

    assert replay._ZERO_FALSE_BOUNDARY == {
        "current_corpus_eligible": False,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
        "raw_payloads_retained_in_output": False,
        "whole_corpus_external_llm_cleanliness_claimed": False,
    }


def test_physical_identity_is_runner_derived_and_sha_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "a" * 40
    monkeypatch.setenv("GITHUB_REPOSITORY", "Oleksii-debug/12-6-ai.")
    monkeypatch.setenv("GITHUB_SHA", head)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_JOB", "physical_replay_a")
    assert replay._physical_identity(head) == {
        "repository": "Oleksii-debug/12-6-ai.",
        "workflow_run_id": 123,
        "workflow_run_attempt": 2,
        "workflow_job": "physical_replay_a",
        "execution_head_sha": head,
    }

    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    with pytest.raises(replay.CleanG05G06ReplayError):
        replay._physical_identity(head)


def test_bound_repaired_engine_blobs_match_integrated_tree() -> None:
    assert replay.QUALITY_MODULE_BLOB_SHA1 == "4659a9d4aba49908f372250904a54361c8d8cf46"
    assert replay.PRIVACY_MODULE_BLOB_SHA1 == "9215287e81c0a82f05ec8405dc4f34c60313c193"
    observed = replay._verify_engine_bindings(ROOT)
    assert observed[str(replay.QUALITY_MODULE)] == replay.QUALITY_MODULE_BLOB_SHA1
    assert observed[str(replay.PRIVACY_MODULE)] == replay.PRIVACY_MODULE_BLOB_SHA1
    assert observed[str(replay.CLEAN_MATERIALIZER)] == replay.CLEAN_MATERIALIZER_BLOB_SHA1


def test_g05_manifest_anchor_is_complete_retained_evidence_identity() -> None:
    retained = {
        "execution_head_sha": replay.RETAINED_EXECUTION_HEAD,
        "evidence_identity_sha256": replay.RETAINED_DATA526_EVIDENCE_IDENTITY_SHA256,
    }
    assert replay._g05_manifest_authority(retained) == (
        replay.RETAINED_DATA526_EVIDENCE_IDENTITY_SHA256
    )

    inventory_alias = dict(retained)
    inventory_alias["evidence_identity_sha256"] = replay.EXPECTED_RECORD_INVENTORY_SHA256
    with pytest.raises(replay.CleanG05G06ReplayError, match="retained evidence identity drift"):
        replay._g05_manifest_authority(inventory_alias)


def test_shared_g05_g06_input_root_is_anchored_to_text_free_inventory() -> None:
    text = "clean payload"
    raw_records = [{"id": "r1", "text": text, "mode": "en"}]
    inventory = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "fixture",
            "modality": "en",
            "payload_sha256": replay._sha256(text.encode("utf-8")),
            "payload_bytes": len(text.encode("utf-8")),
        }
    ]
    expected = replay.input_rows_sha256_from_text_free_inventory(inventory)
    assert replay._verify_shared_input_root(raw_records, expected) == expected

    changed = [{"id": "r1", "text": text + "!", "mode": "en"}]
    with pytest.raises(replay.CleanG05G06ReplayError, match="raw/inventory input root mismatch"):
        replay._verify_shared_input_root(changed, expected)


def test_authenticated_json_rejects_duplicate_object_members(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.json"
    path.write_text(
        '{"schema_version":"first","schema_version":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(replay.CleanG05G06ReplayError, match="duplicate JSON object key"):
        replay._read_json(path)


def test_receipt_source_has_no_training_upgrade_or_global_provenance_upgrade() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert '"status": "PHYSICAL_REPLAY_EXECUTED_ZERO_CREDIT"' in source
    assert '"whole_corpus_external_llm_cleanliness_claimed": False' in source
    assert '"tokenizer_fit_authorized": False' in source
    assert '"training_executed": False' in source
    assert 'b"normalized_payload" not in serialized' in source
    assert "raw normalized payload leaked into replay receipt" in source


def test_fresh_evidence_self_hash_is_not_self_sealable_after_mutation() -> None:
    evidence = _valid_data526_evidence()
    evidence["worker_id"] = "mutated-after-hash"
    with pytest.raises(replay.CleanG05G06ReplayError):
        replay._validate_data526_evidence(evidence)


def test_nomis_identity_and_payload_are_rejected_before_root_acceptance(
    tmp_path: Path,
) -> None:
    inventory = {
        "schema_version": replay._INVENTORY_SCHEMA,
        "record_count": replay.EXPECTED_RECORDS,
        "total_payload_bytes": replay.EXPECTED_PAYLOAD_BYTES,
        "record_inventory_digest_sha256": replay.EXPECTED_RECORD_INVENTORY_SHA256,
        "payload_inventory_digest_sha256": replay.EXPECTED_PAYLOAD_INVENTORY_SHA256,
        "records": [
            {
                "record_id": replay.NOMIS_RECORD_ID,
                "source_id": replay.NOMIS_RECORD_ID,
                "family": "ua.verba.public-domain.nomis1864",
                "modality": "uk",
                "payload_sha256": replay.NOMIS_PAYLOAD_SHA256,
                "payload_bytes": 1,
            }
        ],
    }
    path = tmp_path / "inventory.json"
    path.write_text(replay._canonical(inventory).decode("utf-8"), encoding="utf-8")
    with pytest.raises(
        replay.CleanG05G06ReplayError,
        match="Nomis quarantined identity",
    ):
        replay._validate_inventory(path)


def test_pr462_authority_is_not_promoted_by_replay_contract() -> None:
    assert replay.PR462_AUTHORITY_SHA256 == (
        "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7"
    )
    source = TOOL.read_text(encoding="utf-8")
    assert '"pr462_authority_sha256_not_admitted": PR462_AUTHORITY_SHA256' in source
