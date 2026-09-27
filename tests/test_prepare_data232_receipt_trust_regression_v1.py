from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "prepare_data232_ephemeral_handoff_v1.py"
spec = importlib.util.spec_from_file_location("data232_receipt_regression", MODULE_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _production_receipt() -> dict[str, object]:
    authority = dict(runner._RELEASE_AUTHORITY)
    receipt: dict[str, object] = {
        "schema_version": runner.RECEIPT_SCHEMA,
        "carrier_implementation_git_sha": "a" * 40,
        "clean_release_authority_sha256": runner._release_authority_identity(),
        "physical_release_authority": authority,
        "input_files_sha256": {
            "records_jsonl": authority["records_jsonl_sha256"],
            "record_inventory_json": authority["inventory_json_sha256"],
            "materialization_evidence_json": authority["evidence_json_sha256"],
        },
        "output_files_sha256": dict(runner._PRODUCTION_OUTPUT_FILES_SHA256),
        "materialization_identity_sha256": authority[
            "materialization_identity_sha256"
        ],
        "composition_preflight_identity_sha256": authority[
            "composition_preflight_identity_sha256"
        ],
        "record_inventory_digest_sha256": authority[
            "record_inventory_digest_sha256"
        ],
        "payload_inventory_digest_sha256": authority[
            "payload_inventory_digest_sha256"
        ],
        "records_jsonl_sha256": authority["records_jsonl_sha256"],
        "retained_source_count": authority["retained_source_count"],
        "distinct_physical_source_count": authority[
            "distinct_physical_source_count"
        ],
        "retained_payload_bytes": authority["retained_payload_bytes"],
        "training_records_file_bytes": runner._PRODUCTION_TRAINING_RECORDS_FILE_BYTES,
        "payload_match_proven": True,
        "durable_receipt_hash_only": True,
        "raw_text_persisted_in_receipt": False,
        "record_ids_persisted_in_receipt": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "current_corpus_external_llm_free_claimed_by_this_carrier": False,
    }
    receipt["receipt_identity_sha256"] = runner._sha256(runner._canonical(receipt))
    return receipt


def _reseal(receipt: dict[str, object]) -> None:
    core = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_identity_sha256"
    }
    receipt["receipt_identity_sha256"] = runner._sha256(runner._canonical(core))


def test_production_receipt_exact_bindings_verify() -> None:
    assert runner._release_authority_identity() == (
        runner._PRODUCTION_RELEASE_AUTHORITY_IDENTITY_SHA256
    )
    runner.verify_receipt(_production_receipt())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["input_files_sha256"].__setitem__(
            "records_jsonl", "0" * 64
        ),
        lambda value: value["output_files_sha256"].__setitem__(
            runner.TRAINING_RECORDS_NAME, "0" * 64
        ),
        lambda value: value.__setitem__(
            "materialization_identity_sha256", "0" * 64
        ),
        lambda value: value.__setitem__(
            "composition_preflight_identity_sha256", "0" * 64
        ),
        lambda value: value.__setitem__(
            "record_inventory_digest_sha256", "0" * 64
        ),
        lambda value: value.__setitem__(
            "payload_inventory_digest_sha256", "0" * 64
        ),
        lambda value: value.__setitem__("records_jsonl_sha256", "0" * 64),
        lambda value: value.__setitem__(
            "training_records_file_bytes",
            runner._PRODUCTION_TRAINING_RECORDS_FILE_BYTES + 1,
        ),
    ],
)
def test_coherent_receipt_reseal_cannot_substitute_independent_bindings(
    mutation,
) -> None:
    receipt = deepcopy(_production_receipt())
    mutation(receipt)
    _reseal(receipt)
    with pytest.raises(ValueError):
        runner.verify_receipt(receipt)


def test_receipt_hash_maps_are_closed_world_and_sha_typed() -> None:
    receipt = deepcopy(_production_receipt())
    receipt["input_files_sha256"]["unexpected"] = "0" * 64
    _reseal(receipt)
    with pytest.raises(ValueError, match="key set drift"):
        runner.verify_receipt(receipt)

    receipt = deepcopy(_production_receipt())
    receipt["output_files_sha256"][runner.TRAINING_HANDOFF_NAME] = "XYZ"
    _reseal(receipt)
    with pytest.raises(ValueError, match="lowercase 64-hex"):
        runner.verify_receipt(receipt)


def test_carrier_git_sha_is_strictly_validated_after_coherent_reseal() -> None:
    receipt = deepcopy(_production_receipt())
    receipt["carrier_implementation_git_sha"] = "A" * 40
    _reseal(receipt)
    with pytest.raises(ValueError, match="lowercase 40-hex"):
        runner.verify_receipt(receipt)


def test_payload_transport_is_hashed_and_parsed_from_one_open_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "record_id": "r1",
        "source_id": "s1",
        "family": "fixture",
        "modality": "text",
        "normalized_payload": "payload",
    }
    raw = runner._canonical(row, newline=True)
    records_path = tmp_path / "records.jsonl"
    records_path.write_bytes(raw)
    payload = row["normalized_payload"].encode("utf-8")
    by_record = {
        "r1": {
            "record_id": "r1",
            "source_id": "s1",
            "family": "fixture",
            "modality": "text",
            "payload_sha256": runner._sha256(payload),
            "payload_bytes": len(payload),
        }
    }
    authority = deepcopy(runner._RELEASE_AUTHORITY)
    authority["records_jsonl_sha256"] = runner._sha256(raw)
    authority["retained_source_count"] = 1
    monkeypatch.setattr(runner, "_RELEASE_AUTHORITY", authority)

    original_open = Path.open
    opens = 0

    def tracked_open(self: Path, *args, **kwargs):
        nonlocal opens
        if self == records_path and args and args[0] == "rb":
            opens += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    records, projection = runner._prepare_rows(records_path, by_record)
    assert opens == 1
    assert len(records) == 1
    assert len(projection) == 1
