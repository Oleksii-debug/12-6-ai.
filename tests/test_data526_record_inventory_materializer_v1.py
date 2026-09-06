from __future__ import annotations

import copy

import pytest

from tools.materialize_data526_record_inventory_v1 import materialize


RECORDS = [
    {
        "record_id": "r2",
        "source_id": "source-b",
        "family": "family-b",
        "modality": "code",
        "normalized_payload": "print('привіт')\n",
    },
    {
        "record_id": "r1",
        "source_id": "source-a",
        "family": "family-a",
        "modality": "uk",
        "normalized_payload": "Український текст.\n",
    },
]


def test_materialization_is_order_independent_and_text_free() -> None:
    first = materialize(RECORDS)
    second = materialize(list(reversed(RECORDS)))
    assert first == second
    assert [record["record_id"] for record in first["records"]] == ["r1", "r2"]
    assert "normalized_payload" not in str(first)
    assert first["record_count"] == 2
    assert first["total_payload_bytes"] == sum(len(record["normalized_payload"].encode("utf-8")) for record in RECORDS)


def test_payload_mutation_changes_both_inventory_identities() -> None:
    changed = copy.deepcopy(RECORDS)
    changed[0]["normalized_payload"] += "# mutation\n"
    baseline = materialize(RECORDS)
    mutated = materialize(changed)
    assert mutated["record_inventory_digest_sha256"] != baseline["record_inventory_digest_sha256"]
    assert mutated["payload_inventory_digest_sha256"] != baseline["payload_inventory_digest_sha256"]


def test_metadata_mutation_changes_record_inventory_but_not_payload_inventory() -> None:
    changed = copy.deepcopy(RECORDS)
    changed[0]["family"] = "different-family"
    baseline = materialize(RECORDS)
    mutated = materialize(changed)
    assert mutated["record_inventory_digest_sha256"] != baseline["record_inventory_digest_sha256"]
    assert mutated["payload_inventory_digest_sha256"] == baseline["payload_inventory_digest_sha256"]


@pytest.mark.parametrize("field", ["record_id", "source_id", "family", "modality", "normalized_payload"])
def test_missing_or_empty_required_fields_fail_closed(field: str) -> None:
    changed = copy.deepcopy(RECORDS)
    changed[0][field] = ""
    with pytest.raises(ValueError):
        materialize(changed)


def test_duplicate_record_ids_fail_closed() -> None:
    changed = copy.deepcopy(RECORDS)
    changed[1]["record_id"] = changed[0]["record_id"]
    with pytest.raises(ValueError, match="duplicate record_id"):
        materialize(changed)
