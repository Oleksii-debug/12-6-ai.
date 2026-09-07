from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.loc_public_domain_intake import (
    LocIntakeError,
    canonical_json,
    materialize_records,
    self_identity,
    validate_config,
    verify_report,
)

CONFIG_PATH = Path("configs/data/d03_loc_public_domain_intake_v1.json")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def record(
    record_id: str,
    text: str,
    *,
    source: str = "loc_books",
    license_name: str = "Public Domain",
) -> dict:
    return {
        "id": record_id,
        "text": text,
        "source": source,
        "added": "2024-05-14T16:22:35.184395",
        "metadata": {"license": license_name, "title": "fixture", "year": 1901},
    }


def test_config_is_self_hash_valid_and_zero_credit() -> None:
    config = load_config()
    validate_config(config)
    assert config["training_authorized_bytes"] == 0
    assert config["authorized_unique_loss_positions"] == 0
    assert config["corpus_admitted"] is False
    assert config["model_training_permitted"] is False


def test_materialization_is_deterministic_and_text_free_report() -> None:
    config = load_config()
    payload = "A" * 2048
    records = [record("a", payload), record("b", "B" * 4096)]
    first_rows, first_report = materialize_records(config, records)
    second_rows, second_report = materialize_records(config, records)
    assert first_rows == second_rows
    assert first_report == second_report
    assert first_report["accepted_documents"] == 2
    assert first_report["accepted_normalized_utf8_bytes"] == 6144
    serialized_report = canonical_json(first_report)
    assert payload not in serialized_report
    assert "fixture" not in serialized_report
    verify_report(config, first_report)


def test_wrong_source_and_license_are_rejected_not_promoted() -> None:
    config = load_config()
    rows, report = materialize_records(
        config,
        [
            record("wrong-source", "A" * 2048, source="other"),
            record("wrong-license", "B" * 2048, license_name="CC-BY-4.0"),
            record("accepted", "C" * 2048),
        ],
    )
    assert [row["source_record_id"] for row in rows] == ["accepted"]
    assert report["rejected_counts"]["source"] == 1
    assert report["rejected_counts"]["license"] == 1
    assert rows[0]["training_eligible"] is False
    assert rows[0]["evaluation_eligible"] is False


def test_exact_normalized_duplicate_is_collapsed() -> None:
    config = load_config()
    rows, report = materialize_records(
        config,
        [record("first", "D" * 2048), record("second", "D" * 2048)],
    )
    assert len(rows) == 1
    assert report["rejected_counts"]["duplicate"] == 1


def test_oversized_document_is_skipped_without_truncation() -> None:
    config = load_config()
    max_bytes = config["selection_policy"]["max_document_normalized_utf8_bytes"]
    rows, report = materialize_records(
        config,
        [record("huge", "H" * (max_bytes + 1)), record("small", "S" * 2048)],
    )
    assert [row["source_record_id"] for row in rows] == ["small"]
    assert report["rejected_counts"]["document_too_large"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_authorized_bytes", 1),
        ("corpus_admitted", True),
        ("model_training_permitted", True),
        ("paid_compute_authorized", True),
        ("final_test_payload_accessed", True),
    ],
)
def test_truth_boundary_mutation_fails_closed(field: str, value: object) -> None:
    config = load_config()
    config[field] = value
    config["contract_identity_sha256"] = self_identity(config, "contract_identity_sha256")
    with pytest.raises(LocIntakeError, match="drift"):
        validate_config(config)


def test_self_consistent_shard_substitution_fails_closed() -> None:
    config = load_config()
    config["upstream"]["shard_sha256"] = "0" * 64
    config["contract_identity_sha256"] = self_identity(config, "contract_identity_sha256")
    with pytest.raises(LocIntakeError, match="shard"):
        validate_config(config)


def test_report_tamper_fails_closed() -> None:
    config = load_config()
    _, report = materialize_records(config, [record("ok", "Q" * 2048)])
    tampered = copy.deepcopy(report)
    tampered["accepted_documents"] += 1
    with pytest.raises(LocIntakeError, match="identity"):
        verify_report(config, tampered)


def test_field_schema_drift_is_quarantined() -> None:
    config = load_config()
    malformed = record("bad", "X" * 2048)
    malformed["extra"] = "unexpected"
    rows, report = materialize_records(
        config,
        [malformed, record("good", "Y" * 2048)],
    )
    assert len(rows) == 1
    assert report["rejected_counts"]["field_schema"] == 1
