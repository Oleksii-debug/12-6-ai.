from __future__ import annotations

import gzip
import json
from copy import deepcopy

import pytest

from twelve_six.data.common_pile_loc_intake import (
    LocIntakeError,
    iter_gzip_jsonl_bytes,
    materialize,
    self_identity,
    validate_config,
)

CONFIG_PATH = "configs/data/d03_common_pile_loc_intake_v1.json"


def load_config() -> dict:
    return json.loads(open(CONFIG_PATH, encoding="utf-8").read())


def reseal(config: dict) -> dict:
    config = deepcopy(config)
    config["contract_identity_sha256"] = self_identity(config, "contract_identity_sha256")
    return config


def good_text(seed: str = "history") -> str:
    paragraph = (
        f"This {seed} volume records public history, institutions, communities, and events. "
        "The scanned pages preserve ordinary prose for research and reading. "
    )
    return paragraph * 18


def record(record_id: str, text: str | None = None) -> dict:
    return {
        "id": record_id,
        "text": good_text(record_id) if text is None else text,
        "source": "loc_books",
        "added": "2024-05-13T00:00:00",
        "metadata": {
            "license": "Public Domain",
            "title": "Metadata title must not become training text",
            "author": "Metadata Author",
            "year": 1901,
            "language": "english",
            "item_url": f"https://www.loc.gov/item/{record_id}",
            "text_file_url": f"https://tile.loc.gov/storage-services/{record_id}_djvu.txt",
        },
    }


def test_checked_in_config_is_fail_closed() -> None:
    validate_config(load_config())


def test_training_credit_or_gate_removal_is_rejected_even_when_resealed() -> None:
    config = load_config()
    config["training_authorized_bytes"] = 1
    with pytest.raises(LocIntakeError, match="training_authorized_bytes drift"):
        validate_config(reseal(config))

    config = load_config()
    config["required_downstream_gates"].remove("privacy")
    with pytest.raises(LocIntakeError, match="required gates drift"):
        validate_config(reseal(config))


def test_resealed_upstream_or_audit_authority_substitution_is_rejected() -> None:
    config = load_config()
    config["upstream"]["revision"] = "0" * 40
    config["upstream"]["resolve_url"] = (
        "https://huggingface.co/datasets/common-pile/library_of_congress/resolve/"
        + "0" * 40
        + "/data/00000_loc_books.jsonl.gz"
    )
    with pytest.raises(LocIntakeError, match="upstream revision drift"):
        validate_config(reseal(config))

    config = load_config()
    config["common_pile_audit"]["collector_blob_sha1"] = "0" * 40
    with pytest.raises(LocIntakeError, match="collector_blob_sha1 drift"):
        validate_config(reseal(config))


def test_record_rights_language_source_and_provenance_are_exact() -> None:
    config = load_config()
    for field, value, message in (
        ("license", "CC-BY-4.0", "license"),
        ("language", "french", "language"),
    ):
        candidate = record("abc123")
        candidate["metadata"][field] = value
        with pytest.raises(LocIntakeError, match=message):
            materialize(config, [candidate])

    candidate = record("abc123")
    candidate["source"] = "other"
    with pytest.raises(LocIntakeError, match="source drift"):
        materialize(config, [candidate])

    candidate = record("abc123")
    candidate["metadata"]["item_url"] = "https://www.loc.gov/item/not-abc123"
    with pytest.raises(LocIntakeError, match="bind record id"):
        materialize(config, [candidate])

    candidate = record("abc123")
    candidate["metadata"]["text_file_url"] = "https://example.com/book.txt"
    with pytest.raises(LocIntakeError, match="text file URL"):
        materialize(config, [candidate])


def test_privacy_secret_control_and_low_quality_rows_are_rejected() -> None:
    config = load_config()
    rows = [
        record("mail", good_text() + " contact modern.user@example.com"),
        record("secret", good_text() + " password=correct-horse"),
        record("noise", "1234 !!! ??? " * 100),
        record("ok", good_text("accepted")),
    ]
    candidates, report = materialize(config, rows)
    assert [row["source_record_id"] for row in candidates] == ["ok"]
    assert report["rejection_counts"] == {
        "email_like_contact": 1,
        "low_alpha_content": 1,
        "secret_like_text": 1,
    }


def test_control_character_fails_closed_before_quarantine() -> None:
    config = load_config()
    with pytest.raises(LocIntakeError, match="control"):
        materialize(config, [record("bad", good_text() + "\x00")])


def test_materialization_is_deterministic_zero_credit_and_text_only() -> None:
    config = load_config()
    rows = [record("a1"), record("b2")]
    first_records, first_report = materialize(config, rows)
    second_records, second_report = materialize(config, deepcopy(rows))
    assert first_records == second_records
    assert first_report == second_report
    assert first_report["training_authorized_bytes"] == 0
    assert first_report["authorized_unique_loss_positions"] == 0
    assert first_report["corpus_admitted"] is False
    assert first_report["full_shard_hash_verified"] is False
    assert first_report["privacy_gate"] == "NOT_RUN"
    assert all(row["training_eligible"] is False for row in first_records)
    assert all(row["evaluation_eligible"] is False for row in first_records)
    assert all("Metadata title" not in row["text"] for row in first_records)
    assert all("Metadata Author" not in row["text"] for row in first_records)


def test_exact_normalized_duplicate_is_rejected_without_duplicate_credit() -> None:
    config = load_config()
    same = good_text("same")
    candidates, report = materialize(config, [record("a1", same), record("b2", same)])
    assert [row["source_record_id"] for row in candidates] == ["a1"]
    assert report["rejection_counts"] == {"exact_normalized_duplicate": 1}


def test_duplicate_record_id_fails_closed() -> None:
    config = load_config()
    with pytest.raises(LocIntakeError, match="duplicate record id"):
        materialize(config, [record("same"), record("same", good_text("other"))])


def test_family_byte_cap_is_deterministic() -> None:
    config = load_config()
    sample = good_text("cap")
    sample_bytes = len(sample.encode("utf-8"))
    config["selection_policy"]["max_total_normalized_utf8_bytes"] = sample_bytes + 100
    config = reseal(config)
    candidates, report = materialize(config, [record("a1", sample), record("b2", good_text("b"))])
    assert [row["source_record_id"] for row in candidates] == ["a1"]
    assert report["rejection_counts"] == {"family_byte_cap": 1}


def test_gzip_jsonl_parser_is_bounded_and_strict() -> None:
    rows = [record("a1"), record("b2")]
    payload = b"".join((json.dumps(row) + "\n").encode("utf-8") for row in rows)
    raw = gzip.compress(payload, mtime=0)
    parsed = list(iter_gzip_jsonl_bytes(raw, max_jsonl_line_bytes=100_000))
    assert parsed == rows

    bad = gzip.compress(b"not-json\n", mtime=0)
    with pytest.raises(LocIntakeError, match="malformed JSONL"):
        list(iter_gzip_jsonl_bytes(bad, max_jsonl_line_bytes=100))

    long_line = gzip.compress(b'{"x":"' + b"a" * 200 + b'"}\n', mtime=0)
    with pytest.raises(LocIntakeError, match="line exceeds"):
        list(iter_gzip_jsonl_bytes(long_line, max_jsonl_line_bytes=50))
