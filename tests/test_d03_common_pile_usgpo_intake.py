from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.data.common_pile_usgpo_intake import (
    UsgpoIntakeError,
    iter_gzip_jsonl_bytes,
    materialize,
    self_identity,
    validate_config,
    verify_transport,
)

CONFIG_PATH = Path("configs/data/d03_common_pile_usgpo_intake_v1.json")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def reseal(config: dict) -> dict:
    config = deepcopy(config)
    config["contract_identity_sha256"] = self_identity(config, "contract_identity_sha256")
    return config


def good_text(seed: str = "federal") -> str:
    paragraph = (
        f"This {seed} document records official federal law, policy, history, and procedure. "
        "The public text contains explanatory prose, statutory language, "
        "and administrative detail. "
    )
    return paragraph * 18


def record(
    package_id: str,
    text: str | None = None,
    *,
    category: str = "CFR",
    author: int | str | None = 118,
) -> dict:
    return {
        "id": package_id,
        "created": "2024-05-13",
        "text": good_text(package_id) if text is None else text,
        "source": "usgpo",
        "added": "2025-01-01T00:00:00",
        "metadata": {
            "title": "Metadata title must never become training text",
            "author": author,
            "publisher": "gpo",
            "category": category,
            "license": "Public Domain",
            "url": (
                f"https://www.govinfo.gov/content/pkg/{package_id}/"
                f"html/{package_id}.htm"
            ),
        },
    }


def test_checked_in_config_is_sealed_and_fail_closed() -> None:
    validate_config(load_config())


def test_training_credit_rights_or_transport_drift_is_rejected_even_when_resealed() -> None:
    config = load_config()
    config["training_authorized_bytes"] = 1
    with pytest.raises(UsgpoIntakeError, match="training_authorized_bytes drift"):
        validate_config(reseal(config))

    config = load_config()
    config["rights_policy"]["legal_conclusion_claimed"] = True
    with pytest.raises(UsgpoIntakeError, match="legal_conclusion_claimed drift"):
        validate_config(reseal(config))

    config = load_config()
    config["upstream"]["revision"] = "0" * 40
    config["upstream"]["resolve_url"] = (
        "https://huggingface.co/datasets/common-pile/usgpo/resolve/"
        + "0" * 40
        + "/data/00017_usgpo.jsonl.gz"
    )
    with pytest.raises(UsgpoIntakeError, match="revision drift"):
        validate_config(reseal(config))

    config = load_config()
    config["common_pile_audit"]["collector_blob_sha1"] = "0" * 40
    with pytest.raises(UsgpoIntakeError, match="collector_blob_sha1 drift"):
        validate_config(reseal(config))


def test_required_item_rights_gate_cannot_be_removed() -> None:
    config = load_config()
    config["required_downstream_gates"].remove("item_level_rights_review")
    with pytest.raises(UsgpoIntakeError, match="required gates drift"):
        validate_config(reseal(config))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.update(source="other"), "source drift"),
        (
            lambda row: row["metadata"].update(license="CC-BY-4.0"),
            "license drift",
        ),
        (
            lambda row: row["metadata"].update(publisher="private"),
            "publisher drift",
        ),
        (
            lambda row: row["metadata"].update(
                url="https://example.com/content/pkg/PKG-X/html/PKG-X.htm"
            ),
            "bind exact package id",
        ),
        (
            lambda row: row["metadata"].update(
                url="https://www.govinfo.gov/content/pkg/OTHER/html/OTHER.htm"
            ),
            "bind exact package id",
        ),
        (
            lambda row: row["metadata"].update(
                url="https://www.govinfo.gov/content/pkg/PKG-X/html/PKG-X.htm?download=1"
            ),
            "bind exact package id",
        ),
        (
            lambda row: row["metadata"].update(extra_field="laundered"),
            "metadata schema drift",
        ),
    ],
)
def test_producer_schema_and_govinfo_provenance_are_exact(mutator, message: str) -> None:
    row = record("PKG-X")
    mutator(row)
    with pytest.raises(UsgpoIntakeError, match=message):
        materialize(load_config(), [row])


def test_unsupported_collection_and_ambiguous_author_are_quarantined() -> None:
    candidates, evidence, report = materialize(
        load_config(),
        [
            record("GOVPUB-X", category="GOVPUB"),
            record("PRIVATE-X", author="Private Corporation"),
            record("OK-X", category="USCODE", author=None),
        ],
    )
    assert [row["source_record_id"] for row in candidates] == ["OK-X"]
    assert report["rejection_counts"] == {
        "ambiguous_noncongress_author_metadata": 1,
        "unsupported_federal_scope_collection": 1,
    }
    assert [row["decision"] for row in evidence] == ["REJECT", "REJECT", "CANDIDATE"]


def test_privacy_secret_and_quality_rows_are_quarantined() -> None:
    rows = [
        record("MAIL", good_text() + " Contact modern.user@example.com for details."),
        record("PHONE", good_text() + " Call 202-555-0184 for assistance."),
        record("SECRET", good_text() + " password=correct-horse-battery-staple"),
        record("NOISE", "1234 !!! ??? " * 200),
        record("NONEN", ("Привіт світ та офіційний документ. " * 100)),
        record("OK", good_text("accepted")),
    ]
    candidates, _, report = materialize(load_config(), rows)
    assert [row["source_record_id"] for row in candidates] == ["OK"]
    assert report["rejection_counts"] == {
        "email_like_contact": 1,
        "low_alpha_content": 1,
        "non_english_like_text": 1,
        "phone_like_contact": 1,
        "secret_like_text": 1,
    }


def test_control_character_is_fatal_not_silently_normalized() -> None:
    with pytest.raises(UsgpoIntakeError, match="control"):
        materialize(load_config(), [record("BAD", good_text() + "\x00")])


def test_materialization_is_deterministic_zero_credit_and_payload_only() -> None:
    rows = [record("A1"), record("B2", author=None, category="FR")]
    first = materialize(load_config(), rows)
    second = materialize(load_config(), deepcopy(rows))
    assert first == second
    candidates, evidence, report = first
    assert report["training_authorized_bytes"] == 0
    assert report["authorized_unique_loss_positions"] == 0
    assert report["canonical_capacity_credit_bytes"] == 0
    assert report["corpus_admitted"] is False
    assert report["full_shard_hash_verified"] is False
    assert report["item_level_rights_review"] == "REQUIRED"
    assert all(row["training_eligible"] is False for row in candidates)
    assert all(row["evaluation_eligible"] is False for row in candidates)
    assert all(row["item_rights_review_required"] is True for row in candidates)
    assert all("Metadata title" not in row["text"] for row in candidates)
    assert all("title" not in item for item in evidence)
    assert report["report_identity_sha256"] == self_identity(
        report, "report_identity_sha256"
    )


def test_exact_normalized_duplicate_is_quarantined_and_duplicate_id_is_fatal() -> None:
    same = good_text("same")
    candidates, _, report = materialize(
        load_config(),
        [record("A1", same), record("B2", same)],
    )
    assert [row["source_record_id"] for row in candidates] == ["A1"]
    assert report["rejection_counts"] == {"exact_normalized_duplicate": 1}

    with pytest.raises(UsgpoIntakeError, match="duplicate USGPO package id"):
        materialize(
            load_config(),
            [record("SAME"), record("SAME", good_text("different"))],
        )


def test_family_byte_cap_is_hard_and_not_credit() -> None:
    config = load_config()
    config["selection_policy"]["max_total_normalized_utf8_bytes"] = 2500
    config = reseal(config)
    with pytest.raises(UsgpoIntakeError, match="max_total_normalized_utf8_bytes drift"):
        validate_config(config)


def test_gzip_parser_is_bounded_utf8_strict_and_json_strict() -> None:
    rows = [record("A1"), record("B2")]
    payload = b"".join((json.dumps(row) + "\n").encode("utf-8") for row in rows)
    compressed = gzip.compress(payload, mtime=0)
    parsed = list(
        iter_gzip_jsonl_bytes(
            compressed,
            max_jsonl_line_bytes=100_000,
            max_decompressed_jsonl_bytes=500_000,
        )
    )
    assert parsed == rows

    bad_json = gzip.compress(b"not-json\n", mtime=0)
    with pytest.raises(UsgpoIntakeError, match="malformed JSONL"):
        list(iter_gzip_jsonl_bytes(bad_json, max_jsonl_line_bytes=100))

    bad_utf8 = gzip.compress(b"\xff\n", mtime=0)
    with pytest.raises(UsgpoIntakeError, match="invalid UTF-8"):
        list(iter_gzip_jsonl_bytes(bad_utf8, max_jsonl_line_bytes=100))

    long_line = gzip.compress(b'{"x":"' + b"a" * 200 + b'"}\n', mtime=0)
    with pytest.raises(UsgpoIntakeError, match="line exceeds"):
        list(iter_gzip_jsonl_bytes(long_line, max_jsonl_line_bytes=50))

    budget = gzip.compress(b'{"x":1}\n{"x":2}\n', mtime=0)
    with pytest.raises(UsgpoIntakeError, match="scan budget exceeded"):
        list(
            iter_gzip_jsonl_bytes(
                budget,
                max_jsonl_line_bytes=100,
                max_decompressed_jsonl_bytes=10,
            )
        )


def test_transport_rejects_before_gzip_parse_when_size_is_not_exact(tmp_path: Path) -> None:
    shard = tmp_path / "00017_usgpo.jsonl.gz"
    shard.write_bytes(gzip.compress(b"not-even-read\n", mtime=0))
    with pytest.raises(UsgpoIntakeError, match="compressed size mismatch"):
        verify_transport(load_config(), shard)
