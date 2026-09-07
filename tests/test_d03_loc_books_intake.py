from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.data.loc_books_intake import (
    LocBooksIntakeError,
    classify_row,
    materialize_from_gzip_bytes,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/d03_loc_books_intake_v1.json"


def config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def row(record_id: str, text: str, **meta_overrides):
    metadata = {
        "license": "Public Domain",
        "title": f"Book {record_id}",
        "author": "Public Domain Author",
        "year": 1899,
        "language": "english",
        "item_url": f"https://www.loc.gov/item/{record_id}",
        "text_file_url": f"https://tile.loc.gov/storage-services/service/{record_id}.txt",
    }
    metadata.update(meta_overrides)
    return {
        "id": record_id,
        "text": text,
        "source": "loc_books",
        "added": "2024-05-14T16:22:35.184395",
        "metadata": metadata,
    }


def good_text(seed: str, repetitions: int = 120):
    phrase = f"{seed} This is a historical public domain book passage with readable English prose. "
    return (phrase * repetitions).strip()


def gz_rows(rows):
    payload = b"".join(
        (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for item in rows
    )
    return gzip.compress(payload, mtime=0)


def test_config_and_zero_credit_boundary():
    cfg = config()
    validate_config(cfg)
    for field in (
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "canonical_capacity_credit_bytes",
    ):
        assert cfg[field] == 0
    assert cfg["model_training_permitted"] is False
    assert cfg["evaluation_eligible"] is False
    assert cfg["paid_compute_authorized"] is False


def test_per_record_rights_and_provenance_fail_closed():
    cfg = config()
    assert classify_row(cfg, row("01000001", good_text("ok"))).status == "ACCEPT"
    assert classify_row(
        cfg, row("01000002", good_text("bad"), license="CC-BY-4.0")
    ).reason == "license_not_public_domain"
    assert classify_row(
        cfg, row("01000003", good_text("bad"), language="French")
    ).reason == "language_not_english"
    assert classify_row(
        cfg,
        row(
            "01000004",
            good_text("bad"),
            item_url="https://example.org/item/01000004",
        ),
    ).reason == "invalid_loc_item_url"
    assert classify_row(
        cfg,
        row(
            "01000005",
            good_text("bad"),
            text_file_url="https://example.org/book.txt",
        ),
    ).reason == "invalid_loc_text_url"


def test_text_safety_schema_and_quality_fail_closed():
    cfg = config()
    assert classify_row(cfg, row("01000006", good_text("x") + "\x00")).reason == (
        "disallowed_control_character"
    )
    assert classify_row(
        cfg, row("01000007", good_text("x") + " person@example.com")
    ).reason == "obvious_contact_email"
    assert classify_row(
        cfg, row("01000008", good_text("x") + " AKIAABCDEFGHIJKLMNOP")
    ).reason == "obvious_secret_marker"
    assert classify_row(cfg, row("01000009", "too short")).reason == "text_too_short"
    bad = row("01000010", good_text("x"))
    bad["unexpected"] = True
    assert classify_row(cfg, bad).reason == "row_schema_drift"


def test_deterministic_materialization_dedup_and_zero_credit():
    cfg = config()
    cfg["selection_policy"]["stop_after_min_text_bytes"] = 8_000
    cfg["selection_policy"]["max_total_text_bytes"] = 40_000
    # Recompute test-only identity after bounded test-policy edits.
    from twelve_six.data.loc_books_intake import self_identity

    cfg["contract_identity_sha256"] = self_identity(cfg, "contract_identity_sha256")
    text_a = good_text("alpha", 80)
    text_b = good_text("beta", 80)
    rows = [
        row("02000001", text_a),
        row("02000002", text_a),
        row("02000003", good_text("bad"), license="Unknown"),
        row("02000004", text_b),
    ]
    compressed = gz_rows(rows)
    first_candidates, first_report = materialize_from_gzip_bytes(cfg, compressed)
    second_candidates, second_report = materialize_from_gzip_bytes(cfg, compressed)
    assert first_candidates == second_candidates
    assert first_report == second_report
    assert [item["source_record_id"] for item in first_candidates] == ["02000001", "02000004"]
    assert first_report["rejection_counts"]["exact_normalized_duplicate"] == 1
    assert first_report["rejection_counts"]["license_not_public_domain"] == 1
    assert first_report["training_authorized_bytes"] == 0
    assert first_report["authorized_unique_loss_positions"] == 0
    assert first_report["full_shard_sha256_verified"] is False
    assert all(item["training_eligible"] is False for item in first_candidates)
    assert all(item["evaluation_eligible"] is False for item in first_candidates)


def test_family_cap_is_strictly_below_five_megabytes():
    cfg = config()
    assert cfg["selection_policy"]["max_total_text_bytes"] < 5_000_000
    drift = deepcopy(cfg)
    drift["selection_policy"]["max_total_text_bytes"] = 5_000_000
    from twelve_six.data.loc_books_intake import self_identity

    drift["contract_identity_sha256"] = self_identity(drift, "contract_identity_sha256")
    with pytest.raises(LocBooksIntakeError, match="planning cap"):
        validate_config(drift)


def test_stream_prefix_cannot_claim_full_shard_verification():
    cfg = config()
    with pytest.raises(LocBooksIntakeError, match="cannot assert full-shard"):
        from twelve_six.data.loc_books_intake import materialize_from_gzip_stream
        import io

        stream = io.BytesIO(gz_rows([row("03000001", good_text("x"))]))
        materialize_from_gzip_stream(
            cfg, stream, full_shard_sha256_verified=True
        )
