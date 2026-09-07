from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from twelve_six.data.loc_books_intake import (
    DATASET_REVISION,
    LocBooksIntakeError,
    decide_record,
    iter_gzip_jsonl,
    load_config,
    materialize_records,
    normalize_text,
    self_identity,
    validate_config,
    verify_bound_shard,
    verify_two_builds,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/d03_loc_books_intake_v1.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def valid_text(label: str = "history") -> str:
    paragraph = (
        f"This {label} volume records the public history of institutions and communities. "
        "The chapters describe events, places, laws, observations, and ordinary affairs "
        "in clear English prose for readers and researchers. "
    )
    return (paragraph * 80) + "\n"


def valid_record(record_id: str = "01020304", text: str | None = None) -> dict:
    return {
        "id": record_id,
        "text": valid_text() if text is None else text,
        "source": "loc_books",
        "added": "2024-05-14T16:22:35.184395",
        "metadata": {
            "license": "Public Domain",
            "title": "A historical volume",
            "author": "Example Author",
            "year": 1901,
            "language": "english",
            "item_url": f"https://www.loc.gov/item/{record_id}",
            "text_file_url": (
                "https://tile.loc.gov/storage-services/public/gdcmassbookdig/"
                f"{record_id}/{record_id}_djvu.txt"
            ),
        },
    }


def snapshot() -> dict:
    return {
        "dataset": "synthetic-test-only",
        "revision": DATASET_REVISION,
        "shard_path": "fixture.jsonl.gz",
        "compressed_bytes": 1,
        "shard_sha256": "0" * 64,
        "full_shard_hash_verified": False,
    }


def test_config_is_valid_and_self_bound() -> None:
    value = load_config(CONFIG_PATH)
    validate_config(value)
    assert self_identity(value, "contract_identity_sha256") == value["contract_identity_sha256"]
    assert value["training_authorized_bytes"] == 0
    assert value["authorized_unique_loss_positions"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_authorized_bytes", 1),
        ("corpus_admitted", True),
        ("tokenizer_fit_permitted", True),
        ("model_training_permitted", True),
        ("evaluation_use_permitted", True),
        ("paid_compute_authorized", True),
    ],
)
def test_config_rejects_authority_drift(field: str, value: object) -> None:
    altered = config()
    altered[field] = value
    altered["contract_identity_sha256"] = self_identity(altered, "contract_identity_sha256")
    with pytest.raises(LocBooksIntakeError, match="drift"):
        validate_config(altered)


def test_config_rejects_upstream_hash_drift_even_if_resigned() -> None:
    altered = config()
    altered["upstream"]["shard_sha256"] = "1" * 64
    altered["contract_identity_sha256"] = self_identity(altered, "contract_identity_sha256")
    with pytest.raises(LocBooksIntakeError, match="upstream.shard_sha256 drift"):
        validate_config(altered)


def test_config_rejects_family_ceiling_drift() -> None:
    altered = config()
    altered["selection_policy"]["max_total_candidate_bytes"] = 5_000_000
    altered["contract_identity_sha256"] = self_identity(altered, "contract_identity_sha256")
    with pytest.raises(LocBooksIntakeError, match="family planning ceiling"):
        validate_config(altered)


def test_normalization_is_stable() -> None:
    source = "Cafe\u0301  \r\n\r\n\r\n\r\nLine\t \rTail"
    assert normalize_text(source) == "Café\n\n\nLine\nTail\n"
    assert normalize_text(normalize_text(source)) == normalize_text(source)


def test_valid_record_is_admitted_without_metadata_in_text() -> None:
    value = config()
    record = valid_record()
    decision = decide_record(record, value)
    assert decision.accepted is True
    assert decision.record_id == record["id"]
    assert decision.normalized_bytes > 4_096
    assert "Example Author" not in (decision.normalized_text or "")


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda r: r.update(source="other"), "wrong_source"),
        (lambda r: r["metadata"].update(license="CC-BY"), "wrong_license"),
        (lambda r: r["metadata"].update(language="french"), "wrong_language"),
        (lambda r: r["metadata"].update(year=1499), "year_below_floor"),
        (lambda r: r["metadata"].update(item_url="http://example.com/x"), "invalid_item_url"),
        (
            lambda r: r["metadata"].update(text_file_url="https://example.com/file.txt"),
            "invalid_text_file_url",
        ),
        (lambda r: r.update(text="short text"), "too_short"),
        (
            lambda r: r.update(text=valid_text() + "\ncontact@example.com\n"),
            "contact_or_secret_marker",
        ),
        (
            lambda r: r.update(text=valid_text() + "\n-----BEGIN PRIVATE KEY-----\n"),
            "contact_or_secret_marker",
        ),
        (
            lambda r: r.update(text=valid_text() + "\x00"),
            "control_character",
        ),
    ],
)
def test_record_fail_closed(mutator, reason: str) -> None:
    record = valid_record()
    mutator(record)
    decision = decide_record(record, config())
    assert decision.accepted is False
    assert decision.reason == reason


def test_materialization_dedups_normalized_payload_and_is_text_free_in_report(
    tmp_path: Path,
) -> None:
    value = config()
    first = valid_record("A001", valid_text("archive"))
    duplicate_text = valid_record(
        "A002",
        valid_text("archive").replace("\n", "\r\n"),
    )
    second = valid_record("A003", valid_text("science"))
    rejected = valid_record("A004")
    rejected["metadata"]["license"] = "unknown"

    candidate = tmp_path / "candidate.jsonl"
    report_path = tmp_path / "report.json"
    report = materialize_records(
        [first, duplicate_text, second, rejected],
        value,
        source_snapshot=snapshot(),
        candidate_path=candidate,
        report_path=report_path,
    )

    assert report["accepted_documents"] == 2
    assert report["rejection_counts"] == {
        "duplicate_normalized_text": 1,
        "wrong_license": 1,
    }
    assert report["training_authorized_bytes"] == 0
    assert report["authorized_unique_loss_positions"] == 0
    report_text = report_path.read_text(encoding="utf-8")
    assert "A001" not in report_text
    assert "A003" not in report_text
    assert "Example Author" not in report_text
    rows = [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["A001", "A003"]
    assert all(row["training_eligible"] is False for row in rows)
    assert all(row["evaluation_eligible"] is False for row in rows)


def test_divergent_duplicate_id_aborts_package(tmp_path: Path) -> None:
    with pytest.raises(LocBooksIntakeError, match="record id collision"):
        materialize_records(
            [
                valid_record("DUP", valid_text("first")),
                valid_record("DUP", valid_text("second")),
            ],
            config(),
            source_snapshot=snapshot(),
            candidate_path=tmp_path / "candidate.jsonl",
            report_path=tmp_path / "report.json",
        )
    assert not (tmp_path / "candidate.jsonl").exists()
    assert not (tmp_path / "report.json").exists()


def test_two_independent_builds_are_byte_identical(tmp_path: Path) -> None:
    records = [
        valid_record("B001", valid_text("history")),
        valid_record("B002", valid_text("science")),
    ]
    a = tmp_path / "a"
    b = tmp_path / "b"
    for folder in (a, b):
        materialize_records(
            copy.deepcopy(records),
            config(),
            source_snapshot=snapshot(),
            candidate_path=folder / "candidate.jsonl",
            report_path=folder / "report.json",
        )
    result = verify_two_builds(
        candidate_a=a / "candidate.jsonl",
        report_a=a / "report.json",
        candidate_b=b / "candidate.jsonl",
        report_b=b / "report.json",
    )
    assert result["status"] == "BYTE_IDENTICAL_TWO_BUILD_PASS"


def test_two_build_verifier_detects_tampering(tmp_path: Path) -> None:
    for name, payload in (
        ("a.jsonl", b"same\n"),
        ("b.jsonl", b"different\n"),
        ("ra.json", b"{}\n"),
        ("rb.json", b"{}\n"),
    ):
        (tmp_path / name).write_bytes(payload)
    with pytest.raises(LocBooksIntakeError, match="candidate mismatch"):
        verify_two_builds(
            candidate_a=tmp_path / "a.jsonl",
            report_a=tmp_path / "ra.json",
            candidate_b=tmp_path / "b.jsonl",
            report_b=tmp_path / "rb.json",
        )


def test_gzip_jsonl_reader_rejects_malformed_json(tmp_path: Path) -> None:
    shard = tmp_path / "bad.jsonl.gz"
    with gzip.open(shard, "wb") as handle:
        handle.write(b'{"ok": true}\nnot-json\n')
    iterator = iter_gzip_jsonl(shard, max_line_bytes=1000)
    assert next(iterator) == {"ok": True}
    with pytest.raises(LocBooksIntakeError, match="malformed JSON"):
        next(iterator)


def test_gzip_jsonl_reader_rejects_oversized_line(tmp_path: Path) -> None:
    shard = tmp_path / "large.jsonl.gz"
    with gzip.open(shard, "wb") as handle:
        handle.write(json.dumps({"x": "a" * 200}).encode("utf-8") + b"\n")
    with pytest.raises(LocBooksIntakeError, match="exceeds"):
        list(iter_gzip_jsonl(shard, max_line_bytes=100))


def test_bound_shard_verification_rejects_wrong_size_before_hash(tmp_path: Path) -> None:
    fake = tmp_path / "shard.gz"
    fake.write_bytes(b"not the 358 MB immutable shard")
    with pytest.raises(LocBooksIntakeError, match="size mismatch"):
        verify_bound_shard(fake, config())
