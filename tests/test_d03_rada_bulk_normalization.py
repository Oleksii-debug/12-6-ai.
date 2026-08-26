from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from tools.normalize_d03_rada_bulk_html import (
    NormalizationError,
    materialize_normalized_records,
    normalize_html_bytes,
)
from tools.probe_d03_rada_bulk_source import inventory_archive

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/data/d03_rada_bulk_normalization_v1.json").read_text(
        encoding="utf-8"
    )
)


def _archive(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _probe_config(min_entries: int) -> dict:
    return {
        "schema_version": "12-6.d03-rada-bulk-source-probe.v1",
        "worker_id": "TEST-RADA-PROBE",
        "local_free_only": True,
        "model_training_executed": False,
        "training_authorized_bytes": 0,
        "parent_authority": {
            "head_sha": "f" * 40,
            "registry_identity_sha256": "e" * 64,
        },
        "source": {
            "family_id": "ua.rada.open-data.laws-texts",
            "dataset_id": "laws-texts",
            "archive_url": "https://example.invalid/texts.zip",
        },
        "probe_policy": {
            "canonical_entry_regex": r"d[0-9]+\.htm",
            "min_canonical_entries": min_entries,
            "max_archive_bytes": 1_000_000,
            "max_entry_bytes": 100_000,
            "max_total_uncompressed_bytes": 1_000_000,
        },
    }


def _probe(archive: bytes, min_entries: int) -> dict:
    report = inventory_archive(archive, _probe_config(min_entries))
    assert report["training_authorized_bytes"] == 0
    return report


def _report_sha(report: dict) -> str:
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _materialize(archive: bytes, report: dict) -> tuple[bytes, dict]:
    return materialize_normalized_records(
        archive,
        report,
        CONFIG,
        probe_report_sha256=_report_sha(report),
    )


def test_visible_text_extraction_is_nfkc_and_hides_noncontent() -> None:
    html = (
        b"\xef\xbb\xbf<html><head><title>hidden title</title></head><body>"
        b"<h1>\xd0\x97\xd0\xb0\xd0\xba\xd0\xbe\xd0\xbd</h1>"
        b"<script>secret()</script><style>.x{display:none}</style>"
        b"<p>A\xc2\xa0B &amp; C</p><div>  D\tE  </div></body></html>"
    )
    text = normalize_html_bytes(html, CONFIG)
    assert text == "Закон\nA B & C\nD E"
    assert "hidden title" not in text
    assert "secret" not in text
    assert "display" not in text


def test_two_clean_materializations_are_byte_identical_and_zero_credit() -> None:
    archive = _archive(
        {
            "d2.htm": "<html><body><p>Другий закон</p></body></html>".encode(),
            "nested/d1.htm": (
                "<html><body><h2>Перший</h2><p>Текст&nbsp;акта</p></body></html>"
            ).encode(),
            "README.txt": b"ignored",
        }
    )
    report = _probe(archive, 2)
    jsonl_a, manifest_a = _materialize(archive, report)
    jsonl_b, manifest_b = _materialize(archive, report)

    assert jsonl_a == jsonl_b
    assert manifest_a == manifest_b
    assert manifest_a["normalization"]["record_count"] == 2
    assert manifest_a["training_authorized_bytes"] == 0
    assert manifest_a["normalized_capacity_credited"] == 0
    assert manifest_a["tokenizer_fit_authorized"] is False
    assert manifest_a["model_training_executed"] is False
    assert manifest_a["gates"]["quality"] == "NOT_RUN"
    assert manifest_a["gates"]["evaluation_decontamination"] == "NOT_RUN"

    rows = [json.loads(line) for line in jsonl_a.decode().splitlines()]
    assert [row["record_id"] for row in rows] == [
        "ua.rada.open-data.laws-texts.d1",
        "ua.rada.open-data.laws-texts.d2",
    ]
    assert rows[0]["text"] == "Перший\nТекст акта"


def test_archive_bytes_must_match_exact_probe() -> None:
    archive = _archive({"d1.htm": b"<p>one</p>"})
    report = _probe(archive, 1)
    tampered = _archive({"d1.htm": b"<p>changed</p>"})
    with pytest.raises(NormalizationError, match="archive SHA-256 does not match probe"):
        _materialize(tampered, report)


def test_entry_hash_must_match_probe_even_if_archive_binding_is_resealed() -> None:
    archive = _archive({"d1.htm": b"<p>one</p>"})
    report = _probe(archive, 1)
    broken = copy.deepcopy(report)
    broken["inventory"]["entries"][0]["raw_sha256"] = "0" * 64
    with pytest.raises(NormalizationError, match="raw SHA-256 drift"):
        _materialize(archive, broken)


def test_probe_cannot_hide_a_canonical_archive_entry() -> None:
    archive = _archive({"d1.htm": b"<p>one</p>", "d2.htm": b"<p>two</p>"})
    report = _probe(archive, 2)
    broken = copy.deepcopy(report)
    broken["inventory"]["entries"] = broken["inventory"]["entries"][:1]
    broken["inventory"]["canonical_entry_count"] = 1
    with pytest.raises(NormalizationError, match="absent from probe"):
        _materialize(archive, broken)


def test_invalid_utf8_fails_closed_at_normalization() -> None:
    archive = _archive({"d1.htm": b"<p>bad:\xff</p>"})
    report = _probe(archive, 1)
    with pytest.raises(NormalizationError, match="not strict UTF-8"):
        _materialize(archive, report)


def test_empty_visible_document_is_retained_for_later_quality_rejection() -> None:
    archive = _archive({"d1.htm": b"<html><head>x</head><script>y</script></html>"})
    report = _probe(archive, 1)
    jsonl, manifest = _materialize(archive, report)
    row = json.loads(jsonl.decode())
    assert row["text"] == ""
    assert row["normalized_bytes"] == 0
    assert manifest["normalization"]["nonempty_record_count"] == 0
    assert manifest["gates"]["quality"] == "NOT_RUN"


def test_truth_boundary_mutation_is_rejected() -> None:
    archive = _archive({"d1.htm": b"<p>one</p>"})
    report = _probe(archive, 1)
    config = copy.deepcopy(CONFIG)
    config["claim_boundary"]["training_authorized_bytes"] = 1
    with pytest.raises(NormalizationError, match="authorize zero training bytes"):
        materialize_normalized_records(
            archive,
            report,
            config,
            probe_report_sha256=_report_sha(report),
        )
