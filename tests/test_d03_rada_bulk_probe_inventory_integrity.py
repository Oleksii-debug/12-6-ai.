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
)
from tools.probe_d03_rada_bulk_source import inventory_archive

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/data/d03_rada_bulk_normalization_v1.json").read_text(
        encoding="utf-8"
    )
)


def _archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("d1.htm", b"<html><body><p>one</p></body></html>")
        archive.writestr("nested/d2.htm", b"<html><body><p>two</p></body></html>")
    return buffer.getvalue()


def _probe_config() -> dict:
    return {
        "schema_version": "12-6.d03-rada-bulk-source-probe.v1",
        "worker_id": "TEST-RADA-PROBE-INVENTORY",
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
            "min_canonical_entries": 2,
            "max_archive_bytes": 1_000_000,
            "max_entry_bytes": 100_000,
            "max_total_uncompressed_bytes": 1_000_000,
        },
    }


def _strict_probe(archive: bytes) -> dict:
    return inventory_archive(
        archive,
        _probe_config(),
        expected_md5=hashlib.md5(archive, usedforsecurity=False).hexdigest(),
        expected_bytes=len(archive),
    )


def _normalizer_config(report: dict) -> dict:
    config = copy.deepcopy(CONFIG)
    config["parent_probe"]["probe_worker_id"] = report["worker_id"]
    config["parent_probe"]["probe_config_identity_sha256"] = report[
        "config_identity_sha256"
    ]
    return config


def _report_sha(report: dict) -> str:
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _materialize(archive: bytes, report: dict) -> None:
    materialize_normalized_records(
        archive,
        report,
        _normalizer_config(report),
        probe_report_sha256=_report_sha(report),
    )


def test_mutated_probe_entry_identity_fails_closed() -> None:
    archive = _archive()
    report = _strict_probe(archive)
    report["inventory"]["entry_identity_sha256"] = "0" * 64

    with pytest.raises(NormalizationError, match="entry-identity SHA-256 drift"):
        _materialize(archive, report)


def test_mutated_probe_canonical_raw_total_fails_closed() -> None:
    archive = _archive()
    report = _strict_probe(archive)
    report["inventory"]["canonical_raw_bytes"] += 1

    with pytest.raises(NormalizationError, match="canonical raw-byte total drift"):
        _materialize(archive, report)
