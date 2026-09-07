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
        archive.writestr("d1.htm", "<p>Закон України</p>".encode("utf-8"))
    return buffer.getvalue()


def _probe_config() -> dict:
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
            "min_canonical_entries": 1,
            "max_archive_bytes": 1_000_000,
            "max_entry_bytes": 100_000,
            "max_total_uncompressed_bytes": 1_000_000,
        },
    }


def _materialization_inputs() -> tuple[bytes, dict, dict, str]:
    archive = _archive()
    report = inventory_archive(
        archive,
        _probe_config(),
        expected_md5=hashlib.md5(archive, usedforsecurity=False).hexdigest(),
        expected_bytes=len(archive),
    )
    config = copy.deepcopy(CONFIG)
    config["parent_probe"]["probe_worker_id"] = report["worker_id"]
    config["parent_probe"]["probe_config_identity_sha256"] = report[
        "config_identity_sha256"
    ]
    report_bytes = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return archive, report, config, hashlib.sha256(report_bytes).hexdigest()


def test_output_contract_field_drift_fails_closed() -> None:
    archive, report, config, report_sha = _materialization_inputs()
    config["output_contract"]["jsonl_record_fields"].remove("source_encoding")

    with pytest.raises(NormalizationError, match="JSONL record field contract drift"):
        materialize_normalized_records(
            archive,
            report,
            config,
            probe_report_sha256=report_sha,
        )


def test_output_contract_boolean_weakening_fails_closed() -> None:
    archive, report, config, report_sha = _materialization_inputs()
    config["output_contract"]["manifest_is_text_free"] = False

    with pytest.raises(NormalizationError, match="output contract weakened"):
        materialize_normalized_records(
            archive,
            report,
            config,
            probe_report_sha256=report_sha,
        )


def test_manifest_binds_exact_normalization_config_identity() -> None:
    archive, report, config, report_sha = _materialization_inputs()
    jsonl, manifest = materialize_normalized_records(
        archive,
        report,
        config,
        probe_report_sha256=report_sha,
    )

    canonical_config = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_identity = hashlib.sha256(canonical_config).hexdigest()
    row = json.loads(jsonl.decode("utf-8"))

    assert manifest["normalization_config_identity_sha256"] == expected_identity
    assert manifest["normalization"]["decode_policy"] == config["normalization"]["decode"]
    assert manifest["normalization"]["legacy_fallback_encoding"] == "windows-1251"
    assert row["source_encoding"] == "utf-8"
    assert manifest["training_authorized_bytes"] == 0