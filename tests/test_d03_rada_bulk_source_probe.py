from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from tools.probe_d03_rada_bulk_source import ProbeError, inventory_archive


def _config(min_entries: int = 2) -> dict:
    return {
        "schema_version": "12-6.d03-rada-bulk-source-probe.v1",
        "worker_id": "TEST-RADA-BULK",
        "local_free_only": True,
        "model_training_executed": False,
        "training_authorized_bytes": 0,
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


def _archive(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buffer.getvalue()


def test_inventory_is_deterministic_and_keeps_training_closed() -> None:
    archive = _archive(
        {
            "d2.htm": b"<html>two</html>",
            "nested/d1.htm": b"<html>one</html>",
            "README.txt": b"ignored",
        }
    )
    md5 = hashlib.md5(archive, usedforsecurity=False).hexdigest()
    report_a = inventory_archive(
        archive,
        _config(),
        expected_md5=md5,
        expected_bytes=len(archive),
    )
    report_b = inventory_archive(
        archive,
        _config(),
        expected_md5=md5,
        expected_bytes=len(archive),
    )

    assert report_a == report_b
    assert report_a["inventory"]["canonical_entry_count"] == 2
    assert report_a["inventory"]["ignored_file_count"] == 1
    assert [row["basename"] for row in report_a["inventory"]["entries"]] == [
        "d1.htm",
        "d2.htm",
    ]
    assert report_a["training_authorized_bytes"] == 0
    assert report_a["corpus_admitted"] is False
    assert report_a["gates"]["canonical_normalization"] == "NOT_RUN"


def test_rejects_archive_identity_drift() -> None:
    archive = _archive({"d1.htm": b"a", "d2.htm": b"b"})
    with pytest.raises(ProbeError, match="byte identity drift"):
        inventory_archive(archive, _config(), expected_bytes=len(archive) + 1)


def test_rejects_duplicate_canonical_basenames() -> None:
    archive = _archive({"a/d1.htm": b"a", "b/d1.htm": b"b"})
    with pytest.raises(ProbeError, match="duplicate canonical basename"):
        inventory_archive(archive, _config())


def test_rejects_path_traversal() -> None:
    archive = _archive({"../d1.htm": b"a", "d2.htm": b"b"})
    with pytest.raises(ProbeError, match="unsafe archive path"):
        inventory_archive(archive, _config())


def test_rejects_too_few_canonical_entries() -> None:
    archive = _archive({"d1.htm": b"a", "readme.txt": b"x"})
    with pytest.raises(ProbeError, match="below minimum"):
        inventory_archive(archive, _config(min_entries=2))
