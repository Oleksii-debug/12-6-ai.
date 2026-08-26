from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from tools.probe_d03_rada_bulk_source import (
    DEFAULT_CONFIG,
    ProbeError,
    _load_config,
    inventory_archive,
)


def _config(min_entries: int = 2) -> dict:
    return {
        "schema_version": "12-6.d03-rada-bulk-source-probe.v1",
        "worker_id": "TEST-RADA-BULK",
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


def _archive(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buffer.getvalue()


def _production_config() -> dict:
    return json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


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
    assert report_a["gates"]["exact_archive_identity"] == (
        "PASS_PINNED_DISCOVERY_REVALIDATED"
    )
    assert report_a["safe_result"] == (
        "PINNED_BULK_ARCHIVE_INVENTORIED_DOWNSTREAM_GATES_REQUIRED"
    )
    assert len(report_a["config_identity_sha256"]) == 64


def test_unpinned_observation_is_machine_distinct_from_strict_pass() -> None:
    archive = _archive({"d1.htm": b"a", "d2.htm": b"b"})

    report = inventory_archive(archive, _config())

    assert report["gates"]["exact_archive_identity"] == "OBSERVED_UNPINNED"
    assert report["safe_result"] == "CURRENT_UPSTREAM_OBSERVED_SUCCESSOR_PIN_REQUIRED"
    assert report["training_authorized_bytes"] == 0


def test_rejects_archive_identity_drift() -> None:
    archive = _archive({"d1.htm": b"a", "d2.htm": b"b"})
    md5 = hashlib.md5(archive, usedforsecurity=False).hexdigest()
    with pytest.raises(ProbeError, match="byte identity drift"):
        inventory_archive(
            archive,
            _config(),
            expected_md5=md5,
            expected_bytes=len(archive) + 1,
        )


def test_rejects_partial_expected_identity() -> None:
    archive = _archive({"d1.htm": b"a", "d2.htm": b"b"})
    with pytest.raises(ProbeError, match="must be supplied together"):
        inventory_archive(archive, _config(), expected_bytes=len(archive))


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


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda cfg: cfg["parent_authority"].__setitem__("head_sha", "0" * 40),
            "parent_authority.head_sha drifted",
        ),
        (
            lambda cfg: cfg["parent_authority"].__setitem__(
                "registry_identity_sha256", "0" * 64
            ),
            "parent_authority.registry_identity_sha256 drifted",
        ),
        (
            lambda cfg: cfg["source"].__setitem__(
                "archive_url", "https://example.invalid/other.zip"
            ),
            "source.archive_url drifted",
        ),
        (
            lambda cfg: cfg["probe_policy"].__setitem__("canonical_entry_regex", ".*"),
            "probe_policy.canonical_entry_regex drifted",
        ),
        (
            lambda cfg: cfg["probe_policy"].__setitem__("max_archive_bytes", 9_999_999_999),
            "probe_policy.max_archive_bytes drifted",
        ),
        (
            lambda cfg: cfg["rights_boundary"].__setitem__(
                "bulk_extension_status", "ADMITTED"
            ),
            "bulk extension must remain not admitted",
        ),
        (
            lambda cfg: cfg["claim_boundary"].__setitem__(
                "training_exposure_authorized", True
            ),
            "training_exposure_authorized must remain false",
        ),
    ],
)
def test_production_config_authority_drift_fails_closed(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    config = copy.deepcopy(_production_config())
    mutator(config)

    with pytest.raises(ProbeError, match=message):
        _load_config(_write_config(tmp_path, config))


def test_production_config_loads_under_exact_v1_authority() -> None:
    config = _load_config(DEFAULT_CONFIG)

    assert config["source"]["dataset_id"] == "laws-texts"
    assert config["training_authorized_bytes"] == 0
