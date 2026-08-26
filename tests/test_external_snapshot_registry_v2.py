from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.data.external_snapshot_registry_v2 import (
    ExternalSnapshotRegistryV2Error,
    build_external_snapshot_registry_v2,
    serialize_registry,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "configs/data/data287_external_snapshot_registry_v2.json"
BASE = ROOT / "data/registry/real_snapshots.v1.json"
COMMITTED = ROOT / "data/registry/external_snapshots.v2.json"


def _build() -> dict[str, object]:
    return build_external_snapshot_registry_v2(
        inputs_path=INPUTS,
        base_registry_path=BASE,
    )


def _write_mutated_inputs(tmp_path: Path, mutate) -> Path:
    value = json.loads(INPUTS.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_committed_registry_is_byte_identical_to_two_independent_builds() -> None:
    first = serialize_registry(_build())
    second = serialize_registry(_build())
    assert first == second
    assert COMMITTED.read_bytes() == first


def test_terminal_inventory_and_unique_byte_report_are_exact() -> None:
    registry = _build()
    assert registry["source_count"] == 5
    assert registry["independent_source_family_count"] == 4
    report = registry["byte_report"]
    assert report["unique_raw_object_count"] == 5
    assert report["unique_raw_bytes"] == 448_214
    assert report["unique_normalized_object_count"] == 5
    assert report["unique_normalized_bytes"] == 183_061

    by_language = {row["key"]: row for row in report["by_language"]}
    assert (by_language["uk"]["unique_raw_bytes"], by_language["uk"]["unique_normalized_bytes"]) == (332_400, 88_565)
    assert (by_language["en"]["unique_raw_bytes"], by_language["en"]["unique_normalized_bytes"]) == (106_111, 84_793)
    assert (by_language["python"]["unique_raw_bytes"], by_language["python"]["unique_normalized_bytes"]) == (9_703, 9_703)

    by_modality = {row["key"]: row for row in report["by_modality"]}
    assert (by_modality["text"]["unique_raw_bytes"], by_modality["text"]["unique_normalized_bytes"]) == (438_511, 173_358)
    assert (by_modality["code"]["unique_raw_bytes"], by_modality["code"]["unique_normalized_bytes"]) == (9_703, 9_703)


def test_family_level_dedup_does_not_double_count_standard_ebooks_family() -> None:
    registry = _build()
    rows = {
        row["key"]: row
        for row in registry["byte_report"]["by_independent_source_family"]
    }
    assert set(rows) == {
        "en.standardebooks.manual",
        "github:encode/httpx",
        "github:psf/requests",
        "ua.rada.open-data.laws-texts",
    }
    assert rows["en.standardebooks.manual"]["snapshot_count"] == 2
    assert rows["en.standardebooks.manual"]["unique_raw_bytes"] == 106_111
    assert rows["github:encode/httpx"]["unique_raw_bytes"] == 8_161
    assert rows["github:psf/requests"]["unique_raw_bytes"] == 1_542
    assert registry["family_deduplication"]["excluded_mirror_or_fork_count"] == 0


def test_rights_remain_separate_and_evaluation_fails_closed() -> None:
    registry = _build()
    for source in registry["sources"]:
        assert source["rights"]["model_training"]["status"] == "ALLOWED"
        assert source["rights"]["redistribution"]["status"] == "ALLOWED"
        assert source["rights"]["evaluation"]["status"] == "NOT_SEPARATELY_ADMITTED"
    assert registry["claim_boundary"]["training_authorized_source_count"] == 5
    assert registry["claim_boundary"]["redistribution_authorized_source_count"] == 5
    assert registry["claim_boundary"]["evaluation_authorized_source_count"] == 0


def test_failed_data228_candidates_and_old_rights_blocked_code_are_absent() -> None:
    registry = _build()
    families = {
        source["independent_source_family"]["family_id"]
        for source in registry["sources"]
    }
    assert "kubernetes.website.docs" not in families
    assert "python.cpython.documentation" not in families
    assert "github:pallets/itsdangerous" not in families
    assert "github:pytest-dev/pluggy" not in families
    assert registry["terminal_authorities"]["DATA-228"]["status"] == "TERMINAL_FAILURE"
    assert registry["claim_boundary"]["failed_terminal_candidates_consumed"] is False


def test_data227_code_snapshots_preserve_exact_bytes_and_upstream_identity() -> None:
    registry = _build()
    code = [source for source in registry["sources"] if source["modality"] == "code"]
    assert len(code) == 2
    assert {source["independent_source_family"]["family_id"] for source in code} == {
        "github:encode/httpx",
        "github:psf/requests",
    }
    for source in code:
        snapshot = source["snapshot"]
        assert snapshot["raw_sha256"] == snapshot["normalized_sha256"]
        assert snapshot["raw_bytes"] == snapshot["normalized_bytes"]
        assert len(source["exact_upstream_identity"]["git_blob_sha1"]) == 40


def test_non_terminal_success_source_producer_is_rejected(tmp_path: Path) -> None:
    path = _write_mutated_inputs(
        tmp_path,
        lambda value: value["sources"][0].update({"producer": "DATA-228"}),
    )
    with pytest.raises(ExternalSnapshotRegistryV2Error, match="non-terminal-success producer"):
        build_external_snapshot_registry_v2(inputs_path=path, base_registry_path=BASE)


def test_mirror_or_fork_is_rejected_at_family_boundary(tmp_path: Path) -> None:
    path = _write_mutated_inputs(
        tmp_path,
        lambda value: value["sources"][3].update({"mirror": True}),
    )
    with pytest.raises(ExternalSnapshotRegistryV2Error, match="mirror/fork"):
        build_external_snapshot_registry_v2(inputs_path=path, base_registry_path=BASE)


def test_data229_text_source_version_drift_is_rejected(tmp_path: Path) -> None:
    path = _write_mutated_inputs(
        tmp_path,
        lambda value: value["sources"][0].update({"source_version": "git:deadbeef"}),
    )
    with pytest.raises(ExternalSnapshotRegistryV2Error, match="source_version drift"):
        build_external_snapshot_registry_v2(inputs_path=path, base_registry_path=BASE)
