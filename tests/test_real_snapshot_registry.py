from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.real_snapshot_registry import (
    ALLOWED,
    NOT_ADMITTED,
    RealSnapshotRegistryError,
    build_real_snapshot_registry,
    registry_identity,
    serialize_registry,
    sources_for_corpus,
    sources_for_holdout,
    sources_for_redistribution,
    validate_real_snapshot_registry,
    verify_source_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def _build():
    return build_real_snapshot_registry(
        inputs_path=ROOT / "configs/data/data229_real_snapshot_registry_v1.json",
        data213_plan_path=ROOT / "configs/data/data181_real_snapshot_promotion_v1.json",
        data24_registry_path=ROOT / "data/external/external_sources.json",
        data213_report_path=ROOT / "evidence/data229/data213-promotion-report.json",
        data213_artifact_manifest_path=ROOT / "evidence/data229/data213-artifact-manifest.json",
    )


def _reseal(registry):
    registry["registry_identity_sha256"] = registry_identity(registry)
    return registry


def test_two_independent_builds_are_byte_identical_and_match_committed_registry():
    first = serialize_registry(_build())
    second = serialize_registry(_build())
    assert first == second
    assert first == (ROOT / "data/registry/real_snapshots.v1.json").read_bytes()


def test_current_cutoff_contains_only_terminal_data213_sources():
    registry = _build()
    assert registry["source_count"] == 3
    assert registry["claim_boundary"]["code_source_count"] == 0
    assert registry["claim_boundary"]["missing_terminal_workers"] == ["DATA-227", "DATA-228"]
    assert {source["language"] for source in registry["sources"]} == {"en", "uk"}
    assert {source["origin_class"] for source in registry["sources"]} == {"EXTERNAL_REAL"}


def test_purpose_rights_are_separate_and_evaluation_does_not_inherit_training():
    registry = _build()
    assert len(sources_for_corpus(registry)) == 3
    assert len(sources_for_redistribution(registry)) == 3
    assert sources_for_holdout(registry) == ()
    for source in registry["sources"]:
        assert source["rights"]["model_training"]["status"] == ALLOWED
        assert source["rights"]["redistribution"]["status"] == ALLOWED
        assert source["rights"]["evaluation"]["status"] == NOT_ADMITTED
        assert source["rights"]["model_training"]["decision_identity_sha256"] != source["rights"]["redistribution"]["decision_identity_sha256"]
        assert source["rights"]["model_training"]["decision_identity_sha256"] != source["rights"]["evaluation"]["decision_identity_sha256"]


def test_registry_identity_changes_for_semantic_or_source_byte_identity_change():
    registry = _build()
    original = registry["registry_identity_sha256"]
    semantic = copy.deepcopy(registry)
    semantic["sources"][0]["language"] = "und"
    assert registry_identity(semantic) != original
    byte_identity = copy.deepcopy(registry)
    byte_identity["sources"][0]["raw_identity"]["raw_sha256"] = "0" * 64
    assert registry_identity(byte_identity) != original


def test_origin_namespace_and_raw_identity_prevent_silent_origin_conflation():
    registry = _build()
    changed = copy.deepcopy(registry)
    changed["sources"][0]["origin_class"] = "PROJECT_AUTHORED"
    changed = _reseal(changed)
    with pytest.raises(RealSnapshotRegistryError, match="origin class"):
        validate_real_snapshot_registry(changed)

    duplicated = copy.deepcopy(registry)
    clone = copy.deepcopy(duplicated["sources"][0])
    clone["registry_source_id"] = "project-authored:" + clone["raw_identity"]["source_id"]
    clone["origin_class"] = "PROJECT_AUTHORED"
    duplicated["sources"].append(clone)
    duplicated["sources"].sort(key=lambda item: item["registry_source_id"])
    duplicated["source_count"] += 1
    duplicated = _reseal(duplicated)
    with pytest.raises(RealSnapshotRegistryError, match="multiple origins"):
        validate_real_snapshot_registry(duplicated)


def test_api_filters_do_not_require_source_table_copying():
    registry = _build()
    en = sources_for_corpus(registry, languages=["en"], origins=["EXTERNAL_REAL"])
    uk = sources_for_corpus(registry, languages=["uk"], modalities=["text"])
    assert len(en) == 2
    assert len(uk) == 1
    assert all(item["registry_source_id"].startswith("external-real:") for item in en + uk)


def test_committed_registry_has_valid_self_identity():
    registry = json.loads((ROOT / "data/registry/real_snapshots.v1.json").read_text())
    validate_real_snapshot_registry(registry)
    assert registry["registry_identity_sha256"] == registry_identity(registry)


def test_materialized_source_bytes_must_match_raw_identity():
    payload = b"immutable source bytes"
    source = {"raw_identity": {"raw_sha256": hashlib.sha256(payload).hexdigest(), "raw_size_bytes": len(payload)}}
    verify_source_payload(source, payload)
    with pytest.raises(RealSnapshotRegistryError, match="source bytes"):
        verify_source_payload(source, payload + b"!")
