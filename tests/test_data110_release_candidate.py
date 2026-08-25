from __future__ import annotations

from pathlib import Path

from twelve_six.data110_release_candidate import (
    _classification,
    _model,
    _source_registry,
)


def test_s2_byte_vertical_is_in_requested_parameter_band() -> None:
    spec, _init, provenance = _model(Path("."))
    assert spec.parameter_count() == 836_736
    assert 100_000 <= spec.parameter_count() <= 1_000_000
    assert provenance["only_geometry_change"] == "vocab_size:2048->256 to bind canonical s0-byte-v1"


def test_release_classification_stays_fail_closed() -> None:
    value = _classification(
        [
            {"source_id": "ua.real"},
            {"source_id": "en.real"},
        ],
        0,
    )
    assert value["status"] == "RETEST_REQUIRED"
    codes = {item["code"] for item in value["machine_readable_reasons"]}
    assert "EXTERNAL_SOURCE_DIVERSITY_TOO_NARROW" in codes
    assert "NO_EXTERNAL_CODE_SOURCE" in codes
    assert "D06_PRODUCTION_REGISTRY_SPARSE" in codes
    assert "REPRESENTATIVENESS_NOT_ESTABLISHED" in codes


def test_source_registry_separates_real_and_project_authored_origins() -> None:
    external = [
        {
            "source_id": "ua.real",
            "source_version": "v1",
            "rights_status": "APPROVED_FOR_TRAINING",
            "license_id": "TEST",
            "source_identity_sha256": "1" * 64,
            "raw_sha256": "2" * 64,
            "content_sha256": "3" * 64,
        }
    ]
    project = [
        {
            "source_id": "project-authored:code:test",
            "source_version": "v1",
        }
    ]
    registry = _source_registry(external, project)
    by_id = {row["source_id"]: row for row in registry["sources"]}
    assert by_id["ua.real"]["origin"] == "external_real"
    assert by_id["ua.real"]["allows_model_training"] is True
    assert by_id["project-authored:code:test"]["origin"] == "project_authored"
    assert by_id["project-authored:code:test"]["rights_status"] == "PROJECT_CONTROLLED"
