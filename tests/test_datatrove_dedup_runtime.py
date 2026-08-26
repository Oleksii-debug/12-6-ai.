from __future__ import annotations

import pytest

from twelve_six.data.datatrove_dedup_runtime import (
    DataTroveMinhashError,
    DataTroveMinhashSpec,
)

H = "a" * 64
R = "b" * 64


def _spec(**overrides) -> DataTroveMinhashSpec:
    values = {
        "source_registry_sha256": H,
        "reserved_registry_sha256": R,
        "input_uri": "file:///tmp/input",
        "output_uri": "file:///tmp/output",
        "work_uri": "file:///tmp/minhash-work",
        "logging_uri": "file:///tmp/minhash-logs",
        "language": "uk",
        "signature_tasks": 32,
        "workers": 4,
    }
    values.update(overrides)
    return DataTroveMinhashSpec(**values)


def test_runtime_plan_matches_datatrove_0100_public_minhash_shape() -> None:
    spec = _spec()
    manifest = spec.manifest()

    assert spec.num_buckets == 14
    assert spec.hashes_per_bucket == 8
    assert spec.total_signature_hashes == 112
    assert spec.n_grams == 5
    assert spec.hash_precision == 64
    assert manifest["language"] == "uk"
    assert manifest["datatrove_version"] == "0.10.0"
    assert manifest["total_signature_hashes"] == 112
    assert manifest["within_partition_near_dedup"] is True
    assert manifest["global_cross_partition_dedup_claimed"] is False
    assert len(manifest["plan_sha256"]) == 64


def test_runtime_stage_topology_keeps_bucket_stage_compatible() -> None:
    topology = _spec(signature_tasks=64, workers=8).stage_topology()

    assert topology["signatures"] == {"tasks": 64, "workers": 8}
    assert topology["buckets"] == {"tasks": 14, "workers": 8}
    assert topology["cluster"] == {"tasks": 1, "workers": 1}
    assert topology["filter"] == {"tasks": 64, "workers": 8}


def test_runtime_plan_rejects_unsafe_or_unvalidated_variants() -> None:
    with pytest.raises(DataTroveMinhashError, match="distinct"):
        _spec(output_uri="file:///tmp/input")
    with pytest.raises(DataTroveMinhashError, match="workers"):
        _spec(signature_tasks=2, workers=3)
    with pytest.raises(DataTroveMinhashError, match="hash_precision"):
        _spec(hash_precision=32)
    with pytest.raises(DataTroveMinhashError, match="version"):
        _spec(datatrove_version="0.9.0")
    with pytest.raises(DataTroveMinhashError, match="language"):
        _spec(language="")


def test_runtime_manifest_is_deterministic_and_parameter_sensitive() -> None:
    first = _spec().manifest()
    second = _spec().manifest()
    changed_ngram = _spec(n_grams=6).manifest()
    changed_language = _spec(language="en").manifest()

    assert first == second
    assert first["plan_sha256"] != changed_ngram["plan_sha256"]
    assert first["plan_sha256"] != changed_language["plan_sha256"]
