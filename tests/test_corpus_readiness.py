from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.data.readiness import (
    CorpusReadinessError,
    evaluate_corpus_readiness,
    evaluate_policy_file,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "data" / "20m_capability_readiness.json"
MANIFEST_PATH = ROOT / "data" / "corpus" / "v0.1" / "manifest.json"
MODEL_PATH = ROOT / "configs" / "candidates" / "model341_20m_candidate_a.json"


def _policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_current_20m_corpus_is_explicitly_blocked_for_capability_pretraining() -> None:
    report = evaluate_policy_file(POLICY_PATH, repo_root=ROOT)

    assert report["status"] == "BLOCKED"
    assert report["pass"] is False
    assert report["training_authorization_inferred"] is False
    assert report["training_performed"] is False
    assert report["parameter_count"] == 20_613_440

    floor = report["checks"]["data_parameter_floor"]
    assert floor["train_byte_tokens"] == 20_000_775
    assert floor["required_train_byte_tokens"] == 412_268_800
    assert floor["deficit_train_byte_tokens"] == 392_268_025
    assert floor["tokens_per_parameter"] < 1.0

    assert "data_parameter_floor" in report["blockers"]
    assert "external_training_data" in report["blockers"]
    assert "external_source_count" in report["blockers"]
    assert "external_source_diversity" in report["blockers"]


def test_representative_20x_byte_corpus_passes_metadata_readiness() -> None:
    policy = _policy()
    manifest = _manifest()
    parameter_count = 20_613_440
    required = parameter_count * 20

    manifest["by_split"]["train"]["byte_tokens"] = required
    manifest["external_training_eligible_sources"] = 3
    manifest["truth_boundary"]["contains_external_training_data"] = True
    manifest["truth_boundary"]["external_source_diversity_representative"] = True
    manifest["train_validation_content_overlap"] = 0

    report = evaluate_corpus_readiness(
        manifest, parameter_count=parameter_count, policy=policy
    )

    assert report["status"] == "READY"
    assert report["pass"] is True
    assert report["blockers"] == []
    assert report["training_authorization_inferred"] is False


def test_overlap_blocks_even_when_volume_and_external_requirements_pass() -> None:
    policy = _policy()
    manifest = _manifest()
    parameter_count = 20_613_440

    manifest["by_split"]["train"]["byte_tokens"] = parameter_count * 20
    manifest["external_training_eligible_sources"] = 2
    manifest["truth_boundary"]["contains_external_training_data"] = True
    manifest["truth_boundary"]["external_source_diversity_representative"] = True
    manifest["train_validation_content_overlap"] = 1

    report = evaluate_corpus_readiness(
        manifest, parameter_count=parameter_count, policy=policy
    )

    assert report["pass"] is False
    assert report["blockers"] == ["train_validation_isolation"]


def test_malformed_policy_and_manifest_fail_closed() -> None:
    policy = _policy()
    manifest = _manifest()

    policy["min_train_tokens_per_parameter"] = 0
    with pytest.raises(CorpusReadinessError, match="min_train_tokens_per_parameter"):
        evaluate_corpus_readiness(manifest, parameter_count=1, policy=policy)

    policy = _policy()
    manifest["schema_version"] = "unknown"
    with pytest.raises(CorpusReadinessError, match="manifest schema"):
        evaluate_corpus_readiness(manifest, parameter_count=1, policy=policy)


def test_policy_is_bound_to_canonical_20m_candidate() -> None:
    policy = _policy()
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    assert policy["model_candidate_path"] == "configs/candidates/model341_20m_candidate_a.json"
    assert policy["corpus_manifest_path"] == "data/corpus/v0.1/manifest.json"
    assert model["expected_parameters"] == 20_613_440
