from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.training.scale_1b_readiness import (
    SCALE_1B_TARGET_PARAMETERS,
    Scale1BDependencies,
    assess_scale_1b_readiness,
    meta_parameter_probe,
    validate_scale_1b_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs" / "stages" / "s6_1b.scale06_current_tokenizer.candidate.json"
EXPECTED_PARAMETERS = 999_761_920
EXPECTED_MODEL_IDENTITY = "f691e75eea9ca4c4edae197b5284c2564d3784a87fd3a831c6411af55dfc00be"


def test_scale_1b_candidate_exact_identity_and_geometry() -> None:
    config = validate_scale_1b_candidate(CANDIDATE)
    assert config.target_parameters == SCALE_1B_TARGET_PARAMETERS
    assert config.expected_parameters == EXPECTED_PARAMETERS
    assert config.model.parameter_count() == EXPECTED_PARAMETERS
    assert config.model.identity_sha256() == EXPECTED_MODEL_IDENTITY
    assert config.model.vocab_size == 256
    assert config.model.max_seq_len == 4096
    assert config.model.n_heads == 32
    assert config.model.n_kv_heads == 8


def test_scale_1b_default_assessment_is_fail_closed() -> None:
    report = assess_scale_1b_readiness(CANDIDATE)
    assert report.exact_parameters == EXPECTED_PARAMETERS
    assert report.attention_variant == "gqa"
    assert report.requires_native_gqa is True
    assert report.ready_for_authorization_request is False
    assert report.ready_for_material_compute is False
    assert report.engineering_blockers == (
        "preceding_stage_not_admitted",
        "production_tokenizer_not_qualified",
        "native_gqa_not_qualified",
        "distributed_checkpoint_not_qualified",
        "data_pipeline_not_qualified",
        "accelerator_runtime_not_qualified",
    )
    assert report.authorization_blockers == ("material_compute_not_authorized",)


def test_scale_1b_positive_evidence_must_be_explicit() -> None:
    engineering_only = Scale1BDependencies(
        preceding_stage_admitted=True,
        production_tokenizer_qualified=True,
        native_gqa_qualified=True,
        distributed_checkpoint_qualified=True,
        data_pipeline_qualified=True,
        accelerator_runtime_qualified=True,
        compute_authorized=False,
    )
    report = assess_scale_1b_readiness(CANDIDATE, engineering_only)
    assert report.engineering_blockers == ()
    assert report.ready_for_authorization_request is True
    assert report.ready_for_material_compute is False

    fully_authorized = Scale1BDependencies(
        preceding_stage_admitted=True,
        production_tokenizer_qualified=True,
        native_gqa_qualified=True,
        distributed_checkpoint_qualified=True,
        data_pipeline_qualified=True,
        accelerator_runtime_qualified=True,
        compute_authorized=True,
    )
    report = assess_scale_1b_readiness(CANDIDATE, fully_authorized)
    assert report.engineering_blockers == ()
    assert report.authorization_blockers == ()
    assert report.ready_for_material_compute is True


def test_scale_1b_topology_estimates_shard_persistent_state() -> None:
    report = assess_scale_1b_readiness(CANDIDATE, world_sizes=(1, 4, 8))
    single = report.topology_resource_estimates["1"]
    four = report.topology_resource_estimates["4"]
    eight = report.topology_resource_estimates["8"]

    assert single["fsdp2_sharded"] is False
    assert four["fsdp2_sharded"] is True
    assert eight["fsdp2_sharded"] is True
    assert single["persistent_total_bytes_per_rank"] > four["persistent_total_bytes_per_rank"]
    assert four["persistent_total_bytes_per_rank"] > eight["persistent_total_bytes_per_rank"]
    assert single["full_training_checkpoint_bytes"] == four["full_training_checkpoint_bytes"]
    assert four["full_training_checkpoint_bytes"] == eight["full_training_checkpoint_bytes"]


def test_scale_1b_meta_probe_does_not_materialize_weights() -> None:
    assert meta_parameter_probe(CANDIDATE) == EXPECTED_PARAMETERS


def test_scale_1b_rejects_promotion_open_candidate(tmp_path: Path) -> None:
    payload = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    payload["promotion_allowed"] = True
    path = tmp_path / "unsafe_s6.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fail closed on promotion"):
        validate_scale_1b_candidate(path)


def test_scale_1b_dependency_fields_require_real_booleans() -> None:
    with pytest.raises(TypeError, match="compute_authorized must be bool"):
        Scale1BDependencies(compute_authorized=1)  # type: ignore[arg-type]
