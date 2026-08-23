import json
from pathlib import Path

import pytest

from twelve_six.posttraining.compatibility import (
    CURRENT_RUNTIME_COMPATIBILITY,
    SemanticVersion,
)
from twelve_six.posttraining.contracts import CheckpointRef, LineageKind
from twelve_six.posttraining.interfaces import RolloutRequest
from twelve_six.posttraining.rollout_planning import (
    DryRunRolloutPlan,
    RolloutPlanningError,
    RolloutTarget,
    SamplingPlan,
    build_dry_run_rollout_plan,
)

HEX_A = "a" * 64


def _checkpoint() -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id="s0-base-read-only",
        sha256=HEX_A,
        git_sha="f2e94c7",
        stage="S0",
        lineage=LineageKind.BASE,
    )


def _request(**generation: str) -> RolloutRequest:
    default_generation = {
        "max_new_tokens": "32",
        "seed": "7",
        "temperature": "0.7",
        "top_k": "20",
        "top_p": "0.95",
    }
    default_generation.update(generation)
    return RolloutRequest(
        request_id="dry-run-1",
        prompts=("first prompt", "second prompt"),
        generation=default_generation,
    )


def test_semantic_version_parse_and_order() -> None:
    assert SemanticVersion.parse("0.23.0") < SemanticVersion.parse("0.27.1")
    assert str(SemanticVersion.parse("1.10.0")) == "1.10.0"
    with pytest.raises(ValueError, match="invalid semantic version"):
        SemanticVersion.parse("v0.23")


def test_snapshot_selects_joint_version_instead_of_latest() -> None:
    snapshot = CURRENT_RUNTIME_COMPATIBILITY
    assert snapshot.snapshot_id.endswith("-r2")
    assert str(snapshot.trl_version) == "1.10.0"
    assert str(snapshot.verl_version) == "0.9.0"
    assert str(snapshot.vllm_latest_version) == "0.27.1"
    assert str(snapshot.vllm_selected_version) == "0.23.0"
    assert str(snapshot.trl_vllm_min) == "0.16.0"
    assert str(snapshot.trl_vllm_max) == "0.23.0"
    assert snapshot.selected_is_jointly_supported()
    assert not snapshot.latest_is_jointly_supported()


def test_checked_in_compatibility_snapshot_matches_code_contract() -> None:
    config_path = Path("configs/posttraining/runtime_compatibility_2026-08-23.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    snapshot = CURRENT_RUNTIME_COMPATIBILITY
    assert config["snapshot_id"] == snapshot.snapshot_id
    assert config["trl"]["observed_version"] == str(snapshot.trl_version)
    assert config["trl"]["vllm_supported_min"] == str(snapshot.trl_vllm_min)
    assert config["trl"]["vllm_supported_max"] == str(snapshot.trl_vllm_max)
    assert config["verl"]["observed_version"] == str(snapshot.verl_version)
    assert config["vllm"]["latest_observed_version"] == str(snapshot.vllm_latest_version)
    assert config["vllm"]["selected_joint_compatibility_version"] == str(
        snapshot.vllm_selected_version
    )
    assert config["execution"]["runtime_imported"] is False
    assert config["execution"]["training_authorized"] is False
    assert config["execution"]["weights_mutated"] is False


def test_sampling_plan_normalizes_d07_style_max_new_tokens() -> None:
    plan = SamplingPlan.from_mapping(_request().generation)
    assert plan.max_tokens == 32
    assert plan.as_vllm_sampling()["max_tokens"] == 32
    assert plan.as_vllm_sampling()["n"] == 1


def test_sampling_plan_rejects_unknown_or_ambiguous_keys() -> None:
    with pytest.raises(RolloutPlanningError, match="unsupported generation keys"):
        SamplingPlan.from_mapping({"max_tokens": "8", "surprise": "1"})
    with pytest.raises(RolloutPlanningError, match="only one"):
        SamplingPlan.from_mapping({"max_tokens": "8", "max_new_tokens": "8"})


def test_sampling_plan_preserves_range_validation_diagnostics() -> None:
    with pytest.raises(RolloutPlanningError, match="max_tokens must be > 0"):
        SamplingPlan.from_mapping({"max_tokens": "0"})
    with pytest.raises(RolloutPlanningError, match="top_p must be finite"):
        SamplingPlan.from_mapping({"max_tokens": "8", "top_p": "2.0"})
    with pytest.raises(RolloutPlanningError, match="generation values must be numeric"):
        SamplingPlan.from_mapping({"max_tokens": "eight"})


def test_trl_dry_run_rejects_versions_above_declared_range() -> None:
    for version in ("0.24.0", "0.26.0", "0.27.1"):
        with pytest.raises(RolloutPlanningError, match=f"does not support vLLM {version}"):
            build_dry_run_rollout_plan(
                _request(),
                _checkpoint(),
                RolloutTarget.TRL_VLLM_SERVER,
                vllm_version=SemanticVersion.parse(version),
            )


def test_trl_dry_run_uses_selected_joint_compatibility_version() -> None:
    plan = build_dry_run_rollout_plan(
        _request(),
        _checkpoint(),
        RolloutTarget.TRL_VLLM_SERVER,
    )
    payload = plan.normalized_payload()
    assert payload["execution_enabled"] is False
    assert payload["runtime"] == {"trl": "1.10.0", "vllm": "0.23.0"}
    assert payload["checkpoint"]["lineage"] == "base"
    assert payload["sampling"]["max_tokens"] == 32


def test_verl_dry_run_records_versioned_rollout_contract() -> None:
    plan = build_dry_run_rollout_plan(
        _request(num_generations="2"),
        _checkpoint(),
        RolloutTarget.VERL_VLLM,
    )
    payload = plan.normalized_payload()
    assert payload["runtime"] == {"verl": "0.9.0", "vllm": "0.23.0"}
    assert payload["sampling"]["n"] == 2


def test_plan_hash_is_deterministic_and_seed_sensitive() -> None:
    first = build_dry_run_rollout_plan(
        _request(seed="7"),
        _checkpoint(),
        RolloutTarget.VLLM_OFFLINE,
    )
    repeated = build_dry_run_rollout_plan(
        _request(seed="7"),
        _checkpoint(),
        RolloutTarget.VLLM_OFFLINE,
    )
    changed = build_dry_run_rollout_plan(
        _request(seed="8"),
        _checkpoint(),
        RolloutTarget.VLLM_OFFLINE,
    )
    assert first.plan_sha256 == repeated.plan_sha256
    assert first.plan_sha256 != changed.plan_sha256


def test_dry_run_plan_cannot_be_converted_into_execution_declaration() -> None:
    with pytest.raises(RolloutPlanningError, match="cannot enable execution"):
        DryRunRolloutPlan(
            target=RolloutTarget.VLLM_OFFLINE,
            request=_request(),
            checkpoint=_checkpoint(),
            sampling=SamplingPlan.from_mapping(_request().generation),
            vllm_version=SemanticVersion.parse("0.23.0"),
            compatibility_snapshot_id="test-snapshot",
            execution_enabled=True,
        )
