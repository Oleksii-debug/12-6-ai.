from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.scaling-data-budget-policy.v1"
REFERENCE_MULTIPLIERS = (20, 50, 100)
REQUIRED_EVIDENCE = {
    "immutable_corpus_identity",
    "immutable_train_split_identity",
    "tokenizer_identity",
    "exact_post_tokenization_unique_train_token_count",
    "exact_unique_loss_position_count",
    "document_boundary_loss_mask_identity",
    "evaluation_decontamination_authority",
    "deduplication_authority",
    "quality_and_privacy_authority",
    "checkpoint_resume_authority",
    "preregistered_total_training_token_exposure_budget",
    "preregistered_replay_policy_and_epoch_cap",
    "preregistered_stop_rule",
    "explicit_compute_authorization",
}
FORBIDDEN_SUBSTITUTIONS = {
    "source_bytes_for_training_tokens",
    "normalized_bytes_for_training_tokens",
    "source_capacity_target_for_training_budget",
    "total_training_token_exposures_for_unique_loss_positions",
    "raw_record_count_for_unique_loss_positions",
    "replayed_tokens_for_unique_tokens",
    "queued_ci_for_terminal_success",
    "stale_source_snapshot_for_live_data_authority",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_stage(stage: dict[str, Any]) -> None:
    parameter_count = stage.get("parameter_count")
    _require(
        isinstance(parameter_count, int) and not isinstance(parameter_count, bool),
        "parameter_count must be an integer",
    )
    _require(parameter_count > 0, "parameter_count must be positive")

    references = stage.get("reference_training_token_exposures")
    _require(
        isinstance(references, dict),
        "reference_training_token_exposures must be a mapping",
    )
    for multiplier in REFERENCE_MULTIPLIERS:
        key = f"{multiplier}x"
        _require(
            references.get(key) == parameter_count * multiplier,
            f"{stage.get('stage')} {key} token-exposure reference drift",
        )

    expected_flops = 6 * parameter_count * (parameter_count * 20)
    _require(
        stage.get("approx_dense_training_flops_at_20x") == expected_flops,
        f"{stage.get('stage')} approximate 20x FLOPs drift",
    )
    _require(
        stage.get("status_source") == "external_stage_readiness_authority",
        f"{stage.get('stage')} must not embed volatile readiness status",
    )


def validate(policy: dict[str, Any]) -> dict[str, Any]:
    _require(policy.get("schema_version") == EXPECTED_SCHEMA, "unexpected schema_version")
    _require(
        "current_20m_observation" not in policy,
        "volatile live source snapshots must not be embedded in scaling policy",
    )

    boundary = policy.get("truth_boundary")
    _require(isinstance(boundary, dict), "truth_boundary must be a mapping")
    _require(
        boundary.get("source_capacity_bytes_are_training_tokens") is False,
        "bytes must never be token authority",
    )
    _require(
        boundary.get("source_capacity_bytes_are_unique_loss_positions") is False,
        "bytes must never be loss-position authority",
    )
    _require(
        boundary.get("source_capacity_target_is_training_budget") is False,
        "source-capacity targets must never be training budgets",
    )
    _require(
        boundary.get("training_token_exposures_are_unique_loss_positions") is False,
        "token exposures must not be treated as unique loss positions",
    )
    _require(
        boundary.get("planning_reference_is_minimum_training_requirement") is False,
        "planning reference cannot become a hard minimum",
    )
    _require(
        boundary.get("planning_reference_is_quality_guarantee") is False,
        "planning references cannot guarantee quality",
    )
    _require(
        boundary.get("planning_reference_is_compute_authorization") is False,
        "planning references cannot authorize compute",
    )
    _require(
        boundary.get("this_policy_can_authorize_training") is False,
        "science policy must not self-authorize training",
    )
    _require(
        boundary.get("paid_compute_authorized") is False,
        "this policy must not authorize paid compute",
    )

    reference_policy = policy.get("reference_policy")
    _require(isinstance(reference_policy, dict), "reference_policy must be a mapping")
    _require(
        reference_policy.get("baseline_training_token_exposures_per_parameter") == 20,
        "baseline planning reference must remain 20x",
    )
    _require(
        reference_policy.get("exploration_training_token_exposures_per_parameter")
        == [20, 50, 100],
        "exploration ladder drift",
    )

    stages = policy.get("stages")
    _require(
        isinstance(stages, list) and len(stages) == 3,
        "expected exactly three planning stages",
    )
    expected_stage_names = ["20M_PRIMARY", "100M_TARGET", "1B_TARGET"]
    _require(
        [stage.get("stage") for stage in stages] == expected_stage_names,
        "stage ordering or identity drift",
    )
    for stage in stages:
        _require(isinstance(stage, dict), "each stage must be a mapping")
        _validate_stage(stage)

    data_contract = policy.get("data_authority_contract")
    _require(isinstance(data_contract, dict), "data_authority_contract must be a mapping")
    required_true = {
        "live_source_registry_owned_by_data_lane",
        "source_capacity_targets_are_acquisition_planning_only",
        "promotion_requires_terminal_exact_head_authority",
        "promotion_requires_immutable_materialized_corpus",
        "promotion_requires_exact_post_tokenization_accounting",
        "promotion_requires_unique_vs_replayed_exposure_accounting",
    }
    for key in required_true:
        _require(data_contract.get(key) is True, f"data authority invariant missing: {key}")
    required_false = {
        "volatile_source_capacity_embedded_here",
        "source_registry_is_corpus_identity",
        "source_registry_is_token_count_authority",
        "source_registry_is_loss_position_authority",
    }
    for key in required_false:
        _require(data_contract.get(key) is False, f"data authority firewall missing: {key}")

    gate = policy.get("long_training_gate")
    _require(isinstance(gate, dict), "long_training_gate must be a mapping")
    _require(
        set(gate.get("required_evidence", [])) == REQUIRED_EVIDENCE,
        "required evidence set drift",
    )
    _require(
        set(gate.get("forbidden_substitutions", [])) == FORBIDDEN_SUBSTITUTIONS,
        "forbidden substitution set drift",
    )
    _require(
        gate.get("below_20x_requires_explicit_scaling_rationale") is True,
        "below-reference scaling rationale gate missing",
    )
    _require(
        gate.get("above_20x_requires_explicit_scaling_rationale") is True,
        "above-reference scaling rationale gate missing",
    )

    references = policy.get("research_basis")
    _require(
        isinstance(references, list) and len(references) >= 5,
        "research basis is incomplete",
    )
    _require(
        len({item.get("id") for item in references}) == len(references),
        "research basis IDs must be unique",
    )

    return {
        "status": "PASS",
        "stage_count": len(stages),
        "primary_parameter_count": stages[0]["parameter_count"],
        "primary_20x_training_token_exposures": stages[0][
            "reference_training_token_exposures"
        ]["20x"],
        "volatile_source_snapshot_embedded": False,
        "source_bytes_are_token_authority": False,
        "token_exposures_are_unique_positions": False,
        "policy_can_authorize_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "policy",
        nargs="?",
        default="configs/scaling/data_budget_policy_v1.json",
        type=Path,
    )
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    print(json.dumps(validate(policy), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
