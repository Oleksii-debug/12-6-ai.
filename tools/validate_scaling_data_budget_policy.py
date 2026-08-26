from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.scaling-data-budget-policy.v1"
REFERENCE_MULTIPLIERS = (20, 50, 100)
CURRENT_REGISTRY_PR = 538
CURRENT_REGISTRY_HEAD = "958ebec0f7c9cb00238c7df70566cefd6b504d92"
CURRENT_REGISTRY_IDENTITY = "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526"
CURRENT_SOURCE_BYTES = 565_743
CURRENT_FAMILY_COUNT = 13
SOURCE_TARGET_BYTES = 20_000_000
REQUIRED_EVIDENCE = {
    "immutable_corpus_identity",
    "immutable_train_split_identity",
    "tokenizer_identity",
    "exact_post_tokenization_train_token_count",
    "exact_unique_loss_position_count",
    "document_boundary_loss_mask_identity",
    "evaluation_decontamination_authority",
    "deduplication_authority",
    "quality_and_privacy_authority",
    "checkpoint_resume_authority",
    "preregistered_training_budget_and_stop_rule",
}
FORBIDDEN_SUBSTITUTIONS = {
    "source_bytes_for_training_tokens",
    "normalized_bytes_for_training_tokens",
    "raw_record_count_for_unique_loss_positions",
    "replayed_tokens_for_unique_tokens",
    "queued_ci_for_terminal_success",
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

    references = stage.get("reference_unique_loss_tokens")
    _require(isinstance(references, dict), "reference_unique_loss_tokens must be a mapping")
    for multiplier in REFERENCE_MULTIPLIERS:
        key = f"{multiplier}x"
        _require(
            references.get(key) == parameter_count * multiplier,
            f"{stage.get('stage')} {key} token reference drift",
        )

    expected_flops = 6 * parameter_count * (parameter_count * 20)
    _require(
        stage.get("approx_dense_training_flops_at_20x") == expected_flops,
        f"{stage.get('stage')} approximate 20x FLOPs drift",
    )


def validate(policy: dict[str, Any]) -> dict[str, Any]:
    _require(policy.get("schema_version") == EXPECTED_SCHEMA, "unexpected schema_version")

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
        boundary.get("planning_reference_is_quality_guarantee") is False,
        "planning references cannot guarantee quality",
    )
    _require(
        boundary.get("planning_reference_is_compute_authorization") is False,
        "planning references cannot authorize compute",
    )
    _require(
        boundary.get("paid_compute_authorized") is False,
        "this policy must not authorize paid compute",
    )

    reference_policy = policy.get("reference_policy")
    _require(isinstance(reference_policy, dict), "reference_policy must be a mapping")
    _require(
        reference_policy.get("baseline_unique_loss_tokens_per_parameter") == 20,
        "baseline planning reference must remain 20x",
    )
    _require(
        reference_policy.get("exploration_tokens_per_parameter") == [20, 50, 100],
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
        gate.get("below_20x_classification") == "BOUNDED_SCALING_OR_SMOKE_EXPERIMENT_ONLY",
        "below-reference classification drift",
    )
    _require(
        gate.get("full_stage_attempt_requires_at_least_reference_or_explicit_scaling_exception")
        is True,
        "full-stage fail-closed rule missing",
    )

    current = policy.get("current_20m_observation")
    _require(isinstance(current, dict), "current_20m_observation must be a mapping")
    _require(current.get("source_registry_pr") == CURRENT_REGISTRY_PR, "source registry PR drift")
    _require(
        current.get("source_registry_head_sha") == CURRENT_REGISTRY_HEAD,
        "source registry head drift",
    )
    _require(
        current.get("source_registry_identity") == CURRENT_REGISTRY_IDENTITY,
        "source registry identity drift",
    )
    _require(
        current.get("observed_source_capacity_bytes") == CURRENT_SOURCE_BYTES,
        "observed source-capacity authority drift",
    )
    _require(
        current.get("observed_independent_families") == CURRENT_FAMILY_COUNT,
        "observed source-family authority drift",
    )
    _require(
        current.get("frozen_source_capacity_target_bytes") == SOURCE_TARGET_BYTES,
        "source-capacity target drift",
    )
    _require(
        current.get("remaining_source_capacity_gap_bytes")
        == SOURCE_TARGET_BYTES - CURRENT_SOURCE_BYTES,
        "source-capacity gap arithmetic drift",
    )
    _require(
        current.get("authorized_balanced_no_replay_loss_positions") == 0,
        "source registry cannot fabricate authorized loss positions",
    )
    _require(
        current.get("bytes_to_tokens_conversion") == "FORBIDDEN_WITHOUT_EXACT_TOKENIZATION",
        "bytes-to-token firewall missing",
    )
    _require(
        current.get("learned_20m_long_training_ready") is False,
        "policy must not fabricate 20M training readiness",
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
        "primary_20x_unique_loss_tokens": stages[0]["reference_unique_loss_tokens"]["20x"],
        "source_registry_pr": CURRENT_REGISTRY_PR,
        "source_capacity_bytes_at_cutoff": CURRENT_SOURCE_BYTES,
        "source_bytes_are_token_authority": False,
        "long_training_ready": False,
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
