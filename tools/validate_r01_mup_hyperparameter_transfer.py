#!/usr/bin/env python3
"""Validate the fail-closed R01 maximal-update transfer research contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_PARENT = {
    "path": "configs/research/r01_20m_to_100m_scaling_campaign_v1.json",
    "merged_main_sha": "a73ab38026cb7849f478cc13ad58b93534a76e2f",
    "model341_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
    "modelspec_sha256": (
        "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
    ),
    "parameter_count": 20613440,
}

EXPECTED_WIDTH_PROBES = [
    {"d_model": 192, "n_heads": 6},
    {"d_model": 256, "n_heads": 8},
    {"d_model": 320, "n_heads": 10},
]

REQUIRED_METRICS = {
    "heldout_bits_per_byte",
    "heldout_loss",
    "loss_curve",
    "activation_rms_by_layer",
    "update_to_weight_ratio_by_layer",
    "gradient_norms",
    "nan_inf_events",
    "tokens_per_second",
    "peak_memory_bytes",
    "unique_loss_positions_seen",
    "total_loss_position_exposures",
    "checkpoint_resume_equivalence",
}

REQUIRED_SOURCE_URLS = {
    "https://arxiv.org/abs/2203.03466",
    "https://arxiv.org/abs/2305.16264",
    "https://proceedings.mlr.press/v235/liu24ce.html",
    "https://arxiv.org/abs/2203.15556",
}


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_campaign(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    _expect(errors, data.get("schema_version") == 1, "schema_version must be 1")
    _expect(
        errors,
        data.get("campaign_id") == "R01-MUP-HYPERPARAMETER-TRANSFER-V1",
        "campaign_id mismatch",
    )
    _expect(
        errors,
        data.get("status") == "RESEARCH_CANDIDATE_ONLY",
        "campaign must remain research-only",
    )

    parent = data.get("parent_scaling_campaign")
    _expect(errors, isinstance(parent, dict), "parent_scaling_campaign must be an object")
    if isinstance(parent, dict):
        for key, value in EXPECTED_PARENT.items():
            _expect(errors, parent.get(key) == value, f"parent_scaling_campaign.{key} mismatch")

    boundaries = data.get("hard_boundaries")
    _expect(errors, isinstance(boundaries, dict), "hard_boundaries must be an object")
    if isinstance(boundaries, dict):
        _expect(
            errors,
            boundaries.get("local_free_engineering_only") is True,
            "LOCAL_FREE boundary must remain true",
        )
        for key in (
            "model_training_authorized",
            "paid_compute_authorized",
            "production_modelspec_change_authorized",
            "optimizer_recipe_change_authorized",
            "tokenizer_change_authorized",
            "corpus_change_authorized",
            "standard_parameterization_transfer_may_be_assumed",
            "mup_adoption_may_be_assumed",
        ):
            _expect(errors, boundaries.get(key) is False, f"hard_boundaries.{key} must be false")

    arms = data.get("comparison_arms")
    _expect(errors, isinstance(arms, list), "comparison_arms must be an array")
    if isinstance(arms, list):
        ids = {item.get("id") for item in arms if isinstance(item, dict)}
        _expect(errors, ids == {"SP-CONTROL", "MUP-CANDIDATE"}, "comparison arm ids drift")

    integration = data.get("mup_integration_contract")
    _expect(errors, isinstance(integration, dict), "mup_integration_contract must be an object")
    if isinstance(integration, dict):
        _expect(
            errors,
            integration.get("implementation_choice") == "NOT_FROZEN",
            "muP implementation must not be frozen by research contract",
        )
        for key in (
            "width_transfer_primary",
            "depth_transfer_is_separate_experiment",
            "width_and_depth_may_not_be_conflated_in_one_transfer_claim",
            "tied_embedding_handling_must_be_mup_compatible",
            "attention_logit_scaling_must_be_explicitly_verified",
            "optimizer_parameter_group_scaling_must_be_explicitly_verified",
            "initialization_scaling_must_be_explicitly_verified",
            "scheduler_must_preserve_relative_parameter_group_scaling",
            "ordinary_adamw_hyperparameters_may_not_be_relabelled_mup",
            "checkpoint_identity_must_record_parameterization",
        ):
            _expect(errors, integration.get(key) is True, f"mup_integration_contract.{key} drift")

    coord = data.get("coordinate_check_preregistration")
    _expect(errors, isinstance(coord, dict), "coordinate_check_preregistration must be an object")
    if isinstance(coord, dict):
        _expect(errors, coord.get("production_model") is False, "coord probes are not production models")
        _expect(errors, coord.get("width_probes") == EXPECTED_WIDTH_PROBES, "width probe ladder drift")
        _expect(
            errors,
            coord.get("pass_claim_requires_independent_audit") is True,
            "coordinate PASS must require independent audit",
        )
        shared = coord.get("shared")
        _expect(errors, isinstance(shared, dict), "coordinate shared contract must be an object")
        if isinstance(shared, dict):
            _expect(errors, shared.get("n_layers") == 16, "coordinate depth drift")
            _expect(errors, shared.get("head_dim") == 32, "coordinate head_dim drift")
            _expect(errors, shared.get("n_kv_heads") == 2, "coordinate KV-head drift")
            _expect(errors, shared.get("tie_word_embeddings") is True, "tied embedding drift")

    transfer = data.get("hyperparameter_transfer_preregistration")
    _expect(
        errors,
        isinstance(transfer, dict),
        "hyperparameter_transfer_preregistration must be an object",
    )
    if isinstance(transfer, dict):
        lr_search = transfer.get("learning_rate_search")
        _expect(errors, isinstance(lr_search, dict), "learning_rate_search must be an object")
        if isinstance(lr_search, dict):
            _expect(errors, lr_search.get("multipliers") == [0.5, 1.0, 2.0], "LR grid drift")
        _expect(
            errors,
            transfer.get("target_sizes_for_future_measurement")
            == [20613440, 50000000, 100000000],
            "target-size ladder drift",
        )
        _expect(
            errors,
            transfer.get("zero_shot_transfer_is_evidence_claim_not_default") is True,
            "zero-shot transfer must remain an evidence claim",
        )

    repetition = data.get("data_constrained_pilot")
    _expect(errors, isinstance(repetition, dict), "data_constrained_pilot must be an object")
    if isinstance(repetition, dict):
        _expect(
            errors,
            repetition.get("candidate_epoch_counts") == [1, 2, 4],
            "repetition pilot ladder drift",
        )
        for key in (
            "unique_loss_positions_must_be_reported_separately",
            "repeated_exposure_positions_must_be_reported_separately",
            "same_split_and_decontamination_identity_required",
            "no_repetition_policy_change_authorized_by_this_contract",
        ):
            _expect(errors, repetition.get(key) is True, f"data_constrained_pilot.{key} drift")
        _expect(
            errors,
            repetition.get("repeated_positions_may_increase_unique_capacity") is False,
            "repeated exposure may not inflate unique capacity",
        )

    metrics = data.get("metrics")
    _expect(errors, isinstance(metrics, dict), "metrics must be an object")
    if isinstance(metrics, dict):
        _expect(
            errors,
            metrics.get("primary_cross_tokenizer_quality") == "bits_per_byte",
            "cross-tokenizer quality metric must remain BPB",
        )
        required = metrics.get("required")
        _expect(errors, isinstance(required, list), "metrics.required must be an array")
        if isinstance(required, list):
            _expect(errors, REQUIRED_METRICS.issubset(set(required)), "required metric set incomplete")

    rules = data.get("decision_rules")
    _expect(errors, isinstance(rules, dict), "decision_rules must be an object")
    if isinstance(rules, dict):
        for key, value in rules.items():
            _expect(errors, value is False, f"decision_rules.{key} must remain false")

    sources = data.get("research_sources")
    _expect(errors, isinstance(sources, list), "research_sources must be an array")
    if isinstance(sources, list):
        urls = {item.get("url") for item in sources if isinstance(item, dict)}
        _expect(errors, REQUIRED_SOURCE_URLS.issubset(urls), "required research sources incomplete")

    return errors


def validate_path(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ["campaign root must be an object"]
    return validate_campaign(data)


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(
        "configs/research/r01_mup_hyperparameter_transfer_v1.json"
    )
    errors = validate_path(path)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: R01 muP hyperparameter-transfer research contract is fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
