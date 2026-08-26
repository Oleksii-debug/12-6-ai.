"""Fail-closed readiness assessment for the learned ~20M Base campaign."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

MODEL341_SHA = "e4ff486fd90802fc123bebf60eed4e59196a98df"
MODELSPEC_SHA256 = (
    "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
)
PARAMETER_COUNT = 20_613_440

PILOT_AUTHORITIES = (
    "launch_binding",
    "corpus",
    "unique_loss_ledger",
    "tokenizer",
    "checkpoint",
    "evaluation_firewall",
    "training_recipe",
)
LONG_TRAIN_AUTHORITIES = PILOT_AUTHORITIES + ("bounded_pilot", "compute_authorization")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _hex_text(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(char in "0123456789abcdef" for char in value)


def validate_contract(data: Mapping[str, Any]) -> list[str]:
    """Validate the frozen launch-gate contract itself."""
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("gate_id") != "TRAIN-20M-LAUNCH-GATE-V1":
        errors.append("gate_id mismatch")
    if data.get("status") != "BLOCKED_WAIT_TERMINAL_AUTHORITIES":
        errors.append("contract must remain blocked until evidence is supplied")

    authority = data.get("authority")
    if not isinstance(authority, Mapping):
        errors.append("authority must be an object")
    else:
        if authority.get("model341_sha") != MODEL341_SHA:
            errors.append("authority.model341_sha mismatch")
        if authority.get("modelspec_sha256") != MODELSPEC_SHA256:
            errors.append("authority.modelspec_sha256 mismatch")
        if authority.get("parameter_count") != PARAMETER_COUNT:
            errors.append("authority.parameter_count mismatch")
        if authority.get("canonical_base") != "random_init":
            errors.append("authority.canonical_base must be random_init")

    boundaries = data.get("hard_boundaries")
    if not isinstance(boundaries, Mapping):
        errors.append("hard_boundaries must be an object")
    else:
        for key in (
            "long_training_authorized",
            "paid_compute_authorized",
            "stage_promotion_authorized",
            "final_test_payload_read_authorized",
        ):
            if boundaries.get(key) is not False:
                errors.append(f"hard_boundaries.{key} must be false")
        if boundaries.get("base_lineage") != "PRETRAINING_ONLY":
            errors.append("hard_boundaries.base_lineage drift")

    required = data.get("required_authorities")
    if not isinstance(required, list):
        errors.append("required_authorities must be an array")
    else:
        names = [item.get("name") for item in required if isinstance(item, Mapping)]
        if set(names) != set(LONG_TRAIN_AUTHORITIES):
            errors.append("required authority set mismatch")
        if len(names) != len(set(names)):
            errors.append("required authority names must be unique")
        for item in required:
            if not isinstance(item, Mapping):
                errors.append("required authority entry must be an object")
                continue
            if item.get("terminal") is not False:
                errors.append(f"{item.get('name')}.terminal must start false")
            if item.get("identity") is not None:
                errors.append(f"{item.get('name')}.identity must start null")

    decision = data.get("default_decision")
    if not isinstance(decision, Mapping):
        errors.append("default_decision must be an object")
    else:
        if decision.get("pilot_ready") is not False:
            errors.append("default pilot_ready must be false")
        if decision.get("long_training_ready") is not False:
            errors.append("default long_training_ready must be false")
        if decision.get("max_optimizer_updates") != 0:
            errors.append("default max_optimizer_updates must be 0")

    return errors


def _terminal_identity(evidence: Mapping[str, Any], name: str) -> bool:
    item = evidence.get(name)
    return (
        isinstance(item, Mapping)
        and item.get("terminal") is True
        and _nonempty_text(item.get("identity"))
    )


def _validate_recipe(item: Mapping[str, Any], blockers: list[str]) -> None:
    if item.get("optimizer_family") != "AdamW":
        blockers.append("training_recipe.optimizer_family_not_qualified")
    if not _positive_number(item.get("learning_rate")):
        blockers.append("training_recipe.learning_rate_invalid")
    betas = item.get("betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or not all(_positive_number(value) for value in betas)
        or not all(float(value) < 1.0 for value in betas)
    ):
        blockers.append("training_recipe.betas_invalid")
    weight_decay = item.get("weight_decay")
    if (
        not isinstance(weight_decay, (int, float))
        or isinstance(weight_decay, bool)
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) < 0.0
    ):
        blockers.append("training_recipe.weight_decay_invalid")
    if not _nonempty_text(item.get("scheduler")):
        blockers.append("training_recipe.scheduler_missing")
    if not _nonnegative_int(item.get("warmup_steps")):
        blockers.append("training_recipe.warmup_steps_invalid")
    if item.get("precision") not in {"fp32", "bf16", "fp16"}:
        blockers.append("training_recipe.precision_invalid")
    if not _positive_number(item.get("gradient_clip_norm")):
        blockers.append("training_recipe.gradient_clip_norm_invalid")
    if not _nonnegative_int(item.get("seed")):
        blockers.append("training_recipe.seed_invalid")
    for key in (
        "requested_unique_loss_positions",
        "requested_total_training_exposures",
        "max_exposures_per_unique_position",
        "max_optimizer_updates",
        "checkpoint_interval_steps",
    ):
        if not _positive_int(item.get(key)):
            blockers.append(f"training_recipe.{key}_invalid")
    for key in (
        "stopping_rule",
        "restart_policy",
        "selection_validation_schedule",
        "resource_plan_identity",
    ):
        if not _nonempty_text(item.get(key)):
            blockers.append(f"training_recipe.{key}_missing")
    if not _positive_number(item.get("estimated_training_flops")):
        blockers.append("training_recipe.estimated_training_flops_invalid")
    if not _positive_number(item.get("estimated_wall_clock_seconds")):
        blockers.append("training_recipe.estimated_wall_clock_seconds_invalid")


def _validate_launch_binding(evidence: Mapping[str, Any], blockers: list[str]) -> None:
    binding = evidence.get("launch_binding")
    if not isinstance(binding, Mapping):
        return
    if not _hex_text(binding.get("code_sha"), 40):
        blockers.append("launch_binding.code_sha_invalid")
    if binding.get("modelspec_sha256") != MODELSPEC_SHA256:
        blockers.append("launch_binding.modelspec_sha256_mismatch")
    if not _hex_text(binding.get("config_sha256"), 64):
        blockers.append("launch_binding.config_sha256_invalid")

    links = {
        "corpus_identity": ("corpus", "identity"),
        "loss_ledger_identity": ("unique_loss_ledger", "identity"),
        "tokenizer_identity": ("tokenizer", "identity"),
        "checkpoint_identity": ("checkpoint", "identity"),
        "evaluation_firewall_identity": ("evaluation_firewall", "identity"),
        "training_recipe_identity": ("training_recipe", "identity"),
    }
    for binding_key, (authority_name, authority_key) in links.items():
        expected = evidence.get(authority_name)
        expected_identity = expected.get(authority_key) if isinstance(expected, Mapping) else None
        if not _nonempty_text(binding.get(binding_key)):
            blockers.append(f"launch_binding.{binding_key}_missing")
        elif binding.get(binding_key) != expected_identity:
            blockers.append(f"launch_binding.{binding_key}_mismatch")


def assess_launch(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    material_cost: bool,
) -> dict[str, Any]:
    """Assess bounded-pilot and long-training readiness from terminal evidence."""
    contract_errors = validate_contract(contract)
    blockers = [f"contract:{error}" for error in contract_errors]

    for name in PILOT_AUTHORITIES:
        if not _terminal_identity(evidence, name):
            blockers.append(f"{name}_not_terminal")

    ledger = evidence.get("unique_loss_ledger")
    if isinstance(ledger, Mapping):
        if not _positive_int(ledger.get("authorized_unique_loss_positions")):
            blockers.append("unique_loss_ledger.zero_or_invalid_authorized_positions")
        if ledger.get("replay_used") is not False:
            blockers.append("unique_loss_ledger.replay_must_be_false")

    corpus = evidence.get("corpus")
    if isinstance(corpus, Mapping):
        for key in ("corpus_identity", "split_identity", "packing_identity"):
            if not _nonempty_text(corpus.get(key)):
                blockers.append(f"corpus.{key}_missing")
        if corpus.get("two_clean_builds_identical") is not True:
            blockers.append("corpus.two_clean_builds_not_proven")

    tokenizer = evidence.get("tokenizer")
    if isinstance(tokenizer, Mapping):
        if tokenizer.get("mode") not in {"trained_tokenizer", "byte_baseline"}:
            blockers.append("tokenizer.mode_invalid")
        if tokenizer.get("roundtrip_passed") is not True:
            blockers.append("tokenizer.roundtrip_not_proven")

    checkpoint = evidence.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        if checkpoint.get("corruption_matrix_passed") is not True:
            blockers.append("checkpoint.corruption_matrix_not_terminal")
        if checkpoint.get("fresh_resume_equivalence") is not True:
            blockers.append("checkpoint.fresh_resume_equivalence_missing")

    firewall = evidence.get("evaluation_firewall")
    if isinstance(firewall, Mapping):
        for key in (
            "selection_validation_identity",
            "final_test_identity",
            "decontamination_identity",
        ):
            if not _nonempty_text(firewall.get(key)):
                blockers.append(f"evaluation_firewall.{key}_missing")
        if firewall.get("training_overlap_count") != 0:
            blockers.append("evaluation_firewall.training_overlap_nonzero")
        if firewall.get("tokenizer_fit_overlap_count") != 0:
            blockers.append("evaluation_firewall.tokenizer_fit_overlap_nonzero")
        if firewall.get("final_test_access_before_training") is not False:
            blockers.append("evaluation_firewall.final_test_pretraining_access_forbidden")
        if firewall.get("final_test_access_before_terminal_selection") is not False:
            blockers.append("evaluation_firewall.final_test_preselection_access_forbidden")

    recipe = evidence.get("training_recipe")
    if isinstance(recipe, Mapping):
        _validate_recipe(recipe, blockers)

    if isinstance(ledger, Mapping) and isinstance(recipe, Mapping):
        available = ledger.get("authorized_unique_loss_positions")
        requested_unique = recipe.get("requested_unique_loss_positions")
        requested_total = recipe.get("requested_total_training_exposures")
        max_exposures = recipe.get("max_exposures_per_unique_position")
        if _positive_int(available) and _positive_int(requested_unique):
            if requested_unique > available:
                blockers.append("training_recipe.requests_more_unique_loss_than_authorized")
        if _positive_int(requested_unique) and _positive_int(requested_total):
            if requested_total < requested_unique:
                blockers.append("training_recipe.total_exposure_below_unique_requirement")
        if (
            _positive_int(available)
            and _positive_int(requested_total)
            and _positive_int(max_exposures)
            and requested_total > available * max_exposures
        ):
            blockers.append("training_recipe.total_exposure_exceeds_replay_cap")

    _validate_launch_binding(evidence, blockers)

    pilot_blockers = sorted(set(blockers))
    pilot_ready = not pilot_blockers

    long_blockers = list(pilot_blockers)
    pilot = evidence.get("bounded_pilot")
    if not _terminal_identity(evidence, "bounded_pilot"):
        long_blockers.append("bounded_pilot_not_terminal")
    elif isinstance(pilot, Mapping):
        for key in (
            "finite_loss",
            "loss_decreased",
            "gradient_health_passed",
            "checkpoint_resume_passed",
            "evaluation_isolation_passed",
            "throughput_measured",
            "peak_memory_within_plan",
        ):
            if pilot.get(key) is not True:
                long_blockers.append(f"bounded_pilot.{key}_not_proven")

    authorization = evidence.get("compute_authorization")
    if material_cost:
        if not _terminal_identity(evidence, "compute_authorization"):
            long_blockers.append("compute_authorization_not_terminal")
        elif isinstance(authorization, Mapping):
            if authorization.get("compute_authorized") is not True:
                long_blockers.append("compute_authorization.explicit_authorization_missing")
            if not _positive_number(authorization.get("max_budget_usd")):
                long_blockers.append("compute_authorization.max_budget_usd_invalid")
    elif isinstance(authorization, Mapping) and authorization.get("compute_authorized") is True:
        if not _terminal_identity(evidence, "compute_authorization"):
            long_blockers.append("compute_authorization_unbound_true")

    long_blockers = sorted(set(long_blockers))
    long_training_ready = not long_blockers

    return {
        "pilot_ready": pilot_ready,
        "pilot_blockers": pilot_blockers,
        "long_training_ready": long_training_ready,
        "long_training_blockers": long_blockers,
        "material_cost": material_cost,
    }
