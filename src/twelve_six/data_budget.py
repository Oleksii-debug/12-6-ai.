"""Runtime evaluation for the 12-6 scaling data-budget policy.

This module evaluates only exact post-tokenization unique causal-loss positions.
It cannot authorize training, paid compute, or substitute source-byte capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BudgetEvaluation:
    stage: str
    parameter_count: int
    multiplier: int
    unique_loss_positions: int
    required_unique_loss_positions: int
    observed_positions_per_parameter: float
    shortfall_positions: int
    progress_fraction: float
    data_budget_tier_met: bool
    approximate_dense_training_flops_at_tier: int
    training_authorized: bool = False
    paid_compute_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def required_unique_loss_positions(parameter_count: int, multiplier: int) -> int:
    """Return the exact unique-loss-position reference for a parameter multiplier."""

    parameters = _positive_int("parameter_count", parameter_count)
    ratio = _positive_int("multiplier", multiplier)
    return parameters * ratio


def _validate_truth_boundary(policy: dict[str, Any]) -> None:
    truth = policy.get("truth_boundary")
    if not isinstance(truth, dict):
        raise ValueError("missing truth_boundary")

    forbidden_true = (
        "source_capacity_bytes_are_training_tokens",
        "source_capacity_bytes_are_unique_loss_positions",
        "planning_reference_is_quality_guarantee",
        "planning_reference_is_compute_authorization",
        "paid_compute_authorized",
    )
    for field in forbidden_true:
        if truth.get(field) is not False:
            raise ValueError(f"truth boundary violation: {field} must be false")


def _allowed_multipliers(policy: dict[str, Any]) -> tuple[int, ...]:
    reference = policy.get("reference_policy")
    if not isinstance(reference, dict):
        raise ValueError("missing reference_policy")
    raw = reference.get("exploration_tokens_per_parameter")
    if not isinstance(raw, list) or not raw:
        raise ValueError("missing exploration_tokens_per_parameter")

    values = tuple(_positive_int("exploration multiplier", value) for value in raw)
    if len(set(values)) != len(values):
        raise ValueError("exploration multipliers must be unique")
    baseline = _positive_int(
        "baseline_unique_loss_tokens_per_parameter",
        reference.get("baseline_unique_loss_tokens_per_parameter"),
    )
    if baseline not in values:
        raise ValueError("baseline multiplier must be present in exploration multipliers")
    return values


def _find_stage(policy: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stages = policy.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("missing stages")
    matches = [
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage") == stage_name
    ]
    if len(matches) != 1:
        raise ValueError(f"stage must resolve exactly once: {stage_name}")
    return matches[0]


def evaluate_policy_stage(
    *,
    policy: dict[str, Any],
    stage_name: str,
    unique_loss_positions: int,
    multiplier: int,
) -> BudgetEvaluation:
    """Evaluate one policy tier without granting any training authority."""

    if policy.get("schema_version") != "12-6.scaling-data-budget-policy.v1":
        raise ValueError("unsupported scaling data-budget policy schema")
    _validate_truth_boundary(policy)

    allowed = _allowed_multipliers(policy)
    ratio = _positive_int("multiplier", multiplier)
    if ratio not in allowed:
        raise ValueError(f"multiplier {ratio} is not preregistered in policy")

    stage = _find_stage(policy, stage_name)
    parameters = _positive_int("parameter_count", stage.get("parameter_count"))
    observed = _nonnegative_int("unique_loss_positions", unique_loss_positions)
    required = required_unique_loss_positions(parameters, ratio)

    references = stage.get("reference_unique_loss_tokens")
    if not isinstance(references, dict):
        raise ValueError("stage missing reference_unique_loss_tokens")
    key = f"{ratio}x"
    if references.get(key) != required:
        raise ValueError(f"stage reference drift for {stage_name} {key}")

    shortfall = max(required - observed, 0)
    return BudgetEvaluation(
        stage=stage_name,
        parameter_count=parameters,
        multiplier=ratio,
        unique_loss_positions=observed,
        required_unique_loss_positions=required,
        observed_positions_per_parameter=observed / parameters,
        shortfall_positions=shortfall,
        progress_fraction=min(observed / required, 1.0),
        data_budget_tier_met=observed >= required,
        approximate_dense_training_flops_at_tier=6 * parameters * required,
    )
