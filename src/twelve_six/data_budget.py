"""Runtime evaluation for the 12-6 scaling data-budget policy.

Unique corpus loss positions and total training-token exposure are deliberately
separate quantities. This module evaluates a preregistered exposure reference;
it never turns unique data volume into training or paid-compute authority.
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
    planned_training_token_exposures: int
    reference_training_token_exposures: int
    exposure_delta: int
    exposure_relation: str
    reference_exposure_match: bool
    unique_positions_per_parameter: float
    implied_exposures_per_unique_position: float | None
    approximate_dense_training_flops_at_reference: int
    approximate_dense_training_flops_planned: int
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


def required_training_token_exposures(parameter_count: int, multiplier: int) -> int:
    """Return the exact total-exposure reference for a parameter multiplier."""

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
        "source_capacity_target_is_training_budget",
        "training_token_exposures_are_unique_loss_positions",
        "planning_reference_is_minimum_training_requirement",
        "planning_reference_is_quality_guarantee",
        "planning_reference_is_compute_authorization",
        "this_policy_can_authorize_training",
        "paid_compute_authorized",
    )
    for field in forbidden_true:
        if truth.get(field) is not False:
            raise ValueError(f"truth boundary violation: {field} must be false")


def _allowed_multipliers(policy: dict[str, Any]) -> tuple[int, ...]:
    reference = policy.get("reference_policy")
    if not isinstance(reference, dict):
        raise ValueError("missing reference_policy")
    raw = reference.get("exploration_training_token_exposures_per_parameter")
    if not isinstance(raw, list) or not raw:
        raise ValueError("missing exploration_training_token_exposures_per_parameter")

    values = tuple(_positive_int("exploration multiplier", value) for value in raw)
    if len(set(values)) != len(values):
        raise ValueError("exploration multipliers must be unique")
    baseline = _positive_int(
        "baseline_training_token_exposures_per_parameter",
        reference.get("baseline_training_token_exposures_per_parameter"),
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
    planned_training_token_exposures: int,
    multiplier: int,
) -> BudgetEvaluation:
    """Evaluate one exposure reference without conflating it with unique data."""

    if policy.get("schema_version") != "12-6.scaling-data-budget-policy.v1":
        raise ValueError("unsupported scaling data-budget policy schema")
    _validate_truth_boundary(policy)

    allowed = _allowed_multipliers(policy)
    ratio = _positive_int("multiplier", multiplier)
    if ratio not in allowed:
        raise ValueError(f"multiplier {ratio} is not preregistered in policy")

    stage = _find_stage(policy, stage_name)
    parameters = _positive_int("parameter_count", stage.get("parameter_count"))
    unique_positions = _nonnegative_int("unique_loss_positions", unique_loss_positions)
    planned_exposures = _nonnegative_int(
        "planned_training_token_exposures", planned_training_token_exposures
    )
    reference_exposures = required_training_token_exposures(parameters, ratio)

    references = stage.get("reference_training_token_exposures")
    if not isinstance(references, dict):
        raise ValueError("stage missing reference_training_token_exposures")
    key = f"{ratio}x"
    if references.get(key) != reference_exposures:
        raise ValueError(f"stage reference drift for {stage_name} {key}")

    delta = planned_exposures - reference_exposures
    if delta < 0:
        relation = "BELOW_REFERENCE"
    elif delta > 0:
        relation = "ABOVE_REFERENCE"
    else:
        relation = "MATCHES_REFERENCE"

    replay_pressure = planned_exposures / unique_positions if unique_positions > 0 else None
    return BudgetEvaluation(
        stage=stage_name,
        parameter_count=parameters,
        multiplier=ratio,
        unique_loss_positions=unique_positions,
        planned_training_token_exposures=planned_exposures,
        reference_training_token_exposures=reference_exposures,
        exposure_delta=delta,
        exposure_relation=relation,
        reference_exposure_match=delta == 0,
        unique_positions_per_parameter=unique_positions / parameters,
        implied_exposures_per_unique_position=replay_pressure,
        approximate_dense_training_flops_at_reference=(6 * parameters * reference_exposures),
        approximate_dense_training_flops_planned=6 * parameters * planned_exposures,
    )
