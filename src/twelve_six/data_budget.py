"""Model-size-aware pretraining data budget checks for 12-6 AI.

The accounting unit is post-pack unique causal-loss tokens. Source bytes, replayed
examples, or unmaterialized candidate capacity must never be substituted for this
quantity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Any


@dataclass(frozen=True)
class DataBudgetResult:
    parameter_count: int
    unique_loss_tokens: int
    tokens_per_parameter_required: float
    observed_tokens_per_parameter: float
    required_unique_loss_tokens: int
    shortfall_tokens: int
    progress_fraction: float
    ready: bool
    approximate_dense_training_flops_at_requirement: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric


def required_unique_loss_tokens(parameter_count: int, tokens_per_parameter: float) -> int:
    """Return the minimum integer token count for a planning ratio."""

    parameters = _positive_int("parameter_count", parameter_count)
    ratio = _positive_number("tokens_per_parameter", tokens_per_parameter)
    return ceil(parameters * ratio)


def evaluate_data_budget(
    *,
    parameter_count: int,
    unique_loss_tokens: int,
    tokens_per_parameter: float,
    dense_training_flops_per_parameter_token: float = 6.0,
) -> DataBudgetResult:
    """Evaluate one fail-closed model/data budget gate.

    This function intentionally knows nothing about source bytes or nominal corpus
    capacity. Callers must supply an already materialized, decontaminated,
    post-pack count of unique causal-loss tokens.
    """

    parameters = _positive_int("parameter_count", parameter_count)
    observed = _nonnegative_int("unique_loss_tokens", unique_loss_tokens)
    ratio = _positive_number("tokens_per_parameter", tokens_per_parameter)
    flops_factor = _positive_number(
        "dense_training_flops_per_parameter_token",
        dense_training_flops_per_parameter_token,
    )

    required = required_unique_loss_tokens(parameters, ratio)
    shortfall = max(required - observed, 0)
    progress = min(observed / required, 1.0)

    return DataBudgetResult(
        parameter_count=parameters,
        unique_loss_tokens=observed,
        tokens_per_parameter_required=ratio,
        observed_tokens_per_parameter=observed / parameters,
        required_unique_loss_tokens=required,
        shortfall_tokens=shortfall,
        progress_fraction=progress,
        ready=observed >= required,
        approximate_dense_training_flops_at_requirement=ceil(
            flops_factor * parameters * required
        ),
    )
