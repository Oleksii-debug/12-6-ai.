"""Fail-closed accounting for unique training data versus repeated exposure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any


class ExposureBudgetError(ValueError):
    """Raised when an exposure-budget request is malformed."""


@dataclass(frozen=True)
class ExposureBudgetAssessment:
    """Exact exposure accounting without implying training authorization."""

    status: str
    unique_loss_positions: int
    requested_total_exposures: int
    repeat_exposures: int
    effective_epochs_numerator: int
    effective_epochs_denominator: int
    repeat_policy_required: bool
    max_repeat_epochs_numerator: int | None
    max_repeat_epochs_denominator: int | None
    training_authorized: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_positive_int(value: int, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ExposureBudgetError(f"{field} must be a positive integer")
    return value


def _normalize_repeat_cap(value: Fraction | int | str | None) -> Fraction | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ExposureBudgetError("max_repeat_epochs must not be boolean")
    try:
        cap = value if isinstance(value, Fraction) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ExposureBudgetError("max_repeat_epochs must be a positive rational") from exc
    if cap < 1:
        raise ExposureBudgetError("max_repeat_epochs must be at least 1")
    return cap


def assess_training_exposure(
    *,
    unique_loss_positions: int,
    requested_total_exposures: int,
    max_repeat_epochs: Fraction | int | str | None = None,
) -> ExposureBudgetAssessment:
    """Assess whether a requested exposure fits an explicit data-use envelope.

    ``unique_loss_positions`` is immutable corpus capacity. Repeated optimizer
    exposure never increases it. A request above the unique ledger fails closed
    unless the caller supplies a separately preregistered maximum epoch cap.

    A positive result only means the arithmetic fits that exposure policy. It is
    deliberately never training authorization.
    """

    unique = _require_positive_int(unique_loss_positions, field="unique_loss_positions")
    requested = _require_positive_int(
        requested_total_exposures,
        field="requested_total_exposures",
    )
    cap = _normalize_repeat_cap(max_repeat_epochs)
    epochs = Fraction(requested, unique)
    repeat = max(0, requested - unique)

    if requested <= unique:
        status = "WITHIN_UNIQUE_LEDGER"
        policy_required = False
    elif cap is None:
        status = "BLOCKED_REPEAT_POLICY_REQUIRED"
        policy_required = True
    elif epochs <= cap:
        status = "WITHIN_PREREGISTERED_REPEAT_CAP"
        policy_required = True
    else:
        status = "BLOCKED_REPEAT_CAP_EXCEEDED"
        policy_required = True

    return ExposureBudgetAssessment(
        status=status,
        unique_loss_positions=unique,
        requested_total_exposures=requested,
        repeat_exposures=repeat,
        effective_epochs_numerator=epochs.numerator,
        effective_epochs_denominator=epochs.denominator,
        repeat_policy_required=policy_required,
        max_repeat_epochs_numerator=None if cap is None else cap.numerator,
        max_repeat_epochs_denominator=None if cap is None else cap.denominator,
    )
