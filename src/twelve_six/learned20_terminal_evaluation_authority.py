"""Compose terminal D06 learned-20M evidence with smaller-ladder context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twelve_six.learned20_ladder_context import validate_smaller_ladder_context
from twelve_six.learned20_pilot_evaluation import (
    validate_terminal_pilot_evaluation as _validate_pilot_evaluation,
)


def validate_terminal_pilot_evaluation(evidence: Mapping[str, Any]) -> list[str]:
    """Require terminal pilot measurements plus budget-caveated smaller-ladder context."""

    blockers = list(_validate_pilot_evaluation(evidence))
    pilot = evidence.get("bounded_pilot")
    if not isinstance(pilot, Mapping) or pilot.get("terminal") is not True:
        return sorted(set(blockers))
    d06 = pilot.get("d06_evaluation")
    if isinstance(d06, Mapping):
        blockers.extend(validate_smaller_ladder_context(d06))
    return sorted(set(blockers))
