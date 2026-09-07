"""Compose terminal D06 learned-20M scientific authority."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from twelve_six.learned20_ladder_context import validate_smaller_ladder_context
from twelve_six.learned20_pilot_evaluation import (
    validate_terminal_pilot_evaluation as _validate_pilot_evaluation,
)
from twelve_six.learned20_terminal_run_evidence import validate_terminal_run_evidence

REPOSITORY = "Oleksii-debug/12-6-ai."
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_D06_AUTHORITIES = (
    "heldout_metrics",
    "selection_trajectory",
    "checkpoint_selection",
    "inference_probe",
    "memorization_diagnostic",
    "exposure_accounting",
)


def _terminal_authority(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("repository") == REPOSITORY
        and isinstance(value.get("git_sha"), str)
        and _GIT_SHA.fullmatch(value["git_sha"]) is not None
        and isinstance(value.get("evidence_sha256"), str)
        and _SHA256.fullmatch(value["evidence_sha256"]) is not None
        and isinstance(value.get("workflow_run_id"), int)
        and not isinstance(value.get("workflow_run_id"), bool)
        and value["workflow_run_id"] > 0
        and value.get("workflow_conclusion") == "success"
        and value.get("terminal") is True
    )


def _validate_terminal_subauthorities(d06: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    authorities = d06.get("terminal_authorities")
    if not isinstance(authorities, Mapping):
        return ["bounded_pilot.d06.terminal_authorities_missing"]
    for key in _REQUIRED_D06_AUTHORITIES:
        if not _terminal_authority(authorities.get(key)):
            blockers.append(f"bounded_pilot.d06.terminal_authorities.{key}_missing")
    return blockers


def validate_terminal_pilot_evaluation(evidence: Mapping[str, Any]) -> list[str]:
    """Require terminal measurements, provenance, run accounting, and ladder context."""

    blockers = list(_validate_pilot_evaluation(evidence))
    pilot = evidence.get("bounded_pilot")
    if not isinstance(pilot, Mapping) or pilot.get("terminal") is not True:
        return sorted(set(blockers))
    d06 = pilot.get("d06_evaluation")
    if isinstance(d06, Mapping):
        blockers.extend(_validate_terminal_subauthorities(d06))
        blockers.extend(validate_terminal_run_evidence(pilot, d06))
        blockers.extend(validate_smaller_ladder_context(d06))
    return sorted(set(blockers))
