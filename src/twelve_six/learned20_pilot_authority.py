"""Fail-closed provenance checks for bounded learned-20M pilot evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twelve_six.learned20_checkpoint_authority import (
    assess_launch_with_checkpoint_provenance,
)
from twelve_six.learned20_pilot_evaluation import validate_terminal_pilot_evaluation


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_bounded_pilot_authority(evidence: Mapping[str, Any]) -> list[str]:
    """Require a terminal pilot to bind to the exact launch candidate it exercised."""

    pilot = evidence.get("bounded_pilot")
    if not isinstance(pilot, Mapping) or pilot.get("terminal") is not True:
        return []

    binding = evidence.get("launch_binding")
    if not isinstance(binding, Mapping):
        return ["bounded_pilot.launch_binding_missing"]

    blockers: list[str] = []
    links = {
        "launch_binding_identity": binding.get("identity"),
        "code_sha": binding.get("code_sha"),
        "config_sha256": binding.get("config_sha256"),
        "corpus_identity": binding.get("corpus_identity"),
        "loss_ledger_identity": binding.get("loss_ledger_identity"),
        "tokenizer_identity": binding.get("tokenizer_identity"),
        "checkpoint_identity": binding.get("checkpoint_identity"),
        "evaluation_firewall_identity": binding.get("evaluation_firewall_identity"),
        "training_recipe_identity": binding.get("training_recipe_identity"),
    }
    for key, expected in links.items():
        value = pilot.get(key)
        if not _nonempty_text(value):
            blockers.append(f"bounded_pilot.{key}_missing")
        elif value != expected:
            blockers.append(f"bounded_pilot.{key}_mismatch")

    return sorted(set(blockers))


def assess_launch_with_terminal_provenance(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    material_cost: bool,
) -> dict[str, Any]:
    """Assess launch readiness with checkpoint, pilot-provenance, and D06 evidence."""

    result = assess_launch_with_checkpoint_provenance(
        contract,
        evidence,
        material_cost=material_cost,
    )
    blockers = validate_bounded_pilot_authority(evidence)
    d06_blockers = validate_terminal_pilot_evaluation(evidence)

    if blockers:
        result["long_training_blockers"] = sorted(
            set(result["long_training_blockers"] + blockers)
        )
        result["long_training_ready"] = False

    if d06_blockers:
        result["long_training_blockers"] = sorted(
            set(result["long_training_blockers"] + d06_blockers)
        )
        result["long_training_ready"] = False

    return result
