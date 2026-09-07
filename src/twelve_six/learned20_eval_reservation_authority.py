"""D06 fail-closed evaluation reservation authority for learned-20M launch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REQUIRED_SELECTION_STRATA = ("UA", "EN", "CODE")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_evaluation_reservation_authority(evidence: Mapping[str, Any]) -> list[str]:
    """Require distinct terminal selection/final-test authorities before a pilot.

    This validator consumes metadata only. It deliberately does not open final-test
    payloads or outcomes. A terminal registry must prove that selection-validation
    covers UA/EN/code and that final-test material is permanently ineligible for
    selection, tokenizer fitting, and training.
    """

    registry = evidence.get("evaluation_reservation_registry")
    if not isinstance(registry, Mapping) or registry.get("terminal") is not True:
        return ["evaluation_reservation_registry_not_terminal"]

    blockers: list[str] = []
    registry_identity = registry.get("identity")
    if not _nonempty_text(registry_identity):
        blockers.append("evaluation_reservation_registry.identity_missing")

    selection = registry.get("selection_validation")
    final_test = registry.get("final_test")
    if not isinstance(selection, Mapping) or selection.get("terminal") is not True:
        blockers.append("evaluation_reservation_registry.selection_validation_not_terminal")
    if not isinstance(final_test, Mapping) or final_test.get("terminal") is not True:
        blockers.append("evaluation_reservation_registry.final_test_not_terminal")

    if isinstance(selection, Mapping):
        selection_identity = selection.get("identity")
        if not _nonempty_text(selection_identity):
            blockers.append("evaluation_reservation_registry.selection_validation.identity_missing")
        if selection.get("purpose") != "selection_validation":
            blockers.append("evaluation_reservation_registry.selection_validation.purpose_invalid")
        if selection.get("final_test") is not False:
            blockers.append("evaluation_reservation_registry.selection_validation.final_test_role_drift")
        for key in (
            "training_allowed",
            "tokenizer_fit_allowed",
            "data_selection_source_allowed",
        ):
            if selection.get(key) is not False:
                blockers.append(
                    f"evaluation_reservation_registry.selection_validation.{key}_must_be_false"
                )
        strata = selection.get("strata")
        if not isinstance(strata, Mapping):
            blockers.append("evaluation_reservation_registry.selection_validation.strata_missing")
        else:
            suite_ids: set[str] = set()
            for stratum in REQUIRED_SELECTION_STRATA:
                item = strata.get(stratum)
                prefix = f"evaluation_reservation_registry.selection_validation.strata.{stratum}"
                if not isinstance(item, Mapping):
                    blockers.append(f"{prefix}_missing")
                    continue
                if not _positive_int(item.get("record_count")):
                    blockers.append(f"{prefix}.record_count_invalid")
                suite_identity = item.get("suite_identity")
                if not _nonempty_text(suite_identity):
                    blockers.append(f"{prefix}.suite_identity_missing")
                elif suite_identity in suite_ids:
                    blockers.append(
                        "evaluation_reservation_registry.selection_validation.suite_identity_reused"
                    )
                else:
                    suite_ids.add(suite_identity)

    if isinstance(final_test, Mapping):
        final_identity = final_test.get("identity")
        if not _nonempty_text(final_identity):
            blockers.append("evaluation_reservation_registry.final_test.identity_missing")
        if final_test.get("purpose") != "final_test":
            blockers.append("evaluation_reservation_registry.final_test.purpose_invalid")
        for key in (
            "selection_eligible",
            "hyperparameter_selection_eligible",
            "tokenizer_fit_eligible",
            "training_eligible",
            "payload_accessed_before_selection_lock",
            "outcome_accessed_before_selection_lock",
        ):
            if final_test.get(key) is not False:
                blockers.append(f"evaluation_reservation_registry.final_test.{key}_must_be_false")

    if isinstance(selection, Mapping) and isinstance(final_test, Mapping):
        selection_identity = selection.get("identity")
        final_identity = final_test.get("identity")
        if _nonempty_text(selection_identity) and selection_identity == final_identity:
            blockers.append("evaluation_reservation_registry.selection_and_final_identity_collision")
        if registry.get("selection_final_disjoint") is not True:
            blockers.append("evaluation_reservation_registry.selection_final_disjoint_not_proven")

    firewall = evidence.get("evaluation_firewall")
    if (
        isinstance(firewall, Mapping)
        and firewall.get("terminal") is True
        and firewall.get("reservation_registry_identity") != registry_identity
    ):
        blockers.append("evaluation_firewall.reservation_registry_identity_mismatch")

    return sorted(set(blockers))
