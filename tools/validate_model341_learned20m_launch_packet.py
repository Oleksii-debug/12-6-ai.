#!/usr/bin/env python3
"""Validate and assess the MODEL-341 learned-20M launch packet."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

EXPECTED_AUTHORITY = {
    "repository": "Oleksii-debug/12-6-ai.",
    "template_main_sha": "a73ab38026cb7849f478cc13ad58b93534a76e2f",
    "model341_branch": "model341/20m-candidate-a-20260826",
    "model341_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
    "modelspec_sha256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
    "parameter_count": 20613440,
    "base_lineage": "RANDOM_INIT_PRETRAINING_ONLY",
}

REQUIRED_AUTHORITIES = {
    "qualified_integration_head",
    "research_corpus_v1",
    "reserved_evaluation_decontamination",
    "cluster_safe_split",
    "deterministic_packing",
    "unique_post_pack_loss_ledger",
    "tokenizer_or_byte_baseline_decision",
    "checkpoint_integrity",
    "learned_ladder_independent_verification",
    "selection_validation",
    "training_recipe",
    "runtime_profile",
    "cost_model",
    "independent_launch_audit",
}

TERMINAL_DECISIONS = {"pass", "success", "qualified", "admit", "authorized"}
GITHUB_REF = re.compile(
    r"^github:[A-Za-z0-9._/+-]+@[0-9a-f]{40}:(pass|success|qualified|admit|authorized)$"
)
ARTIFACT_REF = re.compile(
    r"^artifact:[A-Za-z0-9._/+-]+@sha256:[0-9a-f]{64}:(pass|success|qualified|admit|authorized)$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SELF_SCOPE = "model341-learned20m-launch-packet-v1"

EXPECTED_BLOCKED = [
    "terminal_authorities_missing",
    "training_recipe_unbound",
    "evaluation_firewall_unbound",
    "compute_envelope_unbound",
    "explicit_compute_authorization_missing",
    "explicit_training_authorization_missing",
    "bounded_smoke_evidence_missing",
    "short_horizon_evidence_missing",
]


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _is_nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _is_terminal_ref(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if SELF_SCOPE in value.lower():
        return False
    match = GITHUB_REF.fullmatch(value) or ARTIFACT_REF.fullmatch(value)
    return bool(match and match.group(1) in TERMINAL_DECISIONS)


def _is_authorized_ref(value: Any) -> bool:
    return _is_terminal_ref(value) and isinstance(value, str) and value.endswith(":authorized")


def _recipe_ready(recipe: Any) -> bool:
    if not isinstance(recipe, dict):
        return False
    seeds = recipe.get("seed_set")
    if not isinstance(seeds, list) or not seeds:
        return False
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        return False
    if len(seeds) != len(set(seeds)):
        return False
    for key in (
        "optimizer",
        "scheduler",
        "precision",
        "replay_policy",
        "stop_rule",
        "checkpoint_cadence",
        "resume_policy",
    ):
        if not isinstance(recipe.get(key), str) or not recipe[key].strip():
            return False
    if not _is_positive_number(recipe.get("gradient_clip_norm")):
        return False
    unique_positions = recipe.get("unique_loss_positions")
    total_exposure = recipe.get("total_training_exposure")
    if not isinstance(unique_positions, int) or isinstance(unique_positions, bool) or unique_positions <= 0:
        return False
    if not isinstance(total_exposure, int) or isinstance(total_exposure, bool) or total_exposure <= 0:
        return False
    if total_exposure < unique_positions:
        return False
    return (
        recipe.get("unique_loss_unit") == "UNIQUE_AUTHORIZED_CAUSAL_LOSS_POSITIONS"
        and recipe.get("training_exposure_unit") == "CAUSAL_LOSS_POSITION_EXPOSURES"
    )


def _evaluation_ready(firewall: Any) -> bool:
    if not isinstance(firewall, dict):
        return False
    return (
        isinstance(firewall.get("selection_validation_identity"), str)
        and SHA256.fullmatch(firewall["selection_validation_identity"]) is not None
        and isinstance(firewall.get("final_test_identity"), str)
        and SHA256.fullmatch(firewall["final_test_identity"]) is not None
        and firewall.get("training_must_exclude_selection_validation") is True
        and firewall.get("training_must_exclude_final_test") is True
        and firewall.get("final_test_read_before_terminal_training") is False
        and isinstance(firewall.get("selection_schedule"), str)
        and bool(firewall["selection_schedule"].strip())
    )


def _compute_envelope_ready(envelope: Any) -> bool:
    if not isinstance(envelope, dict):
        return False
    return (
        _is_positive_number(envelope.get("estimated_training_flops"))
        and isinstance(envelope.get("resource_shape"), str)
        and bool(envelope["resource_shape"].strip())
        and _is_positive_number(envelope.get("profiled_loss_positions_per_second"))
        and _is_positive_number(envelope.get("wall_clock_upper_bound_seconds"))
        and _is_nonnegative_number(envelope.get("maximum_budget_eur"))
    )


def assess_packet(data: dict[str, Any]) -> dict[str, Any]:
    authorities = data.get("required_authorities")
    all_authorities = (
        isinstance(authorities, dict)
        and set(authorities) == REQUIRED_AUTHORITIES
        and all(_is_terminal_ref(authorities.get(key)) for key in REQUIRED_AUTHORITIES)
    )
    recipe_ready = _recipe_ready(data.get("training_recipe"))
    evaluation_ready = _evaluation_ready(data.get("evaluation_firewall"))
    compute_ready = _compute_envelope_ready(data.get("compute_envelope"))

    envelope = data.get("compute_envelope") if isinstance(data.get("compute_envelope"), dict) else {}
    compute_authorized = _is_authorized_ref(envelope.get("compute_authorization"))
    training_authorized = _is_authorized_ref(envelope.get("training_authorization"))

    phase = data.get("phase_evidence") if isinstance(data.get("phase_evidence"), dict) else {}
    smoke_ready = _is_terminal_ref(phase.get("bounded_smoke"))
    short_ready = _is_terminal_ref(phase.get("short_horizon"))

    ready_for_authorization_request = all_authorities and recipe_ready and evaluation_ready and compute_ready
    ready_for_short_horizon = (
        ready_for_authorization_request
        and compute_authorized
        and training_authorized
        and smoke_ready
    )
    ready_for_long_training = ready_for_short_horizon and short_ready

    blockers: list[str] = []
    if not all_authorities:
        blockers.append("terminal_authorities_missing")
    if not recipe_ready:
        blockers.append("training_recipe_unbound")
    if not evaluation_ready:
        blockers.append("evaluation_firewall_unbound")
    if not compute_ready:
        blockers.append("compute_envelope_unbound")
    if not compute_authorized:
        blockers.append("explicit_compute_authorization_missing")
    if not training_authorized:
        blockers.append("explicit_training_authorization_missing")
    if not smoke_ready:
        blockers.append("bounded_smoke_evidence_missing")
    if not short_ready:
        blockers.append("short_horizon_evidence_missing")

    return {
        "ready_for_authorization_request": ready_for_authorization_request,
        "ready_for_short_horizon": ready_for_short_horizon,
        "ready_for_long_training": ready_for_long_training,
        "current_blockers": blockers,
    }


def validate_packet(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _expect(errors, type(data.get("schema_version")) is int and data["schema_version"] == 1, "schema_version must be integer 1")
    _expect(errors, data.get("packet_id") == "MODEL341-LEARNED-20M-LAUNCH-PACKET-V1", "packet_id mismatch")

    authority = data.get("authority")
    _expect(errors, isinstance(authority, dict), "authority must be an object")
    if isinstance(authority, dict):
        for key, value in EXPECTED_AUTHORITY.items():
            _expect(errors, authority.get(key) == value, f"authority.{key} mismatch")

    truth = data.get("truth_boundary")
    _expect(errors, isinstance(truth, dict), "truth_boundary must be an object")
    if isinstance(truth, dict):
        _expect(errors, truth.get("package_scope") == "CONTROL_PLANE_ONLY", "package scope drift")
        _expect(errors, type(truth.get("optimizer_updates")) is int and truth.get("optimizer_updates") == 0, "optimizer_updates must remain integer zero")
        for key in (
            "corpus_mutation",
            "tokenizer_fit",
            "model_weight_mutation",
            "final_test_access",
            "paid_compute_used",
            "stage_promotion",
            "learned_20m_claim",
        ):
            _expect(errors, truth.get(key) is False, f"truth_boundary.{key} must remain false")

    contract = data.get("authority_contract")
    _expect(errors, isinstance(contract, dict), "authority_contract must be an object")
    if isinstance(contract, dict):
        _expect(errors, contract.get("terminal_only") is True, "authority references must remain terminal-only")
        _expect(errors, contract.get("queued_running_cancelled_not_pass") is True, "queued/running/cancelled firewall drift")
        _expect(errors, contract.get("self_attestation_forbidden") is True, "self-attestation firewall drift")
        _expect(errors, contract.get("accepted_reference_kinds") == ["github", "artifact"], "authority reference kinds drift")
        _expect(errors, set(contract.get("accepted_terminal_decisions", [])) == TERMINAL_DECISIONS, "terminal decision set drift")

    authorities = data.get("required_authorities")
    _expect(errors, isinstance(authorities, dict), "required_authorities must be an object")
    if isinstance(authorities, dict):
        _expect(errors, set(authorities) == REQUIRED_AUTHORITIES, "required authority set mismatch")
        for key, value in authorities.items():
            _expect(errors, value is None or _is_terminal_ref(value), f"required_authorities.{key} is not an immutable terminal reference")

    recipe = data.get("training_recipe")
    _expect(errors, isinstance(recipe, dict), "training_recipe must be an object")
    if isinstance(recipe, dict):
        _expect(errors, recipe.get("unique_loss_unit") == "UNIQUE_AUTHORIZED_CAUSAL_LOSS_POSITIONS", "unique-loss unit drift")
        _expect(errors, recipe.get("training_exposure_unit") == "CAUSAL_LOSS_POSITION_EXPOSURES", "training-exposure unit drift")
        unique_positions = recipe.get("unique_loss_positions")
        total_exposure = recipe.get("total_training_exposure")
        if unique_positions is not None:
            _expect(errors, type(unique_positions) is int and unique_positions > 0, "unique_loss_positions must be a positive integer when bound")
        if total_exposure is not None:
            _expect(errors, type(total_exposure) is int and total_exposure > 0, "total_training_exposure must be a positive integer when bound")
        if type(unique_positions) is int and type(total_exposure) is int:
            _expect(errors, total_exposure >= unique_positions, "training exposure cannot be smaller than unique loss positions")

    firewall = data.get("evaluation_firewall")
    _expect(errors, isinstance(firewall, dict), "evaluation_firewall must be an object")
    if isinstance(firewall, dict):
        _expect(errors, firewall.get("training_must_exclude_selection_validation") is True, "selection-validation exclusion must remain true")
        _expect(errors, firewall.get("training_must_exclude_final_test") is True, "final-test exclusion must remain true")
        _expect(errors, firewall.get("final_test_read_before_terminal_training") is False, "final test must stay unread before terminal training")
        for key in ("selection_validation_identity", "final_test_identity"):
            value = firewall.get(key)
            _expect(errors, value is None or (isinstance(value, str) and SHA256.fullmatch(value) is not None), f"evaluation_firewall.{key} must be null or SHA-256")

    envelope = data.get("compute_envelope")
    _expect(errors, isinstance(envelope, dict), "compute_envelope must be an object")
    if isinstance(envelope, dict):
        for key in ("estimated_training_flops", "profiled_loss_positions_per_second", "wall_clock_upper_bound_seconds"):
            value = envelope.get(key)
            _expect(errors, value is None or _is_positive_number(value), f"compute_envelope.{key} must be null or positive")
        budget = envelope.get("maximum_budget_eur")
        _expect(errors, budget is None or _is_nonnegative_number(budget), "compute_envelope.maximum_budget_eur must be null or non-negative")
        resource_shape = envelope.get("resource_shape")
        _expect(errors, resource_shape is None or (isinstance(resource_shape, str) and bool(resource_shape.strip())), "compute_envelope.resource_shape must be null or non-empty")
        for key in ("compute_authorization", "training_authorization"):
            value = envelope.get(key)
            _expect(errors, value is None or _is_authorized_ref(value), f"compute_envelope.{key} must be null or an immutable :authorized reference")

    phase = data.get("phase_evidence")
    _expect(errors, isinstance(phase, dict), "phase_evidence must be an object")
    if isinstance(phase, dict):
        _expect(errors, set(phase) == {"bounded_smoke", "short_horizon"}, "phase evidence set mismatch")
        for key, value in phase.items():
            _expect(errors, value is None or _is_terminal_ref(value), f"phase_evidence.{key} must be null or terminal evidence")

    assessment = assess_packet(data)
    decision = data.get("decision")
    _expect(errors, isinstance(decision, dict), "decision must be an object")
    if isinstance(decision, dict):
        for key in (
            "ready_for_authorization_request",
            "ready_for_short_horizon",
            "ready_for_long_training",
            "current_blockers",
        ):
            _expect(errors, decision.get(key) == assessment[key], f"decision.{key} does not match computed assessment")

    expected_status = "BLOCKED_PENDING_TERMINAL_AUTHORITIES"
    if assessment["ready_for_long_training"]:
        expected_status = "READY_FOR_LONG_TRAINING"
    elif assessment["ready_for_short_horizon"]:
        expected_status = "READY_FOR_SHORT_HORIZON"
    elif assessment["ready_for_authorization_request"]:
        expected_status = "READY_FOR_AUTHORIZATION_REQUEST"
    _expect(errors, data.get("status") == expected_status, f"status must be {expected_status}")

    return errors


def validate_path(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ["launch packet root must be an object"]
    return validate_packet(data)


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(
        "configs/control/model341_learned20m_launch_packet_v1.json"
    )
    errors = validate_path(path)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    assessment = assess_packet(data)
    print(f"PASS: {data['packet_id']} {data['status']}")
    print("blockers=" + ",".join(assessment["current_blockers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
