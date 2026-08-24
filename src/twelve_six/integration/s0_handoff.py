"""Fail-closed S0 train -> checkpoint -> evaluation handoff validation.

This module is integration-owned. It validates exact lane/CI/provenance evidence and
never implements D06 evaluation semantics or authorizes stage promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_S0_LANES = ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08")
_ALLOWED_DISPOSITIONS = {"accepted", "held"}
_ALLOWED_AUDIT_VERDICTS = {"PASS", "PASS_WITH_NOTES", "CHANGES_REQUIRED", "BLOCKED"}
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HandoffValidationError(ValueError):
    """Raised when an S0 handoff document is not fail-closed and self-consistent."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffValidationError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise HandoffValidationError(f"{field} must be an array")
    return value


def _require_full_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _FULL_GIT_SHA.fullmatch(value) is None:
        raise HandoffValidationError(f"{field} must be a full lowercase 40-hex Git SHA")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HandoffValidationError(f"{field} must be a lowercase SHA-256")
    return value


def component_map_sha256(components: Sequence[Mapping[str, Any]]) -> str:
    """Return the canonical identity of the complete ordered component evidence map."""
    payload = json.dumps(
        list(components), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_s0_handoff(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an S0 handoff and return derived readiness facts.

    Readiness here means only that a LOCAL_FREE train/checkpoint/eval execution handoff
    has complete exact-head component evidence. It is deliberately independent from
    promotion authority: this contract requires ``promotion_allowed`` to remain false.
    """

    if document.get("schema") != "12-6.s0-train-checkpoint-eval-handoff.v1":
        raise HandoffValidationError("unsupported handoff schema")
    if document.get("run_id") != "12-6-AI-SWARM-EXP-01":
        raise HandoffValidationError("unexpected swarm run_id")
    if document.get("repository") != "Oleksii-debug/12-6-ai.":
        raise HandoffValidationError("repository identity mismatch")
    if document.get("stage") != "S0":
        raise HandoffValidationError("handoff stage must be S0")

    composition = _require_mapping(document.get("composition"), "composition")
    _require_full_sha(composition.get("source_sha"), "composition.source_sha")
    if composition.get("ci_conclusion") != "success":
        raise HandoffValidationError("composition exact-head CI must be success")
    if not isinstance(composition.get("ci_run_id"), int) or composition["ci_run_id"] <= 0:
        raise HandoffValidationError("composition.ci_run_id must be a positive integer")
    _require_sha256(composition.get("environment_lock_sha256"), "composition.environment_lock_sha256")

    authorization = _require_mapping(document.get("authorization"), "authorization")
    if authorization.get("execution_class") != "LOCAL_FREE":
        raise HandoffValidationError("only LOCAL_FREE execution may be prepared by this run")
    if authorization.get("local_free_allowed") is not True:
        raise HandoffValidationError("LOCAL_FREE execution must be explicitly allowed")
    if authorization.get("paid_compute_authorized") is not False:
        raise HandoffValidationError("paid compute must remain unauthorized")

    components_raw = _require_sequence(document.get("components"), "components")
    components: list[Mapping[str, Any]] = []
    by_lane: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(components_raw):
        component = _require_mapping(raw, f"components[{index}]")
        lane = component.get("lane")
        if lane not in REQUIRED_S0_LANES:
            raise HandoffValidationError(f"components[{index}].lane is not a required S0 lane")
        if lane in by_lane:
            raise HandoffValidationError(f"duplicate component lane {lane}")
        _require_full_sha(component.get("source_sha"), f"components[{index}].source_sha")
        disposition = component.get("disposition")
        if disposition not in _ALLOWED_DISPOSITIONS:
            raise HandoffValidationError(f"invalid disposition for {lane}")
        if not isinstance(component.get("pr_number"), int) or component["pr_number"] <= 0:
            raise HandoffValidationError(f"{lane}.pr_number must be a positive integer")
        ci = _require_mapping(component.get("ci"), f"{lane}.ci")
        if not isinstance(ci.get("run_id"), int) or ci["run_id"] <= 0:
            raise HandoffValidationError(f"{lane}.ci.run_id must be a positive integer")
        if ci.get("conclusion") not in {"success", "failure"}:
            raise HandoffValidationError(f"{lane}.ci.conclusion must be success or failure")
        if disposition == "accepted":
            if ci.get("conclusion") != "success":
                raise HandoffValidationError(f"accepted lane {lane} requires exact-head success")
            if component.get("contains_behavioral_weights") is not False:
                raise HandoffValidationError(f"accepted lane {lane} may not add behavioral Base weights")
            if component.get("contains_foreign_pretrained_weights") is not False:
                raise HandoffValidationError(f"accepted lane {lane} may not add foreign pretrained weights")
        else:
            reason = component.get("hold_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise HandoffValidationError(f"held lane {lane} requires hold_reason")
        components.append(component)
        by_lane[lane] = component

    if tuple(sorted(by_lane)) != REQUIRED_S0_LANES:
        missing = sorted(set(REQUIRED_S0_LANES) - set(by_lane))
        raise HandoffValidationError(f"component map must contain D01-D08 exactly; missing={missing}")

    declared_map_hash = _require_sha256(document.get("component_map_sha256"), "component_map_sha256")
    calculated_map_hash = component_map_sha256(components)
    if declared_map_hash != calculated_map_hash:
        raise HandoffValidationError("component_map_sha256 does not match component evidence")

    blockers_raw = _require_sequence(document.get("blockers"), "blockers")
    blockers = [str(item).strip() for item in blockers_raw]
    if any(not item for item in blockers):
        raise HandoffValidationError("blockers may not contain empty entries")

    held_lanes = tuple(lane for lane in REQUIRED_S0_LANES if by_lane[lane]["disposition"] == "held")
    accepted_lanes = tuple(lane for lane in REQUIRED_S0_LANES if by_lane[lane]["disposition"] == "accepted")
    execution_ready = not held_lanes and not blockers
    expected_state = "READY_LOCAL_FREE" if execution_ready else "PREPARED_BLOCKED"
    if document.get("handoff_state") != expected_state:
        raise HandoffValidationError(
            f"handoff_state must be {expected_state} for the supplied evidence"
        )

    if document.get("promotion_allowed") is not False:
        raise HandoffValidationError("handoff evidence must never self-authorize promotion")

    audits = _require_mapping(document.get("audits"), "audits")
    if set(audits) != {"AUDIT-A", "AUDIT-B"}:
        raise HandoffValidationError("audits must contain exactly AUDIT-A and AUDIT-B")
    for name, verdict in audits.items():
        if verdict not in _ALLOWED_AUDIT_VERDICTS:
            raise HandoffValidationError(f"unsupported {name} verdict")

    artifact_contract = _require_mapping(document.get("artifact_contract"), "artifact_contract")
    required_artifacts = {
        "resolved_run_manifest",
        "training_metrics",
        "checkpoint_manifest",
        "checkpoint_payload_hashes",
        "evaluation_report",
    }
    if set(artifact_contract) != required_artifacts:
        raise HandoffValidationError("artifact_contract must name the complete train/checkpoint/eval set")
    for name, value in artifact_contract.items():
        if not isinstance(value, str) or not value.strip():
            raise HandoffValidationError(f"artifact_contract.{name} must be a non-empty path template")

    return {
        "execution_ready": execution_ready,
        "handoff_state": expected_state,
        "accepted_lanes": accepted_lanes,
        "held_lanes": held_lanes,
        "component_map_sha256": calculated_map_hash,
        "promotion_allowed": False,
    }
