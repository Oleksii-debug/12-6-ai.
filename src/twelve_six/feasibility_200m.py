"""Evidence-bound builder for the R01 learned-20M -> ~200M feasibility packet.

This module is deliberately a consumer of :mod:`twelve_six.accelerated_scaling`.
It does not grant training, compute, backend-promotion, tokenizer-fit, ModelSpec,
or stage-promotion authority. A packet produced here is prepared evidence that
still needs an independently rooted terminal authority before the canonical R01
roadmap may record ``feasibility_200m.status == "PASS"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Any, Mapping

from twelve_six import accelerated_scaling as r01

PACKET_SCHEMA_VERSION = 1
PACKET_KIND = "R01_200M_FEASIBILITY_PACKET_V1"
PACKET_STATUS = "PREPARED_FOR_INDEPENDENT_VALIDATION"
_ALLOWED_DECISIONS = {"GO", "HOLD", "NO_GO"}
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_EVIDENCE_REF_FIELDS = {
    "repository",
    "git_sha",
    "evidence_sha256",
    "workflow_run_id",
    "workflow_conclusion",
    "terminal",
}
_CANDIDATE_FIELDS = {
    "candidate_id",
    "architecture_sha256",
    "parameter_count",
    "target_unique_loss_positions",
}
_BOUNDARY_FIELDS = {
    "modelspec_frozen_by_packet",
    "tokenizer_fit_authorized_by_packet",
    "backend_promotion_authorized_by_packet",
    "stage_promotion_authorized_by_packet",
    "compute_authorized_by_packet",
    "paid_compute_authorized_by_packet",
    "material_training_authorized_by_packet",
    "optimizer_updates_executed_by_packet",
    "training_executed_by_packet",
    "learned_weights_created_by_packet",
    "final_test_outcomes_read_by_packet",
    "foreign_pretrained_weights_introduced_by_packet",
    "external_llm_or_api_used_for_data_or_intelligence_by_packet",
}
_FALSE_BOUNDARIES: dict[str, Any] = {
    "modelspec_frozen_by_packet": False,
    "tokenizer_fit_authorized_by_packet": False,
    "backend_promotion_authorized_by_packet": False,
    "stage_promotion_authorized_by_packet": False,
    "compute_authorized_by_packet": False,
    "paid_compute_authorized_by_packet": False,
    "material_training_authorized_by_packet": False,
    "optimizer_updates_executed_by_packet": 0,
    "training_executed_by_packet": False,
    "learned_weights_created_by_packet": False,
    "final_test_outcomes_read_by_packet": False,
    "foreign_pretrained_weights_introduced_by_packet": False,
    "external_llm_or_api_used_for_data_or_intelligence_by_packet": False,
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "packet_kind",
    "packet_status",
    "repository",
    "roadmap_id",
    "source_git_sha",
    "roadmap_snapshot_sha256",
    "roadmap_target_parameters",
    "learned_20m_binding",
    "candidate",
    "measurements_20m",
    "measurements_20m_sha256",
    "measurement_authority",
    "requirement_evidence",
    "requirements_covered",
    "decision",
    "authority_boundaries",
    "packet_sha256",
}
_LEARNED_20M_BINDING_FIELDS = {
    "evidence_manifest_sha256",
    "terminal_authority",
    "independent_audit_authority",
}


class FeasibilityPacketError(ValueError):
    """Raised when a requested packet would violate the fail-closed contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FeasibilityPacketError("value_not_canonical_json") from exc
    return rendered.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical UTF-8 JSON with NaN/Inf forbidden."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def compute_packet_sha256(packet: Mapping[str, Any]) -> str:
    """Hash all packet semantics except the self-hash field."""

    if not isinstance(packet, Mapping):
        raise FeasibilityPacketError("packet_not_mapping")
    body = dict(packet)
    body.pop("packet_sha256", None)
    return canonical_sha256(body)


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _valid_evidence_ref(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != _EVIDENCE_REF_FIELDS:
        return False
    return (
        value.get("repository") == r01.REPOSITORY
        and _is_git_sha(value.get("git_sha"))
        and _is_sha256(value.get("evidence_sha256"))
        and _is_positive_int(value.get("workflow_run_id"))
        and value.get("workflow_conclusion") == "success"
        and value.get("terminal") is True
    )


def _validate_measurements(measurements: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(measurements, dict):
        return ["measurements_20m_not_object"]
    if set(measurements) != set(r01.REQUIRED_MEASUREMENTS):
        return ["measurements_20m_fields_mismatch"]

    for name in {
        "tokens_per_second",
        "step_time_seconds",
        "checkpoint_save_seconds",
        "checkpoint_load_seconds",
        "estimated_wall_clock_to_target",
    }:
        if not _is_positive_finite_number(measurements.get(name)):
            errors.append(f"measurement_{name}_invalid")
    for name in {"peak_ram_bytes", "checkpoint_size_bytes"}:
        if not _is_positive_int(measurements.get(name)):
            errors.append(f"measurement_{name}_invalid")

    peak_vram = measurements.get("peak_vram_bytes_or_not_applicable")
    if peak_vram != "NOT_APPLICABLE" and not _is_positive_int(peak_vram):
        errors.append("measurement_peak_vram_bytes_or_not_applicable_invalid")
    energy = measurements.get("energy_or_thermal_observation_if_available")
    if energy is not None and not (isinstance(energy, str) and energy.strip()):
        errors.append("measurement_energy_or_thermal_observation_if_available_invalid")
    return errors


def _product_200m_target(roadmap: Mapping[str, Any]) -> int:
    route = roadmap.get("scale_route")
    if not isinstance(route, list):
        raise FeasibilityPacketError("roadmap_scale_route_missing")
    matches = [
        item
        for item in route
        if isinstance(item, dict) and item.get("id") == "PRODUCT_200M"
    ]
    if len(matches) != 1:
        raise FeasibilityPacketError("roadmap_product_200m_route_invalid")
    target = matches[0].get("approximate_target_parameters")
    if not _is_positive_int(target):
        raise FeasibilityPacketError("roadmap_product_200m_target_invalid")
    return target


def _require_buildable_roadmap(roadmap: Any) -> None:
    if not isinstance(roadmap, dict):
        raise FeasibilityPacketError("roadmap_not_object")
    contract_errors = r01.validate_roadmap(roadmap)
    if contract_errors:
        raise FeasibilityPacketError(
            "roadmap_contract_invalid:" + ",".join(sorted(set(contract_errors)))
        )
    assessment = r01.assess_roadmap(roadmap)
    if (
        not assessment.contract_valid
        or not assessment.terminal_20m_proven
        or not assessment.ready_for_200m_feasibility
        or assessment.next_action != "PREPARE_200M_FEASIBILITY_PACKET"
    ):
        raise FeasibilityPacketError("roadmap_not_at_200m_feasibility_boundary")


def _validate_candidate(candidate: Any) -> list[str]:
    if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_FIELDS:
        return ["candidate_fields_mismatch"]
    errors: list[str] = []
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        errors.append("candidate_id_invalid")
    if not _is_sha256(candidate.get("architecture_sha256")):
        errors.append("candidate_architecture_sha256_invalid")
    if not _is_positive_int(candidate.get("parameter_count")):
        errors.append("candidate_parameter_count_invalid")
    if not _is_positive_int(candidate.get("target_unique_loss_positions")):
        errors.append("candidate_target_unique_loss_positions_invalid")
    return errors


def _validate_requirement_evidence(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["requirement_evidence_not_object"]
    if set(value) != set(r01.REQUIRED_200M_FEASIBILITY):
        return ["requirement_evidence_fields_mismatch"]
    return sorted(
        f"requirement_evidence_{name}_invalid"
        for name, evidence in value.items()
        if not _valid_evidence_ref(evidence)
    )


def _validate_learned_20m_binding(value: Any) -> list[str]:
    if not isinstance(value, dict) or set(value) != _LEARNED_20M_BINDING_FIELDS:
        return ["learned_20m_binding_fields_mismatch"]
    errors: list[str] = []
    if not _is_sha256(value.get("evidence_manifest_sha256")):
        errors.append("learned_20m_evidence_manifest_sha256_invalid")
    for field in ("terminal_authority", "independent_audit_authority"):
        if not _valid_evidence_ref(value.get(field)):
            errors.append(f"learned_20m_{field}_invalid")
    return errors


def build_200m_feasibility_packet(
    *,
    roadmap_snapshot: dict[str, Any],
    source_git_sha: str,
    candidate: dict[str, Any],
    measurements_20m: dict[str, Any],
    measurement_authority: dict[str, Any],
    requirement_evidence: dict[str, dict[str, Any]],
    decision: str,
) -> dict[str, Any]:
    """Build a deterministic non-authorizing ~200M feasibility packet."""

    _require_buildable_roadmap(roadmap_snapshot)
    if not _is_git_sha(source_git_sha):
        raise FeasibilityPacketError("source_git_sha_invalid")

    errors = _validate_candidate(candidate)
    errors.extend(_validate_measurements(measurements_20m))
    errors.extend(_validate_requirement_evidence(requirement_evidence))
    measurements_sha256 = canonical_sha256(measurements_20m)
    candidate_sha256 = canonical_sha256(candidate)
    if not _valid_evidence_ref(measurement_authority):
        errors.append("measurement_authority_invalid")
    elif measurement_authority.get("evidence_sha256") != measurements_sha256:
        errors.append("measurement_authority_payload_mismatch")
    candidate_evidence = requirement_evidence.get(
        "candidate_architecture_and_parameter_count"
    )
    if (
        isinstance(candidate_evidence, dict)
        and candidate_evidence.get("evidence_sha256") != candidate_sha256
    ):
        errors.append("candidate_architecture_evidence_payload_mismatch")
    if decision not in _ALLOWED_DECISIONS:
        errors.append("decision_invalid")
    if errors:
        raise FeasibilityPacketError(",".join(sorted(set(errors))))

    learned_20m = roadmap_snapshot["evidence_state"]["learned_20m"]
    learned_binding = {
        "evidence_manifest_sha256": learned_20m["evidence_manifest_sha256"],
        "terminal_authority": deepcopy(learned_20m["terminal_authority"]),
        "independent_audit_authority": deepcopy(
            learned_20m["independent_audit_authority"]
        ),
    }
    binding_errors = _validate_learned_20m_binding(learned_binding)
    if binding_errors:
        raise FeasibilityPacketError(",".join(binding_errors))

    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_kind": PACKET_KIND,
        "packet_status": PACKET_STATUS,
        "repository": r01.REPOSITORY,
        "roadmap_id": r01.ROADMAP_ID,
        "source_git_sha": source_git_sha,
        "roadmap_snapshot_sha256": canonical_sha256(roadmap_snapshot),
        "roadmap_target_parameters": _product_200m_target(roadmap_snapshot),
        "learned_20m_binding": learned_binding,
        "candidate": deepcopy(candidate),
        "measurements_20m": deepcopy(measurements_20m),
        "measurements_20m_sha256": measurements_sha256,
        "measurement_authority": deepcopy(measurement_authority),
        "requirement_evidence": deepcopy(requirement_evidence),
        "requirements_covered": sorted(r01.REQUIRED_200M_FEASIBILITY),
        "decision": decision,
        "authority_boundaries": deepcopy(_FALSE_BOUNDARIES),
    }
    packet["packet_sha256"] = compute_packet_sha256(packet)
    return packet


def expected_external_identities(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return identities that an external authority must retain for later verify."""

    if not isinstance(packet, Mapping):
        raise FeasibilityPacketError("packet_not_mapping")
    requirement_evidence = packet.get("requirement_evidence")
    if not isinstance(requirement_evidence, dict):
        raise FeasibilityPacketError("requirement_evidence_not_object")
    return {
        "packet_sha256": packet.get("packet_sha256"),
        "roadmap_snapshot_sha256": packet.get("roadmap_snapshot_sha256"),
        "source_git_sha": packet.get("source_git_sha"),
        "measurements_20m_sha256": packet.get("measurements_20m_sha256"),
        "requirement_evidence_sha256": {
            key: value.get("evidence_sha256") if isinstance(value, dict) else None
            for key, value in sorted(requirement_evidence.items())
        },
    }


def validate_200m_feasibility_packet(
    packet: Any,
    *,
    roadmap_snapshot: dict[str, Any],
    expected_packet_sha256: str,
    expected_roadmap_snapshot_sha256: str,
    expected_source_git_sha: str,
    expected_measurements_20m_sha256: str,
    expected_requirement_evidence_sha256: Mapping[str, str],
) -> list[str]:
    """Return fail-closed packet violations against independent expectations."""

    errors: list[str] = []
    if not isinstance(packet, dict):
        return ["packet_not_object"]
    if set(packet) != _TOP_LEVEL_FIELDS:
        errors.append("packet_fields_mismatch")

    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if packet.get("packet_kind") != PACKET_KIND:
        errors.append("packet_kind_mismatch")
    if packet.get("packet_status") != PACKET_STATUS:
        errors.append("packet_status_mismatch")
    if packet.get("repository") != r01.REPOSITORY:
        errors.append("repository_mismatch")
    if packet.get("roadmap_id") != r01.ROADMAP_ID:
        errors.append("roadmap_id_mismatch")
    if not _is_git_sha(packet.get("source_git_sha")):
        errors.append("source_git_sha_invalid")
    if packet.get("source_git_sha") != expected_source_git_sha:
        errors.append("source_git_sha_external_mismatch")

    try:
        _require_buildable_roadmap(roadmap_snapshot)
    except FeasibilityPacketError:
        errors.append("roadmap_not_at_200m_feasibility_boundary")
    try:
        roadmap_sha = canonical_sha256(roadmap_snapshot)
    except FeasibilityPacketError:
        roadmap_sha = None
        errors.append("roadmap_snapshot_not_canonical_json")
    if not _is_sha256(expected_roadmap_snapshot_sha256):
        errors.append("expected_roadmap_snapshot_sha256_invalid")
    if packet.get("roadmap_snapshot_sha256") != roadmap_sha:
        errors.append("roadmap_snapshot_sha256_recompute_mismatch")
    if packet.get("roadmap_snapshot_sha256") != expected_roadmap_snapshot_sha256:
        errors.append("roadmap_snapshot_sha256_external_mismatch")

    try:
        target = _product_200m_target(roadmap_snapshot)
    except FeasibilityPacketError:
        target = None
        errors.append("roadmap_product_200m_target_invalid")
    if packet.get("roadmap_target_parameters") != target:
        errors.append("roadmap_target_parameters_mismatch")

    learned_binding = packet.get("learned_20m_binding")
    errors.extend(_validate_learned_20m_binding(learned_binding))
    learned = roadmap_snapshot.get("evidence_state", {}).get("learned_20m")
    if isinstance(learned, dict):
        expected_binding = {
            "evidence_manifest_sha256": learned.get("evidence_manifest_sha256"),
            "terminal_authority": learned.get("terminal_authority"),
            "independent_audit_authority": learned.get("independent_audit_authority"),
        }
        if learned_binding != expected_binding:
            errors.append("learned_20m_binding_roadmap_mismatch")
    else:
        errors.append("roadmap_learned_20m_missing")

    candidate_value = packet.get("candidate")
    errors.extend(_validate_candidate(candidate_value))
    measurements = packet.get("measurements_20m")
    errors.extend(_validate_measurements(measurements))
    try:
        measurements_sha = canonical_sha256(measurements)
    except FeasibilityPacketError:
        measurements_sha = None
        errors.append("measurements_20m_not_canonical_json")
    if packet.get("measurements_20m_sha256") != measurements_sha:
        errors.append("measurements_20m_sha256_recompute_mismatch")
    if packet.get("measurements_20m_sha256") != expected_measurements_20m_sha256:
        errors.append("measurements_20m_sha256_external_mismatch")

    measurement_authority = packet.get("measurement_authority")
    if not _valid_evidence_ref(measurement_authority):
        errors.append("measurement_authority_invalid")
    elif measurement_authority.get("evidence_sha256") != measurements_sha:
        errors.append("measurement_authority_payload_mismatch")

    requirement_evidence = packet.get("requirement_evidence")
    errors.extend(_validate_requirement_evidence(requirement_evidence))
    if isinstance(requirement_evidence, dict) and isinstance(candidate_value, dict):
        candidate_evidence = requirement_evidence.get(
            "candidate_architecture_and_parameter_count"
        )
        if isinstance(candidate_evidence, dict):
            try:
                candidate_sha = canonical_sha256(candidate_value)
            except FeasibilityPacketError:
                candidate_sha = None
                errors.append("candidate_not_canonical_json")
            if candidate_evidence.get("evidence_sha256") != candidate_sha:
                errors.append("candidate_architecture_evidence_payload_mismatch")

    expected_requirement_keys = set(r01.REQUIRED_200M_FEASIBILITY)
    if not isinstance(expected_requirement_evidence_sha256, Mapping) or set(
        expected_requirement_evidence_sha256
    ) != expected_requirement_keys:
        errors.append("expected_requirement_evidence_sha256_fields_mismatch")
    elif isinstance(requirement_evidence, dict):
        for name in sorted(expected_requirement_keys):
            expected_sha = expected_requirement_evidence_sha256.get(name)
            if not _is_sha256(expected_sha):
                errors.append(f"expected_requirement_evidence_{name}_sha256_invalid")
                continue
            evidence = requirement_evidence.get(name)
            actual_sha = (
                evidence.get("evidence_sha256")
                if isinstance(evidence, dict)
                else None
            )
            if actual_sha != expected_sha:
                errors.append(f"requirement_evidence_{name}_external_mismatch")

    if packet.get("requirements_covered") != sorted(r01.REQUIRED_200M_FEASIBILITY):
        errors.append("requirements_covered_mismatch")
    if packet.get("decision") not in _ALLOWED_DECISIONS:
        errors.append("decision_invalid")

    boundaries = packet.get("authority_boundaries")
    if not isinstance(boundaries, dict) or set(boundaries) != _BOUNDARY_FIELDS:
        errors.append("authority_boundaries_fields_mismatch")
    elif boundaries != _FALSE_BOUNDARIES:
        errors.append("authority_boundaries_must_be_non_authorizing")

    packet_sha = packet.get("packet_sha256")
    if not _is_sha256(packet_sha):
        errors.append("packet_sha256_invalid")
    try:
        recomputed_packet_sha = compute_packet_sha256(packet)
    except FeasibilityPacketError:
        recomputed_packet_sha = None
        errors.append("packet_not_canonical_json")
    if packet_sha != recomputed_packet_sha:
        errors.append("packet_sha256_recompute_mismatch")
    if not _is_sha256(expected_packet_sha256):
        errors.append("expected_packet_sha256_invalid")
    if packet_sha != expected_packet_sha256:
        errors.append("packet_sha256_external_mismatch")
    return sorted(set(errors))
