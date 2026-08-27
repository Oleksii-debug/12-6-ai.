from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MANIFEST_SCHEMA = "12-6.export-runtime-targets.v1"
EVIDENCE_SCHEMA = "12-6.export-runtime-parity-evidence.v1"
REPORT_SCHEMA = "12-6.export-runtime-parity-report.v1"
PROMOTION_SEQUENCE = ("DISCOVERED", "CANDIDATE", "PARITY_PROVEN", "ADOPTED")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ExportRuntimeEvidenceError(ValueError):
    """Raised when export/runtime authority or parity evidence is incomplete."""


@dataclass(frozen=True)
class NumericParity:
    shape: tuple[int, ...]
    element_count: int
    mismatch_count: int
    max_abs_error: float
    max_allowed_error: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "element_count": self.element_count,
            "mismatch_count": self.mismatch_count,
            "max_abs_error": self.max_abs_error,
            "max_allowed_error": self.max_allowed_error,
            "passed": self.passed,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportRuntimeEvidenceError(f"{name} must be a mapping")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportRuntimeEvidenceError(f"{name} must be a non-empty string")
    return value


def _require_hex(value: Any, name: str, pattern: re.Pattern[str]) -> str:
    text = _require_string(value, name)
    if pattern.fullmatch(text) is None:
        raise ExportRuntimeEvidenceError(f"{name} has an invalid digest/commit shape")
    return text


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExportRuntimeEvidenceError(f"{name} must be boolean")
    return value


def _require_nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportRuntimeEvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ExportRuntimeEvidenceError(f"{name} must be finite and non-negative")
    return result


def _validate_target(target: Mapping[str, Any]) -> None:
    expected = {
        "id",
        "upstream_repo",
        "release_tag",
        "upstream_commit_sha",
        "release_published_at",
        "license_spdx",
        "license_path",
        "license_git_blob_sha",
        "candidate_state",
        "parity_evidence_identity",
        "adoption_authority",
    }
    if set(target) != expected:
        raise ExportRuntimeEvidenceError("target fields do not match the v1 schema")
    _require_string(target["id"], "target.id")
    _require_string(target["upstream_repo"], "target.upstream_repo")
    _require_string(target["release_tag"], "target.release_tag")
    _require_hex(target["upstream_commit_sha"], "target.upstream_commit_sha", _HEX40)
    _require_string(target["release_published_at"], "target.release_published_at")
    _require_string(target["license_spdx"], "target.license_spdx")
    _require_string(target["license_path"], "target.license_path")
    _require_hex(target["license_git_blob_sha"], "target.license_git_blob_sha", _HEX40)

    state = target["candidate_state"]
    if state not in PROMOTION_SEQUENCE:
        raise ExportRuntimeEvidenceError("target.candidate_state is not a promotion state")
    parity_identity = target["parity_evidence_identity"]
    adoption_authority = target["adoption_authority"]
    if state in {"DISCOVERED", "CANDIDATE"}:
        if parity_identity is not None or adoption_authority is not None:
            raise ExportRuntimeEvidenceError(
                "candidate/discovered target cannot carry promotion evidence"
            )
    elif state == "PARITY_PROVEN":
        _require_hex(parity_identity, "target.parity_evidence_identity", _HEX64)
        if adoption_authority is not None:
            raise ExportRuntimeEvidenceError("PARITY_PROVEN target cannot claim adoption authority")
    else:
        _require_hex(parity_identity, "target.parity_evidence_identity", _HEX64)
        _require_string(adoption_authority, "target.adoption_authority")


def manifest_identity(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_identity", None)
    return semantic_sha256(payload)


def validate_target_manifest(manifest: Mapping[str, Any]) -> str:
    expected = {
        "schema",
        "project_repo",
        "project_base_sha",
        "parent_issue",
        "lane_issue",
        "claim_issue",
        "promotion_sequence",
        "truth_boundary",
        "targets",
        "manifest_identity",
    }
    if set(manifest) != expected:
        raise ExportRuntimeEvidenceError("manifest fields do not match the v1 schema")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ExportRuntimeEvidenceError("unsupported manifest schema")
    if manifest["project_repo"] != "Oleksii-debug/12-6-ai.":
        raise ExportRuntimeEvidenceError("project repository identity drift")
    _require_hex(manifest["project_base_sha"], "project_base_sha", _HEX40)
    if manifest["parent_issue"] != 720 or manifest["lane_issue"] != 8:
        raise ExportRuntimeEvidenceError("parent/lane authority drift")
    if manifest["claim_issue"] != 744:
        raise ExportRuntimeEvidenceError("claim authority drift")
    if tuple(manifest["promotion_sequence"]) != PROMOTION_SEQUENCE:
        raise ExportRuntimeEvidenceError("promotion sequence drift")

    boundary = _require_mapping(manifest["truth_boundary"], "truth_boundary")
    required_flags = {
        "backend_execution_required": True,
        "adoption_authority_required": True,
        "canonical_base_random_init_only": True,
        "foreign_pretrained_weights_allowed": False,
        "paid_compute_authorized": False,
        "final_test_access_authorized": False,
    }
    if boundary != required_flags:
        raise ExportRuntimeEvidenceError("truth boundary must remain fail-closed")

    targets = manifest["targets"]
    if not isinstance(targets, list) or not targets:
        raise ExportRuntimeEvidenceError("manifest must contain at least one target")
    seen: set[str] = set()
    for raw_target in targets:
        target = _require_mapping(raw_target, "target")
        _validate_target(target)
        target_id = str(target["id"])
        if target_id in seen:
            raise ExportRuntimeEvidenceError("duplicate target id")
        seen.add(target_id)

    expected_identity = manifest_identity(manifest)
    if manifest["manifest_identity"] != expected_identity:
        raise ExportRuntimeEvidenceError("manifest identity mismatch")
    return expected_identity


def load_target_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    manifest = dict(_require_mapping(value, "manifest"))
    validate_target_manifest(manifest)
    return manifest


def _numeric_shape_and_values(
    value: Any, path: str = "output"
) -> tuple[tuple[int, ...], list[float]]:
    if isinstance(value, bool):
        raise ExportRuntimeEvidenceError(f"{path} contains boolean, not numeric evidence")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ExportRuntimeEvidenceError(f"{path} contains non-finite numeric evidence")
        return (), [number]
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ExportRuntimeEvidenceError(f"{path} must be numeric or a nested numeric sequence")
    if not value:
        raise ExportRuntimeEvidenceError(f"{path} cannot be empty")

    child_shape: tuple[int, ...] | None = None
    flattened: list[float] = []
    for index, item in enumerate(value):
        shape, numbers = _numeric_shape_and_values(item, f"{path}[{index}]")
        if child_shape is None:
            child_shape = shape
        elif shape != child_shape:
            raise ExportRuntimeEvidenceError(f"{path} is ragged")
        flattened.extend(numbers)
    assert child_shape is not None
    return (len(value), *child_shape), flattened


def compare_numeric_outputs(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
) -> NumericParity:
    abs_tolerance = _require_nonnegative_finite(atol, "atol")
    rel_tolerance = _require_nonnegative_finite(rtol, "rtol")
    reference_shape, reference_values = _numeric_shape_and_values(reference, "reference")
    candidate_shape, candidate_values = _numeric_shape_and_values(candidate, "candidate")
    if reference_shape != candidate_shape:
        raise ExportRuntimeEvidenceError(
            f"output shape mismatch: reference={reference_shape}, candidate={candidate_shape}"
        )

    mismatch_count = 0
    max_abs_error = 0.0
    max_allowed_error = 0.0
    for expected, observed in zip(reference_values, candidate_values, strict=True):
        error = abs(expected - observed)
        allowed = abs_tolerance + rel_tolerance * abs(expected)
        max_abs_error = max(max_abs_error, error)
        max_allowed_error = max(max_allowed_error, allowed)
        if error > allowed:
            mismatch_count += 1
    return NumericParity(
        shape=reference_shape,
        element_count=len(reference_values),
        mismatch_count=mismatch_count,
        max_abs_error=max_abs_error,
        max_allowed_error=max_allowed_error,
        passed=mismatch_count == 0,
    )


def _target_by_id(manifest: Mapping[str, Any], target_id: str) -> Mapping[str, Any]:
    validate_target_manifest(manifest)
    for target in manifest["targets"]:
        if target["id"] == target_id:
            return target
    raise ExportRuntimeEvidenceError(f"unknown target id: {target_id}")


def assess_parity_evidence(
    manifest: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "target_id",
        "candidate_upstream_commit_sha",
        "candidate_release_tag",
        "backend_execution",
        "project_git_sha",
        "model_identity",
        "input_identity",
        "execution_environment",
        "reference_backend",
        "reference_output",
        "candidate_output",
        "atol",
        "rtol",
    }
    if set(evidence) != expected_fields:
        raise ExportRuntimeEvidenceError("evidence fields do not match the v1 schema")
    if evidence["schema"] != EVIDENCE_SCHEMA:
        raise ExportRuntimeEvidenceError("unsupported evidence schema")

    target_id = _require_string(evidence["target_id"], "evidence.target_id")
    target = _target_by_id(manifest, target_id)
    if evidence["candidate_upstream_commit_sha"] != target["upstream_commit_sha"]:
        raise ExportRuntimeEvidenceError("candidate upstream commit does not match target pin")
    if evidence["candidate_release_tag"] != target["release_tag"]:
        raise ExportRuntimeEvidenceError("candidate release tag does not match target pin")
    if not _require_bool(evidence["backend_execution"], "backend_execution"):
        raise ExportRuntimeEvidenceError("backend_execution=false cannot prove parity")
    _require_hex(evidence["project_git_sha"], "project_git_sha", _HEX40)
    _require_hex(evidence["model_identity"], "model_identity", _HEX64)
    _require_hex(evidence["input_identity"], "input_identity", _HEX64)

    environment = _require_mapping(evidence["execution_environment"], "execution_environment")
    if set(environment) != {"python", "platform", "backend_version", "backend_device"}:
        raise ExportRuntimeEvidenceError("execution environment fields do not match the v1 schema")
    _require_string(environment["python"], "execution_environment.python")
    _require_string(environment["platform"], "execution_environment.platform")
    _require_string(environment["backend_device"], "execution_environment.backend_device")
    if environment["backend_version"] != target["release_tag"]:
        raise ExportRuntimeEvidenceError(
            "executed backend version does not match the target release"
        )
    if evidence["reference_backend"] != "twelve_six_first_party":
        raise ExportRuntimeEvidenceError("reference backend must be the first-party 12-6 path")

    parity = compare_numeric_outputs(
        evidence["reference_output"],
        evidence["candidate_output"],
        atol=evidence["atol"],
        rtol=evidence["rtol"],
    )
    report = {
        "schema": REPORT_SCHEMA,
        "target_id": target_id,
        "target_commit_sha": target["upstream_commit_sha"],
        "project_git_sha": evidence["project_git_sha"],
        "model_identity": evidence["model_identity"],
        "input_identity": evidence["input_identity"],
        "execution_environment": dict(environment),
        "reference_backend": evidence["reference_backend"],
        "reference_output_identity": semantic_sha256(evidence["reference_output"]),
        "candidate_output_identity": semantic_sha256(evidence["candidate_output"]),
        "atol": _require_nonnegative_finite(evidence["atol"], "atol"),
        "rtol": _require_nonnegative_finite(evidence["rtol"], "rtol"),
        "numeric_parity": parity.as_dict(),
        "derived_state": "PARITY_PROVEN" if parity.passed else "CANDIDATE",
        "adopted": False,
    }
    report["evidence_identity"] = semantic_sha256(report)
    return report


def derive_adoption_state(report: Mapping[str, Any], adoption_authority: str | None) -> str:
    if report.get("schema") != REPORT_SCHEMA:
        raise ExportRuntimeEvidenceError("unsupported parity report schema")
    if report.get("derived_state") != "PARITY_PROVEN":
        if adoption_authority is not None:
            raise ExportRuntimeEvidenceError("cannot adopt a target without parity proof")
        return "CANDIDATE"
    if adoption_authority is None:
        return "PARITY_PROVEN"
    _require_string(adoption_authority, "adoption_authority")
    return "ADOPTED"
