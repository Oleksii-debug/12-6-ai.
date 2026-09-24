"""Bind EVAL-647 as a mandatory auxiliary reserved set for future corpus scans.

This module does not authorize EVAL-647 for model selection. It only turns the two
sealed code objects into the incumbent DATA-232 ``auxiliary_reserved`` role so every
future reserved-decontamination execution can exclude exact/near copies before any
training authority is composed. Durable outputs are text-free.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.current_reserved_decontamination_v1 import (
    RESERVED_BINDING_SCHEMA,
    build_reserved_payload_binding,
)

MANIFEST_SCHEMA = "12-6.eval-code-reserve-v1.contract.v1"
MATERIALIZATION_SCHEMA = "12-6.eval-code-reserve-v1.source-materialization-terminal.v1"
AUXILIARY_AUTHORITY_ID = "eval647-code-selection-future-training-exclusion"
AUXILIARY_ROLE = "auxiliary_reserved"
RECEIPT_SCHEMA = "12-6.eval647-future-training-exclusion-consumption.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class Eval647FutureTrainingExclusionError(ValueError):
    """Raised when EVAL-647 cannot be bound to the canonical exclusion path."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Eval647FutureTrainingExclusionError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def _require_git_sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None,
        f"{label} must be lowercase Git SHA",
    )
    return str(value)


def _object_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    parts: list[str] = []
    for key in ("repository", "revision", "path"):
        item = value.get(key)
        _require(isinstance(item, str) and bool(item), f"EVAL-647 object {key} missing")
        parts.append(str(item))
    return parts[0], parts[1], parts[2]


def build_eval647_auxiliary_reserved_set(
    manifest: Mapping[str, Any],
    materialization_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the one canonical text-free DATA-232 auxiliary set for EVAL-647."""
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "EVAL-647 manifest schema drift")
    _require(manifest.get("issue") == 647, "EVAL-647 issue binding drift")
    _require(manifest.get("execution_class") == "LOCAL_FREE", "EVAL-647 execution class drift")
    _require(manifest.get("purpose") == "selection_validation_only", "EVAL-647 purpose drift")

    reservation = manifest.get("reservation")
    _require(isinstance(reservation, Mapping), "EVAL-647 reservation missing")
    for key in ("training_allowed", "tokenizer_fit_allowed"):
        _require(reservation.get(key) is False, f"EVAL-647 reservation widened: {key}")
    _require(
        reservation.get("permanent_future_training_exclusion") is True,
        "EVAL-647 permanent future-training exclusion missing",
    )
    _require(
        reservation.get("historical_training_exposure_required") == 0,
        "EVAL-647 historical training boundary widened",
    )
    _require(
        reservation.get("historical_tokenizer_fit_exposure_required") == 0,
        "EVAL-647 historical tokenizer boundary widened",
    )

    _require(
        materialization_evidence.get("schema_version") == MATERIALIZATION_SCHEMA,
        "EVAL-647 materialization schema drift",
    )
    _require(materialization_evidence.get("reservation_authority_issue") == 647, "EVAL-647 materialization issue drift")
    _require(materialization_evidence.get("execution_profile") == "LOCAL_FREE", "EVAL-647 materialization profile drift")
    _require(materialization_evidence.get("repeat_execution_byte_identical") is True, "EVAL-647 materialization was not repeat-identical")
    _require(materialization_evidence.get("raw_payload_persisted_in_repository") is False, "EVAL-647 raw payload persistence widened")
    _require(materialization_evidence.get("selection_validation_records_authorized") == 0, "EVAL-647 exclusion cannot authorize evaluation records")

    manifest_evidence = manifest.get("materialization_evidence")
    _require(isinstance(manifest_evidence, Mapping), "EVAL-647 manifest materialization binding missing")
    evidence_identity = _require_sha256(
        materialization_evidence.get("evidence_identity_sha256"),
        "EVAL-647 evidence_identity_sha256",
    )
    _require(
        manifest_evidence.get("identity_sha256") == evidence_identity,
        "EVAL-647 materialization evidence identity drift",
    )
    membership_identity = _require_sha256(
        materialization_evidence.get("object_set_identity_sha256"),
        "EVAL-647 object_set_identity_sha256",
    )
    discovery_head = _require_git_sha(
        materialization_evidence.get("discovery_head_sha"),
        "EVAL-647 discovery_head_sha",
    )

    manifest_objects = manifest.get("objects")
    sealed_objects = materialization_evidence.get("objects")
    _require(isinstance(manifest_objects, Sequence) and not isinstance(manifest_objects, (str, bytes)), "EVAL-647 manifest objects missing")
    _require(isinstance(sealed_objects, Sequence) and not isinstance(sealed_objects, (str, bytes)), "EVAL-647 sealed objects missing")
    _require(len(manifest_objects) == 2 == len(sealed_objects), "EVAL-647 must contain exactly two sealed objects")
    sealed_by_key = {
        _object_key(obj): obj
        for obj in sealed_objects
        if isinstance(obj, Mapping)
    }
    _require(len(sealed_by_key) == 2, "EVAL-647 sealed object identities are not unique")

    members: list[dict[str, Any]] = []
    for index, obj in enumerate(manifest_objects):
        _require(isinstance(obj, Mapping), f"EVAL-647 objects[{index}] must be an object")
        key = _object_key(obj)
        sealed = sealed_by_key.get(key)
        _require(isinstance(sealed, Mapping), f"EVAL-647 object is absent from terminal materialization: {key}")
        for flag in ("training_allowed", "tokenizer_fit_allowed"):
            _require(obj.get(flag) is False and sealed.get(flag) is False, f"EVAL-647 object boundary widened: {flag}")
        _require(
            obj.get("permanent_future_training_exclusion") is True
            and sealed.get("permanent_future_training_exclusion") is True,
            "EVAL-647 object future exclusion missing",
        )
        _require(obj.get("evaluation_use") == "selection_validation", "EVAL-647 evaluation use drift")
        for field in ("source_family", "repository", "revision", "path", "git_blob_sha1", "license_spdx", "raw_sha256"):
            _require(obj.get(field) == sealed.get(field), f"EVAL-647 sealed object {field} drift")
        _require(obj.get("expected_raw_bytes") == sealed.get("raw_bytes"), "EVAL-647 sealed object byte-count drift")
        raw_sha = _require_sha256(obj.get("raw_sha256"), f"EVAL-647 objects[{index}].raw_sha256")
        raw_bytes = obj.get("expected_raw_bytes")
        _require(isinstance(raw_bytes, int) and not isinstance(raw_bytes, bool) and raw_bytes > 0, "EVAL-647 raw byte count invalid")
        repository, revision, path = key
        members.append(
            {
                "record_id": f"eval647:{index}:{raw_sha}",
                "source_id": f"{repository}@{revision}:{path}",
                "source_family": str(obj["source_family"]),
                "modality": "code",
                "content_sha256": raw_sha,
                "utf8_bytes": raw_bytes,
                "training_prohibited": True,
                "outcomes_included": False,
            }
        )

    return {
        "authority_id": AUXILIARY_AUTHORITY_ID,
        "identity_sha256": evidence_identity,
        "role": AUXILIARY_ROLE,
        "source_sha": discovery_head,
        "source_membership_identity_sha256": membership_identity,
        "members": members,
    }


def compose_eval647_future_training_exclusion(
    base_reserved_binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    materialization_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose EVAL-647 into the incumbent reserved binding without authority elevation."""
    _require(base_reserved_binding.get("schema_version") == RESERVED_BINDING_SCHEMA, "base reserved binding schema drift")
    raw_sets = base_reserved_binding.get("reserved_sets")
    _require(isinstance(raw_sets, Sequence) and not isinstance(raw_sets, (str, bytes)), "base reserved sets missing")
    canonical_base = build_reserved_payload_binding(list(raw_sets))
    _require(canonical_base == dict(base_reserved_binding), "base reserved binding is not canonical")
    _require(
        all(
            not (isinstance(item, Mapping) and item.get("role") == AUXILIARY_ROLE)
            for item in raw_sets
        ),
        "base reserved binding already contains an auxiliary reserved set",
    )

    auxiliary = build_eval647_auxiliary_reserved_set(manifest, materialization_evidence)
    composed = build_reserved_payload_binding([*raw_sets, auxiliary])
    roles = {
        str(item["role"]): str(item["identity_sha256"])
        for item in composed["reserved_sets"]
    }
    _require(roles.get(AUXILIARY_ROLE) == auxiliary["identity_sha256"], "EVAL-647 auxiliary authority was not consumed")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "READY_FOR_CANONICAL_RESERVED_DECONTAMINATION_EXECUTION",
        "base_reserved_binding_identity_sha256": str(base_reserved_binding["binding_identity_sha256"]),
        "composed_reserved_binding_identity_sha256": str(composed["binding_identity_sha256"]),
        "eval647_materialization_evidence_identity_sha256": str(auxiliary["identity_sha256"]),
        "eval647_object_set_identity_sha256": str(auxiliary["source_membership_identity_sha256"]),
        "eval647_reserved_record_count": len(auxiliary["members"]),
        "eval647_role": AUXILIARY_ROLE,
        "future_training_exclusion_present_in_reserved_binding": True,
        "selection_validation_records_authorized": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "raw_payload_persisted": False,
    }
    receipt["receipt_identity_sha256"] = _sha256(_canonical_bytes(receipt))
    return composed, receipt


def require_eval647_future_training_exclusion(
    reserved_binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    materialization_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a reserved binding contains the exact sealed EVAL-647 set."""
    raw_sets = reserved_binding.get("reserved_sets")
    _require(isinstance(raw_sets, Sequence) and not isinstance(raw_sets, (str, bytes)), "reserved sets missing")
    canonical = build_reserved_payload_binding(list(raw_sets))
    _require(canonical == dict(reserved_binding), "reserved binding is not canonical")
    expected = build_eval647_auxiliary_reserved_set(manifest, materialization_evidence)
    candidates = [
        dict(item)
        for item in raw_sets
        if isinstance(item, Mapping) and item.get("role") == AUXILIARY_ROLE
    ]
    _require(len(candidates) == 1, "exactly one auxiliary reserved set is required")
    _require(candidates[0] == expected, "reserved binding does not contain the exact EVAL-647 future-training exclusion")
    return expected
