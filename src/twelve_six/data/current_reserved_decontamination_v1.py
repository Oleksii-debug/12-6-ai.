"""Bind current retained training rows and reserved evaluation payloads to DATA-232.

This module is an execution adapter only. It does not reconstruct the retained corpus,
resolve evaluation payloads, or implement contamination matching. Callers must provide
already-verified ephemeral payload rows plus independent exact identities from the
canonical upstream authorities. Durable outputs remain text-free.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data.decontamination_authority_v2 import build_report, verify_report

EXECUTION_SCHEMA = "12-6.current-reserved-decontamination-execution.v1"
RESERVED_BINDING_SCHEMA = "12-6.reserved-evaluation-payload-binding.v1"
TRAINING_HANDOFF_SCHEMA = "12-6.postdedup-decontam-handoff.v1"
_ALLOWED_ROLES = {"selection_validation", "final_test", "auxiliary_reserved"}
_REQUIRED_ROLES = {"selection_validation", "final_test"}
_ALLOWED_MODALITIES = {"uk", "ua", "en", "code", "text"}


class CurrentDecontaminationExecutionError(RuntimeError):
    """Fail-closed current-graph decontamination execution error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentDecontaminationExecutionError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def _require_git_sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
        f"{label} must be lowercase Git SHA",
    )
    return str(value)


def _require_nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def _record_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        _require(isinstance(raw, Mapping), f"record[{index}] must be an object")
        values: dict[str, str] = {}
        for key in ("record_id", "source_id", "source_family", "modality", "text"):
            value = raw.get(key)
            _require(
                isinstance(value, str) and bool(value),
                f"record[{index}].{key} must be non-empty text",
            )
            values[key] = value
        modality = values["modality"].lower()
        _require(modality in _ALLOWED_MODALITIES, f"unsupported modality: {modality}")
        record_id = values["record_id"]
        _require(record_id not in seen, f"duplicate record_id: {record_id}")
        seen.add(record_id)
        text_bytes = values["text"].encode("utf-8")
        projected.append(
            {
                "record_id": record_id,
                "source_id": values["source_id"],
                "source_family": values["source_family"],
                "modality": modality,
                "text_sha256": _sha256_bytes(text_bytes),
                "text_utf8_bytes": len(text_bytes),
            }
        )
    return sorted(projected, key=lambda row: row["record_id"])


def _verify_training_handoff(
    training_records: Sequence[Mapping[str, Any]],
    handoff: Mapping[str, Any],
    *,
    expected_inventory_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    expected_handoff_identity_sha256: str,
) -> tuple[str, str, str]:
    _require(isinstance(handoff, Mapping), "training handoff must be an object")
    _require(
        handoff.get("schema_version") == TRAINING_HANDOFF_SCHEMA,
        "training handoff schema drift",
    )
    inventory = _require_sha256(
        handoff.get("postdedup_inventory_identity_sha256"),
        "postdedup_inventory_identity_sha256",
    )
    survivor = _require_sha256(
        handoff.get("input_survivor_authority_sha256"),
        "input_survivor_authority_sha256",
    )
    _require(
        inventory
        == _require_sha256(
            expected_inventory_identity_sha256,
            "expected_inventory_identity_sha256",
        ),
        "training handoff inventory identity is not independently expected",
    )
    _require(
        survivor
        == _require_sha256(
            expected_survivor_authority_sha256,
            "expected_survivor_authority_sha256",
        ),
        "training handoff survivor identity is not independently expected",
    )
    _require(
        handoff.get("raw_text_persisted_in_evidence") is False,
        "training handoff persisted raw text",
    )
    _require(
        handoff.get("final_test_payload_accessed") is False,
        "training handoff accessed final-test payload before decontamination",
    )
    _require(
        handoff.get("final_test_outcomes_accessed") is False,
        "training handoff accessed final-test outcomes",
    )
    _require(
        handoff.get("authorized_training_exposure") == 0,
        "training handoff grants training exposure",
    )
    claimed = _require_sha256(
        handoff.get("handoff_identity_sha256"),
        "handoff_identity_sha256",
    )
    _require(
        claimed
        == _require_sha256(
            expected_handoff_identity_sha256,
            "expected_handoff_identity_sha256",
        ),
        "training handoff identity is not independently expected",
    )
    body = deepcopy(dict(handoff))
    body.pop("handoff_identity_sha256", None)
    _require(
        _sha256_bytes(_canonical_bytes(body)) == claimed,
        "training handoff self-hash mismatch",
    )

    projection = _record_projection(training_records)
    retained_count = _require_nonnegative_int(
        handoff.get("retained_source_count"),
        "retained_source_count",
    )
    _require(
        len(projection) == retained_count,
        "training row count does not match retained-source authority",
    )
    _require(
        handoff.get("matcher_input_projection") == projection,
        "training rows do not reproduce the retained matcher projection",
    )
    _require(
        handoff.get("matcher_input_projection_sha256")
        == _sha256_bytes(_canonical_bytes(projection)),
        "training matcher projection identity mismatch",
    )
    return inventory, survivor, claimed


def build_reserved_payload_binding(
    reserved_sets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create deterministic text-free binding metadata for reserved payloads.

    The returned self-hash is not sufficient external authority. Execution also
    requires the caller to supply the independently expected binding identity.
    """
    normalized_sets: list[dict[str, Any]] = []
    roles: set[str] = set()
    authority_ids: set[str] = set()
    all_record_ids: set[str] = set()
    for set_index, raw_set in enumerate(reserved_sets):
        _require(
            isinstance(raw_set, Mapping),
            f"reserved_set[{set_index}] must be an object",
        )
        authority_id = raw_set.get("authority_id")
        role = raw_set.get("role")
        _require(
            isinstance(authority_id, str) and bool(authority_id),
            f"reserved_set[{set_index}].authority_id must be non-empty text",
        )
        _require(
            isinstance(role, str) and role in _ALLOWED_ROLES,
            f"reserved_set[{set_index}].role is unsupported",
        )
        _require(
            authority_id not in authority_ids,
            "reserved authority_id must be unique",
        )
        authority_ids.add(authority_id)
        roles.add(role)

        members = raw_set.get("members")
        _require(
            isinstance(members, Sequence) and not isinstance(members, (str, bytes)),
            f"reserved_set[{set_index}].members must be a sequence",
        )
        normalized_members: list[dict[str, Any]] = []
        for member_index, member in enumerate(members):
            _require(
                isinstance(member, Mapping),
                f"reserved_set[{set_index}].members[{member_index}] must be an object",
            )
            texts: dict[str, str] = {}
            for key in ("record_id", "source_id", "source_family", "modality"):
                value = member.get(key)
                _require(
                    isinstance(value, str) and bool(value),
                    f"reserved member {key} must be non-empty text",
                )
                texts[key] = value
            modality = texts["modality"].lower()
            _require(
                modality in _ALLOWED_MODALITIES,
                f"unsupported modality: {modality}",
            )
            record_id = texts["record_id"]
            _require(
                record_id not in all_record_ids,
                "reserved record_id must be globally unique",
            )
            all_record_ids.add(record_id)
            _require(
                member.get("training_prohibited") is True,
                f"reserved record is not prohibited from training: {record_id}",
            )
            _require(
                member.get("outcomes_included") is False,
                f"outcome-bearing reserved payload is forbidden: {record_id}",
            )
            normalized_members.append(
                {
                    "record_id": record_id,
                    "source_id": texts["source_id"],
                    "source_family": texts["source_family"],
                    "modality": modality,
                    "content_sha256": _require_sha256(
                        member.get("content_sha256"),
                        f"reserved member content_sha256: {record_id}",
                    ),
                    "utf8_bytes": _require_nonnegative_int(
                        member.get("utf8_bytes"),
                        f"reserved member utf8_bytes: {record_id}",
                    ),
                    "training_prohibited": True,
                    "outcomes_included": False,
                }
            )
        _require(
            bool(normalized_members),
            f"reserved authority has no payload members: {authority_id}",
        )
        normalized_members.sort(key=lambda row: row["record_id"])
        normalized_sets.append(
            {
                "authority_id": authority_id,
                "identity_sha256": _require_sha256(
                    raw_set.get("identity_sha256"),
                    f"reserved_set[{set_index}].identity_sha256",
                ),
                "role": role,
                "source_sha": _require_git_sha(
                    raw_set.get("source_sha"),
                    f"reserved_set[{set_index}].source_sha",
                ),
                "source_membership_identity_sha256": _require_sha256(
                    raw_set.get("source_membership_identity_sha256"),
                    f"reserved_set[{set_index}].source_membership_identity_sha256",
                ),
                "members": normalized_members,
            }
        )
    _require(
        _REQUIRED_ROLES <= roles,
        "selection-validation and final-test payload sets are required",
    )
    normalized_sets.sort(key=lambda row: row["authority_id"])
    core: dict[str, Any] = {
        "schema_version": RESERVED_BINDING_SCHEMA,
        "reserved_sets": normalized_sets,
        "raw_text_persisted": False,
        "outcomes_included": False,
    }
    core["binding_identity_sha256"] = _sha256_bytes(_canonical_bytes(core))
    return core


def _verify_reserved_payloads(
    evaluation_records: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    *,
    expected_binding_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
) -> tuple[dict[str, Any], str, str, int]:
    _require(
        isinstance(binding, Mapping),
        "reserved payload binding must be an object",
    )
    _require(
        binding.get("schema_version") == RESERVED_BINDING_SCHEMA,
        "reserved payload binding schema drift",
    )
    claimed = _require_sha256(
        binding.get("binding_identity_sha256"),
        "binding_identity_sha256",
    )
    _require(
        claimed
        == _require_sha256(
            expected_binding_identity_sha256,
            "expected_binding_identity_sha256",
        ),
        "reserved payload binding is not independently expected",
    )
    body = deepcopy(dict(binding))
    body.pop("binding_identity_sha256", None)
    _require(
        _sha256_bytes(_canonical_bytes(body)) == claimed,
        "reserved payload binding self-hash mismatch",
    )
    _require(
        binding.get("raw_text_persisted") is False,
        "reserved binding persisted raw text",
    )
    _require(
        binding.get("outcomes_included") is False,
        "reserved binding contains outcomes",
    )

    raw_sets = binding.get("reserved_sets")
    _require(
        isinstance(raw_sets, Sequence) and not isinstance(raw_sets, (str, bytes)),
        "reserved_sets must be a sequence",
    )
    authorities: list[dict[str, str]] = []
    membership: dict[str, dict[str, Any]] = {}
    roles: dict[str, str] = {}
    for raw_set in raw_sets:
        _require(isinstance(raw_set, Mapping), "reserved set must be an object")
        authority_id = raw_set.get("authority_id")
        role = raw_set.get("role")
        _require(
            isinstance(authority_id, str) and bool(authority_id),
            "reserved authority_id invalid",
        )
        _require(
            isinstance(role, str) and role in _ALLOWED_ROLES,
            "reserved role invalid",
        )
        _require(role not in roles, f"duplicate reserved role: {role}")
        identity = _require_sha256(
            raw_set.get("identity_sha256"),
            "reserved identity_sha256",
        )
        source_sha = _require_git_sha(raw_set.get("source_sha"), "reserved source_sha")
        _require_sha256(
            raw_set.get("source_membership_identity_sha256"),
            "source_membership_identity_sha256",
        )
        authorities.append(
            {
                "authority_id": authority_id,
                "identity_sha256": identity,
                "role": role,
                "source_sha": source_sha,
            }
        )
        roles[role] = identity
        members = raw_set.get("members")
        _require(
            isinstance(members, Sequence) and not isinstance(members, (str, bytes)),
            "reserved members must be a sequence",
        )
        for member in members:
            _require(isinstance(member, Mapping), "reserved member must be an object")
            record_id = member.get("record_id")
            _require(
                isinstance(record_id, str) and bool(record_id),
                "reserved record_id invalid",
            )
            _require(record_id not in membership, "reserved record_id is duplicated")
            _require(
                member.get("training_prohibited") is True,
                "reserved training boundary weakened",
            )
            _require(
                member.get("outcomes_included") is False,
                "reserved outcome boundary weakened",
            )
            membership[record_id] = dict(member)

    _require(_REQUIRED_ROLES <= set(roles), "required reserved roles are missing")
    selection_identity = _require_sha256(
        expected_selection_validation_identity_sha256,
        "expected_selection_validation_identity_sha256",
    )
    final_identity = _require_sha256(
        expected_final_test_identity_sha256,
        "expected_final_test_identity_sha256",
    )
    _require(
        roles.get("selection_validation") == selection_identity,
        "selection-validation authority identity drift",
    )
    _require(
        roles.get("final_test") == final_identity,
        "final-test authority identity drift",
    )

    projection = _record_projection(evaluation_records)
    _require(
        {row["record_id"] for row in projection} == set(membership),
        "resolved evaluation payload coverage differs from reserved membership",
    )
    for row in projection:
        member = membership[row["record_id"]]
        for key in ("source_id", "source_family"):
            _require(
                row[key] == member.get(key),
                f"reserved payload {key} drift: {row['record_id']}",
            )
        _require(
            row["modality"] == str(member.get("modality", "")).lower(),
            f"reserved payload modality drift: {row['record_id']}",
        )
        _require(
            row["text_sha256"] == member.get("content_sha256"),
            f"reserved payload content SHA-256 drift: {row['record_id']}",
        )
        _require(
            row["text_utf8_bytes"] == member.get("utf8_bytes"),
            f"reserved payload byte-count drift: {row['record_id']}",
        )

    authority_metadata = {
        "schema": "12-6.current-reserved-authorities.v1",
        "authorities": sorted(authorities, key=lambda row: row["authority_id"]),
    }
    final_record_count = sum(
        len(raw_set["members"])
        for raw_set in raw_sets
        if raw_set.get("role") == "final_test"
    )
    return authority_metadata, selection_identity, final_identity, final_record_count


def execute_reserved_decontamination(
    training_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    training_handoff_evidence: Mapping[str, Any],
    reserved_payload_binding: Mapping[str, Any],
    expected_inventory_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    expected_training_handoff_identity_sha256: str,
    expected_reserved_binding_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
    quarantine_cross_source_families: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run incumbent DATA-232 matching over externally bound current payloads."""
    inventory, survivor, handoff_identity = _verify_training_handoff(
        training_records,
        training_handoff_evidence,
        expected_inventory_identity_sha256=expected_inventory_identity_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
        expected_handoff_identity_sha256=expected_training_handoff_identity_sha256,
    )
    authorities, selection_identity, final_identity, final_record_count = (
        _verify_reserved_payloads(
            evaluation_records,
            reserved_payload_binding,
            expected_binding_identity_sha256=(
                expected_reserved_binding_identity_sha256
            ),
            expected_selection_validation_identity_sha256=(
                expected_selection_validation_identity_sha256
            ),
            expected_final_test_identity_sha256=expected_final_test_identity_sha256,
        )
    )
    _require(
        final_record_count > 0,
        "final-test payload set must be non-empty for decontamination",
    )

    report = build_report(
        training_records,
        evaluation_records,
        training_corpus_identity=inventory,
        selection_validation_identity=selection_identity,
        final_test_identity=final_identity,
        authorities=authorities,
        quarantine_cross_source_families=quarantine_cross_source_families,
    )
    verify_report(report)
    _require(
        report.get("training_corpus_identity") == inventory,
        "DATA-232 report training identity drift",
    )
    _require(
        report.get("selection_validation_identity") == selection_identity,
        "DATA-232 report selection identity drift",
    )
    _require(
        report.get("final_test_identity") == final_identity,
        "DATA-232 report final-test identity drift",
    )
    _require(
        report.get("hash_only_evidence") is True,
        "DATA-232 report is not hash-only",
    )
    _require(
        report.get("final_test_outcomes_read") is False,
        "DATA-232 report accessed final-test outcomes",
    )
    _require(
        report.get("model_architecture_or_hyperparameters_selected") is False,
        "decontamination influenced model selection",
    )
    _require(
        report.get("training_executed") is False,
        "decontamination executed training",
    )
    _require(
        report.get("local_free_only") is True,
        "decontamination left LOCAL_FREE boundary",
    )

    core: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "status": report["status"],
        "training_corpus_identity_sha256": inventory,
        "input_survivor_authority_sha256": survivor,
        "training_handoff_identity_sha256": handoff_identity,
        "reserved_payload_binding_identity_sha256": reserved_payload_binding[
            "binding_identity_sha256"
        ],
        "selection_validation_identity_sha256": selection_identity,
        "final_test_identity_sha256": final_identity,
        "decontamination_report_sha256": report["report_sha256"],
        "counts": deepcopy(report["counts"]),
        "final_test_payload_accessed_for_decontamination": True,
        "final_test_outcomes_read": False,
        "durable_evidence_hash_only": True,
        "model_architecture_or_hyperparameters_selected": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    core["execution_identity_sha256"] = _sha256_bytes(_canonical_bytes(core))
    return report, core


def verify_execution_evidence(
    evidence: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    _require(
        evidence.get("schema_version") == EXECUTION_SCHEMA,
        "execution schema drift",
    )
    claimed = _require_sha256(
        evidence.get("execution_identity_sha256"),
        "execution_identity_sha256",
    )
    body = deepcopy(dict(evidence))
    body.pop("execution_identity_sha256", None)
    _require(
        _sha256_bytes(_canonical_bytes(body)) == claimed,
        "execution evidence hash drift",
    )
    verify_report(report)
    _require(
        evidence.get("decontamination_report_sha256") == report.get("report_sha256"),
        "execution/report identity mismatch",
    )
    _require(
        evidence.get("durable_evidence_hash_only") is True,
        "durable text boundary weakened",
    )
    _require(
        evidence.get("final_test_payload_accessed_for_decontamination") is True,
        "final-test decontamination payload access truth was erased",
    )
    _require(
        evidence.get("final_test_outcomes_read") is False,
        "final-test outcomes were read",
    )
    _require(
        evidence.get("authorized_training_exposure") == 0,
        "training exposure fabricated",
    )
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "paid_compute_used",
        "model_architecture_or_hyperparameters_selected",
    ):
        _require(
            evidence.get(key) is False,
            f"execution boundary weakened: {key}",
        )
