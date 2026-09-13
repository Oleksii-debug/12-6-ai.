"""Terminal fail-closed envelope for DATA-526 V8 reserved decontamination.

This module is deliberately thin: contamination matching remains owned by the
incumbent DATA-232/#857 implementation and record binding remains owned by #874.
The envelope adds only terminal scientific authority checks that must not be
optional at the canonical execution surface: independently expected reserved-set
membership identities, conservative cross-source-family quarantine, exact source
Git-head binding, and report/evidence coupling.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data.data526_v8_reserved_decontamination_v1 import (
    execute_data526_v8_reserved_decontamination,
    verify_data526_v8_execution_evidence,
)
from twelve_six.data.decontamination_authority_v2 import verify_report

TERMINAL_EXECUTION_SCHEMA = "12-6.data526-v8-reserved-decontamination-terminal.v1"
TERMINAL_EXECUTION_SCOPE = "FINAL_RECORD_GRAPH"
_REQUIRED_ROLES = {"selection_validation", "final_test"}


class Data526V8TerminalDecontaminationError(RuntimeError):
    """Raised when terminal decontamination authority cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Data526V8TerminalDecontaminationError(message)


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
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def _require_git_sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
        f"{label} must be lowercase Git SHA-1",
    )
    return str(value)


def _verify_reserved_memberships(
    reserved_payload_binding: Mapping[str, Any],
    *,
    expected_selection_membership_identity_sha256: str,
    expected_final_membership_identity_sha256: str,
) -> tuple[str, str]:
    raw_sets = reserved_payload_binding.get("reserved_sets")
    _require(
        isinstance(raw_sets, Sequence) and not isinstance(raw_sets, (str, bytes)),
        "reserved_sets must be a sequence",
    )
    memberships: dict[str, str] = {}
    for index, raw_set in enumerate(raw_sets):
        _require(isinstance(raw_set, Mapping), f"reserved_set[{index}] must be an object")
        role = raw_set.get("role")
        if role not in _REQUIRED_ROLES:
            continue
        _require(role not in memberships, f"duplicate reserved role: {role}")
        memberships[str(role)] = _require_sha256(
            raw_set.get("source_membership_identity_sha256"),
            f"reserved_set[{index}].source_membership_identity_sha256",
        )
    _require(_REQUIRED_ROLES <= set(memberships), "required reserved memberships are missing")
    expected_selection = _require_sha256(
        expected_selection_membership_identity_sha256,
        "expected_selection_membership_identity_sha256",
    )
    expected_final = _require_sha256(
        expected_final_membership_identity_sha256,
        "expected_final_membership_identity_sha256",
    )
    _require(
        memberships["selection_validation"] == expected_selection,
        "selection-validation membership identity drift",
    )
    _require(
        memberships["final_test"] == expected_final,
        "final-test membership identity drift",
    )
    return expected_selection, expected_final


def execute_terminal_data526_v8_reserved_decontamination(
    raw_training_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    record_inventory: Mapping[str, Any],
    materialization_evidence: Mapping[str, Any],
    reserved_payload_binding: Mapping[str, Any],
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    expected_materialization_evidence_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    expected_reserved_binding_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
    expected_selection_membership_identity_sha256: str,
    expected_final_membership_identity_sha256: str,
    decontamination_implementation_git_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the canonical terminal record-level path with no safety bypass."""
    selection_membership, final_membership = _verify_reserved_memberships(
        reserved_payload_binding,
        expected_selection_membership_identity_sha256=(
            expected_selection_membership_identity_sha256
        ),
        expected_final_membership_identity_sha256=expected_final_membership_identity_sha256,
    )
    implementation_sha = _require_git_sha(
        decontamination_implementation_git_sha,
        "decontamination_implementation_git_sha",
    )

    report, inner = execute_data526_v8_reserved_decontamination(
        raw_training_records,
        evaluation_records,
        record_inventory=record_inventory,
        materialization_evidence=materialization_evidence,
        reserved_payload_binding=reserved_payload_binding,
        expected_record_inventory_digest_sha256=expected_record_inventory_digest_sha256,
        expected_payload_inventory_digest_sha256=expected_payload_inventory_digest_sha256,
        expected_materialization_evidence_identity_sha256=(
            expected_materialization_evidence_identity_sha256
        ),
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
        expected_reserved_binding_identity_sha256=expected_reserved_binding_identity_sha256,
        expected_selection_validation_identity_sha256=(
            expected_selection_validation_identity_sha256
        ),
        expected_final_test_identity_sha256=expected_final_test_identity_sha256,
        quarantine_cross_source_families=True,
    )
    verify_data526_v8_execution_evidence(inner)
    verify_report(report)
    _require(
        inner.get("decontamination_report_sha256") == report.get("report_sha256"),
        "inner execution/report identity mismatch",
    )
    _require(inner.get("status") == report.get("status"), "inner execution/report status mismatch")
    _require(inner.get("counts") == report.get("counts"), "inner execution/report counts mismatch")

    core: dict[str, Any] = {
        "schema_version": TERMINAL_EXECUTION_SCHEMA,
        "status": report["status"],
        "execution_scope": TERMINAL_EXECUTION_SCOPE,
        "decontamination_implementation_git_sha": implementation_sha,
        "inner_execution_identity_sha256": inner["execution_identity_sha256"],
        "decontamination_report_sha256": report["report_sha256"],
        "selection_validation_membership_identity_sha256": selection_membership,
        "final_test_membership_identity_sha256": final_membership,
        "data526_record_inventory_digest_sha256": inner[
            "data526_record_inventory_digest_sha256"
        ],
        "data526_payload_inventory_digest_sha256": inner[
            "data526_payload_inventory_digest_sha256"
        ],
        "data526_materialization_evidence_identity_sha256": inner[
            "data526_materialization_evidence_identity_sha256"
        ],
        "v8_survivor_authority_sha256": inner["v8_survivor_authority_sha256"],
        "counts": deepcopy(report["counts"]),
        "cross_source_family_quarantine_enforced": True,
        "durable_evidence_hash_only": True,
        "final_test_payload_accessed_for_decontamination": True,
        "final_test_outcomes_read": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    core["execution_identity_sha256"] = _sha256(_canonical_bytes(core))
    return report, core


def verify_terminal_data526_v8_execution_evidence(
    evidence: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    expected_decontamination_implementation_git_sha: str,
    expected_selection_membership_identity_sha256: str,
    expected_final_membership_identity_sha256: str,
) -> None:
    """Verify terminal outer evidence against independent expectations and report."""
    _require(
        evidence.get("schema_version") == TERMINAL_EXECUTION_SCHEMA,
        "terminal execution schema drift",
    )
    claimed = _require_sha256(evidence.get("execution_identity_sha256"), "execution_identity_sha256")
    body = deepcopy(dict(evidence))
    body.pop("execution_identity_sha256", None)
    _require(_sha256(_canonical_bytes(body)) == claimed, "terminal execution evidence hash drift")

    verify_report(report)
    _require(
        evidence.get("decontamination_report_sha256") == report.get("report_sha256"),
        "terminal execution/report identity mismatch",
    )
    _require(evidence.get("status") == report.get("status"), "terminal execution/report status mismatch")
    _require(evidence.get("counts") == report.get("counts"), "terminal execution/report counts mismatch")

    implementation_sha = _require_git_sha(
        evidence.get("decontamination_implementation_git_sha"),
        "decontamination_implementation_git_sha",
    )
    _require(
        implementation_sha
        == _require_git_sha(
            expected_decontamination_implementation_git_sha,
            "expected_decontamination_implementation_git_sha",
        ),
        "decontamination implementation Git head is not independently expected",
    )
    _require(
        evidence.get("selection_validation_membership_identity_sha256")
        == _require_sha256(
            expected_selection_membership_identity_sha256,
            "expected_selection_membership_identity_sha256",
        ),
        "terminal selection-validation membership identity drift",
    )
    _require(
        evidence.get("final_test_membership_identity_sha256")
        == _require_sha256(
            expected_final_membership_identity_sha256,
            "expected_final_membership_identity_sha256",
        ),
        "terminal final-test membership identity drift",
    )

    for key in (
        "inner_execution_identity_sha256",
        "decontamination_report_sha256",
        "data526_record_inventory_digest_sha256",
        "data526_payload_inventory_digest_sha256",
        "data526_materialization_evidence_identity_sha256",
        "v8_survivor_authority_sha256",
    ):
        _require_sha256(evidence.get(key), key)
    _require(evidence.get("execution_scope") == TERMINAL_EXECUTION_SCOPE, "terminal execution scope drift")
    _require(
        evidence.get("cross_source_family_quarantine_enforced") is True,
        "terminal cross-source-family quarantine was disabled",
    )
    _require(evidence.get("durable_evidence_hash_only") is True, "durable text boundary weakened")
    _require(
        evidence.get("final_test_payload_accessed_for_decontamination") is True,
        "final-test decontamination payload access truth was erased",
    )
    _require(evidence.get("final_test_outcomes_read") is False, "final-test outcomes were read")
    _require(evidence.get("authorized_training_exposure") == 0, "training exposure fabricated")
    for key in ("tokenizer_fit_authorized", "training_executed", "paid_compute_used"):
        _require(evidence.get(key) is False, f"terminal execution boundary weakened: {key}")
