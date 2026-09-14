from __future__ import annotations

import copy
import hashlib
import json

import pytest

import twelve_six.data.data526_v8_reserved_decontamination_terminal_v1 as terminal
from twelve_six.data.data526_v8_reserved_decontamination_terminal_v1 import (
    Data526V8TerminalDecontaminationError,
    execute_terminal_data526_v8_reserved_decontamination,
    verify_terminal_data526_v8_execution_evidence,
)

SELECTION_MEMBERSHIP = "1" * 64
FINAL_MEMBERSHIP = "2" * 64
IMPLEMENTATION_SHA = "3" * 40


def _binding(selection_membership: str = SELECTION_MEMBERSHIP) -> dict[str, object]:
    return {
        "reserved_sets": [
            {
                "role": "selection_validation",
                "source_membership_identity_sha256": selection_membership,
            },
            {
                "role": "final_test",
                "source_membership_identity_sha256": FINAL_MEMBERSHIP,
            },
        ]
    }


def _fake_outputs() -> tuple[dict[str, object], dict[str, object]]:
    report = {
        "status": "PASS_CLEAN",
        "report_sha256": "4" * 64,
        "counts": {"input_training_records": 2, "excluded_training_records": 0},
    }
    inner = {
        "status": "PASS_CLEAN",
        "execution_identity_sha256": "5" * 64,
        "decontamination_report_sha256": report["report_sha256"],
        "data526_record_inventory_digest_sha256": "6" * 64,
        "data526_payload_inventory_digest_sha256": "7" * 64,
        "data526_materialization_evidence_identity_sha256": "8" * 64,
        "v8_survivor_authority_sha256": "9" * 64,
        "counts": copy.deepcopy(report["counts"]),
    }
    return report, inner


def _execute(monkeypatch: pytest.MonkeyPatch, binding: dict[str, object] | None = None):
    delegated: dict[str, object] = {}
    report, inner = _fake_outputs()

    def fake_execute(*args, **kwargs):
        delegated["args"] = args
        delegated["kwargs"] = kwargs
        return copy.deepcopy(report), copy.deepcopy(inner)

    monkeypatch.setattr(terminal, "execute_data526_v8_reserved_decontamination", fake_execute)
    monkeypatch.setattr(terminal, "verify_data526_v8_execution_evidence", lambda *_: None)
    monkeypatch.setattr(terminal, "verify_report", lambda *_: None)
    actual_binding = binding or _binding()
    observed_report, evidence = execute_terminal_data526_v8_reserved_decontamination(
        [],
        [],
        record_inventory={},
        materialization_evidence={},
        reserved_payload_binding=actual_binding,
        expected_record_inventory_digest_sha256="a" * 64,
        expected_payload_inventory_digest_sha256="b" * 64,
        expected_materialization_evidence_identity_sha256="c" * 64,
        expected_survivor_authority_sha256="d" * 64,
        expected_reserved_binding_identity_sha256="e" * 64,
        expected_selection_validation_identity_sha256="f" * 64,
        expected_final_test_identity_sha256="0" * 64,
        expected_selection_membership_identity_sha256=SELECTION_MEMBERSHIP,
        expected_final_membership_identity_sha256=FINAL_MEMBERSHIP,
        decontamination_implementation_git_sha=IMPLEMENTATION_SHA,
    )
    return observed_report, evidence, delegated


def _rehash(evidence: dict[str, object]) -> None:
    body = copy.deepcopy(evidence)
    body.pop("execution_identity_sha256", None)
    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    evidence["execution_identity_sha256"] = hashlib.sha256(raw).hexdigest()


def test_terminal_path_forces_quarantine_and_binds_external_memberships(monkeypatch: pytest.MonkeyPatch):
    report, evidence, delegated = _execute(monkeypatch)
    kwargs = delegated["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["quarantine_cross_source_families"] is True
    assert evidence["selection_validation_membership_identity_sha256"] == SELECTION_MEMBERSHIP
    assert evidence["final_test_membership_identity_sha256"] == FINAL_MEMBERSHIP
    assert evidence["decontamination_implementation_git_sha"] == IMPLEMENTATION_SHA
    assert evidence["execution_scope"] == "FINAL_RECORD_GRAPH"
    assert evidence["cross_source_family_quarantine_enforced"] is True
    assert evidence["authorized_training_exposure"] == 0
    verify_terminal_data526_v8_execution_evidence(
        evidence,
        report,
        expected_decontamination_implementation_git_sha=IMPLEMENTATION_SHA,
        expected_selection_membership_identity_sha256=SELECTION_MEMBERSHIP,
        expected_final_membership_identity_sha256=FINAL_MEMBERSHIP,
    )


def test_rehashed_reserved_membership_substitution_fails_before_matching(monkeypatch: pytest.MonkeyPatch):
    called = False

    def fake_execute(*args, **kwargs):
        nonlocal called
        called = True
        return _fake_outputs()

    monkeypatch.setattr(terminal, "execute_data526_v8_reserved_decontamination", fake_execute)
    with pytest.raises(
        Data526V8TerminalDecontaminationError,
        match="selection-validation membership identity drift",
    ):
        execute_terminal_data526_v8_reserved_decontamination(
            [],
            [],
            record_inventory={},
            materialization_evidence={},
            reserved_payload_binding=_binding("f" * 64),
            expected_record_inventory_digest_sha256="a" * 64,
            expected_payload_inventory_digest_sha256="b" * 64,
            expected_materialization_evidence_identity_sha256="c" * 64,
            expected_survivor_authority_sha256="d" * 64,
            expected_reserved_binding_identity_sha256="e" * 64,
            expected_selection_validation_identity_sha256="f" * 64,
            expected_final_test_identity_sha256="0" * 64,
            expected_selection_membership_identity_sha256=SELECTION_MEMBERSHIP,
            expected_final_membership_identity_sha256=FINAL_MEMBERSHIP,
            decontamination_implementation_git_sha=IMPLEMENTATION_SHA,
        )
    assert called is False


def test_terminal_verifier_rejects_report_substitution_even_when_evidence_self_hash_is_valid(monkeypatch: pytest.MonkeyPatch):
    report, evidence, _ = _execute(monkeypatch)
    substituted_report = copy.deepcopy(report)
    substituted_report["report_sha256"] = "a" * 64
    with pytest.raises(
        Data526V8TerminalDecontaminationError,
        match="terminal execution/report identity mismatch",
    ):
        verify_terminal_data526_v8_execution_evidence(
            evidence,
            substituted_report,
            expected_decontamination_implementation_git_sha=IMPLEMENTATION_SHA,
            expected_selection_membership_identity_sha256=SELECTION_MEMBERSHIP,
            expected_final_membership_identity_sha256=FINAL_MEMBERSHIP,
        )


def test_rehashed_quarantine_disable_cannot_be_promoted(monkeypatch: pytest.MonkeyPatch):
    report, evidence, _ = _execute(monkeypatch)
    evidence["cross_source_family_quarantine_enforced"] = False
    _rehash(evidence)
    with pytest.raises(
        Data526V8TerminalDecontaminationError,
        match="quarantine was disabled",
    ):
        verify_terminal_data526_v8_execution_evidence(
            evidence,
            report,
            expected_decontamination_implementation_git_sha=IMPLEMENTATION_SHA,
            expected_selection_membership_identity_sha256=SELECTION_MEMBERSHIP,
            expected_final_membership_identity_sha256=FINAL_MEMBERSHIP,
        )


def test_rehashed_implementation_head_substitution_requires_external_rebinding(monkeypatch: pytest.MonkeyPatch):
    report, evidence, _ = _execute(monkeypatch)
    evidence["decontamination_implementation_git_sha"] = "a" * 40
    _rehash(evidence)
    with pytest.raises(
        Data526V8TerminalDecontaminationError,
        match="implementation Git head is not independently expected",
    ):
        verify_terminal_data526_v8_execution_evidence(
            evidence,
            report,
            expected_decontamination_implementation_git_sha=IMPLEMENTATION_SHA,
            expected_selection_membership_identity_sha256=SELECTION_MEMBERSHIP,
            expected_final_membership_identity_sha256=FINAL_MEMBERSHIP,
        )
