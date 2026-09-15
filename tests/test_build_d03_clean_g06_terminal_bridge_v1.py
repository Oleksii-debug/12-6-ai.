from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

_TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_d03_clean_g06_terminal_bridge_v1.py"
)
_SPEC = importlib.util.spec_from_file_location("clean_g06_bridge", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)

REPLAY_HEAD = "a" * 40


def _cjson(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _truth():
    return dict(m._REPLAY_ZERO_CREDIT)


def _seal_receipt(receipt):
    core = deepcopy(receipt)
    core.pop("replay_identity_sha256", None)
    core["replay_identity_sha256"] = _sha256(_cjson(core))
    return core


def _receipt(job: str):
    g06_authority = {
        "schema_version": "12-6.g06-privacy-execution-authority.v1",
        "authority_class": "G06_PRIVACY_EXECUTION_ZERO_CREDIT",
        "execution_identity_sha256": "6" * 64,
        "records": [],
    }
    return _seal_receipt(
        {
            "schema_version": "12-6.d03-clean-g05-g06-physical-replay.v1",
            "status": "PHYSICAL_REPLAY_EXECUTED_ZERO_CREDIT",
            "execution_profile": "LOCAL_FREE",
            "physical_identity": {
                "repository": "Oleksii-debug/12-6-ai.",
                "execution_head_sha": REPLAY_HEAD,
                "workflow_run_id": 11,
                "workflow_run_attempt": 1,
                "workflow_job": job,
            },
            "replay_projection_sha256": "1" * 64,
            "engine_bindings": {"engine": "2" * 40},
            "retained_clean_authority": {"authority": "3" * 64},
            "physical_clean_reconstruction": {
                "materialization_authority_head_sha": "3" * 40,
                "evidence_identity_sha256": "4" * 64,
                "record_payload_jsonl_sha256": "5" * 64,
                "record_inventory_digest_sha256": "7" * 64,
                "payload_inventory_digest_sha256": "8" * 64,
                "record_count": 274,
                "total_payload_bytes": 6093662,
                "source_object_count": 261,
            },
            "g05": {"execution_identity_sha256": "9" * 64},
            "g06": {
                "input_rows_sha256": "b" * 64,
                "execution_identity_sha256": "6" * 64,
                "authority": g06_authority,
            },
            "truth_boundary": _truth(),
            "scope_note": "scoped zero-credit replay",
        }
    )


def _terminal(a, b):
    return {
        "authorized_optimized_target_exposure": 0,
        "corpus_materialization_authority_head_sha": "3" * 40,
        "current_corpus_eligible": False,
        "data526_evidence_identity_sha256": "4" * 64,
        "deterministic_replay_projection_sha256": "1" * 64,
        "deterministic_scientific_projection_agrees": True,
        "execution_profile": "LOCAL_FREE",
        "final_test_outcomes_read": False,
        "foreign_pretrained_weights_used": False,
        "g05_execution_identity_sha256": "9" * 64,
        "g06_execution_identity_sha256": "6" * 64,
        "learned_weights_created": False,
        "optimizer_updates_executed": 0,
        "paid_compute_used": False,
        "physical_jobs": [a["physical_identity"]["workflow_job"], b["physical_identity"]["workflow_job"]],
        "product_execution_head_sha": REPLAY_HEAD,
        "raw_payloads_retained_in_output": False,
        "receipt_a_sha256": _sha256(_cjson(a)),
        "receipt_b_sha256": _sha256(_cjson(b)),
        "replay_identity_a_sha256": a["replay_identity_sha256"],
        "replay_identity_b_sha256": b["replay_identity_sha256"],
        "schema_version": "12-6.d03-clean-g05-g06-two-replay-terminal.v1",
        "tokenizer_fit_authorized": False,
        "training_authorized_bytes": 0,
        "training_executed": False,
        "two_distinct_physical_replay_identities": True,
        "whole_corpus_external_llm_cleanliness_claimed": False,
    }


def _build(
    a=None,
    b=None,
    terminal=None,
    *,
    release_terminal=None,
    release_overrides=None,
):
    a = _receipt("a") if a is None else a
    b = _receipt("b") if b is None else b
    terminal = _terminal(a, b) if terminal is None else terminal
    release_terminal = terminal if release_terminal is None else release_terminal
    synthetic_release = {
        "target_pr_number": 2136,
        "target_head_git_sha": "e" * 40,
        "real_replay_head_git_sha": REPLAY_HEAD,
        "real_replay_run_id": 11,
        "real_replay_job_id": 12,
        "final_head_ci_run_id": 13,
        "final_head_ci_job_id": 14,
        "artifact_id": 15,
        "artifact_zip_sha256": "f" * 64,
        "terminal_summary_sha256": _sha256(_cjson(release_terminal)),
        "independent_audit_issue_number": 2147,
        "independent_audit_terminal_comment_id": 123456,
        "independent_audit_status": "PASS_FOR_INTEGRATION_RELEASED",
    }
    call_args = {
        "target_pr": 2136,
        "target_head": "e" * 40,
        "replay_head": REPLAY_HEAD,
        "replay_run": 11,
        "replay_job": 12,
        "final_run": 13,
        "final_job": 14,
        "artifact_id": 15,
        "artifact_zip_sha256": "f" * 64,
        "audit_issue": 2147,
    }
    if release_overrides:
        call_args.update(release_overrides)

    old_release = m._RELEASE_AUTHORITY
    old_identity = m._RELEASE_AUTHORITY_IDENTITY_SHA256
    m._RELEASE_AUTHORITY = synthetic_release
    m._RELEASE_AUTHORITY_IDENTITY_SHA256 = _sha256(_cjson(synthetic_release))
    try:
        return m.build_bridge(a, b, terminal, **call_args)
    finally:
        m._RELEASE_AUTHORITY = old_release
        m._RELEASE_AUTHORITY_IDENTITY_SHA256 = old_identity


def test_builds_zero_credit_terminal_binding():
    envelope, qualification = _build()
    assert envelope["execution_profile"] == "LOCAL_FREE"
    assert qualification["status"] == "PASS_FOR_G06_TERMINAL_CONSUMPTION"
    assert qualification["replay_count"] == 2
    assert qualification["replay_record_count"] == 274
    assert qualification["replay_utf8_bytes"] == 6093662
    assert qualification["g06_envelope_identity_sha256"] == envelope["evidence_identity_sha256"]
    assert qualification["truth_boundary"] == m._OUTPUT_ZERO_CREDIT
    assert qualification["independent_audit_issue_number"] == 2147
    assert qualification["independent_audit_status"] == "PASS_FOR_INTEGRATION_RELEASED"


def test_same_physical_replay_identity_fails_closed():
    a = _receipt("a")
    with pytest.raises(m.BridgeError, match="physically distinct"):
        _build(a=a, b=deepcopy(a), terminal=_terminal(a, a))


def test_projection_drift_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    b["replay_projection_sha256"] = "0" * 64
    b = _seal_receipt(b)
    with pytest.raises(m.BridgeError, match="projection drift"):
        _build(a=a, b=b, terminal=_terminal(a, b))


def test_head_substitution_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    a["physical_identity"]["execution_head_sha"] = "0" * 40
    a = _seal_receipt(a)
    with pytest.raises(m.BridgeError, match="head drift"):
        _build(a=a, b=b, terminal=_terminal(a, b))


def test_terminal_identity_substitution_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    terminal = _terminal(a, b)
    terminal["g06_execution_identity_sha256"] = "0" * 64
    with pytest.raises(m.BridgeError, match="G06 identity drift"):
        _build(a=a, b=b, terminal=terminal)


def test_incomplete_replay_truth_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    del a["truth_boundary"]["training_executed"]
    a = _seal_receipt(a)
    with pytest.raises(m.BridgeError, match="truth boundary drift"):
        _build(a=a, b=b, terminal=_terminal(a, b))


def test_widened_replay_truth_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    a["truth_boundary"]["training_executed"] = True
    a = _seal_receipt(a)
    with pytest.raises(m.BridgeError, match="truth boundary drift"):
        _build(a=a, b=b, terminal=_terminal(a, b))


def test_replay_tamper_with_stale_identity_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    a["scope_note"] = "tampered"
    with pytest.raises(m.BridgeError, match="self-hash mismatch"):
        _build(a=a, b=b, terminal=_terminal(a, b))


def test_replay_extra_field_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    a["unexpected_authority"] = True
    with pytest.raises(m.BridgeError, match="schema is not closed"):
        _build(a=a, b=b, terminal=_terminal(a, b))


def test_terminal_receipt_hash_tamper_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    terminal = _terminal(a, b)
    terminal["receipt_a_sha256"] = "0" * 64
    with pytest.raises(m.BridgeError, match="receipt hash drift"):
        _build(a=a, b=b, terminal=terminal)


def test_terminal_truth_widening_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    terminal = _terminal(a, b)
    terminal["training_authorized_bytes"] = 1
    with pytest.raises(m.BridgeError, match="zero-credit truth drift"):
        _build(a=a, b=b, terminal=terminal)


def test_terminal_artifact_substitution_fails_closed_before_reseal():
    a = _receipt("a")
    b = _receipt("b")
    release_terminal = _terminal(a, b)
    substituted = deepcopy(release_terminal)
    substituted["data526_evidence_identity_sha256"] = "0" * 64
    with pytest.raises(m.BridgeError, match="release authority drift: terminal_summary_sha256"):
        _build(
            a=a,
            b=b,
            terminal=substituted,
            release_terminal=release_terminal,
        )


def test_coherent_release_metadata_substitution_fails_closed():
    with pytest.raises(m.BridgeError, match="release authority drift"):
        _build(
            release_overrides={
                "target_pr": 9999,
                "target_head": "0" * 40,
                "replay_head": "1" * 40,
                "replay_run": 99,
                "replay_job": 98,
                "final_run": 97,
                "final_job": 96,
                "artifact_id": 95,
                "artifact_zip_sha256": "2" * 64,
                "audit_issue": 94,
            }
        )


def test_immutable_release_identity_drift_fails_closed():
    a = _receipt("a")
    b = _receipt("b")
    terminal = _terminal(a, b)
    old_release = m._RELEASE_AUTHORITY
    old_identity = m._RELEASE_AUTHORITY_IDENTITY_SHA256
    synthetic_release = {
        "target_pr_number": 2136,
        "target_head_git_sha": "e" * 40,
        "real_replay_head_git_sha": REPLAY_HEAD,
        "real_replay_run_id": 11,
        "real_replay_job_id": 12,
        "final_head_ci_run_id": 13,
        "final_head_ci_job_id": 14,
        "artifact_id": 15,
        "artifact_zip_sha256": "f" * 64,
        "terminal_summary_sha256": _sha256(_cjson(terminal)),
        "independent_audit_issue_number": 2147,
        "independent_audit_terminal_comment_id": 123456,
        "independent_audit_status": "PASS_FOR_INTEGRATION_RELEASED",
    }
    m._RELEASE_AUTHORITY = synthetic_release
    m._RELEASE_AUTHORITY_IDENTITY_SHA256 = "0" * 64
    try:
        with pytest.raises(m.BridgeError, match="immutable release authority identity drift"):
            m.build_bridge(
                a,
                b,
                terminal,
                target_pr=2136,
                target_head="e" * 40,
                replay_head=REPLAY_HEAD,
                replay_run=11,
                replay_job=12,
                final_run=13,
                final_job=14,
                artifact_id=15,
                artifact_zip_sha256="f" * 64,
                audit_issue=2147,
            )
    finally:
        m._RELEASE_AUTHORITY = old_release
        m._RELEASE_AUTHORITY_IDENTITY_SHA256 = old_identity


def test_output_paths_must_be_distinct(tmp_path):
    output = tmp_path / "same.json"
    with pytest.raises(m.BridgeError, match="must be distinct"):
        m._validate_output_paths(output, output)
