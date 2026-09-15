from __future__ import annotations

import importlib.util
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


def _truth():
    return {"authorized_optimized_target_exposure": 0}


def _receipt(job: str, replay_identity: str):
    g06_authority = {
        "schema_version": "12-6.g06-privacy-execution-authority.v1",
        "authority_class": "G06_PRIVACY_EXECUTION_ZERO_CREDIT",
        "execution_identity_sha256": "6" * 64,
        "records": [],
    }
    return {
        "schema_version": "12-6.d03-clean-g05-g06-physical-replay.v1",
        "status": "PHYSICAL_REPLAY_EXECUTED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "physical_identity": {
            "execution_head_sha": REPLAY_HEAD,
            "workflow_run_id": 11,
            "workflow_job": job,
        },
        "replay_identity_sha256": replay_identity,
        "replay_projection_sha256": "1" * 64,
        "engine_bindings": {"engine": "2" * 40},
        "retained_clean_authority": {"authority": "3" * 64},
        "physical_clean_reconstruction": {
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
    }


def _terminal():
    return {
        "schema_version": "12-6.d03-clean-g05-g06-two-replay-terminal.v1",
        "deterministic_scientific_projection_agrees": True,
        "two_distinct_physical_replay_identities": True,
        "deterministic_replay_projection_sha256": "1" * 64,
        "product_execution_head_sha": REPLAY_HEAD,
        "g05_execution_identity_sha256": "9" * 64,
        "g06_execution_identity_sha256": "6" * 64,
    }


def _build(a=None, b=None, terminal=None):
    return m.build_bridge(
        _receipt("a", "c" * 64) if a is None else a,
        _receipt("b", "d" * 64) if b is None else b,
        _terminal() if terminal is None else terminal,
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


def test_builds_zero_credit_terminal_binding():
    envelope, qualification = _build()
    assert envelope["execution_profile"] == "LOCAL_FREE"
    assert qualification["status"] == "PASS_FOR_G06_TERMINAL_CONSUMPTION"
    assert qualification["replay_count"] == 2
    assert qualification["replay_record_count"] == 274
    assert qualification["replay_utf8_bytes"] == 6093662
    assert qualification["g06_envelope_identity_sha256"] == envelope["evidence_identity_sha256"]
    assert qualification["truth_boundary"]["training_authorized_bytes"] == 0
    assert qualification["truth_boundary"]["authorized_optimized_target_exposure"] == 0


def test_same_physical_replay_identity_fails_closed():
    b = _receipt("b", "c" * 64)
    with pytest.raises(m.BridgeError, match="physically distinct"):
        _build(b=b)


def test_projection_drift_fails_closed():
    b = _receipt("b", "d" * 64)
    b["replay_projection_sha256"] = "0" * 64
    with pytest.raises(m.BridgeError, match="projection drift"):
        _build(b=b)


def test_head_substitution_fails_closed():
    a = _receipt("a", "c" * 64)
    a["physical_identity"]["execution_head_sha"] = "0" * 40
    with pytest.raises(m.BridgeError, match="head drift"):
        _build(a=a)


def test_terminal_identity_substitution_fails_closed():
    terminal = deepcopy(_terminal())
    terminal["g06_execution_identity_sha256"] = "0" * 64
    with pytest.raises(m.BridgeError, match="G06 identity drift"):
        _build(terminal=terminal)
