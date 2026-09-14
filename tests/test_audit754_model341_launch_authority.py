from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/audit754_model341_launch_authority.py"
SPEC = importlib.util.spec_from_file_location("audit754_model341_launch_authority", TOOL)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_report_hash_is_deterministic_and_strict_json() -> None:
    base = {"schema_version": 1, "findings": [], "verdict": "PASS_WITH_NOTES"}
    first = AUDIT._finalize(dict(base))
    second = AUDIT._finalize(dict(base))
    assert first == second
    assert len(first["report_sha256"]) == 64
    json.dumps(first, allow_nan=False)


def test_authority_fixture_rejects_boolean_as_positive_run_id_by_contract() -> None:
    authority = AUDIT._authority()
    assert authority["terminal"] is True
    assert authority["workflow_conclusion"] == "success"
    assert isinstance(authority["workflow_run_id"], int)
    assert not isinstance(authority["workflow_run_id"], bool)


def test_current_canonical_gate_is_redteamed_against_expected_findings() -> None:
    pytest.importorskip("twelve_six.learned20m_readiness")
    report = AUDIT.run_live_checkout_audit(ROOT)
    ids = {finding["id"] for finding in report["findings"]}
    assert report["execution_status"] == "COMPLETE"
    assert report["verdict"] == "CHANGES_REQUIRED"
    assert {
        "AUDIT754-001",
        "AUDIT754-002",
        "AUDIT754-003",
        "AUDIT754-004",
        "AUDIT754-005",
    } <= ids
    assert report["truth_boundary"]["product_code_modified"] is False
    json.dumps(report, allow_nan=False)
