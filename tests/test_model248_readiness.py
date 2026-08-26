from __future__ import annotations

import json
from pathlib import Path

from tools.validate_model248_readiness import validate


REPORT = Path("evidence/model248/readiness_20260826.json")


def _payload() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_committed_readiness_report_is_fail_closed() -> None:
    payload = _payload()
    assert validate(payload) == []
    assert payload["training_executed"] is False
    assert payload["optimizer_updates"] == 0
    assert payload["decision"]["replace_8q2kv_with_8q4kv"] is False
    assert payload["decision"]["current_10m_default"] == "8Q/2KV"


def test_strict_parameter_match_cannot_be_silently_relaxed() -> None:
    payload = _payload()
    payload["parameter_match"]["strict_match"] = True
    assert "strict_parameter_match" in validate(payload)


def test_blocked_report_cannot_claim_training() -> None:
    payload = _payload()
    payload["training_executed"] = True
    payload["optimizer_updates"] = 1
    errors = validate(payload)
    assert "training_executed" in errors
    assert "optimizer_updates" in errors


def test_kv_cache_accounting_is_unexpanded_geometry() -> None:
    payload = _payload()
    kv = payload["unexpanded_kv_cache_reference"]
    assert kv["candidate_8q4kv_bytes"] == 2 * kv["incumbent_8q2kv_bytes"]
