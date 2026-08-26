#!/usr/bin/env python3
"""Validate the fail-closed CHECKPOINT-346 dependency gate.

This validator deliberately performs no optimizer work.  It protects the truth
boundary until the exact RESEARCH-339/MODEL-341 primary ~20M authority exists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/checkpoint346/recovery_20m_dependency_gate_v1.json"
EXPECTED_IDENTITY = "cfb42e2f5f7dd08b6c1556abeeb97399c9a9e110a2cfe151b5f98e96466e1557"
EXPECTED_RECOVERY_HEAD = "349e6db94d4aca81c2d1a0ccc3368a98b6058392"
EXPECTED_RECOVERY_PROOF = "9e002d07e85624da5b9799a08a006f589472769055df6609e17b17e698a8da5b"


def _hash_without_identity(value: dict[str, object]) -> str:
    body = dict(value)
    body.pop("identity_sha256", None)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate(path: Path = EVIDENCE) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema"] == "12-6.checkpoint346-20m-recovery-qualification-gate.v1"
    assert value["worker_id"] == "CHECKPOINT-346-20M-RECOVERY-QUALIFICATION"
    assert value["execution_profile"] == "LOCAL_FREE"
    assert value["verdict"] == "BLOCKED_MISSING_PRIMARY_20M_MODELSPEC"
    assert value["identity_sha256"] == EXPECTED_IDENTITY
    assert _hash_without_identity(value) == EXPECTED_IDENTITY

    predecessors = value["required_predecessors"]
    assert predecessors["research339"]["repository_authority_found"] is False
    assert predecessors["model341"]["repository_authority_found"] is False

    accounting = value["execution_accounting"]
    assert accounting == {
        "final_test_outcomes_read": False,
        "foreign_weights_used": False,
        "long_campaign_executed": False,
        "model_training_started": False,
        "optimizer_updates": 0,
        "paid_compute": False,
        "recovery_injections_executed_on_20m": 0,
    }

    incumbent = value["incumbent_recovery_authority"]
    assert incumbent["head_sha"] == EXPECTED_RECOVERY_HEAD
    assert incumbent["proof_identity_sha256"] == EXPECTED_RECOVERY_PROOF
    assert incumbent["contract_reusable_for_20m"] is True
    assert incumbent["not_a_20m_qualification"] is True

    successor = value["unblock_rule"]["then_execute"]
    assert successor["maximum_optimizer_steps"] == 3
    assert successor["synthetic_fixture_only"] is True
    assert successor["cpu_only"] is True

    missing = set(value["blocking_reason"]["missing_exact_fields"])
    assert "primary 20M ModelSpec identity" in missing
    assert "model constructor/configuration binding" in missing
    return value


def main() -> None:
    value = load_and_validate()
    print(value["verdict"])
    print(value["identity_sha256"])


if __name__ == "__main__":
    main()
