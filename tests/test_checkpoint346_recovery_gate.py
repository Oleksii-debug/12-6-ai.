from __future__ import annotations

import json
from copy import deepcopy

import pytest

from tools.validate_checkpoint346_recovery_gate import EVIDENCE, load_and_validate


def test_checkpoint346_gate_is_fail_closed() -> None:
    value = load_and_validate()
    assert value["verdict"] == "BLOCKED_MISSING_PRIMARY_20M_MODELSPEC"
    assert value["execution_accounting"]["optimizer_updates"] == 0
    assert value["execution_accounting"]["recovery_injections_executed_on_20m"] == 0


def test_checkpoint346_successor_budget_is_bounded() -> None:
    value = load_and_validate()
    successor = value["unblock_rule"]["then_execute"]
    assert successor["maximum_optimizer_steps"] <= 3
    assert successor["synthetic_fixture_only"] is True
    assert successor["cpu_only"] is True


def test_checkpoint346_cannot_relabel_incumbent_as_20m(tmp_path) -> None:
    value = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    mutated = deepcopy(value)
    mutated["incumbent_recovery_authority"]["not_a_20m_qualification"] = False
    target = tmp_path / "mutated.json"
    target.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(AssertionError):
        load_and_validate(target)
