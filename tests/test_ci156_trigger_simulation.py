from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ci156_trigger_simulation", ROOT / "tools" / "ci156_trigger_simulation.py"
)
assert SPEC is not None and SPEC.loader is not None
SIM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIM
SPEC.loader.exec_module(SIM)


def _row(name: str) -> dict[str, object]:
    rows = SIM.report()["scenarios"]
    return next(row for row in rows if row["scenario"] == name)


def test_milestone150_research_sha_contracts_from_eight_to_two_workflows() -> None:
    row = _row("milestone150_research_sha")
    assert row["before_count"] == 8
    assert row["after_count"] == 2
    assert row["workflow_runs_avoided"] == 6
    assert row["after_workflows"] == ["CI", "MILESTONE-150 Learned Base Ladder V1"]
    assert row["arm_portability_job_after"] is False


def test_dependency_lock_change_keeps_ci_and_d08_and_arm_portability() -> None:
    row = _row("dependency_lock_change")
    assert row["after_workflows"] == ["CI", "D08 Purpose Environments"]
    assert row["arm_portability_job_after"] is True


def test_d02_change_does_not_fan_out_other_experiment_workflows() -> None:
    row = _row("d02_real_training_change")
    assert row["after_workflows"] == ["CI", "D02 Real S0 Training"]


def test_docs_only_skips_compute_workflows() -> None:
    row = _row("docs_only")
    assert row["after_count"] == 0
