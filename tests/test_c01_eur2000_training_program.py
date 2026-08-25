from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_c01_eur2000_training_program.py"
SPEC = importlib.util.spec_from_file_location("c01_eur2000_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _program():
    return MODULE.load_program(ROOT / "configs/runs/c01_eur2000_training_program.v1.json")


def test_program_is_valid_planning_but_not_launch_ready() -> None:
    program = _program()
    MODULE.validate_program(program)
    blockers = MODULE.launch_blockers(program)
    assert "P01_SCALE_DATA_IDENTITY" in blockers
    assert "P02_SCALE_TOKENIZER_IDENTITY" in blockers
    assert "P04_SAME_GEOMETRY_GPU_SMOKE" in blockers
    assert "AUTHORITY_PAID_COMPUTE_FALSE" in blockers
    assert "NO_MEASURED_GPU_THROUGHPUT" in blockers


def test_validator_rejects_paid_authorization_in_planning_file() -> None:
    program = copy.deepcopy(_program())
    program["authority"]["materially_paid_compute_authorized"] = True
    with pytest.raises(MODULE.ProgramValidationError, match="must not authorize"):
        MODULE.validate_program(program)


def test_validator_rejects_flop_drift() -> None:
    program = copy.deepcopy(_program())
    program["strategies"][0]["planning_train_flops_6n_per_token"] += 1
    with pytest.raises(MODULE.ProgramValidationError, match="FLOP estimate drift"):
        MODULE.validate_program(program)


def test_validator_rejects_budget_overallocation() -> None:
    program = copy.deepcopy(_program())
    program["budget"]["main_training_cap_eur"] += 1.0
    with pytest.raises(MODULE.ProgramValidationError, match="allocation must equal"):
        MODULE.validate_program(program)


def test_recommended_main_wall_time_requires_same_geometry_pilot() -> None:
    program = copy.deepcopy(_program())
    recommended = next(item for item in program["strategies"] if item.get("recommended"))
    recommended["main_run"]["wall_time_formula"] = "training_tokens / S0_CPU_TOKENS_PER_SECOND"
    with pytest.raises(MODULE.ProgramValidationError, match="GPU pilot"):
        MODULE.validate_program(program)
