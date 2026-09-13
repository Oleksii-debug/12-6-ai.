from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from tools.validate_pretraining_token_budget import TokenBudgetValidationError, validate


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs" / "control" / "pretraining_token_budget_v1.json"
VALIDATOR = ROOT / "tools" / "validate_pretraining_token_budget.py"


def load_control() -> dict:
    return json.loads(CONTROL.read_text(encoding="utf-8"))


def stage(value: dict, name: str) -> dict:
    return next(row for row in value["stages"] if row["name"] == name)


def test_canonical_pretraining_budget_passes() -> None:
    validate(load_control())


def test_control_validator_contains_no_python_assert_statements() -> None:
    tree = ast.parse(VALIDATOR.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_compute_authorization_mutation_fails_closed() -> None:
    value = copy.deepcopy(load_control())
    value["compute_authorized"] = True
    with pytest.raises(TokenBudgetValidationError):
        validate(value)


def test_quality_baseline_cannot_be_silently_promoted() -> None:
    value = copy.deepcopy(load_control())
    value["current_decision"]["20m_quality_baseline_data_budget_ready"] = True
    with pytest.raises(TokenBudgetValidationError):
        validate(value)


def test_direct_chinchilla_to_byte_conversion_is_forbidden() -> None:
    value = copy.deepcopy(load_control())
    value["unit_policy"]["direct_chinchilla_reference_as_byte_budget_allowed"] = True
    stage(value, "MODEL-341-20M")["direct_reference_byte_positions"] = 412_268_800
    with pytest.raises(TokenBudgetValidationError):
        validate(value)


def test_byte_positions_cannot_be_reported_as_fraction_of_external_tokens() -> None:
    value = copy.deepcopy(load_control())
    stage(value, "MODEL-341-20M")["current_request_fraction_of_external_reference"] = 0.048512
    with pytest.raises(TokenBudgetValidationError):
        validate(value)


def test_science_complete_byte_budget_requires_calibration_authority() -> None:
    value = copy.deepcopy(load_control())
    value["current_decision"]["20m_science_complete_byte_budget"] = 412_268_800
    with pytest.raises(TokenBudgetValidationError):
        validate(value)


def test_100m_cannot_be_unblocked_without_measured_scaling_evidence() -> None:
    value = copy.deepcopy(load_control())
    value["current_decision"]["100m_training_ready"] = True
    with pytest.raises(TokenBudgetValidationError):
        validate(value)
