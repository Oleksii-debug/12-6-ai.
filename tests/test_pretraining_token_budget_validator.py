from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from tools.validate_pretraining_token_budget import (
    TokenBudgetValidationError,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs" / "control" / "pretraining_token_budget_v1.json"
VALIDATOR = ROOT / "tools" / "validate_pretraining_token_budget.py"


def load_control() -> dict:
    return json.loads(CONTROL.read_text(encoding="utf-8"))


def test_canonical_pretraining_token_budget_passes() -> None:
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


def test_reference_token_count_is_bound_to_exact_model_size() -> None:
    value = copy.deepcopy(load_control())
    stage20 = next(row for row in value["stages"] if row["name"] == "MODEL-341-20M")
    stage20["compute_optimal_reference_tokens"] += 1
    with pytest.raises(TokenBudgetValidationError):
        validate(value)
