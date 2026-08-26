from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.validate_r01_20m_to_100m_scaling_campaign import validate_campaign

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = json.loads(
    (ROOT / "configs/research/r01_20m_to_100m_scaling_campaign_v1.json").read_text(
        encoding="utf-8"
    )
)


def _entry(campaign: dict[str, object], entry_id: str) -> dict[str, object]:
    matrix = campaign["experiment_matrix"]
    assert isinstance(matrix, list)
    return next(
        item for item in matrix if isinstance(item, dict) and item.get("id") == entry_id
    )


def test_stage_data_budget_contract_is_valid() -> None:
    assert validate_campaign(copy.deepcopy(CAMPAIGN)) == []


def test_20m_long_training_requires_stage_data_budget_authority() -> None:
    weakened = copy.deepcopy(CAMPAIGN)
    _entry(weakened, "R01-E20")["requires_stage_data_budget_authority"] = False

    errors = validate_campaign(weakened)

    assert any("R01-E20.requires_stage_data_budget_authority" in error for error in errors)


def test_100m_sweep_requires_preregistered_replay_policy() -> None:
    weakened = copy.deepcopy(CAMPAIGN)
    _entry(weakened, "R01-E30")["requires_preregistered_replay_policy"] = False

    errors = validate_campaign(weakened)

    assert any("R01-E30.requires_preregistered_replay_policy" in error for error in errors)


def test_tokens_per_parameter_cannot_masquerade_as_unique_data_budget() -> None:
    weakened = copy.deepcopy(CAMPAIGN)
    _entry(weakened, "R01-E20")["tokens_per_parameter_semantics"] = (
        "UNIQUE_DATA_REQUIREMENT"
    )

    errors = validate_campaign(weakened)

    assert any("distinguish exposure tokens from unique-data sufficiency" in error for error in errors)


def test_replay_accounting_cannot_be_disabled() -> None:
    weakened = copy.deepcopy(CAMPAIGN)
    policy = weakened["data_budget_policy"]
    assert isinstance(policy, dict)
    policy["replay_factor_must_be_measured"] = False

    errors = validate_campaign(weakened)

    assert any("data_budget_policy.replay_factor_must_be_measured" in error for error in errors)


def test_four_epoch_reference_remains_nonbinding_planning_evidence() -> None:
    weakened = copy.deepcopy(CAMPAIGN)
    policy = weakened["data_budget_policy"]
    assert isinstance(policy, dict)
    policy["replay_reference_interpretation"] = "UNIVERSAL_HARD_CAP"

    errors = validate_campaign(weakened)

    assert any("planning reference rather than a universal hard cap" in error for error in errors)
