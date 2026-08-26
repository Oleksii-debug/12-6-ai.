from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/scaling/data_budget_policy_v1.json"
VALIDATOR = ROOT / "tools/validate_scaling_data_budget_policy.py"

spec = importlib.util.spec_from_file_location("scaling_data_budget_validator", VALIDATOR)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load_policy() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def test_current_policy_passes() -> None:
    report = module.validate(load_policy())
    assert report == {
        "status": "PASS",
        "stage_count": 3,
        "primary_parameter_count": 20_613_440,
        "primary_20x_unique_loss_tokens": 412_268_800,
        "source_registry_pr": 538,
        "source_capacity_bytes_at_cutoff": 565_743,
        "source_bytes_are_token_authority": False,
        "long_training_ready": False,
    }


def test_bytes_cannot_be_promoted_to_training_tokens() -> None:
    broken = copy.deepcopy(load_policy())
    broken["truth_boundary"]["source_capacity_bytes_are_training_tokens"] = True
    with pytest.raises(ValueError, match="bytes must never be token authority"):
        module.validate(broken)


def test_unique_loss_reference_is_derived_from_parameter_count() -> None:
    broken = copy.deepcopy(load_policy())
    broken["stages"][0]["reference_unique_loss_tokens"]["20x"] += 1
    with pytest.raises(ValueError, match="20M_PRIMARY 20x token reference drift"):
        module.validate(broken)


def test_paid_compute_cannot_be_authorized_by_research_policy() -> None:
    broken = copy.deepcopy(load_policy())
    broken["truth_boundary"]["paid_compute_authorized"] = True
    with pytest.raises(ValueError, match="must not authorize paid compute"):
        module.validate(broken)


def test_long_training_cannot_be_marked_ready_from_source_capacity() -> None:
    broken = copy.deepcopy(load_policy())
    broken["current_20m_observation"]["learned_20m_long_training_ready"] = True
    with pytest.raises(ValueError, match="must not fabricate 20M training readiness"):
        module.validate(broken)


def test_exact_tokenization_evidence_cannot_be_removed() -> None:
    broken = copy.deepcopy(load_policy())
    broken["long_training_gate"]["required_evidence"].remove(
        "exact_post_tokenization_train_token_count"
    )
    with pytest.raises(ValueError, match="required evidence set drift"):
        module.validate(broken)


def test_live_registry_identity_cannot_drift_silently() -> None:
    broken = copy.deepcopy(load_policy())
    broken["current_20m_observation"]["source_registry_identity"] = "0" * 64
    with pytest.raises(ValueError, match="source registry identity drift"):
        module.validate(broken)


def test_source_capacity_gap_is_derived_not_free_text() -> None:
    broken = copy.deepcopy(load_policy())
    broken["current_20m_observation"]["remaining_source_capacity_gap_bytes"] -= 1
    with pytest.raises(ValueError, match="source-capacity gap arithmetic drift"):
        module.validate(broken)
