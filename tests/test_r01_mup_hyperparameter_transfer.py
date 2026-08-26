from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/research/r01_mup_hyperparameter_transfer_v1.json"
VALIDATOR_PATH = ROOT / "tools/validate_r01_mup_hyperparameter_transfer.py"

spec = importlib.util.spec_from_file_location("r01_mup_validator", VALIDATOR_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_research_contract_is_valid() -> None:
    assert validator.validate_campaign(_load()) == []


def test_training_cannot_be_silently_authorized() -> None:
    data = _load()
    data["hard_boundaries"]["model_training_authorized"] = True
    errors = validator.validate_campaign(data)
    assert any("model_training_authorized" in error for error in errors)


def test_standard_parameterization_transfer_cannot_be_assumed() -> None:
    data = _load()
    data["hard_boundaries"]["standard_parameterization_transfer_may_be_assumed"] = True
    errors = validator.validate_campaign(data)
    assert any("standard_parameterization_transfer_may_be_assumed" in error for error in errors)


def test_width_and_depth_transfer_cannot_be_conflated() -> None:
    data = _load()
    data["mup_integration_contract"][
        "width_and_depth_may_not_be_conflated_in_one_transfer_claim"
    ] = False
    errors = validator.validate_campaign(data)
    assert any("width_and_depth" in error for error in errors)


def test_tied_embedding_handling_cannot_be_weakened() -> None:
    data = _load()
    data["mup_integration_contract"]["tied_embedding_handling_must_be_mup_compatible"] = False
    errors = validator.validate_campaign(data)
    assert any("tied_embedding_handling" in error for error in errors)


def test_coordinate_probe_ladder_is_bound() -> None:
    data = _load()
    data["coordinate_check_preregistration"]["width_probes"][0]["d_model"] = 160
    errors = validator.validate_campaign(data)
    assert any("width probe ladder" in error for error in errors)


def test_repeated_exposure_cannot_inflate_unique_capacity() -> None:
    data = _load()
    data["data_constrained_pilot"]["repeated_positions_may_increase_unique_capacity"] = True
    errors = validator.validate_campaign(data)
    assert any("inflate unique capacity" in error for error in errors)


def test_update_ratio_metric_cannot_be_removed() -> None:
    data = _load()
    data["metrics"]["required"].remove("update_to_weight_ratio_by_layer")
    errors = validator.validate_campaign(data)
    assert any("required metric set" in error for error in errors)


def test_final_test_cannot_be_enabled_for_selection() -> None:
    data = _load()
    data["decision_rules"]["final_test_can_be_used_for_hyperparameter_selection"] = True
    errors = validator.validate_campaign(data)
    assert any("final_test_can_be_used" in error for error in errors)


def test_mutating_copy_does_not_change_control() -> None:
    original = _load()
    mutated = copy.deepcopy(original)
    mutated["hyperparameter_transfer_preregistration"]["learning_rate_search"]["multipliers"] = [1.0]
    assert validator.validate_campaign(original) == []
    errors = validator.validate_campaign(mutated)
    assert any("LR grid" in error for error in errors)
