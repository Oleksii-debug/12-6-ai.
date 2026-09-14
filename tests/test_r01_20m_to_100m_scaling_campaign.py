from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/research/r01_20m_to_100m_scaling_campaign_v1.json"
VALIDATOR_PATH = ROOT / "tools/validate_r01_20m_to_100m_scaling_campaign.py"

spec = importlib.util.spec_from_file_location("r01_scaling_validator", VALIDATOR_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_campaign_is_valid() -> None:
    assert validator.validate_campaign(_load()) == []


def test_paid_compute_cannot_be_silently_authorized() -> None:
    data = _load()
    data["hard_boundaries"]["paid_compute_authorized"] = True
    errors = validator.validate_campaign(data)
    assert any("paid_compute_authorized" in error for error in errors)


def test_long_training_experiment_cannot_be_authorized_now() -> None:
    data = _load()
    experiment = next(
        item for item in data["experiment_matrix"] if item["id"] == "R01-E20"
    )
    experiment["authorized_now"] = True
    errors = validator.validate_campaign(data)
    assert any("R01-E20" in error for error in errors)


def test_100m_modelspec_cannot_be_frozen_before_evidence() -> None:
    data = _load()
    experiment = next(
        item for item in data["experiment_matrix"] if item["id"] == "R01-E30"
    )
    experiment["freeze_100m_modelspec_now"] = True
    errors = validator.validate_campaign(data)
    assert any("100M ModelSpec" in error for error in errors)


def test_100m_sweep_requires_cross_scale_hyperparameter_evidence() -> None:
    data = _load()
    experiment = next(
        item for item in data["experiment_matrix"] if item["id"] == "R01-E30"
    )
    experiment["requires_cross_scale_hyperparameter_evidence"] = False
    errors = validator.validate_campaign(data)
    assert any("cross-scale hyperparameter evidence" in error for error in errors)


def test_model341_authority_drift_fails_closed() -> None:
    data = _load()
    data["authority"]["model341_sha"] = "0" * 40
    errors = validator.validate_campaign(data)
    assert any("authority.model341_sha" in error for error in errors)


def test_cross_tokenizer_metric_cannot_drop_bpb() -> None:
    data = _load()
    data["scientific_principles"]["cross_tokenizer_primary_metric"] = "perplexity"
    errors = validator.validate_campaign(data)
    assert any("cross-tokenizer primary metric" in error for error in errors)


def test_parameter_count_alone_cannot_become_primary_scale_axis() -> None:
    data = _load()
    data["compute_accounting"]["parameter_only_6nd_is_primary_scale_axis"] = True
    errors = validator.validate_campaign(data)
    assert any("parameter_only_6nd_is_primary_scale_axis" in error for error in errors)


def test_silent_20m_hyperparameter_copy_is_forbidden() -> None:
    data = _load()
    data["cross_scale_hyperparameter_transfer"][
        "silent_20m_to_50m_or_100m_copy_allowed"
    ] = True
    errors = validator.validate_campaign(data)
    assert any("silent 20M hyperparameter copy" in error for error in errors)


def test_umup_cannot_be_silently_adopted_by_planning_contract() -> None:
    data = _load()
    data["cross_scale_hyperparameter_transfer"]["u_mup_adopted_now"] = True
    errors = validator.validate_campaign(data)
    assert any("u-muP may not be silently adopted" in error for error in errors)


def test_cross_scale_transfer_must_keep_retuning_and_umup_control_paths() -> None:
    data = _load()
    data["cross_scale_hyperparameter_transfer"]["allowed_evidence_paths"] = [
        "u_mup_proxy_transfer_with_matched_standard_parameterization_control"
    ]
    errors = validator.validate_campaign(data)
    assert any("transfer evidence paths" in error for error in errors)


def test_required_promotion_gate_cannot_be_removed() -> None:
    data = _load()
    data["promotion_gates"].remove("reserved_evaluation_decontamination")
    errors = validator.validate_campaign(data)
    assert any("promotion gate set" in error for error in errors)


def test_cross_scale_promotion_gate_cannot_be_removed() -> None:
    data = _load()
    data["promotion_gates"].remove("cross_scale_hyperparameter_evidence")
    errors = validator.validate_campaign(data)
    assert any("promotion gate set" in error for error in errors)


def test_mutating_copy_does_not_change_control() -> None:
    original = _load()
    mutated = copy.deepcopy(original)
    mutated["baseline_model"]["n_layers"] = 17
    assert validator.validate_campaign(original) == []
    errors = validator.validate_campaign(mutated)
    assert any("baseline_model.n_layers" in error for error in errors)
