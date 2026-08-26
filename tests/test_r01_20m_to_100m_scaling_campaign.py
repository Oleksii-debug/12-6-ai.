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


def test_readiness_checkpoint_retest_cannot_be_silently_promoted() -> None:
    data = _load()
    data["current_readiness"]["checkpoint_integrity_terminal_retest"] = True
    errors = validator.validate_campaign(data)
    assert any("checkpoint_integrity_terminal_retest" in error for error in errors)


def test_readiness_selection_validation_cannot_be_silently_promoted() -> None:
    data = _load()
    data["current_readiness"]["selection_validation_terminal"] = True
    errors = validator.validate_campaign(data)
    assert any("selection_validation_terminal" in error for error in errors)


def test_readiness_model_mechanics_authority_cannot_drift() -> None:
    data = _load()
    data["current_readiness"]["model_mechanics"] = "QUALIFIED_BY_PROSE_ONLY"
    errors = validator.validate_campaign(data)
    assert any("current_readiness.model_mechanics" in error for error in errors)


def test_long_training_experiment_cannot_be_authorized_now() -> None:
    data = _load()
    experiment = next(item for item in data["experiment_matrix"] if item["id"] == "R01-E20")
    experiment["authorized_now"] = True
    errors = validator.validate_campaign(data)
    assert any("R01-E20.authorized_now" in error for error in errors)


def test_tokenizer_calibration_cannot_be_authorized_without_corpus() -> None:
    data = _load()
    experiment = next(item for item in data["experiment_matrix"] if item["id"] == "R01-E10")
    experiment["authorized_now"] = True
    errors = validator.validate_campaign(data)
    assert any("R01-E10.authorized_now" in error for error in errors)


def test_learned_20m_cannot_drop_checkpoint_gate() -> None:
    data = _load()
    experiment = next(item for item in data["experiment_matrix"] if item["id"] == "R01-E20")
    experiment["requires_checkpoint_integrity_terminal_retest"] = False
    errors = validator.validate_campaign(data)
    assert any("requires_checkpoint_integrity_terminal_retest" in error for error in errors)


def test_100m_sweep_cannot_drop_learned_20m_gate() -> None:
    data = _load()
    experiment = next(item for item in data["experiment_matrix"] if item["id"] == "R01-E30")
    experiment["requires_20m_learned_evidence"] = False
    errors = validator.validate_campaign(data)
    assert any("requires_20m_learned_evidence" in error for error in errors)


def test_100m_modelspec_cannot_be_frozen_before_evidence() -> None:
    data = _load()
    experiment = next(item for item in data["experiment_matrix"] if item["id"] == "R01-E30")
    experiment["freeze_100m_modelspec_now"] = True
    errors = validator.validate_campaign(data)
    assert any("100M ModelSpec" in error for error in errors)


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


def test_required_promotion_gate_cannot_be_removed() -> None:
    data = _load()
    data["promotion_gates"].remove("reserved_evaluation_decontamination")
    errors = validator.validate_campaign(data)
    assert any("promotion gate set" in error for error in errors)


def test_mutating_copy_does_not_change_control() -> None:
    original = _load()
    mutated = copy.deepcopy(original)
    mutated["baseline_model"]["n_layers"] = 17
    assert validator.validate_campaign(original) == []
    errors = validator.validate_campaign(mutated)
    assert any("baseline_model.n_layers" in error for error in errors)
