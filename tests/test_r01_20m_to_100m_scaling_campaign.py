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


def _experiment(data: dict, experiment_id: str) -> dict:
    return next(item for item in data["experiment_matrix"] if item["id"] == experiment_id)


def test_frozen_campaign_is_valid() -> None:
    assert validator.validate_campaign(_load()) == []


def test_paid_compute_cannot_be_silently_authorized() -> None:
    data = _load()
    data["hard_boundaries"]["paid_compute_authorized"] = True
    errors = validator.validate_campaign(data)
    assert any("paid_compute_authorized" in error for error in errors)


def test_long_training_experiment_cannot_be_authorized_now() -> None:
    data = _load()
    _experiment(data, "R01-E20")["authorized_now"] = True
    errors = validator.validate_campaign(data)
    assert any("R01-E20" in error for error in errors)


def test_tokenizer_calibration_cannot_be_authorized_without_corpus() -> None:
    data = _load()
    _experiment(data, "R01-E10")["authorized_now"] = True
    errors = validator.validate_campaign(data)
    assert any("R01-E10" in error and "authorized" in error for error in errors)


def test_tokenizer_calibration_cannot_drop_corpus_gate() -> None:
    data = _load()
    _experiment(data, "R01-E10")["requires_corpus_identity"] = False
    errors = validator.validate_campaign(data)
    assert any(
        "R01-E10.requires_corpus_identity gate drift" in error for error in errors
    )


def test_learned_sweep_cannot_drop_checkpoint_gate() -> None:
    data = _load()
    _experiment(data, "R01-E20")["requires_checkpoint_integrity_terminal_retest"] = False
    errors = validator.validate_campaign(data)
    assert any(
        "R01-E20.requires_checkpoint_integrity_terminal_retest gate drift" in error
        for error in errors
    )


def test_100m_sweep_cannot_drop_20m_evidence_gate() -> None:
    data = _load()
    _experiment(data, "R01-E30")["requires_20m_learned_evidence"] = False
    errors = validator.validate_campaign(data)
    assert any(
        "R01-E30.requires_20m_learned_evidence gate drift" in error
        for error in errors
    )


def test_checkpoint_terminality_cannot_be_fabricated_in_snapshot() -> None:
    data = _load()
    data["current_readiness"]["checkpoint_integrity_terminal_retest"] = True
    errors = validator.validate_campaign(data)
    assert any("checkpoint terminality" in error for error in errors)


def test_selection_terminality_cannot_be_fabricated_in_snapshot() -> None:
    data = _load()
    data["current_readiness"]["selection_validation_terminal"] = True
    errors = validator.validate_campaign(data)
    assert any("selection-validation terminality" in error for error in errors)


def test_100m_modelspec_cannot_be_frozen_before_evidence() -> None:
    data = _load()
    _experiment(data, "R01-E30")["freeze_100m_modelspec_now"] = True
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
