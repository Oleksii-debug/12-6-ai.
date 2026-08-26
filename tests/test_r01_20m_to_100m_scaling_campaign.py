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


def test_long_training_prerequisite_cannot_be_relaxed() -> None:
    required = (
        "requires_corpus_identity",
        "requires_tokenizer_fit_identity",
        "requires_unique_post_pack_loss_ledger",
        "requires_checkpoint_integrity_terminal_retest",
        "requires_selection_validation_terminal",
        "requires_compute_authorization_if_material_cost",
    )
    for experiment_id in ("R01-E20", "R01-E30"):
        for field in required:
            data = _load()
            _experiment(data, experiment_id)[field] = False
            errors = validator.validate_campaign(data)
            assert any(f"{experiment_id}.{field}" in error for error in errors), (
                experiment_id,
                field,
                errors,
            )


def test_readiness_cannot_fabricate_checkpoint_or_selection_terminality() -> None:
    for field in (
        "checkpoint_integrity_terminal_retest",
        "selection_validation_terminal",
    ):
        data = _load()
        data["current_readiness"][field] = True
        errors = validator.validate_campaign(data)
        assert any(field in error for error in errors)


def test_model_mechanics_readiness_cannot_be_relabelled() -> None:
    data = _load()
    data["current_readiness"]["model_mechanics"] = "LEARNED_TERMINAL"
    errors = validator.validate_campaign(data)
    assert any("current_readiness.model_mechanics" in error for error in errors)


def test_100m_modelspec_cannot_be_frozen_before_evidence() -> None:
    data = _load()
    _experiment(data, "R01-E30")["freeze_100m_modelspec_now"] = True
    errors = validator.validate_campaign(data)
    assert any("100M ModelSpec" in error for error in errors)


def test_100m_sweep_requires_learned_20m_evidence() -> None:
    data = _load()
    _experiment(data, "R01-E30")["requires_20m_learned_evidence"] = False
    errors = validator.validate_campaign(data)
    assert any("learned-20M" in error for error in errors)


def test_tokenizer_calibration_cannot_be_authorized_while_blocked() -> None:
    data = _load()
    _experiment(data, "R01-E10")["authorized_now"] = True
    errors = validator.validate_campaign(data)
    assert any("R01-E10" in error and "authorized" in error for error in errors)


def test_20m_experiment_rows_bind_model341_parameter_count() -> None:
    for experiment_id in ("R01-E00", "R01-E10", "R01-E20"):
        data = _load()
        _experiment(data, experiment_id)["parameters"] = 20000000
        errors = validator.validate_campaign(data)
        assert any(experiment_id in error and "parameter count" in error for error in errors)


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


def test_architecture_policy_cannot_silently_drift() -> None:
    data = _load()
    data["scientific_principles"]["architecture_bias_below_1b"] = ["wide_shallow"]
    errors = validator.validate_campaign(data)
    assert any("architecture-bias" in error for error in errors)


def test_required_promotion_gate_cannot_be_removed() -> None:
    data = _load()
    data["promotion_gates"].remove("reserved_evaluation_decontamination")
    errors = validator.validate_campaign(data)
    assert any("promotion gate set" in error for error in errors)


def test_evaluation_firewall_cannot_be_relaxed() -> None:
    data = _load()
    data["metric_contract"]["evaluation_firewall"] = "training_may_consume_selection_validation"
    errors = validator.validate_campaign(data)
    assert any("evaluation firewall" in error for error in errors)


def test_optimizer_nan_inf_policy_cannot_be_disabled() -> None:
    data = _load()
    data["optimizer_research"]["nan_inf_fail_closed"] = False
    errors = validator.validate_campaign(data)
    assert any("optimizer_research.nan_inf_fail_closed" in error for error in errors)


def test_gradient_clipping_requirement_cannot_be_disabled() -> None:
    data = _load()
    data["optimizer_research"]["gradient_clipping_required"] = False
    errors = validator.validate_campaign(data)
    assert any("optimizer_research.gradient_clipping_required" in error for error in errors)


def test_schema_version_rejects_boolean_alias() -> None:
    data = _load()
    data["schema_version"] = True
    errors = validator.validate_campaign(data)
    assert any("schema_version" in error for error in errors)


def test_baseline_boolean_rejects_integer_alias() -> None:
    data = _load()
    data["baseline_model"]["tie_word_embeddings"] = 1
    errors = validator.validate_campaign(data)
    assert any("baseline_model.tie_word_embeddings" in error for error in errors)


def test_mutating_copy_does_not_change_control() -> None:
    original = _load()
    mutated = copy.deepcopy(original)
    mutated["baseline_model"]["n_layers"] = 17
    assert validator.validate_campaign(original) == []
    errors = validator.validate_campaign(mutated)
    assert any("baseline_model.n_layers" in error for error in errors)
