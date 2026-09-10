from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six import InitSpec, ModelSpec, load_stage_config
from twelve_six.model import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL341_CONFIG = ROOT / "configs" / "candidates" / "model341_20m_candidate_a.json"
MODEL341_HASH = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
INIT_HASH = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


@pytest.mark.parametrize("field", ["rope_theta", "norm_eps"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_model_spec_rejects_non_finite_identity_float(field: str, value: float) -> None:
    payload = load_stage_config(MODEL341_CONFIG).model.to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        ModelSpec.from_dict(payload)


@pytest.mark.parametrize("field", ["rope_theta", "norm_eps"])
def test_model_spec_rejects_boolean_identity_float(field: str) -> None:
    payload = load_stage_config(MODEL341_CONFIG).model.to_dict()
    payload[field] = True
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        ModelSpec.from_dict(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_init_spec_rejects_non_finite_std(value: float) -> None:
    with pytest.raises(ValueError, match=r"InitSpec std must be finite"):
        InitSpec(std=value)


def test_init_spec_rejects_boolean_std() -> None:
    with pytest.raises(ValueError, match=r"InitSpec std must be finite"):
        InitSpec(std=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_hash_rejects_non_standard_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match=r"Out of range float values"):
        canonical_json_sha256({"semantic_float": value})


def test_model341_finite_identity_and_parameter_count_do_not_drift() -> None:
    stage = load_stage_config(MODEL341_CONFIG)
    assert stage.model.identity_sha256() == MODEL341_HASH
    assert stage.init.identity_sha256() == INIT_HASH
    assert stage.model.parameter_count() == 20_613_440
    assert stage.expected_parameters == 20_613_440


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("model", "rope_theta", float("nan"), "rope_theta must be finite"),
        ("model", "norm_eps", float("inf"), "norm_eps must be finite"),
        ("init", "std", float("-inf"), "InitSpec std must be finite"),
    ],
)
def test_stage_config_rejects_poisoned_non_finite_semantics_before_model_construction(
    tmp_path: Path,
    section: str,
    field: str,
    value: float,
    message: str,
) -> None:
    payload = json.loads(MODEL341_CONFIG.read_text(encoding="utf-8"))
    payload[section][field] = value
    poisoned = tmp_path / "poisoned.json"
    poisoned.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_stage_config(poisoned)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("model", "rope_theta", "rope_theta must be finite"),
        ("model", "norm_eps", "norm_eps must be finite"),
        ("init", "std", "InitSpec std must be finite"),
    ],
)
def test_stage_config_rejects_boolean_semantics_before_model_construction(
    tmp_path: Path,
    section: str,
    field: str,
    message: str,
) -> None:
    payload = json.loads(MODEL341_CONFIG.read_text(encoding="utf-8"))
    payload[section][field] = True
    poisoned = tmp_path / "poisoned-boolean.json"
    poisoned.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_stage_config(poisoned)
