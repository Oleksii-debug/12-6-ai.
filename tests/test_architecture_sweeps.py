from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.model import ModelSpec
from twelve_six.training.architecture_sweeps import (
    FFN_SCHEMA,
    HEAD_SCHEMA,
    _compose_spec,
    validate_experiment_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "configs" / "experiments" / name).read_text(encoding="utf-8"))


def test_model12_exact_iso_parameter_family_and_incumbent_control() -> None:
    config = _load("model12_ffn_ratio_1m.json")
    assert config["schema_version"] == FFN_SCHEMA
    validate_experiment_config(config)
    specs = [_compose_spec(config, candidate) for candidate in config["candidates"]]
    assert {spec.parameter_count() for spec in specs} == {992_896}
    control = next(candidate for candidate in config["candidates"] if candidate["control"])
    assert control["model_identity_sha256"] == "18284b303eb31cef5191ddb3ed4ddba5ce51789aadf4b14cc90d4226c5c527b5"
    assert {4 * spec.head_dim + spec.d_ff for spec in specs} == {416}


def test_model13_is_pure_mha_head_granularity_at_exact_parameter_count() -> None:
    config = _load("model13_head_count_100k.json")
    assert config["schema_version"] == HEAD_SCHEMA
    validate_experiment_config(config)
    specs = [_compose_spec(config, candidate) for candidate in config["candidates"]]
    assert {spec.parameter_count() for spec in specs} == {107_856}
    assert {spec.q_dim for spec in specs} == {48}
    assert {spec.kv_dim for spec in specs} == {48}
    assert all(spec.n_heads == spec.n_kv_heads for spec in specs)
    assert [(spec.n_heads, spec.head_dim) for spec in specs] == [(2, 24), (3, 16), (4, 12), (6, 8), (8, 6)]
    assert all(spec.rope_rotary_dim == spec.head_dim for spec in specs)


def test_negative_rope_and_attention_geometries_fail_closed() -> None:
    base = dict(
        schema_version=1,
        vocab_size=256,
        max_seq_len=128,
        d_model=48,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        head_dim=12,
        d_ff=128,
        rope_rotary_dim=12,
    )
    with pytest.raises(ValueError, match="even attention head_dim"):
        ModelSpec(**(base | {"head_dim": 11, "rope_rotary_dim": 10}))
    with pytest.raises(ValueError, match="cannot exceed head_dim"):
        ModelSpec(**(base | {"rope_rotary_dim": 14}))
    with pytest.raises(ValueError, match="divisible by n_kv_heads"):
        ModelSpec(**(base | {"n_heads": 5, "n_kv_heads": 2, "head_dim": 12}))


def test_candidate_identity_or_parameter_drift_is_rejected() -> None:
    config = _load("model12_ffn_ratio_1m.json")
    bad_count = copy.deepcopy(config)
    bad_count["candidates"][0]["expected_parameters"] += 1
    with pytest.raises(ValueError, match="parameter drift"):
        validate_experiment_config(bad_count)

    bad_hash = copy.deepcopy(config)
    bad_hash["candidates"][0]["model_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity drift"):
        validate_experiment_config(bad_hash)
