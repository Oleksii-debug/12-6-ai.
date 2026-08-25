from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.scaling_experiment import controlled_specs
from twelve_six.scaling_vertical_268k import (
    CONFIG_SCHEMA,
    TARGET_PARAMETERS,
    _compare_model_runs,
    _load_config,
)


def test_target_family_reuses_research41_exact_parameter_counts() -> None:
    counts = tuple(spec.parameter_count() for spec in controlled_specs())
    assert counts[:2] == TARGET_PARAMETERS


def test_config_pins_only_geometry_comparison(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema": CONFIG_SCHEMA,
                "model_parameters": [95568, 267912],
                "token_budgets": [4096, 16384, 65536, 131072, 262144],
                "batch_size": 4,
                "sequence_length": 64,
                "seed": 1337,
                "torch_threads": 2,
            }
        ),
        encoding="utf-8",
    )
    loaded = _load_config(path)
    assert loaded["model_parameters"] == [95568, 267912]


def test_config_refuses_batch_trace_drift(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema": CONFIG_SCHEMA,
                "model_parameters": [95568, 267912],
                "token_budgets": [4096],
                "batch_size": 8,
                "sequence_length": 64,
                "seed": 1337,
                "torch_threads": 2,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="batch trace"):
        _load_config(path)


def test_equal_token_comparison_reports_direction() -> None:
    model_runs = [
        {
            "parameters": 95568,
            "checkpoints": [
                {
                    "requested_token_budget": 65536,
                    "optimized_tokens": 65772,
                    "validation_loss": 2.68,
                    "bits_per_byte": 3.87,
                }
            ],
        },
        {
            "parameters": 267912,
            "checkpoints": [
                {
                    "requested_token_budget": 65536,
                    "optimized_tokens": 65772,
                    "validation_loss": 2.28,
                    "bits_per_byte": 3.29,
                }
            ],
        },
    ]
    comparison = _compare_model_runs(model_runs)[0]
    assert comparison["winner"] == 267912
    assert comparison[
        "absolute_validation_loss_reduction_267912_vs_95568"
    ] == pytest.approx(0.4)
