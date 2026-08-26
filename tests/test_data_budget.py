from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from twelve_six.data_budget import evaluate_data_budget, required_unique_loss_tokens


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/scaling/data_budget_v1.json"
CLI = ROOT / "tools/check_data_budget.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("check_data_budget", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_model_341_targets() -> None:
    parameters = 20_613_440
    assert required_unique_loss_tokens(parameters, 5.0) == 103_067_200
    assert required_unique_loss_tokens(parameters, 20.0) == 412_268_800
    assert required_unique_loss_tokens(parameters, 50.0) == 1_030_672_000


def test_budget_blocks_one_token_short() -> None:
    result = evaluate_data_budget(
        parameter_count=20_613_440,
        unique_loss_tokens=412_268_799,
        tokens_per_parameter=20.0,
    )
    assert result.ready is False
    assert result.shortfall_tokens == 1
    assert 0.999 < result.progress_fraction < 1.0


def test_budget_passes_exact_threshold() -> None:
    result = evaluate_data_budget(
        parameter_count=20_613_440,
        unique_loss_tokens=412_268_800,
        tokens_per_parameter=20.0,
    )
    assert result.ready is True
    assert result.shortfall_tokens == 0
    assert result.observed_tokens_per_parameter == 20.0
    assert result.approximate_dense_training_flops_at_requirement == 50_993_820_672_000_000


def test_zero_materialized_tokens_are_valid_but_blocked() -> None:
    result = evaluate_data_budget(
        parameter_count=20_613_440,
        unique_loss_tokens=0,
        tokens_per_parameter=5.0,
    )
    assert result.ready is False
    assert result.progress_fraction == 0.0
    assert result.shortfall_tokens == 103_067_200


@pytest.mark.parametrize(
    ("parameter_count", "unique_loss_tokens", "tokens_per_parameter"),
    [
        (0, 0, 20.0),
        (-1, 0, 20.0),
        (True, 0, 20.0),
        (1, -1, 20.0),
        (1, True, 20.0),
        (1, 0, 0.0),
        (1, 0, -1.0),
        (1, 0, True),
    ],
)
def test_invalid_inputs_fail_closed(
    parameter_count: int,
    unique_loss_tokens: int,
    tokens_per_parameter: float,
) -> None:
    with pytest.raises(ValueError):
        evaluate_data_budget(
            parameter_count=parameter_count,
            unique_loss_tokens=unique_loss_tokens,
            tokens_per_parameter=tokens_per_parameter,
        )


def test_policy_is_self_consistent_and_keeps_compute_unauthorized() -> None:
    cli = _load_cli_module()
    policy = cli.load_policy(POLICY)
    assert policy["status"] == "RESEARCH_REFERENCE_NOT_COMPUTE_AUTHORIZATION"
    assert policy["accounting"]["source_bytes_are_capacity"] is False
    assert policy["accounting"]["replay_counts_as_new_capacity"] is False
    assert policy["policy_boundaries"]["does_not_authorize_training"] is True


def test_policy_mutation_cannot_lower_exact_target_silently(tmp_path: Path) -> None:
    cli = _load_cli_module()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy["exact_targets"]["MODEL-341-20M"]["compute_reference_20x_tokens"] -= 1
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match ratio"):
        cli.load_policy(broken)
