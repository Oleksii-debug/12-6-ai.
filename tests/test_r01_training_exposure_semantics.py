from __future__ import annotations

import copy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest

from twelve_six.training_exposure import ExposureBudgetError, assess_training_exposure

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_r01_training_exposure_semantics.py"
SPEC = importlib.util.spec_from_file_location("r01_exposure_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_contract_validates_against_exact_r01_blob() -> None:
    result = VALIDATOR.validate_contract(ROOT)
    assert result["status"] == "PASS"
    assert result["parameter_count"] == 20_613_440
    assert result["planned_tokens_per_parameter"] == [10, 20, 40]
    assert result["long_training_authorized"] is False


@pytest.mark.parametrize(
    ("requested", "expected_epochs"),
    [
        (206_134_400, Fraction(64_417, 6_250)),
        (412_268_800, Fraction(64_417, 3_125)),
        (824_537_600, Fraction(128_834, 3_125)),
    ],
)
def test_20m_floor_blocks_r01_arms_without_repeat_policy(
    requested: int,
    expected_epochs: Fraction,
) -> None:
    result = assess_training_exposure(
        unique_loss_positions=20_000_000,
        requested_total_exposures=requested,
    )
    assert result.status == "BLOCKED_REPEAT_POLICY_REQUIRED"
    assert Fraction(
        result.effective_epochs_numerator,
        result.effective_epochs_denominator,
    ) == expected_epochs
    assert result.training_authorized is False


def test_four_epoch_reference_does_not_cover_10x_arm() -> None:
    result = assess_training_exposure(
        unique_loss_positions=20_000_000,
        requested_total_exposures=206_134_400,
        max_repeat_epochs=4,
    )
    assert result.status == "BLOCKED_REPEAT_CAP_EXCEEDED"
    assert result.repeat_exposures == 186_134_400
    assert result.training_authorized is False


def test_explicit_repeat_cap_can_bound_exposure_without_relabelling_unique() -> None:
    result = assess_training_exposure(
        unique_loss_positions=20_000_000,
        requested_total_exposures=60_000_000,
        max_repeat_epochs=3,
    )
    assert result.status == "WITHIN_PREREGISTERED_REPEAT_CAP"
    assert result.unique_loss_positions == 20_000_000
    assert result.repeat_exposures == 40_000_000
    assert result.training_authorized is False


def test_request_inside_unique_ledger_needs_no_repeat_policy() -> None:
    result = assess_training_exposure(
        unique_loss_positions=20_000_000,
        requested_total_exposures=10_000_000,
    )
    assert result.status == "WITHIN_UNIQUE_LEDGER"
    assert result.repeat_exposures == 0
    assert result.repeat_policy_required is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unique_loss_positions": 0, "requested_total_exposures": 1},
        {"unique_loss_positions": True, "requested_total_exposures": 1},
        {"unique_loss_positions": 1, "requested_total_exposures": 0},
        {
            "unique_loss_positions": 1,
            "requested_total_exposures": 2,
            "max_repeat_epochs": "1/2",
        },
    ],
)
def test_invalid_exposure_requests_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ExposureBudgetError):
        assess_training_exposure(**kwargs)


def test_contract_identity_mutation_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "configs" / "research" / "r01_training_exposure_semantics_v1.json"
    contract = json.loads(source.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(contract)
    mutated["hard_rules"]["repeat_exposures_are_not_unique"] = False

    config_dir = tmp_path / "configs" / "research"
    config_dir.mkdir(parents=True)
    target = config_dir / source.name
    target.write_text(json.dumps(mutated), encoding="utf-8")

    r01_source = ROOT / contract["base_r01"]["path"]
    r01_target = tmp_path / contract["base_r01"]["path"]
    r01_target.write_bytes(r01_source.read_bytes())

    with pytest.raises(ValueError, match="contract_sha256 mismatch"):
        VALIDATOR.validate_contract(tmp_path)
