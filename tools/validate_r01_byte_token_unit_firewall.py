#!/usr/bin/env python3
"""Validate the R01 byte/token scaling-unit firewall."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.r01-byte-token-unit-firewall.v1"
EXPECTED_UNIT = "UNIQUE_AUTHORIZED_UTF8_BYTE_LOSS_POSITIONS"
EXPECTED_STATUS = "UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION"
EXPECTED_REFERENCE = [10, 20, 40]
EXPECTED_PILOT_POSITIONS = 20_000_000
REQUIRED_CALIBRATIONS = {
    "tokenizer_efficiency_by_ua_en_code",
    "semantic_context_span_calibration",
    "flop_normalized_byte_vs_subword_ablation",
    "heldout_learning_curves",
}


def validate_firewall(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(data.get("schema") == EXPECTED_SCHEMA, "schema mismatch")
    require(data.get("status") == "FAIL_CLOSED", "status must remain FAIL_CLOSED")
    require(data.get("canonical_byte_exposure_unit") == EXPECTED_UNIT, "byte exposure unit drift")
    require(
        data.get("source_reported_token_ratios_are_execution_budgets") is False,
        "source-reported token ratios must never become execution budgets",
    )
    require(
        data.get("source_reported_token_ratios_may_be_converted_to_byte_positions") is False,
        "source-token ratios must not be converted directly to byte positions",
    )
    require(
        data.get("source_reported_tokens_per_parameter_reference_only") == EXPECTED_REFERENCE,
        "reference-only token ratio vector drift",
    )
    require(
        data.get("engineering_early_learning_pilot_unique_loss_positions")
        == EXPECTED_PILOT_POSITIONS,
        "engineering pilot position count drift",
    )
    require(
        data.get("engineering_pilot_is_science_complete_budget") is False,
        "engineering pilot must not masquerade as a science-complete budget",
    )
    require(
        data.get("science_complete_20m_byte_position_budget") is None,
        "science-complete byte-position budget must remain undefined before calibration",
    )
    require(
        data.get("science_complete_budget_status") == EXPECTED_STATUS,
        "science-complete budget status drift",
    )
    calibrations = data.get("required_calibrations")
    require(isinstance(calibrations, list), "required_calibrations must be a list")
    if isinstance(calibrations, list):
        require(
            REQUIRED_CALIBRATIONS.issubset(set(calibrations)),
            "required calibration set is incomplete",
        )
    require(
        data.get("promotion_to_100m_allowed_from_reference_ratio_alone") is False,
        "100M promotion cannot follow from a reference token ratio alone",
    )
    require(data.get("long_training_authorized") is False, "long training must remain blocked")
    require(data.get("paid_compute_authorized") is False, "paid compute must remain blocked")
    return errors


def main(argv: list[str]) -> int:
    path = (
        Path(argv[1])
        if len(argv) > 1
        else Path("configs/research/r01_byte_token_unit_firewall_v1.json")
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print("FAIL: firewall root must be an object")
        return 1
    errors = validate_firewall(data)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: R01 byte/token scaling-unit firewall is fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
