#!/usr/bin/env python3
"""Fail-closed validator for the C01 EUR2000 training program."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PROGRAM_SCHEMA = "12-6.c01.eur2000-training-program.v1"
PROGRAM_PATH = Path("configs/runs/c01_eur2000_training_program.v1.json")


class ProgramValidationError(ValueError):
    """Raised when the program violates a fail-closed invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProgramValidationError(message)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    valid_type = not isinstance(value, bool) and isinstance(value, (int, float))
    _require(valid_type, f"{name} must be numeric")
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    if positive:
        _require(number > 0.0, f"{name} must be > 0")
    else:
        _require(number >= 0.0, f"{name} must be >= 0")
    return number


def _validate_flops(record: dict[str, Any], *, label: str) -> None:
    params = int(record["parameters"])
    tokens = int(record["training_tokens"])
    expected = 6 * params * tokens
    actual = int(record["planning_train_flops_6n_per_token"])
    _require(actual == expected, f"{label} FLOP estimate drift: {actual} != {expected}")


def _validate_state_bytes(record: dict[str, Any], *, label: str) -> None:
    params = int(record["parameters"])
    if "bf16_weight_bytes" in record:
        actual = int(record["bf16_weight_bytes"])
        _require(actual == 2 * params, f"{label} BF16 byte estimate drift")
    if "full_adam_state_planning_bytes" in record:
        actual = int(record["full_adam_state_planning_bytes"])
        _require(actual == 16 * params, f"{label} Adam state byte estimate drift")


def _validate_budget(program: dict[str, Any]) -> None:
    budget = program["budget"]
    ceiling = _number(budget["ceiling_eur"], "budget.ceiling_eur", positive=True)
    _require(ceiling == 2000.0, "EUR2000 program ceiling drift")
    parts = (
        "systems_smoke_and_gpu_calibration_cap_eur",
        "pilot_and_recipe_selection_cap_eur",
        "main_training_cap_eur",
        "storage_and_recovery_cap_eur",
        "restart_and_price_variance_reserve_eur",
    )
    allocated = sum(_number(budget[name], f"budget.{name}") for name in parts)
    message = f"budget allocation must equal ceiling: {allocated} != {ceiling}"
    _require(abs(allocated - ceiling) < 1e-9, message)
    _require(not budget["automatic_overrun_allowed"], "automatic budget overrun enabled")
    _require(budget["budget_is_ceiling_not_spend_target"], "budget is not a ceiling")


def _validate_pricing(program: dict[str, Any]) -> None:
    pricing = program["pricing_assumptions"]
    usd_per_eur = _number(pricing["usd_per_eur"], "pricing.usd_per_eur", positive=True)
    a100_usd = _number(
        pricing["a100_pcie_80gb_usd_per_gpu_hour"],
        "pricing.a100_usd",
        positive=True,
    )
    h100_usd = _number(
        pricing["h100_sxm_80gb_usd_per_gpu_hour"],
        "pricing.h100_usd",
        positive=True,
    )
    a100_eur = float(pricing["a100_pcie_80gb_eur_per_gpu_hour_derived"])
    h100_eur = float(pricing["h100_sxm_80gb_eur_per_gpu_hour_derived"])
    close = dict(rel_tol=0.0, abs_tol=5e-4)
    _require(math.isclose(a100_eur, a100_usd / usd_per_eur, **close), "A100 EUR drift")
    _require(math.isclose(h100_eur, h100_usd / usd_per_eur, **close), "H100 EUR drift")
    _require(not pricing["price_is_authoritative_quote"], "snapshot became a launch quote")
    _require(
        not pricing["vat_storage_egress_and_capacity_premiums_included"],
        "pricing exclusions became implicit",
    )


def _validate_strategies(program: dict[str, Any]) -> None:
    strategies = program["strategies"]
    _require(len(strategies) >= 3, "at least three strategies are required")
    recommended = [item for item in strategies if item.get("recommended") is True]
    _require(len(recommended) == 1, "exactly one strategy must be recommended")

    for strategy in strategies:
        if "parameters" in strategy:
            _validate_flops(strategy, label=strategy["id"])
            _validate_state_bytes(strategy, label=strategy["id"])
        for index, pilot in enumerate(strategy.get("pilot_runs", [])):
            _validate_flops(pilot, label=f"{strategy['id']}.pilot[{index}]")
        if "main_run" in strategy:
            _validate_flops(strategy["main_run"], label=f"{strategy['id']}.main")
            _validate_state_bytes(strategy["main_run"], label=f"{strategy['id']}.main")

    selected = recommended[0]
    _require(selected["id"] == "C_PILOTS_THEN_S4_BALANCED_MAIN", "recommendation drift")
    _require(selected["main_run"]["gpu_count"] == 1, "recommended main is not single GPU")
    wall_formula = selected["main_run"]["wall_time_formula"]
    _require(
        "observed_same_geometry_pilot_tokens_per_second" in wall_formula,
        "main wall time does not derive from the GPU pilot",
    )


def validate_program(program: dict[str, Any]) -> None:
    """Validate planning consistency without granting launch authority."""
    _require(program.get("schema_version") == PROGRAM_SCHEMA, "unexpected schema_version")
    _require(program.get("repository") == "Oleksii-debug/12-6-ai.", "repository drift")

    authority = program["authority"]
    _require(
        authority["materially_paid_compute_authorized"] is False,
        "planning file must not authorize paid compute",
    )
    _require(authority["owner_authorization_required"] is True, "owner gate removed")
    _require(authority["promotion_authority"] is False, "compute plan grants promotion")

    _validate_budget(program)
    _validate_pricing(program)

    observed = program["observed_state"]
    _require(observed["s0"]["parameters"] == 10140, "S0 parameter observation drift")
    _require(observed["s0"]["real_optimized_tokens"] == 10833, "S0 token drift")
    truth = observed["truth_boundary"]
    _require(not truth["gpu_training_throughput_measured"], "fabricated GPU throughput")
    _require(not truth["scale_tokenizer_frozen"], "scale tokenizer incorrectly frozen")
    _require(not truth["scale_corpus_frozen"], "scale corpus incorrectly frozen")

    prerequisites = program["launch_prerequisites"]
    ids = [item["id"] for item in prerequisites]
    expected_ids = [
        "P01_SCALE_DATA_IDENTITY",
        "P02_SCALE_TOKENIZER_IDENTITY",
        "P03_GPU_RUNTIME_EXACT_LOCK",
        "P04_SAME_GEOMETRY_GPU_SMOKE",
        "P05_RESUME_AND_HELDOUT_EVAL",
        "P06_OWNER_PAID_COMPUTE_AUTHORIZATION",
    ]
    _require(ids == expected_ids, "launch prerequisite set/order drift")
    _require(any(not item["ready"] for item in prerequisites), "launch blockers hidden")

    contract = program["common_training_contract"]
    _require(contract["canonical_base"] == "random_init", "non-random base introduced")
    _require(contract["precision"] == "bf16", "planned precision drift")
    _require(contract["optimizer"] == "AdamW", "planned optimizer drift")
    _require(contract["device_count"] == 1, "recommendation is no longer single GPU")
    _require(not contract["distributed_required"], "distributed training falsely required")

    _validate_strategies(program)

    smoke = program["smoke_and_pilot"]["same_geometry_system_smoke"]
    smoke_cap = float(smoke["hard_cost_cap_eur"])
    allocation = float(program["budget"]["systems_smoke_and_gpu_calibration_cap_eur"])
    _require(smoke_cap <= allocation, "smoke cap exceeds allocation")
    _require(smoke["must_checkpoint_reload_and_resume"], "smoke omits resume")
    _require(smoke["must_run_heldout_eval"], "smoke omits held-out evaluation")
    _require(not smoke["quality_claim_allowed"], "systems smoke implies model quality")

    recovery = program["checkpoint_recovery"]
    one = int(recovery["s4_conservative_state_payload_bytes_per_retained_checkpoint"])
    count = int(recovery["s4_max_retained_checkpoint_count"])
    _require(
        one * count == int(recovery["s4_conservative_retained_payload_bytes"]),
        "retained checkpoint payload estimate drift",
    )
    _require(not recovery["blind_retry"], "blind retry enabled")

    required = set(program["owner_authorization_record_required_fields"])
    mandatory = {
        "approved_strategy_id",
        "approved_eur_cap",
        "provider",
        "gpu_sku",
        "refreshed_eur_per_gpu_hour",
        "quote_timestamp_utc",
        "git_sha",
        "modelspec_sha256",
        "dataset_identity_sha256",
        "tokenizer_identity_sha256",
        "gpu_runtime_lock_sha256",
        "pilot_evidence_sha256",
        "projected_wall_hours_from_measured_pilot",
        "projected_compute_cost_eur",
        "owner_authorized_paid_compute",
    }
    _require(mandatory.issubset(required), "authorization record fields incomplete")


def launch_blockers(program: dict[str, Any]) -> list[str]:
    """Return current blockers. An empty list is required before any paid launch."""
    blockers = [
        item["id"] for item in program["launch_prerequisites"] if item["ready"] is not True
    ]
    if program["authority"]["materially_paid_compute_authorized"] is not True:
        blockers.append("AUTHORITY_PAID_COMPUTE_FALSE")
    truth = program["observed_state"]["truth_boundary"]
    if truth["gpu_training_throughput_measured"] is not True:
        blockers.append("NO_MEASURED_GPU_THROUGHPUT")
    return blockers


def load_program(path: Path = PROGRAM_PATH) -> dict[str, Any]:
    """Load the machine-readable program."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, default=PROGRAM_PATH)
    parser.add_argument("--assert-launch-ready", action="store_true")
    args = parser.parse_args()

    program = load_program(args.program)
    validate_program(program)
    blockers = launch_blockers(program)
    if args.assert_launch_ready and blockers:
        raise ProgramValidationError("launch blocked: " + ", ".join(blockers))

    print(
        json.dumps(
            {
                "schema_version": PROGRAM_SCHEMA,
                "planning_valid": True,
                "paid_compute_authorized": False,
                "launch_ready": not blockers,
                "launch_blockers": blockers,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
