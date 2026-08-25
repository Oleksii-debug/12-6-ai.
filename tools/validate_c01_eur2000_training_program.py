#!/usr/bin/env python3
"""Fail-closed validator for the C01 EUR2000 training program.

This validator proves planning/configuration consistency only. It never authorizes paid
compute and deliberately refuses a launch-ready claim until the external prerequisite
and owner-authorization records are supplied.
"""

from __future__ import annotations

import argparse
import copy
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


def _require_number(value: Any, name: str, *, positive: bool = False) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{name} must be numeric")
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    if positive:
        _require(number > 0.0, f"{name} must be > 0")
    else:
        _require(number >= 0.0, f"{name} must be >= 0")
    return number


def _validate_flop_record(record: dict[str, Any], *, label: str) -> None:
    params = int(record["parameters"])
    tokens = int(record["training_tokens"])
    expected = 6 * params * tokens
    actual = int(record["planning_train_flops_6n_per_token"])
    _require(actual == expected, f"{label} FLOP estimate drift: {actual} != {expected}")


def _validate_state_bytes(record: dict[str, Any], *, label: str) -> None:
    params = int(record["parameters"])
    if "bf16_weight_bytes" in record:
        _require(int(record["bf16_weight_bytes"]) == 2 * params, f"{label} BF16 byte estimate drift")
    if "full_adam_state_planning_bytes" in record:
        _require(
            int(record["full_adam_state_planning_bytes"]) == 16 * params,
            f"{label} Adam state byte estimate drift",
        )


def validate_program(program: dict[str, Any]) -> None:
    _require(program.get("schema_version") == PROGRAM_SCHEMA, "unexpected schema_version")
    _require(program.get("repository") == "Oleksii-debug/12-6-ai.", "repository drift")

    authority = program["authority"]
    _require(authority["materially_paid_compute_authorized"] is False, "planning file must not authorize paid compute")
    _require(authority["owner_authorization_required"] is True, "owner authorization must remain required")
    _require(authority["promotion_authority"] is False, "compute program must not grant promotion authority")

    budget = program["budget"]
    ceiling = _require_number(budget["ceiling_eur"], "budget.ceiling_eur", positive=True)
    _require(ceiling == 2000.0, "EUR2000 program ceiling drift")
    parts = (
        "systems_smoke_and_gpu_calibration_cap_eur",
        "pilot_and_recipe_selection_cap_eur",
        "main_training_cap_eur",
        "storage_and_recovery_cap_eur",
        "restart_and_price_variance_reserve_eur",
    )
    allocated = sum(_require_number(budget[name], f"budget.{name}") for name in parts)
    _require(abs(allocated - ceiling) < 1e-9, f"budget allocation must equal ceiling: {allocated} != {ceiling}")
    _require(budget["automatic_overrun_allowed"] is False, "automatic budget overrun must be disabled")
    _require(budget["budget_is_ceiling_not_spend_target"] is True, "budget must remain a ceiling, not a spend target")

    pricing = program["pricing_assumptions"]
    usd_per_eur = _require_number(pricing["usd_per_eur"], "pricing.usd_per_eur", positive=True)
    a100_usd = _require_number(pricing["a100_pcie_80gb_usd_per_gpu_hour"], "pricing.a100_usd", positive=True)
    h100_usd = _require_number(pricing["h100_sxm_80gb_usd_per_gpu_hour"], "pricing.h100_usd", positive=True)
    _require(
        math.isclose(float(pricing["a100_pcie_80gb_eur_per_gpu_hour_derived"]), a100_usd / usd_per_eur, rel_tol=0.0, abs_tol=5e-4),
        "A100 EUR/hour derivation drift",
    )
    _require(
        math.isclose(float(pricing["h100_sxm_80gb_eur_per_gpu_hour_derived"]), h100_usd / usd_per_eur, rel_tol=0.0, abs_tol=5e-4),
        "H100 EUR/hour derivation drift",
    )
    _require(pricing["price_is_authoritative_quote"] is False, "snapshot price must remain an assumption until launch")
    _require(pricing["vat_storage_egress_and_capacity_premiums_included"] is False, "cost exclusions must remain explicit")

    observed = program["observed_state"]
    _require(observed["s0"]["parameters"] == 10140, "S0 parameter observation drift")
    _require(observed["s0"]["real_optimized_tokens"] == 10833, "S0 optimized-token observation drift")
    _require(observed["truth_boundary"]["gpu_training_throughput_measured"] is False, "do not fabricate GPU throughput")
    _require(observed["truth_boundary"]["scale_tokenizer_frozen"] is False, "scale tokenizer is not frozen in current evidence")
    _require(observed["truth_boundary"]["scale_corpus_frozen"] is False, "scale corpus is not frozen in current evidence")

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
    _require(any(item["ready"] is False for item in prerequisites), "planning file must expose unresolved launch blockers")

    contract = program["common_training_contract"]
    _require(contract["canonical_base"] == "random_init", "foreign/pretrained base is not allowed")
    _require(contract["precision"] == "bf16", "planned GPU precision drift")
    _require(contract["optimizer"] == "AdamW", "planned optimizer drift")
    _require(contract["device_count"] == 1, "EUR2000 recommendation is intentionally single-GPU")
    _require(contract["distributed_required"] is False, "distributed training must not be claimed necessary for this program")

    strategies = program["strategies"]
    _require(len(strategies) >= 3, "at least three strategies are required")
    recommended = [item for item in strategies if item.get("recommended") is True]
    _require(len(recommended) == 1, "exactly one strategy must be recommended")

    for strategy in strategies:
        if "parameters" in strategy:
            _validate_flop_record(strategy, label=strategy["id"])
            _validate_state_bytes(strategy, label=strategy["id"])
        for index, pilot in enumerate(strategy.get("pilot_runs", [])):
            _validate_flop_record(pilot, label=f"{strategy['id']}.pilot[{index}]")
        if "main_run" in strategy:
            _validate_flop_record(strategy["main_run"], label=f"{strategy['id']}.main")
            _validate_state_bytes(strategy["main_run"], label=f"{strategy['id']}.main")

    rec = recommended[0]
    _require(rec["id"] == "C_PILOTS_THEN_S4_BALANCED_MAIN", "recommended strategy drift")
    _require(rec["main_run"]["gpu_count"] == 1, "recommended main must remain single-GPU")
    _require(
        "observed_same_geometry_pilot_tokens_per_second" in rec["main_run"]["wall_time_formula"],
        "main wall time must derive from same-geometry GPU pilot",
    )

    smoke = program["smoke_and_pilot"]["same_geometry_system_smoke"]
    _require(float(smoke["hard_cost_cap_eur"]) <= float(budget["systems_smoke_and_gpu_calibration_cap_eur"]), "smoke cap exceeds allocation")
    _require(smoke["must_checkpoint_reload_and_resume"] is True, "smoke must prove checkpoint/reload/resume")
    _require(smoke["must_run_heldout_eval"] is True, "smoke must prove held-out evaluation")
    _require(smoke["quality_claim_allowed"] is False, "systems smoke cannot imply quality")

    recovery = program["checkpoint_recovery"]
    retained = int(recovery["s4_conservative_state_payload_bytes_per_retained_checkpoint"]) * int(recovery["s4_max_retained_checkpoint_count"])
    _require(retained == int(recovery["s4_conservative_retained_payload_bytes"]), "retained checkpoint payload estimate drift")
    _require(recovery["blind_retry"] is False, "blind retry must remain disabled")

    required_fields = set(program["owner_authorization_record_required_fields"])
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
    _require(mandatory.issubset(required_fields), "owner authorization record is missing mandatory fields")


def launch_blockers(program: dict[str, Any]) -> list[str]:
    blockers = [item["id"] for item in program["launch_prerequisites"] if item["ready"] is not True]
    if program["authority"]["materially_paid_compute_authorized"] is not True:
        blockers.append("AUTHORITY_PAID_COMPUTE_FALSE")
    if program["observed_state"]["truth_boundary"]["gpu_training_throughput_measured"] is not True:
        blockers.append("NO_MEASURED_GPU_THROUGHPUT")
    return blockers


def load_program(path: Path = PROGRAM_PATH) -> dict[str, Any]:
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
