#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "12-6.train195-10m-lr-beta-transfer.v1"
BLOCKED = "BLOCKED_MISSING_PREREQUISITE_AUTHORITY"
READY = "READY_FOR_PAIRED_EXECUTION"
REQUIRED_FREEZE_FIELDS = (
    "gradient_clip_norm",
    "weight_decay",
    "epsilon",
    "schedule_family",
    "batch_geometry",
)


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _finite_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def validate_config(cfg: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    errors: list[str] = []
    blockers: list[str] = []

    if cfg.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if cfg.get("execution_class") != "LOCAL_FREE" or cfg.get("paid_compute") is not False:
        errors.append("TRAIN-195 must remain LOCAL_FREE with paid_compute=false")

    prereqs = cfg.get("required_prerequisites")
    if not isinstance(prereqs, dict):
        errors.append("required_prerequisites must be an object")
        prereqs = {}

    train125 = prereqs.get("train125_lr_transfer", {})
    train194 = prereqs.get("train194_clipping", {})

    train125_fields = (
        "source_sha",
        "evidence_sha256",
        "predicted_lr",
        "prediction_method",
        "freeze_weight_decay",
        "freeze_schedule_family",
        "freeze_batch_geometry",
    )
    train194_fields = (
        "source_sha",
        "evidence_sha256",
        "accepted_clip_norm",
        "nonfinite_fail_closed_before_clipping",
    )

    def authority_ready(name: str, item: Any, fields: tuple[str, ...]) -> bool:
        if not isinstance(item, dict):
            errors.append(f"{name} must be an object")
            return False
        present = item.get("authority_present") is True
        values_present = all(item.get(field) is not None for field in fields)
        if present != values_present:
            errors.append(f"{name} authority_present must exactly reflect completion of required authority fields")
        if not present:
            blockers.append(name)
        return present and values_present

    train125_ready = authority_ready("train125_lr_transfer", train125, train125_fields)
    train194_ready = authority_ready("train194_clipping", train194, train194_fields)

    if train125_ready and not _finite_positive_number(train125.get("predicted_lr")):
        errors.append("TRAIN-125 predicted_lr must be finite and positive")
    if train194_ready and not _finite_positive_number(train194.get("accepted_clip_norm")):
        errors.append("TRAIN-194 accepted_clip_norm must be finite and positive")
    if train194_ready and train194.get("nonfinite_fail_closed_before_clipping") is not True:
        errors.append("TRAIN-194 must prove nonfinite fail-closed checks before clipping")

    freeze = cfg.get("freeze_contract")
    if not isinstance(freeze, dict):
        errors.append("freeze_contract must be an object")
        freeze = {}
    freeze_ready = all(freeze.get(field) is not None for field in REQUIRED_FREEZE_FIELDS)
    if not freeze_ready:
        blockers.append("freeze_contract")

    staged = cfg.get("staged_design")
    if not isinstance(staged, dict):
        errors.append("staged_design must be an object")
        staged = {}
    stage_a = staged.get("stage_a_lr_transfer", {})
    stage_b = staged.get("stage_b_beta2_transfer", {})
    multipliers = stage_a.get("lr_multipliers_of_train125_prediction") if isinstance(stage_a, dict) else None
    if multipliers != [0.8, 1.0, 1.25]:
        errors.append("Stage A must remain the preregistered narrow log-symmetric 0.8/1.0/1.25 LR neighborhood")
    if stage_a.get("beta2") != 0.95:
        errors.append("Stage A must hold beta2 at incumbent 0.95")
    if stage_b.get("beta2_candidates") != [0.95, 0.99]:
        errors.append("Stage B must compare only beta2 0.95 vs 0.99")
    for stage_name, stage in (("stage_a_lr_transfer", stage_a), ("stage_b_beta2_transfer", stage_b)):
        if stage.get("minimum_paired_seeds", 0) < 3:
            errors.append(f"{stage_name} requires at least three paired seeds")
        if stage.get("same_initial_weights_within_seed") is not True or stage.get("same_data_trace_within_seed") is not True:
            errors.append(f"{stage_name} must pair identical initialization and data traces")

    beta2_max = max(stage_b.get("beta2_candidates", [0.99]))
    required_updates = math.ceil(float(staged.get("minimum_second_moment_time_constants", 0)) / (1.0 - beta2_max))
    if staged.get("minimum_updates_for_beta2_0_99") != required_updates or required_updates < 300:
        errors.append("beta2=0.99 comparisons must run for at least three nominal second-moment time constants (300 updates)")

    decision = cfg.get("decision_contract", {})
    if decision.get("minimum_paired_repeats", 0) < 3 or decision.get("one_seed_can_promote") is not False:
        errors.append("decision contract must forbid promotion below three paired repeats")
    if decision.get("final_test_eligible_for_selection") is not False:
        errors.append("final-test metrics must not select research winners")
    if decision.get("nonfinite_failure_is_fatal") is not True:
        errors.append("nonfinite numerical failures must be fatal")

    runnable = train125_ready and train194_ready and freeze_ready
    absolute_lrs = staged.get("absolute_lr_candidates")
    if train125_ready:
        expected = [float(train125["predicted_lr"]) * x for x in [0.8, 1.0, 1.25]]
        if not isinstance(absolute_lrs, list) or len(absolute_lrs) != 3:
            errors.append("absolute_lr_candidates must be materialized only after TRAIN-125 authority is present")
        elif any(not math.isclose(float(a), b, rel_tol=0.0, abs_tol=max(1e-15, abs(b) * 1e-12)) for a, b in zip(absolute_lrs, expected)):
            errors.append("absolute_lr_candidates do not match the preregistered TRAIN-125-centered neighborhood")
    elif absolute_lrs is not None:
        errors.append("absolute LR candidates must remain null while TRAIN-125 authority is missing")

    expected_status = READY if runnable else BLOCKED
    if cfg.get("status") != expected_status:
        errors.append(f"status must be {expected_status} for the current prerequisite state")

    truth = cfg.get("truth_boundary", {})
    if not runnable:
        for key in ("training_executed", "numerical_result_claimed", "optimizer_promoted"):
            if truth.get(key) is not False:
                errors.append(f"{key} must remain false while prerequisites are blocked")

    return errors, sorted(set(blockers)), runnable


def make_report(cfg: dict[str, Any]) -> dict[str, Any]:
    errors, blockers, runnable = validate_config(cfg)
    report: dict[str, Any] = {
        "schema": "12-6.train195-prerequisite-gate-report.v1",
        "config_schema": cfg.get("schema"),
        "config_sha256": _sha256(cfg),
        "validation": "PASS" if not errors else "FAIL",
        "runnable": runnable and not errors,
        "scientific_status": READY if runnable and not errors else BLOCKED,
        "blockers": blockers,
        "errors": errors,
        "training_executed": False,
        "numerical_result_claimed": False,
        "optimizer_promoted": False,
    }
    report["self_sha256"] = _sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TRAIN-195 preregistration and fail closed on missing authorities.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-runnable", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    report = make_report(cfg)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(_canonical_bytes(report))
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))

    if report["validation"] != "PASS":
        return 2
    if args.require_runnable and not report["runnable"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
