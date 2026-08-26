#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "12-6.train245-10m-effective-batch.v2"
BLOCKED = "BLOCKED_MISSING_TRAIN244_AUTHORITY"
READY = "READY_FOR_PAIRED_BATCH_EXECUTION"
RESULT_BLOCKED = "INSUFFICIENT_EVIDENCE"
ACCUM_GRID = [1, 2, 4]
TRAIN46_SOURCE = "6f027dd4f89b6e45ef967b256c2c8da0c2c2d4cd"
TRAIN46_RUN = 32862523852
TRAIN46_ARTIFACT = 9571718097
TRAIN46_DIGEST = "sha256:40d050a94345093884528b653f7354b5edabba9a14b2d750b6cc5cd7639fdaf9"
TRAIN46_REPORT = "20eddee576c96b4031b4ab906325bb40440cd280bd72043245f046392471d244"
FREEZE_FIELDS = (
    "learning_rate",
    "beta1",
    "beta2",
    "weight_decay",
    "epsilon",
    "gradient_clip_norm",
    "schedule_family",
    "model_spec_sha256",
    "data_identity",
    "tokenizer_identity",
    "microbatch_size",
    "sequence_length",
    "precision",
)
TRAIN244_FIELDS = FREEZE_FIELDS + (
    "source_sha",
    "evidence_sha256",
    "decision",
    "ordered_train_trace_identity",
    "optimized_token_budget",
)


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def validate_config(cfg: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    errors: list[str] = []
    blockers: list[str] = []

    if cfg.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if cfg.get("worker_id") != "TRAIN-245-10M-EFFECTIVE-BATCH-V2":
        errors.append("worker_id mismatch")
    if cfg.get("execution_class") != "LOCAL_FREE" or cfg.get("paid_compute") is not False:
        errors.append("TRAIN-245 must remain LOCAL_FREE with paid_compute=false")

    authority = cfg.get("consumed_accumulation_authority", {})
    exact_authority = {
        "source_sha": TRAIN46_SOURCE,
        "workflow_run": TRAIN46_RUN,
        "artifact_id": TRAIN46_ARTIFACT,
        "artifact_digest": TRAIN46_DIGEST,
        "report_sha256": TRAIN46_REPORT,
    }
    for key, expected in exact_authority.items():
        if authority.get(key) != expected:
            errors.append(f"TRAIN-46 authority {key} must equal retained executed evidence")
    if authority.get("workflow_conclusion") != "success":
        errors.append("TRAIN-46 workflow conclusion must be success")
    if authority.get("gradient_accumulation_semantics_correct") is not True:
        errors.append("TRAIN-46 must prove gradient accumulation semantics correct")
    if authority.get("valid_token_weighted") is not True or authority.get("variable_valid_token_rows_exercised") is not True:
        errors.append("TRAIN-46 authority must exercise exact valid-token weighting with variable token counts")
    if authority.get("effective_batch_equivalence", {}).get("pass") is not True:
        errors.append("TRAIN-46 effective-batch equivalence must pass")
    if authority.get("checkpoint_boundary_safe") is not True:
        errors.append("TRAIN-46 committed-boundary checkpoint semantics must be preserved")

    prereqs = cfg.get("required_prerequisites", {})
    train244 = prereqs.get("train244_optimizer", {}) if isinstance(prereqs, dict) else {}
    if train244.get("worker_id") != "TRAIN-244-10M-LR-BETA-V2":
        errors.append("TRAIN-244 worker identity must be exact")
    present = train244.get("authority_present") is True
    complete = all(train244.get(field) is not None for field in TRAIN244_FIELDS)
    if present != complete:
        errors.append("TRAIN-244 authority_present must exactly reflect completion of all required optimizer/model/data/trace fields")
    if not present:
        blockers.append("train244_optimizer")
    if present:
        for field in ("learning_rate", "epsilon", "gradient_clip_norm", "optimized_token_budget"):
            if not _finite_positive(train244.get(field)):
                errors.append(f"TRAIN-244 {field} must be finite and positive")
        if train244.get("microbatch_size", 0) < 1 or train244.get("sequence_length", 0) < 2:
            errors.append("TRAIN-244 microbatch_size/sequence_length must define a valid fixed microbatch geometry")
        if not (0.0 <= float(train244.get("beta1", -1)) < 1.0 and 0.0 <= float(train244.get("beta2", -1)) < 1.0):
            errors.append("TRAIN-244 beta values must lie in [0,1)")

    freeze = cfg.get("freeze_contract", {})
    freeze_complete = all(freeze.get(field) is not None for field in FREEZE_FIELDS)
    if present:
        if not freeze_complete:
            errors.append("freeze_contract must be fully materialized from TRAIN-244 before execution")
        else:
            for field in FREEZE_FIELDS:
                if freeze.get(field) != train244.get(field):
                    errors.append(f"freeze_contract.{field} must exactly equal TRAIN-244")
    elif freeze_complete:
        errors.append("freeze_contract must not be populated while TRAIN-244 authority is absent")

    grid = cfg.get("preregistered_batch_grid", {})
    if grid.get("gradient_accumulation_steps") != ACCUM_GRID:
        errors.append("batch grid must remain the preregistered accumulation-only [1,2,4] set")
    if grid.get("candidate_count") != 3 or grid.get("lcm_accumulation_steps") != 4:
        errors.append("candidate count/LCM contract mismatch")
    required_true = (
        "microbatch_shape_fixed",
        "same_initial_weights_within_seed",
        "same_ordered_microbatch_trace_within_seed",
        "same_total_optimized_loss_tokens_within_seed",
        "no_example_or_loss_token_reordering",
        "no_example_or_loss_token_reuse",
        "trace_end_must_be_common_accumulation_boundary",
    )
    for key in required_true:
        if grid.get(key) is not True:
            errors.append(f"{key} must be true")
    if grid.get("microbatch_hardware_efficiency") != "CONTROLLED_FIXED_NOT_SWEPT":
        errors.append("microbatch hardware efficiency must be controlled, not swept with effective batch")
    if "do not extrapolate to GPU" not in str(grid.get("cpu_throughput_interpretation", "")):
        errors.append("CPU throughput must explicitly forbid GPU extrapolation")

    execution = cfg.get("execution_contract", {})
    if execution.get("minimum_paired_seeds", 0) < 3:
        errors.append("at least three paired seeds are required for batch selection")
    if execution.get("optimizer_update_count_expected_to_vary") is not True:
        errors.append("optimizer update count must be reported as a varying consequence of effective batch")
    probe = execution.get("gradient_noise_probe", {})
    if probe.get("name") != "trace_covariance_over_mean_gradient_squared":
        errors.append("gradient-noise proxy definition changed")

    decision = cfg.get("decision_contract", {})
    if decision.get("paired_seed_bootstrap_resamples", 0) < 10000:
        errors.append("paired seed bootstrap must use at least 10,000 resamples")
    if decision.get("quality_equivalence_relative_bpb") != 0.005:
        errors.append("quality-equivalence band must remain preregistered at 0.5% relative BPB")
    if decision.get("gpu_claim_allowed") is not False:
        errors.append("this CPU experiment cannot make GPU throughput claims")

    runnable = present and complete and freeze_complete
    expected_status = READY if runnable else BLOCKED
    if cfg.get("status") != expected_status:
        errors.append(f"status must be {expected_status} for the current prerequisite state")

    result = cfg.get("result", {})
    truth = cfg.get("truth_boundary", {})
    candidate_results = cfg.get("candidate_results", {})
    if not runnable:
        if result.get("decision") != RESULT_BLOCKED:
            errors.append("blocked state must emit INSUFFICIENT_EVIDENCE")
        if result.get("selected_gradient_accumulation_steps") is not None or result.get("selected_effective_loss_tokens_per_update") is not None:
            errors.append("blocked state cannot select a batch")
        if any(candidate_results.get(key) is not None for key in ("accumulation_1", "accumulation_2", "accumulation_4")):
            errors.append("blocked state cannot contain numerical candidate results")
        for key in (
            "training_executed",
            "numerical_batch_comparison_claimed",
            "effective_batch_selected",
            "cpu_result_claimed",
            "gpu_result_claimed",
        ):
            if truth.get(key) is not False:
                errors.append(f"{key} must remain false while TRAIN-244 is missing")

    return errors, sorted(set(blockers)), runnable


def make_report(cfg: dict[str, Any]) -> dict[str, Any]:
    errors, blockers, runnable = validate_config(cfg)
    report: dict[str, Any] = {
        "schema": "12-6.train245-effective-batch-gate-report.v2",
        "config_schema": cfg.get("schema"),
        "config_sha256": _sha256(cfg),
        "validation": "PASS" if not errors else "FAIL",
        "runnable": runnable and not errors,
        "scientific_status": READY if runnable and not errors else BLOCKED,
        "decision": None if runnable and not errors else RESULT_BLOCKED,
        "blockers": blockers,
        "errors": errors,
        "consumed_train46_report_sha256": cfg.get("consumed_accumulation_authority", {}).get("report_sha256"),
        "training_executed": False,
        "numerical_batch_comparison_claimed": False,
        "effective_batch_selected": False,
        "cpu_result_claimed": False,
        "gpu_result_claimed": False,
    }
    report["self_sha256"] = _sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TRAIN-245 effective-batch preregistration and fail closed without TRAIN-244.")
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
