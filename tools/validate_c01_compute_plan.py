from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

QUEUE_SCHEMA = "12-6.c01.s0-run-queue.v3"
SCALE_SCHEMA = "12-6.c01.stage-compute-plan.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."
SHA40 = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_PHASES = {"engineering_validation", "scale_experiment", "production_like_training"}
PAID_CLASSES = {"PAID_SMALL", "PAID_LARGE", "PAID_OR_MATERIAL"}
NON_LAUNCH_STATES = {"PREPARED_NOT_LAUNCHED", "PREPARED_BLOCKED"}
REQUIRED_ARTIFACT_MARKERS = (
    "manifest.json",
    "hardware.json",
    "logs/",
    "metrics.jsonl",
    "checkpoints/",
    "eval/",
)


class PlanValidationError(ValueError):
    """Raised when a compute/run-control plan fails closed."""


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PlanValidationError("top-level payload must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanValidationError(message)


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and SHA40.fullmatch(value) is not None


def validate_queue(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == QUEUE_SCHEMA, "unexpected queue schema")
    _require(payload.get("repository") == REPOSITORY, "repository identity mismatch")

    primary = payload.get("primary_candidate")
    _require(isinstance(primary, dict), "primary_candidate missing")
    candidate_sha = primary.get("sha")
    _require(_is_sha(candidate_sha), "primary candidate must be an exact 40-hex SHA")
    _require(primary.get("exact_head_green") is True, "primary evidence must be exact-head green")

    authorization = payload.get("authorization")
    _require(isinstance(authorization, dict), "authorization missing")
    _require(
        authorization.get("materially_paid_compute") is False,
        "this run must remain fail-closed for materially paid compute",
    )
    _require(authorization.get("promotion_allowed") is False, "C01 may not promote")

    external = payload.get("external_exact_green_evidence")
    _require(isinstance(external, list), "external evidence must be a list")
    for item in external:
        _require(isinstance(item, dict), "external evidence entry must be an object")
        _require(_is_sha(item.get("sha")), "external evidence SHA must be exact")
        _require(
            item.get("state") == "EVIDENCE_AVAILABLE_NOT_COMPOSED",
            "external evidence must not imply composition/launch authority",
        )

    jobs = payload.get("jobs")
    _require(isinstance(jobs, list) and jobs, "queue must contain jobs")
    run_ids: set[str] = set()
    for job in jobs:
        _require(isinstance(job, dict), "job must be an object")
        run_id = job.get("run_id")
        _require(isinstance(run_id, str) and run_id, "run_id required")
        _require(run_id not in run_ids, f"duplicate run_id: {run_id}")
        run_ids.add(run_id)
        _require(job.get("phase") in ALLOWED_PHASES, f"{run_id}: invalid phase")
        _require(isinstance(job.get("state"), str) and job["state"], f"{run_id}: state")
        _require(isinstance(job.get("command"), str) and job["command"], f"{run_id}: command")

        job_sha = job.get("candidate_sha")
        if job_sha is None:
            _require(
                job.get("state") in NON_LAUNCH_STATES,
                f"{run_id}: unresolved SHA allowed only for non-launchable jobs",
            )
        else:
            _require(_is_sha(job_sha), f"{run_id}: candidate SHA must be exact")
            _require(job_sha == candidate_sha, f"{run_id}: stale/mismatched candidate SHA")

        artifacts = job.get("artifacts")
        _require(isinstance(artifacts, str) and artifacts, f"{run_id}: artifacts missing")
        for marker in REQUIRED_ARTIFACT_MARKERS:
            _require(marker in artifacts, f"{run_id}: artifact location missing {marker}")

        cancels = job.get("cancellation_conditions")
        failures = job.get("failure_criteria")
        retry = job.get("retry")
        _require(isinstance(cancels, list) and cancels, f"{run_id}: cancellation criteria missing")
        _require(isinstance(failures, list) and failures, f"{run_id}: failure criteria missing")
        _require(isinstance(retry, str) and retry, f"{run_id}: retry policy missing")

        if job.get("compute_class") in PAID_CLASSES:
            _require(
                job.get("state") == "PREPARED_NOT_LAUNCHED",
                f"{run_id}: paid/material compute cannot be launch-ready",
            )

    measured = payload.get("measured_s0")
    _require(isinstance(measured, dict), "measured_s0 missing")
    _require(measured.get("source_sha") == candidate_sha, "measured S0 is stale")
    tokens = measured.get("optimized_tokens")
    wall = measured.get("wall_seconds")
    throughput = measured.get("optimized_tokens_per_wall_second")
    _require(isinstance(tokens, int) and tokens > 0, "bad token count")
    _require(isinstance(wall, (int, float)) and wall > 0, "bad wall time")
    _require(
        isinstance(throughput, (int, float))
        and math.isclose(throughput, tokens / wall, rel_tol=1e-12),
        "throughput must be derived from tokens/wall_seconds",
    )
    _require(measured.get("validation_optimized_tokens") == 0, "held-out optimization leak")


def validate_scale_plan(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == SCALE_SCHEMA, "unexpected scale schema")
    _require(payload.get("repository") == REPOSITORY, "repository identity mismatch")
    _require(
        payload.get("status") == "PLANNING_ONLY_NOT_COMPUTE_AUTHORIZATION",
        "scale plan must not imply compute authorization",
    )
    stages = payload.get("stages")
    _require(isinstance(stages, list), "stages must be a list")
    _require([row.get("stage") for row in stages] == [f"S{i}" for i in range(1, 15)], "need S1-S14")

    for row in stages:
        stage = row["stage"]
        total = row.get("total_parameters")
        active = row.get("active_parameters_for_flop_estimate")
        _require(isinstance(total, int) and total > 0, f"{stage}: total parameters")
        _require(isinstance(active, int) and 0 < active <= total, f"{stage}: active parameters")
        _require(row.get("bf16_weight_checkpoint_bytes") == 2 * total, f"{stage}: BF16 bytes")
        _require(row.get("full_adam_training_state_bytes") == 16 * total, f"{stage}: Adam bytes")
        expected_hbm = math.ceil(16 * total / 0.70)
        _require(row.get("aggregate_hbm_min_with_30pct_reserve_bytes") == expected_hbm, f"{stage}: HBM lower bound")
        _require(row.get("inference_flops_per_token") == 2 * active, f"{stage}: inference FLOPs")

        scenarios = row.get("scenarios")
        _require(isinstance(scenarios, dict), f"{stage}: scenarios")
        _require(set(scenarios) == {"cheap", "balanced", "fast"}, f"{stage}: scenario set")
        prior_tokens = 0
        for name in ("cheap", "balanced", "fast"):
            scenario = scenarios[name]
            tokens = scenario.get("training_tokens")
            flops = scenario.get("train_flops_estimate")
            _require(isinstance(tokens, int) and tokens > prior_tokens, f"{stage}/{name}: tokens")
            _require(flops == 6 * active * tokens, f"{stage}/{name}: FLOPs")
            prior_tokens = tokens

        gpu_counts = row.get("min_gpus_state_only")
        _require(isinstance(gpu_counts, dict), f"{stage}: gpu counts")
        for label, hbm_gb in (("24gb", 24), ("80gb", 80), ("192gb", 192)):
            expected_count = math.ceil(expected_hbm / (hbm_gb * 1_000_000_000))
            _require(gpu_counts.get(label) == expected_count, f"{stage}: {label} lower bound")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate C01/D13 run queue and scale plan.")
    parser.add_argument("--queue", default="configs/runs/c01_s0_run_queue.v3.json")
    parser.add_argument("--scale", default="configs/runs/c01_stage_compute_plan_s1_s14.v1.json")
    args = parser.parse_args()
    validate_queue(_load(args.queue))
    validate_scale_plan(_load(args.scale))
    print("C01/D13 compute plans: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
