from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/runs/c01_s0_run_queue.v3.json"
SCALE = ROOT / "configs/runs/c01_stage_compute_plan_s1_s14.v1.json"
VALIDATOR = ROOT / "tools/validate_c01_compute_plan.py"

_spec = importlib.util.spec_from_file_location("validate_c01_compute_plan", VALIDATOR)
assert _spec is not None and _spec.loader is not None
_validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validator)
PlanValidationError = _validator.PlanValidationError
validate_queue = _validator.validate_queue
validate_scale_plan = _validator.validate_scale_plan


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_c01_run_queue_is_current_exact_sha_fail_closed_and_complete() -> None:
    payload = _load(QUEUE)
    validate_queue(payload)
    assert payload["primary_candidate"]["sha"] == "403831cf623120da18f0f4c62e830a352afcef91"
    assert payload["authorization"]["materially_paid_compute"] is False
    states = {job["run_id"]: job["state"] for job in payload["jobs"]}
    assert states["S0-E01-REAL-TRAIN-403831"] == "COMPLETED_EVIDENCE"
    assert states["S0-E05-REAL-RESUME-COMPOSE"] == "PREPARED_BLOCKED"
    assert states["S0-E07-D06-COMPOSE"] == "PREPARED_BLOCKED"
    assert states["S1-PROD-TRAIN"] == "PREPARED_NOT_LAUNCHED"
    assert {item["pr"] for item in payload["external_exact_green_evidence"]} == {61, 63, 65}


def test_c01_queue_rejects_stale_candidate_or_paid_launch() -> None:
    payload = _load(QUEUE)
    stale = copy.deepcopy(payload)
    stale["jobs"][0]["candidate_sha"] = "0" * 40
    with pytest.raises(PlanValidationError, match="stale/mismatched"):
        validate_queue(stale)

    paid = copy.deepcopy(payload)
    paid_job = next(job for job in paid["jobs"] if job["run_id"] == "S1-PROD-TRAIN")
    paid_job["state"] = "READY_LOCAL_FREE"
    with pytest.raises(PlanValidationError, match="paid/material"):
        validate_queue(paid)


def test_c01_queue_rejects_missing_artifact_or_retry_policy() -> None:
    payload = _load(QUEUE)
    broken = copy.deepcopy(payload)
    broken["jobs"][0]["artifacts"] = "artifacts/no-metrics-here/"
    with pytest.raises(PlanValidationError, match="artifact"):
        validate_queue(broken)

    broken = copy.deepcopy(payload)
    broken["jobs"][0]["retry"] = ""
    with pytest.raises(PlanValidationError, match="retry"):
        validate_queue(broken)


def test_c01_queue_rejects_uncomposed_external_evidence_overclaim() -> None:
    payload = _load(QUEUE)
    broken = copy.deepcopy(payload)
    broken["external_exact_green_evidence"][0]["state"] = "COMPOSED"
    with pytest.raises(PlanValidationError, match="composition"):
        validate_queue(broken)


def test_s1_s14_compute_plan_formulae_and_scenarios_validate() -> None:
    payload = _load(SCALE)
    validate_scale_plan(payload)
    assert len(payload["stages"]) == 14
    assert payload["stages"][0]["total_parameters"] == 107_856
    assert payload["stages"][-1]["total_parameters"] == 1_000_000_000_000
    assert payload["stages"][-1]["active_parameters_for_flop_estimate"] == 50_000_000_000


def test_scale_plan_rejects_flop_or_checkpoint_drift() -> None:
    payload = _load(SCALE)
    broken = copy.deepcopy(payload)
    broken["stages"][3]["scenarios"]["balanced"]["train_flops_estimate"] += 1
    with pytest.raises(PlanValidationError, match="FLOPs"):
        validate_scale_plan(broken)

    broken = copy.deepcopy(payload)
    broken["stages"][8]["bf16_weight_checkpoint_bytes"] += 2
    with pytest.raises(PlanValidationError, match="BF16"):
        validate_scale_plan(broken)
