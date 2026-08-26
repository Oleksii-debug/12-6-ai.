from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.run_compute252_gpu_campaign import EXPECTED_LABELS, build_dry_run

MANIFEST_PATH = Path("configs/compute/compute252_gpu_campaign.json")
EXPECTED_ORDER = [
    "hardware_environment_preflight",
    "precision_decision",
    "ten_m_semantic_smoke",
    "ten_m_performance",
    "checkpoint",
    "activation_checkpointing",
    "compile",
    "hundred_m_qualification",
    "two_gpu_fsdp",
    "two_gpu_dcp_recovery",
]


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _statuses(report: dict[str, object]) -> dict[str, str]:
    stages = report["stages"]
    assert isinstance(stages, list)
    return {str(stage["id"]): str(stage["status"]) for stage in stages}


def test_cpu_dry_run_validates_order_and_executes_zero_target_measurements() -> None:
    report = build_dry_run(_manifest())
    assert report["dag_valid"] is True
    assert report["mode"] == "CPU_DRY_RUN"
    assert report["target_device_measurements_executed"] == 0
    assert report["paid_compute_authorized"] is False
    assert report["runner_labels"] == EXPECTED_LABELS
    assert report["purpose_environment"] == "linux-x86_64-cuda-training"
    assert report["topological_order"] == EXPECTED_ORDER


def test_current_missing_parent_capabilities_fail_closed_and_propagate() -> None:
    report = build_dry_run(_manifest())
    statuses = _statuses(report)

    assert statuses["hardware_environment_preflight"] == "READY_FOR_MANUAL_TARGET"
    assert statuses["precision_decision"] == "BLOCKED_PARENT_TARGET_EXECUTOR_MISSING"
    assert statuses["ten_m_semantic_smoke"] == "BLOCKED_DEPENDENCY"
    assert statuses["ten_m_performance"] == "BLOCKED_DEPENDENCY"
    assert statuses["checkpoint"] == "BLOCKED_DEPENDENCY"
    assert statuses["activation_checkpointing"] == "BLOCKED_DEPENDENCY"
    assert statuses["compile"] == "BLOCKED_PARENT_NOT_FOUND"
    assert statuses["hundred_m_qualification"] == "BLOCKED_DEPENDENCY"
    assert statuses["two_gpu_fsdp"] == "BLOCKED_PARENT_TARGET_EXECUTOR_MISSING"
    assert statuses["two_gpu_dcp_recovery"] == "BLOCKED_DEPENDENCY"
    assert report["full_campaign_ready"] is False


def test_artifact_identities_are_bound_to_parent_source_and_dependencies() -> None:
    report = build_dry_run(_manifest())
    stages = report["stages"]
    assert isinstance(stages, list)
    digests = [stage["artifact_descriptor_sha256"] for stage in stages]
    assert all(isinstance(value, str) and len(value) == 64 for value in digests)
    assert len(set(digests)) == len(digests)


def test_policy_drift_cannot_authorize_paid_compute() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["policy"]["paid_compute_authorized"] = True
    with pytest.raises(ValueError, match="paid compute"):
        build_dry_run(manifest)


def test_runner_label_drift_is_rejected() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["runner"]["labels"] = ["self-hosted", "gpu"]
    with pytest.raises(ValueError, match="label contract"):
        build_dry_run(manifest)


def test_cycle_is_rejected() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["stages"][0]["depends_on"] = ["two_gpu_dcp_recovery"]
    with pytest.raises(ValueError, match="cycle"):
        build_dry_run(manifest)
