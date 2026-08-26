#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import deque
from pathlib import Path
from typing import Any

EXPECTED_LABELS = ["self-hosted", "linux", "x64", "gpu", "cuda", "twelve-six-ai"]
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("campaign manifest must be a JSON object")
    return value


def _topological_order(stages: list[dict[str, Any]]) -> list[str]:
    ids = [str(stage["id"]) for stage in stages]
    if len(set(ids)) != len(ids):
        raise ValueError("stage ids must be unique")
    known = set(ids)
    indegree = {stage_id: 0 for stage_id in ids}
    children: dict[str, list[str]] = {stage_id: [] for stage_id in ids}
    for stage in stages:
        stage_id = str(stage["id"])
        for dep in stage.get("depends_on", []):
            if dep not in known:
                raise ValueError(f"unknown dependency {dep!r} for {stage_id}")
            indegree[stage_id] += 1
            children[dep].append(stage_id)
    queue = deque(stage_id for stage_id in ids if indegree[stage_id] == 0)
    order: list[str] = []
    while queue:
        stage_id = queue.popleft()
        order.append(stage_id)
        for child in children[stage_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(ids):
        raise ValueError("campaign DAG contains a cycle")
    return order


def _descriptor_digest(stage: dict[str, Any], parent: dict[str, Any]) -> str:
    descriptor = {
        "stage": stage["id"],
        "parent": stage["parent"],
        "source_sha": parent.get("sha"),
        "artifact": stage.get("artifact"),
        "reuse_artifact_from": stage.get("reuse_artifact_from"),
        "depends_on": stage.get("depends_on", []),
        "required_terminal": stage.get("required_terminal"),
    }
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_dry_run(manifest: dict[str, Any]) -> dict[str, Any]:
    policy = manifest["policy"]
    if policy.get("provision_hardware") is not False:
        raise ValueError("hardware provisioning must remain disabled")
    if policy.get("purchase_compute") is not False:
        raise ValueError("compute purchase must remain disabled")
    if policy.get("paid_compute_authorized") is not False:
        raise ValueError("paid compute must remain unauthorized")
    if policy.get("credentials_in_repo") is not False:
        raise ValueError("credentials_in_repo must be false")
    if policy.get("cpu_dry_run_target_measurements") != 0:
        raise ValueError("CPU dry run must execute zero target-device measurements")

    runner = manifest["runner"]
    if runner.get("labels") != EXPECTED_LABELS:
        raise ValueError("self-hosted label contract drift")
    if runner.get("purpose_environment") != "linux-x86_64-cuda-training":
        raise ValueError("CUDA purpose-environment drift")

    parents: dict[str, dict[str, Any]] = manifest["parents"]
    stages: list[dict[str, Any]] = manifest["stages"]
    order = _topological_order(stages)
    by_id = {stage["id"]: stage for stage in stages}
    statuses: dict[str, str] = {}
    details: list[dict[str, Any]] = []

    for stage_id in order:
        stage = by_id[stage_id]
        parent_name = stage["parent"]
        if parent_name not in parents:
            raise ValueError(f"unknown parent {parent_name!r} for {stage_id}")
        parent = parents[parent_name]
        sha = parent.get("sha")
        deps = stage.get("depends_on", [])

        if sha is None:
            status = "BLOCKED_PARENT_NOT_FOUND"
        elif not SHA40.fullmatch(str(sha)):
            raise ValueError(f"invalid parent SHA for {parent_name}")
        elif stage.get("kind") == "target_device" and parent.get("target_executor") is None and parent_name != "GPU-200":
            status = "BLOCKED_PARENT_TARGET_EXECUTOR_MISSING"
        elif any(not statuses[dep].startswith("READY") for dep in deps):
            status = "BLOCKED_DEPENDENCY"
        elif stage.get("kind") == "topology_conditional":
            status = "READY_IF_TOPOLOGY_EXISTS"
        elif stage.get("kind") == "artifact_gate":
            reuse = stage.get("reuse_artifact_from")
            if reuse not in statuses:
                raise ValueError(f"artifact gate {stage_id} has invalid reuse source")
            status = "READY_ARTIFACT_GATE"
        else:
            status = "READY_FOR_MANUAL_TARGET"

        statuses[stage_id] = status
        details.append(
            {
                "id": stage_id,
                "status": status,
                "parent": parent_name,
                "source_sha": sha,
                "depends_on": deps,
                "artifact": stage.get("artifact"),
                "reuse_artifact_from": stage.get("reuse_artifact_from"),
                "artifact_descriptor_sha256": _descriptor_digest(stage, parent),
                "target_measurements_executed": 0,
            }
        )

    blockers = [item for item in details if item["status"].startswith("BLOCKED")]
    return {
        "schema": "12-6.compute252-gpu-campaign-dry-run.v1",
        "worker_id": manifest["worker_id"],
        "mode": "CPU_DRY_RUN",
        "dag_valid": True,
        "target_device_measurements_executed": 0,
        "paid_compute_authorized": False,
        "runner_labels": runner["labels"],
        "purpose_environment": runner["purpose_environment"],
        "topological_order": order,
        "stages": details,
        "full_campaign_ready": not blockers,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/compute/compute252_gpu_campaign.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/compute252/dry_run.json"),
    )
    args = parser.parse_args()

    report = build_dry_run(_load(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
