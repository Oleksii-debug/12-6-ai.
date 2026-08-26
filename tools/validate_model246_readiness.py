#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_WORKER = "MODEL-246-10M-CONTEXT-V2"
EXPECTED_STATUS = "BLOCKED_MISSING_TERMINAL_OPTIMIZER_AND_DATA_AUTHORITIES"
EXPECTED_MODEL197_HEAD = "df224d14b11099f3a36cebc5372bb5a869c37ec2"
EXPECTED_MODEL197_RUN = 32940725569
EXPECTED_DATA230_OBSERVED_HEAD = "6d994e2aece6c44e28c1a2c344ac98b5a8fd5e08"
EXPECTED_TRAIN244_PR = 376
EXPECTED_TRAIN244_HEAD = "a4b0543738545bbb37d26446c56ab5056c982d86"
EXPECTED_TRAIN245_PR = 377
EXPECTED_TRAIN245_HEAD = "7269a96102ecaa9ecc44731abd94cd9a7252bc35"


def canonical_sha(payload: dict) -> str:
    body = dict(payload)
    body.pop("report_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["worker_id"] == EXPECTED_WORKER
    assert report["status"] == EXPECTED_STATUS
    assert report["report_sha256"] == canonical_sha(report)
    assert report["execution"] == {
        "training_started": False,
        "optimizer_updates": 0,
        "paid_compute": False,
        "local_free_only": True,
    }
    assert report["required_context_horizons"] == [256, 512, 1024]

    authorities = report["required_terminal_authorities"]
    optimizer = authorities["optimizer"]
    clipping = authorities["clipping"]
    batch = authorities["batch"]
    data = authorities["data"]

    assert optimizer["preferred_worker"] == "TRAIN-244-10M-LR-BETA-V2"
    assert optimizer["observed_pr"] == EXPECTED_TRAIN244_PR
    assert optimizer["observed_head_sha"] == EXPECTED_TRAIN244_HEAD
    assert optimizer["observed_decision"] == "INSUFFICIENT_EVIDENCE"
    assert optimizer["observed_training_started"] is False

    assert clipping["preferred_worker"] == "TRAIN-243-10M-CLIPPING-AUTHORITY-V2"
    assert clipping["observed_branch_count"] == 0

    assert batch["preferred_worker"] == "TRAIN-245-10M-EFFECTIVE-BATCH-V2"
    assert batch["observed_pr"] == EXPECTED_TRAIN245_PR
    assert batch["observed_head_sha"] == EXPECTED_TRAIN245_HEAD
    assert batch["observed_training_started"] is False
    assert "BLOCKED_MISSING_TRAIN244_AUTHORITY" in batch["observed_decision"]

    assert data["preferred_worker"] == "DATA-230-CORPUS-V03-EXTERNAL-REAL"
    assert data["observed_head_sha"] == EXPECTED_DATA230_OBSERVED_HEAD

    for key in ("optimizer", "clipping", "batch", "data"):
        assert authorities[key]["published_terminal_identity"] is None

    historical = report["historical_control"]
    assert historical["head_sha"] == EXPECTED_MODEL197_HEAD
    assert historical["workflow_run_id"] == EXPECTED_MODEL197_RUN
    assert historical["workflow_conclusion"] == "success"
    assert historical["parameter_count"] == 10_000_640
    assert historical["single_seed"] is True
    assert historical["decision"] == "KEEP_256_NO_LONGER_CONTEXT_BPB_GAIN"
    assert historical["metrics"]["256"]["clip_rate"] >= 0.98
    assert historical["metrics"]["512"]["clip_rate"] >= 0.98
    assert historical["metrics"]["1024"]["clip_rate"] >= 0.90

    # Historical MODEL-197 is a diagnostic control, never a V2 promotion result.
    assert "not_promotable_to_v2_reason" in historical
    assert report["v2_frozen_method"]["truth_boundary"].startswith("RoPE/KV")
    assert len(report["blockers"]) >= 4
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("evidence/model246/readiness.json"))
    args = parser.parse_args()
    report = validate(args.report)
    print(json.dumps({
        "status": report["status"],
        "report_sha256": report["report_sha256"],
        "training_started": report["execution"]["training_started"],
        "optimizer_updates": report["execution"]["optimizer_updates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
