#!/usr/bin/env python3
"""Apply the frozen EVAL-139 checkpoint-selection rule to LEARN-122.

This tool is intentionally outcome-blind at source time. It consumes only the
chronological 500K selection-validation BPB trajectory after training and emits
one deterministic selection decision. Raw best and final checkpoints are
retained as post-hoc comparisons and never change selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

EXPECTED_POLICY_ID = "eval139-canonical-checkpoint-selection-v1"
EXPECTED_POLICY_SHA256 = "58d1683111ce7df9024b4d50399bb1b6062381697ad136d69ebdae7c5e1915dd"
SCHEMA = "12-6.learn122-eval139-selection.v1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy(path: Path) -> dict[str, Any]:
    policy = read_json(path)
    if policy.get("schema") != "12-6.checkpoint-selection-policy.v1":
        raise SystemExit("checkpoint-selection policy schema drift")
    if policy.get("policy_id") != EXPECTED_POLICY_ID:
        raise SystemExit("checkpoint-selection policy id drift")
    if policy.get("policy_sha256") != EXPECTED_POLICY_SHA256:
        raise SystemExit("checkpoint-selection policy identity drift")
    selector = policy.get("selector", {})
    expected = {
        "direction": "minimize",
        "metric_name": "bpb",
        "minimum_improvement": 0.01,
        "smoother": "trailing_median",
        "smoothing_window": 3,
    }
    if selector != expected:
        raise SystemExit("checkpoint-selection selector drift")
    if policy.get("status") != "FROZEN_BY_EVAL_139":
        raise SystemExit("checkpoint-selection policy is not frozen")
    return policy


def select(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if len(rows) < 3:
        raise SystemExit("at least three chronological checkpoints are required")
    rows = sorted(rows, key=lambda row: int(row["actual_optimized_tokens"]))
    checkpoint_ids = [str(row["checkpoint_id"]) for row in rows]
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise SystemExit("checkpoint ids must be unique")
    tokens = [int(row["actual_optimized_tokens"]) for row in rows]
    if tokens != sorted(tokens) or len(set(tokens)) != len(tokens):
        raise SystemExit("optimized-token registry must be strictly chronological")
    values = [float(row["bits_per_byte"]) for row in rows]
    window = int(policy["selector"]["smoothing_window"])
    threshold = float(policy["selector"]["minimum_improvement"])
    incumbent_id: str | None = None
    incumbent_score: float | None = None
    trace: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        score = None
        accepted = False
        reason = "smoothing_warmup"
        if index + 1 >= window:
            score = float(statistics.median(values[index + 1 - window:index + 1]))
            if incumbent_id is None:
                accepted = True
                reason = "first_eligible_smoothed_checkpoint"
            elif score <= float(incumbent_score) - threshold:
                accepted = True
                reason = "minimum_improvement_met"
            else:
                reason = "minimum_improvement_not_met"
            if accepted:
                incumbent_id = str(row["checkpoint_id"])
                incumbent_score = score
        trace.append({
            "ordinal": index,
            "checkpoint_id": str(row["checkpoint_id"]),
            "checkpoint": str(row["checkpoint"]),
            "optimized_tokens": int(row["actual_optimized_tokens"]),
            "raw_selection_metric_bpb": float(row["bits_per_byte"]),
            "smoothed_selection_metric_bpb": score,
            "accepted_as_incumbent": accepted,
            "reason": reason,
        })
    if incumbent_id is None:
        raise SystemExit("selection failed to establish an incumbent")
    selected = next(row for row in rows if str(row["checkpoint_id"]) == incumbent_id)
    raw_best = min(enumerate(rows), key=lambda pair: (float(pair[1]["bits_per_byte"]), pair[0]))[1]
    final = rows[-1]
    decision_core = {
        "schema": SCHEMA,
        "policy_id": policy["policy_id"],
        "policy_sha256": policy["policy_sha256"],
        "selected_checkpoint_id": incumbent_id,
        "trace": trace,
    }
    decision = {
        **decision_core,
        "selection_metric": "heldout_bits_per_byte",
        "selection_rule": dict(policy["selector"]),
        "selected_checkpoint": {
            "checkpoint_id": incumbent_id,
            "checkpoint": selected["checkpoint"],
            "actual_optimized_tokens": int(selected["actual_optimized_tokens"]),
            "bits_per_byte": float(selected["bits_per_byte"]),
            "by_stratum": selected["by_stratum"],
        },
        "posthoc_comparison": {
            "absolute_raw_best_validation_checkpoint_id": str(raw_best["checkpoint_id"]),
            "absolute_raw_best_validation_bpb": float(raw_best["bits_per_byte"]),
            "final_checkpoint_id": str(final["checkpoint_id"]),
            "final_checkpoint_bpb": float(final["bits_per_byte"]),
            "posthoc_best_used_for_selection": False,
            "final_checkpoint_used_for_selection": False,
        },
        "retention_policy": "preserve_all_registered_checkpoints",
        "decision_identity_sha256": hash_json(decision_core),
    }
    return decision


def self_test(policy: dict[str, Any]) -> None:
    fixture = [
        {"checkpoint_id": "a", "checkpoint": "a", "actual_optimized_tokens": 0, "bits_per_byte": 5.0, "by_stratum": {}},
        {"checkpoint_id": "b", "checkpoint": "b", "actual_optimized_tokens": 1, "bits_per_byte": 4.0, "by_stratum": {}},
        {"checkpoint_id": "c", "checkpoint": "c", "actual_optimized_tokens": 2, "bits_per_byte": 3.0, "by_stratum": {}},
        {"checkpoint_id": "d", "checkpoint": "d", "actual_optimized_tokens": 3, "bits_per_byte": 2.995, "by_stratum": {}},
        {"checkpoint_id": "e", "checkpoint": "e", "actual_optimized_tokens": 4, "bits_per_byte": 2.0, "by_stratum": {}},
    ]
    decision = select(fixture, policy)
    if decision["selected_checkpoint"]["checkpoint_id"] != "d":
        raise SystemExit("selector self-test failed: preregistered smoothed incumbent drift")
    if decision["posthoc_comparison"]["absolute_raw_best_validation_checkpoint_id"] != "e":
        raise SystemExit("selector self-test failed: raw posthoc best drift")
    if decision["selected_checkpoint"]["checkpoint_id"] == decision["posthoc_comparison"]["absolute_raw_best_validation_checkpoint_id"]:
        raise SystemExit("selector self-test failed: raw-best leakage was not detected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    if args.self_test:
        self_test(policy)
        print("LEARN122_EVAL139_SELECTOR_SELF_TEST_OK")
        return
    if args.report is None or args.output is None:
        raise SystemExit("--report and --output are required unless --self-test is used")
    report = read_json(args.report)
    trajectory = report.get("500k_primary_long_trajectory")
    if not isinstance(trajectory, list):
        raise SystemExit("LEARN-122 report has no 500K long trajectory")
    decision = select(trajectory, policy)
    decision["source"] = report["source"]
    decision["corpus"] = report["corpus"]
    decision["tokenizer"] = report["tokenizer"]
    decision["report_sha256"] = report["report_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
