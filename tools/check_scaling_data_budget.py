#!/usr/bin/env python3
"""Evaluate one exact scaling data-budget tier without granting training authority."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from twelve_six.data_budget import evaluate_policy_stage  # noqa: E402


DEFAULT_POLICY = REPO_ROOT / "configs/scaling/data_budget_policy_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact post-tokenization unique causal-loss positions against "
            "the preregistered 12-6 scaling data-budget policy."
        )
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--unique-loss-positions", required=True, type=int)
    parser.add_argument("--multiplier", required=True, type=int)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    result = evaluate_policy_stage(
        policy=policy,
        stage_name=args.stage,
        unique_loss_positions=args.unique_loss_positions,
        multiplier=args.multiplier,
    )

    decision = (
        "DATA_BUDGET_TIER_MET" if result.data_budget_tier_met else "BLOCKED_DATA_BUDGET"
    )
    payload = {
        "schema_version": "12-6.scaling-data-budget-runtime-result.v1",
        "decision": decision,
        **result.to_dict(),
        "truth_boundary": {
            "this_result_authorizes_training": False,
            "this_result_authorizes_paid_compute": False,
            "source_bytes_are_accepted_as_capacity": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.data_budget_tier_met else 2


if __name__ == "__main__":
    raise SystemExit(main())
