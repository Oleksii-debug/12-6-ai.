#!/usr/bin/env python3
"""Fail-closed CLI for model-size-aware pretraining data readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from twelve_six.data_budget import evaluate_data_budget, required_unique_loss_tokens


DEFAULT_POLICY = REPO_ROOT / "configs/scaling/data_budget_v1.json"
TOKEN_FIELD_BY_TIER = {
    "pilot_5x": "pilot_5x_tokens",
    "compute_reference_20x": "compute_reference_20x_tokens",
    "extended_50x": "extended_50x_tokens",
}


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "12-6-data-budget-v1":
        raise ValueError("unsupported data-budget schema")
    if policy.get("status") != "RESEARCH_REFERENCE_NOT_COMPUTE_AUTHORIZATION":
        raise ValueError("policy must not imply compute authorization")

    accounting = policy.get("accounting")
    if not isinstance(accounting, dict):
        raise ValueError("missing accounting policy")
    if accounting.get("capacity_unit") != "post_pack_unique_causal_loss_tokens":
        raise ValueError("capacity must use post-pack unique causal-loss tokens")
    if accounting.get("source_bytes_are_capacity") is not False:
        raise ValueError("source bytes must not be accepted as capacity")
    if accounting.get("replay_counts_as_new_capacity") is not False:
        raise ValueError("replay must not count as new capacity")

    tiers = policy.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError("missing tier definitions")

    targets = policy.get("exact_targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("missing exact targets")

    for target_name, target in targets.items():
        if not isinstance(target, dict):
            raise ValueError(f"invalid target: {target_name}")
        parameters = target.get("parameters")
        for tier_name, token_field in TOKEN_FIELD_BY_TIER.items():
            tier = tiers.get(tier_name)
            if not isinstance(tier, dict):
                raise ValueError(f"missing required tier: {tier_name}")
            ratio = tier.get("tokens_per_parameter")
            expected = required_unique_loss_tokens(parameters, ratio)
            if target.get(token_field) != expected:
                raise ValueError(
                    f"target {target_name} field {token_field} does not match ratio"
                )

    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a 12-6 AI pretraining data budget using only materialized, "
            "post-pack unique causal-loss tokens."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target", help="Exact target name from the policy")
    source.add_argument("--parameters", type=int, help="Exact model parameter count")
    parser.add_argument("--unique-loss-tokens", type=int, required=True)
    parser.add_argument(
        "--tier",
        choices=tuple(TOKEN_FIELD_BY_TIER),
        default="compute_reference_20x",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = load_policy(args.policy)
    tiers = policy["tiers"]

    if args.target is not None:
        target = policy["exact_targets"].get(args.target)
        if target is None:
            raise SystemExit(f"unknown target: {args.target}")
        parameter_count = target["parameters"]
    else:
        parameter_count = args.parameters

    tier = tiers[args.tier]
    result = evaluate_data_budget(
        parameter_count=parameter_count,
        unique_loss_tokens=args.unique_loss_tokens,
        tokens_per_parameter=tier["tokens_per_parameter"],
        dense_training_flops_per_parameter_token=policy["planning"][
            "dense_training_flops_per_parameter_token"
        ],
    )

    payload = {
        "schema_version": "12-6-data-budget-result-v1",
        "policy_status": policy["status"],
        "tier": args.tier,
        "capacity_unit": policy["accounting"]["capacity_unit"],
        "decision": "READY_FOR_THIS_DATA_BUDGET_TIER" if result.ready else "BLOCKED_DATA_SHORTFALL",
        **result.to_dict(),
        "compute_authorized": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
