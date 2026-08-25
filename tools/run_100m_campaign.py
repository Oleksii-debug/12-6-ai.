#!/usr/bin/env python3
"""Run CAMPAIGN-47 ~100M qualification steps without initiating a main paid launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.campaign_100m import (
    BUDGET_VARIANTS,
    project_budget,
    run_s2_probe,
    run_s4_construction_preflight,
    run_s4_gpu_pilot,
    wrap_s3_gpu_pilot,
)
from twelve_six.campaign_100m_authority import qualify_campaign_main_launch


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    s2 = sub.add_parser("s2-probe")
    s2.add_argument("--repo-root", type=Path, default=Path.cwd())
    s2.add_argument("--source-sha", required=True)
    s2.add_argument("--output", type=Path, required=True)
    s2.add_argument("--sequence-length", type=int, default=256)
    s2.add_argument("--seed", type=int, default=20260825)

    s4 = sub.add_parser("s4-preflight")
    s4.add_argument("--repo-root", type=Path, default=Path.cwd())
    s4.add_argument("--source-sha", required=True)
    s4.add_argument("--output", type=Path, required=True)
    s4.add_argument("--seed", type=int, default=20260825)

    pilot100 = sub.add_parser("s4-gpu-pilot")
    pilot100.add_argument("--repo-root", type=Path, default=Path.cwd())
    pilot100.add_argument("--source-sha", required=True)
    pilot100.add_argument("--checkpoint-root", type=Path)
    pilot100.add_argument("--provider-label", required=True)
    pilot100.add_argument("--hardware-label", required=True)
    pilot100.add_argument("--hourly-cost-eur", type=float, required=True)
    pilot100.add_argument("--rate-evidence", required=True)
    pilot100.add_argument("--compute-class", choices=("local_free", "paid"), default="local_free")
    pilot100.add_argument("--authorize-paid-compute", action="store_true")
    pilot100.add_argument("--batch-size", type=int, default=1)
    pilot100.add_argument("--sequence-length", type=int, default=2048)
    pilot100.add_argument("--optimizer-steps", type=int, default=4)
    pilot100.add_argument("--gradient-accumulation-steps", type=int, default=8)
    pilot100.add_argument("--learning-rate", type=float, default=3e-4)
    pilot100.add_argument("--seed", type=int, default=20260825)
    pilot100.add_argument("--output", type=Path, required=True)

    pilot10 = sub.add_parser("wrap-s3-gpu-pilot")
    pilot10.add_argument("--s3-evidence", type=Path, required=True)
    pilot10.add_argument("--source-sha", required=True)
    pilot10.add_argument("--provider-label", required=True)
    pilot10.add_argument("--hardware-label", required=True)
    pilot10.add_argument("--hourly-cost-eur", type=float, required=True)
    pilot10.add_argument("--rate-evidence", required=True)
    pilot10.add_argument("--output", type=Path, required=True)

    budget = sub.add_parser("budget")
    budget.add_argument("--pilot", type=Path, required=True)
    budget.add_argument("--variant", choices=sorted(BUDGET_VARIANTS), required=True)
    budget.add_argument("--output", type=Path, required=True)

    qualify = sub.add_parser("qualify")
    qualify.add_argument("--source-sha", required=True)
    qualify.add_argument("--variant", choices=sorted(BUDGET_VARIANTS), required=True)
    qualify.add_argument("--s2", type=Path, required=True)
    qualify.add_argument("--s3", type=Path, required=True)
    qualify.add_argument("--s4", type=Path, required=True)
    qualify.add_argument("--pilot", type=Path, required=True)
    qualify.add_argument("--tokenizer", type=Path, required=True)
    qualify.add_argument("--corpus", type=Path, required=True)
    qualify.add_argument("--evaluation", type=Path, required=True)
    qualify.add_argument("--authorize-paid-compute", action="store_true")
    qualify.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "s2-probe":
        payload = run_s2_probe(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
    elif args.command == "s4-preflight":
        payload = run_s4_construction_preflight(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            seed=args.seed,
        )
    elif args.command == "s4-gpu-pilot":
        payload = run_s4_gpu_pilot(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            checkpoint_root=args.checkpoint_root,
            provider_label=args.provider_label,
            hardware_label=args.hardware_label,
            hourly_cost_eur=args.hourly_cost_eur,
            rate_evidence=args.rate_evidence,
            compute_class=args.compute_class,
            paid_compute_authorized=args.authorize_paid_compute,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            optimizer_steps=args.optimizer_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    elif args.command == "wrap-s3-gpu-pilot":
        payload = wrap_s3_gpu_pilot(
            _read(args.s3_evidence),
            source_sha=args.source_sha,
            provider_label=args.provider_label,
            hardware_label=args.hardware_label,
            hourly_cost_eur=args.hourly_cost_eur,
            rate_evidence=args.rate_evidence,
        )
    elif args.command == "budget":
        payload = project_budget(_read(args.pilot), args.variant)
    else:
        payload = qualify_campaign_main_launch(
            source_sha=args.source_sha,
            variant_name=args.variant,
            s2_evidence=_read(args.s2),
            s3_evidence=_read(args.s3),
            s4_preflight=_read(args.s4),
            gpu_pilot=_read(args.pilot),
            tokenizer_freeze=_read(args.tokenizer),
            corpus_freeze=_read(args.corpus),
            evaluation_freeze=_read(args.evaluation),
            paid_compute_authorized=args.authorize_paid_compute,
        )
    _write(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
