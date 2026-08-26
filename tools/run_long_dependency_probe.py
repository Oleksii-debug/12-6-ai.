from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.inference.loader import load_backend
from twelve_six.long_dependency import (
    DEFAULT_CASES_PER_FAMILY_DISTANCE,
    DEFAULT_DISTANCES,
    DEFAULT_SEED,
    materialize_suite,
    score_suite,
    validate_report,
)


def _parse_distances(value: str) -> tuple[int, ...]:
    distances = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not distances:
        raise argparse.ArgumentTypeError("at least one dependency distance is required")
    return distances


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the reserved EVAL-135 long-distance dependency suite."
    )
    parser.add_argument("--backend-loader", required=True, help="MODULE:CALLABLE backend loader")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--distances",
        type=_parse_distances,
        default=DEFAULT_DISTANCES,
        help="comma-separated exact token distances; unsupported distances are skipped",
    )
    parser.add_argument(
        "--cases-per-family-distance",
        type=int,
        default=DEFAULT_CASES_PER_FAMILY_DISTANCE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    backend = load_backend(args.backend_loader, args.checkpoint)
    suite = materialize_suite(
        backend,
        distances=args.distances,
        cases_per_family_distance=args.cases_per_family_distance,
        seed=args.seed,
    )
    report = score_suite(backend, suite, model_label=args.model_label)
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "suite_identity_sha256": report["suite_identity_sha256"],
                "materialized_identity_sha256": report["materialized_identity_sha256"],
                "model": report["model"],
                "interpretation": report["interpretation"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
