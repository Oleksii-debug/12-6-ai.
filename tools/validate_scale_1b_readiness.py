from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s6_readiness import (
    build_s6_readiness_report,
    validate_s6_readiness_report,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate allocation-safe SCALE-06 S6 ~1B readiness evidence."
    )
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-local-analogue", action="store_true")
    args = parser.parse_args()

    report = build_s6_readiness_report(
        ROOT,
        source_sha=args.source_sha,
        run_analogue=not args.skip_local_analogue,
    )
    validate_s6_readiness_report(report)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": report["candidate"]["stage"],
                "parameters": report["candidate"]["parameters"],
                "meta_instantiated_parameters": report["candidate"][
                    "meta_instantiated_parameters"
                ],
                "fsdp2_world8_persistent_bytes_per_rank": report[
                    "resource_estimates"
                ]["8"]["persistent_total_bytes_per_rank"],
                "pilot_estimated_training_flops": report["pilot"][
                    "estimated_training_flops"
                ],
                "launch_ready": report["launch"]["ready"],
                "blockers": report["launch"]["blockers"],
                "local_analogue": report["local_execution_analogue"]["status"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
