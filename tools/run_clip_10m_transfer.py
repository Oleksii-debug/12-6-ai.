#!/usr/bin/env python3
"""Execute TRAIN-127 on the exact checked-out source tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.clip_10m_transfer import run_clip_10m_transfer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preregistration-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    report = run_clip_10m_transfer(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=args.locked_environment_evidence,
        output=args.output,
        preregistration_output=args.preregistration_output,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    print(json.dumps({
        "report_sha256": report["report_sha256"],
        "parameter_count": report["identity"]["parameter_count"],
        "diagnostic_global_norm_distribution": report["preregistration"]["diagnostic_global_norm_distribution"],
        "thresholds": report["preregistration"]["thresholds"],
        "selection": report["selection"],
        "truth_boundary": report["truth_boundary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
