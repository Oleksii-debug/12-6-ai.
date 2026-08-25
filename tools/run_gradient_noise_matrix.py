from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from twelve_six.gradient_noise import run_gradient_noise_matrix, validate_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/validate RESEARCH-20 gradient diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        report = run_gradient_noise_matrix(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            config_path=args.config.resolve(),
            output_path=args.output.resolve(),
        )
        print(
            json.dumps(
                {
                    "source": report["source"],
                    "decision_support": report["decision_support"],
                    "report_sha256": report["report_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(json.dumps({"valid": True, "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
