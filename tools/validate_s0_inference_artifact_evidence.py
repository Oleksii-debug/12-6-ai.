from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.checkpoint.s0_inference_artifact_evidence import (
    validate_s0_inference_artifact_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate D05 S0 inference artifact evidence.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    validate_s0_inference_artifact_evidence(
        payload,
        checkpoint_dir=args.checkpoint,
        expected_source_sha=args.expected_source_sha,
    )
    print("D05 S0 inference artifact evidence: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
