from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.checkpoint.s0_inference_artifact_evidence import (
    build_s0_inference_artifact_evidence,
    validate_s0_inference_artifact_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a verified trained S0 checkpoint plus first-party inference evidence."
    )
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_s0_inference_artifact_evidence(
        Path.cwd(),
        source_sha=args.source_sha,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    validate_s0_inference_artifact_evidence(
        report,
        checkpoint_dir=args.output_dir / "checkpoint",
        expected_source_sha=args.source_sha,
    )
    report_path = args.output_dir / "inference-evidence.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
