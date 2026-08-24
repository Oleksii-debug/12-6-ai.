"""Validate and materialize an exact-candidate D04 evaluation evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.evaluation_evidence import (
    build_s0_evaluation_bundle,
    validate_s0_evaluation_bundle,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--locked-environment-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-quality-pass", action="store_true")
    args = parser.parse_args()

    paths = {
        name: args.report_dir / name
        for name in (
            "candidate_evidence.json",
            "stage_gate_report.json",
            "promotion_eligibility.json",
        )
    }
    bundle = build_s0_evaluation_bundle(
        candidate_sha=args.candidate_sha,
        candidate_evidence=_load_json(paths["candidate_evidence.json"]),
        stage_gate_report=_load_json(paths["stage_gate_report.json"]),
        promotion_report=_load_json(paths["promotion_eligibility.json"]),
        locked_environment_evidence=_load_json(args.locked_environment_evidence),
        report_hashes={name: _sha256_file(path) for name, path in paths.items()},
        require_quality_pass=args.require_quality_pass,
    )
    validate_s0_evaluation_bundle(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"candidate_sha={bundle['candidate_sha']}")
    print(f"bundle_sha256={bundle['bundle_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
