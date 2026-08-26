#!/usr/bin/env python3
"""Build the bounded CI-161 retro-classification sample from recorded Actions metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twelve_six.experiment_failure import (
    FailurePhase,
    FailureSignal,
    build_report,
    validate_report,
)

SCHEMA = "12-6.historical-failure-classification-sample.v1"


def _cases() -> list[dict]:
    cases = [
        build_report(
            FailureSignal(
                phase=FailurePhase.FOCUSED_TEST,
                return_code=1,
                missing_dependency="pytest",
            ),
            diagnostic_codes=(
                "HISTORICAL_STEP_FAILURE",
                "PYTHON_MODULE_MISSING:pytest",
                "TRAINING_STEP_SKIPPED",
            ),
            diagnostic_summary=(
                "Runtime-only lock omitted pytest; contract-test step failed before phase1 and "
                "all training/resume/report steps were skipped."
            ),
            source_sha="015593b22a600184fb4c8001fe3d70893bfc51d5",
            workflow="MILESTONE-100 First Learned ~1M Base",
            run_id=32901140565,
            historical=True,
        ),
        build_report(
            FailureSignal(
                phase=FailurePhase.FOCUSED_TEST,
                return_code=1,
                missing_dependency="pytest",
            ),
            diagnostic_codes=(
                "HISTORICAL_STEP_FAILURE",
                "PYTHON_MODULE_MISSING:pytest",
                "TRAINING_STEP_SKIPPED",
            ),
            diagnostic_summary=(
                "Toolchain plus runtime locks omitted pytest; contract-test step failed and the "
                "10M phase1, resume and evaluation steps never ran."
            ),
            source_sha="020bf4f01a56ae6e4defaec6db00034385910f38",
            workflow="SCALE-141 10M Learned Continuation",
            run_id=32902872519,
            historical=True,
        ),
        build_report(
            FailureSignal(phase=FailurePhase.PREPARE, return_code=1),
            diagnostic_codes=(
                "HISTORICAL_PRETRAINING_GATE_FAILED",
                "TRAINING_STEP_SKIPPED",
                "RAW_LOG_NOT_RETAINED_IN_RETRO_REPORT",
            ),
            diagnostic_summary=(
                "Locked-environment/repository gate failed; the D02 execution environment and "
                "40-step training step were skipped, so no model failure is evidenced."
            ),
            source_sha="020bf4f01a56ae6e4defaec6db00034385910f38",
            workflow="D02 Real S0 Training",
            run_id=32902872602,
            historical=True,
        ),
        build_report(
            FailureSignal(phase=FailurePhase.STATIC_CHECK, return_code=1),
            diagnostic_codes=(
                "HISTORICAL_AGGREGATED_CHECK_STEP_FAILED",
                "NO_EXPERIMENT_PHASE",
                "RAW_LOG_NOT_RETAINED_IN_RETRO_REPORT",
            ),
            diagnostic_summary=(
                "x86 locked clean-install/package-smoke/full-check aggregate failed while the ARM "
                "job passed; no experiment phase was present in this CI workflow."
            ),
            source_sha="020bf4f01a56ae6e4defaec6db00034385910f38",
            workflow="CI / locked-x86-64",
            run_id=32902872600,
            historical=True,
        ),
    ]
    for case in cases:
        validate_report(case)
    return cases


def build() -> dict:
    payload = {
        "schema": SCHEMA,
        "taxonomy_version": "ci161.v1",
        "evidence_policy": {
            "raw_logs_retained": False,
            "secrets_retained": False,
            "classification_uses": [
                "GitHub Actions run/job/step conclusions",
                "workflow command structure",
                "committed exact dependency locks",
            ],
            "historical_uncertainty_is_not_upgraded_to_model_failure": True,
        },
        "cases": _cases(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "ci161" / "historical_failure_sample.json",
    )
    args = parser.parse_args(argv)
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
