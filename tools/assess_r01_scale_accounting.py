#!/usr/bin/env python3
"""Materialize the fail-closed R01 scale-accounting planning report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.scale_accounting import (
    DecoderScaleSpec,
    ScaleAccountingError,
    assess_candidate,
    parameter_breakdown,
    vocabulary_sensitivity,
)


def assess(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ScaleAccountingError("unsupported scale-accounting schema")
    if config.get("status") != "PLANNING_ONLY":
        raise ScaleAccountingError("scale-accounting status must remain PLANNING_ONLY")
    truth = config.get("truth_boundary")
    if not isinstance(truth, dict) or any(value is not False for value in truth.values()):
        raise ScaleAccountingError("all scale-accounting truth-boundary flags must remain false")

    baseline = DecoderScaleSpec.from_mapping(config["baseline"])
    baseline_record = assess_candidate(baseline)
    expected = config["authority"]["expected_parameter_count"]
    if parameter_breakdown(baseline)["total"] != expected:
        raise ScaleAccountingError("MODEL-341 exact parameter authority drift")

    vocab_rows = vocabulary_sensitivity(baseline, config["vocabulary_sensitivity"])
    candidates = []
    for raw in config.get("candidate_geometries", []):
        if raw.get("status") != "PLANNING_ONLY_NOT_FROZEN":
            raise ScaleAccountingError("candidate geometry must remain planning-only")
        candidate = assess_candidate(
            DecoderScaleSpec.from_mapping(raw["spec"]),
            target_parameters=raw["target_parameters"],
        )
        if abs(candidate["target_relative_error"]) > 0.01:
            raise ScaleAccountingError(
                f"{raw.get('id')}: candidate differs from target by more than 1%"
            )
        candidates.append({"id": raw["id"], **candidate})

    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "status": "PLANNING_ONLY",
        "baseline": baseline_record,
        "vocabulary_sensitivity_fixed_geometry": vocab_rows,
        "candidate_geometries_not_frozen": candidates,
        "comparison_rule": (
            "do_not_compare total-parameter targets across tokenizer vocabularies "
            "without reporting embedding share and FLOPs/token"
        ),
        "truth_boundary": truth,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = assess(config)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
