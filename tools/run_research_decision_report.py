#!/usr/bin/env python3
"""Generate deterministic RESEARCH-140 comparison decisions from a JSON input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.research_decision import (
    DecisionConfig,
    MetricDirection,
    MetricPurpose,
    Pair,
    analyze_paired_runs,
)


def _load_pair(payload: dict[str, object]) -> Pair:
    return Pair(
        run_id=str(payload["run_id"]),
        baseline=float(payload["baseline"]) if "baseline" in payload else None,
        candidate=float(payload["candidate"]) if "candidate" in payload else None,
        oriented_delta=(
            float(payload["oriented_delta"]) if "oriented_delta" in payload else None
        ),
    )


def build_report(document: dict[str, object]) -> dict[str, object]:
    comparisons = document.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("input comparisons must be a list")

    results: list[dict[str, object]] = []
    for raw in comparisons:
        if not isinstance(raw, dict):
            raise ValueError("each comparison must be an object")
        if "final_test_metrics" in raw:
            raise ValueError("final_test_metrics are forbidden in research winner inputs")

        config = DecisionConfig(
            materiality=float(raw["materiality"]),
            metric_name=str(raw["metric_name"]),
            metric_purpose=MetricPurpose(str(raw.get("metric_purpose", "selection_validation"))),
            direction=MetricDirection(str(raw.get("direction", "lower_is_better"))),
        )
        pairs_payload = raw.get("pairs", [])
        if not isinstance(pairs_payload, list):
            raise ValueError("pairs must be a list")
        result = analyze_paired_runs(
            [_load_pair(pair) for pair in pairs_payload],
            candidate=str(raw["candidate"]),
            baseline=str(raw["baseline"]),
            config=config,
        ).to_dict()
        result["comparison_id"] = str(raw["comparison_id"])
        result["source"] = raw.get("source", {})
        result["evidence_notes"] = raw.get("evidence_notes", [])
        result["configured_repeats"] = raw.get("configured_repeats")
        results.append(result)

    return {
        "schema": "12-6.research140-retroactive-decisions.v1",
        "worker_id": "RESEARCH-140-DECISION-RULES",
        "execution_class": "LOCAL_FREE",
        "selection_metric_boundary": {
            "allowed_purpose": "selection_validation",
            "final_test_visible_to_selector": False,
            "diagnostic_only_visible_to_selector": False,
        },
        "policy": document.get("policy", {}),
        "comparisons": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    report = build_report(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
