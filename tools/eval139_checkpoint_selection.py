#!/usr/bin/env python3
"""Build a canonical EVAL-139 checkpoint-selection report from a frozen manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.checkpoint_selection import (
    CheckpointRef,
    EvaluationPurpose,
    MetricObservation,
    SelectionRule,
    SelectionValidationObservation,
    build_experiment_selection_report,
    select_checkpoint,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input manifest must be a JSON object")
    return payload


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON list")
    return value


def build_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse a frozen experiment manifest and execute only the selection channel."""

    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")

    purposes = tuple(
        EvaluationPurpose(**item) for item in _require_list(payload, "evaluation_purposes")
    )
    selection_purposes = [item for item in purposes if item.purpose == "selection_validation"]
    if len(selection_purposes) != 1:
        raise ValueError("input must define exactly one selection_validation purpose")
    selection_purpose = selection_purposes[0]

    rule_data = payload.get("selection_rule", {})
    if not isinstance(rule_data, dict):
        raise ValueError("selection_rule must be a JSON object")
    rule = SelectionRule(**rule_data)

    checkpoints = tuple(CheckpointRef(**item) for item in _require_list(payload, "checkpoints"))
    selection_observations = tuple(
        SelectionValidationObservation(**item)
        for item in _require_list(payload, "selection_observations")
    )
    nonselection_data = payload.get("nonselection_observations", [])
    if not isinstance(nonselection_data, list):
        raise ValueError("nonselection_observations must be a JSON list")
    nonselection = tuple(MetricObservation(**item) for item in nonselection_data)

    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object")

    decision = select_checkpoint(
        checkpoints,
        selection_observations,
        selection_purpose=selection_purpose,
        rule=rule,
    )
    return build_experiment_selection_report(
        experiment_id=experiment_id,
        decision=decision,
        evaluation_purposes=purposes,
        nonselection_observations=nonselection,
        provenance=provenance,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Frozen EVAL-139 input manifest")
    parser.add_argument(
        "--output", required=True, type=Path, help="Machine-readable selection report"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(_load_json(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_checkpoint_id": report["selection"]["selected_checkpoint_id"],
                "selection_decision_sha256": report["selection"]["selection_decision_sha256"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
