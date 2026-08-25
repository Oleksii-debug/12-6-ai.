from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.data.near_dedup import (
    calibration_records,
    load_calibration,
    policy_candidates,
    run_datatrove_policy,
    score_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit fail-closed DATA-30 candidate calibration diagnostics"
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.workspace.exists():
        shutil.rmtree(args.workspace)
    args.workspace.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(args.calibration)
    report: dict[str, object] = {
        "schema_version": "12-6.near-dedup-calibration-diagnostics.v1",
        "authority": "DIAGNOSTIC_ONLY_NOT_POLICY_SELECTION",
        "gates": {"min_recall": 0.75, "max_false_removal_risk": 0.25},
        "modalities": {},
    }

    modalities: dict[str, object] = {}
    for modality in ("natural", "code"):
        records = calibration_records(calibration, modality)
        rows = []
        for policy in policy_candidates()[modality]:
            execution = run_datatrove_policy(
                records,
                policy=policy,
                workspace=args.workspace / modality / policy.name,
                exercise_skip_completed=False,
            )
            metrics = score_calibration(
                calibration, modality=modality, execution=execution
            )
            rows.append(
                {
                    "policy": policy.manifest(),
                    "recall": metrics["recall"],
                    "false_removal_risk": metrics["false_removal_risk"],
                    "true_positive_pairs": metrics["true_positive_pairs"],
                    "false_positive_pairs": metrics["false_positive_pairs"],
                    "pair_detection": metrics["pair_detection"],
                    "category_detection": metrics["category_detection"],
                    "false_positive_review_sample": metrics[
                        "false_positive_review_sample"
                    ],
                    "passes_gates": (
                        float(metrics["recall"]) >= 0.75
                        and float(metrics["false_removal_risk"]) <= 0.25
                    ),
                }
            )
        modalities[modality] = rows
    report["modalities"] = modalities

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
