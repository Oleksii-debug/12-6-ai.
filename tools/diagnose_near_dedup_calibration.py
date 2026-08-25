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
    NearDedupPolicy,
    calibration_records,
    load_calibration,
    policy_candidates,
    run_datatrove_policy,
    score_calibration,
)


def _diagnostic_candidates() -> dict[str, tuple[NearDedupPolicy, ...]]:
    incumbent = policy_candidates()
    return {
        "natural": (
            *incumbent["natural"],
            NearDedupPolicy("diag_natural_7g_12x9", "natural", 7, 12, 9),
            NearDedupPolicy("diag_natural_7g_14x8", "natural", 7, 14, 8),
            NearDedupPolicy("diag_natural_7g_16x7", "natural", 7, 16, 7),
            NearDedupPolicy("diag_natural_5g_12x9", "natural", 5, 12, 9),
            NearDedupPolicy("diag_natural_5g_14x8", "natural", 5, 14, 8),
            NearDedupPolicy("diag_natural_5g_16x7", "natural", 5, 16, 7),
            NearDedupPolicy("diag_natural_5g_20x5", "natural", 5, 20, 5),
            NearDedupPolicy("diag_natural_5g_28x4", "natural", 5, 28, 4),
        ),
        "code": (
            *incumbent["code"],
            NearDedupPolicy("diag_code_4g_10x10", "code", 4, 10, 10),
            NearDedupPolicy("diag_code_4g_12x9", "code", 4, 12, 9),
            NearDedupPolicy("diag_code_4g_14x8", "code", 4, 14, 8),
            NearDedupPolicy("diag_code_4g_15x7", "code", 4, 15, 7),
            NearDedupPolicy("diag_code_4g_16x7", "code", 4, 16, 7),
            NearDedupPolicy("diag_code_4g_16x6", "code", 4, 16, 6),
            NearDedupPolicy("diag_code_4g_18x6", "code", 4, 18, 6),
            NearDedupPolicy("diag_code_4g_18x5", "code", 4, 18, 5),
            NearDedupPolicy("diag_code_4g_20x5", "code", 4, 20, 5),
            NearDedupPolicy("diag_code_4g_28x4", "code", 4, 28, 4),
            NearDedupPolicy("diag_code_3g_10x10", "code", 3, 10, 10),
            NearDedupPolicy("diag_code_3g_12x9", "code", 3, 12, 9),
            NearDedupPolicy("diag_code_3g_14x8", "code", 3, 14, 8),
            NearDedupPolicy("diag_code_3g_15x7", "code", 3, 15, 7),
            NearDedupPolicy("diag_code_3g_16x7", "code", 3, 16, 7),
            NearDedupPolicy("diag_code_3g_16x6", "code", 3, 16, 6),
            NearDedupPolicy("diag_code_3g_18x6", "code", 3, 18, 6),
            NearDedupPolicy("diag_code_3g_18x5", "code", 3, 18, 5),
            NearDedupPolicy("diag_code_3g_20x5", "code", 3, 20, 5),
            NearDedupPolicy("diag_code_3g_28x4", "code", 3, 28, 4),
        ),
    }


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
        "exploratory_grid_changes_policy_selection": False,
        "modalities": {},
    }

    candidates = _diagnostic_candidates()
    modalities: dict[str, object] = {}
    for modality in ("natural", "code"):
        records = calibration_records(calibration, modality)
        rows = []
        for policy in candidates[modality]:
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
                    "incumbent_candidate": policy in policy_candidates()[modality],
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
