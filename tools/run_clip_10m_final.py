from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.clip_10m_final import run_clip_10m_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TRAIN-194 final 10M clipping experiment")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", type=Path, required=True)
    parser.add_argument("--preregistration-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    report = run_clip_10m_final(
        args.repo_root.resolve(),
        source_sha=args.source_sha,
        locked_environment_evidence=args.locked_environment_evidence.resolve(),
        preregistration_output=args.preregistration_output.resolve(),
        output=args.output.resolve(),
        torch_threads=args.torch_threads,
    )
    print(
        json.dumps(
            {
                "source_sha": report["identity"]["source_sha"],
                "parameter_count": report["identity"]["parameter_count"],
                "thresholds": report["preregistration"]["candidate_thresholds"],
                "candidate_summaries": {
                    label: {
                        "mean_final_bpb": summary["mean_final_bpb"],
                        "mean_clip_frequency": summary["mean_clip_frequency"],
                        "post_pre_norm_ratio": summary["post_pre_norm_ratio"]["profile"],
                        "update_weight_ratio": summary["update_weight_ratio"]["profile"],
                        "loss_spike_count_by_seed": summary["loss_spike_count_by_seed"],
                        "numerical_failures": summary["numerical_failures"],
                    }
                    for label, summary in report["candidate_summaries"].items()
                },
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
