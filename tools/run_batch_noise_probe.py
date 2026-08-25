"""Run TRAIN-53 fixed-268K batch-noise experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.batch_noise_probe import run_batch_noise_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    report = run_batch_noise_probe(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    recommendation = report["recommendation"]
    comparison = recommendation["candidate_final_comparison"]
    diagnostics = [
        {
            "optimized_tokens": point["checkpoint_optimized_tokens"],
            "noise_scale_microbatches_proxy": point["diagnostic"]["statistics"][
                "noise_scale_microbatches_proxy"
            ],
            "probe_wall_seconds": point["diagnostic"]["probe_wall_seconds"],
        }
        for point in report["gradient_diagnostics"]
    ]
    print(
        json.dumps(
            {
                "authority": report["authority"],
                "source_sha": report["identity"]["source_sha"],
                "parameter_count": report["identity"]["parameter_count"],
                "base_loss_tokens_per_microbatch": report["identity"][
                    "base_loss_tokens_per_microbatch"
                ],
                "candidate_final_comparison": comparison,
                "gradient_diagnostics": diagnostics,
                "recommendation": recommendation,
                "report_sha256": report["report_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
