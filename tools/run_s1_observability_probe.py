"""Run the current S1 engineering-model observability probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s1_observability_probe import run_s1_observability_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jsonl-output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    evidence = run_s1_observability_probe(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    telemetry = evidence["telemetry"]
    records = [
        {
            "record_type": "run_identity",
            "run_identity": telemetry["run_identity"],
            "run_identity_sha256": telemetry["run_identity_sha256"],
        }
    ]
    records.extend(
        {
            "record_type": "step",
            "run_identity_sha256": telemetry["run_identity_sha256"],
            **sample,
        }
        for sample in telemetry["step_samples"]
    )
    records.extend(
        {
            "record_type": "region",
            "run_identity_sha256": telemetry["run_identity_sha256"],
            **region,
        }
        for region in telemetry["regions"]
    )
    records.append(
        {
            "record_type": "summary",
            "run_identity_sha256": telemetry["run_identity_sha256"],
            "summary": telemetry["summary"],
        }
    )
    args.jsonl_output.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    summary = telemetry["summary"]
    print(
        json.dumps(
            {
                "authority": evidence["authority"],
                "source_sha": evidence["identity"]["source_sha"],
                "parameter_count": evidence["identity"]["parameter_count"],
                "optimized_tokens": summary["counters"]["optimized_tokens"],
                "train_tokens_per_second": summary["throughput"][
                    "train_tokens_per_second"
                ],
                "step_seconds_p95": summary["timing"]["step_seconds_p95"],
                "data_wait_seconds_total": summary["timing"]["data_wait_seconds_total"],
                "checkpoint_seconds_total": summary["timing"]["checkpoint_seconds_total"],
                "evaluation_seconds_total": summary["timing"]["evaluation_seconds_total"],
                "memory_peak": summary["memory_peak"],
                "bottleneck": summary["bottleneck"]["classification"],
                "euro_2000_gate": evidence["paid_compute_decision_support"][
                    "euro_2000_gate"
                ],
                "euro_10000_gate": evidence["paid_compute_decision_support"][
                    "euro_10000_gate"
                ],
                "output": str(args.output),
                "jsonl_output": str(args.jsonl_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
