from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.layer_health import run_layer_health_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TRAIN-54 per-layer health diagnostics")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument(
        "--diagnostic-steps",
        type=int,
        nargs="+",
        default=[0, 4, 16, 64],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    args = parser.parse_args()

    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    if not isinstance(locked, dict):
        raise TypeError("locked environment evidence must be a JSON object")
    report = run_layer_health_matrix(
        Path("."),
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        output_path=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
        diagnostic_steps=tuple(args.diagnostic_steps),
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "report_sha256": report["report_sha256"],
                "cross_scale": report["cross_scale"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
