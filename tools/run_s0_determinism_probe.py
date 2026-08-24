"""Execute one timing-free real S0 determinism probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s0_repeatability import run_s0_determinism_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    probe = run_s0_determinism_probe(
        root,
        source_sha=args.source_sha,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(probe, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seed": probe["identity"]["seed"],
                "stable_result_sha256": probe["stable_result_sha256"],
                "initial_model_sha256": probe["state_fingerprints"]["initial_model_sha256"],
                "final_model_sha256": probe["state_fingerprints"]["final_model_sha256"],
                "step_trace_sha256": probe["step_trace"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
