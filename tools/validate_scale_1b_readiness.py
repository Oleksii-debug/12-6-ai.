from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s6_readiness import build_s6_readiness_report

DEFAULT_CONFIG = Path("configs/stages/s6_1b.scale06_current_tokenizer.candidate.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate allocation-safe 12-6 S6 ~1B readiness")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_s6_readiness_report(
        args.config,
        world_size=args.world_size,
        sequence_length=args.sequence_length,
        microbatch_size=args.microbatch_size,
        activation_checkpointing=True,
        compute_authorized=False,
    )
    payload = report.to_dict()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
