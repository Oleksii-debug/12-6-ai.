"""Run real 12-6 FSDP2 model execution on LOCAL_FREE CPU or under torchrun."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from twelve_six.distributed.fsdp2_training import (
    run_local_cpu_fsdp2,
    run_torchrun_fsdp2,
    write_execution_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    local = subparsers.add_parser("local-cpu", help="spawn bounded CPU/Gloo FSDP2 workers")
    local.add_argument("--stage-config", default="configs/stages/s1_100k.json")
    local.add_argument("--world-size", type=int, default=2)
    local.add_argument("--samples-per-rank", type=int, default=1)
    local.add_argument("--sequence-length", type=int, default=8)
    local.add_argument("--seed", type=int, default=1337)
    local.add_argument("--source-sha", required=True)
    local.add_argument("--output", type=Path, required=True)

    launched = subparsers.add_parser("torchrun", help="execute one rank under torchrun")
    launched.add_argument("--stage-config", default="configs/stages/s1_100k.json")
    launched.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    launched.add_argument("--device-type", choices=("cpu", "cuda"), required=True)
    launched.add_argument("--samples-per-rank", type=int, default=1)
    launched.add_argument("--sequence-length", type=int, default=8)
    launched.add_argument("--seed", type=int, default=1337)
    launched.add_argument("--skip-error-recovery", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "local-cpu":
        result = run_local_cpu_fsdp2(
            args.stage_config,
            world_size=args.world_size,
            samples_per_rank=args.samples_per_rank,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
        evidence = write_execution_evidence(
            result,
            source_sha=args.source_sha,
            output_path=args.output,
        )
        print(json.dumps(evidence, sort_keys=True))
        return 0

    if args.backend == "nccl" and args.device_type != "cuda":
        raise SystemExit("NCCL execution requires --device-type cuda")
    if args.device_type == "cuda" and args.backend != "nccl":
        raise SystemExit("GPU pilot requires --backend nccl")
    record = run_torchrun_fsdp2(
        args.stage_config,
        backend=args.backend,
        device_type=args.device_type,
        samples_per_rank=args.samples_per_rank,
        sequence_length=args.sequence_length,
        seed=args.seed,
        exercise_error_recovery=not args.skip_error_recovery,
    )
    print(json.dumps(asdict(record), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
