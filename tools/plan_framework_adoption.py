from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.framework_adoption import AdoptionSignals, decision_payload
from twelve_six.model import load_stage_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan stage-triggered distributed framework adoption")
    parser.add_argument("stage_config", type=Path)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--shard-model-state", action="store_true")
    parser.add_argument("--native-does-not-fit", action="store_true")
    parser.add_argument("--native-runtime-incomplete", action="store_true")
    parser.add_argument("--nvidia-cluster", action="store_true")
    parser.add_argument("--megatron-runtime-validated", action="store_true")
    parser.add_argument("--megatron-speedup", type=float)
    parser.add_argument("--minimum-migration-speedup", type=float, default=1.15)
    return parser


def main() -> int:
    args = _parser().parse_args()
    stage = load_stage_config(args.stage_config)
    plan = ParallelPlan(
        data_parallel=args.dp,
        tensor_parallel=args.tp,
        pipeline_parallel=args.pp,
        context_parallel=args.cp,
        expert_parallel=args.ep,
        shard_model_state_across_data_parallel=args.shard_model_state,
    )
    signals = AdoptionSignals(
        native_fits_memory=not args.native_does_not_fit,
        native_runtime_complete=not args.native_runtime_incomplete,
        nvidia_cluster=args.nvidia_cluster,
        measured_megatron_speedup=args.megatron_speedup,
        minimum_migration_speedup=args.minimum_migration_speedup,
        megatron_runtime_validated=args.megatron_runtime_validated,
    )
    payload = {
        "schema": "12-6.framework-adoption-plan.v1",
        "stage": stage.stage,
        "expected_parameters": stage.expected_parameters,
        **decision_payload(stage.model, stage.init, plan, signals),
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
