from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.scale_1b_readiness import (
    Scale1BDependencies,
    assess_scale_1b_readiness,
    meta_parameter_probe,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "configs" / "stages" / "s6_1b.scale06_current_tokenizer.candidate.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assess S6 ~1B dependency readiness. Every launch gate remains false "
            "unless the caller supplies positive evidence explicitly."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--preceding-stage-admitted", action="store_true")
    parser.add_argument("--production-tokenizer-qualified", action="store_true")
    parser.add_argument("--native-gqa-qualified", action="store_true")
    parser.add_argument("--distributed-checkpoint-qualified", action="store_true")
    parser.add_argument("--data-pipeline-qualified", action="store_true")
    parser.add_argument("--accelerator-runtime-qualified", action="store_true")
    parser.add_argument("--compute-authorized", action="store_true")
    parser.add_argument("--meta-probe", action="store_true")
    args = parser.parse_args()

    dependencies = Scale1BDependencies(
        preceding_stage_admitted=args.preceding_stage_admitted,
        production_tokenizer_qualified=args.production_tokenizer_qualified,
        native_gqa_qualified=args.native_gqa_qualified,
        distributed_checkpoint_qualified=args.distributed_checkpoint_qualified,
        data_pipeline_qualified=args.data_pipeline_qualified,
        accelerator_runtime_qualified=args.accelerator_runtime_qualified,
        compute_authorized=args.compute_authorized,
    )
    payload = assess_scale_1b_readiness(
        args.config,
        dependencies,
        world_sizes=tuple(args.world_sizes),
        microbatch_size=args.microbatch_size,
    ).to_dict()
    if args.meta_probe:
        payload["meta_parameters"] = meta_parameter_probe(args.config)
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
