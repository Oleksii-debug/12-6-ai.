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
            "Assess S6 ~1B dependency readiness. Every launch gate remains blocked "
            "unless the caller binds it to a durable authority reference."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--preceding-stage-authority")
    parser.add_argument("--production-tokenizer-authority")
    parser.add_argument("--native-gqa-authority")
    parser.add_argument("--distributed-checkpoint-authority")
    parser.add_argument("--data-pipeline-authority")
    parser.add_argument("--accelerator-runtime-authority")
    parser.add_argument(
        "--compute-authorization",
        help=(
            "Explicit owner authorization reference. Must begin with "
            "COMPUTE_AUTHORIZED: or TRAINING_AUTHORIZED:."
        ),
    )
    parser.add_argument("--meta-probe", action="store_true")
    args = parser.parse_args()

    dependencies = Scale1BDependencies(
        preceding_stage_authority=args.preceding_stage_authority,
        production_tokenizer_authority=args.production_tokenizer_authority,
        native_gqa_authority=args.native_gqa_authority,
        distributed_checkpoint_authority=args.distributed_checkpoint_authority,
        data_pipeline_authority=args.data_pipeline_authority,
        accelerator_runtime_authority=args.accelerator_runtime_authority,
        compute_authorization=args.compute_authorization,
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
