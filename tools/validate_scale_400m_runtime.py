from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from twelve_six.model import load_stage_config
from twelve_six.training.scale_runtime import (
    build_meta_decoder,
    estimate_scale_resources,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / "configs" / "stages" / "s5_400m.scale05_candidate.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the SCALE-05 400M candidate without materializing its weights."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--fsdp2-sharded", action="store_true")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    args = parser.parse_args()

    candidate = load_stage_config(args.config)
    checkpointing = not args.no_activation_checkpointing

    model = build_meta_decoder(
        candidate.model,
        candidate.init,
        activation_checkpointing=checkpointing,
    )
    estimate = estimate_scale_resources(
        candidate.model,
        sequence_length=args.sequence_length,
        microbatch_size=args.microbatch_size,
        activation_checkpointing=checkpointing,
        world_size=args.world_size,
        fsdp2_sharded=args.fsdp2_sharded,
    )
    payload = {
        "stage": candidate.stage,
        "model_identity_sha256": candidate.model.identity_sha256(),
        "parameters": candidate.model.parameter_count(),
        "meta_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "all_parameters_meta": all(
            parameter.device.type == "meta" for parameter in model.parameters()
        ),
        "resource_estimate": asdict(estimate),
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
