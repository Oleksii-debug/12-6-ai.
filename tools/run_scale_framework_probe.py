from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.framework_adapter import execute_local_scale_smoke
from twelve_six.model import load_stage_config
from twelve_six.training.config import TrainerConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the LOCAL_FREE 12-6 scale-framework mechanics probe."
    )
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=Path("configs/stages/s1_100k.json"),
    )
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19019)
    args = parser.parse_args()

    stage = load_stage_config(args.stage_config)
    trainer = TrainerConfig(max_steps=1, seed=args.seed)
    evidence = execute_local_scale_smoke(
        stage.model,
        stage.init,
        trainer,
        ParallelPlan(),
        sequence_length=args.sequence_length,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
