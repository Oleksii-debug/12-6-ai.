from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.integrations.lm_eval import component_manifest, simple_evaluate_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one verified 12-6 Base checkpoint with lm-eval 0.4.12."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = simple_evaluate_checkpoint(
        args.checkpoint,
        args.tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    envelope = {
        "integration": component_manifest(),
        "checkpoint": str(args.checkpoint),
        "tasks": args.tasks,
        "result": result,
    }
    text = json.dumps(envelope, ensure_ascii=False, indent=2, default=str) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
