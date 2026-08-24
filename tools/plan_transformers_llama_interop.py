#!/usr/bin/env python3
"""Emit a deterministic non-promoting Transformers Llama interoperability plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.inference.transformers_llama import build_llama_interop_plan
from twelve_six.model import load_stage_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a 12-6 stage ModelSpec against the conservative Transformers "
            "Llama bridge and emit the exact config/tensor conversion plan."
        )
    )
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stage = load_stage_config(args.stage_config)
    plan = build_llama_interop_plan(stage.model)
    payload = plan.payload()
    payload["plan_sha256"] = plan.identity_sha256()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
