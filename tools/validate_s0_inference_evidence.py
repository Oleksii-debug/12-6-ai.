from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.inference.s0_evidence import validate_s0_trained_inference_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate retained trained-checkpoint S0 inference evidence."
    )
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("evidence file must contain a JSON object")
    result = validate_s0_trained_inference_evidence(
        payload,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
