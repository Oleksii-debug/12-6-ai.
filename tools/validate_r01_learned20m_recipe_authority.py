#!/usr/bin/env python3
"""Validate the frozen LEARN-345 learned-20M recipe authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.learned20m_recipe import (
    bind_terminal_authorities,
    blocked_template,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs/research/r01_learned20m_recipe_authority_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--bindings",
        type=Path,
        default=None,
        help="Optional terminal authority bindings JSON. Omit for checked-in blocked template.",
    )
    parser.add_argument(
        "--trusted-authorities",
        type=Path,
        default=None,
        help=(
            "Out-of-packet role-bound tokenizer/D04/D05/D06 authority JSON. "
            "Required whenever --bindings is supplied."
        ),
    )
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    validate_policy(policy)
    if args.bindings is None:
        if args.trusted_authorities is not None:
            parser.error("--trusted-authorities requires --bindings")
        result = blocked_template(policy)
    else:
        if args.trusted_authorities is None:
            parser.error("--trusted-authorities is required with --bindings")
        bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
        trusted_authorities = json.loads(
            args.trusted_authorities.read_text(encoding="utf-8")
        )
        result = bind_terminal_authorities(
            policy,
            bindings,
            trusted_authorities=trusted_authorities,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
