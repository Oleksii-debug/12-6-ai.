#!/usr/bin/env python3
"""Assess the fail-closed learned-20M launch packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.learned20m_readiness import (
    assess_learned20m_readiness,
    trusted_readiness_inputs,
)

DEFAULT_PATH = Path("configs/research/r01_learned20m_launch_readiness_v1.json")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--trusted-bindings",
        type=Path,
        help=(
            "separate trusted authority/metadata bundle; never derive this bundle "
            "from the candidate packet under assessment"
        ),
    )
    args = parser.parse_args(argv[1:])

    try:
        payload = _load_json_object(args.packet, "launch packet")
        verified_scientific: set[str] = set()
        verified_refs: set[str] = set()
        if args.trusted_bindings is not None:
            bindings = _load_json_object(args.trusted_bindings, "trusted bindings")
            resolved = trusted_readiness_inputs(bindings)
            if resolved is None:
                raise ValueError("trusted bindings are malformed or fail closed")
            verified_scientific, verified_refs = resolved
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2

    result = assess_learned20m_readiness(
        payload,
        verified_scientific_authorities=verified_scientific,
        verified_authorization_refs=verified_refs,
    ).as_dict()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["material_training_authorized"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
