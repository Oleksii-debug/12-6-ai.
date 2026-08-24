#!/usr/bin/env python3
"""Validate the exact PR #81 S0 LOCAL_FREE execution handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.integration.s0_exact_handoff import validate_exact_handoff


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--handoff",
        type=Path,
        default=Path("configs/releases/s0_exact_handoff_20260824.prepared.json"),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(
            "configs/releases/s0_candidate_convergence_20260824.experimental.json"
        ),
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path("configs/runs/s0_10k.pr81_exact_candidate.local_free.json"),
    )
    args = parser.parse_args()

    result = validate_exact_handoff(
        _read_json(args.handoff),
        _read_json(args.candidate),
        _read_json(args.run_manifest),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
