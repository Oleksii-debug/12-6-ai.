#!/usr/bin/env python3
"""Evaluate the learned ~20M launch gate from terminal evidence JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.learned20_launch_gate import assess_launch, validate_contract

DEFAULT_CONTRACT = ROOT / "configs/training/learned_20m_launch_gate_v1.json"


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--material-cost", action="store_true")
    args = parser.parse_args(argv)

    contract = _load(args.contract)
    errors = validate_contract(contract)
    if errors:
        print(json.dumps({"contract_valid": False, "errors": errors}, indent=2))
        return 1

    if args.evidence is None:
        print(
            json.dumps(
                {
                    "contract_valid": True,
                    "status": contract["status"],
                    "default_decision": contract["default_decision"],
                },
                indent=2,
            )
        )
        return 0

    evidence = _load(args.evidence)
    result = assess_launch(contract, evidence, material_cost=args.material_cost)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pilot_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
