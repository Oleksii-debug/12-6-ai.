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

from twelve_six.learned20_launch_gate import validate_contract
from twelve_six.learned20_pilot_authority import assess_launch_with_terminal_provenance

DEFAULT_CONTRACT = ROOT / "configs/training/learned_20m_launch_gate_v1.json"
MODEL341_INITSPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _validate_exact_initspec(evidence: dict) -> list[str]:
    """Fail closed unless launch evidence binds the canonical MODEL-341 InitSpec."""
    binding = evidence.get("launch_binding")
    if not isinstance(binding, dict):
        return ["launch_binding.initspec_identity_missing"]
    if binding.get("initspec_identity") != MODEL341_INITSPEC_SHA256:
        return ["launch_binding.initspec_identity_mismatch"]
    return []


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
    initspec_errors = _validate_exact_initspec(evidence)
    if initspec_errors:
        print(
            json.dumps(
                {
                    "pilot_ready": False,
                    "long_training_ready": False,
                    "pilot_blockers": initspec_errors,
                    "long_training_blockers": initspec_errors,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    result = assess_launch_with_terminal_provenance(
        contract,
        evidence,
        material_cost=args.material_cost,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    ready = result["long_training_ready"] if args.material_cost else result["pilot_ready"]
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
