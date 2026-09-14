from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.hplt3_source_policy import HPLT3ContractError, load_and_validate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the fail-closed HPLT 3.0 Ukrainian audit contract.")
    parser.add_argument(
        "contract",
        nargs="?",
        default="configs/data/d03_hplt3_ukr_cyrl_source_audit_v1.json",
        help="Path to the machine-readable audit contract.",
    )
    args = parser.parse_args()
    try:
        result = load_and_validate(Path(args.contract))
    except (OSError, json.JSONDecodeError, HPLT3ContractError) as exc:
        print(f"HPLT3 Ukrainian source audit: FAIL: {exc}")
        return 1
    print("HPLT3 Ukrainian source audit: PASS")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
