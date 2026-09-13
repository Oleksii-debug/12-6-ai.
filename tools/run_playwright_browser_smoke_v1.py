#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twelve_six.playwright_browser import ContractError, RuntimeContract, run_real_smoke


def main() -> int:
    try:
        result = run_real_smoke(RuntimeContract(network_mode="deny"))
    except ContractError as exc:
        print(json.dumps({"status": "NOT_EXECUTED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
