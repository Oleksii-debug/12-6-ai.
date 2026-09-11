#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.learned20m_evaluation_firewall import validate_policy

DEFAULT_POLICY = Path("configs/evaluation/learned20m_evaluation_firewall_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    path = args.policy if args.policy.is_absolute() else args.repo_root / args.policy
    policy = json.loads(path.read_text(encoding="utf-8"))
    result = validate_policy(policy)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
