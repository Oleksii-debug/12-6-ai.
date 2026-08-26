#!/usr/bin/env python3
"""Evaluate a pinned corpus-readiness policy without starting training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.readiness import CorpusReadinessError, evaluate_policy_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = Path("configs/data/20m_capability_readiness.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    try:
        report = evaluate_policy_file(args.policy, repo_root=args.repo_root)
    except CorpusReadinessError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 2

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = args.repo_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
