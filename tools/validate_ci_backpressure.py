from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.ci_backpressure import build_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate PR-scoped cancellation across the active GitHub Actions inventory."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("configs/ci/legacy_workflow_backpressure_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    inventory = args.inventory
    if not inventory.is_absolute():
        inventory = repo_root / inventory

    report = build_report(repo_root, inventory)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = repo_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
