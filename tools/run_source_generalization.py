"""Run or validate the EVAL-137 source-family generalization report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.training.source_generalization import (
    run_source_generalization,
    validate_source_generalization_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source-sha")
    parser.add_argument("--locked-environment-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()

    if args.validate is not None:
        validate_source_generalization_report(_read_json(args.validate))
        print(f"validated {args.validate}")
        return 0

    missing = [
        name
        for name, value in (
            ("--source-sha", args.source_sha),
            ("--locked-environment-evidence", args.locked_environment_evidence),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        parser.error("required for execution: " + ", ".join(missing))

    report = run_source_generalization(
        args.root,
        source_sha=args.source_sha,
        locked_environment_evidence=_read_json(args.locked_environment_evidence),
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())