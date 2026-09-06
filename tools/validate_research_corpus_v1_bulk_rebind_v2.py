#!/usr/bin/env python3
"""Validate the Research Corpus V1 bulk acquisition V2 rebind."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.bulk_acquisition_rebind_v2 import load_and_validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/research_corpus_v1_bulk_rebind_v2.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = load_and_validate(args.config, args.repo_root.resolve())
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
