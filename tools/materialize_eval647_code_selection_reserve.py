#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.eval647_code_selection_reserve_v1 import materialize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--discovery",
        action="store_true",
        help="Allow unsealed raw SHA-256 fields while discovering exact pinned-source hashes.",
    )
    args = parser.parse_args()
    result = materialize(
        args.repo_root.resolve(),
        args.output_dir.resolve(),
        require_sealed=not args.discovery,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
