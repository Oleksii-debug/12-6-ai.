from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.integration.late_wave_intake import load_and_validate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate D01 S0 late-wave intake snapshot")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    facts = load_and_validate(args.snapshot, args.repo_root)
    print(json.dumps(facts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
