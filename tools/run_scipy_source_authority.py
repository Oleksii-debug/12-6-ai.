from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.permissive_repo_source_authority import (
    load_and_validate_source_authority,
)

DEFAULT_AUTHORITY = Path("configs/data/scipy_v118_source_authority_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the bounded SciPy v1.18.0 source authority."
    )
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    args = parser.parse_args()
    summary = load_and_validate_source_authority(args.authority)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
