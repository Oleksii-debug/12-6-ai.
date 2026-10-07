#!/usr/bin/env python3
"""Validate the exact Common Pile LoC source-rights authority against this checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from twelve_six.common_pile_loc_rights import load_and_validate

DEFAULT_POLICY = Path("configs/data/d03_common_pile_loc_source_rights_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    payload = load_and_validate(args.policy, repo_root=args.repo_root)
    print(
        "LOC_SOURCE_RIGHTS_POLICY_OK "
        f"authority={payload['authority_id']} "
        f"decision={payload['project_decision']['decision']} "
        "training_authorized_bytes=0 optimized_target_exposure=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
