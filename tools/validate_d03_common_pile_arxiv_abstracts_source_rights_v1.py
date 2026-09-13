#!/usr/bin/env python3
"""Validate the frozen Common Pile arXiv-abstracts source-rights authority."""

from __future__ import annotations

import argparse
from pathlib import Path

from twelve_six.common_pile_arxiv_abstracts_rights import load_and_validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/data/d03_common_pile_arxiv_abstracts_source_rights_v1.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    policy = load_and_validate(args.policy, repo_root=args.repo_root)
    print(policy["policy_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
