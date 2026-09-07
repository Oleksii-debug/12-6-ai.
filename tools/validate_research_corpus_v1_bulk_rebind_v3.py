"""Validate the D10 V5/dedup Research Corpus V1 bulk-acquisition rebind."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.bulk_acquisition_rebind_v3 import load_and_validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/research_corpus_v1_bulk_rebind_v3.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = load_and_validate(args.config, args.repo_root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
