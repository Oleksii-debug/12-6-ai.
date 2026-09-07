from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from twelve_six.data.loc_books_intake import (
    load_config,
    materialize_bound_shard,
    verify_two_builds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the bounded zero-credit Library of Congress source candidate twice."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    build_a = args.out_dir / "build_a"
    build_b = args.out_dir / "build_b"

    materialize_bound_shard(
        config,
        args.shard,
        candidate_path=build_a / "candidate.jsonl",
        report_path=build_a / "report.json",
    )
    materialize_bound_shard(
        config,
        args.shard,
        candidate_path=build_b / "candidate.jsonl",
        report_path=build_b / "report.json",
    )
    result = verify_two_builds(
        candidate_a=build_a / "candidate.jsonl",
        report_a=build_a / "report.json",
        candidate_b=build_b / "candidate.jsonl",
        report_b=build_b / "report.json",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
