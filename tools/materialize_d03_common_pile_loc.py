#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from twelve_six.data.common_pile_loc_intake import (
    LocIntakeError,
    iter_gzip_jsonl_bytes,
    materialize,
    materialize_verified_shard,
    validate_config,
    write_materialization,
)

CONFIG = Path("configs/data/d03_common_pile_loc_intake_v1.json")


def load_config(repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / CONFIG).read_text(encoding="utf-8"))


def fetch_remote_prefix(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{max_bytes - 1}",
            "User-Agent": "12-6-ai-d03-loc-intake-v1",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        if status not in {200, 206}:
            raise LocIntakeError(f"remote shard returned HTTP {status}")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    config = load_config(root)
    validate_config(config)
    selection = config["selection_policy"]

    if args.shard is not None:
        # Read exactly once. The verified entrypoint hashes and materializes this
        # same immutable byte snapshot, eliminating receipt/record substitution
        # and verify-then-read TOCTOU seams.
        raw = args.shard.resolve().read_bytes()
        candidates, report = materialize_verified_shard(config, raw)
    else:
        raw = fetch_remote_prefix(
            config["upstream"]["resolve_url"],
            selection["max_remote_compressed_prefix_bytes"],
        )
        records = iter_gzip_jsonl_bytes(
            raw,
            max_jsonl_line_bytes=selection["max_jsonl_line_bytes"],
            skip_oversize_lines=True,
        )
        candidates, report = materialize(config, records)

    write_materialization(args.output_dir.resolve(), candidates, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
