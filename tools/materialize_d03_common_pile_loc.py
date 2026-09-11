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
    validate_config,
    verify_full_shard,
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

    full_shard_verification = None
    if args.shard is not None:
        shard = args.shard.resolve()
        full_shard_verification = verify_full_shard(shard, config)
        raw = shard.read_bytes()
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
    candidates, report = materialize(
        config,
        records,
        full_shard_verification=full_shard_verification,
    )
    write_materialization(args.output_dir.resolve(), candidates, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
