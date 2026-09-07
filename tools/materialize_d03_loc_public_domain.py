from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from twelve_six.data.loc_public_domain_intake import (
    materialize_shard,
    validate_config,
    verify_local_rights_registry,
)


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_config(value)
    return value


def download_exact(url: str, destination: Path, max_bytes: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-ai/loc-intake-v1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("download exceeds configured safety envelope")
            out.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/d03_loc_public_domain_intake_v1.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    verify_local_rights_registry(config, args.repo_root)
    shard = args.shard
    created = False
    if shard is None:
        if args.download_dir is None:
            parser.error("provide --shard or --download-dir")
        args.download_dir.mkdir(parents=True, exist_ok=True)
        shard = args.download_dir / Path(config["upstream"]["shard_path"]).name
        download_exact(
            config["upstream"]["download_url"],
            shard,
            max_bytes=config["upstream"]["shard_bytes"],
        )
        created = True

    try:
        rows, report = materialize_shard(config, shard)
        args.candidate_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.candidate_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        if created and shard.exists():
            shard.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
