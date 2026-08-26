from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.snapshot_promotion import promote_snapshots


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote exact DATA-21/22 objects through DATA-24/D03 gates."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/data/data181_real_snapshot_promotion_v1.json"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/external/external_sources.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data181-evidence"))
    parser.add_argument("--max-download-bytes", type=int, default=2_000_000)
    args = parser.parse_args()

    report = promote_snapshots(
        repo_root=args.repo_root,
        plan_path=args.plan,
        registry_path=args.registry,
        evidence_dir=args.output,
        max_download_bytes=args.max_download_bytes,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
