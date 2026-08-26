#!/usr/bin/env python3
"""Build or verify the immutable DATA-229 real snapshot registry."""
from __future__ import annotations

import argparse
from pathlib import Path

from twelve_six.data.real_snapshot_registry import build_real_snapshot_registry, serialize_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("data/registry/real_snapshots.v1.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    registry = build_real_snapshot_registry(
        inputs_path=root / "configs/data/data229_real_snapshot_registry_v1.json",
        data213_plan_path=root / "configs/data/data181_real_snapshot_promotion_v1.json",
        data24_registry_path=root / "data/external/external_sources.json",
        data213_report_path=root / "evidence/data229/data213-promotion-report.json",
        data213_artifact_manifest_path=root / "evidence/data229/data213-artifact-manifest.json",
    )
    payload = serialize_registry(registry)
    output = root / args.output
    if args.verify:
        if not output.is_file():
            raise SystemExit(f"missing committed registry: {output}")
        if output.read_bytes() != payload:
            raise SystemExit("committed registry differs from deterministic rebuild")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    print(registry["registry_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
