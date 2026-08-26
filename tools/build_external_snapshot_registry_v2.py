#!/usr/bin/env python3
"""Build or verify the DATA-287 external snapshot registry V2."""
from __future__ import annotations

import argparse
from pathlib import Path

from twelve_six.data.external_snapshot_registry_v2 import (
    build_external_snapshot_registry_v2,
    serialize_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/registry/external_snapshots.v2.json"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    kwargs = {
        "inputs_path": root / "configs/data/data287_external_snapshot_registry_v2.json",
        "base_registry_path": root / "data/registry/real_snapshots.v1.json",
    }
    first = serialize_registry(build_external_snapshot_registry_v2(**kwargs))
    second = serialize_registry(build_external_snapshot_registry_v2(**kwargs))
    if first != second:
        raise SystemExit("independent DATA-287 registry builds are not byte-identical")
    output = root / args.output
    if args.verify:
        if not output.is_file():
            raise SystemExit(f"missing committed registry: {output}")
        if output.read_bytes() != first:
            raise SystemExit("committed DATA-287 registry differs from deterministic rebuild")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(first)
    registry = build_external_snapshot_registry_v2(**kwargs)
    print(registry["registry_identity_sha256"])
    print(
        "sources={source_count} families={independent_source_family_count} "
        "raw={raw} normalized={normalized}".format(
            source_count=registry["source_count"],
            independent_source_family_count=registry["independent_source_family_count"],
            raw=registry["byte_report"]["unique_raw_bytes"],
            normalized=registry["byte_report"]["unique_normalized_bytes"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
