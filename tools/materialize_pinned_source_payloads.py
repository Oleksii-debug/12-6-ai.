#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.pinned_source_materialization import materialize_pinned_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize exact terminal source payload bytes without running model training."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = materialize_pinned_sources(
        config,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
    )
    print(
        "PINNED_SOURCE_MATERIALIZATION_JSON="
        + json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
