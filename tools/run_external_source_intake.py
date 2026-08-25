#!/usr/bin/env python3
"""Run the DATA-21/22 bounded real external-source intake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.source_intake import load_candidate_registry, run_bounded_intake


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="configs/data/external_source_candidates_ua_en_v1.json",
    )
    parser.add_argument("--output", default="external-source-intake-evidence")
    parser.add_argument("--max-download-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-normalized-chars", type=int, default=50_000)
    args = parser.parse_args()

    registry, _sources = load_candidate_registry(args.registry)
    manifest = run_bounded_intake(
        registry,
        Path(args.output),
        max_download_bytes=args.max_download_bytes,
        max_normalized_chars=args.max_normalized_chars,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
