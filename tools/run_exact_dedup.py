#!/usr/bin/env python3
"""Materialize DATA-25 corpus V0.1 when needed and execute DATA-29 exact dedup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.data.exact_dedup import (
    ExactDedupPolicy,
    rebuild_twice_and_assert_identical,
    run_exact_dedup,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus-manifest", type=Path)
    source.add_argument("--corpus-config", type=Path)
    parser.add_argument("--materialized-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-twice", action="store_true")
    parser.add_argument("--stop-after-input-shards", type=int)
    args = parser.parse_args()

    if args.corpus_config is not None:
        if args.resume:
            parser.error("--resume requires a retained --corpus-manifest; do not rebuild input")
        materialized = args.materialized_input or args.output.parent / "data25-materialized"
        build_corpus(args.corpus_config, materialized)
        manifest = materialized / "manifest.json"
    else:
        manifest = args.corpus_manifest

    policy = ExactDedupPolicy()
    if args.rebuild_twice:
        if args.resume or args.stop_after_input_shards is not None:
            parser.error("--rebuild-twice cannot be combined with resume/partial execution")
        result = rebuild_twice_and_assert_identical(
            corpus_manifest=manifest,
            output_root=args.output,
            policy=policy,
        )
    else:
        result = run_exact_dedup(
            corpus_manifest=manifest,
            output_dir=args.output,
            policy=policy,
            resume=args.resume,
            stop_after_input_shards=args.stop_after_input_shards,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
