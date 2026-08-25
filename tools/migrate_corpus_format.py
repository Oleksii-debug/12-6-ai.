"""Convert an existing verified JSONL corpus to a truth-preserving physical layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.corpus_format import (
    PYARROW_EXPERIMENT_VERSION,
    materialize_layout,
    packed_training_trace,
    verify_layout,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--format", choices=("jsonl", "parquet"), required=True)
    parser.add_argument("--compression", choices=("none", "zstd"), default="zstd")
    parser.add_argument("--row-group-rows", type=int, default=4096)
    parser.add_argument("--verify-packed-trace", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()

    compression = args.compression if args.format == "parquet" else "none"
    manifest = materialize_layout(
        args.source_root,
        args.output_root,
        physical_format=args.format,
        compression=compression,
        row_group_rows=args.row_group_rows,
        expected_pyarrow_version=(
            PYARROW_EXPERIMENT_VERSION if args.format == "parquet" else None
        ),
    )
    verify_layout(args.output_root)
    result: dict[str, object] = {"layout": manifest}
    if args.verify_packed_trace:
        result["packed_training_trace"] = packed_training_trace(
            args.output_root,
            manifest,
            sequence_length=args.sequence_length,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
