#!/usr/bin/env python3
"""Bind or verify the learned-20M canonical byte-tokenizer decision authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.tokenization.decision_authority import (
    bind_byte_baseline_decision,
    verify_byte_baseline_decision,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-authority", type=Path, required=True)
    parser.add_argument("--split-authority", type=Path, required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--corpus-terminal-status", required=True)
    parser.add_argument("--split-terminal-status", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus = _load(args.corpus_authority)
    split = _load(args.split_authority)

    if args.verify_report is not None:
        report = _load(args.verify_report)
        verify_byte_baseline_decision(
            report,
            corpus,
            split,
            expected_corpus_sha256=args.expected_corpus_sha256,
            expected_split_sha256=args.expected_split_sha256,
            corpus_terminal_status=args.corpus_terminal_status,
            split_terminal_status=args.split_terminal_status,
        )
    else:
        report = bind_byte_baseline_decision(
            corpus,
            split,
            expected_corpus_sha256=args.expected_corpus_sha256,
            expected_split_sha256=args.expected_split_sha256,
            corpus_terminal_status=args.corpus_terminal_status,
            split_terminal_status=args.split_terminal_status,
        )

    if args.output is not None:
        _write(args.output, report)
    else:
        print(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
