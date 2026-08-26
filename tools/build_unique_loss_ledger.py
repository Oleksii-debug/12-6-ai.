#!/usr/bin/env python3
"""Build or verify the DATA-294 unique causal-loss exposure ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.unique_loss_ledger import (
    build_unique_loss_ledger,
    canonical_json_bytes,
)


def _load(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: JSON object required")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/data294/unique-loss-ledger.v1.json"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    registry = _load(root / "data/registry/real_snapshots.v1.json")
    reservations = _load(root / "configs/data/data294_reserved_eval_ranges_v1.json")
    ledger = build_unique_loss_ledger(registry, reservations)
    payload = canonical_json_bytes(ledger)
    output = root / args.output
    if args.verify:
        if not output.is_file():
            raise SystemExit(f"missing committed DATA-294 ledger: {output}")
        if output.read_bytes() != payload:
            raise SystemExit("committed DATA-294 ledger differs from deterministic rebuild")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    print(ledger["ledger_identity_sha256"])
    print(ledger["one_pass_max_unique_optimized_targets"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
