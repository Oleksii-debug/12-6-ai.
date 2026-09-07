"""CLI for fresh reserved-evaluation decontamination over retained source payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.data.fresh_reserved_decontamination_v1 import (
    execute_fresh_reserved_decontamination,
    verify_fresh_reserved_decontamination,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _payloads(root: Path, manifest: Path) -> dict[str, bytes]:
    root = root.resolve()
    rows = _read_json(manifest)
    if not isinstance(rows, list) or not rows:
        raise ValueError("payload manifest must be a non-empty list")
    result: dict[str, bytes] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("payload manifest rows must be objects")
        source_id = row.get("source_id")
        relative = row.get("path")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("payload manifest source_id must be non-empty")
        if not isinstance(relative, str) or not relative:
            raise ValueError("payload manifest path must be non-empty")
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ValueError("payload manifest path escapes payload root")
        if source_id in result:
            raise ValueError("payload manifest source_id is not unique")
        result[source_id] = candidate.read_bytes()
    return result


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"refusing to overwrite immutable report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--expected-survivor-sha256", required=True)
    parser.add_argument("--selection-validation-identity", required=True)
    parser.add_argument("--final-test-identity", required=True)
    parser.add_argument("--postdedup-handoff-git-sha", required=True)
    parser.add_argument("--data232-matcher-git-sha", required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan")
    scan.add_argument("--inventory", type=Path, required=True)
    scan.add_argument("--payload-root", type=Path, required=True)
    scan.add_argument("--payload-manifest", type=Path, required=True)
    scan.add_argument("--selection-validation-jsonl", type=Path, required=True)
    scan.add_argument("--authorities", type=Path, required=True)
    scan.add_argument("--report", type=Path, required=True)
    _common(scan)

    verify = sub.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    _common(verify)

    args = parser.parse_args()
    common = {
        "expected_inventory_identity_sha256": args.expected_inventory_sha256,
        "expected_survivor_authority_sha256": args.expected_survivor_sha256,
        "selection_validation_identity": args.selection_validation_identity,
        "final_test_identity": args.final_test_identity,
        "postdedup_handoff_git_sha": args.postdedup_handoff_git_sha,
        "data232_matcher_git_sha": args.data232_matcher_git_sha,
    }
    if args.command == "scan":
        report = execute_fresh_reserved_decontamination(
            _read_json(args.inventory),
            _payloads(args.payload_root, args.payload_manifest),
            _read_jsonl(args.selection_validation_jsonl),
            _read_json(args.authorities),
            **common,
        )
        _write_immutable(args.report, report)
        return 0

    verify_fresh_reserved_decontamination(_read_json(args.report), **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
