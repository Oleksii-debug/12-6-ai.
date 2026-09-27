#!/usr/bin/env python3
"""Run the exact Rada indexed global-dedup carrier twice in fresh processes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from twelve_six.data.rada_two_clean_dedup_execution import (
    RadaTwoCleanExecutionError,
    build_two_clean_authority,
    execute_once,
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_canonical_bytes(dict(value)))
    except FileExistsError as exc:
        raise RadaTwoCleanExecutionError(f"refusing to overwrite output: {path}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RadaTwoCleanExecutionError(f"cannot read generated JSON: {path}") from exc
    if type(value) is not dict:
        raise RadaTwoCleanExecutionError(f"generated JSON root is not object: {path}")
    return value


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dependency-authority", type=Path, required=True)
    parser.add_argument("--expected-dependency-authority-sha256", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--execution-evidence", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-two-clean")
    _add_common(run)
    run.add_argument("--output-root", type=Path, required=True)

    worker = subparsers.add_parser("worker")
    _add_common(worker)
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--output-dir", type=Path, required=True)

    return parser.parse_args()


def _common_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--dependency-authority",
        str(args.dependency_authority),
        "--expected-dependency-authority-sha256",
        args.expected_dependency_authority_sha256,
        "--inventory",
        str(args.inventory),
        "--expected-inventory-sha256",
        args.expected_inventory_sha256,
        "--candidate-jsonl",
        str(args.candidate_jsonl),
        "--quality-report",
        str(args.quality_report),
        "--execution-evidence",
        str(args.execution_evidence),
    ]


def _worker(args: argparse.Namespace) -> int:
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RadaTwoCleanExecutionError(
            f"fresh worker output directory already exists: {args.output_dir}"
        ) from exc

    report, survivor, receipt = execute_once(
        run_id=args.run_id,
        dependency_authority_path=args.dependency_authority,
        expected_dependency_authority_sha256=args.expected_dependency_authority_sha256,
        inventory_path=args.inventory,
        expected_inventory_sha256=args.expected_inventory_sha256,
        candidate_jsonl=args.candidate_jsonl,
        quality_report=args.quality_report,
        execution_evidence=args.execution_evidence,
    )
    _write_create_only(args.output_dir / "dedup-report.json", report)
    _write_create_only(args.output_dir / "survivor-authority.json", survivor)
    _write_create_only(args.output_dir / "run-receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": "PASS_ZERO_CREDIT_SINGLE_CLEAN_RUN",
                "run_id": args.run_id,
                "v3_report_sha256": receipt["v3_report_sha256"],
                "survivor_authority_sha256": receipt["survivor_authority_sha256"],
                "canonical_capacity_credited": 0,
                "authorized_optimized_target_exposure": 0,
                "training_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_two_clean(args: argparse.Namespace) -> int:
    try:
        args.output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RadaTwoCleanExecutionError(
            f"refusing non-fresh output root: {args.output_root}"
        ) from exc

    script = Path(__file__).resolve()
    common = _common_argv(args)
    runs = (("clean-a", args.output_root / "clean-a"), ("clean-b", args.output_root / "clean-b"))
    completed: list[str] = []
    try:
        for run_id, output_dir in runs:
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "worker",
                    *common,
                    "--run-id",
                    run_id,
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
            )
            completed.append(run_id)
    except (OSError, subprocess.CalledProcessError) as exc:
        incomplete = {
            "schema_version": "12-6.d03-rada-two-clean-incomplete.v1",
            "status": "INCOMPLETE_NO_SURVIVOR_AUTHORITY",
            "completed_run_ids": completed,
            "canonical_capacity_credited": 0,
            "authorized_optimized_target_exposure": 0,
            "training_executed": False,
            "paid_compute_used": False,
        }
        _write_create_only(args.output_root / "incomplete.json", incomplete)
        raise RadaTwoCleanExecutionError("two-clean execution did not complete") from exc

    first = _read_json(args.output_root / "clean-a" / "run-receipt.json")
    second = _read_json(args.output_root / "clean-b" / "run-receipt.json")
    authority = build_two_clean_authority(first, second)

    first_survivor = _read_json(args.output_root / "clean-a" / "survivor-authority.json")
    second_survivor = _read_json(args.output_root / "clean-b" / "survivor-authority.json")
    if first_survivor != second_survivor:
        raise RadaTwoCleanExecutionError("two-clean survivor artifact bytes differ")

    _write_create_only(args.output_root / "survivor-authority.json", first_survivor)
    _write_create_only(args.output_root / "two-clean-authority.json", authority)
    print(
        json.dumps(
            {
                "status": "PASS_TWO_CLEAN_RADA_SURVIVOR_AUTHORITY_ZERO_CREDIT",
                "two_clean_authority_sha256": authority["two_clean_authority_sha256"],
                "survivor_authority_sha256": authority["survivor_authority_sha256"],
                "survivor_source_object_count": authority["survivor_source_object_count"],
                "survivor_declared_capacity_bytes": authority[
                    "survivor_declared_capacity_bytes"
                ],
                "canonical_capacity_credited": 0,
                "authorized_optimized_target_exposure": 0,
                "training_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "worker":
        return _worker(args)
    return _run_two_clean(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RadaTwoCleanExecutionError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
