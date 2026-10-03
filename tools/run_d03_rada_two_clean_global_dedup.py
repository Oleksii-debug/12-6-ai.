#!/usr/bin/env python3
"""Run the exact Rada indexed global-dedup carrier twice in fresh processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from twelve_six.data.rada_two_clean_dedup_execution import (
    SELECTION_RULE,
    SURVIVOR_SCHEMA,
    RadaTwoCleanExecutionError,
    build_two_clean_authority,
    execute_once,
    validate_dependency_authority,
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_canonical_bytes(dict(value)))
    except FileExistsError as exc:
        raise RadaTwoCleanExecutionError(f"refusing to overwrite output: {path}") from exc
    except OSError as exc:
        raise RadaTwoCleanExecutionError(f"cannot write output: {path}") from exc


def _identity_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _report_identity_bytes(value: Any) -> bytes:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{rendered}\n".encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RadaTwoCleanExecutionError(f"duplicate generated JSON key: {key}")
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise RadaTwoCleanExecutionError(f"generated JSON contains non-finite number: {token}")
    return value


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RadaTwoCleanExecutionError(
                    f"generated JSON contains non-finite constant: {token}"
                )
            ),
            parse_float=_finite_float,
        )
    except RecursionError as exc:
        raise RadaTwoCleanExecutionError(
            f"generated JSON nesting limit exceeded: {path}"
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RadaTwoCleanExecutionError(f"cannot read strict generated JSON: {path}") from exc
    if type(value) is not dict:
        raise RadaTwoCleanExecutionError(f"generated JSON root is not object: {path}")
    return value, raw


_SURVIVOR_KEYS = frozenset(
    {
        "schema_version",
        "selection_rule",
        "v3_report_sha256",
        "pre_dedup_source_object_count",
        "post_dedup_survivor_source_object_count",
        "pre_dedup_declared_capacity_bytes",
        "post_dedup_declared_capacity_bytes",
        "duplicate_discount_bytes",
        "duplicate_cluster_count",
        "duplicate_clusters",
        "survivors",
        "raw_text_emitted",
        "truth_boundary",
        "survivor_authority_sha256",
    }
)
_SURVIVOR_TRUTH_KEYS = frozenset(
    {
        "source_object_authority_only",
        "canonical_capacity_credited",
        "authorized_optimized_target_exposure",
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
    }
)


def _verify_report_artifact(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> str:
    expected = receipt.get("v3_report_sha256")
    if type(expected) is not str or len(expected) != 64:
        raise RadaTwoCleanExecutionError("receipt report identity is invalid")
    observed = report.get("report_sha256")
    if observed != expected:
        raise RadaTwoCleanExecutionError("report artifact identity differs from receipt")
    core = dict(report)
    core.pop("report_sha256", None)
    if _sha256(_report_identity_bytes(core)) != expected:
        raise RadaTwoCleanExecutionError("report artifact self-hash mismatch")
    if report.get("local_free_only") is not True:
        raise RadaTwoCleanExecutionError("report artifact weakened LOCAL_FREE")
    if report.get("model_training_executed") is not False:
        raise RadaTwoCleanExecutionError("report artifact widened training truth")
    if report.get("raw_text_emitted") is not False:
        raise RadaTwoCleanExecutionError("report artifact emitted raw text")
    if (
        type(report.get("source_count")) is not int
        or report.get("source_count") != receipt.get("source_object_count")
    ):
        raise RadaTwoCleanExecutionError("report source count differs from receipt")
    terminal = report.get("terminal_candidates")
    if type(terminal) is not dict:
        raise RadaTwoCleanExecutionError("report terminal_candidates missing")
    if (
        type(terminal.get("conservative_unique_capacity_bytes_after")) is not int
        or terminal.get("conservative_unique_capacity_bytes_after")
        != receipt.get("survivor_declared_capacity_bytes")
    ):
        raise RadaTwoCleanExecutionError("report survivor capacity differs from receipt")
    return expected


def _verify_survivor_artifact(
    survivor: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> str:
    if set(survivor) != _SURVIVOR_KEYS:
        raise RadaTwoCleanExecutionError("survivor artifact schema drift")
    if survivor.get("schema_version") != SURVIVOR_SCHEMA:
        raise RadaTwoCleanExecutionError("survivor artifact version drift")
    if survivor.get("selection_rule") != SELECTION_RULE:
        raise RadaTwoCleanExecutionError("survivor selection rule drift")
    observed = survivor.get("survivor_authority_sha256")
    if type(observed) is not str or len(observed) != 64:
        raise RadaTwoCleanExecutionError("survivor artifact identity is invalid")
    core = dict(survivor)
    core.pop("survivor_authority_sha256", None)
    if _sha256(_identity_bytes(core)) != observed:
        raise RadaTwoCleanExecutionError("survivor artifact self-hash mismatch")
    if observed != receipt.get("survivor_authority_sha256"):
        raise RadaTwoCleanExecutionError("survivor artifact identity differs from receipt")
    if survivor.get("v3_report_sha256") != receipt.get("v3_report_sha256"):
        raise RadaTwoCleanExecutionError("survivor report identity differs from receipt")
    if (
        type(survivor.get("pre_dedup_source_object_count")) is not int
        or survivor.get("pre_dedup_source_object_count") != receipt.get("source_object_count")
    ):
        raise RadaTwoCleanExecutionError("survivor input count differs from receipt")
    if (
        type(survivor.get("post_dedup_survivor_source_object_count")) is not int
        or survivor.get("post_dedup_survivor_source_object_count")
        != receipt.get("survivor_source_object_count")
    ):
        raise RadaTwoCleanExecutionError("survivor output count differs from receipt")
    if (
        type(survivor.get("post_dedup_declared_capacity_bytes")) is not int
        or survivor.get("post_dedup_declared_capacity_bytes")
        != receipt.get("survivor_declared_capacity_bytes")
    ):
        raise RadaTwoCleanExecutionError("survivor output capacity differs from receipt")
    if survivor.get("raw_text_emitted") is not False:
        raise RadaTwoCleanExecutionError("survivor artifact emitted raw text")
    truth = survivor.get("truth_boundary")
    if type(truth) is not dict or set(truth) != _SURVIVOR_TRUTH_KEYS:
        raise RadaTwoCleanExecutionError("survivor truth boundary schema drift")
    if truth.get("source_object_authority_only") is not True:
        raise RadaTwoCleanExecutionError("survivor source-object authority truth drift")
    for key in ("canonical_capacity_credited", "authorized_optimized_target_exposure"):
        if type(truth.get(key)) is not int or truth.get(key) != 0:
            raise RadaTwoCleanExecutionError(f"survivor truth drift: {key}")
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
    ):
        if truth.get(key) is not False:
            raise RadaTwoCleanExecutionError(f"survivor truth drift: {key}")
    return observed


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dependency-authority", type=Path, required=True)
    parser.add_argument("--expected-dependency-authority-sha256", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--base-payload-map", type=Path, required=True)
    parser.add_argument("--expected-base-payload-map-sha256", required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
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
        "--base-payload-map",
        str(args.base_payload_map),
        "--expected-base-payload-map-sha256",
        args.expected_base_payload_map_sha256,
        "--v7-root",
        str(args.v7_root),
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
        base_payload_map_path=args.base_payload_map,
        expected_base_payload_map_sha256=args.expected_base_payload_map_sha256,
        v7_root=args.v7_root,
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


def _write_incomplete(
    output_root: Path,
    completed_run_ids: list[str],
    *,
    reason: str,
) -> None:
    _write_create_only(
        output_root / "incomplete.json",
        {
            "schema_version": "12-6.d03-rada-two-clean-incomplete.v1",
            "status": "INCOMPLETE_NO_SURVIVOR_AUTHORITY",
            "reason": reason,
            "completed_run_ids": list(completed_run_ids),
            "canonical_capacity_credited": 0,
            "authorized_optimized_target_exposure": 0,
            "training_executed": False,
            "paid_compute_used": False,
        },
    )


def _run_worker(command: list[str], *, timeout_seconds: int) -> None:
    process = subprocess.Popen(command)
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            process.kill()
        finally:
            process.wait()
        raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def _run_two_clean(args: argparse.Namespace) -> int:
    # Reject malformed authority without orphaning a create-only output root.
    dependency_authority = validate_dependency_authority(
        args.dependency_authority,
        expected_raw_sha256=args.expected_dependency_authority_sha256,
    )
    worker_timeout_seconds = dependency_authority["worker_timeout_seconds"]

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
            _run_worker(
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
                timeout_seconds=worker_timeout_seconds,
            )
            completed.append(run_id)
    except subprocess.TimeoutExpired as exc:
        _write_incomplete(args.output_root, completed, reason="worker_timeout")
        raise RadaTwoCleanExecutionError("two-clean worker exceeded authority deadline") from exc
    except KeyboardInterrupt:
        _write_incomplete(args.output_root, completed, reason="operator_interrupt")
        raise
    except (OSError, subprocess.CalledProcessError) as exc:
        _write_incomplete(args.output_root, completed, reason="worker_execution_failed")
        raise RadaTwoCleanExecutionError("two-clean execution did not complete") from exc

    try:
        first, _ = _read_json(args.output_root / "clean-a" / "run-receipt.json")
        second, _ = _read_json(args.output_root / "clean-b" / "run-receipt.json")
        authority = build_two_clean_authority(first, second)

        first_report, first_report_bytes = _read_json(
            args.output_root / "clean-a" / "dedup-report.json"
        )
        second_report, second_report_bytes = _read_json(
            args.output_root / "clean-b" / "dedup-report.json"
        )
        first_report_id = _verify_report_artifact(first_report, first)
        second_report_id = _verify_report_artifact(second_report, second)
        if first_report_id != second_report_id or first_report_bytes != second_report_bytes:
            raise RadaTwoCleanExecutionError("two-clean report artifacts differ")

        first_survivor, first_survivor_bytes = _read_json(
            args.output_root / "clean-a" / "survivor-authority.json"
        )
        second_survivor, second_survivor_bytes = _read_json(
            args.output_root / "clean-b" / "survivor-authority.json"
        )
        first_survivor_id = _verify_survivor_artifact(first_survivor, first)
        second_survivor_id = _verify_survivor_artifact(second_survivor, second)
        if (
            first_survivor_id != second_survivor_id
            or first_survivor != second_survivor
            or first_survivor_bytes != second_survivor_bytes
        ):
            raise RadaTwoCleanExecutionError("two-clean survivor artifacts differ")
        if authority["v3_report_sha256"] != first_report_id:
            raise RadaTwoCleanExecutionError("final authority report identity drift")
        if authority["survivor_authority_sha256"] != first_survivor_id:
            raise RadaTwoCleanExecutionError("final authority survivor identity drift")

        _write_create_only(args.output_root / "two-clean-authority.json", authority)
    except (OSError, RadaTwoCleanExecutionError) as exc:
        _write_incomplete(
            args.output_root,
            completed,
            reason="post_run_convergence_failed",
        )
        if isinstance(exc, RadaTwoCleanExecutionError):
            raise
        raise RadaTwoCleanExecutionError("cannot finalize two-clean authority") from exc
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
