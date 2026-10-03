#!/usr/bin/env python3
"""Run clean-retained + PEP + LoC global dedup through canonical indexed V3."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import twelve_six
import twelve_six.data
from twelve_six.data.expanded_global_dedup_v10 import (
    DEFAULT_MAX_CANDIDATE_PAIRS,
    DEFAULT_MAX_INDEX_POSTINGS,
    DEFAULT_MAX_PAIR_EXPANSIONS,
    ExpandedDedupV10Error,
    run_expanded_dedup_v10,
)

EXPECTED_V7_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
V3_MODULE = "twelve_six.data.cross_source_capacity_audit_v3"
V1_MODULE = "twelve_six.data.cross_source_capacity_audit"
DATA232_MODULE = "twelve_six.data._data232_decontamination_matching"
MAX_INPUT_BYTES = 64 * 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpandedDedupV10Error(message)


def read_exact_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ExpandedDedupV10Error(f"{label} must be a regular file: {path}")
    with path.open("rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ExpandedDedupV10Error(f"{label} must be a regular file: {path}")
        if metadata.st_size > MAX_INPUT_BYTES:
            raise ExpandedDedupV10Error(f"{label} exceeds bounded input limit")
        raw = source.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ExpandedDedupV10Error(f"{label} exceeds bounded input limit")
    if not raw:
        raise ExpandedDedupV10Error(f"{label} is empty: {path}")
    return raw


def _json_bytes(value: dict[str, Any]) -> bytes:
    try:
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
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as exc:
        raise ExpandedDedupV10Error("output JSON is not finite/serializable") from exc


def _remove_if_owned(path: Path, identity: tuple[int, int]) -> None:
    """Best-effort rollback: never unlink a visibly replaced output."""
    try:
        observed = path.lstat()
        if (
            stat.S_ISREG(observed.st_mode)
            and (observed.st_dev, observed.st_ino) == identity
        ):
            path.unlink()
    except FileNotFoundError:
        pass


def write_json(path: Path, payload: bytes) -> tuple[int, int]:
    """Create one durable file exclusively, without replacing existing output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    created = os.fstat(fd)
    identity = (created.st_dev, created.st_ino)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        _remove_if_owned(path, identity)
        raise
    return identity


def publish_outputs(
    report_path: Path,
    survivors_path: Path,
    report: dict[str, Any],
    survivors: dict[str, Any],
) -> None:
    """Fail before writing for invalid targets; roll back an ordinary second failure.

    An unexpected hard crash between file creations still requires operator
    recovery; this is not a crash-atomic multi-file transaction.
    """
    if report_path.resolve() == survivors_path.resolve():
        raise ExpandedDedupV10Error("report and survivors output paths must be distinct")
    for path in (report_path, survivors_path):
        if path.exists() or path.is_symlink():
            raise ExpandedDedupV10Error(f"refusing to overwrite output: {path}")
    report_bytes = _json_bytes(report)
    survivors_bytes = _json_bytes(survivors)
    report_identity = write_json(report_path, report_bytes)
    try:
        write_json(survivors_path, survivors_bytes)
    except BaseException:
        _remove_if_owned(report_path, report_identity)
        raise


def _load_exact_v3(v7_root: Path) -> Any:
    """Load only the historical V3 bytes from the exact clean terminal V7 worktree."""
    try:
        root = v7_root.resolve(strict=True)
    except OSError as exc:
        raise ExpandedDedupV10Error("cannot resolve exact V7 worktree") from exc
    _require(root.is_dir(), "V7 root must be a directory")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExpandedDedupV10Error("cannot attest exact V7 worktree") from exc
    _require(head == EXPECTED_V7_HEAD, "V7 worktree HEAD drift")
    _require(dirty == "", "V7 worktree is not clean")

    historical_package = (root / "src" / "twelve_six").resolve(strict=True)
    historical_data = (historical_package / "data").resolve(strict=True)
    for name in (DATA232_MODULE, V1_MODULE, V3_MODULE):
        _require(name not in sys.modules, f"historical matcher preloaded: {name}")

    current_package_path = list(twelve_six.__path__)
    current_data_path = list(twelve_six.data.__path__)
    historical_package_text = str(historical_package)
    historical_data_text = str(historical_data)
    _require(
        historical_package_text not in current_package_path
        and historical_data_text not in current_data_path,
        "historical V7 package path already injected",
    )

    twelve_six.__path__ = [historical_package_text, *current_package_path]
    twelve_six.data.__path__ = [historical_data_text, *current_data_path]
    importlib.invalidate_caches()
    try:
        v3 = importlib.import_module(V3_MODULE)
    finally:
        twelve_six.__path__ = current_package_path
        twelve_six.data.__path__ = current_data_path
        importlib.invalidate_caches()

    raw_path = getattr(v3, "__file__", None)
    _require(type(raw_path) is str and bool(raw_path), "V3 module source path missing")
    v3_path = Path(raw_path).resolve(strict=True)
    _require(v3_path.is_relative_to(historical_data), "V3 module escaped exact V7 worktree")
    return v3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--clean-training-records", type=Path, required=True)
    parser.add_argument("--clean-training-handoff", type=Path, required=True)
    parser.add_argument("--pep-candidate", type=Path, required=True)
    parser.add_argument("--pep-report", type=Path, required=True)
    parser.add_argument("--loc-candidate", type=Path, required=True)
    parser.add_argument("--loc-report", type=Path, required=True)
    parser.add_argument(
        "--max-candidate-pairs",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_PAIRS,
    )
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    args = parser.parse_args()

    try:
        v3 = _load_exact_v3(args.v7_root)
        report, survivors = run_expanded_dedup_v10(
            v3_module=v3,
            clean_training_records_raw=read_exact_bytes(
                args.clean_training_records,
                label="clean DATA-232 training records",
            ),
            clean_training_handoff_raw=read_exact_bytes(
                args.clean_training_handoff,
                label="clean DATA-232 handoff",
            ),
            pep_candidate_raw=read_exact_bytes(
                args.pep_candidate,
                label="PEP candidate",
            ),
            pep_report_raw=read_exact_bytes(args.pep_report, label="PEP report"),
            loc_candidate_raw=read_exact_bytes(
                args.loc_candidate,
                label="LoC candidate",
            ),
            loc_report_raw=read_exact_bytes(args.loc_report, label="LoC report"),
            max_candidate_pairs=args.max_candidate_pairs,
            max_index_postings=args.max_index_postings,
            max_pair_expansions=args.max_pair_expansions,
        )
        publish_outputs(args.output_report, args.output_survivors, report, survivors)
    except (ExpandedDedupV10Error, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print("D03_EXPANDED_GLOBAL_DEDUP_V10_PEP_LOC=PASS_ZERO_CREDIT")
    print("REPORT_SHA256=" + report["report_sha256"])
    print("SURVIVOR_AUTHORITY_SHA256=" + survivors["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("OPTIMIZER_UPDATES_EXECUTED_ON_REAL_TARGETS=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
