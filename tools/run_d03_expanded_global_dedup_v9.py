#!/usr/bin/env python3
"""Run expanded D03 global dedup with the exact terminal #824 matcher authority."""
from __future__ import annotations

import argparse
import copy
import json
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

import compose_data526_records_from_v8 as data526
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data.expanded_global_dedup_v9 import ExpandedDedupError, run_expanded_dedup

_V7_TRACKED_AUTHORITY_PATHS = (
    "src/twelve_six/data/cross_source_capacity_audit_v7.py",
    "src/twelve_six/data/cross_source_capacity_audit_v6.py",
    "src/twelve_six/data/cross_source_capacity_audit_v5.py",
    "src/twelve_six/data/cross_source_capacity_audit_v4.py",
    "src/twelve_six/data/cross_source_capacity_audit_v3.py",
    "src/twelve_six/data/cross_source_capacity_audit.py",
    "src/twelve_six/data/_data232_decontamination_matching.py",
    "src/twelve_six/data/pipeline.py",
    "configs/data/next100_065_cross_source_dedup_v3.json",
    "configs/data/next100_065b_cross_source_dedup_v4.json",
    "configs/data/next100_065c_cross_source_dedup_v5.json",
    "configs/data/next100_065d_cross_source_dedup_v6.json",
    "configs/data/next100_065e_cross_source_dedup_v7.json",
)


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExpandedDedupError(f"cannot execute git for V7 authority: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise ExpandedDedupError(f"V7 git authority check failed: {detail}")
    return proc.stdout.strip()


def validate_v7_checkout(v7_root: Path) -> Path:
    """Require an exact clean terminal V7 checkout before any historical replay."""

    if v7_root.is_symlink():
        raise ExpandedDedupError("V7 root must not be a symlink")
    try:
        root = v7_root.resolve(strict=True)
    except OSError as exc:
        raise ExpandedDedupError(f"V7 root cannot be resolved: {exc}") from exc
    if not root.is_dir():
        raise ExpandedDedupError("V7 root must be a directory")

    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise ExpandedDedupError("V7 root must be the Git worktree root")
    head = _git(root, "rev-parse", "HEAD")
    if head != v8.EXPECTED_V7_HEAD:
        raise ExpandedDedupError(
            f"V7 checkout head drift: expected {v8.EXPECTED_V7_HEAD}, observed {head}"
        )
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ExpandedDedupError("V7 checkout must be clean with no untracked files")

    # HEAD+clean binds the complete tracked tree.  Explicitly re-hash the executable
    # matcher/reconstruction/config closure so ignored/shadow artifacts cannot stand
    # in for any authority-bearing path.
    for relative in _V7_TRACKED_AUTHORITY_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ExpandedDedupError(f"V7 authority path is not a regular file: {relative}")
        tracked = _git(root, "ls-files", "--error-unmatch", "--", relative)
        if tracked != relative:
            raise ExpandedDedupError(f"V7 authority path is not tracked exactly: {relative}")
        expected_blob = _git(root, "rev-parse", f"HEAD:{relative}")
        observed_blob = _git(root, "hash-object", str(path))
        if observed_blob != expected_blob:
            raise ExpandedDedupError(f"V7 authority blob drift: {relative}")
    return root


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExpandedDedupError(f"JSON root must be object: {path}")
    return value


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ExpandedDedupError(f"JSONL must be a regular file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ExpandedDedupError(f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise ExpandedDedupError(f"blank JSONL line at {line_no}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ExpandedDedupError(f"JSONL row {line_no} must be object")
        rows.append(value)
    return rows, raw


def reconstruct_v8_source_inputs(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    v8_config: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes]]:
    exact_v7_root = validate_v7_checkout(v7_root)
    v7, _, inventory, payloads = v8._capture_terminal_v7(exact_v7_root, v8_config)
    _, bulk_rows, bulk_payloads = v8._materialize_bulk(ROOT, bulk_workspace, v8_config)
    combined_inventory = copy.deepcopy(inventory)
    existing = {str(row["source_id"]) for row in combined_inventory.get("sources", [])}
    if existing & set(bulk_payloads):
        raise ExpandedDedupError("reconstructed V7/bulk source-id collision")
    combined_inventory["sources"] = [*combined_inventory["sources"], *bulk_rows]
    combined_inventory["final_refresh_required"] = False
    combined_inventory["terminal_refresh_rule"] = (
        "Reconstructed exact V8 input graph solely to recover sealed V8 survivor payloads; "
        "expanded matching is performed once by the exact terminal #824 V3 semantic closure."
    )
    combined_payloads = dict(payloads)
    combined_payloads.update(bulk_payloads)
    return v7.v6.v3, combined_inventory, combined_payloads


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ExpandedDedupError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument(
        "--v8-config",
        type=Path,
        default=ROOT / "configs/data/next100_065f_global_dedup_v8.json",
    )
    parser.add_argument(
        "--data526-config",
        type=Path,
        default=ROOT / "configs/data/data526_v8_record_composition_v1.json",
    )
    parser.add_argument("--v8-report", type=Path, required=True)
    parser.add_argument("--v8-survivors", type=Path, required=True)
    parser.add_argument(
        "--data526-evidence",
        type=Path,
        default=ROOT / "evidence/data526/v8/materialization_evidence.json",
    )
    parser.add_argument(
        "--data526-record-inventory",
        type=Path,
        default=ROOT / "evidence/data526/v8/record_inventory.json",
    )
    parser.add_argument(
        "--rada-language-report",
        type=Path,
        default=(
            ROOT / "evidence/d03-rada-trees/secondary-plaintext-language-gate-v1.json"
        ),
    )
    parser.add_argument("--rada-quality-privacy-jsonl", type=Path, required=True)
    parser.add_argument("--rada-quality-privacy-report", type=Path, required=True)
    parser.add_argument("--expected-rada-report-sha256", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    args = parser.parse_args()

    try:
        v8_config = v8.load_config(args.v8_config)
        data526_config = read_json(args.data526_config)
        data526.verify_config(data526_config, require_terminal_v8=True)
        v8_report = read_json(args.v8_report)
        v8_survivors = read_json(args.v8_survivors)
        data526.validate_v8_inputs(v8_report, v8_survivors, data526_config)
        matcher, v8_inventory, v8_payloads = reconstruct_v8_source_inputs(
            v7_root=args.v7_root,
            bulk_workspace=args.bulk_workspace,
            v8_config=v8_config,
        )
        rada_rows, rada_raw = read_jsonl(args.rada_quality_privacy_jsonl)
        report, survivors = run_expanded_dedup(
            matcher_audit=matcher.audit_payloads,
            matcher_verify=matcher.verify_report,
            reconstructed_v8_inventory=v8_inventory,
            reconstructed_v8_payloads=v8_payloads,
            v8_survivor_authority=v8_survivors,
            data526_evidence=read_json(args.data526_evidence),
            data526_record_inventory=read_json(args.data526_record_inventory),
            rada_language_report=read_json(args.rada_language_report),
            rada_quality_privacy_report=read_json(args.rada_quality_privacy_report),
            expected_rada_report_sha256=args.expected_rada_report_sha256,
            rada_rows=rada_rows,
            rada_raw_jsonl=rada_raw,
        )
        write_json(args.output_report, report)
        write_json(args.output_survivors, survivors)
    except (ExpandedDedupError, data526.Data526V8Error, v8.V8Error, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print("D03_EXPANDED_GLOBAL_DEDUP_V9=PASS_ZERO_CREDIT")
    print("REPORT_SHA256=" + report["report_sha256"])
    print("SURVIVOR_AUTHORITY_SHA256=" + survivors["survivor_authority_sha256"])
    print("AUTHORIZED_TRAINING_EXPOSURE=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
