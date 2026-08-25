"""Run DATA-32 project-owned calibration and current-corpus quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.data.document_quality import (
    default_quality_policy,
    evaluate_calibration,
    run_quality_filter,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _current_records(
    repo: Path, view_path: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    view = json.loads(view_path.read_text(encoding="utf-8"))
    source = repo / view["source_path"]
    if not source.is_file():
        raise ValueError(f"current corpus source is missing: {source}")
    if source.stat().st_size != view["source_bytes"]:
        raise ValueError("current corpus source size drift")
    if _sha256(source) != view["source_sha256"]:
        raise ValueError("current corpus source SHA-256 drift")
    lines = source.read_text(encoding="utf-8").splitlines()
    records = []
    seen = set()
    for item in view["records"]:
        record_id = item["id"]
        if record_id in seen:
            raise ValueError(f"duplicate current-view record id: {record_id}")
        seen.add(record_id)
        start = int(item["line_start"])
        end = int(item["line_end"])
        if start <= 0 or end < start or end > len(lines):
            raise ValueError(f"invalid current-view line range for {record_id}")
        records.append(
            {
                "id": record_id,
                "mode": item["mode"],
                "text": "\n".join(lines[start - 1 : end]),
            }
        )
    return records, view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/quality/calibration_uk_en_code_v1.jsonl"),
    )
    parser.add_argument(
        "--current-view",
        type=Path,
        default=Path("configs/data/document_quality_current_corpus_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/d03"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    calibration_path = repo / args.calibration
    current_view_path = repo / args.current_view
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = default_quality_policy()
    policy_manifest = policy.manifest()
    calibration_rows = _jsonl(calibration_path)
    calibration = evaluate_calibration(
        calibration_rows,
        calibration_manifest_sha256=_sha256(calibration_path),
        policy=policy,
    )
    calibration_edges = run_quality_filter(
        calibration_rows,
        input_manifest_sha256=_sha256(calibration_path),
        policy=policy,
        edge_samples_per_class=3,
    )["edge_samples"]
    calibration_report = {
        **calibration,
        "policy_manifest": policy_manifest,
        "edge_samples": calibration_edges,
        "manual_review_instruction": (
            "Inspect the selected excerpts; machine selection is deterministic by "
            "absolute threshold margin, then record id."
        ),
    }

    records, current_view = _current_records(repo, current_view_path)
    current = run_quality_filter(
        records,
        input_manifest_sha256=_sha256(current_view_path),
        policy=policy,
        edge_samples_per_class=3,
    )
    current_report = {
        **current,
        "policy_manifest": policy_manifest,
        "current_view_authority": current_view["authority"],
        "current_view_sha256": _sha256(current_view_path),
        "source_path": current_view["source_path"],
        "source_sha256": current_view["source_sha256"],
        "source_bytes": current_view["source_bytes"],
        "representative_100k_1m_corpus_executed": False,
        "truth_boundary": (
            "At this source head the current committed UK/EN/code data is a "
            "project-authored mechanics corpus, not the requested representative "
            "100K-1M corpus. These counts are current-corpus mechanics evidence only."
        ),
    }

    calibration_out = output_dir / "document_quality_calibration_20260825.json"
    current_out = output_dir / "document_quality_current_corpus_20260825.json"
    calibration_out.write_text(
        json.dumps(calibration_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    current_out.write_text(
        json.dumps(current_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"calibration": calibration, "current": current}, sort_keys=True))


if __name__ == "__main__":
    main()
