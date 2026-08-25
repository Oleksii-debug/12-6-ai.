"""Run DATA-32 calibration and the exact current DATA-25 corpus V0.1 quality pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.data.document_quality import (
    default_quality_policy,
    evaluate_calibration,
    run_quality_filter,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


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


def _manifest_totals(manifest: dict[str, Any]) -> tuple[int, int]:
    by_stratum = manifest.get("by_stratum")
    if not isinstance(by_stratum, dict):
        raise ValueError("corpus manifest is missing by_stratum")
    documents = sum(int(item["documents"]) for item in by_stratum.values())
    byte_tokens = sum(int(item["byte_tokens"]) for item in by_stratum.values())
    return documents, byte_tokens


def _assert_current_binding(view: dict[str, Any], manifest: dict[str, Any]) -> None:
    identity = manifest.get("corpus_identity_sha256")
    if identity != view.get("corpus_identity_sha256"):
        raise ValueError("DATA-25 corpus identity drift; fresh quality pass required")
    documents, byte_tokens = _manifest_totals(manifest)
    if documents != int(view.get("expected_documents", -1)):
        raise ValueError("DATA-25 document count drift; fresh quality binding required")
    if byte_tokens != int(view.get("expected_byte_tokens", -1)):
        raise ValueError("DATA-25 byte-token count drift; fresh quality binding required")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != int(view.get("expected_shards", -1)):
        raise ValueError("DATA-25 shard count drift; fresh quality binding required")
    if manifest.get("external_training_eligible_sources") != view.get(
        "external_training_eligible_sources"
    ):
        raise ValueError("DATA-25 external-source eligibility count drift")


def _rebuild_and_read_records(
    repo: Path, view: dict[str, Any], retained: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    config_path = repo / str(view["builder_config_path"])
    retained_shards = [
        (item["path"], item["sha256"], item["size_bytes"], item["documents"])
        for item in retained["shards"]
    ]

    with tempfile.TemporaryDirectory(prefix="data32-corpus-v01-") as tmp:
        output_dir = Path(tmp) / "corpus"
        rebuilt = build_corpus(config_path, output_dir)
        if rebuilt["corpus_identity_sha256"] != retained["corpus_identity_sha256"]:
            raise ValueError("DATA-25 rebuild identity differs from retained manifest")
        rebuilt_shards = [
            (item["path"], item["sha256"], item["size_bytes"], item["documents"])
            for item in rebuilt["shards"]
        ]
        if rebuilt_shards != retained_shards:
            raise ValueError("DATA-25 rebuilt shard identities differ from retained manifest")

        records: list[dict[str, str]] = []
        verified_bytes = 0
        for shard in rebuilt["shards"]:
            path = output_dir / shard["path"]
            if _sha256(path) != shard["sha256"]:
                raise ValueError(f"rebuilt shard hash mismatch: {shard['path']}")
            if path.stat().st_size != shard["size_bytes"]:
                raise ValueError(f"rebuilt shard size mismatch: {shard['path']}")
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"{shard['path']}:{line_number}: expected JSON object"
                        )
                    mode = row.get("stratum")
                    if mode not in {"uk", "en", "code"}:
                        raise ValueError(f"unsupported corpus stratum: {mode!r}")
                    text = row.get("text")
                    record_id = row.get("record_id")
                    if not isinstance(text, str) or not isinstance(record_id, str):
                        raise ValueError("corpus row requires record_id and text strings")
                    records.append({"id": record_id, "mode": mode, "text": text})
                    verified_bytes += int(row["byte_tokens"])

        expected_documents, expected_bytes = _manifest_totals(retained)
        if len(records) != expected_documents or verified_bytes != expected_bytes:
            raise ValueError("rebuilt corpus totals differ from retained manifest")

        rebuild_evidence = {
            "rebuild_verified": True,
            "ordered_shard_hashes_match": True,
            "reproduced_shards": len(rebuilt["shards"]),
            "reproduced_documents": len(records),
            "reproduced_byte_tokens": verified_bytes,
            "builder_sha256": rebuilt["builder_sha256"],
            "config_sha256": rebuilt["config_sha256"],
        }
        return records, rebuild_evidence


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

    view = _json(current_view_path)
    manifest_path = repo / str(view["corpus_manifest_path"])
    retained = _json(manifest_path)
    _assert_current_binding(view, retained)
    records, rebuild_evidence = _rebuild_and_read_records(repo, view, retained)
    current = run_quality_filter(
        records,
        input_manifest_sha256=retained["corpus_identity_sha256"],
        policy=policy,
        edge_samples_per_class=5,
    )
    current_report = {
        **current,
        "policy_manifest": policy_manifest,
        "data25_head_sha": view["data25_head_sha"],
        "corpus_identity_sha256": retained["corpus_identity_sha256"],
        "corpus_version": retained["corpus_version"],
        "external_training_eligible_sources": retained["external_training_eligible_sources"],
        "truth_boundary": retained["truth_boundary"],
        "representative_100k_1m_corpus_executed": True,
        **rebuild_evidence,
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
