"""Execute DATA-108 calibration, frozen holdout, and complete current-corpus effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from run_document_quality import _assert_current_binding, _json, _rebuild_and_read_records
from twelve_six.data.document_quality import assess_document, default_quality_policy
from twelve_six.data.document_quality_real import (
    evaluate_labeled_rows,
    select_policy_on_calibration,
)

DATA21_ARTIFACT_SHA256 = "3b4ef6c0d42725f7b660935f70ef1f8b41b90eb9d5d73c83455401a434233122"
DATA21_MANIFEST_ID = "9d50c0baf98247c1babc5fca8dead5b1fa87264ad92ea62527c34e342a7dd735"
DATA21_REGISTRY_ID = "678d250ac9910f58ab1b9113cf713a2fea52a6a21e7a8434e6434d95a8045214"
EXPECTED_EXTERNAL_RECORDS = {
    "ext-ba861bf058ce23e02cc569d1a63de897": (
        "ua.rada.open-data.laws-texts",
        "72c301db0b2539f3f7a73c9c15e2e425700a6b758a1114f1a861e2d60c704c50",
        88565,
    ),
    "ext-34108f6cb4826107ff3b53be7b172eb0": (
        "en.standardebooks.manual",
        "154fb4034929714087e75150d678bf65049ddac32e79dcdf97162c8972c2be83",
        48002,
    ),
    "ext-b47c3572ee24c4641b86e91804ed04fe": (
        "en.standardebooks.manual",
        "94eb2f529922d125b3bd40691778886f4d5d80b128b925d0274fb3d94646ec5a",
        36791,
    ),
}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _load_external_artifact(artifact_zip: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if _sha_file(artifact_zip) != DATA21_ARTIFACT_SHA256:
        raise ValueError("DATA-21/22 artifact SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="data108-data21-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(artifact_zip) as archive:
            archive.extractall(root)
        evidence = root / "external-source-intake-evidence"
        manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("manifest_sha256") != DATA21_MANIFEST_ID:
            raise ValueError("DATA-21/22 manifest identity mismatch")
        if manifest.get("candidate_registry_identity_sha256") != DATA21_REGISTRY_ID:
            raise ValueError("DATA-21/22 source registry identity mismatch")
        if manifest.get("authority_boundary") != "REAL_BOUNDED_SAMPLE_NOT_CANONICAL_CORPUS_FREEZE_OR_SOURCE_SNAPSHOT_PROMOTION":
            raise ValueError("unexpected DATA-21/22 authority boundary")
        if manifest.get("record_counts", {}).get("accepted") != 3:
            raise ValueError("expected exactly three accepted external records")

        records: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(evidence / "records.jsonl"):
            record_id = str(row["id"])
            if record_id not in EXPECTED_EXTERNAL_RECORDS:
                raise ValueError(f"unexpected external record {record_id}")
            source_id, content_sha, byte_count = EXPECTED_EXTERNAL_RECORDS[record_id]
            if row.get("source_id") != source_id or row.get("content_sha256") != content_sha:
                raise ValueError(f"external identity mismatch for {record_id}")
            if int(row.get("normalized_utf8_bytes", -1)) != byte_count:
                raise ValueError(f"external byte-count mismatch for {record_id}")
            text = (evidence / str(row["text_path"])).read_text(encoding="utf-8")
            if text.endswith("\n"):
                text = text[:-1]
            encoded = text.encode("utf-8")
            if len(encoded) != byte_count or _sha_bytes(encoded) != content_sha:
                raise ValueError(f"external normalized text mismatch for {record_id}")
            records[record_id] = {**row, "text": text}
        if set(records) != set(EXPECTED_EXTERNAL_RECORDS):
            raise ValueError("external record set mismatch")
        return manifest, records


def _resolve_labels(
    rows: list[dict[str, Any]], external: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for row in rows:
        materialized = dict(row)
        if "source_record_id" in row:
            source_record = external.get(str(row["source_record_id"]))
            if source_record is None:
                raise ValueError(f"unknown source record for {row['id']}")
            if source_record["source_id"] != row.get("source_family"):
                raise ValueError(f"source-family mismatch for {row['id']}")
            nonempty = [line for line in source_record["text"].splitlines() if line.strip()]
            start = int(row["start_nonempty_line"])
            end = int(row["end_nonempty_line"])
            if start < 0 or end < start or end >= len(nonempty):
                raise ValueError(f"invalid source range for {row['id']}")
            text = "\n".join(nonempty[start : end + 1])
            materialized["text"] = text
        else:
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"project control {row['id']} requires inline text")
        resolved.append(materialized)
        evidence.append(
            {
                "id": row["id"],
                "source_family": row["source_family"],
                "source_record_id": row.get("source_record_id"),
                "mode": row["mode"],
                "label": row["label"],
                "label_rationale": row["label_rationale"],
                "start_nonempty_line": row.get("start_nonempty_line"),
                "end_nonempty_line": row.get("end_nonempty_line"),
                "sample_chars": len(text),
                "sample_utf8_bytes": len(text.encode("utf-8")),
                "sample_sha256": _sha_bytes(text.encode("utf-8")),
                "excerpt": " ".join(text.split())[:180],
            }
        )
    return resolved, evidence


def _assert_partitions(calibration: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> None:
    calibration_ids = {str(row["id"]) for row in calibration}
    holdout_ids = {str(row["id"]) for row in holdout}
    if len(calibration_ids) != len(calibration) or len(holdout_ids) != len(holdout):
        raise ValueError("duplicate labeled sample id")
    if calibration_ids & holdout_ids:
        raise ValueError("calibration and holdout ids overlap")
    calibration_ranges = {
        (row.get("source_record_id"), row.get("start_nonempty_line"), row.get("end_nonempty_line"))
        for row in calibration
        if row.get("source_record_id") is not None
    }
    holdout_ranges = {
        (row.get("source_record_id"), row.get("start_nonempty_line"), row.get("end_nonempty_line"))
        for row in holdout
        if row.get("source_record_id") is not None
    }
    if calibration_ranges & holdout_ranges:
        raise ValueError("calibration and holdout source ranges overlap exactly")


def _removed_effects(
    records: list[dict[str, Any]], source_family: dict[str, str], policy: Any
) -> dict[str, Any]:
    removed: list[dict[str, Any]] = []
    totals = {"documents": 0, "utf8_bytes": 0, "byte_tokens": 0}
    removed_totals = {"documents": 0, "utf8_bytes": 0, "byte_tokens": 0}
    by_family: dict[str, dict[str, int]] = {}
    by_mode: dict[str, dict[str, int]] = {}

    for row in records:
        record_id = str(row["id"])
        text = str(row["text"])
        mode = str(row["mode"])
        byte_count = int(row.get("byte_tokens", len(text.encode("utf-8"))))
        utf8_bytes = len(text.encode("utf-8"))
        family = source_family[record_id]
        totals["documents"] += 1
        totals["utf8_bytes"] += utf8_bytes
        totals["byte_tokens"] += byte_count
        family_bucket = by_family.setdefault(family, {"input_documents": 0, "removed_documents": 0, "removed_byte_tokens": 0})
        mode_bucket = by_mode.setdefault(mode, {"input_documents": 0, "removed_documents": 0, "removed_byte_tokens": 0})
        family_bucket["input_documents"] += 1
        mode_bucket["input_documents"] += 1
        decision = assess_document(record_id, text, mode, policy=policy)
        if decision.accepted:
            continue
        removed_totals["documents"] += 1
        removed_totals["utf8_bytes"] += utf8_bytes
        removed_totals["byte_tokens"] += byte_count
        family_bucket["removed_documents"] += 1
        family_bucket["removed_byte_tokens"] += byte_count
        mode_bucket["removed_documents"] += 1
        mode_bucket["removed_byte_tokens"] += byte_count
        removed.append(
            {
                "id": record_id,
                "source_family": family,
                "mode": mode,
                "utf8_bytes": utf8_bytes,
                "byte_tokens": byte_count,
                "reasons": list(decision.reasons),
            }
        )

    byte_fraction = removed_totals["byte_tokens"] / totals["byte_tokens"] if totals["byte_tokens"] else 0.0
    document_fraction = removed_totals["documents"] / totals["documents"] if totals["documents"] else 0.0
    material = byte_fraction >= 0.005 or document_fraction >= 0.01
    return {
        "input": totals,
        "removed": removed_totals,
        "removed_byte_token_fraction": round(byte_fraction, 9),
        "removed_document_fraction": round(document_fraction, 9),
        "materiality_rule": "A/B required when removed byte-token fraction >= 0.5% OR removed document fraction >= 1.0%",
        "ab_training_required": material,
        "by_source_family": dict(sorted(by_family.items())),
        "by_mode": dict(sorted(by_mode.items())),
        "removed_documents": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--data21-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/d03/data108"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    external_manifest, external = _load_external_artifact(args.data21_artifact.resolve())
    calibration_spec = _read_jsonl(repo / "data/quality/calibration_real_sources_v1.jsonl")
    holdout_spec = _read_jsonl(repo / "data/quality/holdout_real_sources_v1.jsonl")
    _assert_partitions(calibration_spec, holdout_spec)
    calibration, calibration_evidence = _resolve_labels(calibration_spec, external)
    holdout, holdout_evidence = _resolve_labels(holdout_spec, external)

    selected, candidate_reports = select_policy_on_calibration(calibration)
    incumbent = default_quality_policy()
    calibration_selected = evaluate_labeled_rows(calibration, selected)
    holdout_selected = evaluate_labeled_rows(holdout, selected)
    holdout_incumbent = evaluate_labeled_rows(holdout, incumbent)

    view = _json(repo / "configs/data/document_quality_current_corpus_v1.json")
    retained = _json(repo / str(view["corpus_manifest_path"]))
    _assert_current_binding(view, retained)
    project_records, rebuild_evidence = _rebuild_and_read_records(repo, view, retained)
    source_family: dict[str, str] = {str(row["id"]): "PROJECT_AUTHORED_DATA25" for row in project_records}
    complete_records: list[dict[str, Any]] = [
        {**row, "byte_tokens": len(str(row["text"]).encode("utf-8"))} for row in project_records
    ]
    for record_id, record in sorted(external.items()):
        text = str(record["text"])
        complete_records.append(
            {
                "id": record_id,
                "mode": record["language"],
                "text": text,
                "byte_tokens": int(record["normalized_utf8_bytes"]),
            }
        )
        source_family[record_id] = str(record["source_id"])

    effects = _removed_effects(complete_records, source_family, selected)
    corpus_binding = {
        "data25_corpus_identity_sha256": retained["corpus_identity_sha256"],
        "data21_manifest_identity_sha256": DATA21_MANIFEST_ID,
        "data21_artifact_sha256": DATA21_ARTIFACT_SHA256,
        "data21_registry_identity_sha256": DATA21_REGISTRY_ID,
        "external_record_content_sha256": {
            record_id: record["content_sha256"] for record_id, record in sorted(external.items())
        },
    }
    corpus_binding_sha256 = _sha_bytes(_canonical(corpus_binding))

    calibration_report = {
        "schema_version": "12-6.data108-calibration-report.v1",
        "partition": "CALIBRATION_THRESHOLD_SELECTION_ONLY",
        "calibration_spec_sha256": _sha_file(repo / "data/quality/calibration_real_sources_v1.jsonl"),
        "selected_policy": selected.manifest(),
        "candidate_reports": candidate_reports,
        "selected_report": calibration_selected,
        "resolved_labeled_evidence": calibration_evidence,
        "external_evidence_authority": external_manifest["authority_boundary"],
    }
    holdout_report = {
        "schema_version": "12-6.data108-holdout-report.v1",
        "partition": "FROZEN_HOLDOUT_NOT_USED_FOR_THRESHOLD_SELECTION",
        "holdout_spec_sha256": _sha_file(repo / "data/quality/holdout_real_sources_v1.jsonl"),
        "selected_policy_sha256": selected.manifest()["policy_sha256"],
        "selected_policy_report": holdout_selected,
        "incumbent_policy_report": holdout_incumbent,
        "resolved_labeled_evidence": holdout_evidence,
    }
    effects_report = {
        "schema_version": "12-6.data108-complete-current-corpus-effects.v1",
        "selected_policy": selected.manifest(),
        "corpus_binding": corpus_binding,
        "corpus_binding_sha256": corpus_binding_sha256,
        "corpus_scope": "DATA25_V0.1_PROJECT_CORPUS_PLUS_EXACT_TERMINAL_SUCCESS_DATA21_22_BOUNDED_REAL_INTAKE",
        "external_admitted_source_families": [
            "ua.rada.open-data.laws-texts",
            "en.standardebooks.manual",
        ],
        "external_real_code_family_status": "NONE_ADMITTED; code-like examples occur inside admitted Standard Ebooks manual",
        "project_rebuild": rebuild_evidence,
        **effects,
    }
    recommendation = {
        "schema_version": "12-6.data108-recommendation.v1",
        "selected_policy_id": selected.policy_id,
        "selected_policy_sha256": selected.manifest()["policy_sha256"],
        "holdout_false_accepts": holdout_selected["overall"]["false_accepts"],
        "holdout_false_rejects": holdout_selected["overall"]["false_rejects"],
        "ab_training_required": effects["ab_training_required"],
        "ab_status": "REQUIRED_NOT_EXECUTED" if effects["ab_training_required"] else "NOT_REQUIRED_NO_MATERIAL_COMPOSITION_CHANGE",
        "quality_definition": "manual labels plus deterministic interpretable features; model loss is not used to define quality",
        "truth_boundary": (
            "Policy is calibrated only for the currently admitted bounded Rada and Standard Ebooks families plus explicit controls. "
            "It is not a universal web-quality classifier and grants no rights, PII, copyright, language-admission, deduplication, or contamination authority."
        ),
    }

    outputs = {
        "calibration.json": calibration_report,
        "holdout.json": holdout_report,
        "complete_current_corpus_effects.json": effects_report,
        "recommendation.json": recommendation,
    }
    for name, value in outputs.items():
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "selected_policy_id": selected.policy_id,
        "selected_policy_sha256": selected.manifest()["policy_sha256"],
        "calibration": calibration_selected["overall"],
        "holdout": holdout_selected["overall"],
        "complete_current_corpus": effects,
        "ab_status": recommendation["ab_status"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
