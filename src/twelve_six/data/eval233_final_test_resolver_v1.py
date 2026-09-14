"""Resolve the terminal EVAL-233 final-test bytes for DATA-232 matching only.

The immutable RECOVER-174 gzip seed and source authority are selectively carried by
exact Git blob identity from terminal EVAL-233. This resolver never exposes outcomes,
metrics, model scores, or selection signals. It verifies the historical final-test set
identity before returning ephemeral matcher rows plus text-free membership metadata.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

EVAL233_HEAD = "b5512b4648cb09dd052b08884dc53f291e1ce935"
EVAL233_SCOPED_RUN = 32957254139
EVAL233_ARTIFACT_ID = 9602456151
EVAL233_ARTIFACT_DIGEST = (
    "sha256:c4631d10bbf373878eec3b47578cb61cd89c5b006f315be4ed6750c6db5ff3c2"
)
SEED_PATH = Path("data/evaluation/recover174_real_holdout_seed.jsonl.gz")
AUTHORITY_PATH = Path("configs/evaluation/recover174_source_authority_v1.json")
SEED_GIT_BLOB_SHA1 = "4bfbfbf29fa9538cabda6068efd3a1fd036a9479"
AUTHORITY_GIT_BLOB_SHA1 = "3ba9f221a82468f971c17eda518cd6f1642fd311"
SEED_SHA256 = "7e6827d22d573dda4c9ff2b5f0ab2b8fe3fdf5aa577970ecb77e94a10cf72367"
SOURCE_AUTHORITY_ID = "c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f"
EVAL233_FINAL_SET_ID = "6b012efc4d627b113b8adc2166e6ab50d9001284083f7a429c665b7752ca18d7"
DATA232_FINAL_TEST_ID = "86d51eb106524cd8e4d0f94d4ff6e2e3426c6321e0698279877dfc4d5fce3116"
NORMALIZATION_POLICY = "PRESERVE_RECOVER174_BYTES_NO_RENORMALIZATION"
SET_SCHEMA = "12-6.eval233-real-holdout-set.v2"
EXPECTED_DOCUMENTS = 16
EXPECTED_MODALITY_COUNTS = {"ua": 8, "en": 8}


class FinalTestResolverError(RuntimeError):
    """Fail-closed EVAL-233 final-test payload-resolution error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalTestResolverError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _read_authority(repo_root: Path) -> tuple[bytes, dict[str, Any]]:
    path = repo_root / AUTHORITY_PATH
    _require(path.is_file(), "RECOVER-174 source authority is missing")
    raw = path.read_bytes()
    _require(
        _git_blob_sha1(raw) == AUTHORITY_GIT_BLOB_SHA1,
        "RECOVER-174 source-authority Git blob identity drift",
    )
    try:
        authority = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalTestResolverError("RECOVER-174 source authority is invalid JSON") from exc
    _require(isinstance(authority, dict), "RECOVER-174 source authority must be an object")
    _require(
        authority.get("authority_identity_sha256") == SOURCE_AUTHORITY_ID,
        "RECOVER-174 semantic authority identity drift",
    )
    _require(
        authority.get("admitted_modalities") == ["ua", "en"],
        "RECOVER-174 admitted modality boundary drift",
    )
    code = authority.get("blocked_modalities", {}).get("code", {})
    _require(
        code.get("status") == "BLOCKED_NO_EVALUATION_USE_AUTHORITY",
        "RECOVER-174 code evaluation-rights boundary drift",
    )
    return raw, authority


def _read_seed(repo_root: Path) -> tuple[bytes, list[tuple[bytes, dict[str, Any]]]]:
    path = repo_root / SEED_PATH
    _require(path.is_file(), "RECOVER-174 final-test seed is missing")
    seed = path.read_bytes()
    _require(
        _git_blob_sha1(seed) == SEED_GIT_BLOB_SHA1,
        "RECOVER-174 final-test seed Git blob identity drift",
    )
    _require(_sha256(seed) == SEED_SHA256, "RECOVER-174 final-test seed SHA-256 drift")
    try:
        decompressed = gzip.decompress(seed)
    except (OSError, EOFError) as exc:
        raise FinalTestResolverError("RECOVER-174 final-test seed gzip is invalid") from exc
    raw_lines = decompressed.splitlines(keepends=True)
    _require(len(raw_lines) == EXPECTED_DOCUMENTS, "RECOVER-174 final-test row count drift")
    rows: list[tuple[bytes, dict[str, Any]]] = []
    for raw_line in raw_lines:
        _require(raw_line.endswith(b"\n"), "RECOVER-174 final-test row lost LF terminator")
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FinalTestResolverError("RECOVER-174 final-test row is invalid JSON") from exc
        _require(isinstance(row, dict), "RECOVER-174 final-test row must be an object")
        rows.append((raw_line, row))
    return seed, rows


def _validate_rows(
    rows: list[tuple[bytes, dict[str, Any]]],
    authority: dict[str, Any],
) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    counts = {"ua": 0, "en": 0}
    validated: list[dict[str, Any]] = []
    for raw_line, row in rows:
        required = (
            "record_id",
            "modality",
            "source_id",
            "source_family",
            "source_version",
            "source_snapshot_sha256",
            "source_kind",
            "evaluation_use_authority_ref",
            "provenance_ref",
            "text",
        )
        missing = [key for key in required if key not in row]
        _require(not missing, f"RECOVER-174 final-test row missing fields: {missing}")
        record_id = row["record_id"]
        modality = row["modality"]
        source_id = row["source_id"]
        source_family = row["source_family"]
        text = row["text"]
        for label, value in (
            ("record_id", record_id),
            ("modality", modality),
            ("source_id", source_id),
            ("source_family", source_family),
            ("text", text),
        ):
            _require(isinstance(value, str) and bool(value), f"invalid final-test {label}")
        _require(modality in counts, f"unsupported final-test modality: {modality}")
        _require(row["source_kind"] == "EXTERNAL_REAL", "final-test source is not external-real")
        source = authority.get("sources", {}).get(source_id)
        _require(isinstance(source, dict), f"final-test source lacks authority: {source_id}")
        _require(
            source.get("evaluation_status") == "APPROVED_FOR_HELDOUT_EVALUATION",
            f"final-test source lacks evaluation approval: {source_id}",
        )
        _require(
            row["source_snapshot_sha256"]
            in source.get("admitted_source_snapshots_sha256", []),
            f"final-test source snapshot is not admitted: {source_id}",
        )
        text_bytes = text.encode("utf-8")
        content_sha = _sha256(text_bytes)
        if "content_sha256" in row:
            _require(row["content_sha256"] == content_sha, "final-test content hash drift")
        if "source_bytes" in row:
            _require(row["source_bytes"] == len(text_bytes), "final-test content byte drift")
        _require(record_id not in seen_ids, "duplicate final-test record_id")
        _require(content_sha not in seen_content, "duplicate exact final-test content")
        seen_ids.add(record_id)
        seen_content.add(content_sha)
        counts[modality] += 1
        validated.append(
            {
                "raw_line": raw_line,
                "row": row,
                "source": source,
                "content_sha256": content_sha,
                "text_utf8_bytes": len(text_bytes),
            }
        )
    _require(counts == EXPECTED_MODALITY_COUNTS, f"final-test modality counts drift: {counts}")
    return validated


def _historical_record_binding(item: dict[str, Any]) -> dict[str, Any]:
    row = item["row"]
    source = item["source"]
    raw_line = item["raw_line"]
    return {
        "record_id": str(row["record_id"]),
        "modality": str(row["modality"]),
        "source_id": str(row["source_id"]),
        "source_family": str(row["source_family"]),
        "source_version": str(row["source_version"]),
        "purpose": "final-test",
        "raw_source_sha256": list(source.get("raw_sha256", [])),
        "extracted_normalized_snapshot_sha256": str(row["source_snapshot_sha256"]),
        "upstream_source_identity_sha256": str(source.get("source_identity_sha256", "")),
        "evaluation_use_authority_ref": str(row["evaluation_use_authority_ref"]),
        "provenance_ref": str(row["provenance_ref"]),
        "normalization_policy": NORMALIZATION_POLICY,
        "content_sha256": item["content_sha256"],
        "source_jsonl_row_bytes_sha256": _sha256(raw_line),
        "source_jsonl_row_bytes": len(raw_line),
    }


def _verify_historical_final_set(
    seed: bytes,
    validated: list[dict[str, Any]],
) -> None:
    unsigned = {
        "schema_version": SET_SCHEMA,
        "worker_id": "EVAL-233-REAL-HOLDOUT-V2",
        "purpose": "final-test",
        "status": "IMMUTABLE_RESERVED_FINAL_TEST",
        "modalities": ["ua", "en"],
        "documents": EXPECTED_DOCUMENTS,
        "modality_documents": EXPECTED_MODALITY_COUNTS,
        "files": {
            "recover174_seed": {
                "path": "final-test/recover174_real_holdout_seed.jsonl.gz",
                "bytes": len(seed),
                "sha256": _sha256(seed),
                "git_blob_sha1": _git_blob_sha1(seed),
            }
        },
        "records": [_historical_record_binding(item) for item in validated],
        "selection_eligible": False,
        "tokenizer_fit_eligible": False,
        "hyperparameter_selection_eligible": False,
        "final_test_exposure_prohibited": True,
        "normalization_policy": NORMALIZATION_POLICY,
        "immutable": True,
    }
    observed = _sha256(_canonical_bytes(unsigned))
    _require(observed == EVAL233_FINAL_SET_ID, "terminal EVAL-233 final-test set identity drift")


def resolve_eval233_final_test(
    repo_root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Return ephemeral DATA-232 rows, reserved-set metadata, and text-free evidence."""
    _, authority = _read_authority(repo_root)
    seed, rows = _read_seed(repo_root)
    validated = _validate_rows(rows, authority)
    _verify_historical_final_set(seed, validated)

    matcher_rows = [
        {
            "record_id": str(item["row"]["record_id"]),
            "source_id": str(item["row"]["source_id"]),
            "source_family": str(item["row"]["source_family"]),
            "modality": str(item["row"]["modality"]),
            "text": str(item["row"]["text"]),
        }
        for item in validated
    ]
    matcher_rows.sort(key=lambda row: row["record_id"])
    members = [
        {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "source_family": row["source_family"],
            "modality": row["modality"],
            "content_sha256": _sha256(row["text"].encode("utf-8")),
            "utf8_bytes": len(row["text"].encode("utf-8")),
            "training_prohibited": True,
            "outcomes_included": False,
        }
        for row in matcher_rows
    ]
    reserved_set: dict[str, Any] = {
        "authority_id": "eval233-final-test",
        "identity_sha256": DATA232_FINAL_TEST_ID,
        "role": "final_test",
        "source_sha": EVAL233_HEAD,
        "source_membership_identity_sha256": EVAL233_FINAL_SET_ID,
        "members": members,
    }
    projection = [
        {
            "record_id_sha256": _sha256(row["record_id"].encode("utf-8")),
            "source_id_sha256": _sha256(row["source_id"].encode("utf-8")),
            "source_family_sha256": _sha256(row["source_family"].encode("utf-8")),
            "modality": row["modality"],
            "content_sha256": member["content_sha256"],
            "utf8_bytes": member["utf8_bytes"],
        }
        for row, member in zip(matcher_rows, members, strict=True)
    ]
    evidence_core: dict[str, Any] = {
        "schema_version": "12-6.eval233-final-test-resolver.v1",
        "source_head_sha": EVAL233_HEAD,
        "source_scoped_run": EVAL233_SCOPED_RUN,
        "source_artifact_id": EVAL233_ARTIFACT_ID,
        "source_artifact_digest": EVAL233_ARTIFACT_DIGEST,
        "seed_git_blob_sha1": SEED_GIT_BLOB_SHA1,
        "seed_sha256": SEED_SHA256,
        "source_authority_git_blob_sha1": AUTHORITY_GIT_BLOB_SHA1,
        "source_authority_identity_sha256": SOURCE_AUTHORITY_ID,
        "source_final_set_identity_sha256": EVAL233_FINAL_SET_ID,
        "data232_final_test_identity_sha256": DATA232_FINAL_TEST_ID,
        "documents": len(matcher_rows),
        "modality_documents": EXPECTED_MODALITY_COUNTS,
        "member_projection": projection,
        "member_projection_sha256": _sha256(_canonical_bytes(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed_for_decontamination": True,
        "final_test_outcomes_read": False,
        "selection_or_hyperparameter_use": False,
        "training_executed": False,
        "authorized_training_exposure": 0,
    }
    evidence_core["resolver_identity_sha256"] = _sha256(_canonical_bytes(evidence_core))
    return matcher_rows, reserved_set, evidence_core
