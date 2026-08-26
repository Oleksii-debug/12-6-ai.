"""Fail-closed materialization for Research Corpus V1 acquisition candidates.

This module turns already-acquired, hash-pinned source bytes into a deterministic
candidate tree.  It deliberately does not download sources, grant rights, credit
corpus capacity, fit a tokenizer, or authorize training.  Source-specific workers
must acquire and rights-review bytes first; downstream corpus gates must still run
on the materialized candidate tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .corpus_acquisition_plan import STRATA, validate_acquisition_plan

SCHEMA = "12-6.research-corpus-v1-candidate-materialization.v1"
MAX_OBJECT_BYTES = 64 * 1024 * 1024
_OBJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")
NORMALIZERS = {"utf8_identity_v1", "utf8_lf_v1", "utf8_nfkc_lf_v1"}


class CandidateMaterializationError(RuntimeError):
    """Raised when candidate bytes or authority metadata fail closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CandidateMaterializationError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CandidateMaterializationError(f"{field} must be a positive integer")
    return value


def _require_source_locator(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateMaterializationError(f"{field} must be non-empty")
    if any(ord(char) < 32 for char in value):
        raise CandidateMaterializationError(f"{field} contains a control character")
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise CandidateMaterializationError(f"{field} must not contain credentials")
    return value


def _normalize(raw: bytes, normalizer: str) -> bytes:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CandidateMaterializationError("candidate object is not strict UTF-8") from exc
    if normalizer == "utf8_identity_v1":
        return text.encode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalizer == "utf8_nfkc_lf_v1":
        text = unicodedata.normalize("NFKC", text)
    elif normalizer != "utf8_lf_v1":
        raise CandidateMaterializationError(f"unsupported normalizer: {normalizer}")
    return text.encode("utf-8")


def _safe_raw_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise CandidateMaterializationError("raw_path must be a non-traversing relative POSIX path")
    candidate = root.joinpath(*path.parts)
    if candidate.is_symlink():
        raise CandidateMaterializationError(f"raw_path must not be a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CandidateMaterializationError(f"raw_path does not exist: {relative}") from exc
    if not resolved.is_relative_to(root):
        raise CandidateMaterializationError(f"raw_path escapes raw root: {relative}")
    if not resolved.is_file():
        raise CandidateMaterializationError(f"raw_path must identify a regular file: {relative}")
    return resolved


def _candidate_plan_rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    validate_acquisition_plan(plan)
    rows = plan.get("candidate_streams")
    assert isinstance(rows, list)
    return {str(row["candidate_id"]): row for row in rows if isinstance(row, Mapping)}


def validate_materialization_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Validate an exact candidate-materialization request against its acquisition plan."""

    _require_sha256(plan_sha256, field="plan_sha256")
    if manifest.get("schema_version") != SCHEMA:
        raise CandidateMaterializationError("unsupported materialization schema")
    if manifest.get("acquisition_plan_sha256") != plan_sha256:
        raise CandidateMaterializationError("materialization manifest does not bind the exact plan")
    if manifest.get("local_free_only") is not True:
        raise CandidateMaterializationError("candidate materialization must be LOCAL_FREE only")
    if manifest.get("training_authorized") is not False:
        raise CandidateMaterializationError("candidate materialization cannot authorize training")
    if manifest.get("credited_capacity_bytes") != 0:
        raise CandidateMaterializationError("candidate materialization cannot credit corpus capacity")
    if manifest.get("authorized_loss_positions") != 0:
        raise CandidateMaterializationError("candidate materialization cannot authorize loss positions")

    plan_rows = _candidate_plan_rows(plan)
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise CandidateMaterializationError("objects must be a non-empty list")

    validated: list[dict[str, Any]] = []
    object_ids: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(objects):
        if not isinstance(raw_row, Mapping):
            raise CandidateMaterializationError(f"objects[{index}] must be an object")
        object_id = raw_row.get("object_id")
        if not isinstance(object_id, str) or _OBJECT_ID_RE.fullmatch(object_id) is None:
            raise CandidateMaterializationError(f"objects[{index}].object_id is invalid")
        if object_id in object_ids:
            raise CandidateMaterializationError(f"duplicate object_id: {object_id}")
        object_ids.add(object_id)

        candidate_id = raw_row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in plan_rows:
            raise CandidateMaterializationError(f"{object_id}: candidate_id is not in the bound plan")
        plan_row = plan_rows[candidate_id]
        stratum = raw_row.get("stratum")
        family = raw_row.get("family_candidate")
        if stratum not in STRATA or stratum != plan_row.get("stratum"):
            raise CandidateMaterializationError(f"{object_id}: stratum does not match the plan")
        if not isinstance(family, str) or family != plan_row.get("family_candidate"):
            raise CandidateMaterializationError(f"{object_id}: family does not match the plan")

        revision = raw_row.get("source_revision")
        if not isinstance(revision, str) or _SOURCE_REVISION_RE.fullmatch(revision) is None:
            raise CandidateMaterializationError(f"{object_id}: source_revision is invalid")
        locator = _require_source_locator(raw_row.get("source_locator"), field=f"{object_id}.source_locator")
        source_key = (locator, revision)
        if source_key in source_keys:
            raise CandidateMaterializationError(f"duplicate source locator/revision: {object_id}")
        source_keys.add(source_key)

        raw_path = raw_row.get("raw_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise CandidateMaterializationError(f"{object_id}: raw_path is required")
        expected_raw_sha256 = _require_sha256(
            raw_row.get("expected_raw_sha256"), field=f"{object_id}.expected_raw_sha256"
        )
        rights_evidence_sha256 = _require_sha256(
            raw_row.get("rights_evidence_sha256"), field=f"{object_id}.rights_evidence_sha256"
        )
        max_raw_bytes = _require_positive_int(
            raw_row.get("max_raw_bytes"), field=f"{object_id}.max_raw_bytes"
        )
        if max_raw_bytes > MAX_OBJECT_BYTES:
            raise CandidateMaterializationError(
                f"{object_id}: max_raw_bytes exceeds the {MAX_OBJECT_BYTES}-byte safety ceiling"
            )
        normalizer = raw_row.get("normalizer")
        if normalizer not in NORMALIZERS:
            raise CandidateMaterializationError(f"{object_id}: unsupported normalizer")

        validated.append(
            {
                "object_id": object_id,
                "candidate_id": candidate_id,
                "stratum": stratum,
                "family_candidate": family,
                "source_revision": revision,
                "source_locator": locator,
                "raw_path": raw_path,
                "expected_raw_sha256": expected_raw_sha256,
                "rights_evidence_sha256": rights_evidence_sha256,
                "max_raw_bytes": max_raw_bytes,
                "normalizer": normalizer,
            }
        )
    return tuple(sorted(validated, key=lambda row: row["object_id"]))


def materialize_candidate(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    raw_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build a deterministic candidate tree while keeping training authority at zero."""

    validated = validate_materialization_manifest(
        manifest,
        plan=plan,
        plan_sha256=plan_sha256,
    )
    try:
        raw_root_path = Path(raw_root).resolve(strict=True)
    except FileNotFoundError as exc:
        raise CandidateMaterializationError("raw_root does not exist") from exc
    if not raw_root_path.is_dir():
        raise CandidateMaterializationError("raw_root must be a directory")

    output_root_path = Path(output_root)
    if output_root_path.exists():
        raise CandidateMaterializationError("output_root must not already exist")
    output_parent = output_root_path.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".{output_root_path.name}.staging"
    if staging.exists():
        raise CandidateMaterializationError("staging path already exists")

    prepared: list[tuple[dict[str, Any], bytes, dict[str, Any]]] = []
    for row in validated:
        source_path = _safe_raw_path(raw_root_path, str(row["raw_path"]))
        size = source_path.stat().st_size
        if size > int(row["max_raw_bytes"]):
            raise CandidateMaterializationError(
                f"{row['object_id']}: raw object exceeds declared max_raw_bytes"
            )
        raw = source_path.read_bytes()
        raw_sha256 = _sha256_bytes(raw)
        if raw_sha256 != row["expected_raw_sha256"]:
            raise CandidateMaterializationError(f"{row['object_id']}: raw SHA-256 mismatch")
        normalized = _normalize(raw, str(row["normalizer"]))
        normalized_sha256 = _sha256_bytes(normalized)
        record = {
            "object_id": row["object_id"],
            "candidate_id": row["candidate_id"],
            "stratum": row["stratum"],
            "family_candidate": row["family_candidate"],
            "source_revision": row["source_revision"],
            "source_locator_sha256": _sha256_bytes(str(row["source_locator"]).encode("utf-8")),
            "rights_evidence_sha256": row["rights_evidence_sha256"],
            "normalizer": row["normalizer"],
            "raw_sha256": raw_sha256,
            "raw_bytes": len(raw),
            "normalized_sha256": normalized_sha256,
            "normalized_bytes": len(normalized),
            "payload_path": f"objects/{row['object_id']}.utf8",
            "capacity_credited": False,
            "training_eligible": False,
            "weight_update_eligible": False,
        }
        prepared.append((row, normalized, record))

    records = [record for _, _, record in prepared]
    materialized_bytes = sum(int(record["normalized_bytes"]) for record in records)
    identity_payload = {
        "schema_version": SCHEMA,
        "acquisition_plan_sha256": plan_sha256,
        "records": records,
    }
    inventory_identity = _sha256_bytes(_canonical_json_bytes(identity_payload))
    receipt = {
        **identity_payload,
        "inventory_identity_sha256": inventory_identity,
        "candidate_materialized_bytes": materialized_bytes,
        "credited_capacity_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_loss_positions": 0,
        "long_training_authorized": False,
        "model_training_executed": False,
        "status": "MATERIALIZED_CANDIDATE_ONLY_DOWNSTREAM_GATES_REQUIRED",
    }

    try:
        objects_dir = staging / "objects"
        objects_dir.mkdir(parents=True, exist_ok=False)
        for _, normalized, record in prepared:
            payload_path = staging / str(record["payload_path"])
            payload_path.write_bytes(normalized)
        (staging / "candidate_inventory.json").write_bytes(_canonical_json_bytes(receipt) + b"\n")
        os.replace(staging, output_root_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def load_plan_and_manifest(
    *,
    plan_path: str | Path,
    manifest_path: str | Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    plan_bytes = Path(plan_path).read_bytes()
    manifest_bytes = Path(manifest_path).read_bytes()
    try:
        plan = json.loads(plan_bytes)
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise CandidateMaterializationError("plan and manifest must be valid JSON") from exc
    if not isinstance(plan, dict) or not isinstance(manifest, dict):
        raise CandidateMaterializationError("plan and manifest roots must be objects")
    plan_sha256 = _sha256_bytes(plan_bytes)
    validate_materialization_manifest(manifest, plan=plan, plan_sha256=plan_sha256)
    return plan, plan_sha256, manifest
