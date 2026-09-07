"""Current-main authority for the immutable RECOVER-174 UA/EN final-test set.

The raw final-test text is loaded only into ephemeral caller memory for contamination
matching. Durable authority output is text-free and preserves the exact historical
RECOVER-174 seed and evaluation-use authority blobs already proven by EVAL-233.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

AUTHORITY_ID = "RECOVER-174-UA-EN-FINAL-TEST-V1"
ROLE = "final_test"
AUTHORITY_SCHEMA = "12-6.recover174-current-final-test-authority.v1"
MEMBERSHIP_SCHEMA = "12-6.recover174-final-test-membership.v1"

RECOVER174_SEED_PATH = Path("data/evaluation/recover174_real_holdout_seed.jsonl.gz")
RECOVER174_AUTHORITY_PATH = Path("configs/evaluation/recover174_source_authority_v1.json")
CURRENT_AUTHORITY_PATH = Path("evidence/eval233/current-final-test-authority.json")

RECOVER174_SEED_GIT_BLOB_SHA1 = "4bfbfbf29fa9538cabda6068efd3a1fd036a9479"
RECOVER174_SEED_SHA256 = "7e6827d22d573dda4c9ff2b5f0ab2b8fe3fdf5aa577970ecb77e94a10cf72367"
RECOVER174_AUTHORITY_GIT_BLOB_SHA1 = "3ba9f221a82468f971c17eda518cd6f1642fd311"
RECOVER174_AUTHORITY_IDENTITY_SHA256 = (
    "c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f"
)
CANONICAL_FINAL_TEST_SET_IDENTITY_SHA256 = (
    "6b012efc4d627b113b8adc2166e6ab50d9001284083f7a429c665b7752ca18d7"
)
SOURCE_MEMBERSHIP_IDENTITY_SHA256 = (
    "d77220397547f049d0946618170410d42f729db932c40fc5c86c9b0c9470eb31"
)
CURRENT_AUTHORITY_EVIDENCE_IDENTITY_SHA256 = (
    "ec5b4067a4cdd7adbcfd71db48373332c483da36349a456e44e954fe5e7d55cc"
)


class Recover174FinalTestAuthorityError(RuntimeError):
    """Fail-closed final-test authority error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Recover174FinalTestAuthorityError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()  # noqa: S324 - Git object identity


def _require_sha(value: str, *, length: int, label: str) -> str:
    _require(
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None,
        f"{label} must be lowercase hex length {length}",
    )
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Recover174FinalTestAuthorityError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _load_evidence(repo_root: Path) -> dict[str, Any]:
    evidence = _load_json(repo_root / CURRENT_AUTHORITY_PATH)
    _require(evidence.get("schema_version") == AUTHORITY_SCHEMA, "authority schema drift")
    claimed = evidence.get("evidence_identity_sha256")
    _require(
        claimed == CURRENT_AUTHORITY_EVIDENCE_IDENTITY_SHA256,
        "authority evidence identity constant drift",
    )
    core = deepcopy(evidence)
    core.pop("evidence_identity_sha256", None)
    _require(_sha256_json(core) == claimed, "authority evidence self-hash mismatch")
    _require(evidence.get("authority_id") == AUTHORITY_ID, "authority id drift")
    _require(evidence.get("role") == ROLE, "authority role drift")
    _require(
        evidence.get("canonical_final_test_set_identity_sha256")
        == CANONICAL_FINAL_TEST_SET_IDENTITY_SHA256,
        "canonical final-test set identity drift",
    )
    _require(
        evidence.get("source_membership_identity_sha256")
        == SOURCE_MEMBERSHIP_IDENTITY_SHA256,
        "source membership identity drift",
    )
    boundaries = evidence.get("boundaries")
    _require(isinstance(boundaries, dict), "authority boundaries missing")
    for key in (
        "training_prohibited",
        "tokenizer_fit_prohibited",
        "hyperparameter_selection_prohibited",
        "raw_text_persisted_in_authority",
        "code_included",
    ):
        expected = key not in {"raw_text_persisted_in_authority", "code_included"}
        _require(boundaries.get(key) is expected, f"authority boundary drift: {key}")
    _require(boundaries.get("outcomes_included") is False, "final-test outcomes leaked")
    _require(boundaries.get("local_free_only") is True, "LOCAL_FREE boundary drift")
    return evidence


def _validate_source_authority(repo_root: Path) -> dict[str, Any]:
    path = repo_root / RECOVER174_AUTHORITY_PATH
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Recover174FinalTestAuthorityError(f"cannot read {path}: {exc}") from exc
    _require(
        git_blob_sha1(raw) == RECOVER174_AUTHORITY_GIT_BLOB_SHA1,
        "RECOVER-174 source-authority Git blob mismatch",
    )
    authority = json.loads(raw.decode("utf-8"))
    _require(isinstance(authority, dict), "RECOVER-174 authority must be an object")
    _require(
        authority.get("authority_identity_sha256")
        == RECOVER174_AUTHORITY_IDENTITY_SHA256,
        "RECOVER-174 semantic authority identity mismatch",
    )
    _require(
        set(authority.get("admitted_modalities", [])) == {"ua", "en"},
        "RECOVER-174 admitted modalities drift",
    )
    blocked = authority.get("blocked_modalities")
    _require(
        isinstance(blocked, dict)
        and isinstance(blocked.get("code"), dict)
        and blocked["code"].get("status") == "BLOCKED_NO_EVALUATION_USE_AUTHORITY",
        "RECOVER-174 code boundary drift",
    )
    return authority


def _materialize_members(
    repo_root: Path, authority: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    path = repo_root / RECOVER174_SEED_PATH
    try:
        seed_blob = path.read_bytes()
    except OSError as exc:
        raise Recover174FinalTestAuthorityError(f"cannot read {path}: {exc}") from exc
    _require(
        git_blob_sha1(seed_blob) == RECOVER174_SEED_GIT_BLOB_SHA1,
        "RECOVER-174 final-test seed Git blob mismatch",
    )
    _require(
        _sha256_bytes(seed_blob) == RECOVER174_SEED_SHA256,
        "RECOVER-174 final-test seed SHA-256 mismatch",
    )
    try:
        raw_rows = gzip.decompress(seed_blob).splitlines()
    except OSError as exc:
        raise Recover174FinalTestAuthorityError("RECOVER-174 seed is not valid gzip") from exc

    sources = authority.get("sources")
    _require(isinstance(sources, dict), "RECOVER-174 source authority has no sources")
    payloads: list[dict[str, str]] = []
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    modality_counts = {"ua": 0, "en": 0}

    for index, raw_line in enumerate(raw_rows):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise Recover174FinalTestAuthorityError(
                f"invalid seed JSON row {index}"
            ) from exc
        _require(isinstance(row, dict), f"seed row {index} must be an object")
        values: dict[str, str] = {}
        for key in (
            "record_id",
            "source_id",
            "source_family",
            "modality",
            "source_snapshot_sha256",
            "source_kind",
            "text",
        ):
            value = row.get(key)
            _require(
                isinstance(value, str) and bool(value),
                f"seed row {index}.{key} must be non-empty text",
            )
            values[key] = value

        record_id = values["record_id"]
        _require(record_id not in seen, f"duplicate final-test record_id: {record_id}")
        seen.add(record_id)
        modality = values["modality"].lower()
        _require(modality in {"ua", "en"}, f"unsupported final-test modality: {modality}")
        _require(values["source_kind"] == "EXTERNAL_REAL", "non-real final-test source")

        source = sources.get(values["source_id"])
        _require(isinstance(source, dict), f"source is not authorized: {values['source_id']}")
        _require(
            source.get("evaluation_status") == "APPROVED_FOR_HELDOUT_EVALUATION",
            f"source lacks held-out evaluation authority: {values['source_id']}",
        )
        _require(
            source.get("modality") == modality,
            f"source modality drift: {values['source_id']}",
        )
        snapshots = source.get("admitted_source_snapshots_sha256")
        _require(
            isinstance(snapshots, list)
            and values["source_snapshot_sha256"] in snapshots,
            f"source snapshot is not admitted: {record_id}",
        )

        text_bytes = values["text"].encode("utf-8")
        _require(bool(text_bytes), f"empty final-test text: {record_id}")
        payloads.append(
            {
                "record_id": record_id,
                "source_id": values["source_id"],
                "source_family": values["source_family"],
                "modality": modality,
                "text": values["text"],
            }
        )
        members.append(
            {
                "record_id": record_id,
                "source_id": values["source_id"],
                "source_family": values["source_family"],
                "modality": modality,
                "content_sha256": _sha256_bytes(text_bytes),
                "utf8_bytes": len(text_bytes),
                "training_prohibited": True,
                "outcomes_included": False,
            }
        )
        modality_counts[modality] += 1

    _require(len(payloads) == 16, "RECOVER-174 final-test must contain exactly 16 records")
    _require(modality_counts == {"ua": 8, "en": 8}, "RECOVER-174 UA/EN balance drift")
    payloads.sort(key=lambda row: row["record_id"])
    members.sort(key=lambda row: row["record_id"])
    return payloads, members


def load_final_test_reserved_set(
    repo_root: str | Path,
    *,
    source_sha: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return ephemeral payload rows plus the text-free #874 reserved-set contract."""
    source_sha = _require_sha(source_sha, length=40, label="source_sha")
    root = Path(repo_root)
    evidence = _load_evidence(root)
    authority = _validate_source_authority(root)
    payloads, members = _materialize_members(root, authority)

    membership = {"schema_version": MEMBERSHIP_SCHEMA, "members": members}
    _require(
        _sha256_json(membership) == SOURCE_MEMBERSHIP_IDENTITY_SHA256,
        "RECOVER-174 final-test membership identity mismatch",
    )
    _require(
        evidence.get("membership") == membership,
        "durable authority membership differs from exact seed",
    )

    reserved_set: dict[str, Any] = {
        "authority_id": AUTHORITY_ID,
        "identity_sha256": CANONICAL_FINAL_TEST_SET_IDENTITY_SHA256,
        "role": ROLE,
        "source_sha": source_sha,
        "source_membership_identity_sha256": SOURCE_MEMBERSHIP_IDENTITY_SHA256,
        "members": deepcopy(members),
    }

    def _has_text_key(value: Any) -> bool:
        if isinstance(value, dict):
            return "text" in value or any(_has_text_key(item) for item in value.values())
        if isinstance(value, list):
            return any(_has_text_key(item) for item in value)
        return False

    _require(not _has_text_key(reserved_set), "raw text key leaked into reserved authority")
    return payloads, reserved_set


def verify_current_final_test_authority(
    repo_root: str | Path,
    *,
    source_sha: str,
    expected_identity_sha256: str = CANONICAL_FINAL_TEST_SET_IDENTITY_SHA256,
    expected_membership_identity_sha256: str = SOURCE_MEMBERSHIP_IDENTITY_SHA256,
) -> dict[str, Any]:
    """Fail closed and return a concise text-free verification summary."""
    payloads, reserved_set = load_final_test_reserved_set(
        repo_root,
        source_sha=source_sha,
    )
    _require(
        reserved_set["identity_sha256"]
        == _require_sha(expected_identity_sha256, length=64, label="expected_identity_sha256"),
        "independently expected final-test identity mismatch",
    )
    _require(
        reserved_set["source_membership_identity_sha256"]
        == _require_sha(
            expected_membership_identity_sha256,
            length=64,
            label="expected_membership_identity_sha256",
        ),
        "independently expected membership identity mismatch",
    )
    return {
        "schema_version": "12-6.recover174-current-final-test-verification.v1",
        "authority_id": AUTHORITY_ID,
        "role": ROLE,
        "source_sha": source_sha,
        "identity_sha256": reserved_set["identity_sha256"],
        "source_membership_identity_sha256": reserved_set[
            "source_membership_identity_sha256"
        ],
        "documents": len(payloads),
        "modality_documents": {
            "ua": sum(row["modality"] == "ua" for row in payloads),
            "en": sum(row["modality"] == "en" for row in payloads),
        },
        "raw_text_persisted": False,
        "outcomes_included": False,
        "training_prohibited": True,
        "local_free_only": True,
    }
