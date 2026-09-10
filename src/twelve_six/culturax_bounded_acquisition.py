from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "12-6.d03-culturax-bounded-acquisition.v1"
_ALLOWED_LANGUAGES = ("en", "uk")
_ALLOWED_LINEAGES = frozenset(
    {"mc4", "OSCAR-20.19", "OSCAR-21.09", "OSCAR-22.01", "OSCAR-23.01"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHARD_RE = re.compile(r"^(?P<lang>en|uk)_part_(?P<index>[0-9]{5})\.parquet$")


class CulturaXContractError(ValueError):
    """Raised when CulturaX acquisition evidence violates the fail-closed contract."""


@dataclass(frozen=True, order=True)
class ShardChecksum:
    language: str
    filename: str
    sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CulturaXContractError(f"{label} must be lowercase SHA-256")
    return value


def _require_git_sha1(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA1_RE.fullmatch(value) is None:
        raise CulturaXContractError(f"{label} must be lowercase 40-hex Git SHA")
    return value


def _require_zero_false_claims(boundary: Mapping[str, Any]) -> None:
    expected = {
        "training_authorized_bytes": 0,
        "authorized_unique_causal_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates": 0,
        "model_training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
    }
    if dict(boundary) != expected:
        raise CulturaXContractError("claim_boundary must remain exact zero/false pre-admission")


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise CulturaXContractError("unsupported schema_version")
    if contract.get("execution_profile") != "LOCAL_FREE":
        raise CulturaXContractError("only LOCAL_FREE execution is allowed by this contract")

    authority = contract.get("authority")
    if not isinstance(authority, Mapping):
        raise CulturaXContractError("authority must be an object")
    _require_git_sha1(authority.get("base_main_sha"), "authority.base_main_sha")
    _require_git_sha1(authority.get("parent_audit_head_sha"), "authority.parent_audit_head_sha")
    _require_git_sha1(authority.get("dataset_card_revision"), "authority.dataset_card_revision")
    if authority.get("claim_issue") != 853 or authority.get("parent_audit_pr") != 762:
        raise CulturaXContractError("claim/audit authority drift")

    upstream = contract.get("upstream")
    if not isinstance(upstream, Mapping):
        raise CulturaXContractError("upstream must be an object")
    required_upstream = {
        "dataset_id": "uonlp/CulturaX",
        "repository_host": "huggingface.co",
        "gated": True,
        "access_acceptance_required": True,
        "access_acceptance_observed": False,
        "payload_download_executed": False,
    }
    for key, expected in required_upstream.items():
        if upstream.get(key) != expected:
            raise CulturaXContractError(f"upstream.{key} drift")

    languages = upstream.get("languages")
    if not isinstance(languages, Mapping) or set(languages) != set(_ALLOWED_LANGUAGES):
        raise CulturaXContractError("upstream.languages must contain exactly en and uk")
    for language in _ALLOWED_LANGUAGES:
        row = languages[language]
        if not isinstance(row, Mapping):
            raise CulturaXContractError(f"upstream.languages.{language} must be an object")
        if row.get("directory") != language or row.get("checksum_file") != "checksum.sha256":
            raise CulturaXContractError(f"invalid {language} checksum location")

    if upstream.get("record_schema") != ["text", "timestamp", "url", "source"]:
        raise CulturaXContractError("record_schema drift")
    if set(upstream.get("allowed_source_lineages", ())) != _ALLOWED_LINEAGES:
        raise CulturaXContractError("allowed source lineage drift")

    selection = contract.get("selection")
    if not isinstance(selection, Mapping):
        raise CulturaXContractError("selection must be an object")
    expected_selection = {
        "selection_rule": "lexicographically_first_shard_per_language",
        "max_shards_per_language": 1,
        "max_total_shards": 2,
        "required_suffix": ".parquet",
        "checksum_algorithm": "sha256",
        "require_checksum_for_every_selected_shard": True,
        "require_project_sha256_after_download": True,
    }
    if dict(selection) != expected_selection:
        raise CulturaXContractError("selection policy drift")

    gates = contract.get("downstream_gates")
    if not isinstance(gates, Mapping) or not gates:
        raise CulturaXContractError("downstream_gates must be a non-empty object")
    if any(value != "BLOCKED" for value in gates.values()):
        raise CulturaXContractError("all downstream gates must remain BLOCKED pre-admission")

    boundary = contract.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise CulturaXContractError("claim_boundary must be an object")
    _require_zero_false_claims(boundary)


def parse_checksum_manifest(text: str, language: str) -> list[ShardChecksum]:
    if language not in _ALLOWED_LANGUAGES:
        raise CulturaXContractError("language must be en or uk")

    entries: list[ShardChecksum] = []
    seen_files: set[str] = set()
    seen_hashes: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise CulturaXContractError(f"checksum line {line_number} must contain hash and file")
        digest, filename = parts
        digest = _require_sha256(digest, f"checksum line {line_number}")
        filename = filename.removeprefix("*")
        match = _SHARD_RE.fullmatch(filename)
        if match is None or match.group("lang") != language:
            raise CulturaXContractError(f"checksum line {line_number} has invalid shard filename")
        if filename in seen_files:
            raise CulturaXContractError("duplicate shard filename")
        if digest in seen_hashes:
            raise CulturaXContractError("duplicate shard SHA-256")
        seen_files.add(filename)
        seen_hashes.add(digest)
        entries.append(ShardChecksum(language=language, filename=filename, sha256=digest))

    if not entries:
        raise CulturaXContractError(f"{language} checksum manifest is empty")
    return sorted(entries)


def build_bounded_plan(
    contract: Mapping[str, Any],
    checksum_manifests: Mapping[str, str],
) -> dict[str, Any]:
    validate_contract(contract)
    if set(checksum_manifests) != set(_ALLOWED_LANGUAGES):
        raise CulturaXContractError("checksum_manifests must contain exactly en and uk")

    selected: list[dict[str, str]] = []
    manifest_sha256: dict[str, str] = {}
    dataset_revision = contract["authority"]["dataset_card_revision"]
    for language in _ALLOWED_LANGUAGES:
        text = checksum_manifests[language]
        entries = parse_checksum_manifest(text, language)
        winner = entries[0]
        manifest_sha256[language] = sha256_bytes(text.encode())
        selected.append(
            {
                "language": language,
                "filename": winner.filename,
                "upstream_sha256": winner.sha256,
                "relative_path": f"{language}/{winner.filename}",
                "revision": dataset_revision,
            }
        )

    selected.sort(key=lambda item: (item["language"], item["filename"]))
    if len(selected) > contract["selection"]["max_total_shards"]:
        raise CulturaXContractError("bounded shard selection exceeds max_total_shards")

    plan = {
        "schema_version": "12-6.d03-culturax-bounded-plan.v1",
        "contract_schema_version": SCHEMA_VERSION,
        "dataset_id": contract["upstream"]["dataset_id"],
        "dataset_revision": dataset_revision,
        "checksum_manifest_sha256": manifest_sha256,
        "selected_shards": selected,
        "access_state": "BLOCKED_GATED_ACCESS_ACCEPTANCE_REQUIRED",
        "training_authorized_bytes": 0,
        "authorized_unique_causal_loss_positions": 0,
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def verify_download_receipt(plan: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    selected = plan.get("selected_shards")
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise CulturaXContractError("plan.selected_shards must be a sequence")
    expected = {
        (item["language"], item["filename"]): item["upstream_sha256"] for item in selected
    }
    rows = receipt.get("shards")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise CulturaXContractError("receipt.shards must be a sequence")
    if len(rows) != len(expected):
        raise CulturaXContractError("receipt shard count mismatch")

    observed: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise CulturaXContractError("receipt shard row must be an object")
        key = (row.get("language"), row.get("filename"))
        if key not in expected or key in observed:
            raise CulturaXContractError("receipt contains unexpected or duplicate shard")
        observed.add(key)
        project_digest = _require_sha256(row.get("project_sha256"), "receipt project_sha256")
        if project_digest != expected[key]:
            raise CulturaXContractError("downloaded shard SHA-256 does not match upstream manifest")
        size = row.get("downloaded_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CulturaXContractError("downloaded_bytes must be a positive integer")

    if receipt.get("training_authorized_bytes") != 0:
        raise CulturaXContractError("download receipt cannot authorize training bytes")


def validate_record(record: Mapping[str, Any]) -> None:
    if set(record) != {"text", "timestamp", "url", "source"}:
        raise CulturaXContractError("record fields must be text,timestamp,url,source exactly")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise CulturaXContractError("record text must be non-empty")
    if not isinstance(record["timestamp"], str) or not record["timestamp"].strip():
        raise CulturaXContractError("record timestamp must be non-empty")
    if record["source"] not in _ALLOWED_LINEAGES:
        raise CulturaXContractError("record source lineage is not approved by the audit boundary")
    if not isinstance(record["url"], str):
        raise CulturaXContractError("record url must be a string")
    parsed = urlparse(record["url"])
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CulturaXContractError("record url must preserve a valid http(s) origin")
