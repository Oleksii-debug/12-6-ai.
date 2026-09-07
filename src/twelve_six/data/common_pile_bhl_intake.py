from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_SCHEMA = "12-6.d03-common-pile-bhl-intake.v1"
_REPORT_SCHEMA = "12-6.d03-common-pile-bhl-materialization.v1"
_REQUIRED_RECORD_KEYS = frozenset(
    {"id", "page_id", "item_id", "page_num", "text", "source", "added", "metadata"}
)
_REQUIRED_METADATA_KEYS = frozenset({"license", "url"})
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGITS_RE = re.compile(r"^[0-9]+$")
_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9.-]+\.[a-z]{2,}(?![\w.-])")
_SECRET_RE = re.compile(
    r"(?i)(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?:api[_ -]?key|access[_ -]?token|password)\s*[:=]\s*\S+)"
)
_ALLOWED_TEXT_CONTROLS = {"\n", "\r", "\t"}

_EXPECTED_AUDIT = {
    "registry_path": "configs/data/common_pile_source_rights_v1.json",
    "registry_blob_sha1": "7b4d6828288672bf25c551e85a5d7f7399e8ef0f",
    "source_key": "biodiversity_heritage_library",
    "audited_code_revision": "9457f04a14cb2355ab00023420369d46ffd4a395",
    "build_index_path": "sources/bhl/build-index.py",
    "build_index_blob_sha1": "e36d7df76fab7db081dc20e4815f4de1e41a75bf",
    "extract_files_path": "sources/bhl/extract-files.py",
    "extract_files_blob_sha1": "43791cf050d0e678e01037e522b2a503c3ce6908",
    "license_whitelist_path": "sources/bhl/license_whitelist.json",
    "license_whitelist_blob_sha1": "5a4bc431ee27825c99f2d1d84f456a3bfbca1fb3",
    "to_dolma_path": "sources/bhl/to-dolma.py",
    "to_dolma_blob_sha1": "0104d471bda7588548ae5fe9a43389e96e907661",
}
_EXPECTED_UPSTREAM = {
    "dataset": "common-pile/biodiversity_heritage_library",
    "revision": "8e332cad3fa3f67a9fe189603a39e5ececc5a5ec",
    "shard_path": "v0/00000_bhl.jsonl.gz",
    "shard_bytes": 349_614_022,
    "shard_sha256": "b8d089aa29ad54a5edf7f5a56f963ee30e37c4276b674ccab8ebea8b45e33003",
    "xet_hash": "f93f797391f5bd9a530c4f60afe9cdce5e257cbac1d69c43a2afd2a11a9a1fd2",
    "source_name": "biodiversity-heritage-library",
}
_REQUIRED_GATES = [
    "source_rights_revalidation",
    "language_quality",
    "privacy",
    "global_exact_near_fragment_lineage_dedup",
    "reserved_evaluation_decontamination",
    "balance_family_caps",
    "cluster_safe_split",
    "deterministic_pack_two_clean_builds",
    "positive_unique_loss_ledger",
]


class BhlIntakeError(ValueError):
    """Fail-closed violation in the bounded BHL source intake."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256_bytes((canonical_json(clone) + "\n").encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BhlIntakeError(message)


def _require_hex(value: object, length: int, field: str) -> str:
    _require(isinstance(value, str), f"{field} must be a string")
    pattern = _HEX40_RE if length == 40 else _HEX64_RE
    _require(bool(pattern.fullmatch(value)), f"{field} must be lowercase {length}-hex")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": _SCHEMA,
        "source_family": "en.bhl.public-domain-pages",
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "final_test_payload_accessed": False,
    }
    for field, value in expected.items():
        _require(config.get(field) == value, f"{field} drift")

    audit = config.get("common_pile_audit")
    _require(isinstance(audit, Mapping), "common_pile_audit missing")
    for field, expected_value in _EXPECTED_AUDIT.items():
        _require(audit.get(field) == expected_value, f"Common Pile audit {field} drift")
    for field in (
        "registry_blob_sha1",
        "audited_code_revision",
        "build_index_blob_sha1",
        "extract_files_blob_sha1",
        "license_whitelist_blob_sha1",
        "to_dolma_blob_sha1",
    ):
        _require_hex(audit.get(field), 40, f"common_pile_audit.{field}")

    upstream = config.get("upstream")
    _require(isinstance(upstream, Mapping), "upstream missing")
    for field, expected_value in _EXPECTED_UPSTREAM.items():
        _require(upstream.get(field) == expected_value, f"upstream {field} drift")
    _require_hex(upstream.get("revision"), 40, "upstream revision")
    _require_hex(upstream.get("shard_sha256"), 64, "upstream shard sha256")
    _require_hex(upstream.get("xet_hash"), 64, "upstream xet hash")
    expected_url = (
        "https://huggingface.co/datasets/common-pile/biodiversity_heritage_library/"
        f"resolve/{upstream['revision']}/{upstream['shard_path']}?download=true"
    )
    _require(upstream.get("download_url") == expected_url, "upstream download URL drift")

    rights = config.get("rights_policy")
    _require(isinstance(rights, Mapping), "rights_policy missing")
    _require(rights.get("required_metadata_license") == "Public Domain", "license authority drift")
    _require(
        rights.get("required_url_prefix") == "https://www.biodiversitylibrary.org/page/",
        "BHL URL authority drift",
    )
    _require(rights.get("dataset_language_context") == "en", "dataset language context drift")
    _require(
        rights.get("per_record_language_authority_available") is False,
        "per-record language authority must remain unavailable",
    )
    _require(
        rights.get("dataset_package_license_is_training_authority") is False,
        "dataset package license may not authorize training",
    )
    _require(rights.get("legal_conclusion_claimed") is False, "legal conclusion may not be claimed")

    selection = config.get("selection_policy")
    _require(isinstance(selection, Mapping), "selection_policy missing")
    _require(selection.get("ordering") == "UPSTREAM_SHARD_ORDER", "selection ordering drift")
    exact_numeric = {
        "max_examined_documents": 512,
        "max_accepted_documents": 256,
        "max_total_normalized_utf8_bytes": 4_500_000,
        "max_document_normalized_utf8_bytes": 250_000,
        "min_document_normalized_utf8_bytes": 512,
        "max_json_line_bytes": 2_000_000,
        "max_remote_compressed_prefix_bytes": 33_554_432,
    }
    for field, expected_value in exact_numeric.items():
        _require(selection.get(field) == expected_value, f"selection_policy {field} drift")
    _require(
        selection["max_total_normalized_utf8_bytes"] <= 5_000_000,
        "family planning ceiling exceeded",
    )

    _require(config.get("required_downstream_gates") == _REQUIRED_GATES, "required gates drift")
    identity = _require_hex(config.get("contract_identity_sha256"), 64, "contract identity")
    _require(
        self_identity(config, "contract_identity_sha256") == identity,
        "contract identity mismatch",
    )


def normalize_text(text: object) -> str:
    _require(isinstance(text, str), "text must be a string")
    normalized_lines = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", normalized_lines).strip()
    _require(bool(normalized), "text is empty")
    for char in normalized:
        if unicodedata.category(char) == "Cc" and char not in _ALLOWED_TEXT_CONTROLS:
            raise BhlIntakeError("text contains forbidden control characters")
    return normalized


def _valid_page_url(url: object, page_id: str) -> bool:
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return (
        url == f"https://www.biodiversitylibrary.org/page/{page_id}"
        and parts.scheme == "https"
        and parts.hostname == "www.biodiversitylibrary.org"
        and parts.username is None
        and parts.password is None
        and not parts.query
        and not parts.fragment
    )


def _record_authority(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, str, str, str, Mapping[str, Any]]:
    _require(set(record) == _REQUIRED_RECORD_KEYS, "record top-level schema drift")
    _require(record.get("source") == config["upstream"]["source_name"], "source drift")

    record_id = record.get("id")
    page_id = record.get("page_id")
    item_id = record.get("item_id")
    page_num = record.get("page_num")
    for value, field in (
        (record_id, "id"),
        (page_id, "page_id"),
        (item_id, "item_id"),
        (page_num, "page_num"),
    ):
        _require(isinstance(value, str) and bool(value), f"invalid {field}")
        no_space = value.strip() == value and not any(ch.isspace() for ch in value)
        _require(no_space, f"{field} whitespace")

    _require(bool(_DIGITS_RE.fullmatch(page_id)), "page_id must be numeric")
    _require(bool(_DIGITS_RE.fullmatch(item_id)), "item_id must be numeric")
    _require(bool(_DIGITS_RE.fullmatch(page_num)), "page_num must be numeric")
    expected_record_id = f"{item_id}-{page_id}-{page_num}"
    _require(record_id == expected_record_id, "record id does not bind BHL components")
    added = record.get("added")
    _require(isinstance(added, str) and bool(added.strip()), "invalid added timestamp")

    metadata = record.get("metadata")
    _require(isinstance(metadata, Mapping), "metadata missing")
    _require(set(metadata) == _REQUIRED_METADATA_KEYS, "metadata schema drift")
    _require(
        metadata.get("license") == config["rights_policy"]["required_metadata_license"],
        "record license is not Public Domain",
    )
    _require(_valid_page_url(metadata.get("url"), page_id), "BHL page URL does not bind page_id")
    return record_id, page_id, item_id, page_num, metadata


def _quality_reason(text: str, config: Mapping[str, Any]) -> str | None:
    payload = text.encode("utf-8")
    policy = config["selection_policy"]
    if len(payload) < policy["min_document_normalized_utf8_bytes"]:
        return "too_short"
    if len(payload) > policy["max_document_normalized_utf8_bytes"]:
        return "too_large"
    if _EMAIL_RE.search(text):
        return "email_like_contact"
    if _SECRET_RE.search(text):
        return "secret_like_text"
    visible = [char for char in text if not char.isspace()]
    alpha = sum(char.isalpha() for char in visible)
    if not visible or alpha < 100 or alpha / len(visible) < 0.15:
        return "low_alpha_content"
    return None


def _rows_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes((canonical_json(list(rows)) + "\n").encode("utf-8"))


def materialize(
    config: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    full_shard_hash_verified: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_config(config)
    policy = config["selection_policy"]
    candidates: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_payloads: set[str] = set()
    examined = 0
    accepted_bytes = 0

    for record in records:
        if (
            examined >= policy["max_examined_documents"]
            or len(candidates) >= policy["max_accepted_documents"]
            or accepted_bytes >= policy["max_total_normalized_utf8_bytes"]
        ):
            break
        _require(isinstance(record, Mapping), "JSONL row must be an object")
        examined += 1
        record_id, page_id, item_id, page_num, metadata = _record_authority(record, config)
        _require(record_id not in seen_ids, f"duplicate record id: {record_id}")
        seen_ids.add(record_id)

        text = normalize_text(record.get("text"))
        payload = text.encode("utf-8")
        payload_sha = sha256_bytes(payload)
        reason = _quality_reason(text, config)
        if reason is None and payload_sha in seen_payloads:
            reason = "exact_normalized_duplicate"
        if (
            reason is None
            and accepted_bytes + len(payload) > policy["max_total_normalized_utf8_bytes"]
        ):
            reason = "family_byte_cap"

        evidence.append(
            {
                "record_id": record_id,
                "page_id": page_id,
                "item_id": item_id,
                "page_num": page_num,
                "url": metadata["url"],
                "text_sha256": payload_sha,
                "normalized_utf8_bytes": len(payload),
                "decision": "ACCEPT" if reason is None else "REJECT",
                "reason": "candidate_zero_credit" if reason is None else reason,
            }
        )
        if reason is not None:
            rejected[reason] += 1
            continue

        seen_payloads.add(payload_sha)
        candidates.append(
            {
                "record_id": f"bhl:{record_id}",
                "source_family": config["source_family"],
                "source_revision": config["upstream"]["revision"],
                "source_shard_path": config["upstream"]["shard_path"],
                "source_shard_sha256": config["upstream"]["shard_sha256"],
                "source_record_id": record_id,
                "page_id": page_id,
                "item_id": item_id,
                "page_num": page_num,
                "provenance_url": metadata["url"],
                "text": text,
                "text_sha256": payload_sha,
                "utf8_bytes": len(payload),
                "training_eligible": False,
                "evaluation_eligible": False,
            }
        )
        accepted_bytes += len(payload)

    _require(bool(candidates), "bounded materialization retained zero candidates")
    inventory = [
        {
            "record_id": row["record_id"],
            "source_record_id": row["source_record_id"],
            "page_id": row["page_id"],
            "item_id": row["item_id"],
            "page_num": row["page_num"],
            "provenance_url": row["provenance_url"],
            "text_sha256": row["text_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
        for row in candidates
    ]
    report: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA,
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": config["source_family"],
        "contract_identity_sha256": config["contract_identity_sha256"],
        "upstream_revision": config["upstream"]["revision"],
        "source_shard_path": config["upstream"]["shard_path"],
        "source_shard_sha256": config["upstream"]["shard_sha256"],
        "full_shard_hash_verified": full_shard_hash_verified,
        "two_independent_materializations_verified": False,
        "examined_documents": examined,
        "accepted_documents": len(candidates),
        "accepted_normalized_utf8_bytes": accepted_bytes,
        "rejection_counts": dict(sorted(rejected.items())),
        "candidate_inventory_identity_sha256": _rows_identity(inventory),
        "rights_provenance_evidence_identity_sha256": _rows_identity(evidence),
        "per_record_language_authority_available": False,
        "language_quality_gate": "NOT_RUN",
        "source_rights_revalidation_gate": "NOT_RUN",
        "privacy_gate": "NOT_RUN",
        "global_dedup_gate": "NOT_RUN",
        "reserved_evaluation_decontamination_gate": "NOT_RUN",
        "balance_family_caps_gate": "NOT_RUN",
        "cluster_safe_split_gate": "NOT_RUN",
        "packing_two_clean_builds_gate": "NOT_RUN",
        "positive_unique_loss_ledger_gate": "NOT_RUN",
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "final_test_payload_accessed": False,
    }
    report["report_identity_sha256"] = self_identity(report, "report_identity_sha256")
    return candidates, report


def materialize_twice(
    config: Mapping[str, Any],
    first_records: Iterable[Mapping[str, Any]],
    second_records: Iterable[Mapping[str, Any]],
    *,
    full_shard_hash_verified: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_rows, first_report = materialize(
        config,
        first_records,
        full_shard_hash_verified=full_shard_hash_verified,
    )
    second_rows, second_report = materialize(
        config,
        second_records,
        full_shard_hash_verified=full_shard_hash_verified,
    )
    _require(first_rows == second_rows, "independent candidate materializations differ")
    _require(first_report == second_report, "independent materialization reports differ")
    final_report = deepcopy(first_report)
    final_report["two_independent_materializations_verified"] = True
    final_report["report_identity_sha256"] = self_identity(final_report, "report_identity_sha256")
    return first_rows, final_report


def verify_report(config: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    validate_config(config)
    _require(report.get("schema_version") == _REPORT_SCHEMA, "report schema drift")
    expected = {
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": config["source_family"],
        "contract_identity_sha256": config["contract_identity_sha256"],
        "upstream_revision": config["upstream"]["revision"],
        "source_shard_path": config["upstream"]["shard_path"],
        "source_shard_sha256": config["upstream"]["shard_sha256"],
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "paid_compute_authorized": False,
        "final_test_payload_accessed": False,
        "per_record_language_authority_available": False,
        "language_quality_gate": "NOT_RUN",
    }
    for field, value in expected.items():
        _require(report.get(field) == value, f"report {field} drift")
    identity = _require_hex(report.get("report_identity_sha256"), 64, "report identity")
    _require(
        self_identity(report, "report_identity_sha256") == identity,
        "report identity mismatch",
    )


def verify_full_shard(path: Path, config: Mapping[str, Any]) -> None:
    validate_config(config)
    upstream = config["upstream"]
    _require(path.stat().st_size == upstream["shard_bytes"], "full shard size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    _require(digest.hexdigest() == upstream["shard_sha256"], "full shard SHA-256 mismatch")


def iter_gzip_jsonl_bytes(
    raw_gzip: bytes,
    *,
    max_json_line_bytes: int,
) -> Iterable[dict[str, Any]]:
    _require(isinstance(raw_gzip, bytes) and bool(raw_gzip), "gzip input is empty")
    with gzip.GzipFile(fileobj=io.BytesIO(raw_gzip), mode="rb") as handle:
        while True:
            raw_line = handle.readline(max_json_line_bytes + 1)
            if not raw_line:
                return
            _require(len(raw_line) <= max_json_line_bytes, "JSONL line exceeds safety bound")
            _require(raw_line.endswith(b"\n"), "truncated JSONL line")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise BhlIntakeError("JSONL is not strict UTF-8") from exc
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BhlIntakeError("malformed JSONL row") from exc
            _require(isinstance(value, dict), "JSONL row must be an object")
            yield value


def iter_gzip_jsonl_path(
    path: Path,
    *,
    max_json_line_bytes: int,
) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        while True:
            raw_line = handle.readline(max_json_line_bytes + 1)
            if not raw_line:
                return
            _require(len(raw_line) <= max_json_line_bytes, "JSONL line exceeds safety bound")
            _require(raw_line.endswith(b"\n"), "truncated JSONL line")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise BhlIntakeError("JSONL is not strict UTF-8") from exc
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BhlIntakeError("malformed JSONL row") from exc
            _require(isinstance(value, dict), "JSONL row must be an object")
            yield value


def write_materialization(
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [canonical_json(dict(row)) for row in candidates]
    (output_dir / "bhl-candidate.jsonl").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "bhl-report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
