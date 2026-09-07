#!/usr/bin/env python3
"""Bounded, zero-credit Common Pile Caselaw candidate materializer."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/d03_common_pile_caselaw_zero_credit_v1.json")
SCHEMA = "12-6.d03-common-pile-caselaw-zero-credit.v1"
REPORT_SCHEMA = "12-6.d03-common-pile-caselaw-zero-credit-report.v1"
RIGHTS_PR = 769
RIGHTS_HEAD = "327a5364f7729f3ffdfc2f8079d02fb7a54638d2"
SOURCE_DATASET = "common-pile/caselaw_access_project"
SOURCE_REVISION = "31e65135501af50f8285b52489bc3b39fd0fc5d5"
SOURCE_FILE = "cap_00044.jsonl.gz"
SOURCE_SHA256 = "e04bdff817b34d5fd2ab0b4aff272adfe6e863aacef7d4cd24db47e4b7c8db33"
SOURCE_BYTES = 9_441_419
SOURCE_FAMILY = "en.common-pile.caselaw"
SOURCE_FIELDS = ("Caselaw Access Project", "Court Listener", "CourtListener")
EXPECTED_FIELDS = ("id", "source", "added", "created", "metadata", "text")
LICENSE_EXACT = "Public Domain"
MAX_LINE_BYTES = 2_000_000
MAX_CANDIDATE_BYTES = 6_500_000
MIN_CANDIDATE_BYTES = 5_500_000
MAX_SCANNED_BYTES = 80_000_000
MIN_TEXT_CHARS = 200
MIN_LATIN_ALPHA_RATIO = 0.80
MAX_CYRILLIC_ALPHA_RATIO = 0.05
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ZERO_KEYS = (
    "canonical_capacity_credited",
    "family_credit_added",
    "training_authorized_bytes",
    "optimizer_updates",
)
FALSE_KEYS = (
    "upstream_public_domain_metadata_is_project_rights_authority",
    "source_level_project_review_complete",
    "post_composition_privacy_complete",
    "training_eligible",
    "evaluation_eligible",
    "tokenizer_fit_authorized",
    "model_training_executed",
    "final_test_accessed",
    "paid_compute_used",
)
REQUIRED_GATES = (
    "SOURCE_LEVEL_RIGHTS_AND_PROVENANCE_REVIEW",
    "CURRENT_GLOBAL_EXACT_NEAR_LINEAGE_DEDUP",
    "FRESH_RESERVED_EVALUATION_DECONTAMINATION",
    "POST_COMPOSITION_QUALITY_PRIVACY",
    "BALANCE_AND_FAMILY_CAPS",
    "CLUSTER_SAFE_SPLIT",
    "DETERMINISTIC_PACKING_TWO_CLEAN_BUILDS",
    "POSITIVE_UNIQUE_LOSS_LEDGER",
)


class CandidateError(RuntimeError):
    """Fail-closed source qualification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _validate_rights_registry(cfg: Mapping[str, Any]) -> dict[str, Any]:
    authority = cfg["rights_authority"]
    require(authority["pr"] == RIGHTS_PR, "rights PR drift")
    require(authority["head_sha"] == RIGHTS_HEAD, "rights head drift")
    require(authority["source_key"] == "caselaw_access_project", "rights source key drift")
    require(
        authority["required_rights_basis_class"] == "PUBLIC_DOMAIN_FILTER",
        "rights basis requirement drift",
    )
    require(authority["required_license_signal"] == "PUBLIC_DOMAIN", "rights signal drift")
    path = Path(authority["registry_path"])
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError("cannot load Common Pile rights registry") from exc
    require(
        registry.get("registry_id") == "COMMON-PILE-SOURCE-RIGHTS-V1",
        "rights registry drift",
    )
    sources = registry.get("sources")
    require(isinstance(sources, list), "rights source vector missing")
    matches = [
        row
        for row in sources
        if isinstance(row, Mapping) and row.get("key") == authority["source_key"]
    ]
    require(len(matches) == 1, "Caselaw rights source missing or ambiguous")
    source = dict(matches[0])
    require(source.get("hf_dataset") == SOURCE_DATASET, "rights dataset drift")
    require(
        source.get("rights_basis_class") == authority["required_rights_basis_class"],
        "rights basis drift",
    )
    signals = source.get("license_or_status_signals")
    require(
        isinstance(signals, list) and authority["required_license_signal"] in signals,
        "public-domain rights signal missing",
    )
    require(
        source.get("canonical_training_authorized") is False,
        "rights audit self-authorized training",
    )
    require(source.get("credited_bytes") == 0, "rights audit credited bytes")
    require(
        source.get("authorized_loss_positions") == 0,
        "rights audit credited loss positions",
    )
    require(
        source.get("evaluation_role") == "TRAINING_CANDIDATE_ONLY",
        "evaluation role drift",
    )
    require(source.get("final_test_excluded") is True, "final-test boundary drift")
    return source


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"cannot load config: {path}") from exc
    require(isinstance(cfg, dict), "config root must be object")
    require(cfg.get("schema_version") == SCHEMA, "schema drift")
    require(cfg.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE boundary drift")
    expected_source = {
        "dataset": SOURCE_DATASET,
        "revision": SOURCE_REVISION,
        "file": SOURCE_FILE,
        "url": (
            "https://huggingface.co/datasets/common-pile/caselaw_access_project/resolve/"
            f"{SOURCE_REVISION}/{SOURCE_FILE}"
        ),
        "sha256": SOURCE_SHA256,
        "bytes": SOURCE_BYTES,
        "family": SOURCE_FAMILY,
        "allowed_source_fields": list(SOURCE_FIELDS),
    }
    require(cfg["source"] == expected_source, "source identity drift")
    contract = cfg["record_contract"]
    require(contract["fields"] == list(EXPECTED_FIELDS), "record field contract drift")
    require(contract["metadata_license_exact"] == LICENSE_EXACT, "license contract drift")
    require(contract["max_json_line_bytes"] == MAX_LINE_BYTES, "line envelope drift")
    require(
        cfg["selection"]
        == {
            "max_candidate_normalized_utf8_bytes": MAX_CANDIDATE_BYTES,
            "min_candidate_normalized_utf8_bytes": MIN_CANDIDATE_BYTES,
            "max_scanned_decompressed_bytes": MAX_SCANNED_BYTES,
            "min_text_chars": MIN_TEXT_CHARS,
            "min_latin_alpha_ratio": MIN_LATIN_ALPHA_RATIO,
            "max_cyrillic_alpha_ratio": MAX_CYRILLIC_ALPHA_RATIO,
        },
        "selection policy drift",
    )
    require(
        cfg["privacy"]
        == {
            "reject_email": True,
            "reject_phone": True,
            "reject_us_ssn_shape": True,
            "reject_control_characters": True,
        },
        "privacy policy drift",
    )
    boundary = cfg["claim_boundary"]
    require(boundary.get("candidate_only") is True, "candidate-only boundary disabled")
    for key in ZERO_KEYS:
        require(boundary.get(key) == 0, f"zero-credit boundary drift: {key}")
    for key in FALSE_KEYS:
        require(boundary.get(key) is False, f"false boundary drift: {key}")
    require(
        tuple(cfg["required_downstream_gates"]) == REQUIRED_GATES,
        "required downstream gates drift",
    )
    _validate_rights_registry(cfg)
    return cfg


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def language_evidence(text: str) -> dict[str, int | float]:
    alpha = [char for char in text if char.isalpha()]
    latin = [char for char in alpha if "LATIN" in unicodedata.name(char, "")]
    cyrillic = [char for char in alpha if "CYRILLIC" in unicodedata.name(char, "")]
    denominator = max(1, len(alpha))
    return {
        "alpha_chars": len(alpha),
        "latin_alpha_ratio": round(len(latin) / denominator, 6),
        "cyrillic_alpha_ratio": round(len(cyrillic) / denominator, 6),
    }


def assess_record(
    record: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[bool, str, dict[str, Any] | None]:
    require(set(record) == set(EXPECTED_FIELDS), "record field drift")
    record_id = record.get("id")
    require(
        isinstance(record_id, str)
        and 1 <= len(record_id) <= 256
        and not CONTROL_RE.search(record_id),
        "record id invalid",
    )
    require(record.get("source") in SOURCE_FIELDS, "record source field drift")
    metadata = record.get("metadata")
    require(isinstance(metadata, Mapping), "record metadata missing")
    require(metadata.get("license") == LICENSE_EXACT, "record public-domain metadata missing")
    text = record.get("text")
    require(isinstance(text, str), "record text must be string")
    if CONTROL_RE.search(text):
        return False, "control_character", None
    if EMAIL_RE.search(text):
        return False, "email", None
    if SSN_RE.search(text):
        return False, "us_ssn_shape", None
    if PHONE_RE.search(text):
        return False, "phone", None
    normalized = normalize(text)
    if len(normalized) < cfg["selection"]["min_text_chars"]:
        return False, "too_short", None
    language = language_evidence(normalized)
    if language["latin_alpha_ratio"] < cfg["selection"]["min_latin_alpha_ratio"]:
        return False, "low_latin_ratio", None
    if language["cyrillic_alpha_ratio"] > cfg["selection"]["max_cyrillic_alpha_ratio"]:
        return False, "high_cyrillic_ratio", None
    payload = normalized.encode("utf-8")
    return True, "accepted", {
        "record_id": record_id,
        "normalized_sha256": sha256(payload),
        "normalized_utf8_bytes": len(payload),
        "text": normalized,
        "source_family": SOURCE_FAMILY,
        "source_kind": record["source"],
        "rights_basis": "UPSTREAM_PUBLIC_DOMAIN_METADATA_REVIEW_REQUIRED",
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _source_file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise CandidateError(f"cannot read source shard: {path}") from exc
    return size, digest.hexdigest()


def materialize(
    source_gzip: Path,
    candidate_jsonl: Path,
    report_path: Path,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    source_size, source_hash = _source_file_identity(source_gzip)
    require(source_size == SOURCE_BYTES, f"source byte count mismatch: {source_size}")
    require(source_hash == SOURCE_SHA256, "source SHA-256 mismatch")

    seen_ids: set[str] = set()
    seen_text_hashes: set[str] = set()
    retained: list[dict[str, Any]] = []
    dispositions: dict[str, int] = {}
    source_kind_counts: dict[str, int] = {}
    candidate_text_bytes = 0
    decompressed_bytes = 0
    rows_scanned = 0
    cap_reached = False

    try:
        with gzip.open(source_gzip, "rb") as handle:
            while True:
                line = handle.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                decompressed_bytes += len(line)
                require(len(line) <= MAX_LINE_BYTES, "JSONL line exceeds safety envelope")
                require(
                    decompressed_bytes <= MAX_SCANNED_BYTES,
                    "decompressed scan envelope exceeded",
                )
                rows_scanned += 1
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CandidateError(f"invalid UTF-8 JSONL at row {rows_scanned}") from exc
                require(isinstance(record, Mapping), "JSONL row must be object")
                raw_id = record.get("id")
                require(isinstance(raw_id, str), "record id missing")
                require(raw_id not in seen_ids, f"duplicate source record id: {raw_id}")
                seen_ids.add(raw_id)
                ok, reason, candidate = assess_record(record, cfg)
                if not ok:
                    dispositions[reason] = dispositions.get(reason, 0) + 1
                    continue
                require(candidate is not None, "accepted row missing candidate payload")
                digest = candidate["normalized_sha256"]
                if digest in seen_text_hashes:
                    dispositions["exact_normalized_duplicate"] = (
                        dispositions.get("exact_normalized_duplicate", 0) + 1
                    )
                    continue
                next_bytes = candidate_text_bytes + candidate["normalized_utf8_bytes"]
                if next_bytes > MAX_CANDIDATE_BYTES:
                    cap_reached = True
                    break
                seen_text_hashes.add(digest)
                retained.append(candidate)
                candidate_text_bytes = next_bytes
                dispositions["accepted"] = dispositions.get("accepted", 0) + 1
                kind = str(candidate["source_kind"])
                source_kind_counts[kind] = source_kind_counts.get(kind, 0) + 1
                if candidate_text_bytes >= MAX_CANDIDATE_BYTES - 8_192:
                    cap_reached = True
                    break
    except (OSError, EOFError) as exc:
        raise CandidateError("cannot decode pinned gzip shard") from exc

    require(candidate_text_bytes >= MIN_CANDIDATE_BYTES, "candidate byte floor not reached")
    require(bool(retained), "candidate retained zero rows")
    payload = b"".join(canonical(row) for row in retained)
    candidate_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_jsonl.write_bytes(payload)
    rights_source = _validate_rights_registry(cfg)
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": SOURCE_BYTES,
        "source_family": SOURCE_FAMILY,
        "rights_authority_pr": RIGHTS_PR,
        "rights_authority_head": RIGHTS_HEAD,
        "rights_source_key": rights_source["key"],
        "rights_basis_class": rights_source["rights_basis_class"],
        "project_review_status": rights_source.get("project_review_status"),
        "upstream_license_signal": LICENSE_EXACT,
        "source_rows_scanned": rows_scanned,
        "decompressed_bytes_scanned": decompressed_bytes,
        "candidate_cap_reached": cap_reached,
        "retained_records": len(retained),
        "retained_normalized_utf8_bytes": candidate_text_bytes,
        "candidate_jsonl_sha256": sha256(payload),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "disposition_counts": dict(sorted(dispositions.items())),
        "source_level_project_review_complete": False,
        "global_dedup_completed": False,
        "reserved_evaluation_decontamination_completed": False,
        "post_composition_privacy_complete": False,
        "canonical_capacity_credited": 0,
        "family_credit_added": 0,
        "training_authorized_bytes": 0,
        "training_eligible": False,
        "evaluation_eligible": False,
        "tokenizer_fit_authorized": False,
        "optimizer_updates": 0,
        "model_training_executed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "required_downstream_gates": list(REQUIRED_GATES),
        "safe_result": "COMMON_PILE_CASELAW_CANDIDATE_ONLY_ZERO_CREDIT",
    }
    report["report_identity_sha256"] = sha256(canonical(report))
    report_path.write_bytes(canonical(report))
    return report


def download_exact(cfg: Mapping[str, Any], output: Path) -> None:
    request = urllib.request.Request(
        cfg["source"]["url"],
        headers={"User-Agent": "12-6-ai-D03-CommonPile-Caselaw/1"},
    )
    partial = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    total = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial.open("wb") as destination,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                require(total <= SOURCE_BYTES, "download exceeds pinned source size")
                digest.update(chunk)
                destination.write(chunk)
        require(total == SOURCE_BYTES, f"downloaded source size mismatch: {total}")
        require(digest.hexdigest() == SOURCE_SHA256, "downloaded source SHA-256 mismatch")
        partial.replace(output)
    except OSError as exc:
        raise CandidateError("exact source download failed") from exc
    finally:
        if partial.exists():
            partial.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-gzip", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.download:
        download_exact(cfg, args.source_gzip)
    report = materialize(args.source_gzip, args.candidate_jsonl, args.report, cfg)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
