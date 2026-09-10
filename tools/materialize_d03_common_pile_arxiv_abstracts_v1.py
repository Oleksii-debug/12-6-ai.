#!/usr/bin/env python3
"""Bounded, zero-credit intake for one immutable Common Pile ArXiv-abstract shard."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from twelve_six.common_pile_rights import load_and_validate as load_common_pile_rights

CONFIG = REPO_ROOT / "configs/data/d03_common_pile_arxiv_abstracts_bounded_v1.json"
SCHEMA = "12-6.d03-common-pile-arxiv-abstracts-bounded.v1"
REPORT_SCHEMA = "12-6.d03-common-pile-arxiv-abstracts-bounded-report.v1"
SOURCE_REVISION = "46de78c48636c0b46f60049dfd1c5a3710d233f9"
SOURCE_FILE = "00003_arxiv-abstracts.jsonl.gz"
SOURCE_SHA256 = "3d781bf7617fd9a6840211227c6c8831280dba457a6d9c79a2a5a357e025a8b7"
SOURCE_BYTES = 232_616_926
SOURCE_URL = (
    "https://huggingface.co/datasets/common-pile/arxiv_abstracts/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_FILE}"
)
SOURCE_LABEL = "arxiv-abstracts"
EXPECTED_FIELDS = ("id", "text", "source", "created", "added", "metadata")
EXPECTED_LICENSE = (
    "Creative Commons Zero - Public Domain - "
    "https://creativecommons.org/publicdomain/zero/1.0/"
)
MAX_JSONL_LINE_BYTES = 2 * 1024 * 1024

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ArxivAbstractsIntakeError(RuntimeError):
    """Fail-closed error for immutable-source or structural contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArxivAbstractsIntakeError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ArxivAbstractsIntakeError(f"cannot read source shard: {path}") from exc
    return size, digest.hexdigest()


def _rights_registry_path(cfg: Mapping[str, Any]) -> Path:
    raw = cfg["parent_rights"]["registry_path"]
    _require(isinstance(raw, str) and raw, "rights registry path missing")
    path = (REPO_ROOT / raw).resolve()
    _require(path.is_relative_to(REPO_ROOT.resolve()), "rights registry escaped repo root")
    return path


def _validate_parent_rights(cfg: Mapping[str, Any]) -> str:
    rights = load_common_pile_rights(_rights_registry_path(cfg))
    rows = rights.get("sources")
    _require(isinstance(rows, list), "validated rights registry lost source rows")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("key") == cfg["parent_rights"]["source_key"]
    ]
    _require(len(matches) == 1, "ArXiv abstracts rights row is not unique")
    row = matches[0]
    parent = cfg["parent_rights"]
    _require(row.get("hf_dataset") == parent["hf_dataset"], "rights dataset identity drift")
    _require(
        row.get("rights_basis_class") == parent["required_rights_basis_class"],
        "rights basis drift",
    )
    signals = row.get("license_or_status_signals")
    _require(
        isinstance(signals, list) and parent["required_rights_signal"] in signals,
        "required CC0 rights signal missing",
    )
    _require(
        row.get("project_review_status") == parent["required_project_review_status"],
        "rights review status drift",
    )
    _require(
        row.get("canonical_training_authorized") is False,
        "rights audit may not authorize training",
    )
    _require(row.get("credited_bytes") == 0, "rights audit may not credit bytes")
    _require(
        row.get("authorized_loss_positions") == 0,
        "rights audit may not authorize loss positions",
    )
    _require(row.get("final_test_excluded") is True, "final-test firewall weakened")
    identity = rights.get("registry_identity_sha256")
    _require(
        isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity) is not None,
        "validated rights registry identity missing",
    )
    return identity


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArxivAbstractsIntakeError(f"cannot load config: {path}") from exc
    _require(isinstance(cfg, dict), "config root must be an object")
    _require(cfg.get("schema_version") == SCHEMA, "config schema drift")
    _require(cfg.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    parent = cfg.get("parent_rights")
    _require(isinstance(parent, Mapping), "parent rights binding missing")
    expected_parent = {
        "merged_pr": 769,
        "registry_path": "configs/data/common_pile_source_rights_v1.json",
        "source_key": "arxiv_abstracts",
        "hf_dataset": "common-pile/arxiv_abstracts",
        "required_rights_basis_class": "METADATA_OPEN_LICENSE",
        "required_rights_signal": "CC0",
        "required_project_review_status": "REVIEW_REQUIRED",
    }
    _require(dict(parent) == expected_parent, "parent rights binding drift")

    source = cfg.get("source")
    _require(isinstance(source, Mapping), "source contract missing")
    expected_source = {
        "revision": SOURCE_REVISION,
        "file": SOURCE_FILE,
        "url": SOURCE_URL,
        "sha256": SOURCE_SHA256,
        "bytes": SOURCE_BYTES,
        "compression": "gzip",
        "expected_source_label": SOURCE_LABEL,
        "expected_row_fields": list(EXPECTED_FIELDS),
        "expected_metadata_license": EXPECTED_LICENSE,
    }
    _require(dict(source) == expected_source, "immutable source contract drift")

    selection = cfg.get("selection")
    _require(isinstance(selection, Mapping), "selection contract missing")
    _require(selection.get("max_scanned_records") == 4096, "scan bound drift")
    _require(selection.get("max_retained_records") == 1024, "retain bound drift")
    _require(
        selection.get("order") == "exact_shard_file_order_after_fail_closed_filters",
        "selection order drift",
    )

    quality = cfg.get("quality")
    _require(isinstance(quality, Mapping), "quality contract missing")
    _require(quality.get("min_chars") == 200, "min_chars drift")
    _require(quality.get("max_chars") == 12000, "max_chars drift")
    _require(quality.get("min_alpha_fraction") == 0.35, "alpha threshold drift")

    privacy = cfg.get("privacy")
    _require(isinstance(privacy, Mapping), "privacy contract missing")
    for key in ("reject_email", "reject_phone", "reject_control_characters"):
        _require(privacy.get(key) is True, f"{key} must remain enabled")
    _require(
        privacy.get("universal_pii_absence_claimed") is False,
        "bounded intake may not claim universal PII absence",
    )

    required_downstream = [
        "global_cross_source_dedup",
        "reserved_evaluation_decontamination",
        "final_source_rights_review",
        "quality_privacy_balance_family_caps",
        "cluster_safe_split",
        "deterministic_tokenizer_packing",
        "positive_unique_loss_ledger",
    ]
    _require(cfg.get("downstream_required") == required_downstream, "downstream gates drift")

    boundary = cfg.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "claim boundary missing")
    for key in (
        "training_authorized_bytes",
        "canonical_capacity_credited",
        "family_credit_added",
        "unique_causal_loss_positions_authorized",
        "optimizer_updates",
    ):
        _require(boundary.get(key) == 0, f"{key} must remain zero")
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_accessed",
        "paid_compute_used",
        "research_corpus_v1_released",
    ):
        _require(boundary.get(key) is False, f"{key} must remain false")

    _validate_parent_rights(cfg)
    return cfg


def download_exact(cfg: Mapping[str, Any], output: Path) -> None:
    source = cfg["source"]
    _require(source["url"] == SOURCE_URL, "download URL drift")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "User-Agent": "12-6-ai-D03-common-pile-arxiv-abstracts/1",
            "Accept": "application/gzip,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    )
    partial = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    total = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial.open("wb") as target,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                _require(total <= SOURCE_BYTES, "download exceeded pinned byte count")
                digest.update(chunk)
                target.write(chunk)
        _require(total == SOURCE_BYTES, f"source byte count mismatch: {total}")
        _require(digest.hexdigest() == SOURCE_SHA256, "source SHA-256 mismatch")
        partial.replace(output)
    except OSError as exc:
        raise ArxivAbstractsIntakeError("exact source download failed") from exc
    finally:
        if partial.exists():
            partial.unlink()


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in value.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _validate_structural_row(row: Mapping[str, Any]) -> None:
    _require(set(row) == set(EXPECTED_FIELDS), "source row field drift")
    _require(
        isinstance(row.get("id"), str) and 1 <= len(row["id"]) <= 64,
        "source record id invalid",
    )
    _require(isinstance(row.get("text"), str), "source text is not a string")
    _require(isinstance(row.get("source"), str), "source label is not a string")
    _require(isinstance(row.get("created"), str), "created field is not a string")
    _require(isinstance(row.get("added"), str), "added field is not a string")
    _require(isinstance(row.get("metadata"), Mapping), "metadata field is not an object")


def assess_row(
    row: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[bool, str, str]:
    _validate_structural_row(row)

    if row["source"] != SOURCE_LABEL:
        return False, "source_label_mismatch", ""
    metadata = row["metadata"]
    if metadata.get("license") != EXPECTED_LICENSE:
        return False, "metadata_license_mismatch", ""

    text = row["text"]
    if CONTROL_RE.search(text):
        return False, "control_character", ""
    if EMAIL_RE.search(text):
        return False, "email", ""
    if PHONE_RE.search(text):
        return False, "phone", ""

    normalized = normalize(text)
    quality = cfg["quality"]
    if len(normalized) < quality["min_chars"]:
        return False, "too_short", ""
    if len(normalized) > quality["max_chars"]:
        return False, "too_long", ""
    alpha = sum(character.isalpha() for character in normalized)
    if alpha / max(1, len(normalized)) < quality["min_alpha_fraction"]:
        return False, "low_alpha_fraction", ""
    return True, "accepted", normalized


def _iter_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with gzip.open(path, "rb") as handle:
            while True:
                raw = handle.readline(MAX_JSONL_LINE_BYTES + 1)
                if not raw:
                    break
                _require(
                    len(raw) <= MAX_JSONL_LINE_BYTES,
                    "source JSONL line exceeds safety cap",
                )
                try:
                    text = raw.decode("utf-8", errors="strict")
                    row = json.loads(text)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ArxivAbstractsIntakeError("source JSONL is malformed") from exc
                _require(isinstance(row, dict), "source JSONL row must be an object")
                yield row
    except (OSError, EOFError) as exc:
        raise ArxivAbstractsIntakeError("cannot stream exact gzip shard") from exc


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    scan_cap = int(cfg["selection"]["max_scanned_records"])
    retain_cap = int(cfg["selection"]["max_retained_records"])
    _require(scan_cap > 0 and retain_cap > 0 and retain_cap <= scan_cap, "invalid selection caps")

    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    scanned = 0

    for row in rows:
        if scanned >= scan_cap:
            break
        scanned += 1
        ok, reason, normalized = assess_row(row, cfg)
        if not ok:
            reasons[reason] += 1
            continue

        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            reasons["exact_normalized_duplicate"] += 1
            continue
        seen_hashes.add(digest)

        if len(accepted) >= retain_cap:
            reasons["retained_cap_reached"] += 1
            continue

        accepted.append(
            {
                "record_id": row["id"],
                "source_key": "arxiv_abstracts",
                "source_label": SOURCE_LABEL,
                "normalized_sha256": digest,
                "normalized_bytes": len(normalized.encode("utf-8")),
                "text": normalized,
            }
        )
        reasons["accepted"] += 1

    _require(scanned > 0, "source shard yielded zero scanned rows")
    _require(sum(reasons.values()) == scanned, "row disposition accounting mismatch")
    _require(accepted, "bounded filters retained zero candidate rows")
    return accepted, reasons, scanned


def build_report(
    cfg: Mapping[str, Any],
    *,
    rights_registry_identity: str,
    accepted: list[dict[str, Any]],
    reasons: Counter[str],
    scanned: int,
    candidate_payload: bytes,
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "parent_rights_pr": 769,
        "rights_registry_identity_sha256": rights_registry_identity,
        "source_key": "arxiv_abstracts",
        "hf_dataset": "common-pile/arxiv_abstracts",
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": SOURCE_BYTES,
        "rows_scanned": scanned,
        "retained_records": len(accepted),
        "retained_normalized_bytes": sum(row["normalized_bytes"] for row in accepted),
        "candidate_jsonl_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "disposition_counts": dict(sorted(reasons.items())),
        "source_level_rights_basis_checked": True,
        "per_record_cc0_metadata_required": True,
        "universal_pii_absence_claimed": False,
        "rejected_text_emitted": False,
        "rejected_hashes_emitted": False,
        "training_authorized_bytes": 0,
        "canonical_capacity_credited": 0,
        "family_credit_added": 0,
        "unique_causal_loss_positions_authorized": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates": 0,
        "model_training_executed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "research_corpus_v1_released": False,
        "downstream_required": list(cfg["downstream_required"]),
        "safe_result": (
            "COMMON_PILE_ARXIV_ABSTRACTS_BOUNDED_CANDIDATE_ONLY_"
            "ZERO_CREDIT_DOWNSTREAM_GATES_REQUIRED"
        ),
    }
    return {
        **core,
        "report_identity_sha256": hashlib.sha256(_canonical(core)).hexdigest(),
    }


def materialize(
    shard_path: Path,
    output_jsonl: Path,
    report_path: Path,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    size, digest = _sha256_file(shard_path)
    _require(size == SOURCE_BYTES, "local source byte count mismatch")
    _require(digest == SOURCE_SHA256, "local source SHA-256 mismatch")
    rights_identity = _validate_parent_rights(cfg)

    accepted, reasons, scanned = select_rows(_iter_gzip_jsonl(shard_path), cfg)
    candidate_payload = b"".join(_canonical(row) + b"\n" for row in accepted)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_bytes(candidate_payload)
    report = build_report(
        cfg,
        rights_registry_identity=rights_identity,
        accepted=accepted,
        reasons=reasons,
        scanned=scanned,
        candidate_payload=candidate_payload,
    )
    report_path.write_bytes(_canonical(report) + b"\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.download:
        download_exact(cfg, args.shard)
    report = materialize(args.shard, args.output_jsonl, args.report, cfg)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())