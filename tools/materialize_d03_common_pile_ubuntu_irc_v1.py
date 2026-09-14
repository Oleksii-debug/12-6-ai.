#!/usr/bin/env python3
"""Bounded, zero-credit intake for one immutable Common Pile Ubuntu IRC shard."""

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

CONFIG = REPO_ROOT / "configs/data/d03_common_pile_ubuntu_irc_bounded_v1.json"
SCHEMA = "12-6.d03-common-pile-ubuntu-irc-bounded.v1"
REPORT_SCHEMA = "12-6.d03-common-pile-ubuntu-irc-bounded-report.v1"
SOURCE_REVISION = "47d55b0534a62bf0766c621297165451969f3de9"
SOURCE_FILE = "v0/documents/00007_ubuntu.jsonl.gz"
SOURCE_SHA256 = "75e38bffcaceb00ed9ce9a63b1d9e74a70f5582b3e3f763a7ec573c7fd6c1e60"
SOURCE_MAX_BYTES = 180_000_000
SOURCE_URL = (
    "https://huggingface.co/datasets/common-pile/ubuntu_irc/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_FILE}"
)
SOURCE_LABEL = "ubuntu-chat"
EXPECTED_LICENSE = "Public Domain"
METADATA_URL_PREFIX = "https://irclogs.ubuntu.com/"
EXPECTED_FIELDS = ("id", "text", "source", "added", "created", "metadata")
MAX_JSONL_LINE_BYTES = 5 * 1024 * 1024

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")
IPV4_RE = re.compile(
    r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.I)
BEARER_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{12,}", re.I)
CREDENTIAL_RE = re.compile(
    r"\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|token)"
    r"\s*[:=]\s*[^\s]{6,}",
    re.I,
)
RECORD_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-#.{1,80}$")


class UbuntuIrcIntakeError(RuntimeError):
    """Fail-closed error for immutable-source or structural contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UbuntuIrcIntakeError(message)


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
                _require(size <= SOURCE_MAX_BYTES, "source exceeds compressed safety cap")
                digest.update(chunk)
    except OSError as exc:
        raise UbuntuIrcIntakeError(f"cannot read source shard: {path}") from exc
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
    key = cfg["parent_rights"]["source_key"]
    matches = [
        row for row in rows if isinstance(row, Mapping) and row.get("key") == key
    ]
    _require(len(matches) == 1, "Ubuntu IRC rights row is not unique")
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
        "required public-domain rights signal missing",
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
        raise UbuntuIrcIntakeError(f"cannot load config: {path}") from exc
    _require(isinstance(cfg, dict), "config root must be an object")
    _require(cfg.get("schema_version") == SCHEMA, "config schema drift")
    _require(cfg.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    _require(cfg.get("claim_issue") == 905, "claim issue drift")

    parent = cfg.get("parent_rights")
    _require(isinstance(parent, Mapping), "parent rights binding missing")
    expected_parent = {
        "merged_pr": 769,
        "registry_path": "configs/data/common_pile_source_rights_v1.json",
        "source_key": "ubuntu_irc",
        "hf_dataset": "common-pile/ubuntu_irc",
        "required_rights_basis_class": "PUBLIC_DOMAIN_ARCHIVE",
        "required_rights_signal": "PUBLIC_DOMAIN",
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
        "max_compressed_bytes": SOURCE_MAX_BYTES,
        "compression": "gzip",
        "expected_source_label": SOURCE_LABEL,
        "expected_row_fields": list(EXPECTED_FIELDS),
        "expected_metadata_license": EXPECTED_LICENSE,
        "required_metadata_url_prefix": METADATA_URL_PREFIX,
    }
    _require(dict(source) == expected_source, "immutable source contract drift")

    selection = cfg.get("selection")
    _require(isinstance(selection, Mapping), "selection contract missing")
    _require(selection.get("max_scanned_records") == 4096, "scan bound drift")
    _require(selection.get("max_retained_records") == 1024, "retain bound drift")
    _require(
        selection.get("max_retained_normalized_utf8_bytes") == 4_800_000,
        "retained byte cap drift",
    )
    _require(
        selection.get("order") == "exact_shard_file_order_after_fail_closed_filters",
        "selection order drift",
    )

    quality = cfg.get("quality")
    _require(isinstance(quality, Mapping), "quality contract missing")
    expected_quality = {
        "min_chars": 300,
        "max_chars": 30000,
        "min_alpha_fraction": 0.30,
        "min_latin_alpha_ratio": 0.75,
        "max_cyrillic_alpha_ratio": 0.05,
    }
    _require(dict(quality) == expected_quality, "quality contract drift")

    privacy = cfg.get("privacy")
    _require(isinstance(privacy, Mapping), "privacy contract missing")
    for key in (
        "reject_email",
        "reject_phone",
        "reject_ipv4",
        "reject_secret_markers",
        "reject_control_characters",
    ):
        _require(privacy.get(key) is True, f"{key} must remain enabled")
    for key in (
        "metadata_authors_emitted",
        "metadata_url_emitted",
        "metadata_channel_emitted",
        "universal_pii_absence_claimed",
    ):
        _require(privacy.get(key) is False, f"{key} must remain false")

    required_downstream = [
        "global_cross_source_dedup",
        "reserved_evaluation_decontamination",
        "final_source_rights_review",
        "post_composition_quality_privacy",
        "balance_and_family_caps",
        "cluster_safe_split",
        "deterministic_packing_two_clean_builds",
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
        "training_eligible",
        "evaluation_eligible",
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
            "User-Agent": "12-6-ai-D03-common-pile-ubuntu-irc/1",
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
                _require(total <= SOURCE_MAX_BYTES, "download exceeded compressed safety cap")
                digest.update(chunk)
                target.write(chunk)
        _require(digest.hexdigest() == SOURCE_SHA256, "source SHA-256 mismatch")
        partial.replace(output)
    except OSError as exc:
        raise UbuntuIrcIntakeError("exact source download failed") from exc
    finally:
        if partial.exists():
            partial.unlink()


def verify_exact_source(path: Path) -> int:
    size, digest = _sha256_file(path)
    _require(digest == SOURCE_SHA256, "source SHA-256 mismatch")
    return size


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in value.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _validate_structural_row(row: Mapping[str, Any]) -> None:
    _require(set(row) == set(EXPECTED_FIELDS), "source row field drift")
    record_id = row.get("id")
    _require(
        isinstance(record_id, str) and RECORD_ID_RE.fullmatch(record_id) is not None,
        "source record id invalid",
    )
    _require(isinstance(row.get("text"), str), "source text is not a string")
    _require(isinstance(row.get("source"), str), "source label is not a string")
    _require(isinstance(row.get("created"), str), "created field is not a string")
    _require(isinstance(row.get("added"), str), "added field is not a string")
    _require(isinstance(row.get("metadata"), Mapping), "metadata field is not an object")


def _metadata_matches(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("license") != EXPECTED_LICENSE:
        return False
    url = metadata.get("url")
    channel = metadata.get("channel")
    authors = metadata.get("authors")
    if not isinstance(url, str) or not url.startswith(METADATA_URL_PREFIX):
        return False
    if not url.endswith(".txt"):
        return False
    if not isinstance(channel, str) or not channel.startswith("#"):
        return False
    if not isinstance(authors, list) or not all(isinstance(item, str) for item in authors):
        return False
    return True


def _script_ratios(text: str) -> tuple[float, float]:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0, 0.0
    latin = sum("LATIN" in unicodedata.name(character, "") for character in letters)
    cyrillic = sum("CYRILLIC" in unicodedata.name(character, "") for character in letters)
    return latin / len(letters), cyrillic / len(letters)


def assess_row(
    row: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[bool, str, str]:
    _validate_structural_row(row)
    if row["source"] != SOURCE_LABEL:
        return False, "source_label_mismatch", ""
    metadata = row["metadata"]
    if not _metadata_matches(metadata):
        return False, "metadata_rights_or_origin_mismatch", ""

    text = row["text"]
    if CONTROL_RE.search(text):
        return False, "control_character", ""
    if EMAIL_RE.search(text):
        return False, "email", ""
    if IPV4_RE.search(text):
        return False, "ipv4", ""
    if PHONE_RE.search(text):
        return False, "phone", ""
    if PRIVATE_KEY_RE.search(text) or BEARER_RE.search(text) or CREDENTIAL_RE.search(text):
        return False, "secret_marker", ""

    normalized = normalize(text)
    quality = cfg["quality"]
    if len(normalized) < quality["min_chars"]:
        return False, "too_short", ""
    if len(normalized) > quality["max_chars"]:
        return False, "too_long", ""
    alpha = sum(character.isalpha() for character in normalized)
    if alpha / max(1, len(normalized)) < quality["min_alpha_fraction"]:
        return False, "low_alpha_fraction", ""
    latin_ratio, cyrillic_ratio = _script_ratios(normalized)
    if latin_ratio < quality["min_latin_alpha_ratio"]:
        return False, "low_latin_alpha_ratio", ""
    if cyrillic_ratio > quality["max_cyrillic_alpha_ratio"]:
        return False, "high_cyrillic_alpha_ratio", ""
    return True, "accepted", normalized


def _iter_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with gzip.open(path, "rb") as handle:
            while True:
                raw = handle.readline(MAX_JSONL_LINE_BYTES + 1)
                if not raw:
                    break
                _require(len(raw) <= MAX_JSONL_LINE_BYTES, "source JSONL line exceeds safety cap")
                try:
                    row = json.loads(raw.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise UbuntuIrcIntakeError("source JSONL is malformed") from exc
                _require(isinstance(row, dict), "source JSONL row must be an object")
                yield row
    except (OSError, EOFError) as exc:
        raise UbuntuIrcIntakeError("cannot stream exact gzip shard") from exc


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    scan_cap = int(cfg["selection"]["max_scanned_records"])
    retain_cap = int(cfg["selection"]["max_retained_records"])
    byte_cap = int(cfg["selection"]["max_retained_normalized_utf8_bytes"])
    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_text: set[str] = set()
    seen_ids: set[str] = set()
    retained_bytes = 0
    scanned = 0

    for row in rows:
        if scanned >= scan_cap:
            break
        scanned += 1
        record_id = row.get("id")
        if isinstance(record_id, str) and record_id in seen_ids:
            raise UbuntuIrcIntakeError("duplicate source record id")
        if isinstance(record_id, str):
            seen_ids.add(record_id)

        ok, reason, text = assess_row(row, cfg)
        if not ok:
            reasons[reason] += 1
            continue
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_sha in seen_text:
            reasons["exact_normalized_duplicate"] += 1
            continue
        seen_text.add(text_sha)
        text_bytes = len(text.encode("utf-8"))
        if len(accepted) >= retain_cap:
            reasons["retained_record_cap_reached"] += 1
            continue
        if retained_bytes + text_bytes > byte_cap:
            reasons["retained_byte_cap_reached"] += 1
            continue
        accepted.append(
            {
                "record_id": row["id"],
                "source_key": "ubuntu_irc",
                "source_label": SOURCE_LABEL,
                "normalized_sha256": text_sha,
                "normalized_bytes": text_bytes,
                "training_eligible": False,
                "evaluation_eligible": False,
                "text": text,
            }
        )
        retained_bytes += text_bytes
        reasons["accepted"] += 1

    _require(sum(reasons.values()) == scanned, "selection accounting mismatch")
    return accepted, reasons, scanned


def candidate_payload(accepted: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [_canonical(dict(row)) for row in accepted]
    return b"".join(line + b"\n" for line in lines)


def build_report(
    cfg: Mapping[str, Any],
    *,
    rights_registry_identity: str,
    accepted: list[Mapping[str, Any]],
    reasons: Counter[str],
    scanned: int,
    candidate: bytes,
    observed_source_bytes: int,
) -> dict[str, Any]:
    boundary = cfg["claim_boundary"]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "COMMON_PILE_UBUNTU_IRC_CANDIDATE_ONLY_ZERO_CREDIT",
        "claim_issue": 905,
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "observed_source_bytes": observed_source_bytes,
        "rights_registry_identity_sha256": rights_registry_identity,
        "scanned_records": scanned,
        "retained_records": len(accepted),
        "retained_normalized_bytes": sum(int(row["normalized_bytes"]) for row in accepted),
        "selection_reasons": dict(sorted(reasons.items())),
        "candidate_payload_sha256": hashlib.sha256(candidate).hexdigest(),
        "candidate_payload_bytes": len(candidate),
        "durable_report_contains_source_text": False,
        "metadata_authors_emitted": False,
        "metadata_url_emitted": False,
        "metadata_channel_emitted": False,
        "universal_pii_absence_claimed": False,
        "downstream_required": list(cfg["downstream_required"]),
        "training_authorized_bytes": boundary["training_authorized_bytes"],
        "canonical_capacity_credited": boundary["canonical_capacity_credited"],
        "family_credit_added": boundary["family_credit_added"],
        "unique_causal_loss_positions_authorized": boundary[
            "unique_causal_loss_positions_authorized"
        ],
        "training_eligible": boundary["training_eligible"],
        "evaluation_eligible": boundary["evaluation_eligible"],
        "tokenizer_fit_authorized": boundary["tokenizer_fit_authorized"],
        "optimizer_updates": boundary["optimizer_updates"],
        "model_training_executed": boundary["model_training_executed"],
        "final_test_accessed": boundary["final_test_accessed"],
        "paid_compute_used": boundary["paid_compute_used"],
        "research_corpus_v1_released": boundary["research_corpus_v1_released"],
    }


def write_outputs(
    accepted: list[Mapping[str, Any]],
    report: Mapping[str, Any],
    output_jsonl: Path,
    report_path: Path,
) -> None:
    payload = candidate_payload(accepted)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_bytes(payload)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def run(
    shard: Path,
    output_jsonl: Path,
    report_path: Path,
    *,
    download: bool,
) -> dict[str, Any]:
    cfg = load_config()
    if download:
        download_exact(cfg, shard)
    observed_source_bytes = verify_exact_source(shard)
    rights_identity = _validate_parent_rights(cfg)
    accepted, reasons, scanned = select_rows(_iter_gzip_jsonl(shard), cfg)
    _require(bool(accepted), "bounded source intake retained zero records")
    candidate = candidate_payload(accepted)
    report = build_report(
        cfg,
        rights_registry_identity=rights_identity,
        accepted=accepted,
        reasons=reasons,
        scanned=scanned,
        candidate=candidate,
        observed_source_bytes=observed_source_bytes,
    )
    write_outputs(accepted, report, output_jsonl, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(
            args.shard,
            args.output_jsonl,
            args.report,
            download=args.download,
        )
    except UbuntuIrcIntakeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
