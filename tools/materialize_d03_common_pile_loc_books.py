from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twelve_six.common_pile_rights import validate_registry  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/data/d03_common_pile_loc_books_bounded_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/data/common_pile_source_rights_v1.json"
USER_AGENT = "12-6-ai-D03-LoCBooks/1.0"
EMAIL_RE = re.compile(r"[^@\s<>]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s().-]*)?"
    r"(?:\(?\d{3}\)?[\s.-]*)\d{3}[\s.-]*\d{4}(?!\d)"
)


class MaterializationError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "12-6.d03-common-pile-loc-books-bounded.v1":
        raise MaterializationError("schema drift")
    if config.get("execution_profile") != "LOCAL_FREE":
        raise MaterializationError("LOCAL_FREE boundary drift")
    if config.get("claim_issue") != 887:
        raise MaterializationError("ownership claim drift")

    rights = config.get("rights_authority", {})
    expected_rights = {
        "merged_pr": 769,
        "head_sha": "327a5364f7729f3ffdfc2f8079d02fb7a54638d2",
        "registry_path": "configs/data/common_pile_source_rights_v1.json",
        "source_key": "library_of_congress",
        "required_rights_basis_class": "PUBLIC_DOMAIN_COLLECTION",
        "required_license_signal": "PUBLIC_DOMAIN",
        "required_project_review_status": "REVIEW_REQUIRED",
    }
    if any(rights.get(key) != value for key, value in expected_rights.items()):
        raise MaterializationError("rights authority drift")

    collector = config.get("collector_authority", {})
    expected_collector = {
        "repository": "r-three/common-pile",
        "revision": "9457f04a14cb2355ab00023420369d46ffd4a395",
        "path": "sources/loc_books/books.py",
        "git_blob_sha1": "0f8633d9a0c12bd04140420d74a95d82a065119f",
        "readme_path": "sources/loc_books/README.md",
        "readme_git_blob_sha1": "c364b6dec4232e128b7adee92747f0d728090714",
    }
    if any(
        collector.get(key) != value
        for key, value in expected_collector.items()
    ):
        raise MaterializationError("collector authority drift")

    source = config.get("source", {})
    expected_source = {
        "dataset": "common-pile/library_of_congress",
        "revision": "d31bdba02cdad5104ccec2c02ae799c0bcb5a9a7",
        "file": "data/00000_loc_books.jsonl.gz",
        "sha256": "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f",
        "bytes": 358594502,
        "compression": "gzip",
        "family": "en.common-pile.library-of-congress",
        "expected_source_label": "loc_books",
        "expected_metadata_license": "Public Domain",
        "expected_metadata_language": "english",
        "minimum_metadata_year": 1500,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise MaterializationError("source authority drift")
    expected_url = (
        "https://huggingface.co/datasets/common-pile/"
        "library_of_congress/resolve/"
        "d31bdba02cdad5104ccec2c02ae799c0bcb5a9a7/"
        "data/00000_loc_books.jsonl.gz"
    )
    if source.get("url") != expected_url:
        raise MaterializationError("source URL drift")

    record = config.get("record_contract", {})
    exact_fields = ["id", "text", "source", "added", "metadata"]
    if record.get("exact_fields") != exact_fields:
        raise MaterializationError("record schema drift")
    expected_metadata = [
        "license",
        "title",
        "author",
        "year",
        "language",
        "item_url",
        "text_file_url",
    ]
    if record.get("required_metadata_fields") != expected_metadata:
        raise MaterializationError("metadata schema drift")
    if record.get("max_json_line_bytes") != 20000000:
        raise MaterializationError("JSON line bound drift")
    if record.get("require_unique_source_ids") is not True:
        raise MaterializationError("source-id uniqueness weakened")

    selection = config.get("selection", {})
    expected_selection = {
        "max_scanned_decompressed_bytes": 90000000,
        "min_candidate_normalized_utf8_bytes": 5500000,
        "max_candidate_normalized_utf8_bytes": 6500000,
        "min_text_utf8_bytes": 2000,
        "max_text_utf8_bytes": 1000000,
        "min_alphabetic_chars": 1000,
        "min_alpha_fraction": 0.55,
        "min_latin_share_of_alpha": 0.9,
        "normalization": "NFC_LF_OUTER_TRIM",
    }
    if any(
        selection.get(key) != value
        for key, value in expected_selection.items()
    ):
        raise MaterializationError("selection policy drift")

    privacy = config.get("privacy_quality", {})
    for key in (
        "reject_control_characters",
        "reject_replacement_character",
        "reject_email_like",
        "reject_phone_like",
        "deduplicate_exact_normalized_text",
    ):
        if privacy.get(key) is not True:
            raise MaterializationError(f"privacy/quality gate weakened: {key}")
    if privacy.get("universal_pii_absence_claimed") is not False:
        raise MaterializationError("universal PII claim forbidden")
    if privacy.get("ocr_gold_quality_claimed") is not False:
        raise MaterializationError("OCR gold-quality claim forbidden")

    boundary = config.get("claim_boundary", {})
    exact_boundary = {
        "candidate_only": True,
        "canonical_capacity_credited": 0,
        "family_credit_added": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "corpus_admitted": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "tokenizer_fit_authorized": False,
        "optimizer_updates": 0,
        "model_training_executed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "learned_20m_promoted": False,
    }
    if any(boundary.get(key) != value for key, value in exact_boundary.items()):
        raise MaterializationError("claim boundary drift")


def load_rights_row(
    config: dict[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    source_key = config["rights_authority"]["source_key"]
    matches = [row for row in registry["sources"] if row["key"] == source_key]
    if len(matches) != 1:
        raise MaterializationError("rights row missing or duplicated")
    row = matches[0]
    authority = config["rights_authority"]
    if row.get("hf_dataset") != config["source"]["dataset"]:
        raise MaterializationError("rights dataset mismatch")
    if row.get("rights_basis_class") != authority["required_rights_basis_class"]:
        raise MaterializationError("rights basis mismatch")
    signals = row.get("license_or_status_signals", [])
    if authority["required_license_signal"] not in signals:
        raise MaterializationError("rights signal missing")
    if row.get("project_review_status") != authority["required_project_review_status"]:
        raise MaterializationError("source review status drift")
    if row.get("canonical_training_authorized") is not False:
        raise MaterializationError("rights audit may not authorize training")
    if row.get("credited_bytes") != 0:
        raise MaterializationError("rights audit byte credit must remain zero")
    if row.get("authorized_loss_positions") != 0:
        raise MaterializationError("rights audit loss credit must remain zero")
    if row.get("final_test_excluded") is not True:
        raise MaterializationError("final-test firewall weakened")
    return row


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


def _has_forbidden_control(text: str) -> bool:
    return any(
        unicodedata.category(char) == "Cc" and char not in "\n\t"
        for char in text
    )


def _latin_share(alpha: list[str]) -> float:
    if not alpha:
        return 0.0
    latin = sum("LATIN" in unicodedata.name(char, "") for char in alpha)
    return latin / len(alpha)


def classify_text(text: str, config: dict[str, Any]) -> str | None:
    selection = config["selection"]
    privacy = config["privacy_quality"]
    raw = text.encode("utf-8")
    if len(raw) < selection["min_text_utf8_bytes"]:
        return "too_short"
    if len(raw) > selection["max_text_utf8_bytes"]:
        return "too_large"
    if privacy["reject_replacement_character"] and "\ufffd" in text:
        return "replacement_character"
    if privacy["reject_control_characters"] and _has_forbidden_control(text):
        return "control_character"
    if privacy["reject_email_like"] and EMAIL_RE.search(text):
        return "email_like"
    if privacy["reject_phone_like"] and PHONE_RE.search(text):
        return "phone_like"
    alpha = [char for char in text if char.isalpha()]
    if len(alpha) < selection["min_alphabetic_chars"]:
        return "too_few_alphabetic"
    nonspace = sum(not char.isspace() for char in text)
    if not nonspace:
        return "low_alpha_fraction"
    if len(alpha) / nonspace < selection["min_alpha_fraction"]:
        return "low_alpha_fraction"
    if _latin_share(alpha) < selection["min_latin_share_of_alpha"]:
        return "low_latin_share"
    return None


def validate_row(
    row: object,
    config: dict[str, Any],
) -> tuple[str, str] | str:
    if not isinstance(row, dict):
        return "row_not_object"
    record = config["record_contract"]
    expected_fields = set(record["exact_fields"])
    if set(row) != expected_fields or len(row) != len(expected_fields):
        return "row_schema_mismatch"
    source = config["source"]
    source_id = row.get("id")
    text = row.get("text")
    metadata = row.get("metadata")
    if not isinstance(source_id, str) or not source_id.strip():
        return "invalid_source_id"
    if not isinstance(text, str):
        return "invalid_text"
    if row.get("source") != source["expected_source_label"]:
        return "source_label_mismatch"
    if not isinstance(metadata, dict):
        return "metadata_not_object"
    required_metadata = record["required_metadata_fields"]
    if any(key not in metadata for key in required_metadata):
        return "metadata_schema_mismatch"
    if metadata.get("license") != source["expected_metadata_license"]:
        return "rights_metadata_mismatch"
    language = str(metadata.get("language", "")).lower()
    if language != source["expected_metadata_language"]:
        return "language_metadata_mismatch"
    year = metadata.get("year")
    if not isinstance(year, int) or year < source["minimum_metadata_year"]:
        return "year_metadata_mismatch"
    return source_id.strip(), normalize_text(text)


def _source_id_if_present(row: object) -> str | None:
    if not isinstance(row, dict):
        return None
    source_id = row.get("id")
    if not isinstance(source_id, str):
        return None
    source_id = source_id.strip()
    return source_id or None


def materialize_stream(
    compressed: BinaryIO,
    config: dict[str, Any],
    output: BinaryIO,
) -> dict[str, Any]:
    selection = config["selection"]
    record = config["record_contract"]
    rejected: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    inventory: list[dict[str, Any]] = []
    candidate_digest = hashlib.sha256()
    scanned_bytes = 0
    rows_seen = 0
    accepted_bytes = 0

    with gzip.GzipFile(fileobj=compressed, mode="rb") as stream:
        for raw_line in stream:
            rows_seen += 1
            scanned_bytes += len(raw_line)
            if scanned_bytes > selection["max_scanned_decompressed_bytes"]:
                break
            if len(raw_line) > record["max_json_line_bytes"]:
                rejected["json_line_too_large"] += 1
                continue
            try:
                row = json.loads(raw_line.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                rejected["invalid_json_utf8"] += 1
                continue

            source_id = _source_id_if_present(row)
            if source_id is not None:
                if source_id in seen_ids:
                    raise MaterializationError(f"duplicate source id: {source_id}")
                seen_ids.add(source_id)

            validated = validate_row(row, config)
            if isinstance(validated, str):
                rejected[validated] += 1
                continue
            validated_id, text = validated
            if source_id != validated_id:
                raise MaterializationError("source id normalization mismatch")

            reason = classify_text(text, config)
            if reason is not None:
                rejected[reason] += 1
                continue
            text_bytes = text.encode("utf-8")
            text_hash = sha256_bytes(text_bytes)
            if text_hash in seen_text:
                rejected["exact_normalized_duplicate"] += 1
                continue
            candidate_max = selection["max_candidate_normalized_utf8_bytes"]
            if accepted_bytes + len(text_bytes) > candidate_max:
                rejected["candidate_budget_exceeds"] += 1
                continue

            seen_text.add(text_hash)
            payload = {
                "language": "en",
                "modality": "natural_text",
                "record_id": f"loc:{validated_id}",
                "source_family": config["source"]["family"],
                "source_id": validated_id,
                "text": text,
                "text_sha256": text_hash,
                "text_utf8_bytes": len(text_bytes),
            }
            encoded = canonical_bytes(payload)
            output.write(encoded)
            candidate_digest.update(encoded)
            accepted_bytes += len(text_bytes)
            inventory.append(
                {
                    "record_id": payload["record_id"],
                    "text_sha256": text_hash,
                    "text_utf8_bytes": len(text_bytes),
                }
            )
            if accepted_bytes >= selection["min_candidate_normalized_utf8_bytes"]:
                break

    if accepted_bytes < selection["min_candidate_normalized_utf8_bytes"]:
        raise MaterializationError(
            "bounded scan retained only "
            f"{accepted_bytes} normalized bytes; minimum not met"
        )
    return {
        "rows_seen": rows_seen,
        "scanned_decompressed_bytes": scanned_bytes,
        "accepted_records": len(inventory),
        "accepted_normalized_utf8_bytes": accepted_bytes,
        "rejected_records": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "candidate_jsonl_sha256": candidate_digest.hexdigest(),
        "inventory_identity_sha256": sha256_bytes(canonical_bytes(inventory)),
    }


def acquire_exact(config: dict[str, Any], destination: Path) -> None:
    source = config["source"]
    request = Request(
        source["url"],
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(request, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    verify_exact_source(config, destination)


def verify_exact_source(
    config: dict[str, Any],
    path: Path,
) -> tuple[str, int]:
    digest, size = sha256_file(path)
    source = config["source"]
    if size != source["bytes"]:
        raise MaterializationError(f"source byte-count mismatch: {size}")
    if digest != source["sha256"]:
        raise MaterializationError("source sha256 mismatch")
    return digest, size


def build_report(
    config: dict[str, Any],
    rights_row: dict[str, Any],
    source_sha: str,
    source_size: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": "12-6.d03-common-pile-loc-books-report.v1",
        "decision": "LOC_BOOKS_CANDIDATE_ONLY_ZERO_CREDIT",
        "source_family": config["source"]["family"],
        "source_revision": config["source"]["revision"],
        "source_sha256": source_sha,
        "source_bytes": source_size,
        "rights_row_identity_sha256": sha256_bytes(canonical_bytes(rights_row)),
        "materialization": stats,
        "claim_boundary": config["claim_boundary"],
        "required_downstream_gates": config["required_downstream_gates"],
        "execution_profile": "LOCAL_FREE",
    }
    return {
        **core,
        "report_identity_sha256": sha256_bytes(canonical_bytes(core)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--input")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    rights_row = load_rights_row(config, Path(args.registry))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    candidate_partial = output_dir / "loc-books-candidate.jsonl.partial"
    candidate_path = output_dir / "loc-books-candidate.jsonl"
    candidate_partial.unlink(missing_ok=True)
    candidate_path.unlink(missing_ok=True)

    try:
        if args.input:
            source_path = Path(args.input)
            source_sha, source_size = verify_exact_source(config, source_path)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="d03-loc-")
            source_path = Path(temporary.name) / "source.jsonl.gz"
            acquire_exact(config, source_path)
            source_sha, source_size = verify_exact_source(config, source_path)

        try:
            with source_path.open("rb") as compressed:
                with candidate_partial.open("wb") as candidate:
                    stats = materialize_stream(compressed, config, candidate)
            candidate_partial.replace(candidate_path)
        except Exception:
            candidate_partial.unlink(missing_ok=True)
            raise

        report = build_report(
            config,
            rights_row,
            source_sha,
            source_size,
            stats,
        )
        report_path = output_dir / "loc-books-report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
