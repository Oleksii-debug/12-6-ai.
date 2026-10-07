from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_CONFIG = Path("configs/data/d03_franko1901_exact_materialization_v1.json")
USER_AGENT = "12-6-ai-D03-Franko1901/1.0"
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"[^@\s<>]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def fetch_bounded(url: str, max_bytes: int) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    )
    with urlopen(request, timeout=60) as response:
        final_url = response.geturl()
        if final_url != url:
            raise RuntimeError(f"unexpected redirect: {final_url}")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise RuntimeError("object exceeds configured bound before read")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError("object exceeds configured bound")
    return payload


def normalize_source_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def _has_forbidden_control(text: str) -> bool:
    for char in text:
        category = unicodedata.category(char)
        if category == "Cc" and char not in "\t\n\r":
            return True
    return False


def classify_text(text: str, policy: dict[str, object]) -> str | None:
    if not text:
        return "empty"
    if "\ufffd" in text and policy["reject_replacement_character"] is True:
        return "replacement_character"
    if policy["reject_control_characters"] is True and _has_forbidden_control(text):
        return "control_character"
    encoded = text.encode()
    if len(encoded) > int(policy["max_text_utf8_bytes"]):
        return "oversize_text"
    alpha = [char for char in text if char.isalpha()]
    if len(alpha) < int(policy["min_alphabetic_chars"]):
        return "too_few_alphabetic"
    cyrillic = [char for char in alpha if "\u0400" <= char <= "\u052f"]
    if len(cyrillic) / len(alpha) < float(policy["min_cyrillic_share_of_alpha"]):
        return "low_cyrillic_share"
    if policy["reject_url_like"] is True and URL_RE.search(text):
        return "url_like"
    if policy["reject_email_like"] is True and EMAIL_RE.search(text):
        return "email_like"
    return None


def _exact_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        actual_dict = actual
        expected_dict = expected
        if set(actual_dict) != set(expected_dict):
            return False
        return all(
            _exact_value(actual_dict[key], expected_dict[key])
            for key in expected_dict
        )
    if type(expected) is list:
        actual_list = actual
        expected_list = expected
        return len(actual_list) == len(expected_list) and all(
            _exact_value(left, right)
            for left, right in zip(actual_list, expected_list, strict=True)
        )
    return actual == expected


def _require_exact_mapping(
    name: str,
    actual: object,
    expected: dict[str, object],
) -> dict[str, object]:
    if type(actual) is not dict:
        raise RuntimeError(f"{name} must be an exact object")
    if set(actual) != set(expected):
        raise RuntimeError(f"{name} key-set drift")
    for key, expected_value in expected.items():
        if not _exact_value(actual[key], expected_value):
            raise RuntimeError(f"{name} drift: {key}")
    return actual


def validate_truth_boundary(config: dict[str, object]) -> None:
    expected_truth = {
        "candidate_only": True,
        "canonical_capacity_credit_bytes": 0,
        "family_credit_authorized": False,
        "corpus_admitted": False,
        "global_dedup": "NOT_RUN",
        "evaluation_decontamination": "NOT_RUN",
        "post_composition_quality_privacy": "NOT_RUN",
        "balance_family_caps": "NOT_RUN_FOR_THIS_ADDITION",
        "tokenizer_fit_authorized": False,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "learned_20m_promoted": False,
    }
    _require_exact_mapping(
        "truth boundary",
        config.get("truth_boundary"),
        expected_truth,
    )


def validate_contract(config: dict[str, object]) -> None:
    expected_top_level_keys = {
        "schema_version",
        "worker_id",
        "local_free_only",
        "base_main_sha",
        "source",
        "corroborating_verba_authority",
        "historical_work",
        "rights_boundary",
        "acquisition",
        "filter",
        "truth_boundary",
    }
    if set(config) != expected_top_level_keys:
        raise RuntimeError("top-level contract key-set drift")
    if config.get("schema_version") != "12-6.d03-franko1901-exact-materialization.v1":
        raise RuntimeError("schema version drift")
    if config.get("worker_id") != "D03-FRANKO1901-EXACT-MATERIALIZATION-20260907":
        raise RuntimeError("worker identity drift")
    if config.get("local_free_only") is not True:
        raise RuntimeError("LOCAL_FREE boundary is not asserted")
    if config.get("base_main_sha") != "56d708415e7f7a2ec72d4f80f1d571abdff428e3":
        raise RuntimeError("base-main authority drift")

    expected_source = {
        "source_id": "ua.verba.franko1901",
        "source_family": "ua.verba.public-domain.franko1901",
        "language": "uk",
        "modality": "natural_text",
        "upstream_repository": "MurzikVasilyevich/ukr-proverbs-franko",
        "upstream_revision": "7f62a9d8f0673d325b0a565d508461f4b44dae5b",
        "source_path": "franko.csv",
        "source_git_blob_sha1": "45f33ac620907e1d1ed727524975b3a0fd1a0994",
        "source_bytes": 6225761,
        "license_id": "CC0-1.0",
        "license_path": "LICENSE",
        "license_git_blob_sha1": "0e259d42c996742e9e3cba14c677129b2c1b6311",
        "raw_url": (
            "https://raw.githubusercontent.com/MurzikVasilyevich/"
            "ukr-proverbs-franko/7f62a9d8f0673d325b0a565d508461f4b44dae5b/"
            "franko.csv"
        ),
        "license_url": (
            "https://raw.githubusercontent.com/MurzikVasilyevich/"
            "ukr-proverbs-franko/7f62a9d8f0673d325b0a565d508461f4b44dae5b/"
            "LICENSE"
        ),
        "commit_message": (
            "Simplifying structure / Reducing amount of columns and leaving only "
            "clear proverbs"
        ),
    }
    source = _require_exact_mapping(
        "source authority",
        config.get("source"),
        expected_source,
    )

    expected_corroborating = {
        "repository": "dmytro-yemelianov/verbacorpus",
        "revision": "34a2c10ac35e1febad6c270a88fc8b83790407da",
        "vendored_source_path": "data/sources/franko.csv",
        "vendored_source_git_blob_sha1": source["source_git_blob_sha1"],
        "adapter_path": "adapters/franko.py",
        "adapter_git_blob_sha1": "840f4970367879eff718def3842fc065614c8b25",
        "datacard_path": "DATACARD.md",
        "datacard_git_blob_sha1": "3c880b7b6f891c4b0b6aa8ced367bbd5f7280fe3",
        "source_registry_path": "sources.csv",
        "source_registry_git_blob_sha1": "6d1ec72fb41f580557adf91152f6b0c9d08937b5",
        "same_source_blob_as_primary": True,
        "data_card_classification": "EXISTING_DIGITAL_TRANSCRIPTION",
        "data_card_text_boundary": "VERBATIM_SOURCE_ORTHOGRAPHY_NEVER_MODIFIED",
    }
    _require_exact_mapping(
        "corroborating Verba authority",
        config.get("corroborating_verba_authority"),
        expected_corroborating,
    )

    expected_historical_work = {
        "creator": "Ivan Franko",
        "title": "Halytsko-ruski narodni prypovidky",
        "publication_year_start": 1901,
        "publication_year_end": 1910,
        "creator_death_year": 1916,
        "verba_source_registry_note": "public_domain",
    }
    _require_exact_mapping(
        "historical-work authority",
        config.get("historical_work"),
        expected_historical_work,
    )

    expected_rights = {
        "historical_source_text": "PUBLIC_DOMAIN",
        "primary_dataset_layer": "CC0-1.0",
        "primary_dataset_license_git_blob_sha1": source["license_git_blob_sha1"],
        "verba_compilation_enrichment_layer": "CC-BY-4.0_CORROBORATING_ONLY",
        "required_attribution": (
            "Ivan Franko, Halytsko-ruski narodni prypovidky (1901-1910); "
            "digital transcription dataset MurzikVasilyevich/ukr-proverbs-franko; "
            "corroborated by Yemelianov, Dmytro (2026), verba v1.0.2"
        ),
        "training_purpose_source_candidate": "ALLOWED_PENDING_PROJECT_CORPUS_GATES",
        "evaluation": "NOT_GRANTED",
        "modern_text_allowed": False,
        "category_allowed": False,
        "cleaned_explanation_allowed": False,
        "variant_group_allowed": False,
        "payload_field_allowed": "prov_clean_only",
    }
    _require_exact_mapping(
        "rights boundary",
        config.get("rights_boundary"),
        expected_rights,
    )

    expected_acquisition = {
        "fetch_count_required": 2,
        "byte_identical_fetches_required": True,
        "max_source_bytes": 7000000,
        "max_license_bytes": 20000,
        "max_source_plus_license_bytes": 6300000,
        "strict_utf8": True,
        "accept_encoding": "identity",
        "required_csv_columns": ["prov_clean", "term", "letter", "description"],
        "max_rows": 40000,
    }
    _require_exact_mapping(
        "acquisition contract",
        config.get("acquisition"),
        expected_acquisition,
    )

    expected_filter = {
        "unicode_normalization": "NFC",
        "strip_outer_whitespace": True,
        "min_alphabetic_chars": 3,
        "min_cyrillic_share_of_alpha": 0.65,
        "max_text_utf8_bytes": 4096,
        "reject_control_characters": True,
        "reject_replacement_character": True,
        "reject_url_like": True,
        "reject_email_like": True,
        "deduplicate_exact_normalized_text": True,
    }
    _require_exact_mapping(
        "filter policy",
        config.get("filter"),
        expected_filter,
    )
    validate_truth_boundary(config)


def validate_license_bytes(
    config: dict[str, object],
    license_raw: bytes,
) -> dict[str, object]:
    source = config["source"]
    acquisition = config["acquisition"]
    if len(license_raw) > int(acquisition["max_license_bytes"]):
        raise RuntimeError("license exceeds configured bound")
    if git_blob_sha1(license_raw) != source["license_git_blob_sha1"]:
        raise RuntimeError("license Git-blob identity mismatch")
    try:
        license_text = license_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("license is not strict UTF-8") from exc
    if "CC0 1.0 Universal" not in license_text:
        raise RuntimeError("license semantic marker mismatch")
    return {
        "license_id": source["license_id"],
        "license_git_blob_sha1": git_blob_sha1(license_raw),
        "license_sha256": sha256_bytes(license_raw),
        "license_bytes": len(license_raw),
    }


def parse_and_filter(
    config: dict[str, object],
    raw: bytes,
) -> tuple[bytes, dict[str, object]]:
    source = config["source"]
    acquisition = config["acquisition"]
    policy = config["filter"]
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("source is not strict UTF-8") from exc

    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if reader.fieldnames is None:
        raise RuntimeError("CSV header is missing")
    required_columns = list(acquisition["required_csv_columns"])
    missing = [name for name in required_columns if name not in reader.fieldnames]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    rows_seen = 0
    accepted: list[dict[str, object]] = []
    rejected = Counter()
    seen_text_hashes: set[str] = set()
    aggregate_alpha = 0
    aggregate_cyrillic = 0

    for row_index, row in enumerate(reader, start=1):
        rows_seen += 1
        if rows_seen > int(acquisition["max_rows"]):
            raise RuntimeError("CSV exceeds configured row bound")
        value = row.get("prov_clean")
        if not isinstance(value, str):
            raise RuntimeError("prov_clean is missing from a row")
        text = normalize_source_text(value)
        reason = classify_text(text, policy)
        text_bytes = text.encode()
        text_hash = sha256_bytes(text_bytes)
        if reason is None and policy["deduplicate_exact_normalized_text"] is True:
            if text_hash in seen_text_hashes:
                reason = "exact_duplicate"
        if reason is not None:
            rejected[reason] += 1
            continue

        seen_text_hashes.add(text_hash)
        alpha = [char for char in text if char.isalpha()]
        cyrillic = [char for char in alpha if "\u0400" <= char <= "\u052f"]
        aggregate_alpha += len(alpha)
        aggregate_cyrillic += len(cyrillic)
        accepted.append(
            {
                "language": "uk",
                "modality": "natural_text",
                "record_id": f"franko1901:{row_index:06d}:{text_hash[:16]}",
                "source_family": source["source_family"],
                "source_id": source["source_id"],
                "text": text,
                "text_sha256": text_hash,
                "text_utf8_bytes": len(text_bytes),
            }
        )

    if not accepted:
        raise RuntimeError("filter retained zero rows")

    jsonl = b"".join(canonical_json_bytes(record) for record in accepted)
    inventory_projection = [
        {
            "record_id": record["record_id"],
            "text_sha256": record["text_sha256"],
            "text_utf8_bytes": record["text_utf8_bytes"],
        }
        for record in accepted
    ]
    stats = {
        "rows_seen": rows_seen,
        "accepted_rows": len(accepted),
        "rejected_rows": rows_seen - len(accepted),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "accepted_text_utf8_bytes": sum(
            int(record["text_utf8_bytes"]) for record in accepted
        ),
        "accepted_payload_jsonl_bytes": len(jsonl),
        "accepted_payload_jsonl_sha256": sha256_bytes(jsonl),
        "record_inventory_identity_sha256": sha256_bytes(
            canonical_json_bytes(inventory_projection)
        ),
        "aggregate_alphabetic_chars": aggregate_alpha,
        "aggregate_cyrillic_chars": aggregate_cyrillic,
        "aggregate_cyrillic_share_of_alpha": (
            aggregate_cyrillic / aggregate_alpha if aggregate_alpha else 0.0
        ),
    }
    return jsonl, stats


def build_report(
    config: dict[str, object],
    raw: bytes,
    license_raw: bytes,
    stats: dict[str, object],
) -> dict[str, object]:
    source = config["source"]
    license_evidence = validate_license_bytes(config, license_raw)
    core = {
        "schema_version": "12-6.d03-franko1901-materialization-report.v1",
        "worker_id": config["worker_id"],
        "decision": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_id": source["source_id"],
        "source_family": source["source_family"],
        "upstream_revision": source["upstream_revision"],
        "source_git_blob_sha1": git_blob_sha1(raw),
        "source_sha256": sha256_bytes(raw),
        "source_bytes": len(raw),
        "license_evidence": license_evidence,
        "materialization": stats,
        "rights_boundary": config["rights_boundary"],
        "truth_boundary": config["truth_boundary"],
        "local_free_only": True,
    }
    return {
        **core,
        "report_identity_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def materialize_from_bytes(
    config: dict[str, object],
    raw: bytes,
    license_raw: bytes,
) -> tuple[bytes, dict[str, object]]:
    validate_contract(config)
    source = config["source"]
    acquisition = config["acquisition"]
    if len(raw) != int(source["source_bytes"]):
        raise RuntimeError("source byte-count mismatch")
    if git_blob_sha1(raw) != source["source_git_blob_sha1"]:
        raise RuntimeError("source Git-blob identity mismatch")
    if len(raw) + len(license_raw) > int(acquisition["max_source_plus_license_bytes"]):
        raise RuntimeError("source plus license exceeds configured aggregate bound")
    validate_license_bytes(config, license_raw)
    jsonl, stats = parse_and_filter(config, raw)
    return jsonl, build_report(config, raw, license_raw, stats)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=None)
    parser.add_argument("--license-input", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_contract(config)
    source = config["source"]
    acquisition = config["acquisition"]
    max_source_bytes = int(acquisition["max_source_bytes"])
    max_license_bytes = int(acquisition["max_license_bytes"])

    if args.input or args.license_input:
        if not (args.input and args.license_input):
            raise RuntimeError("--input and --license-input must be supplied together")
        raw_a = Path(args.input).read_bytes()
        raw_b = raw_a
        license_a = Path(args.license_input).read_bytes()
        license_b = license_a
    else:
        raw_a = fetch_bounded(source["raw_url"], max_source_bytes)
        raw_b = fetch_bounded(source["raw_url"], max_source_bytes)
        license_a = fetch_bounded(source["license_url"], max_license_bytes)
        license_b = fetch_bounded(source["license_url"], max_license_bytes)

    if raw_a != raw_b:
        raise RuntimeError("two exact source acquisitions are not byte-identical")
    if license_a != license_b:
        raise RuntimeError("two exact license acquisitions are not byte-identical")

    candidate_jsonl, report = materialize_from_bytes(config, raw_a, license_a)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "franko1901-candidate.jsonl").write_bytes(candidate_jsonl)
    (output_dir / "franko1901-report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "LICENSE.cc0.txt").write_bytes(license_a)
    (output_dir / "ATTRIBUTION.txt").write_text(
        config["rights_boundary"]["required_attribution"] + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
