#!/usr/bin/env python3
"""Materialize the exact Rada_Trees rights-scoped survivor set as zero-credit D03 records.

This is deliberately a source-specific handoff, not a new quality/privacy/dedup
implementation.  It consumes the exact immutable secondary archive and the compact
full-scan + provenance/rights authorities carried by the canonical Rada_Trees lane.
Every admitted payload is re-extracted, byte/hash checked, decoded with the encoding
already established by the terminal full scan, and normalized to UTF-8 JSONL.

The emitted records remain ineligible for corpus/training/evaluation use.  They are
only an executable input to the existing current-main quality/privacy -> global dedup
-> reserved-evaluation decontamination -> balance/family-cap chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import derive_d03_rada_trees_provenance_rights as rights
import scan_d03_rada_trees_secondary_plaintext as scan

ROOT = Path(__file__).resolve().parents[1]
FULL_SCAN = ROOT / "evidence/d03-rada-trees/secondary-plaintext-full-scan-terminal-v1.json"
RIGHTS_CONFIG = ROOT / "configs/data/d03_rada_trees_provenance_rights_v1.json"
RIGHTS_REPORT = ROOT / "evidence/d03-rada-trees/secondary-plaintext-provenance-rights-v1.json"

SCHEMA = "12-6.d03-rada-trees-rights-handoff.v1"
SOURCE_ID = "d03-rada-trees-plenary-transcripts-1990-2024"
FAMILY = "ua.rada.open-data.plenary-transcripts"
LANGUAGE = "uk"
LICENSE = "CC-BY-4.0"
EXPECTED_RIGHTS_REPORT_SHA256 = "7eea6d0b79353ef565910738dce9de93008a961059cab503b6f05686c0f27a7d"
EXPECTED_ACCEPTED_MEMBERS = 4_384
EXPECTED_ACCEPTED_SOURCE_BYTES = 877_899_128
EXPECTED_ACCEPTED_INVENTORY = "7b93056f38fbc87c11e14df9069066e21380db1f1b8d8016314330f841bfa6fc"
EXPECTED_ARCHIVE_BYTES = 697_768_591
EXPECTED_ARCHIVE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
ALLOWED_ENCODINGS = {"utf-8-sig", "windows-1251"}


class HandoffError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical(obj: Any) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"cannot read authority JSON: {path}") from exc
    require(isinstance(value, dict), f"authority JSON root is not an object: {path}")
    return value


def _accepted_rows(
    full_scan: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    metadata = rights.verify_full_scan(full_scan, config)
    survivors = rights.exact_unique_survivors(metadata)
    policy = config["path_provenance_policy"]
    pattern = re.compile(policy["dated_plenary_path_regex"])
    holds = {row["path"] for row in policy["explicit_holds"]}
    accepted: list[dict[str, Any]] = []
    compact_for_identity: list[dict[str, Any]] = []

    for row in survivors:
        path = str(row["path"])
        if path in holds:
            continue
        match = pattern.fullmatch(path)
        require(match is not None, f"survivor is neither dated plenary path nor explicit hold: {path}")
        parsed = date.fromisoformat(match.group("date"))
        require(
            int(policy["minimum_year"]) <= parsed.year <= int(policy["maximum_year"]),
            f"accepted path outside rights period: {path}",
        )
        encoding = row.get("decoded_encoding")
        require(encoding in ALLOWED_ENCODINGS, f"terminal decoded encoding is not admitted: {path}: {encoding}")
        require(row.get("classification") == "PLAIN_TEXT_CANDIDATE", f"non-plaintext survivor: {path}")
        require(row.get("text_emitted") is False, f"terminal raw-text boundary drift: {path}")
        accepted.append(row)
        compact_for_identity.append(
            {
                "path": path,
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
                "date": parsed.isoformat(),
            }
        )

    accepted.sort(key=lambda row: str(row["path"]))
    compact_for_identity.sort(key=lambda row: row["path"])
    require(len(accepted) == EXPECTED_ACCEPTED_MEMBERS, "accepted member count drift")
    require(
        sum(int(row["size_bytes"]) for row in accepted) == EXPECTED_ACCEPTED_SOURCE_BYTES,
        "accepted source-byte total drift",
    )
    require(
        rights.inventory_identity(compact_for_identity) == EXPECTED_ACCEPTED_INVENTORY,
        "accepted rights inventory identity drift",
    )
    return accepted


def verify_authority() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_scan = load_json(FULL_SCAN)
    config = load_json(RIGHTS_CONFIG)
    persisted = load_json(RIGHTS_REPORT)

    derived = rights.derive(full_scan, config)
    require(derived == persisted, "persisted provenance/rights report does not exactly replay")
    require(
        persisted.get("report_sha256") == EXPECTED_RIGHTS_REPORT_SHA256,
        "rights report identity drift",
    )
    decision = persisted.get("decision", {})
    require(
        decision.get("status")
        == "DATED_PARLIAMENT_TRANSCRIPT_RIGHTS_SCOPE_SUPPORTED_WITH_ATTRIBUTION_ONE_SOURCE_SCOPE_HOLD",
        "rights-scope decision drift",
    )
    require(decision.get("attribution_required") is True, "mandatory attribution boundary weakened")
    require(decision.get("rights_scope_supported_bytes_are_training_credit") is False, "rights bytes promoted to training credit")
    boundary = persisted.get("claim_boundary", {})
    require(boundary.get("training_authorized_bytes") == 0, "training-byte authority must remain zero")
    require(boundary.get("unique_causal_loss_positions_authorized") == 0, "loss authority must remain zero")
    for key in ("tokenizer_fit_authorized", "model_training_executed", "final_test_payload_accessed", "paid_compute_used"):
        require(boundary.get(key) is False, f"authority boundary weakened: {key}")

    accepted = _accepted_rows(full_scan, config)
    provenance = persisted.get("provenance", {})
    require(provenance.get("accepted_exact_unique_members") == len(accepted), "persisted accepted count drift")
    require(provenance.get("accepted_exact_unique_bytes") == EXPECTED_ACCEPTED_SOURCE_BYTES, "persisted accepted bytes drift")
    require(provenance.get("accepted_path_inventory_sha256") == EXPECTED_ACCEPTED_INVENTORY, "persisted accepted inventory drift")
    return persisted, accepted


def decode_normalized(payload: bytes, encoding: str) -> tuple[str, bytes]:
    require(encoding in ALLOWED_ENCODINGS, f"unsupported terminal encoding: {encoding}")
    try:
        text = payload.decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise HandoffError(f"payload no longer decodes as terminal encoding {encoding}") from exc
    normalized = text.encode("utf-8")
    require(len(normalized) > 0, "normalized payload unexpectedly empty")
    return text, normalized


def build_record(row: dict[str, Any], payload: bytes, text: str, normalized: bytes) -> dict[str, Any]:
    path = str(row["path"])
    source_sha = str(row["sha256"])
    record_id = sha256_bytes((SOURCE_ID + "\0" + path + "\0" + source_sha).encode("utf-8"))
    match = re.fullmatch(r"texts/(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})__.+\.txt", path)
    require(match is not None, f"materialized path no longer matches plenary contract: {path}")
    return {
        "record_id": record_id,
        "source_id": SOURCE_ID,
        "family": FAMILY,
        "language": LANGUAGE,
        "source_commit": scan.probe.REVISION,
        "source_dataset": scan.probe.DATASET,
        "source_archive": scan.probe.ARCHIVE,
        "source_path": path,
        "session_date": match.group("date"),
        "source_bytes": len(payload),
        "source_sha256": source_sha,
        "source_encoding": str(row["decoded_encoding"]),
        "normalized_bytes": len(normalized),
        "normalized_sha256": sha256_bytes(normalized),
        "text": text,
        "license": LICENSE,
        "attribution_required": True,
        "source_url": "https://data.rada.gov.ua/open/data/stenogram",
        "rights_report_sha256": EXPECTED_RIGHTS_REPORT_SHA256,
        "current_corpus_eligible": False,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def materialize(archive: Path, out_dir: Path) -> dict[str, Any]:
    persisted, accepted = verify_authority()
    require(archive.is_file() and not archive.is_symlink(), "archive must be a regular non-symlink file")
    require(archive.stat().st_size == EXPECTED_ARCHIVE_BYTES, "archive byte-count drift")
    require(sha256_file(archive) == EXPECTED_ARCHIVE_SHA256, "archive SHA-256 drift")

    probe_config = scan.probe.load_config()
    extractor = scan.inventory.find_extractor(
        None, list(probe_config["inventory_policy"]["accepted_extractors"])
    )
    require(
        scan.inventory.extractor_version(extractor) == scan.EXPECTED_EXTRACTOR_VERSION,
        "7z runtime identity drift",
    )
    expected = [
        {"path": str(row["path"]), "size_bytes": int(row["size_bytes"])}
        for row in accepted
    ]

    with tempfile.TemporaryDirectory(prefix="rada-rights-handoff-") as tmp:
        extracted = Path(tmp) / "accepted"
        observed = scan.extract_exact_selected(extractor, archive, extracted, expected)
        by_path = {str(row["path"]): row for row in observed}
        require(len(by_path) == len(accepted), "materialized member count drift")

        records: list[dict[str, Any]] = []
        for row in accepted:
            path = str(row["path"])
            actual = by_path.get(path)
            require(actual is not None, f"accepted member missing after extraction: {path}")
            require(int(actual["size_bytes"]) == int(row["size_bytes"]), f"accepted member size drift: {path}")
            require(str(actual["sha256"]) == str(row["sha256"]), f"accepted member SHA drift: {path}")
            payload = (extracted / Path(path)).read_bytes()
            require(len(payload) == int(row["size_bytes"]), f"accepted payload size drift: {path}")
            require(sha256_bytes(payload) == str(row["sha256"]), f"accepted payload SHA drift: {path}")
            text, normalized = decode_normalized(payload, str(row["decoded_encoding"]))
            records.append(build_record(row, payload, text, normalized))

    records.sort(key=lambda row: row["source_path"])
    require(len(records) == EXPECTED_ACCEPTED_MEMBERS, "record count drift")
    require(sum(row["source_bytes"] for row in records) == EXPECTED_ACCEPTED_SOURCE_BYTES, "record source-byte total drift")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = b"".join(canonical(record) for record in records)
    (out_dir / "rada_trees_rights_handoff.jsonl").write_bytes(jsonl)
    attribution_urls = [
        authority["url"] for authority in persisted["rights_authorities"]
    ]
    attribution = (
        "Rada_Trees parliamentary transcript handoff\n"
        "Dataset: uacorpus/Rada_Trees (CC BY 4.0)\n"
        "Original source: Verkhovna Rada of Ukraine Open Data Portal\n"
        "Attribution/source URLs:\n- " + "\n- ".join(attribution_urls) + "\n"
    ).encode("utf-8")
    (out_dir / "ATTRIBUTION.txt").write_bytes(attribution)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "source_id": SOURCE_ID,
        "family": FAMILY,
        "language": LANGUAGE,
        "source_dataset": scan.probe.DATASET,
        "source_revision": scan.probe.REVISION,
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "rights_report_sha256": EXPECTED_RIGHTS_REPORT_SHA256,
        "accepted_path_inventory_sha256": EXPECTED_ACCEPTED_INVENTORY,
        "record_count": len(records),
        "source_bytes": sum(row["source_bytes"] for row in records),
        "normalized_utf8_bytes": sum(row["normalized_bytes"] for row in records),
        "jsonl_sha256": sha256_bytes(jsonl),
        "attribution_sha256": sha256_bytes(attribution),
        "license": LICENSE,
        "attribution_required": True,
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "current_corpus_eligible": False,
        "requires_language_quality_privacy": True,
        "requires_global_exact_near_lineage_dedup": True,
        "requires_reserved_evaluation_decontamination": True,
        "requires_family_cap_mix_recompute": True,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
    }
    manifest["manifest_identity_sha256"] = sha256_bytes(canonical(manifest))
    (out_dir / "manifest.json").write_bytes(canonical(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--verify-authority-only", action="store_true")
    args = parser.parse_args()
    try:
        persisted, accepted = verify_authority()
        if args.verify_authority_only:
            print("D03_RADA_TREES_HANDOFF_AUTHORITY=PASS_ZERO_CREDIT")
            print(f"RIGHTS_REPORT_SHA256={persisted['report_sha256']}")
            print(f"ACCEPTED_MEMBERS={len(accepted)}")
            print(f"ACCEPTED_SOURCE_BYTES={sum(int(row['size_bytes']) for row in accepted)}")
            print("TRAINING_AUTHORIZED_BYTES=0")
            return 0
        require(args.archive is not None, "--archive is required for materialization")
        require(args.out_dir is not None, "--out-dir is required for materialization")
        manifest = materialize(args.archive, args.out_dir)
        print("D03_RADA_TREES_HANDOFF=PASS_ZERO_CREDIT")
        print(f"RECORDS={manifest['record_count']}")
        print(f"SOURCE_BYTES={manifest['source_bytes']}")
        print(f"NORMALIZED_UTF8_BYTES={manifest['normalized_utf8_bytes']}")
        print(f"JSONL_SHA256={manifest['jsonl_sha256']}")
        print("TRAINING_AUTHORIZED_BYTES=0")
        return 0
    except (OSError, ValueError, KeyError, TypeError, HandoffError, rights.ProvenanceRightsError, scan.PlaintextScanError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
