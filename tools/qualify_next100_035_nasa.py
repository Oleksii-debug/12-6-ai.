#!/usr/bin/env python3
"""Bounded, fail-closed NASA NTRS technical-text qualification.

Only structured title+abstract metadata can become training bytes. PDF/full-text bodies,
figures, tables and images are deliberately outside this authority.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_035_nasa_ntrs_usgov_abstracts_v1.json")
THIRD_PARTY_MARKERS = (
    "©",
    "copyright",
    "all rights reserved",
    "reprinted with permission",
    "reproduced with permission",
    "adapted with permission",
    "courtesy of",
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_RE = re.compile(r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fetch_record(document_id: int) -> tuple[bytes, dict[str, Any]]:
    url = f"https://ntrs.nasa.gov/api/citations/{document_id}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "12-6-ai NEXT100-035 NASA source qualification (LOCAL_FREE)",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("citation endpoint did not return one JSON object")
    return body, value


def normalize_text(title: str, abstract: str) -> str:
    def clean(value: str) -> str:
        value = html.unescape(value)
        value = unicodedata.normalize("NFKC", value)
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[ \t\f\v]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    return f"{clean(title)}\n\n{clean(abstract)}\n"


def nasa_author_gate(record: dict[str, Any]) -> tuple[bool, list[str]]:
    rows = record.get("authorAffiliations")
    if not isinstance(rows, list) or not rows:
        return False, ["NO_AUTHOR_AFFILIATIONS"]
    reasons: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"AUTHOR_{index}_MALFORMED")
            continue
        user_type = str(row.get("userType") or "").upper()
        meta = row.get("meta") or {}
        org = ((meta.get("organization") or {}).get("name") or "") if isinstance(meta, dict) else ""
        org_l = str(org).lower()
        civil = user_type == "CIVIL"
        explicit_nasa = "nasa" in org_l or "national aeronautics and space administration" in org_l
        if not (civil or explicit_nasa):
            reasons.append(f"AUTHOR_{index}_NOT_NASA_CIVIL:{org or user_type or 'UNKNOWN'}")
    return not reasons, reasons


def privacy_gate(text: str) -> list[str]:
    reasons: list[str] = []
    if EMAIL_RE.search(text):
        reasons.append("EMAIL_ADDRESS_IN_TRAINING_TEXT")
    if SSN_RE.search(text):
        reasons.append("SSN_PATTERN_IN_TRAINING_TEXT")
    if PHONE_RE.search(text):
        reasons.append("PHONE_PATTERN_IN_TRAINING_TEXT")
    return reasons


def quality_metrics(text: str) -> dict[str, Any]:
    words = TOKEN_RE.findall(text)
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t") / max(1, len(text))
    alpha = sum(1 for ch in text if ch.isalpha()) / max(1, len(text))
    return {
        "bytes": len(text.encode("utf-8")),
        "characters": len(text),
        "words": len(words),
        "printable_ratio": printable,
        "alphabetic_ratio": alpha,
    }


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def qualify_one(config: dict[str, Any], document_id: int, out_dir: Path) -> dict[str, Any]:
    body, record = fetch_record(document_id)
    reasons: list[str] = []
    if int(record.get("id", -1)) != document_id:
        reasons.append("DOCUMENT_ID_MISMATCH")
    if record.get("distribution") != config["required_distribution"]:
        reasons.append("NOT_PUBLIC_DISTRIBUTION")
    if record.get("status") not in (None, "CURATED"):
        reasons.append("NOT_CURATED")

    copyright_data = record.get("copyright")
    if not isinstance(copyright_data, dict):
        reasons.append("NO_DOCUMENT_COPYRIGHT_METADATA")
        copyright_data = {}
    if copyright_data.get("determinationType") not in config["copyright_determinations_allowed"]:
        reasons.append("COPYRIGHT_DETERMINATION_NOT_GOV_PUBLIC_USE")
    if copyright_data.get("licenseType") != config["required_license_type"]:
        reasons.append("UNEXPECTED_LICENSE_TYPE")
    if copyright_data.get("containsThirdPartyMaterial") is True:
        reasons.append("THIRD_PARTY_CONTENT_POSITIVE")
    if copyright_data.get("belongsToContractor") is True:
        reasons.append("CONTRACTOR_COPYRIGHT_POSITIVE")
    if copyright_data.get("belongsToPublisher") is True:
        reasons.append("PUBLISHER_COPYRIGHT_POSITIVE")

    authors_ok, author_reasons = nasa_author_gate(record)
    if not authors_ok:
        reasons.extend(author_reasons)

    export_control = record.get("exportControl") or {}
    if str(export_control.get("ear") or "NO").upper() not in ("NO", "FALSE", "0"):
        reasons.append("EAR_EXPORT_CONTROL")
    if str(export_control.get("itar") or "NO").upper() not in ("NO", "FALSE", "0"):
        reasons.append("ITAR_EXPORT_CONTROL")

    # NTRS defines public sensitiveInformation value 2 as NONE.
    sensitive = record.get("sensitiveInformation")
    if sensitive not in (None, 2, "2"):
        reasons.append("SENSITIVE_INFORMATION_FLAG")

    title = record.get("title")
    abstract = record.get("abstract")
    if not isinstance(title, str) or not title.strip():
        reasons.append("MISSING_TITLE")
        title = ""
    if not isinstance(abstract, str) or not abstract.strip():
        reasons.append("MISSING_ABSTRACT")
        abstract = ""

    normalized = normalize_text(title, abstract) if title and abstract else ""
    lower = normalized.lower()
    for marker in THIRD_PARTY_MARKERS:
        if marker.lower() in lower:
            reasons.append(f"THIRD_PARTY_TEXT_MARKER:{marker}")
    reasons.extend(privacy_gate(normalized))
    metrics = quality_metrics(normalized)
    if metrics["words"] < int(config["minimum_words_per_record"]):
        reasons.append("TOO_FEW_WORDS")
    if metrics["printable_ratio"] < 0.99:
        reasons.append("LOW_PRINTABLE_RATIO")
    if metrics["alphabetic_ratio"] < 0.55:
        reasons.append("LOW_ALPHABETIC_RATIO")

    canonical_record = canonical_bytes(record)
    raw_dir = out_dir / "raw"
    norm_dir = out_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{document_id}.json").write_bytes(canonical_record)

    normalized_bytes = normalized.encode("utf-8")
    normalized_sha = sha256(normalized_bytes)
    if not reasons:
        (norm_dir / f"{normalized_sha}.txt").write_bytes(normalized_bytes)

    body_exclusions_explicit = all(
        copyright_data.get(key) is False
        for key in ("containsThirdPartyMaterial", "belongsToContractor", "belongsToPublisher")
    )

    return {
        "document_id": document_id,
        "citation_url": f"https://ntrs.nasa.gov/citations/{document_id}",
        "api_url": f"https://ntrs.nasa.gov/api/citations/{document_id}",
        "title": title,
        "modified": record.get("modified"),
        "sti_type": record.get("stiType"),
        "sti_type_details": record.get("stiTypeDetails"),
        "subject_categories": record.get("subjectCategories") or [],
        "distribution": record.get("distribution"),
        "copyright": copyright_data,
        "all_authors_nasa_affiliated": authors_ok,
        "full_document_body_rights": (
            "EXPLICIT_THIRD_PARTY_EXCLUSIONS_PRESENT_BUT_BODY_STILL_OUT_OF_SCOPE"
            if body_exclusions_explicit
            else "NOT_ADMITTED_THIRD_PARTY_EXCLUSIONS_NOT_EXPLICIT"
        ),
        "http_body_sha256": sha256(body),
        "http_body_bytes": len(body),
        "canonical_record_sha256": sha256(canonical_record),
        "canonical_record_bytes": len(canonical_record),
        "normalization_policy": config["normalization_policy"],
        "normalized_sha256": normalized_sha,
        "normalized_bytes": len(normalized_bytes),
        "quality": metrics,
        "privacy_training_text": "PASS_NO_EMAIL_PHONE_SSN_PATTERNS" if not privacy_gate(normalized) else "FAIL",
        "decision": "ADMIT" if not reasons else "REJECT",
        "reasons": sorted(set(reasons)),
    }


def build_evidence(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    candidates = [qualify_one(config, int(doc_id), out_dir) for doc_id in config["candidate_document_ids"]]
    admitted = [row for row in candidates if row["decision"] == "ADMIT"]

    duplicate_pairs: list[dict[str, Any]] = []
    normalized_texts: dict[int, str] = {}
    for row in admitted:
        path = out_dir / "normalized" / f"{row['normalized_sha256']}.txt"
        normalized_texts[row["document_id"]] = path.read_text(encoding="utf-8")
    for i, left in enumerate(admitted):
        for right in admitted[i + 1 :]:
            if left["normalized_sha256"] == right["normalized_sha256"]:
                duplicate_pairs.append({"left": left["document_id"], "right": right["document_id"], "kind": "EXACT"})
                continue
            score = jaccard(
                shingles(normalized_texts[left["document_id"]]),
                shingles(normalized_texts[right["document_id"]]),
            )
            if score >= float(config["near_duplicate_jaccard_threshold"]):
                duplicate_pairs.append(
                    {
                        "left": left["document_id"],
                        "right": right["document_id"],
                        "kind": "NEAR",
                        "jaccard": score,
                    }
                )

    terminal_reasons: list[str] = []
    if len(admitted) < int(config["minimum_admitted_records"]):
        terminal_reasons.append("INSUFFICIENT_RIGHTS_CLEAR_RECORDS")
    if duplicate_pairs:
        terminal_reasons.append("INTERNAL_DUPLICATE_OR_NEAR_DUPLICATE")

    family_identity = sha256(
        canonical_bytes(
            {
                "family_id": config["family_id"],
                "document_ids": [row["document_id"] for row in admitted],
                "normalized_sha256": [row["normalized_sha256"] for row in admitted],
            }
        )
    )

    evidence: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "worker_id": config["worker_id"],
        "authority": "ADMIT" if not terminal_reasons else "RETEST",
        "authority_reasons": terminal_reasons,
        "family": {
            "family_id": config["family_id"],
            "family_identity_sha256": family_identity,
            "independent_family_credit": 1 if admitted else 0,
            "lineage_rule": "ALL_ADMITTED_NTRS_RECORDS_COUNT_AS_ONE_NASA_STI_NTRS_FAMILY",
            "mirror_or_url_variants_create_family_credit": False,
        },
        "scope": {
            "candidate_document_ids": config["candidate_document_ids"],
            "admitted_document_ids": [row["document_id"] for row in admitted],
            "training_fields": config["training_payload_fields"],
            "pdf_body_admitted": False,
            "full_text_download_admitted": False,
            "images_tables_figures_admitted": False,
        },
        "rights": {
            "retained_metadata_gate": "PUBLIC + GOV_PUBLIC_USE_PERMITTED + no positive third-party/contractor/publisher flag + NASA civil authors",
            "model_training": "ALLOWED" if admitted else "NOT_ADMITTED",
            "redistribution": config["redistribution_decision"] if admitted else "NOT_ADMITTED",
            "evaluation": config["evaluation_decision"],
            "full_document_body": "NOT_ADMITTED_BY_THIS_AUTHORITY",
            "full_document_body_requires_explicit_third_party_false": True,
            "attribution": config["attribution_text"],
            "rights_evidence": config["rights_evidence"],
            "no_endorsement": True,
        },
        "privacy": {
            "training_payload_excludes_author_names_and_affiliations": True,
            "training_payload_pattern_scan": "email+phone+SSN",
            "raw_metadata_is_provenance_only_not_training": True,
        },
        "quality": {
            "structured_text_extraction": "NTRS_OPENAPI_TITLE_PLUS_ABSTRACT",
            "minimum_words_per_record": config["minimum_words_per_record"],
            "image_or_table_ocr_used": False,
        },
        "deduplication": {
            "within_family_exact_and_5_token_jaccard": True,
            "threshold": config["near_duplicate_jaccard_threshold"],
            "duplicate_pairs": duplicate_pairs,
            "cross_registry_exact_hash_check_required_before_corpus_merge": True,
            "cross_registry_near_copy_audit_required_before_corpus_merge": True,
        },
        "candidates": candidates,
        "totals": {
            "candidate_records": len(candidates),
            "admitted_records": len(admitted),
            "normalized_training_bytes": sum(row["normalized_bytes"] for row in admitted),
        },
        "local_free_only": bool(config["local_free_only"]),
        "training_executed": False,
    }
    identity_payload = dict(evidence)
    evidence["authority_identity_sha256"] = sha256(canonical_bytes(identity_payload))
    return evidence


def pins_from_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    admitted = [row for row in evidence["candidates"] if row["decision"] == "ADMIT"]
    return {
        "authority": evidence["authority"],
        "authority_identity_sha256": evidence["authority_identity_sha256"],
        "family_identity_sha256": evidence["family"]["family_identity_sha256"],
        "admitted_document_ids": [row["document_id"] for row in admitted],
        "records": [
            {
                "document_id": row["document_id"],
                "modified": row["modified"],
                "canonical_record_sha256": row["canonical_record_sha256"],
                "normalized_sha256": row["normalized_sha256"],
                "normalized_bytes": row["normalized_bytes"],
            }
            for row in admitted
        ],
    }


def verify_pins(config: dict[str, Any], evidence: dict[str, Any]) -> None:
    pins = config.get("pins")
    if not isinstance(pins, dict):
        raise SystemExit("pins are not sealed; probe only")
    current = pins_from_evidence(evidence)
    if current != pins:
        print("PIN_MISMATCH", file=sys.stderr)
        print(
            json.dumps({"expected": pins, "current": current}, indent=2, ensure_ascii=False),
            file=sys.stderr,
        )
        raise SystemExit(2)
    if evidence["authority"] != "ADMIT":
        raise SystemExit("sealed authority is not ADMIT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "verify"))
    parser.add_argument("--out", default="next100-035-nasa-evidence")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence(config, out_dir)
    evidence_path = out_dir / "nasa_ntrs_usgov_abstracts_authority.json"
    evidence_path.write_bytes(canonical_bytes(evidence) + b"\n")
    pins = pins_from_evidence(evidence)
    print(
        json.dumps(
            {"authority": evidence["authority"], "pins": pins, "totals": evidence["totals"]},
            sort_keys=True,
        )
    )
    if args.mode == "verify":
        verify_pins(config, evidence)
    if evidence["authority"] != "ADMIT":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
