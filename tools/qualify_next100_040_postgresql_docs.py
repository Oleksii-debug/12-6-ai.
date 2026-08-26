#!/usr/bin/env python3
"""Deterministically qualify a bounded PostgreSQL 18.6 documentation source family.

LOCAL_FREE only.  Raw upstream bytes are fetched from immutable Git commit URLs,
never committed to this repository, and represented by cryptographic identities.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

WORKER_ID = "NEXT100-040-DATA-EN-POSTGRES-DOCS"
SCHEMA = "12-6.next100-040.postgresql-docs-source-authority.v1"
UPSTREAM_REPO = "https://github.com/postgres/postgres"
UPSTREAM_COMMIT = "724edf9bde9d356724ad384a2e196edc3c9f80f7"
UPSTREAM_TAG = "REL_18_6"
UPSTREAM_VERSION = "18.6"
FAMILY_ID = "github:postgres/postgres:documentation"
NORMALIZATION_ID = "POSTGRES_SGML_TEXT_V1"
REGISTRY_PATH = Path("data/registry/external_snapshots.v2.json")
LICENSE_PATH = "COPYRIGHT"
DOC_PATHS = (
    "doc/src/sgml/syntax.sgml",
    "doc/src/sgml/ddl.sgml",
    "doc/src/sgml/dml.sgml",
    "doc/src/sgml/queries.sgml",
    "doc/src/sgml/datatype.sgml",
    "doc/src/sgml/functions.sgml",
    "doc/src/sgml/typeconv.sgml",
    "doc/src/sgml/indexes.sgml",
    "doc/src/sgml/full-text.sgml",
    "doc/src/sgml/mvcc.sgml",
    "doc/src/sgml/performance-tips.sgml",
    "doc/src/sgml/parallel.sgml",
)
RAW_BASE = f"https://raw.githubusercontent.com/postgres/postgres/{UPSTREAM_COMMIT}/"

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
DECL_RE = re.compile(r"<![^>]*>", re.S)
TAG_RE = re.compile(r"<[^>]+>", re.S)
ENTITY_RE = re.compile(r"&[A-Za-z][A-Za-z0-9_.:-]*;")
BLOCK_END_RE = re.compile(
    r"</(?:para|title|programlisting|screen|synopsis|entry|row|listitem|term|simpara|blockquote|sect[1-6]|chapter|table|figure)>",
    re.I,
)
WS_RE = re.compile(r"[\t\x0b\x0c\r ]+")
BLANK_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
AWS_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")
OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
CREDENTIAL_URI_RE = re.compile(r"\b(?:postgres(?:ql)?|https?)://[^\s/:]+:[^\s/@]{6,}@", re.I)

ENGLISH_ANCHORS = {
    "the", "and", "of", "to", "a", "is", "in", "that", "for", "this", "with",
    "can", "be", "are", "as", "by", "from", "on", "or", "an", "when", "which",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256(canonical_json_bytes(value))


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def fetch(path: str) -> bytes:
    url = RAW_BASE + path
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": WORKER_ID})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            if not data:
                raise RuntimeError(f"empty upstream object: {path}")
            return data
        except Exception as exc:  # pragma: no cover - network retry path
            last = exc
            if attempt < 3:
                time.sleep(1 + attempt)
    raise RuntimeError(f"failed immutable fetch for {path}: {last}")


def normalize_sgml(raw: bytes) -> bytes:
    text = raw.decode("utf-8", errors="strict")
    text = unicodedata.normalize("NFC", text)
    text = COMMENT_RE.sub(" ", text)
    text = BLOCK_END_RE.sub("\n", text)
    text = DECL_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    # PostgreSQL build-time SGML entities whose definitions live outside the selected
    # document are not silently expanded; they are removed rather than materializing
    # another source object's bytes into this document.
    text = ENTITY_RE.sub(" ", text)
    lines: list[str] = []
    for line in text.splitlines():
        clean = WS_RE.sub(" ", line).strip()
        if clean:
            lines.append(clean)
    normalized = "\n".join(lines) + "\n"
    normalized = BLANK_RE.sub("\n\n", normalized)
    return normalized.encode("utf-8")


def quality_metrics(normalized: bytes) -> dict[str, Any]:
    text = normalized.decode("utf-8")
    words = WORD_RE.findall(text)
    lowered = [w.lower() for w in words]
    alpha_chars = sum(ch.isalpha() for ch in text)
    ascii_alpha = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
    anchors = sum(1 for w in lowered if w in ENGLISH_ANCHORS)
    unique_ratio = len(set(lowered)) / len(lowered) if lowered else 0.0
    residual_markup = len(re.findall(r"</?[A-Za-z][^>]*>", text))
    passed = (
        len(normalized) >= 4_000
        and len(words) >= 500
        and anchors >= 80
        and (ascii_alpha / alpha_chars if alpha_chars else 0.0) >= 0.90
        and unique_ratio >= 0.02
        and residual_markup == 0
    )
    return {
        "normalized_utf8_bytes": len(normalized),
        "word_count": len(words),
        "english_anchor_count": anchors,
        "ascii_alpha_fraction": round(ascii_alpha / alpha_chars, 6) if alpha_chars else 0.0,
        "unique_word_ratio": round(unique_ratio, 6),
        "residual_markup_count": residual_markup,
        "decision": "PASS" if passed else "FAIL",
    }


def secret_counts(text: str) -> dict[str, int]:
    patterns = {
        "private_key": PRIVATE_KEY_RE,
        "aws_access_key": AWS_RE,
        "github_token": GITHUB_TOKEN_RE,
        "openai_style_key": OPENAI_KEY_RE,
        "jwt": JWT_RE,
        "credential_uri": CREDENTIAL_URI_RE,
    }
    return {name: len(regex.findall(text)) for name, regex in patterns.items()}


def shingle_set(text: str, width: int = 7) -> set[str]:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < width:
        return set(words)
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}


def aggregate_identity(rows: list[dict[str, Any]], which: str) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda x: x["path"]):
        h.update(row["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(str(row[f"{which}_bytes"]).encode("ascii"))
        h.update(b"\0")
        h.update(bytes.fromhex(row[f"{which}_sha256"]))
        h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit-base64", action="store_true")
    args = parser.parse_args()

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    live_family_ids = {row["key"] for row in registry["byte_report"]["by_independent_source_family"]}
    live_raw_hashes = {src["snapshot"]["raw_sha256"] for src in registry["sources"]}
    live_norm_hashes = {src["snapshot"]["normalized_sha256"] for src in registry["sources"]}

    license_raw = fetch(LICENSE_PATH)
    license_text = license_raw.decode("utf-8", errors="strict")
    required_license_phrases = (
        "Permission to use, copy, modify, and distribute this software and its",
        "documentation for any purpose, without fee, and without a written agreement",
        "provided that the above copyright notice",
    )
    license_ok = all(phrase in license_text for phrase in required_license_phrases)

    rows: list[dict[str, Any]] = []
    normalized_text: dict[str, str] = {}
    raw_hash_seen: set[str] = set()
    norm_hash_seen: set[str] = set()
    secret_total: dict[str, int] = {}
    email_total = 0

    for path in DOC_PATHS:
        raw = fetch(path)
        normalized = normalize_sgml(raw)
        raw_hash = sha256(raw)
        norm_hash = sha256(normalized)
        quality = quality_metrics(normalized)
        counts = secret_counts(normalized.decode("utf-8"))
        for key, value in counts.items():
            secret_total[key] = secret_total.get(key, 0) + value
        emails = len(EMAIL_RE.findall(normalized.decode("utf-8")))
        email_total += emails
        row = {
            "path": path,
            "raw_url": RAW_BASE + path,
            "git_blob_sha1": git_blob_sha1(raw),
            "raw_sha256": raw_hash,
            "raw_bytes": len(raw),
            "normalized_sha256": norm_hash,
            "normalized_bytes": len(normalized),
            "normalization_id": NORMALIZATION_ID,
            "quality": quality,
            "privacy": {
                "secret_finding_counts": counts,
                "email_shape_count": emails,
                "matched_values_retained": False,
            },
            "registry_exact_collision": raw_hash in live_raw_hashes or norm_hash in live_norm_hashes,
        }
        rows.append(row)
        normalized_text[path] = normalized.decode("utf-8")
        raw_hash_seen.add(raw_hash)
        norm_hash_seen.add(norm_hash)

    pairwise: list[dict[str, Any]] = []
    max_jaccard = 0.0
    max_pair: list[str] | None = None
    shingles = {path: shingle_set(text) for path, text in normalized_text.items()}
    paths = sorted(shingles)
    for i, left in enumerate(paths):
        for right in paths[i + 1 :]:
            a, b = shingles[left], shingles[right]
            union = len(a | b)
            score = len(a & b) / union if union else 0.0
            if score > max_jaccard:
                max_jaccard = score
                max_pair = [left, right]
            if score >= 0.25:
                pairwise.append({"left": left, "right": right, "seven_word_shingle_jaccard": round(score, 6)})

    exact_internal_unique = len(raw_hash_seen) == len(rows) and len(norm_hash_seen) == len(rows)
    all_quality_pass = all(row["quality"]["decision"] == "PASS" for row in rows)
    no_secret_findings = sum(secret_total.values()) == 0
    no_registry_collision = all(not row["registry_exact_collision"] for row in rows)
    no_near_duplicate = max_jaccard < 0.85
    family_new = FAMILY_ID not in live_family_ids

    family_basis = {
        "canonical_upstream_repository": UPSTREAM_REPO,
        "lineage": "official PostgreSQL Global Development Group documentation source",
        "family_id": FAMILY_ID,
        "version_invariant": True,
        "rendered_html_pdf_alias_same_family": True,
    }
    family_identity = sha256_json(family_basis)

    rights_basis = {
        "license": "PostgreSQL License",
        "license_path": LICENSE_PATH,
        "license_sha256": sha256(license_raw),
        "upstream_commit": UPSTREAM_COMMIT,
        "model_training": "ALLOWED",
        "redistribution": "ALLOWED_WITH_NOTICE",
        "evaluation": "NOT_SEPARATELY_ADMITTED",
        "redistribution_condition": "carry PostgreSQL copyright notice, permission paragraph, and disclaimer with redistributed copies",
    }
    rights_identity = sha256_json(rights_basis)

    gates = {
        "exact_release_identity": True,
        "license_bound_to_exact_commit": license_ok,
        "training_rights": license_ok,
        "redistribution_rights_with_notice": license_ok,
        "document_provenance_complete": len(rows) == len(DOC_PATHS),
        "quality_all_documents_pass": all_quality_pass,
        "privacy_no_secret_patterns": no_secret_findings,
        "privacy_public_docs_no_private_collection_surface": True,
        "internal_exact_dedup_unique": exact_internal_unique,
        "internal_near_duplicate_below_0_85": no_near_duplicate,
        "live_registry_exact_hash_collision_absent": no_registry_collision,
        "independent_family_not_in_bound_registry": family_new,
        "generated_renderings_excluded": True,
    }
    verdict = "ADMIT" if all(gates.values()) else "RETEST"

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "verdict": verdict,
        "scope": {
            "source_kind": "official English technical documentation",
            "upstream_repository": UPSTREAM_REPO,
            "version": UPSTREAM_VERSION,
            "tag": UPSTREAM_TAG,
            "commit": UPSTREAM_COMMIT,
            "canonical_form": "source SGML only",
            "selected_document_count": len(rows),
            "generated_html_included": False,
            "generated_pdf_included": False,
            "code_examples": "retained only where embedded in selected documentation; same license; no separate code-family credit",
        },
        "license": {
            "name": "PostgreSQL License",
            "path": LICENSE_PATH,
            "raw_sha256": sha256(license_raw),
            "git_blob_sha1": git_blob_sha1(license_raw),
            "raw_bytes": len(license_raw),
            "phrase_verification": license_ok,
            "rights_identity_sha256": rights_identity,
            "model_training": "ALLOWED",
            "redistribution": "ALLOWED_WITH_NOTICE",
            "evaluation": "NOT_SEPARATELY_ADMITTED",
            "notice_requirement": rights_basis["redistribution_condition"],
            "interpretation": "explicit permission to use/copy/modify/distribute software and documentation for any purpose covers model-training use; redistribution must preserve the stated notice/disclaimer conditions",
        },
        "family": {
            **family_basis,
            "family_identity_sha256": family_identity,
            "independent_family_credit": 1 if family_new else 0,
            "document_count_does_not_equal_family_count": True,
        },
        "normalization": {
            "normalization_id": NORMALIZATION_ID,
            "algorithm": "UTF-8 strict; NFC; drop SGML comments/declarations; line-break block endings; strip tags; HTML-unescape standard entities; drop unresolved build-time entities; trim/collapse horizontal whitespace; retain non-empty lines",
            "source_only_no_rendered_duplicate": True,
        },
        "documents": rows,
        "aggregate": {
            "raw_bytes": sum(row["raw_bytes"] for row in rows),
            "normalized_bytes": sum(row["normalized_bytes"] for row in rows),
            "raw_manifest_sha256": aggregate_identity(rows, "raw"),
            "normalized_manifest_sha256": aggregate_identity(rows, "normalized"),
        },
        "privacy": {
            "source_class": "public official documentation, not user/private data collection",
            "secret_finding_counts": dict(sorted(secret_total.items())),
            "email_shape_count": email_total,
            "matched_values_retained": False,
            "decision": "PASS" if no_secret_findings else "RETEST",
            "boundary": "public contact/example identifiers, if present, are counted but not persisted as matched values; this is not a universal PII-absence claim",
        },
        "dedup": {
            "internal_exact_unique": exact_internal_unique,
            "maximum_seven_word_shingle_jaccard": round(max_jaccard, 6),
            "maximum_pair": max_pair,
            "pairs_at_or_above_0_25": pairwise,
            "near_duplicate_threshold": 0.85,
            "bound_registry_exact_collision_absent": no_registry_collision,
            "family_lineage_collision_absent": family_new,
            "rendering_rule": "SGML source is canonical; HTML/PDF/manpage/rendered mirrors are aliases and MUST NOT be counted or trained as additional copies",
            "downstream_boundary": "incumbent DATA-232/DATA-298 corpus-level mirror/fragment/near-copy checks remain mandatory when this family is materialized with other sources",
        },
        "registry_concurrency_binding": {
            "path": str(REGISTRY_PATH),
            "schema_version": registry.get("schema_version"),
            "registry_identity_sha256": registry.get("registry_identity_sha256"),
            "source_count": registry.get("source_count"),
            "independent_source_family_count": registry.get("independent_source_family_count"),
            "note": "must be refreshed immediately before final publication; this evidence binds the registry present on the qualification branch",
        },
        "gates": gates,
        "claim_boundary": {
            "source_family_qualified_for_training": verdict == "ADMIT",
            "corpus_registry_mutated": False,
            "corpus_frozen": False,
            "representative_corpus_claimed": False,
            "evaluation_use_authorized": False,
            "downstream_d03_materialization_required": True,
            "downstream_incumbent_quality_privacy_dedup_required": True,
        },
    }
    evidence["authority_identity_sha256"] = sha256_json(evidence)

    payload = canonical_json_bytes(evidence)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(json.dumps({
        "verdict": verdict,
        "authority_identity_sha256": evidence["authority_identity_sha256"],
        "raw_manifest_sha256": evidence["aggregate"]["raw_manifest_sha256"],
        "normalized_manifest_sha256": evidence["aggregate"]["normalized_manifest_sha256"],
        "raw_bytes": evidence["aggregate"]["raw_bytes"],
        "normalized_bytes": evidence["aggregate"]["normalized_bytes"],
        "family_identity_sha256": family_identity,
        "license_sha256": evidence["license"]["raw_sha256"],
        "registry_identity_sha256": registry.get("registry_identity_sha256"),
        "document_count": len(rows),
        "max_shingle_jaccard": round(max_jaccard, 6),
        "secret_findings": sum(secret_total.values()),
        "quality_pass_count": sum(row["quality"]["decision"] == "PASS" for row in rows),
    }, sort_keys=True))
    if args.emit_base64:
        print("NEXT100040_EVIDENCE_BASE64=" + base64.b64encode(payload).decode("ascii"))
    return 0 if verdict == "ADMIT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
