#!/usr/bin/env python3
"""Materialize the exact training-eligible CPython documentation chunk ledger.

This is a stdlib-only replay of the DATA-228 normalization, chunking and quality
preview. It converts the existing 14 accepted chunk hashes into an auditable
byte ledger without admitting either of the two rejected chunks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
SCHEMA = "12-6.next100-037-python-docs-accepted-chunk-ledger.v1"


class MaterializationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def chunk_text(text: str, *, max_chars: int = 1200, min_chars: int = 80) -> tuple[str, ...]:
    require(max_chars >= min_chars and min_chars >= 20, "invalid chunk limits")
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            value = "\n".join(current).strip()
            if len(value) >= min_chars:
                chunks.append(value)
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces = [paragraph]
        else:
            pieces: list[str] = []
            part: list[str] = []
            part_len = 0
            for word in paragraph.split():
                needed = len(word) if not part else len(word) + 1
                if part and part_len + needed > max_chars:
                    pieces.append(" ".join(part))
                    part = [word]
                    part_len = len(word)
                else:
                    part.append(word)
                    part_len += needed
            if part:
                pieces.append(" ".join(part))

        for piece in pieces:
            needed = len(piece) if not current else len(piece) + 1
            if current and current_len + needed > max_chars:
                flush()
            current.append(piece)
            current_len += needed
    flush()
    return tuple(chunks)


def quality_reason(text: str) -> str | None:
    if len(text) < 60:
        return "too_short"
    if len(text) > 1600:
        return "too_long"
    if any(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in text):
        return "control_character"
    if EMAIL_RE.search(text):
        return "pii_email"
    if PHONE_RE.search(text):
        return "pii_phone"
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return "empty"
    if sum(char.isalpha() for char in visible) / len(visible) < 0.35:
        return "low_alpha_ratio"
    return None


def fetch_exact(url: str, *, max_bytes: int = 100_000) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-NEXT100-037-materializer/1",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(max_bytes + 1)
    except Exception as exc:
        raise MaterializationError(f"source acquisition failed: {exc}") from exc
    require(len(payload) <= max_bytes, "source exceeds bounded acquisition size")
    return payload


def materialize(authority: dict[str, Any], *, source_sha: str) -> dict[str, Any]:
    require(authority["schema_version"] == "12-6.next100-037-python-docs-source-authority.v1", "authority schema drift")
    require(authority["terminal_verdict"] == "ADMIT", "source authority is not ADMIT")
    require(authority["local_free_only"] is True, "source authority is not LOCAL_FREE")
    require(authority["claim_boundary"]["training_executed"] is False, "source authority training boundary drift")

    source = authority["source"]
    quality = authority["quality_privacy"]
    expected_hashes = list(quality["accepted_normalized_sha256"])
    require(len(expected_hashes) == quality["accepted_chunk_count"] == 14, "accepted hash cardinality drift")
    require(len(set(expected_hashes)) == 14, "accepted hash list contains duplicates")
    require(quality["rejected_chunk_count"] == 2, "rejected chunk cardinality drift")
    require(quality["rejection_reasons"] == {"pii_phone": 2}, "quality rejection reason drift")

    raw_url = (
        "https://raw.githubusercontent.com/python/cpython/"
        + source["upstream_commit"]
        + "/"
        + source["file_set"][0]
    )
    raw = fetch_exact(raw_url)
    require(len(raw) == source["raw_bytes"], "raw source byte count drift")
    require(sha256(raw) == source["raw_sha256"], "raw source SHA-256 drift")
    git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()  # noqa: S324
    require(git_blob == source["source_git_blob_sha1"], "raw source Git blob drift")

    text = raw.decode("utf-8", errors="strict")[: int(source["normalization"]["truncate_chars"])]
    normalized = normalize_text(text)
    normalized_bytes = normalized.encode("utf-8")
    require(len(normalized_bytes) == source["normalization"]["normalized_utf8_bytes"], "normalized source byte count drift")
    require(sha256(normalized_bytes) == source["normalization"]["normalized_sha256"], "normalized source SHA-256 drift")

    chunks = chunk_text(
        normalized,
        max_chars=int(quality["chunking"]["max_chars"]),
        min_chars=int(quality["chunking"]["min_chars"]),
    )
    require(len(chunks) == quality["chunk_count"] == 16, "chunk count drift")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        reason = quality_reason(chunk)
        normalized_chunk = normalize_text(chunk)
        payload = normalized_chunk.encode("utf-8")
        digest = sha256(payload)
        row = {
            "chunk_index": chunk_index,
            "normalized_sha256": digest,
            "normalized_utf8_bytes": len(payload),
        }
        if reason is None:
            accepted.append(row)
        else:
            rejected.append({**row, "reason": reason})

    observed_hashes = [row["normalized_sha256"] for row in accepted]
    require(observed_hashes == expected_hashes, "accepted chunk hash/order drift from DATA-228 preview")
    require(len(accepted) == 14, "accepted chunk count drift")
    require(len(rejected) == 2, "rejected chunk count drift")
    require([row["reason"] for row in rejected] == ["pii_phone", "pii_phone"], "rejection replay drift")

    total_accepted = sum(int(row["normalized_utf8_bytes"]) for row in accepted)
    require(0 < total_accepted < len(normalized_bytes), "accepted byte capacity is invalid")
    require(sum(int(row["normalized_utf8_bytes"]) for row in rejected) > 0, "rejected bytes missing")

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": "NEXT100-037-DATA-EN-PYTHON-DOCS",
        "worker_source_sha": source_sha,
        "source_authority_identity_sha256": authority["authority_identity_sha256"],
        "source_id": source["source_id"],
        "source_family": source["source_family"],
        "upstream_commit": source["upstream_commit"],
        "raw_sha256": source["raw_sha256"],
        "normalized_source_sha256": source["normalization"]["normalized_sha256"],
        "chunk_count": len(chunks),
        "accepted_chunk_count": len(accepted),
        "rejected_chunk_count": len(rejected),
        "accepted_normalized_utf8_bytes": total_accepted,
        "accepted_chunks": accepted,
        "rejected_chunks": rejected,
        "training_eligibility": "ONLY_LISTED_ACCEPTED_CHUNKS_ELIGIBLE",
        "capacity_semantics": "SUM_OF_EXACT_ACCEPTED_NORMALIZED_UTF8_CHUNK_BYTES_NOT_SOURCE_BYTES_NOT_LOSS_POSITIONS",
        "execution": {
            "class": "LOCAL_FREE",
            "model_training_executed": False,
            "tokenizer_fit_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "final_test_material_consumed": False,
        },
        "claim_boundary": {
            "source_capacity_materialized": True,
            "global_cross_source_dedup_complete": False,
            "evaluation_decontamination_complete": False,
            "corpus_frozen": False,
            "unique_loss_positions_authorized": 0,
        },
    }
    return {**core, "ledger_identity_sha256": sha256(canonical_bytes(core))}


def verify_ledger(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == SCHEMA, "ledger schema drift")
    supplied = value.get("ledger_identity_sha256")
    core = dict(value)
    core.pop("ledger_identity_sha256", None)
    require(supplied == sha256(canonical_bytes(core)), "ledger self-hash mismatch")
    require(value["accepted_chunk_count"] == len(value["accepted_chunks"]) == 14, "accepted ledger cardinality drift")
    require(value["rejected_chunk_count"] == len(value["rejected_chunks"]) == 2, "rejected ledger cardinality drift")
    require(value["accepted_normalized_utf8_bytes"] == sum(row["normalized_utf8_bytes"] for row in value["accepted_chunks"]), "accepted capacity arithmetic drift")
    require(value["execution"]["model_training_executed"] is False, "training executed")
    require(value["execution"]["tokenizer_fit_executed"] is False, "tokenizer fit executed")
    require(value["execution"]["paid_compute_used"] is False, "paid compute used")
    require(value["claim_boundary"]["unique_loss_positions_authorized"] == 0, "loss positions prematurely authorized")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authority",
        type=Path,
        default=Path("configs/data/next100_037_python_docs_source_authority_v1.json"),
    )
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is not None, "exact worker source SHA required")
    authority = json.loads(args.authority.read_text(encoding="utf-8"))
    ledger = materialize(authority, source_sha=args.source_sha)
    verify_ledger(ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "accepted_chunk_count": ledger["accepted_chunk_count"],
        "accepted_normalized_utf8_bytes": ledger["accepted_normalized_utf8_bytes"],
        "ledger_identity_sha256": ledger["ledger_identity_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
