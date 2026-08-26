#!/usr/bin/env python3
"""Materialize the exact accepted-chunk byte ledger for NEXT100-037 CPython docs.

This successor does one narrow job: reacquire the immutable CPython tutorial
object, reproduce the exact DATA-228/DATA-181 normalization, chunking and
privacy/quality predicates, and emit byte counts only for the 14 chunk hashes
already admitted by NEXT100-037. Rejected chunk text and hashes are never
written to the ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

SCHEMA = "12-6.next100-101-cpython-accepted-chunk-ledger.v1"
LEDGER_SCHEMA = "12-6.next100-101-cpython-accepted-chunk-ledger-report.v1"
WORKER = "NEXT100-101-CPYTHON-ACCEPTED-CHUNK-LEDGER"
DEFAULT_CONFIG = Path("configs/data/next100_101_cpython_accepted_chunk_ledger_v1.json")
DEFAULT_OUTPUT = Path("evidence/next100_101/cpython_accepted_chunk_ledger_v1.json")
MAX_SOURCE_BYTES = 100_000

EXPECTED_AUTHORITY = {
    "worker_id": "NEXT100-037-DATA-EN-PYTHON-DOCS",
    "pr": 467,
    "head_sha": "5a6a495a24bce449334cbc5126d0114f61a9f57c",
    "workflow_run": 32998356906,
    "workflow_name": "NEXT100-037 Python Docs Source Authority",
    "workflow_conclusion": "success",
    "authority_blob_sha1": "b15abac8744ccda9fe58d1351f7925b6ab328034",
    "authority_identity_sha256": "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
    "terminal_verdict": "ADMIT",
}
EXPECTED_SOURCE_ID = "en.python.docs.tutorial-introduction"
EXPECTED_FAMILY = "python.cpython.documentation"
EXPECTED_COMMIT = "7f0ccd6c0e3f85fbaeceb2f67b06ab3631db0480"
EXPECTED_PATH = "Doc/tutorial/introduction.rst"
EXPECTED_BLOB_SHA1 = "465c32d0b72431cc446aae7edeb6b829c657b243"
EXPECTED_RAW_BYTES = 19188
EXPECTED_RAW_SHA256 = "cf1674daf9568abeb5fc22f62a991e17751fea4deb06f598362ce6e7de264808"
EXPECTED_NORMALIZED_BYTES = 17901
EXPECTED_NORMALIZED_SHA256 = "64a4ec4fd7574ba4c22e615a032b157e446b9c7f5a7917cb7f10fa214a05bd1a"
EXPECTED_CHUNKS = 16
EXPECTED_ACCEPTED = 14
EXPECTED_REJECTED = 2
EXPECTED_ACCEPTED_BYTES = 15540
EXPECTED_ACCEPTED_MIN = 290
EXPECTED_ACCEPTED_MAX = 1196
EXPECTED_ACCEPTED_MEAN = 1110.0
EXPECTED_REJECTION_REASONS = {"pii_phone": 2}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")


class LedgerError(RuntimeError):
    """Fail-closed ledger materialization error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LedgerError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalize_text(text: str) -> str:
    """Exact DATA-228 plain-text normalization."""
    text = unicodedata.normalize(
        "NFKC", text.replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _chunk_text(text: str, *, max_chars: int = 1200, min_chars: int = 80) -> tuple[str, ...]:
    """Exact DATA-228/DATA-181 generic natural-text chunking semantics."""
    if max_chars < min_chars or min_chars < 20:
        raise LedgerError("invalid chunk limits")
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
            current, current_len = [], 0

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces = [paragraph]
        else:
            pieces: list[str] = []
            words = paragraph.split()
            part: list[str] = []
            part_len = 0
            for word in words:
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


def _quality_reason(
    text: str,
    *,
    min_chars: int = 60,
    max_chars: int = 1600,
    min_alpha_ratio: float = 0.35,
) -> str | None:
    """Exact DATA-228/D03 privacy/quality predicate."""
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\t" for ch in text):
        return "control_character"
    if EMAIL_RE.search(text):
        return "pii_email"
    if PHONE_RE.search(text):
        return "pii_phone"
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return "empty"
    alpha_ratio = sum(ch.isalpha() for ch in visible) / len(visible)
    if alpha_ratio < min_alpha_ratio:
        return "low_alpha_ratio"
    return None


def _load_contract(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"cannot read contract: {exc}") from exc
    _require(data.get("schema_version") == SCHEMA, "schema drift")
    _require(data.get("worker_id") == WORKER, "worker drift")
    _require(data.get("local_free_only") is True, "LOCAL_FREE boundary weakened")

    authority = data.get("source_authority")
    _require(isinstance(authority, dict), "source_authority missing")
    for key, expected in EXPECTED_AUTHORITY.items():
        _require(authority.get(key) == expected, f"source authority binding drift: {key}")

    source = data.get("source")
    _require(isinstance(source, dict), "source missing")
    _require(source.get("source_id") == EXPECTED_SOURCE_ID, "source id drift")
    _require(source.get("source_family") == EXPECTED_FAMILY, "source family drift")
    _require(source.get("language") == "en", "language drift")
    _require(source.get("upstream_commit") == EXPECTED_COMMIT, "upstream commit drift")
    _require(source.get("path") == EXPECTED_PATH, "source path drift")
    _require(source.get("source_git_blob_sha1") == EXPECTED_BLOB_SHA1, "source blob drift")
    _require(source.get("raw_bytes") == EXPECTED_RAW_BYTES, "raw byte contract drift")
    _require(source.get("raw_sha256") == EXPECTED_RAW_SHA256, "raw hash contract drift")
    normalization = source.get("normalization")
    _require(isinstance(normalization, dict), "normalization missing")
    _require(normalization.get("truncate_chars") == 50000, "truncate rule drift")
    _require(
        normalization.get("normalized_utf8_bytes") == EXPECTED_NORMALIZED_BYTES,
        "normalized byte contract drift",
    )
    _require(
        normalization.get("normalized_sha256") == EXPECTED_NORMALIZED_SHA256,
        "normalized hash contract drift",
    )

    chunking = data.get("chunking")
    _require(isinstance(chunking, dict), "chunking contract missing")
    _require(chunking.get("max_chars") == 1200, "max_chars drift")
    _require(chunking.get("min_chars") == 80, "min_chars drift")
    _require(chunking.get("expected_chunk_count") == EXPECTED_CHUNKS, "chunk count drift")

    quality = data.get("quality_privacy")
    _require(isinstance(quality, dict), "quality/privacy contract missing")
    _require(quality.get("min_chars") == 60, "quality min_chars drift")
    _require(quality.get("max_chars") == 1600, "quality max_chars drift")
    _require(quality.get("min_alpha_ratio") == 0.35, "quality alpha threshold drift")
    for flag in ("reject_control_characters", "reject_email", "reject_phone"):
        _require(quality.get(flag) is True, f"privacy predicate weakened: {flag}")
    _require(
        quality.get("expected_accepted_chunk_count") == EXPECTED_ACCEPTED,
        "accepted chunk count drift",
    )
    _require(
        quality.get("expected_rejected_chunk_count") == EXPECTED_REJECTED,
        "rejected chunk count drift",
    )
    _require(
        quality.get("expected_rejection_reasons") == EXPECTED_REJECTION_REASONS,
        "rejection evidence drift",
    )
    hashes = quality.get("accepted_normalized_sha256_in_order")
    _require(isinstance(hashes, list) and len(hashes) == EXPECTED_ACCEPTED, "accepted hash list invalid")
    _require(len(set(hashes)) == EXPECTED_ACCEPTED, "accepted hash list contains duplicates")
    _require(
        all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes),
        "accepted hash identity invalid",
    )

    historical = data.get("historical_probe_evidence")
    _require(isinstance(historical, dict), "historical probe evidence missing")
    _require(historical.get("head_sha") == "46a70c990dab6ff72bb84ddb54cff1156b491b40", "DATA-228 head drift")
    _require(historical.get("report_blob_sha1") == "19cb931424c81dfef49a72c6c8fcc14f5843e035", "DATA-228 report blob drift")
    _require(historical.get("report_identity_sha256") == "860b5bd9aed72d9bc754a4f73445d18ff3807408a0d6f5a18a83eca14b9f1712", "DATA-228 report identity drift")
    _require(historical.get("accepted_chunk_count") == EXPECTED_ACCEPTED, "DATA-228 accepted count drift")
    _require(historical.get("accepted_utf8_bytes_min") == EXPECTED_ACCEPTED_MIN, "DATA-228 min drift")
    _require(historical.get("accepted_utf8_bytes_max") == EXPECTED_ACCEPTED_MAX, "DATA-228 max drift")
    _require(historical.get("accepted_utf8_bytes_mean") == EXPECTED_ACCEPTED_MEAN, "DATA-228 mean drift")
    _require(
        historical.get("derived_expected_accepted_utf8_bytes_total") == EXPECTED_ACCEPTED_BYTES,
        "derived accepted byte total drift",
    )

    ledger_contract = data.get("ledger_contract")
    _require(isinstance(ledger_contract, dict), "ledger contract missing")
    _require(ledger_contract.get("emit_only_accepted_chunk_records") is True, "rejected records could be emitted")
    _require(ledger_contract.get("emit_rejected_chunk_text") is False, "rejected text emission enabled")
    _require(ledger_contract.get("emit_rejected_chunk_hashes") is False, "rejected hash emission enabled")
    _require(ledger_contract.get("expected_record_count") == EXPECTED_ACCEPTED, "ledger record count drift")
    _require(
        ledger_contract.get("expected_total_eligible_utf8_bytes") == EXPECTED_ACCEPTED_BYTES,
        "ledger byte total drift",
    )
    _require(ledger_contract.get("no_replay") is True, "replay prohibition weakened")
    _require(ledger_contract.get("no_duplicate_hashes") is True, "duplicate prohibition weakened")

    firewall = data.get("purpose_firewall")
    _require(isinstance(firewall, dict), "purpose firewall missing")
    _require(firewall.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "evaluation permission leaked")
    _require(firewall.get("final_test") == "PROHIBITED", "final-test firewall weakened")
    _require(
        firewall.get("canonical_training_capacity_credit")
        == "NOT_YET_PROMOTED_REQUIRES_SUCCESSOR_REGISTRY_CONVERGENCE",
        "ledger prematurely promoted itself to canonical capacity",
    )
    return data


def _fetch_exact_source(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "12-6-ai-NEXT100-101/1.0",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > MAX_SOURCE_BYTES:
                raise LedgerError("source Content-Length exceeds bounded maximum")
            payload = response.read(MAX_SOURCE_BYTES + 1)
    except LedgerError:
        raise
    except Exception as exc:  # network class varies across Python/platform
        raise LedgerError(f"exact source acquisition failed: {exc}") from exc
    _require(len(payload) <= MAX_SOURCE_BYTES, "source exceeds bounded maximum")
    return payload


def _read_source(contract: dict[str, Any], source_file: Path | None) -> bytes:
    if source_file is not None:
        try:
            return source_file.read_bytes()
        except OSError as exc:
            raise LedgerError(f"cannot read source file: {exc}") from exc
    return _fetch_exact_source(str(contract["source"]["acquisition_url"]))


def build_ledger(contract: dict[str, Any], raw: bytes) -> dict[str, Any]:
    source = contract["source"]
    _require(len(raw) == EXPECTED_RAW_BYTES, "source raw byte count mismatch")
    _require(_sha256(raw) == EXPECTED_RAW_SHA256, "source raw SHA-256 mismatch")
    _require(_git_blob_sha1(raw) == EXPECTED_BLOB_SHA1, "source Git blob SHA-1 mismatch")

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LedgerError("source is not strict UTF-8") from exc
    bounded = text[: int(source["normalization"]["truncate_chars"])]
    normalized = _normalize_text(bounded)
    normalized_bytes = normalized.encode("utf-8")
    _require(len(normalized_bytes) == EXPECTED_NORMALIZED_BYTES, "normalized byte count mismatch")
    _require(_sha256(normalized_bytes) == EXPECTED_NORMALIZED_SHA256, "normalized SHA-256 mismatch")

    chunking = contract["chunking"]
    chunks = _chunk_text(
        normalized,
        max_chars=int(chunking["max_chars"]),
        min_chars=int(chunking["min_chars"]),
    )
    _require(len(chunks) == EXPECTED_CHUNKS, "materialized chunk count mismatch")

    expected_hashes = contract["quality_privacy"]["accepted_normalized_sha256_in_order"]
    accepted_records: list[dict[str, Any]] = []
    accepted_hashes: list[str] = []
    rejected: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()

    for index, chunk in enumerate(chunks):
        reason = _quality_reason(chunk)
        if reason is not None:
            rejection_reasons[reason] += 1
            rejected.append({"source_chunk_index": index, "reason": reason})
            continue
        canonical_chunk = _normalize_text(chunk).encode("utf-8")
        chunk_sha = _sha256(canonical_chunk)
        accepted_hashes.append(chunk_sha)
        accepted_records.append(
            {
                "record_id": f"{EXPECTED_SOURCE_ID}#chunk-{index:02d}",
                "source_chunk_index": index,
                "normalized_sha256": chunk_sha,
                "utf8_bytes": len(canonical_chunk),
            }
        )

    _require(len(accepted_records) == EXPECTED_ACCEPTED, "accepted record count mismatch")
    _require(len(rejected) == EXPECTED_REJECTED, "rejected record count mismatch")
    _require(dict(sorted(rejection_reasons.items())) == EXPECTED_REJECTION_REASONS, "rejection reason mismatch")
    _require(accepted_hashes == expected_hashes, "accepted chunk identities/order do not match NEXT100-037")
    _require(len(set(accepted_hashes)) == EXPECTED_ACCEPTED, "accepted chunk hash duplicate")

    byte_values = [int(row["utf8_bytes"]) for row in accepted_records]
    total_bytes = sum(byte_values)
    _require(total_bytes == EXPECTED_ACCEPTED_BYTES, "accepted byte total mismatch")
    _require(min(byte_values) == EXPECTED_ACCEPTED_MIN, "accepted byte minimum mismatch")
    _require(max(byte_values) == EXPECTED_ACCEPTED_MAX, "accepted byte maximum mismatch")
    _require(sum(byte_values) / len(byte_values) == EXPECTED_ACCEPTED_MEAN, "accepted byte mean mismatch")

    core: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA,
        "worker_id": WORKER,
        "local_free_only": True,
        "source_authority": {
            "worker_id": EXPECTED_AUTHORITY["worker_id"],
            "head_sha": EXPECTED_AUTHORITY["head_sha"],
            "workflow_run": EXPECTED_AUTHORITY["workflow_run"],
            "authority_blob_sha1": EXPECTED_AUTHORITY["authority_blob_sha1"],
            "authority_identity_sha256": EXPECTED_AUTHORITY["authority_identity_sha256"],
        },
        "source_identity": {
            "source_id": EXPECTED_SOURCE_ID,
            "source_family": EXPECTED_FAMILY,
            "upstream_commit": EXPECTED_COMMIT,
            "path": EXPECTED_PATH,
            "git_blob_sha1": EXPECTED_BLOB_SHA1,
            "raw_sha256": EXPECTED_RAW_SHA256,
            "normalized_sha256": EXPECTED_NORMALIZED_SHA256,
        },
        "algorithm_binding": {
            "normalization": contract["source"]["normalization"]["name"],
            "chunking": contract["chunking"]["algorithm"],
            "quality_privacy": contract["quality_privacy"]["algorithm"],
        },
        "eligible_record_count": len(accepted_records),
        "eligible_utf8_bytes": total_bytes,
        "eligible_utf8_bytes_min": min(byte_values),
        "eligible_utf8_bytes_max": max(byte_values),
        "eligible_utf8_bytes_mean": sum(byte_values) / len(byte_values),
        "records": accepted_records,
        "rejected_summary": {
            "count": len(rejected),
            "reasons": dict(sorted(rejection_reasons.items())),
            "records": rejected,
            "raw_text_emitted": False,
            "chunk_hashes_emitted": False,
        },
        "purpose_firewall": contract["purpose_firewall"],
        "claim_boundary": contract["claim_boundary"],
    }
    ledger = {
        **core,
        "ledger_identity_sha256": _sha256(_canonical_bytes(core)),
    }
    return ledger


def _write_ledger(ledger: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument(
        "--check-contract-only",
        action="store_true",
        help="Validate immutable contract bindings without acquiring source bytes.",
    )
    args = parser.parse_args(argv)
    try:
        contract = _load_contract(args.config)
        if args.check_contract_only:
            print(json.dumps({"status": "PASS", "worker_id": WORKER}, sort_keys=True))
            return 0
        raw = _read_source(contract, args.source_file)
        ledger = build_ledger(contract, raw)
        _write_ledger(ledger, args.output)
    except LedgerError as exc:
        print(f"NEXT100-101 FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "ledger_identity_sha256": ledger["ledger_identity_sha256"],
                "eligible_record_count": ledger["eligible_record_count"],
                "eligible_utf8_bytes": ledger["eligible_utf8_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
