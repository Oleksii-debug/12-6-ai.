#!/usr/bin/env python3
"""Probe document-specific public-domain CISA/DHS publications without admitting training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

CONFIG_PATH = Path("configs/data/cisa_public_domain_probe_v1.json")
MAX_PDF_BYTES = 32 * 1024 * 1024
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)?")


class ProbeError(RuntimeError):
    """Raised when a source cannot satisfy the fail-closed probe contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(text: str) -> tuple[str, int]:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text, email_count = EMAIL_RE.subn("<EMAIL_REDACTED>", text)

    normalized_lines: list[str] = []
    previous_blank = False
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t\v]+", " ", raw_line).strip()
        if not line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()
    return "\n".join(normalized_lines) + "\n", email_count


def quality_metrics(text: str) -> dict[str, int | float]:
    characters = len(text)
    alphabetic = sum(character.isalpha() for character in text)
    controls = sum(ord(character) < 32 and character != "\n" for character in text)
    return {
        "characters": characters,
        "words": len(WORD_RE.findall(text)),
        "alphabetic_character_ratio": (
            round(alphabetic / characters, 6) if characters else 0.0
        ),
        "unexpected_control_chars": controls,
        "unicode_replacement_chars": text.count("\ufffd"),
    }


def assert_rights_phrases(text: str, required_phrases: list[str]) -> None:
    folded = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
    missing = [
        phrase
        for phrase in required_phrases
        if re.sub(r"\s+", " ", phrase.casefold()).strip() not in folded
    ]
    if missing:
        raise ProbeError(f"document-specific public-domain phrase missing: {missing!r}")


def _download_pdf(url: str, allowed_hosts: set[str]) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "12-6-ai-cisa-source-probe/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        final_url = response.geturl()
        final_host = (urlparse(final_url).hostname or "").lower()
        if final_host not in allowed_hosts:
            raise ProbeError(f"download redirected to unapproved host: {final_host!r}")
        raw = response.read(MAX_PDF_BYTES + 1)
    if len(raw) > MAX_PDF_BYTES:
        raise ProbeError(f"PDF exceeds {MAX_PDF_BYTES} byte probe limit")
    if not raw.startswith(b"%PDF"):
        raise ProbeError("downloaded payload is not a PDF")
    return raw, final_url


def _pdftotext_version() -> str:
    completed = subprocess.run(
        ["pdftotext", "-v"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = (completed.stderr or completed.stdout).strip().splitlines()
    if not output:
        raise ProbeError("pdftotext version could not be determined")
    return output[0]


def _extract_pdf_text(raw: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="twelve-six-cisa-") as tmp:
        pdf_path = Path(tmp) / "source.pdf"
        text_path = Path(tmp) / "source.txt"
        pdf_path.write_bytes(raw)
        subprocess.run(
            [
                "pdftotext",
                "-enc",
                "UTF-8",
                "-nopgbrk",
                str(pdf_path),
                str(text_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return text_path.read_text(encoding="utf-8")


def probe_document(
    document: dict[str, object],
    *,
    allowed_hosts: set[str],
    required_phrases: list[str],
    minimums: dict[str, object],
) -> dict[str, object]:
    publication_id = str(document["publication_id"])
    raw, final_url = _download_pdf(str(document["url"]), allowed_hosts)
    extracted = _extract_pdf_text(raw)
    assert_rights_phrases(extracted, required_phrases)
    normalized, emails_redacted = normalize_text(extracted)
    metrics = quality_metrics(normalized)
    normalized_bytes = len(normalized.encode("utf-8"))

    if normalized_bytes < int(minimums["normalized_utf8_bytes_per_document"]):
        raise ProbeError(f"{publication_id}: normalized payload is below byte minimum")
    if int(metrics["words"]) < int(minimums["word_count_per_document"]):
        raise ProbeError(f"{publication_id}: word count is below minimum")
    if float(metrics["alphabetic_character_ratio"]) < float(
        minimums["alphabetic_character_ratio"]
    ):
        raise ProbeError(f"{publication_id}: alphabetic ratio is below minimum")
    if int(metrics["unexpected_control_chars"]) != 0:
        raise ProbeError(f"{publication_id}: unexpected control characters remain")
    if int(metrics["unicode_replacement_chars"]) != 0:
        raise ProbeError(f"{publication_id}: Unicode replacement characters remain")

    return {
        "publication_id": publication_id,
        "title": str(document["title"]),
        "publisher": str(document["publisher"]),
        "year": int(document["year"]),
        "requested_url": str(document["url"]),
        "final_url": final_url,
        "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_utf8_bytes": normalized_bytes,
        "normalized_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "emails_redacted": emails_redacted,
        "quality": metrics,
        "rights_phrase_check": "PASS_DOCUMENT_SPECIFIC_PUBLIC_DOMAIN",
        "training_credit": 0,
    }


def build_report(config: dict[str, object]) -> dict[str, object]:
    if config.get("status") != "PROBE_ONLY_NO_TRAINING_AUTHORITY":
        raise ProbeError("probe config status must remain fail-closed")
    boundary = dict(config["truth_boundary"])
    if boundary.get("training_authorized_exposure") != 0:
        raise ProbeError("probe config must authorize zero training exposure")
    if boundary.get("corpus_admitted") is not False:
        raise ProbeError("probe config must not admit a corpus")
    if boundary.get("paid_compute_authorized") is not False:
        raise ProbeError("probe config must not authorize paid compute")

    allowed_hosts = {str(host).lower() for host in config["allowed_download_hosts"]}
    required = [str(value) for value in config["required_rights_phrases_casefold"]]
    minimums = dict(config["quality_minimums"])
    documents = [
        probe_document(
            dict(document),
            allowed_hosts=allowed_hosts,
            required_phrases=required,
            minimums=minimums,
        )
        for document in config["documents"]
    ]
    report: dict[str, object] = {
        "schema": "twelve-six.cisa-public-domain-probe-report.v1",
        "status": "PROBE_PASS_REQUIRES_TERMINAL_AUTHORITY",
        "source_family": config["source_family"],
        "language": config["language"],
        "modality": config["modality"],
        "pdftotext_version": _pdftotext_version(),
        "documents": documents,
        "candidate_normalized_utf8_bytes": sum(
            int(document["normalized_utf8_bytes"]) for document in documents
        ),
        "training_authorized_exposure": 0,
        "corpus_admitted": False,
        "truth_boundary": boundary,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_report(config)
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
