#!/usr/bin/env python3
"""DATA-228 immutable UA/EN diversity source probe.

The probe acquires only commit-pinned public source objects and their license
texts, computes content identities, applies deterministic extraction and the
incumbent D03 privacy/quality preview, and writes machine-readable evidence.
It deliberately has no project-package or torch dependency so rights/data
probing remains LOCAL_FREE and independent of model runtime installation.
It does not itself authorize training; a DATA-24 registry decision is required
before admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

MAX_BYTES_DEFAULT = 2_000_000
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")


@dataclass(frozen=True)
class _D03PreviewConfig:
    min_chars: int = 60
    max_chars: int = 1600
    min_alpha_ratio: float = 0.35


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _fetch(url: str, max_bytes: int) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "12-6-ai-DATA-228/1.0", "Accept-Encoding": "identity"},
    )
    with urlopen(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > max_bytes:
            raise RuntimeError(f"oversized source: {url}")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError(f"oversized source: {url}")
    return payload


def _normalize_text(text: str) -> str:
    """Exact incumbent D03 natural-text normalization identity."""
    text = unicodedata.normalize(
        "NFKC", text.replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _quality_reason(text: str, config: _D03PreviewConfig) -> str | None:
    """Exact DATA-181/D03 quality/privacy predicate needed by this probe."""
    if len(text) < config.min_chars:
        return "too_short"
    if len(text) > config.max_chars:
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
    if alpha_ratio < config.min_alpha_ratio:
        return "low_alpha_ratio"
    return None


def _chunk_text(text: str, *, max_chars: int = 1200, min_chars: int = 80) -> tuple[str, ...]:
    """Exact DATA-181 generic natural-text chunking semantics."""
    if max_chars < min_chars or min_chars < 20:
        raise RuntimeError("invalid chunk limits")
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


def _markdown_visible(text: str) -> str:
    """Extract visible Markdown while dropping translation-source comments."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5 :]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{<.*?>\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\{\{%.*?%\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_#>|]+", " ", text)
    return text


NORMALIZATION_DESCRIPTORS = {
    "plain_text": {
        "name": "plain_text",
        "decode": "strict-utf8",
        "truncate_chars": 50000,
        "normalize": "NFKC+LF+per-line-whitespace-collapse+drop-empty-lines+strip",
    },
    "markdown_visible_v1": {
        "name": "markdown_visible_v1",
        "decode": "strict-utf8",
        "extract": [
            "strip-leading-yaml-frontmatter",
            "strip-html-comments",
            "strip-hugo-shortcodes",
            "retain-markdown-link-labels",
            "strip-basic-markdown-punctuation",
        ],
        "truncate_chars": 50000,
        "normalize": "NFKC+LF+per-line-whitespace-collapse+drop-empty-lines+strip",
    },
}


def run_probe(
    *,
    config_path: str | Path,
    report_path: str | Path,
    evidence_dir: str | Path,
    max_bytes: int = MAX_BYTES_DEFAULT,
) -> dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "12-6.data228-source-probe.v1":
        raise RuntimeError("unsupported DATA-228 source probe schema")
    if config.get("local_free_only") is not True:
        raise RuntimeError("DATA-228 probe must remain LOCAL_FREE")

    evidence_root = Path(evidence_dir)
    evidence_root.mkdir(parents=True, exist_ok=True)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quality_config = _D03PreviewConfig()

    reports: list[dict[str, object]] = []
    for item in config["candidates"]:
        adapter = item["adapter"]
        if adapter not in NORMALIZATION_DESCRIPTORS:
            raise RuntimeError(f"unsupported adapter {adapter!r}")
        raw = _fetch(item["acquisition_url"], max_bytes)
        license_raw = _fetch(item["license_url"], max_bytes)
        text = raw.decode("utf-8", errors="strict")
        extracted = _markdown_visible(text) if adapter == "markdown_visible_v1" else text
        bounded = extracted[:50_000]
        normalized = _normalize_text(bounded)
        normalized_bytes = normalized.encode("utf-8")
        chunks = _chunk_text(normalized)

        quality_reasons: dict[str, int] = {}
        accepted_lengths: list[int] = []
        accepted_sha: list[str] = []
        for chunk in chunks:
            reason = _quality_reason(chunk, quality_config)
            if reason is not None:
                quality_reasons[reason] = quality_reasons.get(reason, 0) + 1
                continue
            encoded = chunk.encode("utf-8")
            accepted_lengths.append(len(encoded))
            accepted_sha.append(_sha256(_normalize_text(chunk).encode("utf-8")))

        if len(accepted_sha) != len(set(accepted_sha)):
            raise RuntimeError(f"{item['source_id']}: exact duplicate chunks in candidate")
        if not accepted_lengths:
            raise RuntimeError(
                f"{item['source_id']}: no chunks pass privacy/quality preview"
            )

        evidence_name = (
            "kubernetes-website-cc-by-4.0-25f3dcb.txt"
            if item["language"] == "uk"
            else "cpython-psf-license-7f0ccd6.txt"
        )
        evidence_path = evidence_root / evidence_name
        evidence_path.write_bytes(license_raw)

        descriptor = NORMALIZATION_DESCRIPTORS[adapter]
        reports.append(
            {
                **item,
                "raw_sha256": _sha256(raw),
                "raw_bytes": len(raw),
                "git_blob_sha1": hashlib.sha1(
                    b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
                ).hexdigest(),
                "license_evidence_path": evidence_path.as_posix(),
                "license_evidence_sha256": _sha256(license_raw),
                "license_evidence_bytes": len(license_raw),
                "normalization_descriptor": descriptor,
                "normalization_identity_sha256": _sha256(_canonical(descriptor)),
                "normalized_sha256": _sha256(normalized_bytes),
                "normalized_utf8_bytes": len(normalized_bytes),
                "privacy_quality_preview": {
                    "chunk_count": len(chunks),
                    "accepted_chunk_count": len(accepted_lengths),
                    "rejected_chunk_count": len(chunks) - len(accepted_lengths),
                    "rejection_reasons": quality_reasons,
                    "accepted_document_utf8_bytes": {
                        "min": min(accepted_lengths),
                        "max": max(accepted_lengths),
                        "mean": sum(accepted_lengths) / len(accepted_lengths),
                    },
                    "accepted_normalized_sha256": accepted_sha,
                    "exact_duplicate_chunks": 0,
                },
            }
        )

    family_keys = {(item["language"], item["source_family"]) for item in reports}
    if len(family_keys) != len(reports):
        raise RuntimeError("candidate source families are not independent by language")
    incumbent_families = {"rada.open-data.laws-texts", "standardebooks.manual"}
    if any(item["source_family"] in incumbent_families for item in reports):
        raise RuntimeError("candidate aliases an incumbent source family")

    core: dict[str, object] = {
        "schema_version": "12-6.data228-source-probe-report.v1",
        "status": "PASS",
        "local_free_only": True,
        "authority_boundary": "DATA_181_BASELINE_USED_BECAUSE_DATA_213_PUBLICATION_NOT_DISCOVERABLE",
        "baseline_source_sha": config["baseline"]["source_sha"],
        "d03_preview_semantics": {
            "source_commit": config["baseline"]["source_sha"],
            "chunking": {"max_chars": 1200, "min_chars": 80},
            "quality": {
                "min_chars": quality_config.min_chars,
                "max_chars": quality_config.max_chars,
                "min_alpha_ratio": quality_config.min_alpha_ratio,
                "reject_control_characters": True,
                "reject_email": True,
                "reject_phone": True,
            },
        },
        "candidates": reports,
    }
    report = {**core, "report_sha256": _sha256(_canonical(core))}
    destination.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/data/data228_source_probe_v1.json"
    )
    parser.add_argument("--report", default="reports/data228/source-probe.json")
    parser.add_argument(
        "--evidence-dir", default="data/external/rights-evidence/data228"
    )
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES_DEFAULT)
    args = parser.parse_args()
    report = run_probe(
        config_path=args.config,
        report_path=args.report,
        evidence_dir=args.evidence_dir,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
