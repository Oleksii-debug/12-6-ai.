"""Conservative, traceable Ukrainian/natural-text normalization.

The policy intentionally prefers canonical-equivalence cleanup over compatibility
folding. Natural text and code use different normalization paths: code receives
only newline canonicalization so indentation, spaces, Unicode literals, and other
layout-sensitive bytes are not rewritten by natural-text cleanup.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

NORMALIZATION_SCHEMA = "12-6.ua-normalization-v1"
NORMALIZATION_UNICODE_FORM = "NFC"

_BLOCK_BREAK_RE = re.compile(
    r"(?is)<\s*(?:br|/p|/div|/li|/h[1-6]|/tr)\s*/?\s*>"
)
_HTML_TAG_RE = re.compile(r"(?is)<\s*/?\s*[a-z][^<>]*>")
_LEADING_TRAILING_BLANK_RE = re.compile(r"^(?:[ \t]*\n)+|(?:\n[ \t]*)+$")
_TRAILING_HORIZONTAL_RE = re.compile(r"[ \t]+(?=\n|$)")


class NormalizationError(ValueError):
    """Raised when text cannot enter the deterministic normalization contract."""


@dataclass(frozen=True)
class NormalizationTrace:
    schema: str
    modality: Literal["natural", "code"]
    source_id: str | None
    source_version: str | None
    raw_document_id: str | None
    raw_source_sha256: str | None
    raw_text_sha256: str
    normalized_text_sha256: str
    raw_codepoints: int
    normalized_codepoints: int
    raw_utf8_bytes: int
    normalized_utf8_bytes: int
    byte_token_delta: int
    reason_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    trace: NormalizationTrace


@dataclass(frozen=True)
class ChangeSummary:
    documents: int
    changed_documents: int
    raw_codepoints: int
    normalized_codepoints: int
    codepoint_delta: int
    raw_byte_tokens: int
    normalized_byte_tokens: int
    byte_token_delta: int
    reason_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if "\ufffd" in text:
        raise NormalizationError("replacement character U+FFFD is forbidden")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise NormalizationError("surrogate code points are forbidden")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise NormalizationError("text is not strict UTF-8 encodable") from exc


def _replace_with_count(text: str, old: str, new: str, reason: str, reasons: Counter[str]) -> str:
    count = text.count(old)
    if count:
        reasons[reason] += count
        text = text.replace(old, new)
    return text


def _normalize_newlines(text: str, reasons: Counter[str]) -> str:
    crlf = text.count("\r\n")
    if crlf:
        reasons["crlf_to_lf"] += crlf
        text = text.replace("\r\n", "\n")
    cr = text.count("\r")
    if cr:
        reasons["cr_to_lf"] += cr
        text = text.replace("\r", "\n")
    return text


def _normalize_natural(text: str, reasons: Counter[str]) -> str:
    text = _normalize_newlines(text, reasons)

    # A UTF-8 BOM decoded into text is transport residue, not Ukrainian content.
    if text.startswith("\ufeff"):
        reasons["leading_bom_removed"] += 1
        text = text[1:]

    # Decode entities before whitespace cleanup; this turns &nbsp; into U+00A0.
    unescaped = html.unescape(text)
    if unescaped != text:
        reasons["html_entities_unescaped"] += 1
        text = unescaped

    block_matches = len(_BLOCK_BREAK_RE.findall(text))
    if block_matches:
        reasons["html_block_breaks_to_lf"] += block_matches
        text = _BLOCK_BREAK_RE.sub("\n", text)
    tag_matches = len(_HTML_TAG_RE.findall(text))
    if tag_matches:
        reasons["html_tags_removed"] += tag_matches
        text = _HTML_TAG_RE.sub("", text)

    text = _replace_with_count(text, "\u00a0", " ", "nbsp_to_space", reasons)
    text = _replace_with_count(text, "\u202f", " ", "narrow_nbsp_to_space", reasons)
    text = _replace_with_count(text, "\u00ad", "", "soft_hyphen_removed", reasons)

    # NFC composes canonically equivalent sequences such as decomposed ї/й while
    # avoiding the compatibility folding performed by the incumbent NFKC path.
    nfc = unicodedata.normalize(NORMALIZATION_UNICODE_FORM, text)
    if nfc != text:
        reasons["unicode_nfc"] += 1
        text = nfc

    trailing_matches = len(_TRAILING_HORIZONTAL_RE.findall(text))
    if trailing_matches:
        reasons["trailing_horizontal_space_removed"] += trailing_matches
        text = _TRAILING_HORIZONTAL_RE.sub("", text)

    trimmed = _LEADING_TRAILING_BLANK_RE.sub("", text)
    if trimmed != text:
        reasons["outer_blank_lines_removed"] += 1
        text = trimmed
    return text


def _normalize_code(text: str, reasons: Counter[str]) -> str:
    # Do not NFC/NFKC code, decode HTML entities, remove soft hyphens, normalize
    # NBSP, trim indentation, collapse spaces, or strip lines. Source bytes may
    # carry semantics in identifiers, string literals, indentation, or fixtures.
    return _normalize_newlines(text, reasons)


def normalize_document(
    text: str,
    *,
    modality: Literal["natural", "code"] = "natural",
    source_id: str | None = None,
    source_version: str | None = None,
    raw_document_id: str | None = None,
    raw_source_sha256: str | None = None,
) -> NormalizationResult:
    """Normalize deterministically and emit before/after identity metadata."""
    _validate_text(text)
    if modality not in {"natural", "code"}:
        raise NormalizationError("modality must be natural or code")

    reasons: Counter[str] = Counter()
    normalized = (
        _normalize_code(text, reasons)
        if modality == "code"
        else _normalize_natural(text, reasons)
    )
    _validate_text(normalized)

    raw_bytes = len(text.encode("utf-8"))
    normalized_bytes = len(normalized.encode("utf-8"))
    trace = NormalizationTrace(
        schema=NORMALIZATION_SCHEMA,
        modality=modality,
        source_id=source_id,
        source_version=source_version,
        raw_document_id=raw_document_id,
        raw_source_sha256=raw_source_sha256,
        raw_text_sha256=_sha256_text(text),
        normalized_text_sha256=_sha256_text(normalized),
        raw_codepoints=len(text),
        normalized_codepoints=len(normalized),
        raw_utf8_bytes=raw_bytes,
        normalized_utf8_bytes=normalized_bytes,
        byte_token_delta=normalized_bytes - raw_bytes,
        reason_counts=dict(sorted(reasons.items())),
    )
    return NormalizationResult(normalized, trace)


def summarize_changes(results: tuple[NormalizationResult, ...]) -> ChangeSummary:
    """Aggregate character/current-byte-token effects for an audited sample."""
    reasons: Counter[str] = Counter()
    for result in results:
        reasons.update(result.trace.reason_counts)
    raw_codepoints = sum(result.trace.raw_codepoints for result in results)
    normalized_codepoints = sum(result.trace.normalized_codepoints for result in results)
    raw_tokens = sum(result.trace.raw_utf8_bytes for result in results)
    normalized_tokens = sum(result.trace.normalized_utf8_bytes for result in results)
    return ChangeSummary(
        documents=len(results),
        changed_documents=sum(
            result.trace.raw_text_sha256 != result.trace.normalized_text_sha256
            for result in results
        ),
        raw_codepoints=raw_codepoints,
        normalized_codepoints=normalized_codepoints,
        codepoint_delta=normalized_codepoints - raw_codepoints,
        raw_byte_tokens=raw_tokens,
        normalized_byte_tokens=normalized_tokens,
        byte_token_delta=normalized_tokens - raw_tokens,
        reason_counts=dict(sorted(reasons.items())),
    )


def normalize_natural_text(text: str) -> str:
    """Compatibility wrapper for callers that need only normalized natural text."""
    return normalize_document(text, modality="natural").text


def normalize_code_text(text: str) -> str:
    """Compatibility wrapper for layout-sensitive code text."""
    return normalize_document(text, modality="code").text
