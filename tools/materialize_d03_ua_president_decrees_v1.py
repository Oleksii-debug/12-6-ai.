#!/usr/bin/env python3
"""Bounded LOCAL_FREE intake for official Ukrainian Presidential decrees.

The tool deliberately produces body-free evidence only. It discovers decree URLs on the
President of Ukraine's official catalogue, fetches each selected decree twice, extracts
only the official decree text, applies conservative privacy/quality gates, and records
hashes/byte counts. No corpus or training credit is granted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, build_opener, HTTPSHandler, HTTPRedirectHandler

BASE: Final = "https://www.president.gov.ua"
CATALOG: Final = f"{BASE}/documents/decrees/"
DOC_TITLE_RE: Final = re.compile(r"^УКАЗ ПРЕЗИДЕНТА УКРАЇНИ №\s*\d+/\d{4}$", re.IGNORECASE)
DOC_PATH_RE: Final = re.compile(r"^/documents/[a-z0-9-]*\d[a-z0-9-]*$", re.IGNORECASE)
PAGE_RE: Final = re.compile(r"(?:^|[?&])page=(\d+)(?:&|$)")
FOOTER_MARKERS: Final = {"Новини", "Фото", "Відео", "Документи"}
SKIP_TAGS: Final = {"script", "style", "noscript", "svg"}
BLOCK_TAGS: Final = {
    "p", "div", "section", "article", "header", "footer", "main", "aside", "li",
    "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "td", "th",
}
SUBJECT_DENY_RE: Final = re.compile(
    r"(?i)("
    r"призначення|звільнення|відзначення|нагороджен|присвоєння|"
    r"громадянств|помилуван|стипенді|персональн|"
    r"спеціальн(?:их|і) економічн(?:их|і).*обмежувальн|санкц|"
    r"склад(?:у)? .*коміс|склад(?:у)? .*делегац|"
    r"призначити|звільнити"
    r")"
)
EMAIL_RE: Final = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE: Final = re.compile(
    r"(?<!\d)(?:\+?38[\s().-]*)?0[\s().-]*\d{2}[\s().-]*"
    r"\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)"
)
LONG_DIGITS_RE: Final = re.compile(r"(?<!\d)\d{8,}(?!\d)")
ADDRESS_RE: Final = re.compile(
    r"(?i)\b(?:вул\.|вулиц[яі]|буд\.|будинок|квартир[аи]|пров\.|провулок)\b"
)
NAME_INITIAL_RE: Final = re.compile(
    r"\b[А-ЯІЇЄҐ][а-яіїєґ'’\-]{2,}\s+[А-ЯІЇЄҐ]\.\s*[А-ЯІЇЄҐ]\."
)
SIGNATORY_RE: Final = re.compile(r"^Президент України\b", re.IGNORECASE)
USER_AGENT: Final = "12-6-ai-D03-president-decrees-intake/1.0 (+LOCAL_FREE research; bounded)"


class TextAndLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._href_stack: list[str | None] = []
        self._anchor_text: list[list[str]] = []
        self.links: list[tuple[str, str]] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        if self._skip_depth:
            return
        if tag in BLOCK_TAGS:
            self._text_parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            self._href_stack.append(href)
            self._anchor_text.append([])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a" and self._href_stack:
            href = self._href_stack.pop()
            text = " ".join(self._anchor_text.pop()).strip()
            if href:
                self.links.append((href, text))
        if tag in BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._text_parts.append(data)
        if self._anchor_text:
            self._anchor_text[-1].append(data)

    def lines(self) -> list[str]:
        text = "".join(self._text_parts).replace("\xa0", " ")
        out: list[str] = []
        for raw in text.splitlines():
            value = " ".join(raw.split())
            if value:
                out.append(value)
        return out


class SameOriginRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname != "www.president.gov.ua":
            raise RuntimeError(f"cross-origin redirect refused: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class FetchResult:
    requested_url: str
    final_url: str
    content_type: str
    body: bytes


class Fetcher:
    def __init__(self, delay_seconds: float, max_bytes: int = 8_000_000) -> None:
        self.delay_seconds = delay_seconds
        self.max_bytes = max_bytes
        self._last_request = 0.0
        context = ssl.create_default_context()
        self._opener = build_opener(SameOriginRedirect(), HTTPSHandler(context=context))

    def fetch(self, url: str) -> FetchResult:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "www.president.gov.ua":
            raise RuntimeError(f"refusing non-canonical URL: {url}")
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        with self._opener.open(request, timeout=45) as response:
            self._last_request = time.monotonic()
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            data = response.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise RuntimeError(f"response exceeds {self.max_bytes} bytes: {url}")
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise RuntimeError(f"unexpected content type {content_type}: {url}")
        final = urlparse(final_url)
        if final.scheme != "https" or final.hostname != "www.president.gov.ua":
            raise RuntimeError(f"final URL left canonical origin: {final_url}")
        return FetchResult(url, final_url, content_type, data)


def parse_html(data: bytes) -> TextAndLinkParser:
    text = data.decode("utf-8", errors="strict")
    parser = TextAndLinkParser()
    parser.feed(text)
    parser.close()
    return parser


def canonical_url(href: str, base_url: str) -> str | None:
    url = urljoin(base_url, href)
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.president.gov.ua":
        return None
    return parsed._replace(fragment="").geturl()


def discover_document_urls(data: bytes, page_url: str) -> list[str]:
    parser = parse_html(data)
    urls: list[str] = []
    seen: set[str] = set()
    for href, anchor_text in parser.links:
        if not DOC_TITLE_RE.match(" ".join(anchor_text.split())):
            continue
        url = canonical_url(href, page_url)
        if not url:
            continue
        if not DOC_PATH_RE.match(urlparse(url).path):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def discover_next_catalog_url(data: bytes, page_url: str, current_page: int) -> str | None:
    parser = parse_html(data)
    choices: list[tuple[int, str]] = []
    for href, _text in parser.links:
        url = canonical_url(href, page_url)
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.path.rstrip("/") != "/documents/decrees":
            continue
        page_values = parse_qs(parsed.query).get("page")
        if not page_values or not page_values[0].isdigit():
            continue
        page = int(page_values[0])
        if page > current_page:
            choices.append((page, url))
    if not choices:
        return None
    choices.sort(key=lambda item: item[0])
    return choices[0][1]


def normalize_line(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\u00a0", " "))
    return " ".join(value.split()).strip()


def extract_document(data: bytes) -> tuple[str, str, list[str]]:
    parser = parse_html(data)
    lines = [normalize_line(line) for line in parser.lines()]
    title_indexes = [i for i, line in enumerate(lines) if DOC_TITLE_RE.match(line)]
    if not title_indexes:
        raise ValueError("decree title marker not found")
    start = title_indexes[-1]
    body_lines: list[str] = []
    for line in lines[start:]:
        if body_lines and line in FOOTER_MARKERS:
            break
        if line.startswith("Офіс Президента України. Всі матеріали"):
            break
        body_lines.append(line)
    if len(body_lines) < 3:
        raise ValueError("official decree body too short to parse")
    title = body_lines[0]
    subject = body_lines[1]
    cleaned = [line for line in body_lines if not SIGNATORY_RE.match(line)]
    return title, subject, cleaned


def normalize_document(lines: list[str]) -> bytes:
    normalized = "\n".join(normalize_line(line) for line in lines if normalize_line(line)) + "\n"
    return normalized.encode("utf-8")


def privacy_reasons(subject: str, text: str) -> list[str]:
    reasons: list[str] = []
    if SUBJECT_DENY_RE.search(subject):
        reasons.append("subject_personnel_or_personal_measure")
    if EMAIL_RE.search(text):
        reasons.append("email")
    if PHONE_RE.search(text):
        reasons.append("phone")
    if ADDRESS_RE.search(text):
        reasons.append("street_address")
    if LONG_DIGITS_RE.search(text):
        reasons.append("long_identifier")
    if len(NAME_INITIAL_RE.findall(text)) >= 4:
        reasons.append("high_person_name_density")
    return sorted(set(reasons))


def quality_reasons(payload: bytes) -> list[str]:
    text = payload.decode("utf-8")
    reasons: list[str] = []
    if len(payload) < 1500:
        reasons.append("below_minimum_bytes")
    ua_letters = len(re.findall(r"[А-Яа-яІіЇїЄєҐґ]", text))
    latin_letters = len(re.findall(r"[A-Za-z]", text))
    denom = ua_letters + latin_letters
    if denom == 0 or ua_letters / denom < 0.85:
        reasons.append("ua_letter_ratio")
    control = sum(ord(ch) < 32 and ch not in "\n\t\r" for ch in text)
    if control:
        reasons.append("control_characters")
    return reasons


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def probe_document(fetcher: Fetcher, url: str) -> dict[str, object]:
    first = fetcher.fetch(url)
    second = fetcher.fetch(url)
    title1, subject1, lines1 = extract_document(first.body)
    title2, subject2, lines2 = extract_document(second.body)
    payload1 = normalize_document(lines1)
    payload2 = normalize_document(lines2)
    if title1 != title2 or subject1 != subject2 or payload1 != payload2:
        return {
            "url": url,
            "accepted": False,
            "reason": "double_fetch_extracted_payload_mismatch",
            "raw_sha256_a": sha256(first.body),
            "raw_sha256_b": sha256(second.body),
        }
    text = payload1.decode("utf-8")
    reasons = privacy_reasons(subject1, text) + quality_reasons(payload1)
    title_match = re.search(r"№\s*(\d+/\d{4})", title1)
    record: dict[str, object] = {
        "url": url,
        "document_number": title_match.group(1) if title_match else None,
        "raw_sha256_a": sha256(first.body),
        "raw_sha256_b": sha256(second.body),
        "raw_byte_identical": first.body == second.body,
        "normalized_sha256": sha256(payload1),
        "normalized_bytes": len(payload1),
        "accepted": not reasons,
    }
    if reasons:
        record["reason"] = "+".join(sorted(set(reasons)))
    return record


def run(max_pages: int, max_documents: int, delay_seconds: float) -> dict[str, object]:
    fetcher = Fetcher(delay_seconds=delay_seconds)
    current_url = CATALOG
    current_page = 1
    document_urls: list[str] = []
    seen_documents: set[str] = set()
    catalog_pages: list[dict[str, object]] = []

    for _ in range(max_pages):
        a = fetcher.fetch(current_url)
        b = fetcher.fetch(current_url)
        urls_a = discover_document_urls(a.body, a.final_url)
        urls_b = discover_document_urls(b.body, b.final_url)
        if urls_a != urls_b:
            raise RuntimeError(f"catalog discovery mismatch across double fetch: {current_url}")
        catalog_pages.append(
            {
                "url": current_url,
                "page": current_page,
                "document_count": len(urls_a),
                "document_url_set_sha256": sha256("\n".join(urls_a).encode()),
            }
        )
        for url in urls_a:
            if url not in seen_documents:
                seen_documents.add(url)
                document_urls.append(url)
                if len(document_urls) >= max_documents:
                    break
        if len(document_urls) >= max_documents:
            break
        next_url = discover_next_catalog_url(a.body, a.final_url, current_page)
        if not next_url:
            break
        next_page_values = parse_qs(urlparse(next_url).query).get("page", [])
        if not next_page_values or not next_page_values[0].isdigit():
            raise RuntimeError(f"invalid next-page URL: {next_url}")
        next_page = int(next_page_values[0])
        if next_page <= current_page:
            raise RuntimeError(f"non-monotonic pagination: {next_url}")
        current_page, current_url = next_page, next_url

    records: list[dict[str, object]] = []
    duplicate_payloads: set[str] = set()
    accepted_bytes = 0
    accepted_count = 0
    rejected = Counter()
    for url in document_urls:
        try:
            record = probe_document(fetcher, url)
        except Exception as exc:  # fail closed per object, retain only bounded machine reason
            record = {
                "url": url,
                "accepted": False,
                "reason": f"fetch_or_parse:{type(exc).__name__}",
            }
        if record.get("accepted"):
            digest = str(record["normalized_sha256"])
            if digest in duplicate_payloads:
                record["accepted"] = False
                record["reason"] = "normalized_exact_duplicate"
            else:
                duplicate_payloads.add(digest)
                accepted_count += 1
                accepted_bytes += int(record["normalized_bytes"])
        if not record.get("accepted"):
            rejected[str(record.get("reason", "unknown"))] += 1
        records.append(record)

    evidence: dict[str, object] = {
        "schema": "12-6.d03-ua-president-decrees-intake.v1",
        "status": "PASS_OBSERVED_YIELD" if accepted_count else "PASS_ZERO_YIELD",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "source_id": "ua.president.official-decrees",
            "family_id": "ua.president.official-decrees",
            "stratum": "uk",
            "catalog_url": CATALOG,
            "allowed_origin": BASE,
            "rights_scope": "OFFICIAL_DECREE_TEXT_ONLY",
            "website_blanket_license_not_used_as_training_authority": True,
            "ukraine_copyright_law": {
                "law": "2811-IX",
                "article": "8(1)(3)",
                "authority_url": "https://zakon.rada.gov.ua/laws/show/2811-20",
                "scope": (
                    "official acts including decrees; no blanket right for "
                    "site chrome/news/third-party material"
                ),
            },
        },
        "execution": {
            "class": "LOCAL_FREE",
            "max_catalog_pages": max_pages,
            "max_documents": max_documents,
            "request_delay_seconds": delay_seconds,
            "catalog_pages_observed": len(catalog_pages),
            "documents_probed": len(records),
            "double_fetch_required": True,
            "final_test_accessed": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
        },
        "catalog_pages": catalog_pages,
        "observed_yield": {
            "accepted_documents": accepted_count,
            "accepted_normalized_bytes": accepted_bytes,
            "one_conservative_family": True,
            "rejected_documents": len(records) - accepted_count,
            "rejection_counts": dict(sorted(rejected.items())),
        },
        "records": records,
        "downstream_required": [
            "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
            "RESERVED_EVALUATION_DECONTAMINATION",
            "POST_COMPOSITION_QUALITY_PRIVACY_BALANCE_FAMILY_CAPS",
            "CLUSTER_SAFE_SPLIT",
            "DETERMINISTIC_TOKENIZER_PACKING_DOUBLE_BUILD",
            "POSITIVE_EXACT_UNIQUE_CAUSAL_LOSS_LEDGER",
        ],
        "claims": {
            "canonical_capacity_credit_bytes": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "tokenizer_fit_authorized": False,
            "research_corpus_released": False,
            "learned_20m_claim": False,
        },
    }
    identity_view = dict(evidence)
    identity_view.pop("generated_at_utc", None)
    canonical = json.dumps(
        identity_view, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    evidence["evidence_identity_sha256"] = sha256(canonical)

    stable_records = []
    for record in records:
        stable_records.append(
            {
                key: record[key]
                for key in (
                    "url",
                    "document_number",
                    "normalized_sha256",
                    "normalized_bytes",
                    "accepted",
                    "reason",
                )
                if key in record
            }
        )
    materialization_view = {
        "source_id": "ua.president.official-decrees",
        "catalog_pages": catalog_pages,
        "observed_yield": evidence["observed_yield"],
        "records": stable_records,
    }
    materialization_bytes = json.dumps(
        materialization_view,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    evidence["materialization_identity_sha256"] = sha256(materialization_bytes)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--max-documents", type=int, default=60)
    parser.add_argument("--request-delay", type=float, default=1.05)
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 8:
        parser.error("--max-pages must be in [1,8]")
    if not 1 <= args.max_documents <= 120:
        parser.error("--max-documents must be in [1,120]")
    if args.request_delay < 1.0:
        parser.error("--request-delay must be >= 1.0 seconds")
    evidence = run(args.max_pages, args.max_documents, args.request_delay)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence["observed_yield"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
