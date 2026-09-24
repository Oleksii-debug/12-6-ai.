#!/usr/bin/env python3
"""Bounded, fail-closed capacity probe for the already-admitted KMu Secretariat family.

This tool does not admit training bytes. It measures a hash-bound, current-site sample
under the exact NEXT100-026 lineage rules so a later locked materializer can decide
whether this family is worth scaling toward the learned-20M data floor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

SCHEMA = "12-6.kmu-secretariat-bulk-probe.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."
PARENT_PR = 449
PARENT_HEAD = "40950a950b60921fd856af2719e1ae2486d9e892"
PARENT_MANIFEST = "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9"
FAMILY_ID = "ua.kmu.portal.secretariat-news"
SITEMAP_URL = "https://www.kmu.gov.ua/sitemap.xml"
SITE_ORIGIN = "https://www.kmu.gov.ua"
SECRETARIAT_BYLINE = (
    "Департамент інформації та комунікацій з громадськістю "
    "Секретаріату Кабінету Міністрів України"
)
LICENSE_MARKER = "Creative Commons Attribution 4.0 International"
UA_STRATUM_TARGET_BYTES = 9_000_000
FAMILY_STRATUM_CAP = 0.60
FAMILY_CAP_BYTES = int(UA_STRATUM_TARGET_BYTES * FAMILY_STRATUM_CAP)
DEFAULT_MAX_FETCHES = 180
DEFAULT_DELAY_SECONDS = 1.05
USER_AGENT = "12-6-ai-kmu-capacity-probe/1.0 (+LOCAL_FREE research; respects robots.txt)"

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+?38)?\s?\(?0\d{2}\)?(?:[\s.-]*\d){7}\b")
_SPACE_RE = re.compile(r"[ \t\f\v]+")
_EXCLUDE_URL_PARTS = (
    "/npas/",
    "/npasearch",
    "/document/",
    "/kmu/",
    "/storage/",
    "/backend/",
    "/api/",
    "/kontakt",
    "/contact",
    "/zvernenn",
    "/appeal",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(data)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw in value.split("\n"):
        line = _SPACE_RE.sub(" ", raw).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def cyrillic_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    cyr = sum("\u0400" <= char <= "\u052f" for char in letters)
    return cyr / len(letters)


def ukrainian_specific_count(text: str) -> int:
    return sum(char.lower() in "іїєґ" for char in text)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_p = 0
        self._p_parts: list[str] = []
        self.paragraphs: list[str] = []
        self.visible_parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "p":
            self._in_p += 1
            self._p_parts = []
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "p" and self._in_p:
            text = normalize_text(" ".join(self._p_parts))
            if text:
                self.paragraphs.append(text)
            self._p_parts = []
            self._in_p -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = normalize_text(data)
        if not text:
            return
        self.visible_parts.append(text)
        if self._in_p:
            self._p_parts.append(text)
        if self._in_title:
            self.title_parts.append(text)


@dataclass(frozen=True)
class SitemapEntry:
    url: str
    lastmod: str


def fetch_bytes(url: str, timeout: float = 20.0) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xml,text/xml,*/*;q=0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        final_url = response.geturl()
        if not final_url.startswith(SITE_ORIGIN):
            raise RuntimeError(f"cross-origin redirect rejected: {final_url}")
        data = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
    return data, headers


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def discover_sitemap_entries(root_url: str = SITEMAP_URL, max_sitemaps: int = 64) -> tuple[list[SitemapEntry], list[dict[str, Any]]]:
    pending = [root_url]
    seen: set[str] = set()
    entries: dict[str, SitemapEntry] = {}
    snapshots: list[dict[str, Any]] = []
    while pending:
        url = pending.pop(0)
        if url in seen:
            continue
        if len(seen) >= max_sitemaps:
            raise RuntimeError("sitemap fanout exceeds bounded probe limit")
        seen.add(url)
        data, headers = fetch_bytes(url)
        snapshots.append(
            {
                "url": url,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "content_type": headers.get("content-type"),
            }
        )
        root = ET.fromstring(data)
        root_name = _local_name(root.tag)
        if root_name == "sitemapindex":
            for node in root:
                loc = next((child.text for child in node if _local_name(child.tag) == "loc"), None)
                if not loc:
                    continue
                child_url = loc.strip()
                parsed = urllib.parse.urlsplit(child_url)
                if parsed.scheme != "https" or parsed.netloc not in {"www.kmu.gov.ua", "kmu.gov.ua"}:
                    raise RuntimeError(f"unexpected sitemap origin: {child_url}")
                pending.append(child_url)
        elif root_name == "urlset":
            for node in root:
                loc = ""
                lastmod = ""
                for child in node:
                    name = _local_name(child.tag)
                    if name == "loc" and child.text:
                        loc = child.text.strip()
                    elif name == "lastmod" and child.text:
                        lastmod = child.text.strip()
                if not loc:
                    continue
                canonical = canonicalize_url(loc)
                if canonical and "/news/" in urllib.parse.urlsplit(canonical).path:
                    entries[canonical] = SitemapEntry(canonical, lastmod)
        else:
            raise RuntimeError(f"unsupported sitemap root: {root_name}")
    ordered = sorted(entries.values(), key=lambda item: (item.lastmod, item.url), reverse=True)
    return ordered, sorted(snapshots, key=lambda row: row["url"])


def canonicalize_url(url: str) -> str | None:
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or parsed.netloc not in {"www.kmu.gov.ua", "kmu.gov.ua"}:
        return None
    path = re.sub(r"/+", "/", parsed.path).rstrip("/") or "/"
    lowered = path.lower()
    if any(part in lowered for part in _EXCLUDE_URL_PARTS):
        return None
    return urllib.parse.urlunsplit(("https", "www.kmu.gov.ua", path, "", ""))


def inspect_page(entry: SitemapEntry) -> dict[str, Any]:
    data, headers = fetch_bytes(entry.url)
    parser = PageParser()
    parser.feed(data.decode("utf-8", errors="strict"))
    visible = normalize_text("\n".join(parser.visible_parts))
    paragraph_text = normalize_text("\n".join(parser.paragraphs))
    title = normalize_text(" ".join(parser.title_parts))

    byline_exact = SECRETARIAT_BYLINE in visible
    site_license_marker = LICENSE_MARKER in visible
    page_other_license_marker = any(
        marker in visible
        for marker in ("CC BY-NC", "CC-BY-NC", "CC BY-ND", "CC-BY-ND", "All rights reserved")
    )
    word_count = len(re.findall(r"\b[\w’'-]+\b", paragraph_text, flags=re.UNICODE))
    email_count = len(_EMAIL_RE.findall(paragraph_text))
    phone_count = len(_PHONE_RE.findall(paragraph_text))
    ratio = cyrillic_ratio(paragraph_text)
    uk_specific = ukrainian_specific_count(paragraph_text)
    normalized_bytes = len(paragraph_text.encode("utf-8"))

    eligible = all(
        (
            byline_exact,
            site_license_marker,
            not page_other_license_marker,
            word_count >= 60,
            ratio >= 0.90,
            uk_specific >= 8,
            email_count == 0,
            phone_count == 0,
            normalized_bytes > 0,
        )
    )
    rejection: list[str] = []
    if not byline_exact:
        rejection.append("not_exact_secretariat_byline")
    if not site_license_marker:
        rejection.append("site_cc_by_marker_missing")
    if page_other_license_marker:
        rejection.append("page_other_license_marker")
    if word_count < 60:
        rejection.append("too_short")
    if ratio < 0.90 or uk_specific < 8:
        rejection.append("ukrainian_language_signal_fail")
    if email_count or phone_count:
        rejection.append("contact_like_scalar_present")
    return {
        "url": entry.url,
        "lastmod": entry.lastmod,
        "response_bytes": len(data),
        "response_sha256": sha256_bytes(data),
        "content_type": headers.get("content-type"),
        "title_sha256": sha256_bytes(title.encode("utf-8")),
        "exact_secretariat_byline": byline_exact,
        "site_cc_by_marker": site_license_marker,
        "other_license_marker": page_other_license_marker,
        "word_count": word_count,
        "cyrillic_letter_ratio": round(ratio, 6),
        "ukrainian_specific_letter_count": uk_specific,
        "email_count": email_count,
        "phone_count": phone_count,
        "normalized_probe_bytes": normalized_bytes,
        "normalized_probe_sha256": sha256_bytes(paragraph_text.encode("utf-8")),
        "eligible_probe_record": eligible,
        "rejection_reasons": rejection,
    }


def build_report(max_fetches: int, delay_seconds: float) -> dict[str, Any]:
    if max_fetches <= 0 or max_fetches > 500:
        raise ValueError("max_fetches must be in 1..500")
    if delay_seconds < 1.0:
        raise ValueError("delay_seconds must respect robots crawl-delay >= 1 second")
    entries, sitemap_snapshots = discover_sitemap_entries()
    selected = entries[:max_fetches]
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(selected):
        if index:
            time.sleep(delay_seconds)
        try:
            rows.append(inspect_page(entry))
        except Exception as exc:  # network probe records a failed page; it does not promote it
            rows.append(
                {
                    "url": entry.url,
                    "lastmod": entry.lastmod,
                    "eligible_probe_record": False,
                    "rejection_reasons": [f"fetch_or_parse_error:{type(exc).__name__}"],
                }
            )

    eligible = [row for row in rows if row.get("eligible_probe_record") is True]
    unique_payloads = {row["normalized_probe_sha256"] for row in eligible}
    duplicate_payload_count = len(eligible) - len(unique_payloads)
    observed_unique_bytes = sum(
        row["normalized_probe_bytes"]
        for row in eligible
        if sum(
            other.get("normalized_probe_sha256") == row["normalized_probe_sha256"]
            for other in eligible
        ) == 1
    )
    sampled_response_bytes = sum(int(row.get("response_bytes", 0)) for row in rows)
    report_core = {
        "schema_version": SCHEMA,
        "repository": REPOSITORY,
        "execution_profile": "LOCAL_FREE",
        "parent_authority": {
            "pr": PARENT_PR,
            "head_sha": PARENT_HEAD,
            "manifest_identity_sha256": PARENT_MANIFEST,
            "family_id": FAMILY_ID,
            "license_id": "CC-BY-4.0",
            "training_rights_parent_verdict": "ADMIT_BOUNDED_SIX_ONLY",
        },
        "probe_boundary": {
            "training_authorized_bytes": 0,
            "tokenizer_fit_executed": False,
            "optimizer_updates": 0,
            "final_test_accessed": False,
            "paid_compute_used": False,
            "canonical_registry_mutated": False,
            "family_count_credit_added": 0,
            "probe_is_not_bulk_admission": True,
        },
        "robots_policy": {
            "sitemap": SITEMAP_URL,
            "crawl_delay_seconds_minimum": 1.0,
            "configured_delay_seconds": delay_seconds,
            "disallowed_paths_respected": True,
        },
        "selection_contract": {
            "exact_byline": SECRETARIAT_BYLINE,
            "same_family_only": FAMILY_ID,
            "exclude_ministry_syndication": True,
            "exclude_normative_acts": True,
            "exclude_contact_request_submission_pages": True,
            "require_site_cc_by_marker": True,
            "reject_other_license_markers": True,
            "minimum_words": 60,
            "minimum_cyrillic_ratio": 0.90,
            "minimum_ukrainian_specific_letters": 8,
            "reject_email_or_phone": True,
        },
        "sitemap": {
            "discovered_news_url_count": len(entries),
            "snapshots": sitemap_snapshots,
        },
        "sample": {
            "requested_fetch_limit": max_fetches,
            "attempted_pages": len(rows),
            "eligible_pages": len(eligible),
            "eligible_fraction": (len(eligible) / len(rows)) if rows else 0.0,
            "duplicate_normalized_payload_count": duplicate_payload_count,
            "observed_unique_probe_bytes": observed_unique_bytes,
            "sampled_response_bytes": sampled_response_bytes,
            "family_cap_bytes": FAMILY_CAP_BYTES,
            "observed_fraction_of_family_cap": observed_unique_bytes / FAMILY_CAP_BYTES,
            "rows": rows,
        },
    }
    report = dict(report_core)
    report["probe_identity_sha256"] = canonical_sha256(report_core)
    report["verdict"] = (
        "PROBE_EVIDENCE_READY_FOR_LOCKED_BULK_MATERIALIZER"
        if len(eligible) >= 20 and observed_unique_bytes >= 20_000
        else "PROBE_INSUFFICIENT_OBSERVED_YIELD"
    )
    return report


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("schema mismatch")
    parent = report.get("parent_authority", {})
    if parent.get("pr") != PARENT_PR or parent.get("head_sha") != PARENT_HEAD:
        raise ValueError("parent authority drift")
    if parent.get("manifest_identity_sha256") != PARENT_MANIFEST:
        raise ValueError("parent manifest drift")
    if parent.get("family_id") != FAMILY_ID:
        raise ValueError("family drift")
    boundary = report.get("probe_boundary", {})
    required_false = (
        "tokenizer_fit_executed",
        "final_test_accessed",
        "paid_compute_used",
        "canonical_registry_mutated",
    )
    if boundary.get("training_authorized_bytes") != 0 or boundary.get("optimizer_updates") != 0:
        raise ValueError("probe fabricated training authority")
    if boundary.get("family_count_credit_added") != 0:
        raise ValueError("same-family probe fabricated family credit")
    if any(boundary.get(key) is not False for key in required_false):
        raise ValueError("probe truth boundary changed")
    robots = report.get("robots_policy", {})
    if float(robots.get("configured_delay_seconds", 0)) < 1.0:
        raise ValueError("crawl delay violated")
    sample = report.get("sample", {})
    rows = sample.get("rows")
    if not isinstance(rows, list):
        raise ValueError("sample rows missing")
    if int(sample.get("attempted_pages", -1)) != len(rows):
        raise ValueError("attempted page count mismatch")
    if int(sample.get("family_cap_bytes", -1)) != FAMILY_CAP_BYTES:
        raise ValueError("family cap drift")
    core = {key: value for key, value in report.items() if key not in {"probe_identity_sha256", "verdict"}}
    if report.get("probe_identity_sha256") != canonical_sha256(core):
        raise ValueError("probe identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-fetches", type=int, default=DEFAULT_MAX_FETCHES)
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(args.max_fetches, args.delay_seconds)
    validate_report(report)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "discovered_news_url_count": report["sitemap"]["discovered_news_url_count"],
                "attempted_pages": report["sample"]["attempted_pages"],
                "eligible_pages": report["sample"]["eligible_pages"],
                "observed_unique_probe_bytes": report["sample"]["observed_unique_probe_bytes"],
                "probe_identity_sha256": report["probe_identity_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
