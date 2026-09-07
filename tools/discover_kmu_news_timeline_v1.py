#!/usr/bin/env python3
"""Bounded same-origin KMu timeline discovery fallback for PR #815.

Discovery only: emits canonical /news/... URLs and page hashes. It never admits
training bytes and deliberately leaves content qualification to the existing probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

ORIGIN = "https://www.kmu.gov.ua"
# Live KMu exposes the news timeline through category_id=3. Pin it rather than
# silently accepting arbitrary category filters from discovered links.
NEWS_CATEGORY_ID = "3"
SEED = f"{ORIGIN}/timeline?category_id={NEWS_CATEGORY_ID}&type=posts"
USER_AGENT = "12-6-ai-kmu-timeline-discovery/1.0 (+LOCAL_FREE research)"
EXCLUDED = ("/npas/", "/api/", "/contact", "/kontakt", "/zvernenn", "/appeal")
CLIENT_RENDERED_MARKERS = ("Завантажуємо ще", "Loading more")


def canonical_news_url(raw: str, base: str = SEED) -> str | None:
    absolute = urllib.parse.urljoin(base, raw)
    parsed = urllib.parse.urlsplit(absolute)
    if parsed.scheme != "https" or parsed.netloc not in {"kmu.gov.ua", "www.kmu.gov.ua"}:
        return None
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    low = path.lower()
    if not low.startswith("/news/") or any(part in low for part in EXCLUDED):
        return None
    return urllib.parse.urlunsplit(("https", "www.kmu.gov.ua", path, "", ""))


def canonical_timeline_page(raw: str, base: str = SEED) -> str | None:
    absolute = urllib.parse.urljoin(base, raw)
    parsed = urllib.parse.urlsplit(absolute)
    if parsed.scheme != "https" or parsed.netloc not in {"kmu.gov.ua", "www.kmu.gov.ua"}:
        return None
    if parsed.path.rstrip("/") != "/timeline":
        return None
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if set(query) - {"type", "page", "category_id"}:
        return None
    if query.get("type") != ["posts"]:
        return None
    if query.get("category_id") != [NEWS_CATEGORY_ID]:
        return None
    page_values = query.get("page", [])
    if len(page_values) > 1:
        return None
    if page_values and (not page_values[0].isdigit() or int(page_values[0]) < 1):
        return None
    normalized = [("category_id", NEWS_CATEGORY_ID), ("type", "posts")]
    if page_values:
        normalized.append(("page", str(int(page_values[0]))))
    return urllib.parse.urlunsplit(
        ("https", "www.kmu.gov.ua", "/timeline", urllib.parse.urlencode(normalized), "")
    )


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


def fetch(url: str, timeout: float = 20.0) -> bytes:
    expected = canonical_timeline_page(url)
    if expected is None:
        raise RuntimeError("non-canonical timeline request rejected")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        final_url = response.geturl()
        final = canonical_timeline_page(final_url, url)
        if final != expected:
            raise RuntimeError(f"timeline redirect drift rejected: {final_url}")
        return response.read()


def _explicit_page_vector(max_pages: int) -> list[str]:
    """Return the exact bounded HTTP pagination vector, independent of HTML links."""
    pages = [SEED]
    for page_number in range(2, max_pages + 1):
        candidate = canonical_timeline_page(f"{SEED}&page={page_number}")
        if candidate is None:
            raise RuntimeError("failed to construct canonical timeline pagination")
        pages.append(candidate)
    return pages


def discover(max_pages: int, delay_seconds: float) -> dict[str, object]:
    if max_pages < 1 or max_pages > 20:
        raise ValueError("max_pages must be in 1..20")
    if delay_seconds < 1.0:
        raise ValueError("delay_seconds must be >=1.0")
    # Do not rely on the returned HTML to advertise pagination. The live seed can be a
    # client-rendered shell with no links while later canonical page variants still exist.
    pending = _explicit_page_vector(max_pages)
    visited: set[str] = set()
    news: set[str] = set()
    snapshots: list[dict[str, object]] = []
    client_rendered_shell_detected = False
    while pending and len(visited) < max_pages:
        url = pending.pop(0)
        if url in visited:
            continue
        if visited:
            time.sleep(delay_seconds)
        data = fetch(url)
        visited.add(url)
        snapshots.append({"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        text = data.decode("utf-8", errors="strict")
        if any(marker in text for marker in CLIENT_RENDERED_MARKERS):
            client_rendered_shell_detected = True
        parser = LinkParser()
        parser.feed(text)
        for href in parser.links:
            candidate = canonical_news_url(href, url)
            if candidate:
                news.add(candidate)
            timeline_page = canonical_timeline_page(href, url)
            if timeline_page and timeline_page not in visited and timeline_page not in pending:
                pending.append(timeline_page)
    if news:
        discovery_status = "DISCOVERED_ARTICLE_URLS"
    elif client_rendered_shell_detected:
        discovery_status = "BLOCKED_CLIENT_RENDERED_TIMELINE_NO_ARTICLE_URLS"
    else:
        discovery_status = "BLOCKED_EMPTY_TIMELINE_NO_ARTICLE_URLS"
    core = {
        "schema_version": "12-6.kmu-timeline-discovery.v1",
        "seed": SEED,
        "news_category_id": NEWS_CATEGORY_ID,
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "family_count_credit_added": 0,
        "pages_visited": len(visited),
        "client_rendered_shell_detected": client_rendered_shell_detected,
        "discovery_status": discovery_status,
        "snapshots": sorted(snapshots, key=lambda x: str(x["url"])),
        "news_urls": sorted(news),
    }
    core["identity_sha256"] = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=1.05)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = discover(args.max_pages, args.delay_seconds)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
