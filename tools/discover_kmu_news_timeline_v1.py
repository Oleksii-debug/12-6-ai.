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
SEED = f"{ORIGIN}/timeline?type=posts"
USER_AGENT = "12-6-ai-kmu-timeline-discovery/1.0 (+LOCAL_FREE research)"
EXCLUDED = ("/npas/", "/api/", "/contact", "/kontakt", "/zvernenn", "/appeal")


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
    if query.get("type") != ["posts"]:
        return None
    page_values = query.get("page", [])
    if len(page_values) > 1:
        return None
    if page_values and (not page_values[0].isdigit() or int(page_values[0]) < 1):
        return None
    normalized = [("type", "posts")]
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
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        final = urllib.parse.urlsplit(response.geturl())
        if final.scheme != "https" or final.netloc not in {"kmu.gov.ua", "www.kmu.gov.ua"}:
            raise RuntimeError("cross-origin redirect rejected")
        return response.read()


def discover(max_pages: int, delay_seconds: float) -> dict[str, object]:
    if max_pages < 1 or max_pages > 20:
        raise ValueError("max_pages must be in 1..20")
    if delay_seconds < 1.0:
        raise ValueError("delay_seconds must be >=1.0")
    pending = [SEED]
    visited: set[str] = set()
    news: set[str] = set()
    snapshots: list[dict[str, object]] = []
    while pending and len(visited) < max_pages:
        url = pending.pop(0)
        if url in visited:
            continue
        if visited:
            time.sleep(delay_seconds)
        data = fetch(url)
        visited.add(url)
        snapshots.append({"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        parser = LinkParser()
        parser.feed(data.decode("utf-8", errors="strict"))
        for href in parser.links:
            candidate = canonical_news_url(href, url)
            if candidate:
                news.add(candidate)
            timeline_page = canonical_timeline_page(href, url)
            if timeline_page and timeline_page not in visited and timeline_page not in pending:
                pending.append(timeline_page)
    core = {
        "schema_version": "12-6.kmu-timeline-discovery.v1",
        "seed": SEED,
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "family_count_credit_added": 0,
        "pages_visited": len(visited),
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
