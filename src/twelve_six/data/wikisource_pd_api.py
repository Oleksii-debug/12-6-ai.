"""MediaWiki acquisition adapter for the pinned Lesia 1892 Wikisource edition."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from twelve_six.data.wikisource_pd_contract import (
    API_URL,
    INDEX_REVISION_ID,
    INDEX_TITLE,
    PAGE_PREFIX,
    WikisourceIntakeError,
    normalize_rendered_text,
    validate_page_title,
    validate_ua_page_text,
)

VALIDATED_PROOFREAD_QUALITY = 4


@dataclass(frozen=True)
class PageSnapshot:
    page_number: int
    title: str
    revision_id: int
    normalized_text: str
    sha256: str
    utf8_bytes: int


class _VisibleTextParser(HTMLParser):
    _HIDDEN = frozenset({"script", "style", "noscript", "template", "svg", "math"})
    _BLOCK = frozenset({"p", "div", "br", "li", "poem", "section", "h1", "h2", "h3"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._HIDDEN:
            self._hidden_depth += 1
        if not self._hidden_depth and tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._HIDDEN and self._hidden_depth:
            self._hidden_depth -= 1
        if not self._hidden_depth and tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def rendered_html_to_text(value: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(value)
    parser.close()
    return normalize_rendered_text(parser.text())


def request_json(params: dict[str, str], *, timeout: float = 30.0) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": "12-6-ai-local-free-wikisource-intake/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict) or "error" in value:
        raise WikisourceIntakeError("MediaWiki API returned an error")
    return value


def _validated_sorted_page_titles(rows: Any, *, source: str) -> list[str]:
    if not isinstance(rows, list):
        raise WikisourceIntakeError(f"{source} page list is missing")
    titles: list[tuple[int, str]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("title"), str):
            raise WikisourceIntakeError(f"{source} page list contains malformed entry")
        title = row["title"]
        if not title.startswith(PAGE_PREFIX):
            continue
        titles.append((validate_page_title(title), title))
    if not titles:
        raise WikisourceIntakeError(f"{source} contains no numeric page links")
    if len({number for number, _ in titles}) != len(titles):
        raise WikisourceIntakeError(f"{source} contains duplicate numeric page links")
    return [title for _, title in sorted(titles)]


def _current_index_revision_id(
    get_json: Callable[[dict[str, str]], dict[str, Any]],
) -> int:
    response = get_json(
        {
            "action": "query",
            "titles": INDEX_TITLE,
            "prop": "revisions",
            "rvprop": "ids",
        }
    )
    query = response.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise WikisourceIntakeError("index revision response is ambiguous")
    page = pages[0]
    if page.get("missing") is True or page.get("title") != INDEX_TITLE:
        raise WikisourceIntakeError("pinned index title is missing or drifted")
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise WikisourceIntakeError("index revision identity is ambiguous")
    revision = revisions[0]
    revision_id = revision.get("revid") if isinstance(revision, dict) else None
    if not isinstance(revision_id, int) or isinstance(revision_id, bool) or revision_id <= 0:
        raise WikisourceIntakeError("invalid index revision id")
    return revision_id


def _proofread_quality(row: dict[str, Any], *, source: str) -> int:
    proofread = row.get("proofread")
    if not isinstance(proofread, dict):
        raise WikisourceIntakeError(f"{source} native ProofreadPage quality is missing")
    quality = proofread.get("quality")
    if not isinstance(quality, int) or isinstance(quality, bool) or not 0 <= quality <= 4:
        raise WikisourceIntakeError(f"{source} native ProofreadPage quality is invalid")
    return quality


def _discover_proofread_index_titles(
    *,
    get_json: Callable[[dict[str, str]], dict[str, Any]],
) -> list[str]:
    before_revision = _current_index_revision_id(get_json)
    if before_revision != INDEX_REVISION_ID:
        raise WikisourceIntakeError(
            "current index revision drifted from pinned authority; refusing live pagination"
        )
    response = get_json(
        {
            "action": "query",
            "generator": "proofreadpagesinindex",
            "gprppiititle": INDEX_TITLE,
            "gprppiilimit": "500",
            "prop": "proofread",
        }
    )
    if response.get("continue") is not None:
        raise WikisourceIntakeError("qualified index pagination unexpectedly exceeds one bounded page")
    query = response.get("query")
    rows = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(rows, list):
        raise WikisourceIntakeError("qualified proofread index page list is missing")
    validated_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("title"), str):
            raise WikisourceIntakeError("qualified proofread index contains malformed entry")
        title = row["title"]
        if not title.startswith(PAGE_PREFIX):
            continue
        validate_page_title(title)
        if row.get("missing") is True:
            continue
        if _proofread_quality(row, source="qualified proofread index") == VALIDATED_PROOFREAD_QUALITY:
            validated_rows.append(row)
    if not validated_rows:
        raise WikisourceIntakeError("qualified proofread index contains no validated numeric page")
    titles = _validated_sorted_page_titles(validated_rows, source="qualified proofread index")
    after_revision = _current_index_revision_id(get_json)
    if after_revision != INDEX_REVISION_ID:
        raise WikisourceIntakeError(
            "index revision changed during live pagination; refusing materialization"
        )
    return titles


def discover_index_titles(
    *,
    get_json: Callable[[dict[str, str]], dict[str, Any]] = request_json,
) -> list[str]:
    response = get_json({"action": "parse", "oldid": str(INDEX_REVISION_ID), "prop": "links"})
    parse = response.get("parse")
    if not isinstance(parse, dict):
        raise WikisourceIntakeError("index parse response missing")
    parsed_revision = parse.get("revid")
    if parsed_revision is not None and parsed_revision != INDEX_REVISION_ID:
        raise WikisourceIntakeError("parsed index revision does not match pinned authority")
    links = parse.get("links")
    if not isinstance(links, list):
        raise WikisourceIntakeError("index links missing")
    if any(
        isinstance(row, dict)
        and isinstance(row.get("title"), str)
        and row["title"].startswith(PAGE_PREFIX)
        for row in links
    ):
        return _validated_sorted_page_titles(links, source="qualified index")
    return _discover_proofread_index_titles(get_json=get_json)


def _current_page_revision_and_approval(
    title: str,
    *,
    get_json: Callable[[dict[str, str]], dict[str, Any]],
) -> tuple[int, bool]:
    response = get_json(
        {
            "action": "query",
            "titles": title,
            "prop": "revisions|proofread",
            "rvprop": "ids",
        }
    )
    query_root = response.get("query")
    pages = query_root.get("pages") if isinstance(query_root, dict) else None
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise WikisourceIntakeError("page metadata response is ambiguous")
    page = pages[0]
    if page.get("missing") is True or page.get("title") != title:
        raise WikisourceIntakeError("linked page is missing or title drifted")
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise WikisourceIntakeError("page revision identity is ambiguous")
    revision = revisions[0]
    revision_id = revision.get("revid") if isinstance(revision, dict) else None
    if not isinstance(revision_id, int) or isinstance(revision_id, bool) or revision_id <= 0:
        raise WikisourceIntakeError("invalid page revision id")
    return revision_id, _proofread_quality(page, source="page metadata") == VALIDATED_PROOFREAD_QUALITY


def fetch_page_snapshot(
    title: str,
    *,
    get_json: Callable[[dict[str, str]], dict[str, Any]] = request_json,
) -> PageSnapshot:
    import hashlib

    page_number = validate_page_title(title)
    revision_id, approved_before = _current_page_revision_and_approval(
        title, get_json=get_json
    )
    if not approved_before:
        raise WikisourceIntakeError("exact page revision is not approved")
    rendered = get_json(
        {"action": "parse", "oldid": str(revision_id), "prop": "text|revid"}
    )
    parse = rendered.get("parse")
    if not isinstance(parse, dict) or not isinstance(parse.get("text"), str):
        raise WikisourceIntakeError("page rendered text is missing")
    parsed_revision = parse.get("revid")
    if parsed_revision != revision_id:
        raise WikisourceIntakeError("parsed revision does not match sealed page revision")
    after_revision_id, approved_after = _current_page_revision_and_approval(
        title, get_json=get_json
    )
    if after_revision_id != revision_id:
        raise WikisourceIntakeError("page revision changed during exact render")
    if not approved_after:
        raise WikisourceIntakeError("page approval changed during exact render")
    normalized = rendered_html_to_text(parse["text"])
    validate_ua_page_text(normalized)
    payload = normalized.encode("utf-8")
    return PageSnapshot(
        page_number=page_number,
        title=title,
        revision_id=revision_id,
        normalized_text=normalized,
        sha256=hashlib.sha256(payload).hexdigest(),
        utf8_bytes=len(payload),
    )
