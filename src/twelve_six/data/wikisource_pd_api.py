"""MediaWiki acquisition adapter for the pinned Lesia 1892 Wikisource edition."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable

from twelve_six.data.wikisource_pd_contract import (
    API_URL,
    APPROVED_CATEGORY,
    INDEX_REVISION_ID,
    PAGE_PREFIX,
    WikisourceIntakeError,
    normalize_rendered_text,
    validate_page_title,
    validate_ua_page_text,
)


@dataclass(frozen=True)
class PageSnapshot:
    page_number: int
    title: str
    revision_id: int
    normalized_text: str
    sha256: str
    utf8_bytes: int


class _VisibleTextParser(HTMLParser):
    _HIDDEN = {"script", "style", "noscript", "template", "svg", "math"}
    _BLOCK = {"p", "div", "br", "li", "poem", "section", "h1", "h2", "h3"}

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
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict) or "error" in value:
        raise WikisourceIntakeError("MediaWiki API returned an error")
    return value


def discover_index_titles(
    *,
    get_json: Callable[[dict[str, str]], dict[str, Any]] = request_json,
) -> list[str]:
    response = get_json({"action": "parse", "oldid": str(INDEX_REVISION_ID), "prop": "links"})
    parse = response.get("parse")
    if not isinstance(parse, dict):
        raise WikisourceIntakeError("index parse response missing")
    links = parse.get("links")
    if not isinstance(links, list):
        raise WikisourceIntakeError("index links missing")
    titles: list[tuple[int, str]] = []
    for row in links:
        if not isinstance(row, dict) or not isinstance(row.get("title"), str):
            continue
        title = row["title"]
        if title.startswith(PAGE_PREFIX):
            titles.append((validate_page_title(title), title))
    if not titles:
        raise WikisourceIntakeError("qualified index contains no numeric page links")
    if len({number for number, _ in titles}) != len(titles):
        raise WikisourceIntakeError("qualified index contains duplicate numeric page links")
    return [title for _, title in sorted(titles)]


def _category_names(rows: Any) -> set[str]:
    if not isinstance(rows, list):
        raise WikisourceIntakeError("exact revision categories are missing")
    names: set[str] = set()
    for row in rows:
        if isinstance(row, str):
            names.add(row.removeprefix("Категорія:"))
        elif isinstance(row, dict):
            value = row.get("category", row.get("*", row.get("title")))
            if isinstance(value, str):
                names.add(value.removeprefix("Категорія:"))
    return names


def fetch_page_snapshot(
    title: str,
    *,
    get_json: Callable[[dict[str, str]], dict[str, Any]] = request_json,
) -> PageSnapshot:
    import hashlib

    page_number = validate_page_title(title)
    query = get_json(
        {"action": "query", "titles": title, "prop": "revisions", "rvprop": "ids"}
    )
    query_root = query.get("query")
    pages = query_root.get("pages") if isinstance(query_root, dict) else None
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise WikisourceIntakeError("page metadata response is ambiguous")
    page = pages[0]
    if page.get("missing") is True:
        raise WikisourceIntakeError("linked page is missing")
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise WikisourceIntakeError("page revision identity is ambiguous")
    revision = revisions[0]
    revision_id = revision.get("revid") if isinstance(revision, dict) else None
    if not isinstance(revision_id, int) or isinstance(revision_id, bool) or revision_id <= 0:
        raise WikisourceIntakeError("invalid page revision id")
    rendered = get_json(
        {"action": "parse", "oldid": str(revision_id), "prop": "text|categories"}
    )
    parse = rendered.get("parse")
    if not isinstance(parse, dict) or not isinstance(parse.get("text"), str):
        raise WikisourceIntakeError("page rendered text is missing")
    parsed_revision = parse.get("revid")
    if parsed_revision is not None and parsed_revision != revision_id:
        raise WikisourceIntakeError("parsed revision does not match sealed page revision")
    if APPROVED_CATEGORY not in _category_names(parse.get("categories")):
        raise WikisourceIntakeError("exact page revision is not approved")
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
