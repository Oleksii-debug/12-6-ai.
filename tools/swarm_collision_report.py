#!/usr/bin/env python3
"""Detect duplicate ephemeral swarm task ownership in open GitHub work.

The guard is intentionally narrow: one open issue plus one open PR for the
same ephemeral task key is the normal issue-to-implementation lifecycle.
Multiple open issues or multiple open PRs claiming the same key are blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

API_ROOT = "https://api.github.com"
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
WORKER_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?SWARM_WORKER_ID\s*:\s*`?([A-Za-z0-9][A-Za-z0-9._/-]*)`?\s*$"
)
# Ephemeral orchestration identifiers are globally unique work claims.
# Permanent lane ids (D01, D02, ...) are deliberately excluded because they
# may legitimately own multiple successive PRs.
EPHEMERAL_TASK_PATTERNS = (
    re.compile(r"\b(NEXT100-\d{3})\b", re.IGNORECASE),
    re.compile(r"\b(G\d{2}-T\d{2})\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class WorkRecord:
    number: int
    kind: str
    title: str
    html_url: str
    task_keys: tuple[str, ...]


def _canonical_task_key(value: str) -> str:
    return value.upper()


def extract_task_keys(title: str, body: str | None) -> tuple[str, ...]:
    """Return ephemeral task keys claimed by a title or SWARM_WORKER_ID line."""
    candidates: list[str] = [title]
    if body:
        candidates.extend(WORKER_LINE_RE.findall(body))

    keys: set[str] = set()
    for candidate in candidates:
        for pattern in EPHEMERAL_TASK_PATTERNS:
            keys.update(_canonical_task_key(match.group(1)) for match in pattern.finditer(candidate))
    return tuple(sorted(keys))


def records_from_github_items(items: Iterable[dict[str, Any]]) -> list[WorkRecord]:
    records: list[WorkRecord] = []
    for item in items:
        title = str(item.get("title") or "")
        body = item.get("body")
        keys = extract_task_keys(title, body if isinstance(body, str) else None)
        if not keys:
            continue
        records.append(
            WorkRecord(
                number=int(item["number"]),
                kind="pull_request" if "pull_request" in item else "issue",
                title=title,
                html_url=str(item.get("html_url") or ""),
                task_keys=keys,
            )
        )
    return sorted(records, key=lambda record: (record.number, record.kind))


def _record_dict(record: WorkRecord) -> dict[str, Any]:
    return {
        "number": record.number,
        "title": record.title,
        "html_url": record.html_url,
    }


def detect_collisions(records: Iterable[WorkRecord]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, list[WorkRecord]]] = defaultdict(
        lambda: {"issue": [], "pull_request": []}
    )
    for record in records:
        for key in record.task_keys:
            groups[key][record.kind].append(record)

    collisions: list[dict[str, Any]] = []
    for key in sorted(groups):
        issues = sorted(groups[key]["issue"], key=lambda record: record.number)
        prs = sorted(groups[key]["pull_request"], key=lambda record: record.number)
        reasons: list[str] = []
        if len(issues) > 1:
            reasons.append("MULTIPLE_OPEN_ISSUES_CLAIM_TASK")
        if len(prs) > 1:
            reasons.append("MULTIPLE_OPEN_PRS_CLAIM_TASK")
        if not reasons:
            continue
        collisions.append(
            {
                "task_key": key,
                "reasons": reasons,
                "issues": [_record_dict(record) for record in issues],
                "pull_requests": [_record_dict(record) for record in prs],
            }
        )
    return collisions


def _task_groups(records: Iterable[WorkRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[WorkRecord]]] = defaultdict(
        lambda: {"issue": [], "pull_request": []}
    )
    for record in records:
        for key in record.task_keys:
            grouped[key][record.kind].append(record)

    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        result.append(
            {
                "task_key": key,
                "issues": [
                    _record_dict(record)
                    for record in sorted(grouped[key]["issue"], key=lambda record: record.number)
                ],
                "pull_requests": [
                    _record_dict(record)
                    for record in sorted(
                        grouped[key]["pull_request"], key=lambda record: record.number
                    )
                ],
            }
        )
    return result


def build_report(repo: str, items: Iterable[dict[str, Any]], generated_at: str) -> dict[str, Any]:
    records = records_from_github_items(items)
    collisions = detect_collisions(records)
    report: dict[str, Any] = {
        "schema_version": "12-6.swarm-collision-report.v1",
        "repository": repo,
        "generated_at_utc": generated_at,
        "scope": {
            "ephemeral_task_patterns": [pattern.pattern for pattern in EPHEMERAL_TASK_PATTERNS],
            "normal_lifecycle": "ONE_OPEN_ISSUE_PLUS_ONE_OPEN_PR_PER_TASK",
            "permanent_lane_ids_checked": False,
        },
        "open_items_with_ephemeral_task_key": len(records),
        "task_groups": _task_groups(records),
        "collisions": collisions,
        "verdict": "BLOCK_DUPLICATE_EPHEMERAL_TASK_OWNERSHIP" if collisions else "PASS",
    }
    digest_payload = json.dumps(
        {key: value for key, value in report.items() if key != "report_sha256"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    return report


def _github_get_json(url: str, token: str | None) -> tuple[Any, dict[str, str]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "12-6-ai-swarm-collision-guard",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
        return payload, {key.lower(): value for key, value in response.headers.items()}


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        match = re.match(r"<([^>]+)>", section)
        if match:
            return match.group(1)
    return None


def fetch_open_items(repo: str, token: str | None) -> list[dict[str, Any]]:
    if not REPO_RE.fullmatch(repo):
        raise ValueError("repository must be in owner/name form")
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/", 1))
    url: str | None = f"{API_ROOT}/repos/{quoted_repo}/issues?state=open&per_page=100"
    items: list[dict[str, Any]] = []
    while url:
        payload, headers = _github_get_json(url, token)
        if not isinstance(payload, list):
            raise RuntimeError("GitHub issues endpoint returned a non-list payload")
        items.extend(item for item in payload if isinstance(item, dict))
        url = _next_link(headers.get("link"))
    return items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repository in owner/name form (default: GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the machine-readable report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 when duplicate ephemeral task ownership is detected",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.repo:
        print("error: --repo or GITHUB_REPOSITORY is required", file=sys.stderr)
        return 1
    try:
        items = fetch_open_items(args.repo, os.environ.get("GITHUB_TOKEN"))
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        report = build_report(args.repo, items, generated_at)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.strict and report["collisions"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
