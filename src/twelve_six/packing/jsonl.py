"""Strict D03 JSONL consumption adapters for D04 packing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .core import TextRecord


class JsonlRecordError(ValueError):
    """Raised when a packaged D03 JSONL record violates the D04 consumption contract."""


def records_from_jsonl_lines(
    lines: Iterable[str],
    *,
    split: str,
) -> Iterator[TextRecord]:
    """Convert D03 packaged JSONL lines to split-bound TextRecord objects.

    D03 remains owner of provenance/hash validation and split assignment. D04
    requires only stable ``id`` and normalized ``text`` fields and binds the
    caller-supplied split explicitly.
    """
    if not split:
        raise ValueError("split must be non-empty")
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise JsonlRecordError(f"invalid JSON at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise JsonlRecordError(f"line {line_number} must contain a JSON object")

        record_id = payload.get("id")
        text = payload.get("text")
        if not isinstance(record_id, str) or not record_id:
            raise JsonlRecordError(f"line {line_number} has invalid or missing id")
        if not isinstance(text, str):
            raise JsonlRecordError(f"line {line_number} has invalid or missing text")
        if record_id in seen_ids:
            raise JsonlRecordError(f"duplicate record id {record_id!r}")
        seen_ids.add(record_id)
        yield TextRecord(record_id=record_id, text=text, split=split)


def load_jsonl_records(path: str | Path, *, split: str) -> Iterator[TextRecord]:
    """Stream a local D03 packaged JSONL file without reshuffling its committed order."""
    with Path(path).open("r", encoding="utf-8") as handle:
        yield from records_from_jsonl_lines(handle, split=split)
