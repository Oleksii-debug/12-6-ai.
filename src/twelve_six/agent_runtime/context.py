"""Model-neutral bounded context assembly. Plan 5 / Section 1.

The trusted caller, not retrieved text or a model, is responsible for source
classification and the critical-invariant bit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

SCHEMA = "12-6.agent-context.v1"
POLICY = "critical-pinned-priority-desc-recency-desc-v1"
SourceKind = Literal["system", "owner", "tool", "retrieval", "observation", "model"]


class ContextError(ValueError):
    """Invalid provenance or insufficient budget for protected invariants."""


@dataclass(frozen=True)
class ContextEntry:
    entry_id: str
    source_id: str
    source_kind: SourceKind
    recency: int
    priority: int
    content: str
    critical: bool = False

    def validate(self) -> None:
        for field in ("entry_id", "source_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ContextError(f"{field} must be nonempty")
        if self.source_kind not in ("system", "owner", "tool", "retrieval", "observation", "model"):
            raise ContextError("invalid source_kind")
        if type(self.recency) is not int or self.recency < 0:
            raise ContextError("invalid recency")
        if type(self.priority) is not int or not (0 <= self.priority <= 100):
            raise ContextError("invalid priority")
        if not isinstance(self.content, str):
            raise ContextError("content must be text")
        if type(self.critical) is not bool:
            raise ContextError("critical must be boolean")
        if self.critical and self.source_kind not in ("system", "owner"):
            raise ContextError("untrusted source cannot assert a protected invariant")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def build_context(
    entries: Sequence[ContextEntry], *, max_bytes: int, max_entries: int
) -> dict[str, object]:
    """Select important entries deterministically, never truncate critical bytes.

    Omissions are explicitly counted and hashed. Encoded output stays within
    max_bytes and has no model/provider/tokenizer-specific dependency.
    """
    if type(max_bytes) is not int or max_bytes <= 0 or type(max_entries) is not int or max_entries <= 0:
        raise ContextError("budgets must be positive integers")
    by_id: dict[str, ContextEntry] = {}
    for entry in entries:
        if not isinstance(entry, ContextEntry):
            raise ContextError("invalid entry type")
        entry.validate()
        if entry.entry_id in by_id:
            raise ContextError("duplicate entry_id")
        by_id[entry.entry_id] = entry
    protected = sorted((e for e in by_id.values() if e.critical), key=lambda e: (e.recency, e.entry_id))
    if len(protected) > max_entries:
        raise ContextError("protected invariants exceed max_entries")
    optional = sorted(
        (e for e in by_id.values() if not e.critical),
        key=lambda e: (-e.priority, -e.recency, e.entry_id),
    )
    selected = {e.entry_id for e in protected}

    def render(ids: set[str], *, proof: bool) -> dict[str, object]:
        ordered = sorted((by_id[key] for key in ids), key=lambda e: (e.recency, e.entry_id))
        omitted = (
            sorted((e for key, e in by_id.items() if key not in ids), key=lambda e: e.entry_id)
            if proof else ()
        )
        digest = (
            hashlib.sha256(canonical_bytes([asdict(e) for e in omitted])).hexdigest()
            if proof else "0" * 64
        )
        return {
            "schema": SCHEMA,
            "policy": POLICY,
            "entries": [asdict(e) for e in ordered],
            "omitted_count": len(by_id) - len(ids),
            "omitted_sha256": digest,
        }

    if len(canonical_bytes(render(selected, proof=False))) > max_bytes:
        raise ContextError("protected invariants cannot fit max_bytes")
    for entry in optional:
        if len(selected) >= max_entries:
            break
        candidate = selected | {entry.entry_id}
        if len(canonical_bytes(render(candidate, proof=False))) <= max_bytes:
            selected = candidate
    result = render(selected, proof=True)
    if len(canonical_bytes(result)) > max_bytes:
        raise AssertionError("context budget invariant")
    return result
