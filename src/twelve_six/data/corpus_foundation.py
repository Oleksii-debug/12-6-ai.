"""Minimal current-main policy-hook contract for D03 data gates.

This selectively preserves the incumbent DATA-33 policy evidence seam without
reintroducing the obsolete corpus-foundation dedup/sharding stack.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

POLICY_SCHEMA = "12-6.record-policy-metadata.v1"
_ALLOWED_HOOK_STATUS = frozenset({"NOT_RUN", "PASS", "REJECT", "REVIEW_REQUIRED"})


class CorpusFoundationError(ValueError):
    """Raised when policy-hook evidence is unsafe or inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusFoundationError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if text != text.lower() or len(text) != 64:
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    if any(char not in "0123456789abcdef" for char in text):
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    return text


@dataclass(frozen=True)
class PolicyHookEvidence:
    """One data-policy hook result bound to exact tool/policy evidence."""

    hook_id: str
    status: str
    policy_version: str
    tool_ref: str
    executed_at: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        for field in ("hook_id", "policy_version", "tool_ref", "executed_at"):
            _require_text(getattr(self, field), field)
        if self.status not in _ALLOWED_HOOK_STATUS:
            raise CorpusFoundationError(f"unsupported hook status: {self.status}")
        _require_sha256(self.evidence_sha256, "evidence_sha256")


@dataclass(frozen=True)
class RecordPolicyMetadata:
    """Quality/LID/PII/copyright evidence; metadata never grants source rights."""

    quality: PolicyHookEvidence
    language: PolicyHookEvidence
    pii: PolicyHookEvidence
    copyright: PolicyHookEvidence
    schema_version: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA:
            raise CorpusFoundationError("unsupported policy metadata schema")

    def assert_passed(self) -> None:
        failures = [
            hook.hook_id
            for hook in (self.quality, self.language, self.pii, self.copyright)
            if hook.status != "PASS"
        ]
        if failures:
            raise CorpusFoundationError(
                "record policy metadata is not train-eligible; non-PASS hooks: "
                + ", ".join(sorted(failures))
            )

    def manifest(self) -> dict[str, Any]:
        core = asdict(self)
        return {**core, "metadata_sha256": _sha256_bytes(_canonical_json_bytes(core))}
