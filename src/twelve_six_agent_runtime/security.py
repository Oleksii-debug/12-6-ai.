"""Plan 5 Section 10: host-attested runtime trust, secrets and effect boundaries.

This module grants no model, tool or external-content authority. The incumbent
ContextEntry, ToolRegistry and TaskStore remain the sole canonical authorities.
No credential bytes are accepted, logged, decoded, transmitted or persisted here.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Callable, Literal

from .context import ContextEntry, ContextError
from .task_state import _digest, _json, _uint
from .tools import ToolCall, ToolRegistry

SourceKind = Literal["system", "owner", "host", "model", "tool", "external", "observation"]
Issuer = Literal["system", "owner", "host"]
LOG_EVENTS = ("admit", "deny", "issue", "resolve", "reconcile")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
VAULT_HANDLE = re.compile(r"vault:[A-Za-z0-9_-]{1,128}\Z")
SCHEMA = "12-6.agent-trust-boundary.v1"
SOURCES = {
    "system": "system", "owner": "owner", "host": "observation",
    "model": "model", "tool": "tool", "external": "retrieval",
    "observation": "observation",
}
SECRET_KEYS = frozenset((
    "authorization", "api_key", "apikey", "access_token", "refresh_token",
    "password", "passwd", "secret", "private_key", "credential",
))


class TrustBoundaryError(ValueError):
    """Unauthenticated authority, unsafe content, secret or effect request."""


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise TrustBoundaryError(f"invalid {field}")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise TrustBoundaryError("invalid SHA-256 identity")
    return value


def _nonnegative(value: object, label: str) -> int:
    try:
        return _uint(value, label)
    except ValueError as exc:
        raise TrustBoundaryError(f"invalid {label}") from exc


@dataclass(frozen=True)
class SourceEvidence:
    source_id: str
    kind: SourceKind
    evidence_id: str
    content_sha256: str

    def validate(self) -> None:
        _safe_id(self.source_id, "source")
        _safe_id(self.evidence_id, "source evidence")
        if self.kind not in SOURCES:
            raise TrustBoundaryError("unrecognized trust origin")
        _sha(self.content_sha256)


def admit_context_entry(
    entry: ContextEntry, evidence: SourceEvidence, *,
    verify_source: Callable[[SourceEvidence], bool],
    verify_secret_safety: Callable[[ContextEntry], bool],
    verify_privileged: Callable[[SourceEvidence], bool],
) -> ContextEntry:
    """Classify external/model text as data, never owner/system instructions.

    Caller-side metadata is not evidence: trusted host verifies origin, content
    confidentiality and privileged role independently. No text grants permission.
    """
    if not isinstance(entry, ContextEntry) or not isinstance(evidence, SourceEvidence):
        raise TrustBoundaryError("typed context and provenance required")
    evidence.validate()
    if (not isinstance(entry.content, str)
            or evidence.content_sha256
            != hashlib.sha256(entry.content.encode("utf-8")).hexdigest()
            or entry.source_id != evidence.source_id
            or entry.source_kind != SOURCES[evidence.kind]):
        raise TrustBoundaryError("source/content/classification mismatch")
    if (not callable(verify_source) or verify_source(evidence) is not True
            or not callable(verify_secret_safety)
            or verify_secret_safety(entry) is not True):
        raise TrustBoundaryError("unverified origin or possible secret exposure")
    privileged = evidence.kind in ("system", "owner")
    if privileged:
        if not callable(verify_privileged) or verify_privileged(evidence) is not True:
            raise TrustBoundaryError("privileged instruction lacks host attestation")
    elif entry.critical:
        raise TrustBoundaryError("untrusted data cannot be a protected instruction")
    try:
        entry.validate()
    except ContextError as exc:
        raise TrustBoundaryError("invalid canonical context") from exc
    return entry


@dataclass(frozen=True)
class SecretHandle:
    """Opaque host-only reference: contains no credential bytes or secret value."""

    handle_id: str
    allowed_tool_id: str
    evidence_id: str

    def validate(self) -> None:
        if not isinstance(self.handle_id, str) or VAULT_HANDLE.fullmatch(
            self.handle_id
        ) is None:
            raise TrustBoundaryError("secret must be a vault handle")
        _safe_id(self.allowed_tool_id, "scoped secret tool")
        _safe_id(self.evidence_id, "secret evidence")


def verify_secret_handle(
    handle: SecretHandle, *, requested_tool_id: str,
    verify_host_handle: Callable[[SecretHandle], bool],
) -> None:
    """Resolve permissions in the trusted host, never return or log the secret."""
    if not isinstance(handle, SecretHandle):
        raise TrustBoundaryError("opaque secret handle required")
    handle.validate()
    if (handle.allowed_tool_id != _safe_id(requested_tool_id, "requested tool")
            or not callable(verify_host_handle)
            or verify_host_handle(handle) is not True):
        raise TrustBoundaryError("secret handle is unverified or out of scope")


@dataclass(frozen=True)
class EffectGrant:
    """Host-attested grant bound to one exact canonical S7 tool invocation."""

    issuer: Issuer
    evidence_id: str
    task_id: str
    effect_id: str
    tool_id: str
    descriptor_digest: str
    request_sha256: str
    permissions: tuple[str, ...]
    control_epoch: int
    issued_at_ms: int
    expires_at_ms: int

    def validate(self) -> None:
        if self.issuer not in ("system", "owner", "host"):
            raise TrustBoundaryError("model/external/tool cannot mint grant")
        for key in ("evidence_id", "task_id", "effect_id", "tool_id"):
            _safe_id(getattr(self, key), key)
        _sha(self.descriptor_digest)
        _sha(self.request_sha256)
        if (type(self.permissions) is not tuple
                or self.permissions != tuple(sorted(set(self.permissions)))
                or any(_safe_id(p, "permission") != p for p in self.permissions)):
            raise TrustBoundaryError("invalid declared least-privilege permissions")
        for field in ("control_epoch", "issued_at_ms", "expires_at_ms"):
            _nonnegative(getattr(self, field), field)
        if self.expires_at_ms <= self.issued_at_ms:
            raise TrustBoundaryError("expired or inverted grant")


def _reject_secret_material(value: object, depth: int = 0) -> None:
    """Block structured credentials and vault handles from model-visible tools.

    Arbitrary content still requires the independent host payload-safety proof.
    """
    if depth > 8:
        raise TrustBoundaryError("unbounded nested tool content")
    if isinstance(value, SecretHandle):
        raise TrustBoundaryError("host-only secret handle cannot enter tool request")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TrustBoundaryError("non-string tool field")
            normalized = key.lower().replace("-", "_")
            if normalized in SECRET_KEYS:
                raise TrustBoundaryError("secret-bearing tool field forbidden")
            _reject_secret_material(child, depth + 1)
    elif isinstance(value, (tuple, list)):
        if len(value) > 128:
            raise TrustBoundaryError("unbounded tool array")
        for child in value:
            _reject_secret_material(child, depth + 1)
    elif isinstance(value, str):
        if ("vault:" in value.lower() or "bearer " in value.lower()
                or len(value) > 65_536):
            raise TrustBoundaryError("credential marker in model-visible data")


def prepare_secured_tool(
    registry: ToolRegistry, grant: EffectGrant, *,
    request: dict[str, object], expected_revision: int, now_ms: int,
    verify_grant: Callable[..., bool], verify_payload_safety: Callable[..., bool],
    authorize_host: Callable[..., bool],
) -> ToolCall:
    """Additional fail-closed preflight; Section-7 ToolRegistry remains authority."""
    if not isinstance(registry, ToolRegistry) or not isinstance(grant, EffectGrant):
        raise TrustBoundaryError("canonical registry/grant required")
    grant.validate()
    _nonnegative(now_ms, "host time")
    _nonnegative(expected_revision, "task revision")
    if not isinstance(request, dict):
        raise TrustBoundaryError("typed tool request required")
    _reject_secret_material(request)
    descriptor, digest = registry.get(grant.tool_id)
    if (grant.descriptor_digest != digest
            or grant.permissions != descriptor.permissions
            or grant.request_sha256 != _digest(_json(request))
            or not grant.issued_at_ms <= now_ms < grant.expires_at_ms):
        raise TrustBoundaryError("stale or broadened request/grant/permission")
    if (not callable(verify_grant) or verify_grant(grant, descriptor, request) is not True
            or not callable(verify_payload_safety)
            or verify_payload_safety(grant, request) is not True
            or not callable(authorize_host)):
        raise TrustBoundaryError("missing independent host authorization")
    return registry.prepare(
        task_id=grant.task_id, effect_id=grant.effect_id,
        tool_id=grant.tool_id, request=request, permissions=grant.permissions,
        grant_evidence_id=grant.evidence_id,
        expected_epoch=grant.control_epoch, expected_revision=expected_revision,
        authorize=lambda desc, state, payload, evidence: (
            authorize_host(desc, state, payload, evidence) is True
        ),
    )


@dataclass(frozen=True)
class AuditReceipt:
    """Strict allowlist logging: NEVER log payloads, free text or credentials."""

    event: str
    task_id: str
    effect_id: str
    tool_id: str
    evidence_digest: str
    outcome: Literal["allow", "deny", "unknown"]

    def encode(self) -> str:
        if self.event not in LOG_EVENTS or self.outcome not in ("allow", "deny", "unknown"):
            raise TrustBoundaryError("invalid audit event/outcome")
        for field in ("task_id", "effect_id", "tool_id"):
            _safe_id(getattr(self, field), field)
        _sha(self.evidence_digest)
        return _json({"schema": SCHEMA, **asdict(self)})
