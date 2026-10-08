"""Plan 5 Section 9: typed, evidence-bound multimodal reference adapters.

No media bytes, model service, microphone, filesystem or external tool is invoked.
The trusted host verifies bytes, origin, transforms and outbound permission.
The incumbent Section-7 registry and Section-2 task ledger own tool effects.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Literal

from .task_state import _digest, _id, _json, _uint
from .tools import DESCRIPTOR_VERSION, ToolCall, ToolDescriptor, ToolRegistry

Kind = Literal["image", "audio", "asr", "tts", "vision"]
Direction = Literal["input", "output"]
Privacy = Literal["public", "private", "restricted"]
Locality = Literal["local", "remote"]
SCHEMA = "12-6.agent-media-artifact.v1"
MEDIA_TOOL = "media.transfer"
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
MIMES = {
    "image": ("image/png", "image/jpeg", "image/webp"),
    "audio": ("audio/wav", "audio/flac", "audio/ogg"),
    "asr": ("text/plain", "application/json"),
    "tts": ("audio/wav", "audio/ogg"),
    "vision": ("application/json", "text/plain"),
}


class MediaBoundaryError(ValueError):
    """Malformed media, unverified provenance or disallowed model egress."""


def _name(value: object, field: str) -> str:
    try:
        return _id(value, field)
    except ValueError as exc:
        raise MediaBoundaryError(f"invalid {field}") from exc


def _sha(value: object) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise MediaBoundaryError("invalid SHA-256 identity")
    return value


@dataclass(frozen=True)
class MediaTransform:
    operation: str
    input_sha256: str
    output_sha256: str
    evidence_id: str
    timestamp_ms: int

    def validate(self) -> None:
        _name(self.operation, "transform operation")
        _name(self.evidence_id, "transform evidence")
        _sha(self.input_sha256)
        _sha(self.output_sha256)
        try:
            _uint(self.timestamp_ms, "transform time")
        except ValueError as exc:
            raise MediaBoundaryError("invalid transform time") from exc


@dataclass(frozen=True)
class MediaArtifact:
    artifact_id: str
    kind: Kind
    direction: Direction
    mime_type: str
    byte_size: int
    content_sha256: str
    source_sha256: str
    source_id: str
    evidence_id: str
    timestamp_ms: int
    privacy: Privacy
    locality: Locality
    transforms: tuple[MediaTransform, ...] = ()
    schema: str = SCHEMA

    def validate(self) -> None:
        if self.schema != SCHEMA or self.kind not in MIMES:
            raise MediaBoundaryError("unsupported media schema/kind")
        if self.direction not in ("input", "output"):
            raise MediaBoundaryError("invalid media direction")
        if self.mime_type not in MIMES[self.kind]:
            raise MediaBoundaryError("unsupported media MIME for kind")
        if self.privacy not in ("public", "private", "restricted"):
            raise MediaBoundaryError("invalid privacy")
        if self.locality not in ("local", "remote"):
            raise MediaBoundaryError("invalid source locality")
        for key in ("artifact_id", "source_id", "evidence_id"):
            _name(getattr(self, key), key)
        try:
            _uint(self.byte_size, "byte size")
            _uint(self.timestamp_ms, "source time")
        except ValueError as exc:
            raise MediaBoundaryError("invalid size/source time") from exc
        if self.byte_size == 0 or self.byte_size > 64 * 1024 * 1024:
            raise MediaBoundaryError("empty or unbounded payload")
        _sha(self.source_sha256)
        _sha(self.content_sha256)
        if type(self.transforms) is not tuple or len(self.transforms) > 16:
            raise MediaBoundaryError("invalid bounded transform chain")
        previous, when = self.source_sha256, self.timestamp_ms
        for step in self.transforms:
            if not isinstance(step, MediaTransform):
                raise MediaBoundaryError("untyped transform")
            step.validate()
            if step.input_sha256 != previous or step.timestamp_ms < when:
                raise MediaBoundaryError("broken or reordered transform provenance")
            previous, when = step.output_sha256, step.timestamp_ms
        if previous != self.content_sha256:
            raise MediaBoundaryError("transforms do not bind payload hash")
        if len(_json(asdict(self)).encode("utf-8")) > 16384:
            raise MediaBoundaryError("oversized media metadata")

    @property
    def identity(self) -> str:
        self.validate()
        return _digest(_json(asdict(self)))


@dataclass(frozen=True)
class MediaPolicy:
    allowed_kinds: tuple[str, ...]
    allowed_privacy: tuple[str, ...]
    max_bytes: int
    allow_remote: bool
    allowed_directions: tuple[str, ...] = ("input", "output")

    def validate(self) -> None:
        for values, choices in ((self.allowed_kinds, MIMES),
                                (self.allowed_privacy, ("public", "private", "restricted")),
                                (self.allowed_directions, ("input", "output"))):
            if (type(values) is not tuple or not values
                    or values != tuple(sorted(set(values)))
                    or any(item not in choices for item in values)):
                raise MediaBoundaryError("invalid or unbounded media policy")
        try:
            _uint(self.max_bytes, "policy byte ceiling")
        except ValueError as exc:
            raise MediaBoundaryError("invalid policy ceiling") from exc
        if self.max_bytes == 0 or self.max_bytes > 64 * 1024 * 1024:
            raise MediaBoundaryError("invalid byte ceiling")
        if type(self.allow_remote) is not bool:
            raise MediaBoundaryError("invalid remote policy")


@dataclass(frozen=True)
class MediaAdmission:
    artifact: MediaArtifact
    artifact_identity: str
    destination: Locality
    model_id: str
    host_evidence_id: str


def admit_media(
    artifact: MediaArtifact, policy: MediaPolicy, *, destination: Locality,
    model_id: str, model_capabilities: tuple[str, ...], host_evidence_id: str,
    verify_content: Callable[[MediaArtifact], bool],
    verify_provenance: Callable[[MediaArtifact], bool],
    authorize: Callable[[MediaArtifact, str, str], bool],
) -> MediaAdmission:
    """Fail closed before any transport. Callbacks must be trusted host code."""
    if not isinstance(artifact, MediaArtifact) or not isinstance(policy, MediaPolicy):
        raise MediaBoundaryError("typed artifact and policy required")
    artifact.validate()
    policy.validate()
    _name(model_id, "model id")
    _name(host_evidence_id, "host evidence")
    if destination not in ("local", "remote"):
        raise MediaBoundaryError("invalid model locality")
    if (artifact.kind not in policy.allowed_kinds
            or artifact.direction not in policy.allowed_directions
            or artifact.privacy not in policy.allowed_privacy
            or artifact.byte_size > policy.max_bytes
            or (destination == "remote" and not policy.allow_remote)):
        raise MediaBoundaryError("privacy/size/locality policy denied media")
    if (type(model_capabilities) is not tuple
            or artifact.kind not in model_capabilities):
        raise MediaBoundaryError("model lacks declared media capability")
    if (not callable(verify_content) or verify_content(artifact) is not True
            or not callable(verify_provenance)
            or verify_provenance(artifact) is not True
            or not callable(authorize)
            or authorize(artifact, destination, model_id) is not True):
        raise MediaBoundaryError("missing independently verified host admission")
    return MediaAdmission(artifact, artifact.identity, destination, model_id,
                          host_evidence_id)


def media_tool_descriptor() -> ToolDescriptor:
    """Register through incumbent ToolRegistry; discovery grants no permission."""
    def string(length: int) -> dict[str, object]:
        return {"type": "string", "maxLength": length}

    def closed(fields: dict[str, dict[str, object]]) -> dict[str, object]:
        return {"type": "object", "properties": fields,
                "required": list(fields), "additionalProperties": False}

    return ToolDescriptor(
        tool_id=MEDIA_TOOL, version=DESCRIPTOR_VERSION, available=True,
        capabilities=("asr", "audio", "image", "tts", "vision"),
        permissions=("media.transfer",), side_effect="external",
        input_schema=closed({
            "artifact_identity": string(64), "destination": string(6),
            "model_id": string(256), "host_evidence_id": string(256),
        }),
        output_schema=closed({
            "artifact_identity": string(64), "accepted": {"type": "boolean"},
        }),
    )


def prepare_media_transfer(
    registry: ToolRegistry, admission: MediaAdmission, *,
    task_id: str, effect_id: str, expected_epoch: int, expected_revision: int,
    grant_evidence_id: str, authorize_tool: Callable[..., bool],
    verify_admission: Callable[[MediaAdmission], bool],
) -> ToolCall:
    """Reserve only: actual egress requires host issue and postcondition receipts."""
    if not isinstance(registry, ToolRegistry) or not isinstance(admission, MediaAdmission):
        raise MediaBoundaryError("typed registry and admission required")
    if (not isinstance(admission.artifact, MediaArtifact)
            or admission.artifact_identity != admission.artifact.identity
            or admission.destination not in ("local", "remote")):
        raise MediaBoundaryError("stale or forged media admission")
    if not callable(verify_admission) or verify_admission(admission) is not True:
        raise MediaBoundaryError("host did not attest admission")
    descriptor, _ = registry.get(MEDIA_TOOL)
    if descriptor != media_tool_descriptor():
        raise MediaBoundaryError("noncanonical media tool descriptor")
    return registry.prepare(
        task_id=task_id, effect_id=effect_id, tool_id=MEDIA_TOOL,
        request={"artifact_identity": admission.artifact_identity,
                 "destination": admission.destination, "model_id": admission.model_id,
                 "host_evidence_id": admission.host_evidence_id},
        permissions=("media.transfer",), grant_evidence_id=grant_evidence_id,
        expected_epoch=expected_epoch, expected_revision=expected_revision,
        authorize=authorize_tool,
    )
