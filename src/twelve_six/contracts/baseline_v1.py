"""Versioned wire adapters over canonical Section 1-2 authorities (no new runtime)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from twelve_six.artifact_identity import (
    ArtifactKind, ArtifactManifest, ArtifactRef, GenerationIdentityManifest,
)
from twelve_six.system_architecture import (
    canonical_runtime_shell_v1, canonical_system_architecture_v1,
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z")
_SCHEMAS = {
    "artifact_ref": ArtifactRef,
    "artifact_manifest": ArtifactManifest,
    "generation_manifest": GenerationIdentityManifest,
}
_LIMIT = 1024 * 1024


class ContractPackageError(ValueError):
    """Versioned contract rejects unknown or ambiguous representation."""


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fields(value: Any, names: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != names:
        raise ContractPackageError("contract fields mismatch")
    return value


def _id(value: Any) -> None:
    if type(value) is not str or not _ID.fullmatch(value):
        raise ContractPackageError("invalid contract identifier")


def _digest(value: Any) -> None:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ContractPackageError("invalid contract digest")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    artifact: ArtifactRef
    evidence_sha256: str
    source_ref: str

    def __post_init__(self) -> None:
        if type(self.artifact) is not ArtifactRef:
            raise ContractPackageError("noncanonical evidence artifact")
        self.artifact.to_dict()
        _digest(self.evidence_sha256)
        _id(self.source_ref)

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {"artifact": self.artifact.to_dict(), "evidence_sha256": self.evidence_sha256,
                "source_ref": self.source_ref, "admitted": False}

    @classmethod
    def from_dict(cls, value: Any) -> EvidenceRef:
        v = _fields(value, {"artifact", "evidence_sha256", "source_ref", "admitted"})
        if v["admitted"] is not False:
            raise ContractPackageError("no evidence admission authority")
        return cls(ArtifactRef.from_dict(v["artifact"]), v["evidence_sha256"], v["source_ref"])


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    code: str
    description: str
    retryable: bool

    def __post_init__(self) -> None:
        _id(self.code)
        if type(self.description) is not str or not 0 < len(self.description) <= 1024:
            raise ContractPackageError("invalid error description")
        if type(self.retryable) is not bool:
            raise ContractPackageError("retryable must be bool")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {"code": self.code, "description": self.description, "retryable": self.retryable,
                "policy_authority": False}

    @classmethod
    def from_dict(cls, value: Any) -> ErrorRecord:
        v = _fields(value, {"code", "description", "retryable", "policy_authority"})
        if v["policy_authority"] is not False:
            raise ContractPackageError("no error policy authority")
        return cls(v["code"], v["description"], v["retryable"])


@dataclass(frozen=True, slots=True)
class LifecycleObservation:
    event_id: str
    phase: str
    sequence: int
    parent_sha256: str

    def __post_init__(self) -> None:
        _id(self.event_id)
        if type(self.phase) is not str or self.phase not in {
            "created", "queued", "executed", "stopped", "recovered"
        }:
            raise ContractPackageError("unknown lifecycle phase")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ContractPackageError("invalid lifecycle sequence")
        _digest(self.parent_sha256)

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {"event_id": self.event_id, "phase": self.phase, "sequence": self.sequence,
                "parent_sha256": self.parent_sha256, "effect_authority": False}

    @classmethod
    def from_dict(cls, value: Any) -> LifecycleObservation:
        v = _fields(value, {"event_id", "phase", "sequence", "parent_sha256", "effect_authority"})
        if v["effect_authority"] is not False:
            raise ContractPackageError("no lifecycle effect authority")
        return cls(v["event_id"], v["phase"], v["sequence"], v["parent_sha256"])


_SCHEMAS.update({
    "evidence_ref": EvidenceRef, "error_record": ErrorRecord,
    "lifecycle_observation": LifecycleObservation,
})
_SCHEMAS = MappingProxyType(_SCHEMAS)  # closed catalog: no runtime schema promotion



def baseline_manifest_v1() -> dict[str, Any]:
    document = {
        "schema": "12-6.migration-contract-baseline.v1",
        "architecture_sha256": canonical_system_architecture_v1().identity_sha256(),
        "runtime_shell_sha256": canonical_runtime_shell_v1().identity_sha256(),
        "artifact_kinds": [kind.value for kind in ArtifactKind],
        "schemas": sorted(_SCHEMAS),
        "authorities": ["twelve_six.artifact_identity", "twelve_six.system_architecture"],
        "plans_2_to_8_nonblocking": True,
        "contains_execution_authority": False,
    }
    document["baseline_sha256"] = _sha(_json(document))
    return document


def encode_v1(kind: str, value: Any) -> bytes:
    if type(kind) is not str or kind not in _SCHEMAS:
        raise ContractPackageError("unknown contract type")
    if type(value) is not _SCHEMAS[kind]:
        raise ContractPackageError("contract class identity mismatch")
    envelope = {"schema": "12-6.contract-envelope.v1", "kind": kind, "version": 1,
                "baseline_sha256": baseline_manifest_v1()["baseline_sha256"],
                "payload": value.to_dict()}
    result = _json(envelope)
    if len(result) > _LIMIT:
        raise ContractPackageError("contract exceeds size limit")
    return result


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ContractPackageError("duplicate contract field")
        value[key] = item
    return value


def _reject_constant(text: str) -> None:
    raise ContractPackageError("non-finite contract number")


def decode_v1(data: bytes) -> Any:
    if type(data) is not bytes or len(data) > _LIMIT:
        raise ContractPackageError("invalid/oversized contract bytes")
    try:
        raw = json.loads(data.decode("utf-8", "strict"), object_pairs_hook=_no_duplicates,
                         parse_constant=_reject_constant)
    except ContractPackageError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ContractPackageError("invalid contract JSON") from exc
    envelope = _fields(raw, {"schema", "kind", "version", "baseline_sha256", "payload"})
    kind = envelope["kind"]
    if (envelope["schema"] != "12-6.contract-envelope.v1"
            or type(envelope["version"]) is not int or envelope["version"] != 1
            or type(kind) is not str or kind not in _SCHEMAS
            or envelope["baseline_sha256"] != baseline_manifest_v1()["baseline_sha256"]):
        raise ContractPackageError("unsupported version/baseline/kind")
    item = _SCHEMAS[kind].from_dict(envelope["payload"])
    if encode_v1(kind, item) != data:
        raise ContractPackageError("noncanonical contract bytes")
    return item
