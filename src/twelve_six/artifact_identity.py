from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_MANIFEST_BYTES = 1024 * 1024


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError("non-finite JSON constants are not allowed")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON numbers are not allowed")
    return parsed


def _strict_json_object(data: bytes) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise ValueError("manifest input must be bytes")
    if len(data) > _MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds maximum encoded size")
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("manifest is not strict unambiguous UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("manifest root must be a JSON object")
    return value


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


class ArtifactKind(str, Enum):
    MODEL_SPEC = "model_spec"
    INIT_SPEC = "init_spec"
    CORPUS = "corpus"
    TOKENIZER = "tokenizer"
    SPLIT = "split"
    PACKING = "packing"
    EXPOSURE_LEDGER = "exposure_ledger"
    TRAINING_RUN = "training_run"
    CHECKPOINT = "checkpoint"
    EVALUATION = "evaluation"
    EXPORT = "export"
    RELEASE = "release"


CANONICAL_ARTIFACT_KINDS = tuple(ArtifactKind)


_GENERATION_PARENT_POLICY: dict[ArtifactKind, dict[str, ArtifactKind]] = {
    ArtifactKind.MODEL_SPEC: {},
    ArtifactKind.INIT_SPEC: {},
    ArtifactKind.CORPUS: {},
    ArtifactKind.TOKENIZER: {
        "corpus": ArtifactKind.CORPUS,
    },
    ArtifactKind.SPLIT: {
        "corpus": ArtifactKind.CORPUS,
    },
    ArtifactKind.PACKING: {
        "split": ArtifactKind.SPLIT,
        "tokenizer": ArtifactKind.TOKENIZER,
    },
    ArtifactKind.EXPOSURE_LEDGER: {
        "packing": ArtifactKind.PACKING,
    },
    ArtifactKind.TRAINING_RUN: {
        "corpus": ArtifactKind.CORPUS,
        "exposure_ledger": ArtifactKind.EXPOSURE_LEDGER,
        "init_spec": ArtifactKind.INIT_SPEC,
        "model_spec": ArtifactKind.MODEL_SPEC,
        "packing": ArtifactKind.PACKING,
        "split": ArtifactKind.SPLIT,
        "tokenizer": ArtifactKind.TOKENIZER,
    },
    ArtifactKind.CHECKPOINT: {
        "training_run": ArtifactKind.TRAINING_RUN,
    },
    ArtifactKind.EVALUATION: {
        "checkpoint": ArtifactKind.CHECKPOINT,
        "split": ArtifactKind.SPLIT,
    },
    ArtifactKind.EXPORT: {
        "checkpoint": ArtifactKind.CHECKPOINT,
    },
    ArtifactKind.RELEASE: {
        "checkpoint": ArtifactKind.CHECKPOINT,
        "evaluation": ArtifactKind.EVALUATION,
        "export": ArtifactKind.EXPORT,
    },
}


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Versioned reference to one existing domain artifact identity."""

    kind: ArtifactKind
    schema_version: int
    identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ArtifactKind):
            raise ValueError("kind must be an ArtifactKind")
        _require_positive_int("schema_version", self.schema_version)
        _require_sha256("identity_sha256", self.identity_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "schema_version": self.schema_version,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ArtifactRef:
        if not isinstance(value, dict) or set(value) != {
            "kind",
            "schema_version",
            "identity_sha256",
        }:
            raise ValueError("ArtifactRef fields mismatch")
        try:
            kind = ArtifactKind(value["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("ArtifactRef kind is unsupported") from exc
        return cls(
            kind=kind,
            schema_version=value["schema_version"],
            identity_sha256=value["identity_sha256"],
        )


@dataclass(frozen=True, slots=True)
class ParentBinding:
    """Named exact parent reference of a derived artifact."""

    role: str
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or _ROLE_RE.fullmatch(self.role) is None:
            raise ValueError("parent role must be canonical lower_snake_case")
        if not isinstance(self.artifact, ArtifactRef):
            raise ValueError("parent artifact must be an ArtifactRef")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "artifact": self.artifact.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ParentBinding:
        if not isinstance(value, dict) or set(value) != {"role", "artifact"}:
            raise ValueError("ParentBinding fields mismatch")
        return cls(
            role=value["role"],
            artifact=ArtifactRef.from_dict(value["artifact"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """One artifact plus exact versioned parent bindings."""

    schema_version: int
    artifact: ArtifactRef
    parents: tuple[ParentBinding, ...]

    def __post_init__(self) -> None:
        _require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError("unsupported ArtifactManifest schema_version")
        if not isinstance(self.artifact, ArtifactRef):
            raise ValueError("artifact must be an ArtifactRef")
        if not isinstance(self.parents, tuple):
            raise ValueError("parents must be an immutable tuple")
        if any(not isinstance(parent, ParentBinding) for parent in self.parents):
            raise ValueError("parents must contain only ParentBinding values")

        roles = tuple(parent.role for parent in self.parents)
        if roles != tuple(sorted(roles)):
            raise ValueError("parent bindings must use canonical role order")
        if len(set(roles)) != len(roles):
            raise ValueError("parent binding roles must be unique")

        refs = tuple(parent.artifact for parent in self.parents)
        if len(set(refs)) != len(refs):
            raise ValueError("the same exact parent artifact cannot be bound twice")
        if self.artifact in refs:
            raise ValueError("artifact cannot bind itself as a parent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact": self.artifact.to_dict(),
            "parents": [parent.to_dict() for parent in self.parents],
        }

    def manifest_identity_sha256(self) -> str:
        return _canonical_json_sha256(self.to_dict())

    def parents_by_role(self) -> dict[str, ArtifactRef]:
        return {parent.role: parent.artifact for parent in self.parents}

    @classmethod
    def from_dict(cls, value: object) -> ArtifactManifest:
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "artifact",
            "parents",
        }:
            raise ValueError("ArtifactManifest fields mismatch")
        parents = value["parents"]
        if not isinstance(parents, list):
            raise ValueError("ArtifactManifest parents must be a JSON array")
        return cls(
            schema_version=value["schema_version"],
            artifact=ArtifactRef.from_dict(value["artifact"]),
            parents=tuple(ParentBinding.from_dict(parent) for parent in parents),
        )


def bind_artifact(
    artifact: ArtifactRef,
    *,
    parents: Mapping[str, ArtifactRef] | None = None,
) -> ArtifactManifest:
    """Create a canonically ordered cross-binding around an existing identity."""

    if not isinstance(artifact, ArtifactRef):
        raise ValueError("artifact must be an ArtifactRef")
    if parents is None:
        normalized: dict[str, ArtifactRef] = {}
    else:
        if not isinstance(parents, Mapping):
            raise ValueError("parents must be a mapping")
        normalized = {}
        for role, parent in parents.items():
            if not isinstance(role, str) or _ROLE_RE.fullmatch(role) is None:
                raise ValueError("parent role must be canonical lower_snake_case")
            if not isinstance(parent, ArtifactRef):
                raise ValueError("parent mapping values must be ArtifactRef values")
            normalized[role] = parent

    return ArtifactManifest(
        schema_version=1,
        artifact=artifact,
        parents=tuple(
            ParentBinding(role, normalized[role])
            for role in sorted(normalized)
        ),
    )


def verify_parent_bindings(
    manifest: ArtifactManifest,
    *,
    expected_parents: Mapping[str, ArtifactRef],
) -> None:
    """Fail closed unless role names and exact versioned identities match."""

    if not isinstance(manifest, ArtifactManifest):
        raise ValueError("manifest must be an ArtifactManifest")
    if not isinstance(expected_parents, Mapping):
        raise ValueError("expected_parents must be a mapping")

    normalized: dict[str, ArtifactRef] = {}
    for role, parent in expected_parents.items():
        if not isinstance(role, str) or _ROLE_RE.fullmatch(role) is None:
            raise ValueError("expected parent role must be canonical lower_snake_case")
        if not isinstance(parent, ArtifactRef):
            raise ValueError("expected parent values must be ArtifactRef values")
        normalized[role] = parent

    observed = manifest.parents_by_role()
    if set(observed) != set(normalized):
        raise ValueError("artifact parent role set mismatch")
    for role in sorted(normalized):
        if observed[role] != normalized[role]:
            raise ValueError(f"artifact parent identity mismatch for role: {role}")


@dataclass(frozen=True, slots=True)
class GenerationIdentityManifest:
    """Closed-world identity graph for one selected 12-6 product generation."""

    schema_version: int
    artifacts: tuple[ArtifactManifest, ...]

    def __post_init__(self) -> None:
        _require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError("unsupported GenerationIdentityManifest schema_version")
        if not isinstance(self.artifacts, tuple):
            raise ValueError("artifacts must be an immutable tuple")
        if len(self.artifacts) != len(CANONICAL_ARTIFACT_KINDS):
            raise ValueError("generation must contain exactly one artifact of every canonical kind")
        if any(not isinstance(item, ArtifactManifest) for item in self.artifacts):
            raise ValueError("generation artifacts must contain only ArtifactManifest values")

        kinds = tuple(item.artifact.kind for item in self.artifacts)
        if kinds != CANONICAL_ARTIFACT_KINDS:
            raise ValueError("generation artifact kind order or set is non-canonical")

        by_kind = {item.artifact.kind: item for item in self.artifacts}
        if len(by_kind) != len(CANONICAL_ARTIFACT_KINDS):
            raise ValueError("generation artifact kinds must be unique")

        for kind in CANONICAL_ARTIFACT_KINDS:
            manifest = by_kind[kind]
            expected_policy = _GENERATION_PARENT_POLICY[kind]
            observed = manifest.parents_by_role()
            if tuple(observed) != tuple(sorted(expected_policy)):
                raise ValueError(f"{kind.value} parent role set is non-canonical")
            for role, parent_kind in expected_policy.items():
                parent = observed[role]
                if parent.kind is not parent_kind:
                    raise ValueError(
                        f"{kind.value}.{role} must reference {parent_kind.value}"
                    )
                canonical_parent = by_kind[parent_kind].artifact
                if parent != canonical_parent:
                    raise ValueError(
                        f"{kind.value}.{role} parent identity does not match generation"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    def identity_sha256(self) -> str:
        return _canonical_json_sha256(self.to_dict())

    def artifact_ref(self, kind: ArtifactKind) -> ArtifactRef:
        if not isinstance(kind, ArtifactKind):
            raise ValueError("kind must be an ArtifactKind")
        return self.artifacts[CANONICAL_ARTIFACT_KINDS.index(kind)].artifact

    def artifact_manifest(self, kind: ArtifactKind) -> ArtifactManifest:
        if not isinstance(kind, ArtifactKind):
            raise ValueError("kind must be an ArtifactKind")
        return self.artifacts[CANONICAL_ARTIFACT_KINDS.index(kind)]

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: object) -> GenerationIdentityManifest:
        if not isinstance(value, dict) or set(value) != {"schema_version", "artifacts"}:
            raise ValueError("GenerationIdentityManifest fields mismatch")
        artifacts = value["artifacts"]
        if not isinstance(artifacts, list):
            raise ValueError("GenerationIdentityManifest artifacts must be a JSON array")
        return cls(
            schema_version=value["schema_version"],
            artifacts=tuple(ArtifactManifest.from_dict(item) for item in artifacts),
        )


def parse_generation_identity_manifest(data: bytes) -> GenerationIdentityManifest:
    """Decode and validate one durable closed-world generation manifest."""

    return GenerationIdentityManifest.from_dict(_strict_json_object(data))


def build_generation_identity_manifest(
    refs: Mapping[ArtifactKind, ArtifactRef],
) -> GenerationIdentityManifest:
    """Cross-bind one exact reference per required identity kind into a closed graph."""

    if not isinstance(refs, Mapping):
        raise ValueError("refs must be a mapping")
    if set(refs) != set(CANONICAL_ARTIFACT_KINDS):
        raise ValueError("refs must contain exactly every canonical artifact kind")

    normalized: dict[ArtifactKind, ArtifactRef] = {}
    for kind in CANONICAL_ARTIFACT_KINDS:
        ref = refs[kind]
        if not isinstance(ref, ArtifactRef):
            raise ValueError("refs values must be ArtifactRef values")
        if ref.kind is not kind:
            raise ValueError(f"ref kind mismatch for key: {kind.value}")
        normalized[kind] = ref

    manifests = []
    for kind in CANONICAL_ARTIFACT_KINDS:
        policy = _GENERATION_PARENT_POLICY[kind]
        manifests.append(
            bind_artifact(
                normalized[kind],
                parents={
                    role: normalized[parent_kind]
                    for role, parent_kind in policy.items()
                },
            )
        )
    return GenerationIdentityManifest(schema_version=1, artifacts=tuple(manifests))
