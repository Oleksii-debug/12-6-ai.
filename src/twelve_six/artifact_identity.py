from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
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
    if not _is_exact_type(data, bytes):
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
    if not _is_exact_type(value, dict):
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
    if not _is_exact_type(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _is_exact_type(value: object, expected: type[object]) -> bool:
    # Closed manifest schemas reject behavioral subclasses that can override serialization.
    return type(value) is expected  # noqa: E721


def _require_exact_object_fields(
    schema_name: str,
    value: object,
    expected_fields: set[str],
) -> dict[str, Any]:
    if not _is_exact_type(value, dict):
        raise ValueError(f"{schema_name} fields mismatch")
    if any(not _is_exact_type(key, str) for key in value):
        raise ValueError(f"{schema_name} fields mismatch")
    if set(value) != expected_fields:
        raise ValueError(f"{schema_name} fields mismatch")
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


def _artifact_kind_wire_value(value: object) -> str:
    if not _is_exact_type(value, ArtifactKind):
        raise ValueError("kind must be an ArtifactKind")
    raw_value = str.__str__(value)
    stored_value = object.__getattribute__(value, "_value_")
    if not _is_exact_type(stored_value, str) or stored_value != raw_value:
        raise ValueError("ArtifactKind wire value is non-canonical")
    return raw_value


CANONICAL_ARTIFACT_KINDS = tuple(ArtifactKind)
_CANONICAL_ARTIFACT_KIND_VALUES = tuple(
    _artifact_kind_wire_value(kind) for kind in CANONICAL_ARTIFACT_KINDS
)


def _require_canonical_artifact_kind(
    kind: object,
    _sealed_kinds: tuple[ArtifactKind, ...] = CANONICAL_ARTIFACT_KINDS,
    _sealed_values: tuple[str, ...] = _CANONICAL_ARTIFACT_KIND_VALUES,
) -> ArtifactKind:
    if not _is_exact_type(kind, ArtifactKind):
        raise ValueError("kind must be an ArtifactKind")
    for index, canonical_kind in enumerate(_sealed_kinds):
        if kind is canonical_kind:
            if _artifact_kind_wire_value(canonical_kind) != _sealed_values[index]:
                raise ValueError("ArtifactKind wire value is non-canonical")
            return canonical_kind
    raise ValueError("kind must be a canonical ArtifactKind")


def _artifact_kind_from_wire_value(
    value: object,
    _sealed_kinds: tuple[ArtifactKind, ...] = CANONICAL_ARTIFACT_KINDS,
    _sealed_values: tuple[str, ...] = _CANONICAL_ARTIFACT_KIND_VALUES,
) -> ArtifactKind:
    if not _is_exact_type(value, str):
        raise ValueError("ArtifactRef kind must be an exact string")
    try:
        index = _sealed_values.index(value)
    except ValueError as exc:
        raise ValueError("ArtifactRef kind is unsupported") from exc
    kind = _sealed_kinds[index]
    return _require_canonical_artifact_kind(kind)


_GENERATION_PARENT_POLICY_SOURCE: dict[ArtifactKind, dict[str, ArtifactKind]] = {
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

_GENERATION_PARENT_POLICY: Mapping[ArtifactKind, Mapping[str, ArtifactKind]] = MappingProxyType(
    {
        kind: MappingProxyType(dict(parents))
        for kind, parents in _GENERATION_PARENT_POLICY_SOURCE.items()
    }
)
_GENERATION_PARENT_POLICY_VALUES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        _artifact_kind_wire_value(kind): MappingProxyType(
            {
                role: _artifact_kind_wire_value(parent_kind)
                for role, parent_kind in parents.items()
            }
        )
        for kind, parents in _GENERATION_PARENT_POLICY.items()
    }
)
del _GENERATION_PARENT_POLICY_SOURCE


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Versioned reference to one existing domain artifact identity."""

    kind: ArtifactKind
    schema_version: int
    identity_sha256: str

    def __post_init__(self) -> None:
        _require_canonical_artifact_kind(self.kind)
        _require_positive_int("schema_version", self.schema_version)
        _require_sha256("identity_sha256", self.identity_sha256)

    def to_dict(self) -> dict[str, Any]:
        ArtifactRef.__post_init__(self)
        return {
            "kind": _artifact_kind_wire_value(self.kind),
            "schema_version": self.schema_version,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ArtifactRef:
        if cls is not ArtifactRef:
            raise ValueError("ArtifactRef decoder class must be exact")
        value = _require_exact_object_fields(
            "ArtifactRef",
            value,
            {"kind", "schema_version", "identity_sha256"},
        )
        kind = _artifact_kind_from_wire_value(value["kind"])
        return cls(
            kind=kind,
            schema_version=value["schema_version"],
            identity_sha256=value["identity_sha256"],
        )


def _artifact_ref_signature(value: object) -> tuple[str, int, str]:
    if not _is_exact_type(value, ArtifactRef):
        raise ValueError("artifact identity must be an ArtifactRef")
    ArtifactRef.__post_init__(value)
    return (
        _artifact_kind_wire_value(value.kind),
        value.schema_version,
        value.identity_sha256,
    )


@dataclass(frozen=True, slots=True)
class ParentBinding:
    """Named exact parent reference plus its transitive manifest identity."""

    role: str
    artifact: ArtifactRef
    parent_manifest_identity_sha256: str

    def __post_init__(self) -> None:
        if not _is_exact_type(self.role, str) or _ROLE_RE.fullmatch(self.role) is None:
            raise ValueError("parent role must be canonical lower_snake_case")
        if not _is_exact_type(self.artifact, ArtifactRef):
            raise ValueError("parent artifact must be an ArtifactRef")
        ArtifactRef.__post_init__(self.artifact)
        _require_sha256(
            "parent_manifest_identity_sha256",
            self.parent_manifest_identity_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        ParentBinding.__post_init__(self)
        return {
            "role": self.role,
            "artifact": self.artifact.to_dict(),
            "parent_manifest_identity_sha256": self.parent_manifest_identity_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ParentBinding:
        if cls is not ParentBinding:
            raise ValueError("ParentBinding decoder class must be exact")
        value = _require_exact_object_fields(
            "ParentBinding",
            value,
            {"role", "artifact", "parent_manifest_identity_sha256"},
        )
        return cls(
            role=value["role"],
            artifact=ArtifactRef.from_dict(value["artifact"]),
            parent_manifest_identity_sha256=value["parent_manifest_identity_sha256"],
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
        if not _is_exact_type(self.artifact, ArtifactRef):
            raise ValueError("artifact must be an ArtifactRef")
        if not _is_exact_type(self.parents, tuple):
            raise ValueError("parents must be an immutable tuple")
        if any(not _is_exact_type(parent, ParentBinding) for parent in self.parents):
            raise ValueError("parents must contain only ParentBinding values")
        ArtifactRef.__post_init__(self.artifact)
        for parent in self.parents:
            ParentBinding.__post_init__(parent)

        roles = tuple(parent.role for parent in self.parents)
        if roles != tuple(sorted(roles)):
            raise ValueError("parent bindings must use canonical role order")
        if len(set(roles)) != len(roles):
            raise ValueError("parent binding roles must be unique")

        ref_signatures = tuple(
            _artifact_ref_signature(parent.artifact) for parent in self.parents
        )
        if len(set(ref_signatures)) != len(ref_signatures):
            raise ValueError("the same exact parent artifact cannot be bound twice")
        if _artifact_ref_signature(self.artifact) in ref_signatures:
            raise ValueError("artifact cannot bind itself as a parent")

    def to_dict(self) -> dict[str, Any]:
        ArtifactManifest.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "artifact": self.artifact.to_dict(),
            "parents": [parent.to_dict() for parent in self.parents],
        }

    def manifest_identity_sha256(
        self,
        _sealed_hash_payload=_canonical_json_sha256,
    ) -> str:
        return _sealed_hash_payload(self.to_dict())

    def parents_by_role(self) -> dict[str, ArtifactRef]:
        ArtifactManifest.__post_init__(self)
        return {parent.role: parent.artifact for parent in self.parents}

    def parent_bindings_by_role(self) -> dict[str, ParentBinding]:
        ArtifactManifest.__post_init__(self)
        return {parent.role: parent for parent in self.parents}

    @classmethod
    def from_dict(cls, value: object) -> ArtifactManifest:
        if cls is not ArtifactManifest:
            raise ValueError("ArtifactManifest decoder class must be exact")
        value = _require_exact_object_fields(
            "ArtifactManifest",
            value,
            {"schema_version", "artifact", "parents"},
        )
        parents = value["parents"]
        if not _is_exact_type(parents, list):
            raise ValueError("ArtifactManifest parents must be a JSON array")
        return cls(
            schema_version=value["schema_version"],
            artifact=ArtifactRef.from_dict(value["artifact"]),
            parents=tuple(ParentBinding.from_dict(parent) for parent in parents),
        )


def _artifact_manifest_identity_from_stored_state(
    manifest: ArtifactManifest,
    _sealed_hash_payload=_canonical_json_sha256,
) -> str:
    """Hash exact stored manifest state without dispatching mutable manifest view methods."""

    if not _is_exact_type(manifest, ArtifactManifest):
        raise ValueError("manifest must be an ArtifactManifest")
    ArtifactManifest.__post_init__(manifest)
    payload = {
        "schema_version": manifest.schema_version,
        "artifact": {
            "kind": _artifact_kind_wire_value(manifest.artifact.kind),
            "schema_version": manifest.artifact.schema_version,
            "identity_sha256": manifest.artifact.identity_sha256,
        },
        "parents": [
            {
                "role": parent.role,
                "artifact": {
                    "kind": _artifact_kind_wire_value(parent.artifact.kind),
                    "schema_version": parent.artifact.schema_version,
                    "identity_sha256": parent.artifact.identity_sha256,
                },
                "parent_manifest_identity_sha256": (
                    parent.parent_manifest_identity_sha256
                ),
            }
            for parent in manifest.parents
        ],
    }
    return _sealed_hash_payload(payload)


def bind_artifact(
    artifact: ArtifactRef,
    *,
    parents: Mapping[str, ArtifactManifest] | None = None,
) -> ArtifactManifest:
    """Create a canonical binding that commits to each parent's bound lineage."""

    if not _is_exact_type(artifact, ArtifactRef):
        raise ValueError("artifact must be an ArtifactRef")
    ArtifactRef.__post_init__(artifact)
    if parents is None:
        normalized: dict[str, ArtifactManifest] = {}
    else:
        if not _is_exact_type(parents, dict):
            raise ValueError("parents must be an exact dict mapping")
        normalized = {}
        for role, parent in parents.items():
            if not _is_exact_type(role, str) or _ROLE_RE.fullmatch(role) is None:
                raise ValueError("parent role must be canonical lower_snake_case")
            if not _is_exact_type(parent, ArtifactManifest):
                raise ValueError("parent mapping values must be ArtifactManifest values")
            ArtifactManifest.__post_init__(parent)
            normalized[role] = parent

    return ArtifactManifest(
        schema_version=1,
        artifact=artifact,
        parents=tuple(
            ParentBinding(
                role=role,
                artifact=normalized[role].artifact,
                parent_manifest_identity_sha256=(
                    normalized[role].manifest_identity_sha256()
                ),
            )
            for role in sorted(normalized)
        ),
    )


def verify_parent_bindings(
    manifest: ArtifactManifest,
    *,
    expected_parents: Mapping[str, ArtifactManifest],
) -> None:
    """Fail closed unless exact parent artifacts and bound lineages both match."""

    if not _is_exact_type(manifest, ArtifactManifest):
        raise ValueError("manifest must be an ArtifactManifest")
    ArtifactManifest.__post_init__(manifest)
    if not _is_exact_type(expected_parents, dict):
        raise ValueError("expected_parents must be an exact dict mapping")

    normalized: dict[str, ArtifactManifest] = {}
    for role, parent in expected_parents.items():
        if not _is_exact_type(role, str) or _ROLE_RE.fullmatch(role) is None:
            raise ValueError("expected parent role must be canonical lower_snake_case")
        if not _is_exact_type(parent, ArtifactManifest):
            raise ValueError("expected parent values must be ArtifactManifest values")
        ArtifactManifest.__post_init__(parent)
        normalized[role] = parent

    observed = manifest.parent_bindings_by_role()
    if set(observed) != set(normalized):
        raise ValueError("artifact parent role set mismatch")
    for role in sorted(normalized):
        expected = normalized[role]
        binding = observed[role]
        if _artifact_ref_signature(binding.artifact) != _artifact_ref_signature(
            expected.artifact
        ):
            raise ValueError(f"artifact parent identity mismatch for role: {role}")
        if (
            binding.parent_manifest_identity_sha256
            != expected.manifest_identity_sha256()
        ):
            raise ValueError(f"artifact parent lineage mismatch for role: {role}")


@dataclass(frozen=True, slots=True)
class GenerationIdentityManifest:
    """Closed-world identity graph for one selected 12-6 product generation."""

    schema_version: int
    artifacts: tuple[ArtifactManifest, ...]

    def __post_init__(
        self,
        _sealed_artifact_kind_values: tuple[str, ...] = _CANONICAL_ARTIFACT_KIND_VALUES,
        _sealed_parent_policy_values: Mapping[str, Mapping[str, str]] = (
            _GENERATION_PARENT_POLICY_VALUES
        ),
        _sealed_manifest_identity: Callable[[ArtifactManifest], str] = (
            _artifact_manifest_identity_from_stored_state
        ),
    ) -> None:
        _require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError("unsupported GenerationIdentityManifest schema_version")
        if not _is_exact_type(self.artifacts, tuple):
            raise ValueError("artifacts must be an immutable tuple")
        if len(self.artifacts) != len(_sealed_artifact_kind_values):
            raise ValueError("generation must contain exactly one artifact of every canonical kind")
        if any(not _is_exact_type(item, ArtifactManifest) for item in self.artifacts):
            raise ValueError("generation artifacts must contain only ArtifactManifest values")
        for item in self.artifacts:
            ArtifactManifest.__post_init__(item)

        kind_values = tuple(
            _artifact_kind_wire_value(item.artifact.kind) for item in self.artifacts
        )
        if kind_values != _sealed_artifact_kind_values:
            raise ValueError("generation artifact kind order or set is non-canonical")

        by_kind_value = {
            _artifact_kind_wire_value(item.artifact.kind): item
            for item in self.artifacts
        }
        if len(by_kind_value) != len(_sealed_artifact_kind_values):
            raise ValueError("generation artifact kinds must be unique")

        for kind_value in _sealed_artifact_kind_values:
            manifest = by_kind_value[kind_value]
            expected_policy = _sealed_parent_policy_values[kind_value]
            observed = {parent.role: parent for parent in manifest.parents}
            if tuple(observed) != tuple(sorted(expected_policy)):
                raise ValueError(f"{kind_value} parent role set is non-canonical")
            for role, parent_kind_value in expected_policy.items():
                binding = observed[role]
                if _artifact_kind_wire_value(binding.artifact.kind) != parent_kind_value:
                    raise ValueError(
                        f"{kind_value}.{role} must reference {parent_kind_value}"
                    )
                canonical_parent = by_kind_value[parent_kind_value]
                if _artifact_ref_signature(binding.artifact) != _artifact_ref_signature(
                    canonical_parent.artifact
                ):
                    raise ValueError(
                        f"{kind_value}.{role} parent identity does not match generation"
                    )
                if (
                    binding.parent_manifest_identity_sha256
                    != _sealed_manifest_identity(canonical_parent)
                ):
                    raise ValueError(
                        f"{kind_value}.{role} parent lineage identity "
                        "does not match generation"
                    )

    def to_dict(self) -> dict[str, Any]:
        GenerationIdentityManifest.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    def identity_sha256(
        self,
        _sealed_hash_payload=_canonical_json_sha256,
    ) -> str:
        return _sealed_hash_payload(self.to_dict())

    def artifact_ref(
        self,
        kind: ArtifactKind,
        _sealed_artifact_kind_values: tuple[str, ...] = _CANONICAL_ARTIFACT_KIND_VALUES,
    ) -> ArtifactRef:
        GenerationIdentityManifest.__post_init__(self)
        if not _is_exact_type(kind, ArtifactKind):
            raise ValueError("kind must be an ArtifactKind")
        kind_value = _artifact_kind_wire_value(kind)
        return self.artifacts[_sealed_artifact_kind_values.index(kind_value)].artifact

    def artifact_manifest(
        self,
        kind: ArtifactKind,
        _sealed_artifact_kind_values: tuple[str, ...] = _CANONICAL_ARTIFACT_KIND_VALUES,
    ) -> ArtifactManifest:
        GenerationIdentityManifest.__post_init__(self)
        if not _is_exact_type(kind, ArtifactKind):
            raise ValueError("kind must be an ArtifactKind")
        kind_value = _artifact_kind_wire_value(kind)
        return self.artifacts[_sealed_artifact_kind_values.index(kind_value)]

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
        if cls is not GenerationIdentityManifest:
            raise ValueError("GenerationIdentityManifest decoder class must be exact")
        value = _require_exact_object_fields(
            "GenerationIdentityManifest",
            value,
            {"schema_version", "artifacts"},
        )
        artifacts = value["artifacts"]
        if not _is_exact_type(artifacts, list):
            raise ValueError("GenerationIdentityManifest artifacts must be a JSON array")
        return cls(
            schema_version=value["schema_version"],
            artifacts=tuple(ArtifactManifest.from_dict(item) for item in artifacts),
        )


def parse_generation_identity_manifest(data: bytes) -> GenerationIdentityManifest:
    """Decode one exact canonical durable closed-world generation manifest."""

    manifest = GenerationIdentityManifest.from_dict(_strict_json_object(data))
    if data != manifest.canonical_json_bytes():
        raise ValueError("manifest must use canonical JSON encoding")
    return manifest


def build_generation_identity_manifest(
    refs: Mapping[ArtifactKind, ArtifactRef],
) -> GenerationIdentityManifest:
    """Cross-bind one exact reference per required identity kind into a transitive graph."""

    if not _is_exact_type(refs, dict):
        raise ValueError("refs must be an exact dict mapping")
    if any(not _is_exact_type(kind, ArtifactKind) for kind in refs):
        raise ValueError("refs keys must be ArtifactKind values")
    if set(refs) != set(CANONICAL_ARTIFACT_KINDS):
        raise ValueError("refs must contain exactly every canonical artifact kind")

    normalized: dict[ArtifactKind, ArtifactRef] = {}
    for kind in CANONICAL_ARTIFACT_KINDS:
        ref = refs[kind]
        if not _is_exact_type(ref, ArtifactRef):
            raise ValueError("refs values must be ArtifactRef values")
        ArtifactRef.__post_init__(ref)
        if ref.kind is not kind:
            raise ValueError(
                f"ref kind mismatch for key: {_artifact_kind_wire_value(kind)}"
            )
        normalized[kind] = ref

    manifests: list[ArtifactManifest] = []
    by_kind: dict[ArtifactKind, ArtifactManifest] = {}
    for kind in CANONICAL_ARTIFACT_KINDS:
        policy = _GENERATION_PARENT_POLICY[kind]
        manifest = bind_artifact(
            normalized[kind],
            parents={
                role: by_kind[parent_kind]
                for role, parent_kind in policy.items()
            },
        )
        manifests.append(manifest)
        by_kind[kind] = manifest
    return GenerationIdentityManifest(schema_version=1, artifacts=tuple(manifests))
