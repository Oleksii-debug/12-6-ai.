from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


_HEX = frozenset("0123456789abcdef")


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field} must be an exact lowercase SHA-256 digest")
    return value


def _require_positive_version(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


class ArtifactKind(StrEnum):
    """Canonical Section-1 identity domains."""

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


_REQUIRED_PARENT_KINDS: dict[ArtifactKind, frozenset[ArtifactKind]] = {
    ArtifactKind.MODEL_SPEC: frozenset(),
    ArtifactKind.INIT_SPEC: frozenset(),
    ArtifactKind.CORPUS: frozenset(),
    ArtifactKind.TOKENIZER: frozenset({ArtifactKind.CORPUS}),
    ArtifactKind.SPLIT: frozenset({ArtifactKind.CORPUS}),
    ArtifactKind.PACKING: frozenset({ArtifactKind.TOKENIZER, ArtifactKind.SPLIT}),
    ArtifactKind.EXPOSURE_LEDGER: frozenset({ArtifactKind.PACKING}),
    ArtifactKind.TRAINING_RUN: frozenset(
        {
            ArtifactKind.MODEL_SPEC,
            ArtifactKind.INIT_SPEC,
            ArtifactKind.EXPOSURE_LEDGER,
        }
    ),
    ArtifactKind.CHECKPOINT: frozenset({ArtifactKind.TRAINING_RUN}),
    ArtifactKind.EVALUATION: frozenset({ArtifactKind.CHECKPOINT}),
    ArtifactKind.EXPORT: frozenset({ArtifactKind.CHECKPOINT}),
    ArtifactKind.RELEASE: frozenset(
        {
            ArtifactKind.CHECKPOINT,
            ArtifactKind.EVALUATION,
            ArtifactKind.EXPORT,
        }
    ),
}


@dataclass(frozen=True, slots=True)
class IdentityRef:
    """Exact reference to an upstream artifact and its transitive binding."""

    artifact_id: str
    kind: ArtifactKind
    schema_version: int
    artifact_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty text")
        if not isinstance(self.kind, ArtifactKind):
            raise ValueError("kind must be an ArtifactKind")
        _require_positive_version(self.schema_version, "schema_version")
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        _require_sha256(self.binding_sha256, "binding_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "schema_version": self.schema_version,
            "artifact_sha256": self.artifact_sha256,
            "binding_sha256": self.binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Versioned identity whose binding commits to exact upstream identities."""

    artifact_id: str
    kind: ArtifactKind
    schema_version: int
    artifact_sha256: str
    parents: tuple[IdentityRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty text")
        if not isinstance(self.kind, ArtifactKind):
            raise ValueError("kind must be an ArtifactKind")
        _require_positive_version(self.schema_version, "schema_version")
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        if not isinstance(self.parents, tuple):
            raise ValueError("parents must be a tuple of IdentityRef values")
        if any(not isinstance(parent, IdentityRef) for parent in self.parents):
            raise ValueError("parents must contain only IdentityRef values")
        parent_ids = [parent.artifact_id for parent in self.parents]
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("parent artifact_id values must be unique")
        if self.artifact_id in parent_ids:
            raise ValueError("artifact cannot directly reference itself")

    def _binding_payload(self) -> dict[str, object]:
        parents = sorted(
            (parent.to_dict() for parent in self.parents),
            key=lambda item: (
                str(item["kind"]),
                str(item["artifact_id"]),
                int(item["schema_version"]),
            ),
        )
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "schema_version": self.schema_version,
            "artifact_sha256": self.artifact_sha256,
            "parents": parents,
        }

    @property
    def binding_sha256(self) -> str:
        return _canonical_sha256(self._binding_payload())

    def as_ref(self) -> IdentityRef:
        return IdentityRef(
            artifact_id=self.artifact_id,
            kind=self.kind,
            schema_version=self.schema_version,
            artifact_sha256=self.artifact_sha256,
            binding_sha256=self.binding_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        payload = self._binding_payload()
        payload["binding_sha256"] = self.binding_sha256
        return payload


@dataclass(frozen=True, slots=True)
class IdentityManifest:
    """Fail-closed DAG of Section-1 identities and cryptographic cross-bindings."""

    schema_version: int
    artifacts: tuple[ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported IdentityManifest schema_version: {self.schema_version}")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise ValueError("artifacts must be a non-empty tuple")
        if any(not isinstance(artifact, ArtifactIdentity) for artifact in self.artifacts):
            raise ValueError("artifacts must contain only ArtifactIdentity values")

        by_id = {artifact.artifact_id: artifact for artifact in self.artifacts}
        if len(by_id) != len(self.artifacts):
            raise ValueError("artifact_id values must be unique")

        self._reject_cycles(by_id)

        for artifact in self.artifacts:
            resolved_parent_kinds: set[ArtifactKind] = set()
            for parent_ref in artifact.parents:
                parent = by_id.get(parent_ref.artifact_id)
                if parent is None:
                    raise ValueError(
                        f"{artifact.artifact_id} references unknown parent "
                        f"{parent_ref.artifact_id}"
                    )
                expected = parent.as_ref()
                if parent_ref != expected:
                    raise ValueError(
                        f"{artifact.artifact_id} parent reference does not match exact "
                        f"identity for {parent_ref.artifact_id}"
                    )
                resolved_parent_kinds.add(parent.kind)

            required = _REQUIRED_PARENT_KINDS[artifact.kind]
            missing = required.difference(resolved_parent_kinds)
            if missing:
                names = sorted(kind.value for kind in missing)
                raise ValueError(
                    f"{artifact.artifact_id} missing required upstream kinds: {names}"
                )

    @staticmethod
    def _reject_cycles(by_id: dict[str, ArtifactIdentity]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(artifact_id: str) -> None:
            if artifact_id in visited:
                return
            if artifact_id in visiting:
                raise ValueError(f"identity manifest contains a cycle at {artifact_id}")
            visiting.add(artifact_id)
            artifact = by_id[artifact_id]
            for parent in artifact.parents:
                if parent.artifact_id in by_id:
                    visit(parent.artifact_id)
            visiting.remove(artifact_id)
            visited.add(artifact_id)

        for artifact_id in by_id:
            visit(artifact_id)

    def artifact(self, artifact_id: str) -> ArtifactIdentity:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise KeyError(artifact_id)

    def to_dict(self) -> dict[str, object]:
        artifacts = sorted(
            (artifact.to_dict() for artifact in self.artifacts),
            key=lambda item: str(item["artifact_id"]),
        )
        return {
            "schema_version": self.schema_version,
            "artifacts": artifacts,
        }

    @property
    def manifest_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())
