"""Stable, dependency-light schemas for future 12-6 post-training work."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PostTrainingMethod(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    REWARD_MODEL = "reward_model"
    GRPO = "grpo"
    PPO = "ppo"
    PROCESS_SUPERVISION = "process_supervision"
    TEST_TIME_REASONING = "test_time_reasoning"


class FrameworkKind(str, Enum):
    CONTRACT_ONLY = "contract_only"
    TRL = "trl"
    VERL = "verl"
    CUSTOM_ADAPTER = "custom_adapter"


class ExecutionMode(str, Enum):
    CONTRACT_ONLY = "contract_only"
    DRY_RUN = "dry_run"
    TRAIN = "train"


class SourceKind(str, Enum):
    HUMAN = "human"
    SYNTHETIC = "synthetic"
    IMPORTED = "imported"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize schema data deterministically for manifests and fingerprints."""
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Content-addressed reference supplied by checkpoint/artifact owners."""

    artifact_id: str
    lineage: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        if not self.lineage.strip():
            raise ValueError("lineage must be non-empty")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must contain exactly 64 lowercase hex characters")


@dataclass(frozen=True, slots=True)
class DataProvenance:
    source_kind: SourceKind
    source_ids: tuple[str, ...]
    manifest_sha256: str
    license_or_terms_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.source_ids or any(not item.strip() for item in self.source_ids):
            raise ValueError("source_ids must contain at least one non-empty id")
        if not _SHA256_RE.fullmatch(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SyntheticProvenance:
    """Required provenance envelope for generated candidates or examples."""

    generator: ArtifactRef
    generation_config_sha256: str
    prompt_template_sha256: str
    seed: int
    external_generator: bool = False
    owner_policy_ref: str | None = None

    def __post_init__(self) -> None:
        for name, digest in (
            ("generation_config_sha256", self.generation_config_sha256),
            ("prompt_template_sha256", self.prompt_template_sha256),
        ):
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if self.external_generator and not (self.owner_policy_ref and self.owner_policy_ref.strip()):
            raise ValueError(
                "external synthetic generators require an explicit owner_policy_ref"
            )


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("role must be non-empty")
        if not self.content:
            raise ValueError("content must be non-empty")


@dataclass(frozen=True, slots=True)
class SFTRecord:
    record_id: str
    messages: tuple[Message, ...]
    provenance: DataProvenance | SyntheticProvenance

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")
        if not self.messages:
            raise ValueError("SFTRecord requires at least one message")


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    record_id: str
    prompt: str
    chosen: str
    rejected: str
    provenance: DataProvenance | SyntheticProvenance

    def __post_init__(self) -> None:
        if not self.record_id.strip() or not self.prompt or not self.chosen or not self.rejected:
            raise ValueError("preference fields must be non-empty")
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected candidates must differ")


@dataclass(frozen=True, slots=True)
class VerifierRecord:
    record_id: str
    prompt: str
    candidate: str
    provenance: DataProvenance | SyntheticProvenance
    reference_answer: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id.strip() or not self.prompt or not self.candidate:
            raise ValueError("verifier fields must be non-empty")
