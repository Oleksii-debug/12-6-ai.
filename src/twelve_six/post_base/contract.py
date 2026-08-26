"""Versioned boundary for future post-Base communication experiments.

This module deliberately contains no trainer, optimizer, gradient, model-update, chat
runtime, or external-model integration. It only describes how an immutable canonical
Base checkpoint may be referenced and copied into a separate post-Base workspace.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from twelve_six.posttraining.contracts import CheckpointRef, LineageKind

CONTRACT_SCHEMA = "12-6.post-base.communication-consumption.v1"
CANONICAL_BASE_EVIDENCE_NAMESPACE = "evidence/base"
POST_BASE_EVIDENCE_NAMESPACE = "evidence/post_base"
POST_BASE_ARTIFACT_NAMESPACE = "artifacts/post_base"
_SHA256_LENGTH = 64


def _require_text(value: str, *, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _require_sha256(value: str, *, field: str) -> None:
    if len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256 digest")


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


class PostBaseStage(StrEnum):
    """Future communication-training stage identities.

    The enum identifies a future stage; it does not authorize execution.
    """

    COMMUNICATION_SUPERVISION = "post_base.communication.supervision"
    PREFERENCE_OPTIMIZATION = "post_base.communication.preference_optimization"
    REINFORCEMENT_OPTIMIZATION = "post_base.communication.reinforcement_optimization"


@dataclass(frozen=True, slots=True)
class CanonicalBasePolicy:
    """Assertions that must remain true for the retained canonical Base artifact."""

    random_init_pretraining_origin: bool = True
    sft_applied: bool = False
    rlhf_applied: bool = False
    dpo_applied: bool = False
    personality_applied: bool = False
    chat_template_applied: bool = False
    external_llm_inference_used_for_base: bool = False

    def __post_init__(self) -> None:
        if not self.random_init_pretraining_origin:
            raise ValueError("canonical Base must originate from random-init pretraining")
        forbidden = {
            "sft_applied": self.sft_applied,
            "rlhf_applied": self.rlhf_applied,
            "dpo_applied": self.dpo_applied,
            "personality_applied": self.personality_applied,
            "chat_template_applied": self.chat_template_applied,
            "external_llm_inference_used_for_base": self.external_llm_inference_used_for_base,
        }
        active = sorted(name for name, enabled in forbidden.items() if enabled)
        if active:
            raise ValueError(f"canonical Base policy violation: {', '.join(active)}")


@dataclass(frozen=True, slots=True)
class TokenizerCompatibility:
    tokenizer_id: str
    config_sha256: str
    vocab_sha256: str
    vocab_size: int

    def __post_init__(self) -> None:
        _require_text(self.tokenizer_id, field="tokenizer_id")
        _require_sha256(self.config_sha256, field="config_sha256")
        _require_sha256(self.vocab_sha256, field="vocab_sha256")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

    def require_exact_match(self, other: TokenizerCompatibility) -> None:
        if self != other:
            raise ValueError("post-Base tokenizer must exactly match the canonical Base tokenizer")


@dataclass(frozen=True, slots=True)
class DialogueFormatLayer:
    """Optional post-Base-only text-formatting boundary.

    A v1 dialogue layer may format examples outside the Base artifact, but it may not
    mutate tokenizer IDs, add special tokens, or install a chat template into Base.
    """

    format_id: str
    format_version: str
    formatter_module: str
    mutates_tokenizer: bool = False
    adds_special_tokens: bool = False
    installs_base_chat_template: bool = False

    def __post_init__(self) -> None:
        _require_text(self.format_id, field="format_id")
        _require_text(self.format_version, field="format_version")
        _require_text(self.formatter_module, field="formatter_module")
        if self.mutates_tokenizer or self.adds_special_tokens or self.installs_base_chat_template:
            raise ValueError(
                "dialogue formatting must remain external to canonical Base/tokenizer in v1"
            )


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    dataset_id: str
    manifest_sha256: str
    source_registry_sha256: str
    train_split_sha256: str
    evaluation_split_sha256: str
    purpose: str = "post_base_communication"

    def __post_init__(self) -> None:
        _require_text(self.dataset_id, field="dataset_id")
        for field_name in (
            "manifest_sha256",
            "source_registry_sha256",
            "train_split_sha256",
            "evaluation_split_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        if self.purpose != "post_base_communication":
            raise ValueError("dataset purpose must be post_base_communication")
        if self.train_split_sha256 == self.evaluation_split_sha256:
            raise ValueError("post-Base training and evaluation split identities must differ")


@dataclass(frozen=True, slots=True)
class EvaluationSeparation:
    canonical_base_namespace: str = CANONICAL_BASE_EVIDENCE_NAMESPACE
    post_base_namespace: str = POST_BASE_EVIDENCE_NAMESPACE

    def __post_init__(self) -> None:
        if self.canonical_base_namespace != CANONICAL_BASE_EVIDENCE_NAMESPACE:
            raise ValueError("canonical Base evaluation namespace is immutable")
        if self.post_base_namespace != POST_BASE_EVIDENCE_NAMESPACE:
            raise ValueError("post-Base evaluation must use evidence/post_base")
        if self.canonical_base_namespace == self.post_base_namespace:
            raise ValueError("Base and post-Base evaluation namespaces must be distinct")


@dataclass(frozen=True, slots=True)
class PostBaseConsumptionContract:
    """Immutable handoff from canonical Base into a future communication stage."""

    contract_id: str
    base_checkpoint: CheckpointRef
    base_policy: CanonicalBasePolicy
    tokenizer: TokenizerCompatibility
    dataset: DatasetProvenance
    stage: PostBaseStage
    evaluation: EvaluationSeparation = EvaluationSeparation()
    dialogue_format: DialogueFormatLayer | None = None
    schema: str = CONTRACT_SCHEMA
    output_lineage: LineageKind = LineageKind.POSTTRAIN
    output_namespace: str = POST_BASE_ARTIFACT_NAMESPACE
    execution_authorized: bool = False

    def __post_init__(self) -> None:
        _require_text(self.contract_id, field="contract_id")
        if self.schema != CONTRACT_SCHEMA:
            raise ValueError("unsupported post-Base contract schema")
        if self.base_checkpoint.lineage is not LineageKind.BASE:
            raise ValueError("post-Base input checkpoint must be canonical BASE lineage")
        if self.output_lineage is not LineageKind.POSTTRAIN:
            raise ValueError("post-Base output lineage must remain POSTTRAIN")
        if self.output_namespace != POST_BASE_ARTIFACT_NAMESPACE:
            raise ValueError("post-Base outputs must stay under artifacts/post_base")
        if self.execution_authorized:
            raise ValueError("POSTBASE-253 v1 is contract-only; communication training is not authorized")

    @property
    def rollback_checkpoint(self) -> CheckpointRef:
        """Return the exact untouched Base identity used as the rollback target."""
        return self.base_checkpoint

    def require_tokenizer(self, candidate: TokenizerCompatibility) -> None:
        self.tokenizer.require_exact_match(candidate)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DirectorySnapshot:
    root: Path
    files: tuple[FileIdentity, ...]
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedPostBaseWorkspace:
    canonical_base_root: Path
    experiment_root: Path
    cloned_checkpoint_root: Path
    canonical_snapshot_sha256: str
    cloned_snapshot_sha256: str
    shared_file_inodes: bool


def snapshot_directory(root: Path) -> DirectorySnapshot:
    """Create a deterministic identity for a regular-file-only checkpoint directory."""
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("checkpoint root must be a real directory")
    identities: list[FileIdentity] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("checkpoint snapshot forbids symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("checkpoint snapshot accepts regular files only")
        payload = path.read_bytes()
        identities.append(
            FileIdentity(
                relative_path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    if not identities:
        raise ValueError("checkpoint root must contain at least one file")
    canonical = json.dumps(
        [
            {
                "path": item.relative_path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in identities
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return DirectorySnapshot(
        root=root,
        files=tuple(identities),
        identity_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def prepare_post_base_workspace(
    canonical_base_root: Path,
    experiment_root: Path,
) -> PreparedPostBaseWorkspace:
    """Copy canonical Base into a disjoint workspace and prove no in-place mutation.

    The copy uses independent files rather than hard links. The source is hashed before
    and after publication. Any overlap, symlink, shared inode, or source-byte change
    fails closed.
    """
    source = canonical_base_root.resolve()
    destination_root = experiment_root.resolve()
    if _is_within(destination_root, source) or _is_within(source, destination_root):
        raise ValueError("post-Base workspace must be disjoint from canonical Base storage")
    if destination_root.exists():
        raise FileExistsError("post-Base experiment root must not already exist")

    before = snapshot_directory(source)
    cloned = destination_root / "input_checkpoint"
    shutil.copytree(source, cloned, copy_function=shutil.copy2, symlinks=False)

    after = snapshot_directory(source)
    clone_snapshot = snapshot_directory(cloned)
    if before.identity_sha256 != after.identity_sha256 or before.files != after.files:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise RuntimeError("canonical Base changed while preparing post-Base workspace")
    if clone_snapshot.files != before.files:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise RuntimeError("post-Base checkpoint clone is not byte-identical to canonical Base")

    shared_inode = False
    for identity in before.files:
        source_file = source / identity.relative_path
        clone_file = cloned / identity.relative_path
        source_stat = source_file.stat()
        clone_stat = clone_file.stat()
        if (source_stat.st_dev, source_stat.st_ino) == (clone_stat.st_dev, clone_stat.st_ino):
            shared_inode = True
            break
    if shared_inode:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise RuntimeError("post-Base clone must not share file inodes with canonical Base")

    return PreparedPostBaseWorkspace(
        canonical_base_root=source,
        experiment_root=destination_root,
        cloned_checkpoint_root=cloned,
        canonical_snapshot_sha256=before.identity_sha256,
        cloned_snapshot_sha256=clone_snapshot.identity_sha256,
        shared_file_inodes=False,
    )
