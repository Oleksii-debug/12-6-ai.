from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.post_base import (
    CANONICAL_BASE_EVIDENCE_NAMESPACE,
    CONTRACT_SCHEMA,
    POST_BASE_ARTIFACT_NAMESPACE,
    POST_BASE_EVIDENCE_NAMESPACE,
    CanonicalBasePolicy,
    DatasetProvenance,
    DialogueFormatLayer,
    EvaluationSeparation,
    PostBaseConsumptionContract,
    PostBaseStage,
    TokenizerCompatibility,
    prepare_post_base_workspace,
    snapshot_directory,
)
from twelve_six.posttraining.contracts import CheckpointRef, LineageKind


def _digest(char: str) -> str:
    return char * 64


def _base_checkpoint(*, lineage: LineageKind = LineageKind.BASE) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id="canonical-base-best",
        sha256=_digest("a"),
        git_sha="b" * 40,
        stage="base_pretraining",
        lineage=lineage,
    )


def _tokenizer(*, vocab_digest: str | None = None) -> TokenizerCompatibility:
    return TokenizerCompatibility(
        tokenizer_id="s0-byte-v1",
        config_sha256=_digest("c"),
        vocab_sha256=vocab_digest or _digest("d"),
        vocab_size=256,
    )


def _dataset() -> DatasetProvenance:
    return DatasetProvenance(
        dataset_id="communication-corpus-v1",
        manifest_sha256=_digest("e"),
        source_registry_sha256=_digest("f"),
        train_split_sha256=_digest("1"),
        evaluation_split_sha256=_digest("2"),
    )


def _contract(**overrides: object) -> PostBaseConsumptionContract:
    values: dict[str, object] = {
        "contract_id": "postbase253-contract-fixture",
        "base_checkpoint": _base_checkpoint(),
        "base_policy": CanonicalBasePolicy(),
        "tokenizer": _tokenizer(),
        "dataset": _dataset(),
        "stage": PostBaseStage.COMMUNICATION_SUPERVISION,
    }
    values.update(overrides)
    return PostBaseConsumptionContract(**values)  # type: ignore[arg-type]


def test_contract_is_versioned_separate_and_non_executable() -> None:
    contract = _contract()

    assert contract.schema == CONTRACT_SCHEMA
    assert contract.base_checkpoint.lineage is LineageKind.BASE
    assert contract.output_lineage is LineageKind.POSTTRAIN
    assert contract.output_namespace == POST_BASE_ARTIFACT_NAMESPACE
    assert contract.evaluation.canonical_base_namespace == CANONICAL_BASE_EVIDENCE_NAMESPACE
    assert contract.evaluation.post_base_namespace == POST_BASE_EVIDENCE_NAMESPACE
    assert contract.execution_authorized is False
    assert contract.rollback_checkpoint == contract.base_checkpoint
    assert not hasattr(contract, "train")

    with pytest.raises(ValueError, match="communication training is not authorized"):
        _contract(execution_authorized=True)


def test_canonical_base_policy_rejects_communication_contamination() -> None:
    for field_name in (
        "sft_applied",
        "rlhf_applied",
        "dpo_applied",
        "personality_applied",
        "chat_template_applied",
        "external_llm_inference_used_for_base",
    ):
        with pytest.raises(ValueError, match="canonical Base policy violation"):
            CanonicalBasePolicy(**{field_name: True})

    with pytest.raises(ValueError, match="random-init pretraining"):
        CanonicalBasePolicy(random_init_pretraining_origin=False)


def test_contract_rejects_non_base_input_or_base_output_identity() -> None:
    with pytest.raises(ValueError, match="canonical BASE lineage"):
        _contract(base_checkpoint=_base_checkpoint(lineage=LineageKind.POSTTRAIN))

    with pytest.raises(ValueError, match="output lineage"):
        _contract(output_lineage=LineageKind.BASE)

    with pytest.raises(ValueError, match="artifacts/post_base"):
        _contract(output_namespace="artifacts/base")


def test_tokenizer_compatibility_is_exact_and_dialogue_layer_cannot_mutate_it() -> None:
    contract = _contract()
    contract.require_tokenizer(_tokenizer())

    with pytest.raises(ValueError, match="exactly match"):
        contract.require_tokenizer(_tokenizer(vocab_digest=_digest("3")))

    layer = DialogueFormatLayer(
        format_id="dialogue-lines",
        format_version="1",
        formatter_module="twelve_six.post_base.formats.dialogue_lines",
    )
    assert _contract(dialogue_format=layer).dialogue_format == layer

    for field_name in (
        "mutates_tokenizer",
        "adds_special_tokens",
        "installs_base_chat_template",
    ):
        with pytest.raises(ValueError, match="external to canonical Base/tokenizer"):
            DialogueFormatLayer(
                format_id="invalid",
                format_version="1",
                formatter_module="fixture",
                **{field_name: True},
            )


def test_dataset_and_evaluation_authorities_remain_separate() -> None:
    with pytest.raises(ValueError, match="split identities must differ"):
        DatasetProvenance(
            dataset_id="bad",
            manifest_sha256=_digest("4"),
            source_registry_sha256=_digest("5"),
            train_split_sha256=_digest("6"),
            evaluation_split_sha256=_digest("6"),
        )

    with pytest.raises(ValueError, match="purpose"):
        DatasetProvenance(
            dataset_id="bad-purpose",
            manifest_sha256=_digest("4"),
            source_registry_sha256=_digest("5"),
            train_split_sha256=_digest("6"),
            evaluation_split_sha256=_digest("7"),
            purpose="base_pretraining",
        )

    with pytest.raises(ValueError, match="canonical Base evaluation namespace is immutable"):
        EvaluationSeparation(canonical_base_namespace="evidence/post_base")

    with pytest.raises(ValueError, match="evidence/post_base"):
        EvaluationSeparation(post_base_namespace="evidence/base")


def _write_checkpoint(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text('{"checkpoint":"base"}\n', encoding="utf-8")
    weights = root / "weights"
    weights.mkdir()
    (weights / "model.bin").write_bytes(b"canonical-base-weights")
    (weights / "optimizer.bin").write_bytes(b"canonical-base-optimizer")


def test_post_base_workspace_is_copy_on_write_and_base_is_rollback_safe(tmp_path: Path) -> None:
    base_root = tmp_path / "retained-canonical-base"
    experiment_root = tmp_path / "post-base-experiments" / "experiment-001"
    _write_checkpoint(base_root)

    before = snapshot_directory(base_root)
    prepared = prepare_post_base_workspace(base_root, experiment_root)
    after_prepare = snapshot_directory(base_root)

    assert prepared.canonical_base_root == base_root.resolve()
    assert prepared.experiment_root == experiment_root.resolve()
    assert prepared.cloned_checkpoint_root == (experiment_root / "input_checkpoint").resolve()
    assert prepared.canonical_snapshot_sha256 == before.identity_sha256
    assert prepared.cloned_snapshot_sha256 == before.identity_sha256
    assert prepared.shared_file_inodes is False
    assert after_prepare == before

    source_weight = base_root / "weights" / "model.bin"
    clone_weight = prepared.cloned_checkpoint_root / "weights" / "model.bin"
    assert source_weight.stat().st_ino != clone_weight.stat().st_ino

    clone_weight.write_bytes(b"future-post-base-experiment-mutation")
    assert source_weight.read_bytes() == b"canonical-base-weights"
    assert snapshot_directory(base_root) == before
    assert snapshot_directory(prepared.cloned_checkpoint_root).identity_sha256 != before.identity_sha256


def test_post_base_workspace_rejects_in_place_or_overlapping_destinations(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    _write_checkpoint(base_root)

    with pytest.raises(ValueError, match="disjoint"):
        prepare_post_base_workspace(base_root, base_root)

    with pytest.raises(ValueError, match="disjoint"):
        prepare_post_base_workspace(base_root, base_root / "post-base-run")

    parent_destination = tmp_path
    with pytest.raises(ValueError, match="disjoint"):
        prepare_post_base_workspace(base_root, parent_destination)


def test_existing_post_base_workspace_is_never_overwritten(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    experiment_root = tmp_path / "post-base-run"
    _write_checkpoint(base_root)
    experiment_root.mkdir()
    marker = experiment_root / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must not already exist"):
        prepare_post_base_workspace(base_root, experiment_root)

    assert marker.read_text(encoding="utf-8") == "preserve"
