from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from twelve_six.checkpoint import CheckpointIdentity, hash_json, save_trainer_checkpoint, sha256_file
from twelve_six.scale141_recovery import (
    RecoveryLifecycleError,
    cleanup_recovery_generations,
    publish_recovery_generation,
    resolve_recovery_generation,
)
from twelve_six.training import Trainer, TrainerConfig

SOURCE_SHA = "2" * 40
RUN_HASH = "3" * 64
ENV_HASH = "4" * 64
TOKENIZER_HASH = "5" * 64
VOCAB_HASH = "6" * 64
DATA_HASH = "7" * 64


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def _stack() -> tuple[TinyLM, Trainer, TrainerConfig]:
    cfg = TrainerConfig(max_steps=8, learning_rate=1e-3, seed=919)
    torch.manual_seed(cfg.seed)
    model = TinyLM()
    return model, Trainer(model, cfg, device="cpu"), cfg


def _identity(model: TinyLM, trainer: Trainer, cfg: TrainerConfig) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=SOURCE_SHA,
        model_spec={"kind": "content-addressed-recovery-probe", "vocab": 16, "width": 8},
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_hash=VOCAB_HASH,
        dataset_manifest_hash=DATA_HASH,
        run_manifest_hash=RUN_HASH,
        training_config={"trainer": asdict(cfg), "proof": "content-addressed-recovery"},
        seed=cfg.seed,
        precision=cfg.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": cfg.learning_rate,
            "betas": list(cfg.betas),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
        },
        scheduler=None,
        environment_lock_hash=ENV_HASH,
    )


def _save(path: Path, model: TinyLM, trainer: Trainer, cfg: TrainerConfig):
    return save_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        identity=_identity(model, trainer, cfg),
    )


def _publish(root: Path, model: TinyLM, trainer: Trainer, cfg: TrainerConfig, *, seen=None):
    def save(path: Path):
        if seen is not None:
            seen.append(path)
        return _save(path, model, trainer, cfg)

    return publish_recovery_generation(
        root,
        save_generation=save,
        expected_source_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
    )


def _step(trainer: Trainer, offset: int = 0) -> None:
    values = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 16
    trainer.train_microbatch({"input_ids": values})


def test_publication_stages_privately_then_returns_verified_content_derived_receipt(
    tmp_path: Path,
) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / "recovery"
    seen: list[Path] = []

    reference = _publish(root, model, trainer, cfg, seen=seen)
    checkpoint_id = reference["checkpoint_id"]
    expected_key = f"checkpoints/{checkpoint_id}"
    content_path = root / expected_key

    assert len(seen) == 1
    assert seen[0].name == "checkpoint"
    assert seen[0].parent.name.startswith(".checkpoint-stage-")
    assert seen[0] != root / "generations/generation-00000001"
    assert reference["object_key"] == expected_key
    assert reference["manifest_sha256"] == sha256_file(content_path / "manifest.json")

    resolved = resolve_recovery_generation(root, expected_reference=reference)
    assert resolved.content_path == content_path
    assert resolved.manifest["checkpoint_id"] == checkpoint_id
    assert resolved.path == root / "generations/generation-00000001"


def test_resealed_pointer_cannot_substitute_object_key_for_same_checkpoint_id(
    tmp_path: Path,
) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / "recovery"
    _publish(root, model, trainer, cfg)

    pointer_path = root / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["object_key"] = "checkpoints/" + "0" * 64
    pointer.pop("pointer_sha256")
    pointer["pointer_sha256"] = hash_json(pointer)
    pointer_path.write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RecoveryLifecycleError, match="object_key/checkpoint_id mismatch"):
        resolve_recovery_generation(root)


def test_identical_content_reuses_same_derived_object_without_overwrite(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "recovery"
    _step(trainer)

    first = _publish(root, model, trainer, cfg)
    first_object = root / first["object_key"]
    first_manifest = (first_object / "manifest.json").read_bytes()
    first_checksum = (first_object / "MANIFEST.sha256").read_bytes()

    second = _publish(root, model, trainer, cfg)

    assert second["checkpoint_id"] == first["checkpoint_id"]
    assert second["object_key"] == first["object_key"]
    assert (first_object / "manifest.json").read_bytes() == first_manifest
    assert (first_object / "MANIFEST.sha256").read_bytes() == first_checksum
    assert [entry.name for entry in (root / "checkpoints").iterdir()] == [first["checkpoint_id"]]


def test_different_checkpoint_content_uses_different_object_key_and_cleanup_removes_orphan(
    tmp_path: Path,
) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "recovery"
    _step(trainer)
    first = _publish(root, model, trainer, cfg)

    _step(trainer, 1)
    second = _publish(root, model, trainer, cfg)

    assert second["checkpoint_id"] != first["checkpoint_id"]
    assert second["object_key"] != first["object_key"]
    assert (root / first["object_key"]).is_dir()
    assert (root / second["object_key"]).is_dir()

    cleaned = cleanup_recovery_generations(root, keep=1)
    assert first["checkpoint_id"] in cleaned["removed_content_objects"]
    assert not (root / first["object_key"]).exists()
    assert (root / second["object_key"]).is_dir()
