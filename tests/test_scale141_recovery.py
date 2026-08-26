from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from twelve_six.checkpoint import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.scale141_recovery import (
    RecoveryLifecycleError,
    RecoveryPointerUpdateInterrupted,
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
    cfg = TrainerConfig(max_steps=8, learning_rate=1e-3, seed=211)
    torch.manual_seed(cfg.seed)
    model = TinyLM()
    trainer = Trainer(model, cfg, device="cpu")
    return model, trainer, cfg


def _identity(model: TinyLM, trainer: Trainer, cfg: TrainerConfig) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=SOURCE_SHA,
        model_spec={"kind": "checkpoint211-tiny-lm", "vocab": 16, "width": 8},
        parameter_count=sum(p.numel() for p in model.parameters()),
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_hash=VOCAB_HASH,
        dataset_manifest_hash=DATA_HASH,
        run_manifest_hash=RUN_HASH,
        training_config={"trainer": asdict(cfg), "proof": "checkpoint211"},
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


def _save(
    path: Path,
    model: TinyLM,
    trainer: Trainer,
    cfg: TrainerConfig,
    *,
    overwrite: bool = False,
):
    return save_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        identity=_identity(model, trainer, cfg),
        overwrite=overwrite,
    )


def _publish(
    root: Path,
    model: TinyLM,
    trainer: Trainer,
    cfg: TrainerConfig,
    *,
    failpoint: str | None = None,
):
    return publish_recovery_generation(
        root,
        save_generation=lambda path: _save(path, model, trainer, cfg),
        expected_source_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
        failpoint=failpoint,
    )


def _step(trainer: Trainer, offset: int = 0) -> None:
    values = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 16
    trainer.train_microbatch({"input_ids": values})


def _file_hashes(path: Path) -> dict[str, str]:
    return {
        child.name: sha256_file(child)
        for child in path.iterdir()
        if child.is_file()
    }


def test_historical_recovery_latest_overwrite_failure_is_reproduced(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    legacy = tmp_path / "recovery-latest"
    _save(legacy, model, trainer, cfg, overwrite=True)
    with pytest.raises(FileExistsError, match="checkpoint-v1 is immutable"):
        _save(legacy, model, trainer, cfg, overwrite=True)
    assert verify_checkpoint(legacy)["identity"]["step"] == 1


def test_multiple_immutable_generations_advance_only_pointer_and_fresh_load_exact_state(
    tmp_path: Path,
) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "recovery"

    _step(trainer)
    first = _publish(root, model, trainer, cfg)
    first_path = root / "generations" / "generation-00000001"
    first_hashes = _file_hashes(first_path)

    _step(trainer, 1)
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    second = _publish(root, model, trainer, cfg)
    second_path = root / "generations" / "generation-00000002"

    assert first["generation"] == 1
    assert second["generation"] == 2
    assert _file_hashes(first_path) == first_hashes
    assert verify_checkpoint(first_path)["checkpoint_id"] == first["checkpoint_id"]
    assert verify_checkpoint(second_path)["checkpoint_id"] == second["checkpoint_id"]

    resolution = resolve_recovery_generation(
        root,
        expected_reference=second,
        expected_source_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=2,
        expected_tokens_seen=6,
    )
    assert resolution.path == second_path

    rng_expected = torch.rand(4)
    torch.manual_seed(999)
    fresh_model, fresh_trainer, _ = _stack()
    torch.manual_seed(999)
    loaded = load_trainer_checkpoint(
        resolution.path,
        model=fresh_model,
        trainer=fresh_trainer,
        restore_rng=True,
        expected_git_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_tokenizer_hash=TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=VOCAB_HASH,
        expected_dataset_manifest_hash=DATA_HASH,
        expected_environment_lock_hash=ENV_HASH,
        expected_seed=cfg.seed,
    )
    assert loaded.manifest["checkpoint_id"] == second["checkpoint_id"]
    assert fresh_trainer.optimizer_step == 2
    assert fresh_trainer.tokens_seen == 6
    assert fresh_trainer.optimizer.state_dict()["state"]
    assert (
        fresh_trainer.optimizer.state_dict()["state"].keys()
        == expected_optimizer["state"].keys()
    )
    assert torch.equal(torch.rand(4), rng_expected)


def test_interrupted_pointer_update_keeps_last_known_good_and_unreferenced_newer_is_ignored(
    tmp_path: Path,
) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "recovery"
    _step(trainer)
    first = _publish(root, model, trainer, cfg)

    _step(trainer, 1)
    with pytest.raises(RecoveryPointerUpdateInterrupted):
        _publish(
            root,
            model,
            trainer,
            cfg,
            failpoint="before_pointer_replace",
        )

    current = resolve_recovery_generation(root, expected_reference=first)
    assert current.reference["generation"] == 1
    orphan = root / "generations" / "generation-00000002"
    assert verify_checkpoint(orphan)["identity"]["step"] == 2

    corrupt_newer = root / "generations" / "generation-00000003"
    corrupt_newer.mkdir()
    (corrupt_newer / "incomplete").write_text("not committed\n", encoding="utf-8")
    still_current = resolve_recovery_generation(root, expected_reference=first)
    assert still_current.reference["generation"] == 1


def test_corrupt_current_target_fails_closed_under_d05_authority(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "recovery"
    _step(trainer)
    _publish(root, model, trainer, cfg)
    current = root / "generations" / "generation-00000001"
    with (current / "weights.safetensors").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(CheckpointIntegrityError):
        resolve_recovery_generation(root)


def test_cleanup_always_preserves_verified_current_generation(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "recovery"
    refs = []
    for index in range(3):
        _step(trainer, index)
        refs.append(_publish(root, model, trainer, cfg))

    result = cleanup_recovery_generations(root, keep=1)
    assert result["current_generation"] == 3
    assert result["retained_generation_count"] == 1
    assert not (root / "generations" / "generation-00000001").exists()
    assert not (root / "generations" / "generation-00000002").exists()
    resolution = resolve_recovery_generation(root, expected_reference=refs[-1])
    assert resolution.reference["checkpoint_id"] == refs[-1]["checkpoint_id"]


def test_pointer_binding_mismatch_fails_closed(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "recovery"
    _step(trainer)
    reference = _publish(root, model, trainer, cfg)
    with pytest.raises(RecoveryLifecycleError, match="binding mismatch"):
        resolve_recovery_generation(root, expected_source_sha="a" * 40)
    bad_reference = dict(reference)
    bad_reference["optimizer_step"] += 1
    with pytest.raises(RecoveryLifecycleError, match="phase boundary reference"):
        resolve_recovery_generation(root, expected_reference=bad_reference)
