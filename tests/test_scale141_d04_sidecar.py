from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from twelve_six.checkpoint import (
    CheckpointIdentity,
    D04_RESUME_BINDING_SCHEMA,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.scale141_recovery import (
    RecoveryLifecycleError,
    RecoveryPointerUpdateInterrupted,
    cleanup_recovery_generations,
    publish_recovery_generation,
    resolve_recovery_generation,
)
from twelve_six.scale141_resume_sidecar import ResumeSidecarContext
from twelve_six.training import Trainer, TrainerConfig

SOURCE_SHA = "2" * 40
RUN_HASH = "3" * 64
ENV_HASH = "4" * 64
TOKENIZER_HASH = "5" * 64
VOCAB_HASH = "6" * 64
DATA_HASH = "7" * 64
LEDGER_HASH = "8" * 64
MATERIALIZATION_HASH = "9" * 64
PACKING_IDENTITY_HASH = "a" * 64
EXPOSURE_PLAN_HASH = "b" * 64
ORDERED_NEXT_HASH = "c" * 64
SEGMENT_HASH = "d" * 64


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
        model_spec={"kind": "scale141-d04-sidecar-probe", "vocab": 16, "width": 8},
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_hash=VOCAB_HASH,
        dataset_manifest_hash=DATA_HASH,
        run_manifest_hash=RUN_HASH,
        training_config={
            "trainer": asdict(cfg),
            "data": {
                "resume_binding_schema": D04_RESUME_BINDING_SCHEMA,
                "ledger_identity_sha256": LEDGER_HASH,
                "materialization_identity_sha256": MATERIALIZATION_HASH,
                "packing_identity_sha256": PACKING_IDENTITY_HASH,
                "exposure_plan_identity_sha256": EXPOSURE_PLAN_HASH,
                "ordered_next_exposure_identity_sha256": ORDERED_NEXT_HASH,
            },
        },
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


def _state_hash(value: dict[str, object]) -> str:
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_state(context: ResumeSidecarContext) -> dict[str, object]:
    consumed = context.tokens_seen
    value: dict[str, object] = {
        "schema_version": "12-6.unique-loss-exposure-state.v2",
        "ledger_identity_sha256": LEDGER_HASH,
        "materialization_identity_sha256": MATERIALIZATION_HASH,
        "packing_identity_sha256": PACKING_IDENTITY_HASH,
        "authorized_budget": 1_000_000,
        "one_pass_maximum": 1_000_000,
        "consumed_loss_positions": consumed,
        "claim_sequence": context.optimizer_step,
        "claims": ({SEGMENT_HASH: [[0, consumed]]} if consumed else {}),
        "trainer_state_binding": {
            "checkpoint_generation": context.generation,
            "checkpoint_manifest_sha256": context.checkpoint_manifest_sha256,
            "optimizer_step": context.optimizer_step,
            "trainer_nonignored_target_count": consumed,
        },
    }
    value["state_identity_sha256"] = _state_hash(value)
    return value


def _publish(
    root: Path,
    model: TinyLM,
    trainer: Trainer,
    cfg: TrainerConfig,
    *,
    builder=_resume_state,
    failpoint: str | None = None,
):
    return publish_recovery_generation(
        root,
        save_generation=lambda path: _save(path, model, trainer, cfg),
        expected_source_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
        build_resume_state=builder,
        failpoint=failpoint,
    )


def _step(trainer: Trainer, offset: int = 0) -> None:
    values = torch.tensor([[1 + offset, 2 + offset, 3 + offset, 4 + offset]]) % 16
    trainer.train_microbatch({"input_ids": values})


def test_sidecar_publishes_after_manifest_and_resolves_exact_d04_state(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / "recovery"

    reference = _publish(root, model, trainer, cfg)
    resolution = resolve_recovery_generation(root, expected_reference=reference)

    assert resolution.resume_state is not None
    binding = resolution.resume_state["trainer_state_binding"]
    sidecar_reference = reference["resume_state"]
    assert binding["checkpoint_generation"] == "generation-00000001"
    assert binding["checkpoint_manifest_sha256"] == sidecar_reference[
        "checkpoint_manifest_sha256"
    ]
    assert binding["optimizer_step"] == trainer.optimizer_step
    assert resolution.resume_state["state_identity_sha256"] == sidecar_reference[
        "state_identity_sha256"
    ]
    assert sidecar_reference["ordered_next_exposure_identity_sha256"] == ORDERED_NEXT_HASH

    fresh_model, fresh_trainer, _ = _stack()
    load_trainer_checkpoint(
        resolution.path,
        model=fresh_model,
        trainer=fresh_trainer,
        restore_rng=False,
        expected_git_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
        expected_ledger_identity_sha256=LEDGER_HASH,
        expected_materialization_identity_sha256=MATERIALIZATION_HASH,
        expected_packing_identity_sha256=PACKING_IDENTITY_HASH,
        expected_exposure_plan_identity_sha256=EXPOSURE_PLAN_HASH,
        expected_ordered_next_exposure_identity_sha256=ORDERED_NEXT_HASH,
    )
    assert fresh_trainer.optimizer_step == trainer.optimizer_step
    assert fresh_trainer.tokens_seen == trainer.tokens_seen


@pytest.mark.parametrize("mutation", ["missing", "truncated"])
def test_missing_or_truncated_current_sidecar_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / mutation
    _publish(root, model, trainer, cfg)
    state_path = root / "resume-states/generation-00000001/state.json"

    if mutation == "missing":
        state_path.unlink()
    else:
        state_path.write_text("{", encoding="utf-8")

    with pytest.raises(RecoveryLifecycleError, match="resume sidecar"):
        resolve_recovery_generation(root)


def test_swapped_sidecar_from_other_generation_fails_closed(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "swapped"
    _step(trainer)
    _publish(root, model, trainer, cfg)
    first_bytes = (root / "resume-states/generation-00000001/state.json").read_bytes()

    _step(trainer, 1)
    _publish(root, model, trainer, cfg)
    second_path = root / "resume-states/generation-00000002/state.json"
    second_path.write_bytes(first_bytes)

    with pytest.raises(RecoveryLifecycleError, match="resume sidecar"):
        resolve_recovery_generation(root)


def test_wrong_checkpoint_binding_never_advances_pointer(tmp_path: Path) -> None:
    model, trainer, cfg = _stack()
    _step(trainer)
    root = tmp_path / "wrong-binding"

    def wrong_binding(context: ResumeSidecarContext) -> dict[str, object]:
        state = _resume_state(context)
        state["trainer_state_binding"]["checkpoint_manifest_sha256"] = "0" * 64
        state_without_hash = dict(state)
        state_without_hash.pop("state_identity_sha256")
        state["state_identity_sha256"] = _state_hash(state_without_hash)
        return state

    with pytest.raises(RecoveryLifecycleError, match="sidecar publication failed"):
        _publish(root, model, trainer, cfg, builder=wrong_binding)

    assert not (root / "current.json").exists()
    assert (root / "generations/generation-00000001").is_dir()
    assert not (root / "resume-states/generation-00000001").exists()


def test_interruption_after_sidecar_keeps_last_known_good_and_cleanup_removes_orphan_pair(
    tmp_path: Path,
) -> None:
    model, trainer, cfg = _stack()
    root = tmp_path / "interrupted"
    _step(trainer)
    first = _publish(root, model, trainer, cfg)

    _step(trainer, 1)
    with pytest.raises(RecoveryPointerUpdateInterrupted, match="after D04 sidecar"):
        _publish(
            root,
            model,
            trainer,
            cfg,
            failpoint="after_sidecar_before_pointer",
        )

    current = resolve_recovery_generation(root, expected_reference=first)
    assert current.reference["generation"] == 1
    assert (root / "generations/generation-00000002").is_dir()
    assert (root / "resume-states/generation-00000002").is_dir()

    cleaned = cleanup_recovery_generations(root, keep=1)
    assert "generation-00000002" in cleaned["removed"]
    assert not (root / "generations/generation-00000002").exists()
    assert not (root / "resume-states/generation-00000002").exists()
    assert resolve_recovery_generation(root, expected_reference=first).resume_state is not None
