from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import CheckpointIdentity, hash_json, save_checkpoint
from twelve_six.training.resilience import (
    FailureClass,
    RecoveryPolicy,
    RecoveryStateError,
    RecoveryStore,
    RetryBudgetExceeded,
    recommend_checkpoint_interval,
)


class NumpyModel:
    def __init__(self) -> None:
        self.weights = np.array([0.1, -0.2], dtype=np.float64)

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        assert not strict or set(state) == {"weights"}
        self.weights = state["weights"].copy()


class StubTrainer:
    def __init__(self, *, optimizer_step: int = 0, tokens_seen: int = 0) -> None:
        self.optimizer_step = optimizer_step
        self.tokens_seen = tokens_seen

    def assert_checkpoint_safe(self) -> None:
        return None


def _run_manifest() -> dict:
    return {
        "run_id": "train39-test-run",
        "candidate": {"git_sha": "a" * 40},
        "recovery": {
            "topology": {
                "backend": "single-process-test",
                "world_size": 1,
                "resume_policy": "exact_topology",
            }
        },
    }


def _identity(run_manifest: dict, *, step: int, tokens_seen: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "numpy-resilience-test", "width": 2},
        parameter_count=2,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash=hash_json(run_manifest),
        training_config={"run_id": run_manifest["run_id"], "kind": "test"},
        seed=7,
        precision="float64-test",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "none-test"},
        scheduler=None,
        environment_lock_hash="e" * 64,
    )


def _store(tmp_path: Path, *, max_restarts: int = 2) -> RecoveryStore:
    return RecoveryStore(
        tmp_path,
        run_manifest=_run_manifest(),
        policy=RecoveryPolicy(
            checkpoint_every_steps=2,
            retain_last=3,
            max_restarts=max_restarts,
            max_preemptions=2,
        ),
    )


def _commit(store: RecoveryStore, trainer: StubTrainer, model: NumpyModel):
    manifest = _run_manifest()

    def save(path: Path):
        return save_checkpoint(
            path,
            model=model,
            trainer_state={"optimizer_step": trainer.optimizer_step},
            identity=_identity(
                manifest,
                step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
            ),
        )

    return store.commit_checkpoint(trainer, save)


def test_corrupt_latest_checkpoint_falls_back_to_previous_verified(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_attempt()
    model = NumpyModel()
    trainer = StubTrainer(optimizer_step=2, tokens_seen=20)
    first = _commit(store, trainer, model)
    trainer.optimizer_step = 4
    trainer.tokens_seen = 40
    model.weights += 1.0
    second = _commit(store, trainer, model)
    assert store.last_known_good() == second

    weights_path = store.checkpoints_dir / second.directory / "weights.safetensors"
    payload = bytearray(weights_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    weights_path.write_bytes(payload)

    fallback = store.last_known_good()
    assert fallback == first
    state = store.open()
    assert second.directory in state["invalid_checkpoint_directories"]
    assert state["last_known_good"]["optimizer_step"] == 2


def test_restart_requires_supervisor_failure_transition_and_respects_budget(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, max_restarts=1)
    first = store.begin_attempt()
    assert first["attempt"] == 1
    with pytest.raises(RecoveryStateError, match="supervisor"):
        store.begin_attempt()

    store.record_failure(
        FailureClass.PROCESS_LOSS,
        optimizer_step=3,
        detail_code="injected-process-loss",
    )
    second = store.begin_attempt()
    assert second["attempt"] == 2
    assert second["restarts_used"] == 1

    store.record_failure(
        FailureClass.PROCESS_LOSS,
        optimizer_step=5,
        detail_code="injected-second-process-loss",
    )
    with pytest.raises(RetryBudgetExceeded, match="restart budget exhausted"):
        store.begin_attempt()
    assert store.open()["phase"] == "FAILED"


def test_corrupt_recovery_journal_reconstructs_attempt_and_last_good(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_attempt()
    trainer = StubTrainer(optimizer_step=2, tokens_seen=20)
    record = _commit(store, trainer, NumpyModel())

    store.state_path.write_text("{not-json", encoding="utf-8")
    rebuilt = store.open()
    assert rebuilt["journal_reconstructed"] is True
    assert rebuilt["attempt"] == 1
    assert rebuilt["restarts_used"] == 0
    assert rebuilt["last_known_good"]["checkpoint_id"] == record.checkpoint_id
    assert rebuilt["phase"] == "RECOVERING"


def test_checkpoint_interval_uses_measured_overhead_and_recovery_bounds() -> None:
    result = recommend_checkpoint_interval(
        optimizer_step_seconds=2.0,
        checkpoint_seconds=6.0,
        target_overhead_fraction=0.05,
        max_recovery_window_seconds=300.0,
    )
    assert result.minimum_interval_steps_for_overhead == 29
    assert result.maximum_interval_steps_for_recovery == 150
    assert result.recommended_interval_steps == 29
    assert result.predicted_checkpoint_overhead_fraction <= 0.05
    assert result.predicted_max_recompute_seconds == 58.0
    assert result.constraints_satisfied is True
