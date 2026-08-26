from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

import twelve_six.checkpoint.core as checkpoint_core
import twelve_six.checkpoint.trainer_adapter as trainer_adapter
from twelve_six.checkpoint import CheckpointIdentity


class _NumpyModel:
    def __init__(self, values: list[float]) -> None:
        self.weight = np.asarray(values, dtype=np.float32)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weight": self.weight.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        if strict and set(state) != {"weight"}:
            raise ValueError("unexpected model state")
        self.weight = np.asarray(state["weight"], dtype=np.float32).copy()


class _GenericTrainer:
    def __init__(self, velocity: list[float]) -> None:
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 4}
        self.velocity = list(velocity)
        self.loads = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "micro_step": 1,
            "optimizer_step": 1,
            "tokens_seen": 2,
            "optimizer": {"velocity": list(self.velocity)},
            "scheduler": None,
            "scaler": None,
            "config": copy.deepcopy(self.config),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state["config"] != self.config:
            raise ValueError("trainer config mismatch")
        optimizer = state["optimizer"]
        if not isinstance(optimizer, dict):
            raise ValueError("optimizer state must be a dict")
        velocity = optimizer.get("velocity")
        if not isinstance(velocity, list) or len(velocity) != 2:
            raise ValueError("velocity geometry mismatch")
        self.loads += 1
        self.velocity = [float(value) for value in velocity]


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "single-decode-probe", "parameters": 2},
        parameter_count=2,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"probe": "single-decode"},
        seed=19,
        precision="float32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "generic-probe"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def test_trainer_resume_decodes_verified_snapshot_exactly_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_model = _NumpyModel([1.25, -2.5])
    source_trainer = _GenericTrainer([3.0, 4.0])
    checkpoint = tmp_path / "single-decode"
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(),
    )

    target_model = _NumpyModel([9.0, 10.0])
    target_trainer = _GenericTrainer([7.0, 8.0])

    original_decode = checkpoint_core._decode_verified_state
    decode_calls = 0

    def counted_decode(verified):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(verified)

    # The pre-fix adapter decoded through its imported symbol and then delegated
    # to core.load_verified_checkpoint(), which decoded the same snapshot again.
    # Patch both references so this regression detects either decode path.
    monkeypatch.setattr(checkpoint_core, "_decode_verified_state", counted_decode)
    monkeypatch.setattr(trainer_adapter, "_decode_verified_state", counted_decode)

    result = trainer_adapter.load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
    )

    assert decode_calls == 1
    np.testing.assert_array_equal(target_model.weight, source_model.weight)
    assert target_trainer.velocity == [3.0, 4.0]
    assert target_trainer.loads == 1
    assert result.manifest["identity"]["step"] == 1
    assert result.manifest["identity"]["tokens_seen"] == 2
