"""D05 negative contract: reject malformed caller expectations before checkpoint I/O.

Non-owning, deliberately red adapter regression against PR #2628 at ef7d035.
The progress loader already enforces this boundary.  A passing result requires
both public restore entrypoints to make the same early fail-closed decision.
No model weights, real dataset, training, or checkpoint artifact is needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    progress_trainer,
    trainer_adapter,
)


class _PassiveTrainer:
    def load_state_dict(self, _state: Any) -> None:
        raise AssertionError("invalid expected binding must reject before trainer load")


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "expected_seed",
    [True, 1.0, -1, "7"],
    ids=["bool-as-int", "float-as-int", "negative", "string"],
)
def test_invalid_expected_seed_rejected_before_checkpoint_io(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    expected_seed: Any,
) -> None:
    attempts: list[str] = []

    def unexpected_read(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("checkpoint-open")
        raise AssertionError("invalid expected_seed must reject before checkpoint read")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", unexpected_read)
    with pytest.raises(CheckpointCompatibilityError, match="expected_seed"):
        loader.load_trainer_checkpoint(
            tmp_path / "not-a-checkpoint",
            model=object(),
            trainer=_PassiveTrainer(),
            expected_seed=expected_seed,
        )
    assert attempts == []


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "field",
    [
        "expected_init_spec_hash",
        "expected_packing_hash",
        "expected_training_config_hash",
        "expected_environment_lock_hash",
    ],
)
def test_invalid_expected_hash_rejected_before_checkpoint_io(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    field: str,
) -> None:
    attempts: list[str] = []

    def unexpected_read(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("checkpoint-open")
        raise AssertionError("malformed expected SHA-256 must reject before checkpoint read")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", unexpected_read)
    with pytest.raises(CheckpointCompatibilityError, match=field):
        loader.load_trainer_checkpoint(
            tmp_path / "not-a-checkpoint",
            model=object(),
            trainer=_PassiveTrainer(),
            **{field: "not-a-sha256"},
        )
    assert attempts == []


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "field", ["expected_split_identity", "expected_packing_version"],
)
def test_empty_expected_identity_rejected_before_checkpoint_io(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    field: str,
) -> None:
    attempts: list[str] = []

    def unexpected_read(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("checkpoint-open")
        raise AssertionError("empty expected identity must reject before checkpoint read")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", unexpected_read)
    with pytest.raises(CheckpointCompatibilityError, match=field):
        loader.load_trainer_checkpoint(
            tmp_path / "not-a-checkpoint",
            model=object(),
            trainer=_PassiveTrainer(),
            **{field: ""},
        )
    assert attempts == []

@pytest.mark.parametrize(
    "field", ["expected_step", "expected_tokens_seen"],
)
@pytest.mark.parametrize(
    "invalid", [True, 1.0, -1, "1"],
    ids=["bool-as-int", "float-as-int", "negative", "string"],
)
def test_invalid_expected_progress_rejected_before_checkpoint_io(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: Any,
) -> None:
    attempts: list[str] = []

    def forbidden_read(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("checkpoint-open")
        raise AssertionError("invalid expected progress must reject before checkpoint I/O")

    monkeypatch.setattr(progress_trainer, "prepare_checkpoint_load", forbidden_read)
    with pytest.raises(CheckpointCompatibilityError, match=field):
        progress_trainer.load_trainer_checkpoint(
            tmp_path / "not-a-checkpoint",
            model=object(),
            trainer=_PassiveTrainer(),
            **{field: invalid},
        )
    assert attempts == []


@pytest.mark.parametrize(
    "field",
    [
        "expected_ledger_identity_sha256",
        "expected_materialization_identity_sha256",
        "expected_packing_identity_sha256",
        "expected_exposure_plan_identity_sha256",
        "expected_ordered_next_exposure_identity_sha256",
    ],
)
@pytest.mark.parametrize(
    "invalid", ["not-sha256", "A" * 64, True],
    ids=["malformed", "uppercase", "boolean"],
)
def test_invalid_expected_d04_sha_rejected_before_checkpoint_io(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: Any,
) -> None:
    attempts: list[str] = []

    def forbidden_read(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("checkpoint-open")
        raise AssertionError("invalid expected D04 hash must reject before checkpoint I/O")

    monkeypatch.setattr(progress_trainer, "prepare_checkpoint_load", forbidden_read)
    with pytest.raises(CheckpointCompatibilityError, match=field):
        progress_trainer.load_trainer_checkpoint(
            tmp_path / "not-a-checkpoint",
            model=object(),
            trainer=_PassiveTrainer(),
            **{field: invalid},
        )
    assert attempts == []


@pytest.mark.parametrize(
    "field,valid",
    [
        ("expected_step", 0),
        ("expected_tokens_seen", 0),
        ("expected_ledger_identity_sha256", "a" * 64),
        ("expected_materialization_identity_sha256", "b" * 64),
        ("expected_packing_identity_sha256", "c" * 64),
        ("expected_exposure_plan_identity_sha256", "d" * 64),
        ("expected_ordered_next_exposure_identity_sha256", "e" * 64),
    ],
)
def test_valid_progress_and_d04_expectations_reach_checkpoint_io(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    valid: Any,
) -> None:
    class ReachedCheckpointRead(Exception):
        pass

    attempts: list[str] = []

    def reached_read(*_args: Any, **_kwargs: Any) -> None:
        attempts.append("checkpoint-open")
        raise ReachedCheckpointRead()

    monkeypatch.setattr(progress_trainer, "prepare_checkpoint_load", reached_read)
    with pytest.raises(ReachedCheckpointRead):
        progress_trainer.load_trainer_checkpoint(
            tmp_path / "not-a-checkpoint",
            model=object(),
            trainer=_PassiveTrainer(),
            **{field: valid},
        )
    assert attempts == ["checkpoint-open"]
