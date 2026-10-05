"""Reject malformed independently expected core identities before checkpoint I/O.

Non-owning D05 regression for both public trainer checkpoint restore paths.
These cases are deliberately red until the expected-binding source owner adds
strict validation for the six remaining core identity expectations.
"""

from __future__ import annotations

from typing import Any

import pytest

from twelve_six.checkpoint import CheckpointCompatibilityError, progress_trainer, trainer_adapter


class _PassiveTrainer:
    def load_state_dict(self, _state: Any) -> None:
        raise AssertionError("expected identity must be checked before trainer load")


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "field",
    [
        "expected_git_sha",
        "expected_model_spec_hash",
        "expected_tokenizer_hash",
        "expected_tokenizer_vocab_hash",
        "expected_dataset_manifest_hash",
        "expected_run_manifest_hash",
    ],
)
@pytest.mark.parametrize(
    "invalid", ["not-hex", True, "A" * 64],
    ids=["wrong-format", "boolean", "uppercase"],
)
def test_malformed_expected_core_identity_rejected_before_checkpoint_io(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    field: str,
    invalid: Any,
) -> None:
    reads: list[str] = []

    def unexpected_read(*_args: Any, **_kwargs: Any) -> None:
        reads.append("checkpoint-open")
        raise AssertionError("malformed expected identity must reject before read")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", unexpected_read)
    with pytest.raises(CheckpointCompatibilityError, match=field):
        loader.load_trainer_checkpoint(
            tmp_path / "nonexistent-checkpoint",
            model=object(),
            trainer=_PassiveTrainer(),
            **{field: invalid},
        )
    assert reads == []


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "field,valid",
    [
        ("expected_git_sha", "a" * 40),
        ("expected_git_sha", "a" * 64),
        ("expected_model_spec_hash", "b" * 64),
        ("expected_tokenizer_hash", "c" * 64),
        ("expected_tokenizer_vocab_hash", "d" * 64),
        ("expected_dataset_manifest_hash", "e" * 64),
        ("expected_run_manifest_hash", "f" * 64),
    ],
)
def test_valid_expected_core_identity_reaches_checkpoint_io(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    field: str,
    valid: str,
) -> None:
    class ReachedCheckpointRead(Exception):
        pass

    reads: list[str] = []

    def reached_read(*_args: Any, **_kwargs: Any) -> None:
        reads.append("checkpoint-open")
        raise ReachedCheckpointRead()

    monkeypatch.setattr(loader, "prepare_checkpoint_load", reached_read)
    with pytest.raises(ReachedCheckpointRead):
        loader.load_trainer_checkpoint(
            tmp_path / "nonexistent-checkpoint",
            model=object(),
            trainer=_PassiveTrainer(),
            **{field: valid},
        )
    assert reads == ["checkpoint-open"]
