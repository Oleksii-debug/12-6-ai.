"""D05 post-load authorities must not alter committed resume state.

Synthetic CPU coverage only; this grants no corpus, training, or learned-weight evidence.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, core
from twelve_six.checkpoint import progress_trainer, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


class _TinyLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "postflight-state-seal", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=821,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler={"name": "cosine"},
        environment_lock_hash="f" * 64,
    )


def _source() -> Trainer:
    source = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=821, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    return source


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "restore_rng",
    [False, True],
    ids=["opt-out", "exact-rng"],
)
@pytest.mark.parametrize(
    "mutation",
    ["lookup-counter", "call-pending"],
)
def test_postflight_authority_cannot_hide_successful_state_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    restore_rng: bool,
    mutation: str,
) -> None:
    source = _source()
    checkpoint = tmp_path / f"postflight-{mutation}-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    original_postflight = loader._postflight_trainer_state
    original_policy = Trainer.__dict__["_require_deterministic_policy"]
    armed = False

    class EffectfulPolicyLookup:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            bound = original_policy.__get__(instance, owner)
            if instance is target and armed:
                instance.tokens_seen += 1
            return bound

    def policy_call_then_mutate(self: Trainer) -> None:
        original_policy(self)
        if self is target and armed:
            self._pending_tokens = 1

    def arm_then_postflight(trainer: Any, state: Any) -> None:
        nonlocal armed
        armed = True
        if mutation == "lookup-counter":
            monkeypatch.setattr(
                Trainer,
                "_require_deterministic_policy",
                EffectfulPolicyLookup(),
            )
        else:
            monkeypatch.setattr(
                Trainer,
                "_require_deterministic_policy",
                policy_call_then_mutate,
            )
        original_postflight(trainer, state)

    monkeypatch.setattr(loader, "_postflight_trainer_state", arm_then_postflight)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    expected = (
        "tokens_seen disagrees with checkpoint"
        if mutation == "lookup-counter"
        else "retained pending accumulation"
    )
    with pytest.raises(core.CheckpointCompatibilityError, match=expected):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert vars(target)["_failure_reason"] == "checkpoint_restore_apply_failed"
    assert vars(target)["_update_incomplete"] is True
    if mutation == "lookup-counter":
        assert target.tokens_seen == 3
        assert target._pending_tokens == 0
    else:
        assert target.tokens_seen == 2
        assert target._pending_tokens == 1
