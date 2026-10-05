"""D05 post-load authorities must not alter committed resume state.

Synthetic CPU coverage only; this grants no corpus, training, or learned-weight evidence.
"""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
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


@pytest.fixture(autouse=True)
def preserve_process_state():
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
        torch.set_rng_state(torch_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(
            policy_before[0],
            warn_only=policy_before[1],
        )


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
def test_trainer_model_mode_drift_rejects_before_postflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    restore_rng: bool,
) -> None:
    source = _source()
    checkpoint = tmp_path / "trainer-eval-before-postflight-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    original_loader = Trainer.__dict__["load_state_dict"]
    postflight_calls: list[bool] = []

    def load_then_eval(self: Trainer, state: Any) -> None:
        original_loader(self, state)
        if self is target:
            target.model.eval()

    def forbid_postflight(*args: Any, **kwargs: Any) -> None:
        postflight_calls.append(True)
        raise AssertionError("invalid eval target reached postflight")

    monkeypatch.setattr(Trainer, "load_state_dict", load_then_eval)
    monkeypatch.setattr(loader, "_postflight_trainer_state", forbid_postflight)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="requires model training mode",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert postflight_calls == []
    assert target.model.training is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    assert vars(target)["_failure_reason"] == "checkpoint_restore_apply_failed"
    assert vars(target)["_update_incomplete"] is True

@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "final_phase",
    ["rng-replay", "opt-out-policy"],
)
def test_final_effectful_resume_callout_cannot_hide_committed_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    final_phase: str,
) -> None:
    source = _source()
    checkpoint = tmp_path / f"final-{final_phase}-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    restore_rng = final_phase == "rng-replay"

    if restore_rng:
        original_replay = loader._restore_checkpoint_rng_preserving_warn_only

        def replay_then_mutate(*args: Any, **kwargs: Any) -> None:
            original_replay(*args, **kwargs)
            target.tokens_seen += 1

        monkeypatch.setattr(
            loader,
            "_restore_checkpoint_rng_preserving_warn_only",
            replay_then_mutate,
        )
    else:
        original_policy = loader._assert_live_d02_determinism

        def final_policy_then_mutate(trainer: Any) -> Any:
            result = original_policy(trainer)
            if trainer is target and target.optimizer_step == 1:
                target.tokens_seen += 1
            return result

        monkeypatch.setattr(
            loader,
            "_assert_live_d02_determinism",
            final_policy_then_mutate,
        )

    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="post-load tokens_seen disagrees with checkpoint",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert target.tokens_seen == 3
    assert vars(target)["_failure_reason"] == "checkpoint_restore_apply_failed"
    assert vars(target)["_update_incomplete"] is True


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_post_rng_exact_seal_does_not_reenter_effectful_tensor_comparator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
) -> None:
    source = _source()
    checkpoint = tmp_path / "post-rng-inert-seal-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    original_replay = loader._restore_checkpoint_rng_preserving_warn_only
    original_equal = torch.equal
    replay_complete = False
    post_replay_equal_calls = 0

    def replay_then_mark(*args: Any, **kwargs: Any) -> None:
        nonlocal replay_complete
        original_replay(*args, **kwargs)
        replay_complete = True

    def track_equal(left: Any, right: Any) -> bool:
        nonlocal post_replay_equal_calls
        if replay_complete:
            post_replay_equal_calls += 1
        return original_equal(left, right)

    monkeypatch.setattr(
        loader,
        "_restore_checkpoint_rng_preserving_warn_only",
        replay_then_mark,
    )
    monkeypatch.setattr(torch, "equal", track_equal)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    loader.load_trainer_checkpoint(
        checkpoint,
        model=target.model,
        trainer=target,
        strict_model=False,
        restore_rng=True,
        **extra,
    )

    assert replay_complete is True
    assert post_replay_equal_calls == 0


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "final_phase",
    ["rng-replay", "opt-out-policy"],
)
@pytest.mark.parametrize(
    "mutation",
    ["model", "optimizer", "gradient", "policy"],
)
def test_final_effectful_resume_callout_cannot_hide_exact_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    final_phase: str,
    mutation: str,
) -> None:
    source = _source()
    checkpoint = tmp_path / f"final-exact-{final_phase}-{mutation}-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    restore_rng = final_phase == "rng-replay"
    weight_before = target.model.weight.detach().clone()
    lr_before = target.optimizer.param_groups[0]["lr"]
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    mutation_calls = 0

    def mutate_exact_state() -> None:
        nonlocal mutation_calls
        mutation_calls += 1
        if mutation == "model":
            with torch.no_grad():
                target.model.weight.add_(0.25)
        elif mutation == "optimizer":
            target.optimizer.param_groups[0]["lr"] *= 0.5
        elif mutation == "gradient":
            target.model.weight.grad = torch.full_like(target.model.weight, 0.25)
        else:
            torch.use_deterministic_algorithms(
                torch.are_deterministic_algorithms_enabled(),
                warn_only=not torch.is_deterministic_algorithms_warn_only_enabled(),
            )

    if restore_rng:
        original_replay = loader._restore_checkpoint_rng_preserving_warn_only

        def replay_then_mutate(*args: Any, **kwargs: Any) -> None:
            original_replay(*args, **kwargs)
            mutate_exact_state()

        monkeypatch.setattr(
            loader,
            "_restore_checkpoint_rng_preserving_warn_only",
            replay_then_mutate,
        )
    else:
        original_policy = loader._assert_live_d02_determinism

        def final_policy_then_mutate(trainer: Any) -> Any:
            result = original_policy(trainer)
            if trainer is target and target.optimizer_step == 1:
                mutate_exact_state()
            return result

        monkeypatch.setattr(
            loader,
            "_assert_live_d02_determinism",
            final_policy_then_mutate,
        )

    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    expected = (
        "auxiliary state changed during final checkpoint restore seal"
        if mutation == "optimizer"
        else (
            "live torch deterministic policy disagrees"
            if mutation == "policy"
            else "model changed during final checkpoint restore seal"
        )
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

    assert mutation_calls >= 1
    if mutation == "model":
        assert not torch.equal(target.model.weight.detach(), weight_before)
    elif mutation == "optimizer":
        assert target.optimizer.param_groups[0]["lr"] == lr_before * 0.5
    elif mutation == "gradient":
        assert target.model.weight.grad is not None
        torch.testing.assert_close(
            target.model.weight.grad,
            torch.full_like(target.model.weight, 0.25),
            rtol=0,
            atol=0,
        )
    else:
        assert (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ) == policy_before
    assert vars(target)["_failure_reason"] == "checkpoint_restore_apply_failed"
    assert vars(target)["_update_incomplete"] is True

