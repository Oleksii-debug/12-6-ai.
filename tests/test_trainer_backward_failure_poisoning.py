"""Adversarial D02 backward interruption tests (synthetic inputs, no training credit)."""

from __future__ import annotations

import pytest
import torch

from twelve_six.training import Trainer, TrainerConfig, TrainingStateInvalidError


class _FaultDuringBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, failure_type):
        ctx.failure_type = failure_type
        return values.clone()

    @staticmethod
    def backward(ctx, incoming):
        raise ctx.failure_type("synthetic backward interruption")


class _TinyLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.failure_type = None
        self.forward_failure_type = None

    def forward(self, input_ids):
        if self.forward_failure_type is not None:
            torch.rand(())  # Model forward may have already consumed RNG.
            raise self.forward_failure_type("synthetic forward interruption")
        values = self.weight
        if self.failure_type is not None:
            values = _FaultDuringBackward.apply(values, self.failure_type)
        return values.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
def test_non_runtime_backward_failure_poisoned_and_cannot_replay(failure_type):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )

    first = trainer.train_microbatch(_BATCH)
    assert first.micro_step == 1
    assert first.optimizer_step == 0
    assert model.weight.grad is not None  # One accumulation microbatch is pending.
    before_weights = model.weight.detach().clone()

    model.failure_type = failure_type
    with pytest.raises(failure_type, match="synthetic backward interruption"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 2
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None  # Partial and prior gradients were discarded.

    model.failure_type = None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.assert_checkpoint_safe()
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_clean_backward_still_updates_after_exact_accumulation_boundary():
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    before = model.weight.detach().clone()

    first = trainer.train_microbatch(_BATCH)
    second = trainer.train_microbatch(_BATCH)

    assert first.optimizer_stepped is False
    assert second.optimizer_stepped is True
    assert trainer.micro_step == 2
    assert trainer.optimizer_step == 1
    assert trainer.tokens_seen == 4
    assert not torch.equal(before, model.weight)
    assert trainer.state_dict().optimizer_step == 1


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
def test_forward_interruption_clears_pending_gradients_and_requires_recovery(failure_type):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    before_weights = model.weight.detach().clone()
    model.forward_failure_type = failure_type

    with pytest.raises(failure_type, match="synthetic forward interruption"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 2
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    model.forward_failure_type = None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


@pytest.mark.parametrize(
    ("broken_batch", "expected_error"),
    [
        (
            {"input_ids": torch.tensor([[0, 1]]), "target_ids": torch.tensor([[999, 2]])},
            IndexError,
        ),
        (
            {**_BATCH, "loss_mask": torch.tensor([[2, 1]])},
            ValueError,
        ),
    ],
)
def test_loss_failure_requires_recovery_even_with_prior_accumulation(
    broken_batch, expected_error
):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    before_weights = model.weight.detach().clone()

    with pytest.raises(expected_error):
        trainer.train_microbatch(broken_batch)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 2
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_preflight_shape_error_before_forward_does_not_poison_trainer():
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    bad = {"input_ids": torch.tensor([0, 1]), "target_ids": torch.tensor([1, 2])}
    with pytest.raises(ValueError, match="shape"):
        trainer.train_microbatch(bad)

    assert trainer.micro_step == 0
    assert trainer.tokens_seen == 0
    trainer.train_microbatch(_BATCH)
    trainer.train_microbatch(_BATCH)
    assert trainer.optimizer_step == 1


def test_post_backward_accounting_error_cannot_reuse_completed_gradients():
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    before_weights = model.weight.detach().clone()

    trainer.optimizer.param_groups[0]["lr"] = "invalid-rate"
    with pytest.raises(
        TrainingStateInvalidError,
        match="default constant optimizer rate is malformed",
    ):
        trainer.train_microbatch(_BATCH)

    # The hardened first-party optimizer contract rejects before the second
    # microbatch consumes exposure. The earlier pending accumulation is poisoned
    # and its gradients are discarded rather than silently replayed.
    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 2
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_optimizer_interrupt_cannot_be_replayed(monkeypatch):
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    before_weights = model.weight.detach().clone()

    def interrupted_step(*args, **kwargs):
        raise KeyboardInterrupt("synthetic optimizer interruption")

    monkeypatch.setattr(trainer.optimizer, "step", interrupted_step)
    with pytest.raises(KeyboardInterrupt, match="synthetic optimizer interruption"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.assert_checkpoint_safe()


@pytest.mark.parametrize("cleanup_type", [RuntimeError, KeyboardInterrupt])
def test_backward_and_cleanup_double_fault_preserves_primary_and_poison(
    monkeypatch, cleanup_type
):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    before_weights = model.weight.detach().clone()
    model.failure_type = ValueError

    def broken_cleanup(*args, **kwargs):
        raise cleanup_type("synthetic zero_grad cleanup failure")

    monkeypatch.setattr(trainer.optimizer, "zero_grad", broken_cleanup)
    with pytest.raises(ValueError, match="synthetic backward interruption"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert "backward failed" in trainer._failure_reason
    assert f"gradient cleanup failed: {cleanup_type.__name__}" in trainer._failure_reason
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


@pytest.mark.parametrize("cleanup_type", [RuntimeError, KeyboardInterrupt])
def test_optimizer_and_cleanup_double_fault_preserves_primary_and_poison(
    monkeypatch, cleanup_type
):
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    before_weights = model.weight.detach().clone()

    def broken_step(*args, **kwargs):
        raise ValueError("synthetic primary optimizer failure")

    def broken_cleanup(*args, **kwargs):
        raise cleanup_type("synthetic zero_grad cleanup failure")

    monkeypatch.setattr(trainer.optimizer, "step", broken_step)
    monkeypatch.setattr(trainer.optimizer, "zero_grad", broken_cleanup)
    with pytest.raises(ValueError, match="synthetic primary optimizer failure"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer._update_incomplete is True
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert "optimizer/scheduler update failed" in trainer._failure_reason
    assert f"gradient cleanup failed: {cleanup_type.__name__}" in trainer._failure_reason
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.assert_checkpoint_safe()


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_train_mode_failure_poisoned_without_replaying_accumulation(
    monkeypatch, failure_type
):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    before_weights = model.weight.detach().clone()

    def broken_train(*args, **kwargs):
        torch.rand(())  # train-mode hooks may already have consumed RNG.
        raise failure_type("synthetic train-mode failure")

    monkeypatch.setattr(model, "train", broken_train)
    with pytest.raises(failure_type, match="synthetic train-mode failure"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 2
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_invalid_batch_does_not_invoke_effectful_train_mode(monkeypatch):
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    train_calls = []

    def record_train(*args, **kwargs):
        train_calls.append(True)
        return model

    monkeypatch.setattr(model, "train", record_train)
    with pytest.raises(ValueError, match="shape"):
        trainer.train_microbatch(
            {"input_ids": torch.tensor([1, 2]), "target_ids": torch.tensor([1, 2])}
        )
    assert train_calls == []
    assert trainer.micro_step == 0
    assert trainer._failure_reason is None
    trainer.train_microbatch(_BATCH)
    assert train_calls == [True]
    assert trainer.optimizer_step == 1


class _OrphanGradientLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.orphan = torch.tensor([0.3, -0.1, 0.2], requires_grad=True)

    def forward(self, input_ids):
        # Loss has an autograd graph, but none reaches the registered parameter.
        return self.orphan.reshape(1, 1, 3).expand(*input_ids.shape, 3)


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_backward_without_model_parameter_gradients_cannot_count_optimizer_step(
    accumulation_steps
):
    model = _OrphanGradientLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=accumulation_steps, seed=17),
    )
    before_weights = model.weight.detach().clone()

    for _ in range(accumulation_steps - 1):
        metrics = trainer.train_microbatch(_BATCH)
        assert metrics.optimizer_stepped is False
    with pytest.raises(RuntimeError, match="no model-parameter gradients"):
        trainer.train_microbatch(_BATCH)

    assert model.orphan.grad is not None
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert trainer.micro_step == accumulation_steps
    assert trainer.optimizer_step == 0
    assert trainer._update_incomplete is True
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_actual_zero_gradient_tensor_is_not_confused_with_absent_gradient():
    class ZeroGradientModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

        def forward(self, input_ids):
            return (self.weight * 0).reshape(1, 1, 3).expand(*input_ids.shape, 3)

    model = ZeroGradientModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    metrics = trainer.train_microbatch(_BATCH)
    assert metrics.optimizer_stepped is True
    assert trainer.optimizer_step == 1
    assert trainer.state_dict().optimizer_step == 1


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_effectful_batch_transfer_failure_poisoned_and_no_replay(failure_type):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    before_weights = model.weight.detach().clone()

    class FailingTransfer:
        ndim = 2
        shape = (1, 2)

        def to(self, device):
            torch.rand(())  # Simulate side effects before failed device transfer.
            raise failure_type("synthetic device transfer failure")

    broken = {"input_ids": FailingTransfer(), "target_ids": _BATCH["target_ids"]}
    with pytest.raises(failure_type, match="synthetic device transfer failure"):
        trainer.train_microbatch(broken)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 2
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_pure_shape_preflight_error_keeps_pending_accumulation_retryable():
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    pending_grad = model.weight.grad.detach().clone()
    bad = {"input_ids": torch.tensor([0, 1]), "target_ids": torch.tensor([1, 2])}

    with pytest.raises(ValueError, match="shape"):
        trainer.train_microbatch(bad)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer._failure_reason is None
    torch.testing.assert_close(model.weight.grad, pending_grad, rtol=0, atol=0)
    assert trainer.train_microbatch(_BATCH).optimizer_stepped is True
    assert trainer.optimizer_step == 1


def test_interrupted_target_accounting_poisoned_before_new_forward(monkeypatch):
    import twelve_six.training.trainer as trainer_module

    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    before_weights = model.weight.detach().clone()

    def interrupted_count(*args, **kwargs):
        torch.rand(())  # A device reduction can fail after pending activity.
        raise KeyboardInterrupt("synthetic target-count synchronization interrupt")

    monkeypatch.setattr(trainer_module, "_count_training_tokens", interrupted_count)
    with pytest.raises(KeyboardInterrupt, match="target-count synchronization"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_zero_target_count_preflight_retains_valid_pending_gradients():
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    trainer.train_microbatch(_BATCH)
    before_grad = model.weight.grad.detach().clone()
    empty_targets = {
        "input_ids": _BATCH["input_ids"],
        "target_ids": torch.full((1, 2), -100, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="at least one valid target"):
        trainer.train_microbatch(empty_targets)

    assert trainer._failure_reason is None
    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    torch.testing.assert_close(model.weight.grad, before_grad, rtol=0, atol=0)
    assert trainer.train_microbatch(_BATCH).optimizer_stepped is True
    assert trainer.optimizer_step == 1


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_run_does_not_fetch_one_extra_batch_after_final_optimizer_step(
    accumulation_steps
):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=accumulation_steps, seed=17),
    )
    fetched = []

    def counted_batches():
        for index in range(accumulation_steps + 1):
            fetched.append(index)
            yield _BATCH

    result = trainer.run(counted_batches())
    assert result.optimizer_steps_completed == 1
    assert result.microbatches_consumed == accumulation_steps
    assert fetched == list(range(accumulation_steps))
    assert trainer.state_dict().optimizer_step == 1


def test_run_poisoned_if_batch_iterator_faults_with_pending_accumulation():
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    before_weights = model.weight.detach().clone()

    def broken_source():
        yield _BATCH
        torch.rand(())  # Iterator may mutate RNG/cursor before the error.
        raise ValueError("synthetic corpus cursor failure")

    with pytest.raises(ValueError, match="synthetic corpus cursor failure"):
        trainer.run(broken_source())

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_run_poisoned_on_interrupt_during_iterator_creation():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))

    class BrokenIterable:
        def __iter__(self):
            torch.rand(())
            raise KeyboardInterrupt("synthetic iterable construction interrupt")

    with pytest.raises(KeyboardInterrupt, match="iterable construction interrupt"):
        trainer.run(BrokenIterable())
    assert trainer.optimizer_step == 0
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_run_natural_exhaustion_at_committed_boundary_stays_recoverable():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=2, seed=17))
    with pytest.raises(RuntimeError, match="batch iterable exhausted"):
        trainer.run([_BATCH])
    assert trainer.optimizer_step == 1
    assert trainer._failure_reason is None
    assert trainer.state_dict().optimizer_step == 1


def test_run_already_at_limit_does_not_even_construct_data_iterator():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    trainer.train_microbatch(_BATCH)

    class MustNotBeRead:
        def __iter__(self):
            raise AssertionError("completed run must not touch data source")

    outcome = trainer.run(MustNotBeRead())
    assert outcome.start_optimizer_step == 1
    assert outcome.end_optimizer_step == 1
    assert outcome.optimizer_steps_completed == 0
    assert outcome.microbatches_consumed == 0
    assert outcome.tokens_consumed == 0
    assert outcome.final_metrics is None
    assert trainer.state_dict().optimizer_step == 1


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_metrics_hook_error_after_consumed_batch_requires_verified_recovery(
    failure_type, accumulation_steps
):
    from twelve_six.training import CheckpointHookError

    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=accumulation_steps, seed=17),
    )
    checkpoint_calls = []

    def failed_metrics(metrics):
        raise failure_type("synthetic metrics persistence failure")

    def checkpoint_callback(*args):
        checkpoint_calls.append(True)

    expected_error = failure_type if failure_type is KeyboardInterrupt else CheckpointHookError
    with pytest.raises(expected_error):
        trainer.run(
            [_BATCH] * accumulation_steps,
            on_metrics=failed_metrics,
            on_checkpoint=checkpoint_callback,
            checkpoint_every_steps=1,
        )

    assert checkpoint_calls == []
    assert trainer.micro_step == 1
    assert trainer.optimizer_step == (1 if accumulation_steps == 1 else 0)
    assert trainer._failure_reason.startswith("metrics hook failed")
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
def test_checkpoint_hook_error_after_committed_update_requires_verified_recovery(
    failure_type
):
    from twelve_six.training import CheckpointHookError

    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    recorded_metrics = []

    def failed_checkpoint(trainer_instance, metrics):
        assert trainer_instance.optimizer_step == 1
        raise failure_type("synthetic checkpoint publication failure")

    expected_error = failure_type if failure_type is KeyboardInterrupt else CheckpointHookError
    with pytest.raises(expected_error):
        trainer.run(
            [_BATCH],
            on_metrics=recorded_metrics.append,
            on_checkpoint=failed_checkpoint,
            checkpoint_every_steps=1,
        )

    assert len(recorded_metrics) == 1
    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 1
    assert trainer._failure_reason.startswith("checkpoint hook failed")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.run([_BATCH])


def test_finite_gradients_with_overflowed_aggregate_norm_never_update_model():
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, seed=17, gradient_clip_norm=None),
    )
    before_weights = model.weight.detach().clone()
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, 1e30))

    with pytest.raises(NonFiniteTrainingError, match="non-finite gradient norm"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer._update_incomplete is True
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


@pytest.mark.parametrize("unsafe_rate", [float("nan"), float("inf"), float("-inf"), -0.01])
def test_runtime_unsafe_learning_rate_cannot_commit_optimizer_step(unsafe_rate):
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    before_weights = model.weight.detach().clone()
    trainer.optimizer.param_groups[0]["lr"] = unsafe_rate

    with pytest.raises(
        TrainingStateInvalidError,
        match="default constant optimizer rate differs from configured learning rate",
    ):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 0
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 0
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


@pytest.mark.parametrize("accumulation_steps", [2, 3])
def test_run_exhaustion_mid_accumulation_discards_uncheckpointable_gradients(
    accumulation_steps
):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=accumulation_steps, seed=17),
    )
    before_weights = model.weight.detach().clone()

    with pytest.raises(RuntimeError, match="mid-accumulation"):
        trainer.run([_BATCH] * (accumulation_steps - 1))

    assert trainer.micro_step == accumulation_steps - 1
    assert trainer.optimizer_step == 0
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert "exhausted mid-accumulation" in trainer._failure_reason
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_reject_optimizer_over_unrelated_parameter_at_construction():
    model = _TinyLogitModel()
    foreign = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([foreign], lr=3e-4)
    before = model.weight.detach().clone()

    with pytest.raises(ValueError, match="not owned by the model"):
        Trainer(model, TrainerConfig(max_steps=1, seed=17), optimizer=optimizer)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


def test_reject_optimizer_omitting_a_trainable_model_parameter():
    class TwoParameterModel(_TinyLogitModel):
        def __init__(self) -> None:
            super().__init__()
            self.additional = torch.nn.Parameter(torch.zeros(3))

    model = TwoParameterModel()
    optimizer = torch.optim.AdamW([model.weight], lr=3e-4)
    with pytest.raises(ValueError, match="omits trainable model parameters"):
        Trainer(model, TrainerConfig(max_steps=1, seed=17), optimizer=optimizer)


def test_reject_duplicate_parameter_assignment_across_optimizer_groups():
    model = _TinyLogitModel()
    optimizer = torch.optim.AdamW([model.weight], lr=3e-4)
    optimizer.param_groups[0]["params"].append(model.weight)
    with pytest.raises(ValueError, match="duplicate parameter"):
        Trainer(model, TrainerConfig(max_steps=1, seed=17), optimizer=optimizer)



def test_optimizer_group_nonsequence_is_type_error_and_poisons_before_exposure():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    before = model.weight.detach().clone()
    trainer.optimizer.param_groups[0]["params"] = iter([model.weight])

    with pytest.raises(TypeError, match="concrete parameter sequence"):
        trainer.train_microbatch(_BATCH)

    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(model.weight.detach(), before, rtol=0, atol=0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_cleanup_interrupt_preserves_original_backward_failure(monkeypatch):
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    model.failure_type = ValueError

    def interrupted_optimizer_cleanup(*args, **kwargs):
        raise KeyboardInterrupt("synthetic cleanup interruption")

    monkeypatch.setattr(trainer.optimizer, "zero_grad", interrupted_optimizer_cleanup)
    with pytest.raises(ValueError, match="synthetic backward interruption"):
        trainer.train_microbatch(_BATCH)

    assert "gradient cleanup failed: KeyboardInterrupt" in trainer._failure_reason
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_optimizer_group_swap_before_microbatch_poisoned_without_exposure():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    original = model.weight.detach().clone()
    foreign = torch.nn.Parameter(torch.ones(3))
    trainer.optimizer.param_groups[0]["params"] = [foreign]

    with pytest.raises(ValueError, match="not owned by the model"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 0
    assert trainer.optimizer_step == 0
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_optimizer_group_swap_during_forward_cannot_commit_phantom_step():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    original_forward = model.forward
    original = model.weight.detach().clone()
    foreign = torch.nn.Parameter(torch.ones(3))

    def swapping_forward(input_ids):
        logits = original_forward(input_ids)
        trainer.optimizer.param_groups[0]["params"] = [foreign]
        return logits

    model.forward = swapping_forward
    with pytest.raises(ValueError, match="not owned by the model"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_optimizer_coverage_allows_frozen_model_parameters_in_group():
    class PartiallyFrozenModel(_TinyLogitModel):
        def __init__(self) -> None:
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    model = PartiallyFrozenModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    assert trainer.train_microbatch(_BATCH).optimizer_stepped is True
    assert trainer.optimizer_step == 1


def test_primary_backward_exception_survives_both_gradient_cleanup_failures(
    monkeypatch,
):
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    model.failure_type = ValueError

    def broken_model_cleanup(*args, **kwargs):
        raise RuntimeError("synthetic model cleanup fault")

    def broken_optimizer_cleanup(*args, **kwargs):
        raise KeyboardInterrupt("synthetic optimizer cleanup fault")

    monkeypatch.setattr(model, "zero_grad", broken_model_cleanup)
    monkeypatch.setattr(trainer.optimizer, "zero_grad", broken_optimizer_cleanup)
    with pytest.raises(ValueError, match="synthetic backward interruption"):
        trainer.train_microbatch(_BATCH)

    assert trainer.optimizer_step == 0
    assert "backward failed" in trainer._failure_reason
    assert "model gradient cleanup failed: RuntimeError" in trainer._failure_reason
    assert "gradient cleanup failed: KeyboardInterrupt" in trainer._failure_reason
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


class _TwoGroupModel(_TinyLogitModel):
    def __init__(self) -> None:
        super().__init__()
        self.extra = torch.nn.Parameter(torch.tensor([0.03, 0.04, -0.02]))

    def forward(self, input_ids):
        return super().forward(input_ids) + self.extra


@pytest.mark.parametrize(
    "bad_rate", [float("nan"), float("inf"), float("-inf"), -0.01, True],
)
def test_secondary_optimizer_group_unsafe_lr_never_updates_either_group(bad_rate):
    from twelve_six.training import NonFiniteTrainingError

    model = _TwoGroupModel()
    optimizer = torch.optim.AdamW(
        [{"params": [model.weight], "lr": 1e-3},
         {"params": [model.extra], "lr": 2e-3}],
    )
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17), optimizer=optimizer)
    original_weight = model.weight.detach().clone()
    original_extra = model.extra.detach().clone()
    optimizer.param_groups[1]["lr"] = bad_rate

    with pytest.raises(NonFiniteTrainingError, match="learning rate must be finite"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert model.weight.grad is None
    assert model.extra.grad is None
    torch.testing.assert_close(model.weight, original_weight, rtol=0, atol=0)
    torch.testing.assert_close(model.extra, original_extra, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_distinct_valid_optimizer_group_learning_rates_both_commit():
    model = _TwoGroupModel()
    optimizer = torch.optim.AdamW(
        [{"params": [model.weight], "lr": 1e-3},
         {"params": [model.extra], "lr": 2e-3}],
    )
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17), optimizer=optimizer)
    original_weight = model.weight.detach().clone()
    original_extra = model.extra.detach().clone()

    result = trainer.train_microbatch(_BATCH)

    assert result.optimizer_stepped is True
    assert result.learning_rate == 1e-3
    assert trainer.optimizer_step == 1
    assert not torch.equal(model.weight, original_weight)
    assert not torch.equal(model.extra, original_extra)
    assert trainer.state_dict().optimizer_step == 1


def test_direct_restore_scheduler_presence_mismatch_is_retryable_preflight(monkeypatch):
    from dataclasses import replace

    config = TrainerConfig(max_steps=2, warmup_steps=1, scheduler="cosine", seed=17)
    producer = Trainer(_TinyLogitModel(), config)
    state = producer.state_dict()
    receiver = Trainer(_TinyLogitModel(), config)
    original_load = receiver.optimizer.load_state_dict
    calls = []

    def record_load(value):
        calls.append(True)
        return original_load(value)

    monkeypatch.setattr(receiver.optimizer, "load_state_dict", record_load)
    with pytest.raises(ValueError, match="scheduler state/config mismatch"):
        receiver.load_state_dict(replace(state, scheduler=None))

    assert calls == []
    assert receiver._failure_reason is None
    assert receiver._update_incomplete is False
    receiver.load_state_dict(state)
    assert calls == [True]
    assert receiver.state_dict().optimizer_step == 0


def test_direct_restore_partial_optimizer_load_poisoned_and_original_error(monkeypatch):
    config = TrainerConfig(max_steps=1, seed=17)
    state = Trainer(_TinyLogitModel(), config).state_dict()
    receiver = Trainer(_TinyLogitModel(), config)

    def corrupt_then_fail(value):
        receiver.optimizer.param_groups[0]["lr"] = -0.1
        raise ValueError("synthetic partial optimizer restore")

    monkeypatch.setattr(receiver.optimizer, "load_state_dict", corrupt_then_fail)
    with pytest.raises(ValueError, match="synthetic partial optimizer restore"):
        receiver.load_state_dict(state)

    assert receiver._update_incomplete is True
    assert receiver._failure_reason.startswith("trainer state restore failed")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        receiver.state_dict()
    with pytest.raises(TrainingStateInvalidError, match="verified model"):
        receiver.load_state_dict(state)


def test_direct_restore_partial_scheduler_apply_poisoned(monkeypatch):
    config = TrainerConfig(max_steps=2, warmup_steps=1, scheduler="cosine", seed=17)
    state = Trainer(_TinyLogitModel(), config).state_dict()
    receiver = Trainer(_TinyLogitModel(), config)

    def corrupt_then_fail(value):
        receiver.scheduler.last_epoch = 99
        raise RuntimeError("synthetic partial scheduler restore")

    monkeypatch.setattr(receiver.scheduler, "load_state_dict", corrupt_then_fail)
    with pytest.raises(RuntimeError, match="synthetic partial scheduler restore"):
        receiver.load_state_dict(state)
    assert receiver._update_incomplete is True
    assert receiver._failure_reason.startswith("trainer state restore failed")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        receiver.train_microbatch(_BATCH)


def test_direct_restore_scaler_interrupt_preserves_original_and_poison(monkeypatch):
    config = TrainerConfig(max_steps=1, seed=17)
    state = Trainer(_TinyLogitModel(), config).state_dict()
    receiver = Trainer(_TinyLogitModel(), config)

    def interrupted_scaler_load(value):
        raise KeyboardInterrupt("synthetic scaler restore interruption")

    monkeypatch.setattr(receiver.scaler, "load_state_dict", interrupted_scaler_load)
    with pytest.raises(KeyboardInterrupt, match="synthetic scaler restore interruption"):
        receiver.load_state_dict(state)
    assert receiver._update_incomplete is True
    assert receiver._failure_reason.startswith("trainer state restore failed")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        receiver.state_dict()


def test_direct_restore_cleanup_fault_keeps_primary_failure(monkeypatch):
    config = TrainerConfig(max_steps=1, seed=17)
    state = Trainer(_TinyLogitModel(), config).state_dict()
    receiver = Trainer(_TinyLogitModel(), config)

    def refused_cleanup(*args, **kwargs):
        raise ValueError("synthetic final zero_grad failure")

    monkeypatch.setattr(receiver.optimizer, "zero_grad", refused_cleanup)
    with pytest.raises(ValueError, match="synthetic final zero_grad failure"):
        receiver.load_state_dict(state)
    assert receiver._update_incomplete is True
    assert "trainer state restore failed" in receiver._failure_reason
    assert "gradient cleanup failed: ValueError" in receiver._failure_reason
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        receiver.state_dict()


def test_valid_direct_restore_retains_exact_next_step_trajectory():
    config = TrainerConfig(max_steps=2, seed=17)
    original = Trainer(_TinyLogitModel(), config)
    original.train_microbatch(_BATCH)
    snapshot = original.state_dict()
    weights = original.model.state_dict()

    restored = Trainer(_TinyLogitModel(), config)
    restored.model.load_state_dict(weights)
    restored.load_state_dict(snapshot)

    original.train_microbatch(_BATCH)
    restored.train_microbatch(_BATCH)
    torch.testing.assert_close(
        original.model.weight, restored.model.weight, rtol=0, atol=0,
    )
    assert original.state_dict().optimizer_step == restored.state_dict().optimizer_step == 2


def test_enabled_scaler_missing_state_rejected_before_optimizer_mutation(monkeypatch):
    from dataclasses import replace

    config = TrainerConfig(max_steps=1, seed=17)
    state = Trainer(_TinyLogitModel(), config).state_dict()
    receiver = Trainer(_TinyLogitModel(), config)
    calls = []

    def forbidden_optimizer_load(value):
        calls.append(True)
        raise AssertionError("missing scaler state must fail before optimizer load")

    monkeypatch.setattr(receiver.scaler, "is_enabled", lambda: True)
    monkeypatch.setattr(receiver.optimizer, "load_state_dict", forbidden_optimizer_load)
    with pytest.raises(ValueError, match="enabled gradient scaler checkpoint state missing"):
        receiver.load_state_dict(replace(state, scaler=None))

    assert calls == []
    assert receiver._failure_reason is None
    assert receiver._update_incomplete is False
    monkeypatch.undo()
    receiver.load_state_dict(state)
    assert receiver.state_dict().optimizer_step == 0


def test_disabled_scaler_legacy_none_state_remains_restoreable():
    from dataclasses import replace

    config = TrainerConfig(max_steps=1, seed=17)
    state = Trainer(_TinyLogitModel(), config).state_dict()
    receiver = Trainer(_TinyLogitModel(), config)
    assert receiver.scaler.is_enabled() is False
    receiver.load_state_dict(replace(state, scaler=None))
    assert receiver.train_microbatch(_BATCH).optimizer_stepped is True
    assert receiver.state_dict().optimizer_step == 1


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("weight_decay", float("nan")),
        ("weight_decay", -0.1),
        ("eps", float("nan")),
        ("eps", 0.0),
        ("betas", (float("nan"), 0.9)),
        ("betas", (0.9, 1.0)),
        ("betas", (True, 0.9)),
    ],
)
def test_secondary_group_corrupt_adamw_hyperparameters_cannot_write_nan_weights(
    field, bad_value
):
    from twelve_six.training import NonFiniteTrainingError

    model = _TwoGroupModel()
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.weight], "lr": 1e-3},
            {"params": [model.extra], "lr": 2e-3},
        ],
    )
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17), optimizer=optimizer)
    original_weight = model.weight.detach().clone()
    original_extra = model.extra.detach().clone()
    trainer.optimizer.param_groups[1][field] = bad_value

    with pytest.raises(NonFiniteTrainingError, match=f"optimizer {field}"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert model.weight.grad is None
    assert model.extra.grad is None
    torch.testing.assert_close(model.weight, original_weight, rtol=0, atol=0)
    torch.testing.assert_close(model.extra, original_extra, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_distinct_valid_adamw_group_hyperparameters_complete_update():
    model = _TwoGroupModel()
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.weight], "lr": 1e-3, "weight_decay": 0.0, "eps": 1e-8},
            {"params": [model.extra], "lr": 2e-3, "weight_decay": 0.1, "eps": 1e-6},
        ],
    )
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17), optimizer=optimizer)
    optimizer.param_groups[1]["betas"] = (0.85, 0.95)
    previous_weight = model.weight.detach().clone()
    previous_extra = model.extra.detach().clone()

    metrics = trainer.train_microbatch(_BATCH)
    assert metrics.optimizer_stepped is True
    assert trainer.optimizer_step == 1
    assert not torch.equal(model.weight, previous_weight)
    assert not torch.equal(model.extra, previous_extra)
    assert trainer.state_dict().optimizer_step == 1


def test_silent_noop_optimizer_zero_grad_cannot_publish_clean_step(monkeypatch):
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    initial_weights = model.weight.detach().clone()
    monkeypatch.setattr(trainer.optimizer, "zero_grad", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="residual model gradients"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 1  # Parameter mutation already committed.
    assert not torch.equal(model.weight, initial_weights)
    assert model.weight.grad is None  # Independent model cleanup still runs.
    assert trainer._update_incomplete is True
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_checkpoint_export_rejects_optimizer_group_drift_after_committed_step():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    trainer.train_microbatch(_BATCH)
    foreign = torch.nn.Parameter(torch.ones(3))
    trainer.optimizer.param_groups[0]["params"] = [foreign]

    with pytest.raises(ValueError, match="not owned by the model"):
        trainer.state_dict()

    assert trainer.optimizer_step == 1
    assert "checkpoint boundary has invalid optimizer" in trainer._failure_reason
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.assert_checkpoint_safe()


def test_checkpoint_export_rejects_residual_gradient_after_committed_step():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    trainer.train_microbatch(_BATCH)
    model.weight.grad = torch.ones_like(model.weight)

    with pytest.raises(
        TrainingStateInvalidError,
        match="residual parameter gradients",
    ):
        trainer.state_dict()

    assert trainer.optimizer_step == 1
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_checkpoint_refusal_mid_accumulation_preserves_valid_pending_gradients():
    model = _TinyLogitModel()
    trainer = Trainer(
        model, TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17)
    )
    trainer.train_microbatch(_BATCH)
    saved_gradient = model.weight.grad.detach().clone()

    with pytest.raises(RuntimeError, match="mid-accumulation"):
        trainer.state_dict()

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer._failure_reason is None
    torch.testing.assert_close(model.weight.grad, saved_gradient, rtol=0, atol=0)
    assert trainer.train_microbatch(_BATCH).optimizer_stepped is True
    assert trainer.state_dict().optimizer_step == 1


def test_adamw_finite_gradients_but_overflowed_weights_cannot_earn_step_credit():
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    config = TrainerConfig(max_steps=2, seed=17, gradient_clip_norm=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e35, weight_decay=1.0)
    trainer = Trainer(model, config, optimizer=optimizer)

    first = trainer.train_microbatch(_BATCH)
    assert first.optimizer_stepped is True
    assert trainer.optimizer_step == 1
    assert torch.isfinite(model.weight).all().item() is True

    with pytest.raises(NonFiniteTrainingError, match="non-finite model weights"):
        trainer.train_microbatch(_BATCH)

    # Step 2 may already have physically mutated the parameter to Inf, but
    # cannot be credited as a valid learned-target transition or replayed.
    assert trainer.micro_step == 2
    assert trainer.optimizer_step == 1
    assert trainer._update_incomplete is True
    assert trainer._failure_reason.startswith("optimizer/scheduler update failed")
    assert torch.isfinite(model.weight).all().item() is False
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_nonfinite_optimizer_moment_with_finite_weights_cannot_earn_step_credit(
    monkeypatch,
):
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    real_step = trainer.optimizer.step

    def corrupt_moment_after_step(*args, **kwargs):
        result = real_step(*args, **kwargs)
        trainer.optimizer.state[model.weight]["exp_avg"].fill_(float("nan"))
        return result

    monkeypatch.setattr(trainer.optimizer, "step", corrupt_moment_after_step)
    with pytest.raises(NonFiniteTrainingError, match="non-finite state"):
        trainer.train_microbatch(_BATCH)

    assert torch.isfinite(model.weight).all().item() is True
    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer._update_incomplete is True
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_normal_adamw_step_keeps_finite_weights_and_optimizer_state():
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    result = trainer.train_microbatch(_BATCH)
    assert result.optimizer_stepped is True
    assert trainer.optimizer_step == 1
    assert torch.isfinite(model.weight).all().item() is True
    for state in trainer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all().item() is True
    assert trainer.state_dict().optimizer_step == 1


def test_scheduler_can_never_publish_nonfinite_next_step_rate(monkeypatch):
    from twelve_six.training import NonFiniteTrainingError

    config = TrainerConfig(max_steps=2, warmup_steps=1, scheduler="cosine", seed=17)
    model = _TinyLogitModel()
    trainer = Trainer(model, config)

    def corrupt_scheduler():
        trainer.optimizer.param_groups[0]["lr"] = float("nan")

    monkeypatch.setattr(trainer.scheduler, "step", corrupt_scheduler)
    with pytest.raises(NonFiniteTrainingError, match="learning rate must be finite"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 1  # The optimizer already committed.
    assert trainer._update_incomplete is True
    assert trainer._failure_reason.startswith("optimizer/scheduler update failed")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_scaler_update_cannot_publish_nonfinite_scaling_state(monkeypatch):
    from twelve_six.training import NonFiniteTrainingError

    trainer = Trainer(_TinyLogitModel(), TrainerConfig(max_steps=1, seed=17))
    updated = []

    def corrupt_scaler():
        updated.append(True)

    monkeypatch.setattr(trainer.scaler, "update", corrupt_scaler)
    monkeypatch.setattr(
        trainer.scaler, "state_dict",
        lambda: {"scale": float("inf")} if updated else {},
    )
    with pytest.raises(NonFiniteTrainingError, match="gradient scaler has non-finite"):
        trainer.train_microbatch(_BATCH)

    assert updated == [True]
    assert trainer.optimizer_step == 1
    assert trainer._update_incomplete is True
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_optimizer_group_swap_after_step_with_cleared_model_gradients_poisoned(
    monkeypatch,
):
    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    initial = model.weight.detach().clone()
    original_step = trainer.optimizer.step
    foreign = torch.nn.Parameter(torch.ones(3))

    def swap_after_successful_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        model.zero_grad(set_to_none=True)
        trainer.optimizer.param_groups[0]["params"] = [foreign]
        return result

    monkeypatch.setattr(trainer.optimizer, "step", swap_after_successful_step)
    with pytest.raises(ValueError, match="not owned by the model"):
        trainer.train_microbatch(_BATCH)

    assert not torch.equal(model.weight, initial)  # Physical step committed.
    assert trainer.optimizer_step == 1
    assert trainer._update_incomplete is True
    assert model.weight.grad is None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_direct_restore_rejects_consumed_partial_accumulation_before_optimizer_io(
    monkeypatch,
):
    config = TrainerConfig(max_steps=2, gradient_accumulation_steps=2, seed=17)
    source = Trainer(_TinyLogitModel(), config)
    source_state = source.state_dict()
    model = _TinyLogitModel()
    target = Trainer(model, config)
    target.train_microbatch(_BATCH)
    original_grad = model.weight.grad.detach().clone()
    original_weights = model.weight.detach().clone()
    optimizer_loads = []

    def forbidden_optimizer_load(_state):
        optimizer_loads.append(True)
        raise AssertionError("partial accumulation must fail before optimizer restore")

    monkeypatch.setattr(target.optimizer, "load_state_dict", forbidden_optimizer_load)
    with pytest.raises(TrainingStateInvalidError, match="requires a fresh trainer"):
        target.load_state_dict(source_state)

    assert optimizer_loads == []
    assert target.micro_step == 1
    assert target.optimizer_step == 0
    assert target._failure_reason is None
    torch.testing.assert_close(model.weight.grad, original_grad, rtol=0, atol=0)
    torch.testing.assert_close(model.weight, original_weights, rtol=0, atol=0)
    # The rejected pure preflight must not discard a valid accumulation group.
    assert target.train_microbatch(_BATCH).optimizer_stepped is True


def test_direct_restore_rejects_already_committed_trainer_before_optimizer_io(
    monkeypatch,
):
    config = TrainerConfig(max_steps=2, seed=17)
    source_state = Trainer(_TinyLogitModel(), config).state_dict()
    model = _TinyLogitModel()
    target = Trainer(model, config)
    target.train_microbatch(_BATCH)
    committed = model.weight.detach().clone()
    assert target.optimizer_step == 1
    optimizer_loads = []

    def forbidden_optimizer_load(_state):
        optimizer_loads.append(True)
        raise AssertionError("already committed trainer must not accept restore")

    monkeypatch.setattr(target.optimizer, "load_state_dict", forbidden_optimizer_load)
    with pytest.raises(TrainingStateInvalidError, match="requires a fresh trainer"):
        target.load_state_dict(source_state)

    assert optimizer_loads == []
    assert target.optimizer_step == 1
    assert target.micro_step == 1
    assert target._failure_reason is None
    torch.testing.assert_close(model.weight, committed, rtol=0, atol=0)


def test_direct_restore_rejects_fresh_trainer_with_unowned_residual_gradients():
    config = TrainerConfig(max_steps=1, seed=17)
    source_state = Trainer(_TinyLogitModel(), config).state_dict()
    model = _TinyLogitModel()
    target = Trainer(model, config)
    model.weight.grad = torch.full_like(model.weight, 0.75)
    saved_grad = model.weight.grad.detach().clone()

    with pytest.raises(TrainingStateInvalidError, match="pending gradients"):
        target.load_state_dict(source_state)

    torch.testing.assert_close(model.weight.grad, saved_grad, rtol=0, atol=0)
    assert target._failure_reason is None
    fresh = Trainer(_TinyLogitModel(), config)
    fresh.load_state_dict(source_state)
    assert fresh.state_dict().optimizer_step == 0


def test_scheduler_corrupting_weights_after_optimizer_step_cannot_report_success(
    monkeypatch,
):
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    config = TrainerConfig(max_steps=2, warmup_steps=1, scheduler="cosine", seed=17)
    trainer = Trainer(model, config)
    real_scheduler_step = trainer.scheduler.step

    def corrupt_weights_after_scheduler():
        real_scheduler_step()
        model.weight.data.fill_(float("inf"))

    monkeypatch.setattr(trainer.scheduler, "step", corrupt_weights_after_scheduler)
    with pytest.raises(NonFiniteTrainingError, match="non-finite model weights"):
        trainer.train_microbatch(_BATCH)

    assert trainer.optimizer_step == 1  # Physical AdamW update already happened.
    assert trainer._update_incomplete is True
    assert trainer._failure_reason.startswith("optimizer/scheduler update failed")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_zero_grad_hook_corrupting_optimizer_moment_fails_final_update_check(
    monkeypatch,
):
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    actual_zero_grad = trainer.optimizer.zero_grad

    def corrupt_after_zero_grad(*args, **kwargs):
        result = actual_zero_grad(*args, **kwargs)
        trainer.optimizer.state[model.weight]["exp_avg"].fill_(float("nan"))
        return result

    monkeypatch.setattr(trainer.optimizer, "zero_grad", corrupt_after_zero_grad)
    with pytest.raises(NonFiniteTrainingError, match="non-finite state"):
        trainer.train_microbatch(_BATCH)

    assert trainer.optimizer_step == 1
    assert trainer._update_incomplete is True
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_checkpoint_export_refuses_nonfinite_weights_modified_after_valid_step():
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    assert trainer.train_microbatch(_BATCH).optimizer_stepped is True
    model.weight.data.fill_(float("inf"))

    with pytest.raises(NonFiniteTrainingError, match="non-finite model weights"):
        trainer.state_dict()

    assert trainer.optimizer_step == 1
    assert "checkpoint boundary has invalid optimizer" in trainer._failure_reason
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_direct_restore_rejects_nonfinite_adamw_moment_before_clean_status():
    from copy import deepcopy
    from dataclasses import replace

    from twelve_six.training import NonFiniteTrainingError

    config = TrainerConfig(max_steps=2, seed=17)
    source = Trainer(_TinyLogitModel(), config)
    source.train_microbatch(_BATCH)
    snapshot = source.state_dict()
    bad_optimizer = deepcopy(snapshot.optimizer)
    first_state = next(iter(bad_optimizer["state"].values()))
    first_state["exp_avg"].fill_(float("nan"))
    receiver = Trainer(_TinyLogitModel(), config)

    with pytest.raises(NonFiniteTrainingError, match="non-finite state"):
        receiver.load_state_dict(replace(snapshot, optimizer=bad_optimizer))

    assert receiver._failure_reason.startswith("trainer state restore failed")
    assert receiver._update_incomplete is True
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        receiver.state_dict()
    with pytest.raises(TrainingStateInvalidError, match="verified model"):
        receiver.load_state_dict(snapshot)


def test_direct_restore_rejects_nonfinite_optimizer_group_rate_before_mutation():
    from copy import deepcopy
    from dataclasses import replace

    from twelve_six.training import NonFiniteTrainingError

    config = TrainerConfig(max_steps=1, seed=17)
    snapshot = Trainer(_TinyLogitModel(), config).state_dict()
    bad_optimizer = deepcopy(snapshot.optimizer)
    bad_optimizer["param_groups"][0]["lr"] = float("inf")
    receiver = Trainer(_TinyLogitModel(), config)

    with pytest.raises(NonFiniteTrainingError, match="learning rate must be finite"):
        receiver.load_state_dict(replace(snapshot, optimizer=bad_optimizer))

    assert receiver._failure_reason is None
    assert receiver._update_incomplete is False
    assert receiver.optimizer_step == 0
    assert not receiver.optimizer.state
    receiver.load_state_dict(snapshot)
    assert receiver.train_microbatch(_BATCH).optimizer_stepped is True


def test_nonfinite_model_buffer_after_optimizer_step_never_earns_step_credit(
    monkeypatch,
):
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    model.register_buffer("running_statistic", torch.tensor(1.0))
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    original_step = trainer.optimizer.step

    def corrupt_buffer_after_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        model.running_statistic.fill_(float("inf"))
        return result

    monkeypatch.setattr(trainer.optimizer, "step", corrupt_buffer_after_step)
    with pytest.raises(NonFiniteTrainingError, match="non-finite buffer"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer._update_incomplete is True
    assert torch.isinf(model.running_statistic).item()
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_checkpoint_export_rejects_corrupted_persistent_model_buffer():
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    model.register_buffer("running_statistic", torch.tensor(1.0))
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))
    assert trainer.train_microbatch(_BATCH).optimizer_stepped is True
    assert trainer.optimizer_step == 1
    model.running_statistic.fill_(float("nan"))

    with pytest.raises(NonFiniteTrainingError, match="non-finite buffer"):
        trainer.state_dict()

    assert "checkpoint boundary has invalid optimizer" in trainer._failure_reason
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_finite_float_and_integer_model_buffers_allow_normal_checkpoint():
    model = _TinyLogitModel()
    model.register_buffer("running_statistic", torch.tensor(1.0))
    model.register_buffer("completed_batches", torch.tensor(0, dtype=torch.long))
    trainer = Trainer(model, TrainerConfig(max_steps=1, seed=17))

    result = trainer.train_microbatch(_BATCH)

    assert result.optimizer_stepped is True
    assert trainer.optimizer_step == 1
    assert trainer.state_dict().optimizer_step == 1



def test_scheduler_corrupting_model_buffer_poisoned_after_committed_step(
    monkeypatch,
):
    from twelve_six.training import NonFiniteTrainingError

    model = _TinyLogitModel()
    model.register_buffer("running_statistic", torch.tensor(1.0))
    config = TrainerConfig(max_steps=2, warmup_steps=1, scheduler="cosine", seed=17)
    trainer = Trainer(model, config)
    real_scheduler_step = trainer.scheduler.step

    def corrupt_buffer_after_scheduler():
        real_scheduler_step()
        model.running_statistic.fill_(float("nan"))

    monkeypatch.setattr(trainer.scheduler, "step", corrupt_buffer_after_scheduler)
    with pytest.raises(NonFiniteTrainingError, match="non-finite buffer"):
        trainer.train_microbatch(_BATCH)

    assert trainer.optimizer_step == 1  # Optimizer already committed.
    assert trainer._update_incomplete is True
    assert trainer._failure_reason.startswith("optimizer/scheduler update failed")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_direct_restore_rejects_existing_corrupted_model_buffer():
    from twelve_six.training import NonFiniteTrainingError

    config = TrainerConfig(max_steps=1, seed=17)
    source = Trainer(_TinyLogitModel(), config)
    snapshot = source.state_dict()
    model = _TinyLogitModel()
    model.register_buffer("running_statistic", torch.tensor(1.0))
    target = Trainer(model, config)
    model.running_statistic.fill_(float("inf"))

    with pytest.raises(NonFiniteTrainingError, match="non-finite buffer"):
        target.load_state_dict(snapshot)

    assert target._failure_reason.startswith("trainer state restore failed")
    assert target._update_incomplete is True
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        target.state_dict()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("lr", "0.001"),
        ("weight_decay", float("nan")),
        ("weight_decay", -0.1),
        ("weight_decay", "0.01"),
        ("eps", float("inf")),
        ("eps", 0.0),
        ("eps", "1e-8"),
        ("betas", (float("nan"), 0.9)),
        ("betas", (0.9, 1.0)),
        ("betas", ("0.9", "0.999")),
    ],
)
def test_restore_rejects_invalid_adamw_hyperparameters_before_mutation(
    field, invalid,
):
    from copy import deepcopy
    from dataclasses import replace

    from twelve_six.training import NonFiniteTrainingError

    config = TrainerConfig(max_steps=1, seed=17)
    original = Trainer(_TinyLogitModel(), config)
    snapshot = original.state_dict()
    corrupt = deepcopy(snapshot.optimizer)
    corrupt["param_groups"][0][field] = invalid
    receiver = Trainer(_TinyLogitModel(), config)

    label = "learning rate" if field == "lr" else field
    with pytest.raises(NonFiniteTrainingError, match=f"optimizer {label}"):
        receiver.load_state_dict(replace(snapshot, optimizer=corrupt))

    assert receiver.optimizer_step == 0
    assert receiver._update_incomplete is False
    assert receiver._failure_reason is None
    assert not receiver.optimizer.state
    receiver.load_state_dict(snapshot)
    assert receiver.state_dict().optimizer_step == 0


def test_checkpoint_export_rejects_nonfinite_group_weight_decay():
    from twelve_six.training import NonFiniteTrainingError

    trainer = Trainer(_TinyLogitModel(), TrainerConfig(max_steps=1, seed=17))
    assert trainer.train_microbatch(_BATCH).optimizer_stepped is True
    trainer.optimizer.param_groups[0]["weight_decay"] = float("nan")

    with pytest.raises(NonFiniteTrainingError, match="optimizer weight_decay"):
        trainer.state_dict()

    assert trainer._failure_reason.startswith("checkpoint boundary")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)


def test_restore_rejects_nonfinite_scheduler_base_lr():
    from copy import deepcopy
    from dataclasses import replace

    from twelve_six.training import NonFiniteTrainingError

    config = TrainerConfig(max_steps=2, scheduler="cosine", warmup_steps=1, seed=17)
    original = Trainer(_TinyLogitModel(), config)
    original.train_microbatch(_BATCH)
    snapshot = original.state_dict()
    corrupt_scheduler = deepcopy(snapshot.scheduler)
    assert corrupt_scheduler is not None
    corrupt_scheduler["base_lrs"] = [float("inf")]
    receiver = Trainer(_TinyLogitModel(), config)

    with pytest.raises(
        TrainingStateInvalidError,
        match="default scheduler rate differs from configured committed schedule",
    ):
        receiver.load_state_dict(replace(snapshot, scheduler=corrupt_scheduler))

    assert receiver._failure_reason is None
    assert receiver._update_incomplete is False
    assert not receiver.optimizer.state
    receiver.load_state_dict(snapshot)
    assert receiver.state_dict().optimizer_step == snapshot.optimizer_step


def test_scheduler_corrupting_internal_base_lrs_rejects_completed_update(
    monkeypatch,
):
    from twelve_six.training import NonFiniteTrainingError

    config = TrainerConfig(max_steps=2, scheduler="cosine", warmup_steps=1, seed=17)
    trainer = Trainer(_TinyLogitModel(), config)
    real_scheduler_step = trainer.scheduler.step

    def corrupt_scheduler():
        real_scheduler_step()
        trainer.scheduler.base_lrs[0] = float("nan")

    monkeypatch.setattr(trainer.scheduler, "step", corrupt_scheduler)
    with pytest.raises(NonFiniteTrainingError, match="scheduler has non-finite state"):
        trainer.train_microbatch(_BATCH)

    assert trainer.optimizer_step == 1  # Physical step already occurred.
    assert trainer._update_incomplete is True
    assert trainer._failure_reason.startswith("optimizer/scheduler update failed")
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()
