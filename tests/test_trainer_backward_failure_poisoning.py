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
    with pytest.raises(ValueError, match="convert string to float"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 2  # Backward happened but no update was committed.
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 4
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

    with pytest.raises(NonFiniteTrainingError, match="learning rate must be finite"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()
