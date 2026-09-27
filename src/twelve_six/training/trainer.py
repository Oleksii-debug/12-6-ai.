"""Reusable PyTorch trainer with explicit numerical-safety and resume contracts."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from .config import TrainerConfig
from .loss import causal_lm_loss, causal_pair_loss
from .numeric_forensics import (
    ActivationHealthProvider,
    AffectedParameters,
    NumericFailureDiagnostics,
    build_numeric_failure_diagnostics,
    model_parameters_are_finite,
    nonfinite_gradient_parameters,
    nonfinite_update_parameters,
)
from .precision import (
    PrecisionRuntime,
    autocast_dtype,
    resolve_precision_runtime,
    validate_master_weight_semantics,
)

Batch = Mapping[str, Tensor]


class NonFiniteTrainingError(FloatingPointError):
    """Raised when a non-finite transition poisons the current Trainer state."""

    def __init__(
        self,
        message: str,
        diagnostics: NumericFailureDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class TrainingStateInvalidError(RuntimeError):
    """Raised when training must restore a verified checkpoint before continuing."""


class CheckpointHookError(RuntimeError):
    """A checkpoint hook failed after an optimizer step was already committed."""


@dataclass(frozen=True, slots=True)
class StepMetrics:
    micro_step: int
    optimizer_step: int
    loss: float
    update_loss: float | None
    learning_rate: float
    grad_norm: float | None
    tokens: int
    optimizer_stepped: bool


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    start_optimizer_step: int
    end_optimizer_step: int
    optimizer_steps_completed: int
    microbatches_consumed: int
    tokens_consumed: int
    final_metrics: StepMetrics | None


@dataclass(frozen=True, slots=True)
class TrainerState:
    """Serializable trainer-owned state; D05 owns durable checkpoint file formats."""

    micro_step: int
    optimizer_step: int
    tokens_seen: int
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None
    scaler: dict[str, Any] | None
    config: dict[str, Any]


def _typed_state_equal(left: Any, right: Any) -> bool:
    """Compare checkpoint metadata without Python numeric/bool equality aliases."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_typed_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_typed_state_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _numerical_state_is_finite(value: Any) -> bool:
    """Recursively reject non-finite behavior-affecting numerical state."""
    if isinstance(value, Tensor):
        if value.is_floating_point() or value.is_complex():
            return bool(torch.isfinite(value.detach()).all().item())
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, complex):
        return math.isfinite(value.real) and math.isfinite(value.imag)
    if isinstance(value, Mapping):
        return all(_numerical_state_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numerical_state_is_finite(item) for item in value)
    return True


def _lr_lambda(config: TrainerConfig):
    def factor(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        if config.scheduler == "constant":
            return 1.0
        if config.scheduler == "linear_warmup":
            return 1.0
        progress_denominator = max(config.max_steps - config.warmup_steps, 1)
        progress = min(max((step - config.warmup_steps) / progress_denominator, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def build_optimizer(model: nn.Module, config: TrainerConfig) -> Optimizer:
    """Construct the S0 default optimizer without owning model architecture."""
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: Optimizer, config: TrainerConfig) -> LRScheduler | None:
    """Build the configured per-optimizer-step schedule."""
    if config.scheduler == "constant" and config.warmup_steps == 0:
        return None
    return LambdaLR(optimizer, lr_lambda=_lr_lambda(config))


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), Tensor):
        return output["logits"]
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    raise TypeError(
        "model output must be a Tensor, mapping['logits'], or object with .logits Tensor"
    )


def _count_training_tokens(
    targets: Tensor,
    *,
    aligned_targets: bool,
    loss_mask: Tensor | None,
    ignore_index: int = -100,
) -> int:
    if aligned_targets:
        valid = targets.ne(ignore_index)
        if loss_mask is not None:
            valid = valid & loss_mask.bool()
        return int(valid.sum().item())
    return int(targets[:, 1:].ne(ignore_index).sum().item())


class Trainer:
    """Small backend-clean trainer for S0 and later stage-specific composition.

    One call to :meth:`train_microbatch` consumes one microbatch. An optimizer update
    happens exactly every ``gradient_accumulation_steps`` calls. :meth:`run` adds a
    boundary-safe reusable loop without owning dataset iteration semantics.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        device: str | torch.device = "cpu",
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        activation_health_provider: ActivationHealthProvider | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.precision_runtime: PrecisionRuntime = resolve_precision_runtime(
            config.precision,
            self.device,
        )
        validate_master_weight_semantics(self.model, self.precision_runtime)

        self.model.to(self.device)
        self._configure_determinism(config)
        self.optimizer = optimizer or build_optimizer(model, config)
        self.scheduler = (
            scheduler if scheduler is not None else build_scheduler(self.optimizer, config)
        )
        self.scaler = self._build_scaler()

        self.micro_step = 0
        self.optimizer_step = 0
        self.tokens_seen = 0
        self._pending_tokens = 0
        self._pending_loss_sum = 0.0
        self._update_incomplete = False
        self._failure_reason: str | None = None
        self._failure_diagnostics: NumericFailureDiagnostics | None = None
        self._activation_health_provider = activation_health_provider
        self.optimizer.zero_grad(set_to_none=True)

    def _configure_determinism(self, config: TrainerConfig) -> None:
        if not hasattr(self, "precision_runtime"):
            self.precision_runtime = resolve_precision_runtime(
                config.precision,
                self.device,
            )
            validate_master_weight_semantics(self.model, self.precision_runtime)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(
            config.deterministic_algorithms,
            warn_only=config.deterministic_warn_only,
        )

    def _build_scaler(self):
        enabled = self.precision_runtime.grad_scaler_enabled
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast_context(self):
        if not self.precision_runtime.autocast_enabled:
            return nullcontext()
        return torch.autocast(
            device_type=self.precision_runtime.device_type,
            dtype=autocast_dtype(self.precision_runtime),
        )

    @property
    def failure_diagnostics(self) -> NumericFailureDiagnostics | None:
        """Return the first bounded forensic record for the current poisoned state."""
        return self._failure_diagnostics

    def _mark_failed(
        self,
        reason: str,
        diagnostics: NumericFailureDiagnostics | None = None,
    ) -> None:
        if self._failure_reason is None:
            self._failure_reason = reason
            self._failure_diagnostics = diagnostics
        self.optimizer.zero_grad(set_to_none=True)

    def _diagnostics(
        self,
        *,
        kind: str,
        batch: Batch,
        batch_tokens: int,
        gradient_norm: float | None,
        gradient_norm_finite: bool | None,
        affected: AffectedParameters,
        attempted_micro_step: int | None = None,
    ) -> NumericFailureDiagnostics:
        return build_numeric_failure_diagnostics(
            kind=kind,  # type: ignore[arg-type]
            micro_step=self.micro_step if attempted_micro_step is None else attempted_micro_step,
            optimizer_step=self.optimizer_step,
            tokens_seen=self.tokens_seen,
            batch_tokens=batch_tokens,
            pending_tokens=self._pending_tokens,
            precision=str(self.config.precision),
            device_type=self.precision_runtime.device_type,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            gradient_norm=gradient_norm,
            gradient_norm_finite=gradient_norm_finite,
            affected=affected,
            batch=batch,
            activation_health_provider=self._activation_health_provider,
        )

    def _raise_nonfinite(
        self,
        reason: str,
        diagnostics: NumericFailureDiagnostics,
    ) -> None:
        self._mark_failed(reason, diagnostics)
        raise NonFiniteTrainingError(reason, diagnostics)

    def _assert_trainable(self) -> None:
        if self._failure_reason is not None:
            raise TrainingStateInvalidError(
                "trainer state is invalid after a failed training transition; "
                f"construct a fresh trainer and restore a verified checkpoint: {self._failure_reason}"
            )
        if self._update_incomplete:
            raise TrainingStateInvalidError(
                "optimizer/scheduler update has ambiguous committed state; "
                "construct a fresh trainer and restore a verified checkpoint"
            )

    def _model_state_is_finite(self) -> bool:
        return _numerical_state_is_finite(self.model.state_dict())

    def _optimizer_state_is_finite(self) -> bool:
        return _numerical_state_is_finite(self.optimizer.state_dict())

    def _prepare_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor | None, bool]:
        if "input_ids" not in batch:
            raise KeyError("batch must contain input_ids")
        if "labels" in batch and "target_ids" in batch:
            raise ValueError("batch must not contain both labels and target_ids")

        input_ids = batch["input_ids"].to(self.device)
        aligned_targets = "target_ids" in batch
        targets = batch.get(
            "target_ids",
            batch.get("labels", batch["input_ids"]),
        ).to(self.device)
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            if not aligned_targets:
                raise ValueError("loss_mask is only valid with already-aligned target_ids")
            loss_mask = loss_mask.to(self.device)

        if input_ids.ndim != 2 or targets.ndim != 2:
            raise ValueError("input_ids and training targets must have shape [batch, time]")
        if input_ids.shape != targets.shape:
            raise ValueError("input_ids and training targets must have identical shape")
        return input_ids, targets, loss_mask, aligned_targets

    def _forward_loss(
        self,
        input_ids: Tensor,
        targets: Tensor,
        *,
        loss_mask: Tensor | None,
        aligned_targets: bool,
    ) -> Tensor:
        with self._autocast_context():
            logits = _extract_logits(self.model(input_ids))
            if aligned_targets:
                return causal_pair_loss(logits, targets, loss_mask=loss_mask)
            return causal_lm_loss(logits, targets)

    def _normalize_gradients_and_norm(self, token_count: int) -> Tensor | None:
        if token_count <= 0:
            raise RuntimeError("optimizer update requires at least one valid target token")
        squared_norm = torch.zeros((), device=self.device)
        found = False
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            found = True
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all().item():
                return None
            grad.div_(token_count)
            squared_norm += torch.sum(grad.float() * grad.float())
        if not found:
            return torch.zeros((), device=self.device)
        return torch.sqrt(squared_norm)

    def train_microbatch(self, batch: Batch) -> StepMetrics:
        """Backpropagate one microbatch and update only at the accumulation boundary."""
        self._assert_trainable()
        if self.optimizer_step >= self.config.max_steps:
            raise RuntimeError("configured max_steps already reached")
        self.model.train()
        input_ids, targets, loss_mask, aligned_targets = self._prepare_batch(batch)
        tokens = _count_training_tokens(
            targets,
            aligned_targets=aligned_targets,
            loss_mask=loss_mask,
        )
        if tokens <= 0:
            raise ValueError("microbatch must contain at least one valid target token")

        loss = self._forward_loss(
            input_ids,
            targets,
            loss_mask=loss_mask,
            aligned_targets=aligned_targets,
        )
        if not torch.isfinite(loss).item():
            attempted_micro_step = self.micro_step + 1
            diagnostics = self._diagnostics(
                kind="loss",
                batch=batch,
                batch_tokens=tokens,
                gradient_norm=None,
                gradient_norm_finite=None,
                affected=nonfinite_update_parameters(self.model, self.optimizer),
                attempted_micro_step=attempted_micro_step,
            )
            self._raise_nonfinite(
                f"non-finite loss at micro_step={attempted_micro_step}", diagnostics
            )

        try:
            self.scaler.scale(loss * tokens).backward()
        except RuntimeError:
            self._mark_failed(f"backward failed at micro_step={self.micro_step + 1}")
            raise

        self.micro_step += 1
        self.tokens_seen += tokens
        self._pending_tokens += tokens
        self._pending_loss_sum += float(loss.detach().float().item()) * tokens

        should_step = self.micro_step % self.config.gradient_accumulation_steps == 0
        grad_norm_value: float | None = None
        update_loss: float | None = None
        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        if should_step:
            self._update_incomplete = True
            try:
                self.scaler.unscale_(self.optimizer)
                raw_grad_norm = self._normalize_gradients_and_norm(self._pending_tokens)
                if raw_grad_norm is None or not torch.isfinite(raw_grad_norm).item():
                    diagnostics = self._diagnostics(
                        kind="gradient",
                        batch=batch,
                        batch_tokens=tokens,
                        gradient_norm=None,
                        gradient_norm_finite=False,
                        affected=nonfinite_gradient_parameters(self.model),
                    )
                    self._raise_nonfinite(
                        f"non-finite gradient at micro_step={self.micro_step}", diagnostics
                    )

                grad_norm_value = float(raw_grad_norm.item())
                update_loss = self._pending_loss_sum / self._pending_tokens

                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )

                self.scaler.step(self.optimizer)
                if not model_parameters_are_finite(self.model):
                    diagnostics = self._diagnostics(
                        kind="update",
                        batch=batch,
                        batch_tokens=tokens,
                        gradient_norm=grad_norm_value,
                        gradient_norm_finite=True,
                        affected=nonfinite_update_parameters(self.model, self.optimizer),
                    )
                    self._raise_nonfinite(
                        f"non-finite update at micro_step={self.micro_step}", diagnostics
                    )
                if not self._optimizer_state_is_finite():
                    diagnostics = self._diagnostics(
                        kind="update",
                        batch=batch,
                        batch_tokens=tokens,
                        gradient_norm=grad_norm_value,
                        gradient_norm_finite=True,
                        affected=nonfinite_update_parameters(self.model, self.optimizer),
                    )
                    self._raise_nonfinite(
                        f"non-finite optimizer state at micro_step={self.micro_step}",
                        diagnostics,
                    )

                self.optimizer_step += 1
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                if not self._optimizer_state_is_finite():
                    diagnostics = self._diagnostics(
                        kind="update",
                        batch=batch,
                        batch_tokens=tokens,
                        gradient_norm=grad_norm_value,
                        gradient_norm_finite=True,
                        affected=nonfinite_update_parameters(self.model, self.optimizer),
                    )
                    self._raise_nonfinite(
                        f"non-finite optimizer/scheduler state at micro_step={self.micro_step}",
                        diagnostics,
                    )
            except Exception:
                self._mark_failed(
                    f"optimizer/scheduler update failed at micro_step={self.micro_step}"
                )
                raise
            self._pending_tokens = 0
            self._pending_loss_sum = 0.0
            self._update_incomplete = False

        return StepMetrics(
            micro_step=self.micro_step,
            optimizer_step=self.optimizer_step,
            loss=float(loss.detach().float().item()),
            update_loss=update_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm_value,
            tokens=tokens,
            optimizer_stepped=should_step,
        )

    def run(
        self,
        batches: Iterable[Batch],
        *,
        on_metrics: Callable[[StepMetrics], None] | None = None,
        on_checkpoint: Callable[[Trainer, StepMetrics], None] | None = None,
        checkpoint_every_steps: int | None = None,
    ) -> TrainingRunResult:
        """Train from the current state until ``config.max_steps`` optimizer steps.

        The iterable controls data order/epochs and must contain enough microbatches.
        Checkpoint hooks run only after committed optimizer/scheduler steps. A final
        hook is emitted at ``max_steps`` even when it is off cadence. Hook failure is
        explicit because the optimizer step must not be blindly replayed.
        """
        self.assert_checkpoint_safe()
        if checkpoint_every_steps is not None and checkpoint_every_steps <= 0:
            raise ValueError("checkpoint_every_steps must be > 0")
        if checkpoint_every_steps is not None and on_checkpoint is None:
            raise ValueError("checkpoint_every_steps requires on_checkpoint")

        start_step = self.optimizer_step
        start_tokens = self.tokens_seen
        consumed = 0
        final_metrics: StepMetrics | None = None

        for batch in batches:
            if self.optimizer_step >= self.config.max_steps:
                break
            metrics = self.train_microbatch(batch)
            consumed += 1
            final_metrics = metrics
            if on_metrics is not None:
                on_metrics(metrics)

            if metrics.optimizer_stepped and on_checkpoint is not None:
                on_cadence = (
                    checkpoint_every_steps is not None
                    and metrics.optimizer_step % checkpoint_every_steps == 0
                )
                is_final = metrics.optimizer_step == self.config.max_steps
                if on_cadence or is_final:
                    try:
                        on_checkpoint(self, metrics)
                    except Exception as exc:
                        raise CheckpointHookError(
                            "checkpoint hook failed after committed "
                            f"optimizer_step={metrics.optimizer_step}; do not replay blindly"
                        ) from exc

        if self.optimizer_step < self.config.max_steps:
            self.assert_checkpoint_safe()
            raise RuntimeError(
                "batch iterable exhausted before max_steps: "
                f"optimizer_step={self.optimizer_step}, max_steps={self.config.max_steps}"
            )
        self.assert_checkpoint_safe()
        return TrainingRunResult(
            start_optimizer_step=start_step,
            end_optimizer_step=self.optimizer_step,
            optimizer_steps_completed=self.optimizer_step - start_step,
            microbatches_consumed=consumed,
            tokens_consumed=self.tokens_seen - start_tokens,
            final_metrics=final_metrics,
        )

    def assert_accumulation_boundary(self) -> None:
        """Require no incomplete gradient-accumulation group."""
        remainder = self.micro_step % self.config.gradient_accumulation_steps
        if remainder:
            raise RuntimeError(
                "training stopped mid-accumulation: "
                f"{remainder}/{self.config.gradient_accumulation_steps} microbatches pending"
            )

    def assert_checkpoint_safe(self) -> None:
        """Require committed, numerically finite state before checkpoint publication."""
        self._assert_trainable()
        if self.optimizer_step > self.config.max_steps:
            raise RuntimeError("optimizer_step exceeds configured max_steps")
        self.assert_accumulation_boundary()
        expected_micro_steps = self.optimizer_step * self.config.gradient_accumulation_steps
        if self.micro_step != expected_micro_steps:
            raise RuntimeError(
                "trainer has consumed but uncommitted microbatches: "
                f"micro_step={self.micro_step}, committed_expected={expected_micro_steps}"
            )
        if self._pending_tokens != 0 or self._pending_loss_sum != 0.0:
            raise RuntimeError("trainer has pending accumulation statistics")
        if not self._model_state_is_finite():
            reason = "non-finite model state blocks checkpoint publication"
            self._mark_failed(reason)
            raise NonFiniteTrainingError(reason)
        if not self._optimizer_state_is_finite():
            reason = "non-finite optimizer state blocks checkpoint publication"
            self._mark_failed(reason)
            raise NonFiniteTrainingError(reason)

    def state_dict(self) -> TrainerState:
        """Return checkpoint-safe trainer state only after committed optimizer steps."""
        self.assert_checkpoint_safe()
        return TrainerState(
            micro_step=self.micro_step,
            optimizer_step=self.optimizer_step,
            tokens_seen=self.tokens_seen,
            optimizer=copy.deepcopy(self.optimizer.state_dict()),
            scheduler=(
                None if self.scheduler is None else copy.deepcopy(self.scheduler.state_dict())
            ),
            scaler=None if self.scaler is None else copy.deepcopy(self.scaler.state_dict()),
            config=asdict(self.config),
        )

    def load_state_dict(self, state: TrainerState | Mapping[str, Any]) -> None:
        """Restore checkpoint state into a clean trainer instance.

        A trainer that has entered a poisoned or ambiguous state cannot be repaired
        in place because trainer-only state cannot prove that model weights were also
        restored. Construct a fresh Trainer around the verified checkpoint model and
        then load the trainer state. Component restore including gradient cleanup is
        transactional: a rejected resume either restores the exact prior component
        and gradient state or poisons the Trainer so it cannot continue training.
        """
        if self._failure_reason is not None or self._update_incomplete:
            raise TrainingStateInvalidError(
                "failed trainer cannot be repaired in place; construct a fresh trainer "
                "and restore the verified model + trainer checkpoint"
            )
        if not self._model_state_is_finite() or not self._optimizer_state_is_finite():
            reason = "non-finite live Trainer state before restore"
            self._mark_failed(reason)
            raise TrainingStateInvalidError(
                "non-finite live Trainer cannot be repaired in place; construct a fresh "
                "Trainer and restore a verified checkpoint"
            )
        if isinstance(state, Mapping):
            state = TrainerState(**state)

        counters = {
            "micro_step": state.micro_step,
            "optimizer_step": state.optimizer_step,
            "tokens_seen": state.tokens_seen,
        }
        for name, value in counters.items():
            if type(value) is not int:
                raise ValueError(f"trainer checkpoint {name} must be an exact integer")
        if not _typed_state_equal(state.config, asdict(self.config)):
            raise ValueError("trainer config mismatch; refusing unsafe resume")
        if state.micro_step < 0 or state.optimizer_step < 0 or state.tokens_seen < 0:
            raise ValueError("trainer counters must be non-negative")
        expected_micro_steps = state.optimizer_step * self.config.gradient_accumulation_steps
        if state.micro_step != expected_micro_steps:
            raise ValueError(
                "checkpoint is not at a complete committed accumulation boundary: "
                f"micro_step={state.micro_step}, expected={expected_micro_steps}"
            )
        if state.optimizer_step > self.config.max_steps:
            raise ValueError("checkpoint optimizer_step exceeds configured max_steps")
        if (state.scheduler is None) != (self.scheduler is None):
            raise ValueError("scheduler state/config mismatch")
        if (state.scaler is None) != (self.scaler is None):
            raise ValueError("scaler state/runtime mismatch")
        if not _numerical_state_is_finite(state.optimizer):
            raise NonFiniteTrainingError(
                "trainer checkpoint optimizer state contains NaN/Inf"
            )

        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        scheduler_before = (
            None if self.scheduler is None else copy.deepcopy(self.scheduler.state_dict())
        )
        scaler_before = None if self.scaler is None else copy.deepcopy(self.scaler.state_dict())
        gradients_before: list[tuple[Tensor, Tensor | None]] = []
        seen_parameters: set[int] = set()
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                parameter_id = id(parameter)
                if parameter_id in seen_parameters:
                    continue
                seen_parameters.add(parameter_id)
                gradient = parameter.grad
                gradients_before.append(
                    (
                        parameter,
                        None if gradient is None else gradient.detach().clone(),
                    )
                )

        try:
            self.optimizer.load_state_dict(state.optimizer)
            if not self._optimizer_state_is_finite():
                raise NonFiniteTrainingError(
                    "restored optimizer state contains NaN/Inf"
                )
            if not self._model_state_is_finite():
                raise NonFiniteTrainingError(
                    "live model state contains NaN/Inf during trainer restore"
                )
            if self.scheduler is not None and state.scheduler is not None:
                self.scheduler.load_state_dict(state.scheduler)
            if self.scaler is not None and state.scaler is not None:
                self.scaler.load_state_dict(state.scaler)
            self.optimizer.zero_grad(set_to_none=True)
        except Exception as restore_error:
            rollback_errors: list[BaseException] = []
            # Rollback must catch every failure to prove or deny a clean restore.
            try:
                self.optimizer.load_state_dict(optimizer_before)
            except Exception as rollback_error:  # noqa: BLE001
                rollback_errors.append(rollback_error)
            if self.scheduler is not None and scheduler_before is not None:
                try:
                    self.scheduler.load_state_dict(scheduler_before)
                except Exception as rollback_error:  # noqa: BLE001
                    rollback_errors.append(rollback_error)
            if self.scaler is not None and scaler_before is not None:
                try:
                    self.scaler.load_state_dict(scaler_before)
                except Exception as rollback_error:  # noqa: BLE001
                    rollback_errors.append(rollback_error)
            for parameter, gradient in gradients_before:
                try:
                    parameter.grad = gradient
                except Exception as rollback_error:  # noqa: BLE001
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                self._failure_reason = "trainer state restore rollback failed"
                raise TrainingStateInvalidError(
                    "trainer state restore failed and rollback could not prove a clean state; "
                    "construct a fresh trainer and restore a verified checkpoint"
                ) from restore_error
            raise

        self.micro_step = state.micro_step
        self.optimizer_step = state.optimizer_step
        self.tokens_seen = state.tokens_seen
        self._pending_tokens = 0
        self._pending_loss_sum = 0.0
        self._update_incomplete = False
        self._failure_reason = None
        self._failure_diagnostics = None
