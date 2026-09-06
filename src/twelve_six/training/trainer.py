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
from .precision import (
    PrecisionRuntime,
    autocast_dtype,
    resolve_precision_runtime,
    validate_master_weight_semantics,
)

Batch = Mapping[str, Tensor]


class NonFiniteTrainingError(FloatingPointError):
    """Raised before an unsafe optimizer update when training becomes non-finite."""


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

    def _mark_failed(self, reason: str) -> None:
        if self._failure_reason is None:
            self._failure_reason = reason
        self.optimizer.zero_grad(set_to_none=True)

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
                loss = causal_pair_loss(logits, targets, loss_mask=loss_mask)
            else:
                loss = causal_lm_loss(logits, targets)
        if not torch.isfinite(loss).item():
            reason = f"non-finite loss at micro_step={self.micro_step + 1}"
            self._mark_failed(reason)
            raise NonFiniteTrainingError(reason)
        return loss

    def _normalize_gradients_and_norm(self, token_count: int) -> Tensor:
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
                reason = f"non-finite gradient at micro_step={self.micro_step}"
                self._mark_failed(reason)
                raise NonFiniteTrainingError(reason)
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
                grad_norm_value = float(raw_grad_norm.item())
                update_loss = self._pending_loss_sum / self._pending_tokens

                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )

                self.scaler.step(self.optimizer)
                self.optimizer_step += 1
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
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
        """Require all consumed microbatches to belong to committed optimizer steps."""
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
        then load the trainer state.
        """
        if self._failure_reason is not None or self._update_incomplete:
            raise TrainingStateInvalidError(
                "failed trainer cannot be repaired in place; construct a fresh trainer "
                "and restore the verified model + trainer checkpoint"
            )
        if isinstance(state, Mapping):
            state = TrainerState(**state)

        if state.config != asdict(self.config):
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

        self.optimizer.load_state_dict(state.optimizer)
        if (state.scheduler is None) != (self.scheduler is None):
            raise ValueError("scheduler state/config mismatch")
        if self.scheduler is not None and state.scheduler is not None:
            self.scheduler.load_state_dict(state.scheduler)
        if state.scaler is not None:
            self.scaler.load_state_dict(state.scaler)

        self.micro_step = state.micro_step
        self.optimizer_step = state.optimizer_step
        self.tokens_seen = state.tokens_seen
        self._pending_tokens = 0
        self._pending_loss_sum = 0.0
        self._update_incomplete = False
        self._failure_reason = None
        self.optimizer.zero_grad(set_to_none=True)
