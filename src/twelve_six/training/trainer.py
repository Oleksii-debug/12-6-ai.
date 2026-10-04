"""Reusable PyTorch trainer with explicit numerical-safety and resume contracts."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from .config import TrainerConfig
from .loss import causal_lm_loss, causal_pair_loss

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
        self.model.to(self.device)

        self._configure_determinism(config)
        self.optimizer = optimizer or build_optimizer(model, config)
        self._require_optimizer_parameter_coverage()
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

    @staticmethod
    def _configure_determinism(config: TrainerConfig) -> None:
        random.seed(config.seed)
        # NumPy's legacy global RandomState accepts only 32-bit seeds.
        np.random.seed(config.seed % (2 ** 32))
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(
            config.deterministic_algorithms,
            warn_only=config.deterministic_warn_only,
        )

    def _require_optimizer_parameter_coverage(self) -> None:
        """Require the optimizer to own every trainable model parameter exactly once."""
        model_parameters = tuple(self.model.parameters())
        model_ids = {id(parameter) for parameter in model_parameters}
        trainable_ids = {
            id(parameter) for parameter in model_parameters if parameter.requires_grad
        }
        if not trainable_ids:
            raise ValueError("model has no trainable parameters for optimizer")
        optimizer_ids: list[int] = []
        for group in self.optimizer.param_groups:
            parameters = group.get("params")
            if not isinstance(parameters, (list, tuple)):
                raise TypeError("optimizer group must contain a concrete parameter sequence")
            for parameter in parameters:
                if not isinstance(parameter, Tensor) or id(parameter) not in model_ids:
                    raise ValueError("optimizer contains a parameter not owned by the model")
                optimizer_ids.append(id(parameter))
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError("optimizer contains duplicate parameter assignments")
        if not trainable_ids.issubset(optimizer_ids):
            raise ValueError("optimizer omits trainable model parameters")

    def _optimizer_parameter_name_groups(self) -> list[list[str]]:
        """Bind optimizer slots to first canonical model names, including tied weights."""
        self._require_optimizer_parameter_coverage()
        by_id = {id(parameter): name for name, parameter in self.model.named_parameters()}
        ordered: list[list[str]] = []
        for group in self.optimizer.param_groups:
            names: list[str] = []
            for parameter in group["params"]:
                name = by_id.get(id(parameter))
                if name is None:
                    raise ValueError(
                        "optimizer parameter has no stable named model identity"
                    )
                names.append(name)
            ordered.append(names)
        return ordered

    def _require_optimizer_state_parameter_order(self, state: Any) -> None:
        """Refuse positional state remapping even between same-shaped parameters."""
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint optimizer state must be a mapping")
        source_groups = state.get("param_groups")
        expected = self._optimizer_parameter_name_groups()
        if not isinstance(source_groups, list) or len(source_groups) != len(expected):
            raise ValueError("checkpoint optimizer parameter-name group count differs")
        next_id = 0
        for index, (saved_group, names) in enumerate(
            zip(source_groups, expected, strict=True)
        ):
            if (
                not isinstance(saved_group, Mapping)
                or not isinstance(saved_group.get("param_names"), list)
                or saved_group["param_names"] != names
            ):
                raise ValueError(
                    "checkpoint optimizer parameter order/identity differs "
                    f"in group {index}; legacy unnamed optimizer state is not exact-resumable"
                )
            # PyTorch assigns unique parameters ordinal IDs in optimizer-group
            # order. Derive this layout from the live parameter groups already
            # checked above; state_dict() may have harmful export side effects.
            saved_ids = saved_group.get("params")
            canonical_ids = list(range(next_id, next_id + len(names)))
            next_id += len(names)
            if (
                not isinstance(saved_ids, list)
                or len(saved_ids) != len(canonical_ids)
                or any(
                    type(saved_id) is not int or saved_id != canonical_id
                    for saved_id, canonical_id in zip(
                        saved_ids, canonical_ids, strict=True
                    )
                )
            ):
                raise ValueError(
                    f"checkpoint optimizer parameter ID/order differs in group {index}"
                )
        # State-map keys are the same positional IDs, not arbitrary values
        # that merely compare equal (False == 0 and 0.0 == 0 in Python).
        # A partial state map is valid for a fresh optimizer; every present
        # entry must nevertheless bind to a canonical live parameter slot.
        saved_state = state.get("state")
        if not isinstance(saved_state, Mapping) or any(
            type(saved_id) is not int or not 0 <= saved_id < next_id
            for saved_id in saved_state
        ):
            raise ValueError("checkpoint optimizer state parameter ID is noncanonical")

    def _require_no_residual_model_gradients(self) -> None:
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("completed optimizer step left residual model gradients")

    def _require_finite_committed_update(self) -> None:
        """Reject optimizer corruption before crediting an optimizer transition."""
        for parameter in self.model.parameters():
            if not torch.isfinite(parameter.detach()).all().item():
                raise NonFiniteTrainingError(
                    f"optimizer produced non-finite model weights at micro_step={self.micro_step}"
                )
        # Buffers are durable model state too (for example normalization
        # statistics). They can be corrupted by forward/scheduler hooks even
        # when every optimizer-managed parameter and moment remains finite.
        for buffer in self.model.buffers():
            if (buffer.is_floating_point() or buffer.is_complex()) and not (
                torch.isfinite(buffer.detach()).all().item()
            ):
                raise NonFiniteTrainingError(
                    f"model contains non-finite buffer at micro_step={self.micro_step}"
                )
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, Tensor) and not torch.isfinite(value).all().item():
                    raise NonFiniteTrainingError(
                        f"optimizer produced non-finite state at micro_step={self.micro_step}"
                    )

    def _require_safe_optimizer_hyperparameters(self) -> None:
        """Validate all group hyperparameters, not only the reported group LR."""
        for group in self.optimizer.param_groups:
            for field in ("lr", "weight_decay", "eps"):
                if field not in group:
                    continue  # Other injected optimizer families may omit these fields.
                value = group[field]
                label = "learning rate" if field == "lr" else field
                if isinstance(value, bool) or not math.isfinite(float(value)):
                    raise NonFiniteTrainingError(
                        f"optimizer {label} must be finite and >= 0"
                    )
                if float(value) < 0 or (field == "eps" and float(value) == 0):
                    raise NonFiniteTrainingError(
                        f"optimizer {label} must be finite and >= 0"
                    )
            if "betas" in group:
                betas = group["betas"]
                if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                    raise NonFiniteTrainingError("optimizer betas must be valid")
                for beta in betas:
                    if isinstance(beta, bool) or not math.isfinite(float(beta)):
                        raise NonFiniteTrainingError("optimizer betas must be valid")
                    if not 0 <= float(beta) < 1:
                        raise NonFiniteTrainingError("optimizer betas must be valid")

    @staticmethod
    def _require_finite_state_tree(value: Any, label: str) -> None:
        """Reject nonfinite numeric leaves in nested scheduler/scaler state."""
        if isinstance(value, Tensor):
            if (value.is_floating_point() or value.is_complex()) and not (
                torch.isfinite(value).all().item()
            ):
                raise NonFiniteTrainingError(f"{label} has non-finite state")
        elif isinstance(value, Mapping):
            for child in value.values():
                Trainer._require_finite_state_tree(child, label)
        elif isinstance(value, (list, tuple)):
            for child in value:
                Trainer._require_finite_state_tree(child, label)
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(value)
        ):
            raise NonFiniteTrainingError(f"{label} has non-finite state")

    def _require_finite_auxiliary_state(self) -> None:
        # The optimizer or scheduler may have changed groups after the
        # pre-backward check; a completed step must remain checkpoint-safe.
        self._require_optimizer_parameter_coverage()
        self._require_safe_optimizer_hyperparameters()
        self._require_finite_state_tree(self.scaler.state_dict(), "gradient scaler")
        if self.scheduler is not None:
            self._require_finite_state_tree(self.scheduler.state_dict(), "scheduler")

    def _build_scaler(self):
        enabled = self.config.precision == "fp16" and self.device.type == "cuda"
        if self.config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires a CUDA device; use fp32 or bf16 on CPU")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast_context(self):
        if self.config.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _mark_failed(self, reason: str) -> None:
        if self._failure_reason is not None:
            return
        self._failure_reason = reason
        cleanup_faults: list[str] = []
        # An injected optimizer may have swapped or dropped parameter groups.
        # Clear model-owned gradients independently, then best-effort clear any
        # optimizer-owned state without replacing the primary exception.
        for owner, label in (
            (self.model, "model gradient cleanup failed"),
            (self.optimizer, "gradient cleanup failed"),
        ):
            try:
                owner.zero_grad(set_to_none=True)
            except BaseException as cleanup_error:  # noqa: BLE001 - keep interrupt-safe cleanup
                cleanup_faults.append(f"{label}: {type(cleanup_error).__name__}")
        if cleanup_faults:
            self._failure_reason = f"{reason}; " + "; ".join(cleanup_faults)

    def _require_deterministic_policy(self) -> None:
        """Fail closed when another trainer or callback changed global torch mode."""
        if (
            torch.are_deterministic_algorithms_enabled()
            != self.config.deterministic_algorithms
            or torch.is_deterministic_algorithms_warn_only_enabled()
            != self.config.deterministic_warn_only
        ):
            raise TrainingStateInvalidError(
                "live PyTorch deterministic policy disagrees with trainer configuration; "
                "restore a verified checkpoint before continuing"
            )

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
        self._require_deterministic_policy()

    def _prepare_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor | None, bool]:
        if "input_ids" not in batch:
            raise KeyError("batch must contain input_ids")
        if "labels" in batch and "target_ids" in batch:
            raise ValueError("batch must not contain both labels and target_ids")

        raw_inputs = batch["input_ids"]
        aligned_targets = "target_ids" in batch
        raw_targets = batch.get("target_ids", batch.get("labels", raw_inputs))
        raw_mask = batch.get("loss_mask")
        if raw_mask is not None and not aligned_targets:
            raise ValueError("loss_mask is only valid with already-aligned target_ids")
        if raw_inputs.ndim != 2 or raw_targets.ndim != 2:
            raise ValueError("input_ids and training targets must have shape [batch, time]")
        if raw_inputs.shape != raw_targets.shape:
            raise ValueError("input_ids and training targets must have identical shape")
        if raw_mask is not None and raw_mask.shape != raw_targets.shape:
            raise ValueError("loss_mask must match target_ids shape")

        try:
            input_ids = raw_inputs.to(self.device)
            targets = raw_targets.to(self.device)
            loss_mask = None if raw_mask is None else raw_mask.to(self.device)
        except BaseException:
            # Transfers can fail after asynchronous device activity. A pending
            # accumulation group is no longer safe to retry in-place.
            self._mark_failed(f"batch transfer failed at micro_step={self.micro_step + 1}")
            raise
        return input_ids, targets, loss_mask, aligned_targets

    def _forward_loss(
        self,
        input_ids: Tensor,
        targets: Tensor,
        *,
        loss_mask: Tensor | None,
        aligned_targets: bool,
    ) -> Tensor:
        try:
            with self._autocast_context():
                logits = _extract_logits(self.model(input_ids))
                # A forward hook may change process-global torch policy.
                # Refuse to compute loss or backpropagate under that drift.
                self._require_deterministic_policy()
                if aligned_targets:
                    loss = causal_pair_loss(logits, targets, loss_mask=loss_mask)
                else:
                    loss = causal_lm_loss(logits, targets)
            if not torch.isfinite(loss).item():
                reason = f"non-finite loss at micro_step={self.micro_step + 1}"
                self._mark_failed(reason)
                raise NonFiniteTrainingError(reason)
        except BaseException:
            # Forward, loss or device synchronization may have changed model
            # buffers/RNG; an earlier microbatch can have pending gradients.
            if self._failure_reason is None:
                self._mark_failed(f"forward/loss failed at micro_step={self.micro_step + 1}")
            raise
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
            # Backward can succeed through tensors not owned by the model. Such
            # a step would advance optimizer/exposure accounting without even
            # one model-parameter gradient and must fail closed.
            raise RuntimeError("optimizer update has no model-parameter gradients")
        gradient_norm = torch.sqrt(squared_norm)
        if not torch.isfinite(gradient_norm).item():
            raise NonFiniteTrainingError(
                f"non-finite gradient norm at micro_step={self.micro_step}"
            )
        return gradient_norm

    def train_microbatch(self, batch: Batch) -> StepMetrics:
        """Backpropagate one microbatch and update only at the accumulation boundary."""
        self._assert_trainable()
        if self.optimizer_step >= self.config.max_steps:
            raise RuntimeError("configured max_steps already reached")
        try:
            self._require_optimizer_parameter_coverage()
        except BaseException:
            self._mark_failed("optimizer parameter coverage changed before microbatch")
            raise
        input_ids, targets, loss_mask, aligned_targets = self._prepare_batch(batch)
        try:
            tokens = _count_training_tokens(
                targets,
                aligned_targets=aligned_targets,
                loss_mask=loss_mask,
            )
        except BaseException:
            # A device-side reduction/synchronization can fail after successful
            # transfers; do not reuse earlier accumulated gradients afterward.
            self._mark_failed(f"target accounting failed at micro_step={self.micro_step + 1}")
            raise
        if tokens <= 0:
            raise ValueError("microbatch must contain at least one valid target token")

        try:
            self.model.train()
            # A custom train-mode hook may have changed process-global policy.
            self._require_deterministic_policy()
        except BaseException:
            # Custom train-mode hooks can mutate buffers or consume RNG before
            # failing; prior accumulated gradients must not be replayed.
            self._mark_failed(f"train-mode transition failed at micro_step={self.micro_step + 1}")
            raise

        loss = self._forward_loss(
            input_ids,
            targets,
            loss_mask=loss_mask,
            aligned_targets=aligned_targets,
        )
        # A loss callback can change torch's global policy after model.forward.
        # Check before backward, before gradients can be accumulated.
        try:
            self._require_deterministic_policy()
        except BaseException:
            self._mark_failed(
                f"deterministic policy drift before backward at micro_step={self.micro_step + 1}"
            )
            raise
        try:
            scaled_loss = self.scaler.scale(loss * tokens)
            # A scaler hook may drift global policy after the loss-side guard.
            # Do not run backward on any tensor under a mismatched mode.
            self._require_deterministic_policy()
            scaled_loss.backward()
        except BaseException:
            # Autograd may raise non-RuntimeError exceptions or be interrupted after
            # partially accumulating gradients. A retry requires verified recovery.
            self._mark_failed(f"backward failed at micro_step={self.micro_step + 1}")
            raise

        try:
            # Backward hooks can switch global policy after the entry guard.
            self._require_deterministic_policy()
            observed_loss = float(loss.detach().float().item())
            self.micro_step += 1
            self.tokens_seen += tokens
            self._pending_tokens += tokens
            self._pending_loss_sum += observed_loss * tokens

            should_step = self.micro_step % self.config.gradient_accumulation_steps == 0
            grad_norm_value: float | None = None
            update_loss: float | None = None
            # A custom optimizer can have distinct schedules per group. Never
            # validate only the first group while another can write NaN weights.
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self._require_safe_optimizer_hyperparameters()
        except BaseException:
            # Backward already ran; do not allow a partial accounting transition
            # or an interrupted device synchronization to reuse these gradients.
            self._mark_failed(f"post-backward accounting failed at micro_step={self.micro_step}")
            raise

        if should_step:
            self._update_incomplete = True
            try:
                # Forward/backward hooks may have modified external optimizer
                # groups after entry preflight. Never count a phantom update.
                self._require_optimizer_parameter_coverage()
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

                self._require_deterministic_policy()
                self.scaler.step(self.optimizer)
                self._require_deterministic_policy()
                # A finite gradient and finite LR do not guarantee a finite
                # AdamW update (e.g. weight-decay overflow). The update may
                # already have mutated tensors, but must never earn step credit.
                self._require_finite_committed_update()
                self.optimizer_step += 1
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                # An effectful scheduler or scaler can corrupt the NEXT step's
                # state after the finite optimizer update. Reject that now.
                self._require_finite_auxiliary_state()
                # Effectful zero_grad/scheduler hooks may modify parameters
                # or optimizer moments *after* the first post-step check.
                self._require_finite_committed_update()
                # A custom optimizer may silently ignore zero_grad or swap
                # groups inside step(). Never expose that as a clean boundary.
                self._require_no_residual_model_gradients()
                self._require_deterministic_policy()
            except BaseException:
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
            loss=observed_loss,
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

        if self.optimizer_step == self.config.max_steps:
            # Already complete: even constructing a custom iterable may touch
            # data/RNG. Do not access it after the authorized run boundary.
            return TrainingRunResult(start_step, start_step, 0, 0, 0, None)

        try:
            iterator = iter(batches)
        except BaseException:
            self._mark_failed("batch iterator construction failed")
            raise
        while self.optimizer_step < self.config.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            except BaseException:
                # A failing source may have consumed bytes, advanced its cursor
                # or changed RNG before raising. In-place retry is not safe.
                self._mark_failed("batch iterator failed after possible cursor advancement")
                raise
            metrics = self.train_microbatch(batch)
            consumed += 1
            final_metrics = metrics
            if on_metrics is not None:
                try:
                    on_metrics(metrics)
                except BaseException as exc:
                    # The batch has already been consumed. The optimizer may
                    # also have committed, and the checkpoint hook has not run.
                    self._mark_failed(
                        f"metrics hook failed after micro_step={metrics.micro_step}, "
                        f"optimizer_step={metrics.optimizer_step}"
                    )
                    if isinstance(exc, Exception):
                        raise CheckpointHookError(
                            "metrics hook failed after consumed microbatch; "
                            "restore a verified checkpoint before retry"
                        ) from exc
                    raise

            if metrics.optimizer_stepped and on_checkpoint is not None:
                on_cadence = (
                    checkpoint_every_steps is not None
                    and metrics.optimizer_step % checkpoint_every_steps == 0
                )
                is_final = metrics.optimizer_step == self.config.max_steps
                if on_cadence or is_final:
                    try:
                        on_checkpoint(self, metrics)
                    except BaseException as exc:
                        self._mark_failed(
                            f"checkpoint hook failed after optimizer_step={metrics.optimizer_step}"
                        )
                        if isinstance(exc, Exception):
                            raise CheckpointHookError(
                                "checkpoint hook failed after committed "
                                f"optimizer_step={metrics.optimizer_step}; do not replay blindly"
                            ) from exc
                        raise

        if self.optimizer_step < self.config.max_steps:
            try:
                self.assert_checkpoint_safe()
            except RuntimeError:
                # An exhausted source cannot complete this accumulation group.
                # Pending gradients do not belong to a committed checkpoint and
                # cannot be reattached to an arbitrary successor data iterator.
                self._mark_failed(
                    f"batch iterable exhausted mid-accumulation at micro_step={self.micro_step}"
                )
                raise
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
        # A normal mid-accumulation checkpoint attempt must remain retryable:
        # its gradients are legitimately pending and no state was exported.
        self.assert_accumulation_boundary()
        expected_micro_steps = self.optimizer_step * self.config.gradient_accumulation_steps
        if self.micro_step != expected_micro_steps:
            raise RuntimeError(
                "trainer has consumed but uncommitted microbatches: "
                f"micro_step={self.micro_step}, committed_expected={expected_micro_steps}"
            )
        if self._pending_tokens != 0 or self._pending_loss_sum != 0.0:
            raise RuntimeError("trainer has pending accumulation statistics")
        try:
            self._require_finite_auxiliary_state()
            self._require_finite_committed_update()
            self._require_no_residual_model_gradients()
            self._require_deterministic_policy()
        except BaseException:
            self._mark_failed("checkpoint boundary has invalid optimizer or residual gradients")
            raise

    def state_dict(self) -> TrainerState:
        """Return checkpoint-safe trainer state only after committed optimizer steps."""
        self.assert_checkpoint_safe()
        try:
            optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            saved_groups = optimizer_state.get("param_groups")
            name_groups = self._optimizer_parameter_name_groups()
            if not isinstance(saved_groups, list) or len(saved_groups) != len(name_groups):
                raise TrainingStateInvalidError(
                    "optimizer state cannot bind named parameter groups"
                )
            for saved_group, names in zip(saved_groups, name_groups, strict=True):
                if not isinstance(saved_group, dict):
                    raise TrainingStateInvalidError(
                        "optimizer parameter group is not a mutable mapping"
                    )
                # Never trust caller-provided param_names over live model identity.
                saved_group["param_names"] = names
            snapshot = TrainerState(
                micro_step=self.micro_step,
                optimizer_step=self.optimizer_step,
                tokens_seen=self.tokens_seen,
                optimizer=optimizer_state,
                scheduler=(
                    None if self.scheduler is None else copy.deepcopy(self.scheduler.state_dict())
                ),
                scaler=None if self.scaler is None else copy.deepcopy(self.scaler.state_dict()),
                config=asdict(self.config),
            )
            # State-dict hooks can mutate weights, moments, gradients or policy.
            # Refuse publication unless the extracted state remains checkpoint-safe.
            self.assert_checkpoint_safe()
            # Hooks can also return a detached, corrupt snapshot without
            # changing their live component. Validate the bytes to publish.
            self._require_finite_state_tree(snapshot.optimizer, "checkpoint optimizer")
            if snapshot.scheduler is not None:
                self._require_finite_state_tree(snapshot.scheduler, "checkpoint scheduler")
            if snapshot.scaler is not None:
                self._require_finite_state_tree(snapshot.scaler, "checkpoint gradient scaler")
        except BaseException:
            self._mark_failed("checkpoint state extraction failed after possible mutation")
            raise
        return snapshot

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
        if (
            self.micro_step != 0
            or self.optimizer_step != 0
            or self.tokens_seen != 0
            or self._pending_tokens != 0
            or self._pending_loss_sum != 0.0
            or any(parameter.grad is not None for parameter in self.model.parameters())
        ):
            raise TrainingStateInvalidError(
                "trainer state restore requires a fresh trainer with no consumed "
                "exposure or pending gradients; restore the verified model too"
            )
        if isinstance(state, Mapping):
            state = TrainerState(**state)

        if not _typed_state_equal(state.config, asdict(self.config)):
            raise ValueError("trainer config mismatch; refusing unsafe resume")
        # Validate exact counter types before any optimizer/scheduler/scaler mutation.
        # Python considers False == 0 and 0.0 == 0; those are not durable
        # training/exposure accounting identities.
        if any(
            type(value) is not int or value < 0
            for value in (state.micro_step, state.optimizer_step, state.tokens_seen)
        ):
            raise ValueError("trainer counters must be non-negative integers")
        expected_micro_steps = state.optimizer_step * self.config.gradient_accumulation_steps
        if state.micro_step != expected_micro_steps:
            raise ValueError(
                "checkpoint is not at a complete committed accumulation boundary: "
                f"micro_step={state.micro_step}, expected={expected_micro_steps}"
            )
        if state.optimizer_step > self.config.max_steps:
            raise ValueError("checkpoint optimizer_step exceeds configured max_steps")

        # Reject known contract mismatches before touching optimizer state.
        if (state.scheduler is None) != (self.scheduler is None):
            raise ValueError("scheduler state/config mismatch")
        if self.scaler.is_enabled() and not state.scaler:
            raise ValueError("enabled gradient scaler checkpoint state missing")
        # PyTorch maps optimizer slot IDs by group position, ignoring shape-equal
        # parameter identity. Reject missing/reordered names before mutation.
        self._require_optimizer_state_parameter_order(state.optimizer)

        # From the first component load onward a failure may leave optimizer,
        # scheduler, scaler or counters partially applied. No same-instance
        # retry is safe without also restoring the verified model/RNG state.
        self._update_incomplete = True
        try:
            self.optimizer.load_state_dict(state.optimizer)
            self._require_optimizer_parameter_coverage()
            if self.scheduler is not None and state.scheduler is not None:
                self.scheduler.load_state_dict(state.scheduler)
            if state.scaler is not None:
                self.scaler.load_state_dict(state.scaler)

            self.micro_step = state.micro_step
            self.optimizer_step = state.optimizer_step
            self.tokens_seen = state.tokens_seen
            self._pending_tokens = 0
            self._pending_loss_sum = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            # PyTorch's load_state_dict accepts NaN optimizer moments and
            # malformed-but-type-compatible group rates. A restore must not
            # return a supposedly checkpoint-safe trainer with those values.
            self._require_finite_auxiliary_state()
            self._require_finite_committed_update()
            self._require_no_residual_model_gradients()
            self._require_deterministic_policy()
        except BaseException:
            self._mark_failed("trainer state restore failed after possible partial apply")
            raise
        self._update_incomplete = False
        self._failure_reason = None
