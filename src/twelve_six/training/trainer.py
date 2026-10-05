"""Reusable PyTorch trainer with explicit numerical-safety and resume contracts."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import struct
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import Enum
from types import FunctionType
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
        # Only the unmodified first-party optimizer/scheduler pair has a
        # configuration-derived rate oracle. Do not call arbitrary callbacks.
        self._canonical_default_schedule = (
            optimizer is None and scheduler is None and type(self.scheduler) is LambdaLR
        )
        # The default constant/no-warmup path has no LambdaLR object, but
        # its AdamW LR is still an immutable part of the training contract.
        self._canonical_unscheduled_default_optimizer = (
            optimizer is None and scheduler is None and self.scheduler is None
        )
        # Freeze small constructor-owned AdamW options before external hooks
        # can change otherwise finite optimizer behavior at a resume boundary.
        self._canonical_default_optimizer_options = (
            {
                key: copy.deepcopy(value)
                for key, value in self.optimizer.param_groups[0].items()
                if key not in ("params", "lr", "initial_lr", "param_names")
            }
            if optimizer is None and scheduler is None else None
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
                if isinstance(value, Tensor):
                    if not torch.isfinite(value).all().item():
                        raise NonFiniteTrainingError(
                            f"optimizer produced non-finite state at micro_step={self.micro_step}"
                        )
                else:
                    # Non-Torch optimizer families can store NumPy or nested
                    # numeric moments. They must not earn phantom step credit.
                    try:
                        self._require_finite_state_tree(value, "optimizer")
                    except NonFiniteTrainingError as exc:
                        raise NonFiniteTrainingError(
                            f"optimizer produced non-finite state at micro_step={self.micro_step}"
                        ) from exc

    def _require_safe_optimizer_hyperparameters(
        self, optimizer_state: Any | None = None,
    ) -> None:
        """Validate live or checkpoint group hyperparameters before use."""
        live_groups = self.optimizer.param_groups
        groups = live_groups
        compare_checkpoint_types = optimizer_state is not None
        if optimizer_state is not None:
            if not isinstance(optimizer_state, Mapping):
                raise NonFiniteTrainingError("optimizer parameter groups must be valid")
            groups = optimizer_state.get("param_groups")
            if (
                not isinstance(groups, list)
                or not isinstance(live_groups, list)
                or len(groups) != len(live_groups)
            ):
                raise NonFiniteTrainingError("optimizer parameter groups must be valid")
        for group_index, group in enumerate(groups):
            if not isinstance(group, Mapping):
                raise NonFiniteTrainingError("optimizer parameter groups must be valid")
            live_group = live_groups[group_index] if compare_checkpoint_types else group
            for field in ("lr", "weight_decay", "eps"):
                if field not in group:
                    continue  # Other injected optimizer families may omit these fields.
                value = group[field]
                label = "learning rate" if field == "lr" else field
                if (
                    compare_checkpoint_types
                    and field in live_group
                    and type(value) is not type(live_group[field])
                ):
                    raise NonFiniteTrainingError(
                        f"optimizer {label} type differs from live optimizer"
                    )
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
                if compare_checkpoint_types and "betas" in live_group:
                    live_betas = live_group["betas"]
                    if (
                        type(betas) is not type(live_betas)
                        or len(live_betas) != 2
                        or any(
                            type(beta) is not type(live_beta)
                            for beta, live_beta in zip(betas, live_betas, strict=True)
                        )
                    ):
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
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in {"f", "c"}:
                # flatiter slices bound copies even for non-contiguous arrays.
                for start in range(0, value.size, 1_048_576):
                    if not np.isfinite(value.flat[start:start + 1_048_576]).all():
                        raise NonFiniteTrainingError(f"{label} has non-finite state")
        elif isinstance(value, np.generic):
            if value.dtype.kind in {"f", "c"} and not np.isfinite(value):
                raise NonFiniteTrainingError(f"{label} has non-finite state")
        elif isinstance(value, complex):
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
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
        self._require_default_optimizer_options(
            {"param_groups": self.optimizer.param_groups},
        )
        self._require_constant_default_rate({"param_groups": self.optimizer.param_groups})
        scaler_state = self.scaler.state_dict()
        self._require_finite_state_tree(scaler_state, "gradient scaler")
        # Live scaler state must also be restorable: finite subnormal scales
        # can yield an infinite float32 inverse on the next unscale_.
        self._require_checkpoint_scaler_state(scaler_state)
        if self.scheduler is not None:
            self._require_finite_state_tree(self.scheduler.state_dict(), "scheduler")
            # The canonical LambdaLR advances exactly once per committed
            # optimizer step. Finite live corruption is not an exact-resume
            # chronology, even if state_dict accurately exports that corruption.
            if (
                isinstance(self.scheduler, LambdaLR)
                and (
                    type(self.scheduler.last_epoch) is not int
                    or self.scheduler.last_epoch != self.optimizer_step
                    or (
                        type(self.scheduler) is LambdaLR
                        and (
                            type(self.scheduler._step_count) is not int
                            or self.scheduler._step_count != self.optimizer_step + 1
                        )
                    )
                )
            ):
                raise TrainingStateInvalidError(
                    "scheduler chronology differs from committed optimizer step"
                )
            self._require_default_schedule_rates(
                vars(self.scheduler), self.optimizer_step,
                {"param_groups": self.optimizer.param_groups},
            )

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
            # Detect drift between committed steps before consuming another batch.
            self._require_first_party_optimizer_contract()
        except BaseException:
            self._mark_failed("optimizer identity or configured update contract changed")
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
            # Forward/backward hooks may change otherwise finite AdamW options
            # or rates. Refuse before scaler/optimizer.step can mutate weights.
            self._require_first_party_optimizer_contract()
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
                # unscale_/gradient clipping may invoke effectful callbacks.
                # Recheck after them, immediately before the real update.
                self._require_first_party_optimizer_contract()
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

    def _canonical_lambda_lr_live_state(self) -> dict[str, Any] | None:
        """Obtain the canonical built-in scheduler's state without invoking hooks."""
        if self.scheduler is None or type(self.scheduler) is not LambdaLR:
            return None
        live = vars(self.scheduler)
        functions = live.get("lr_lambdas")
        if not isinstance(functions, list) or any(
            not isinstance(fn, FunctionType) for fn in functions
        ):
            raise TrainingStateInvalidError(
                "LambdaLR callback state lacks a pure export authority"
            )
        snapshot = {
            key: value for key, value in live.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        snapshot["lr_lambdas"] = [None] * len(functions)
        # Freeze mutable scheduler lists: an export hook may change them in place.
        return copy.deepcopy(snapshot)

    def _model_export_fingerprint(self) -> str:
        """Hash model weights and buffers without overridable model iterators."""
        digest = hashlib.sha256()
        seen_modules: set[int] = set()
        seen_parameters: set[int] = set()
        seen_buffers: set[int] = set()

        def hash_tensor(value: Tensor) -> None:
            if type(value) not in {Tensor, nn.Parameter}:
                raise TrainingStateInvalidError(
                    "checkpoint model state must use canonical torch tensors"
                )
            if value.layout != torch.strided:
                raise TrainingStateInvalidError(
                    "checkpoint model contains unsupported non-strided state"
                )
            detached = value.detach()
            if detached.is_contiguous():
                flat = detached.reshape(-1)
                for index in range(0, flat.numel(), 262_144):
                    block = flat[index:index + 262_144]
                    raw = block.to(device="cpu").contiguous().view(torch.uint8)
                    digest.update(raw.numpy().tobytes())
            elif detached.numel() <= 262_144:
                bounded = detached.to(device="cpu").contiguous().reshape(-1)
                digest.update(bounded.view(torch.uint8).numpy().tobytes())
            elif detached.ndim == 1:
                for index in range(0, detached.numel(), 262_144):
                    hash_tensor(detached[index:index + 262_144])
            else:
                for child in detached.unbind(0):
                    hash_tensor(child)

        def hash_member(label: str, name: str, value: Tensor) -> None:
            metadata = (
                label,
                name,
                id(value),
                str(value.dtype),
                str(value.device),
                tuple(value.shape),
                tuple(value.stride()),
                value.requires_grad,
            )
            digest.update(repr(metadata).encode("utf-8"))
            hash_tensor(value)

        def walk(module: nn.Module, prefix: str) -> None:
            module_id = id(module)
            if module_id in seen_modules:
                return
            seen_modules.add(module_id)
            try:
                attrs = object.__getattribute__(module, "__dict__")
            except (AttributeError, TypeError) as exc:
                raise TrainingStateInvalidError(
                    "checkpoint model module state is unavailable"
                ) from exc
            parameters = attrs.get("_parameters")
            buffers = attrs.get("_buffers")
            modules = attrs.get("_modules")
            if not all(type(value) is dict for value in (parameters, buffers, modules)):
                raise TrainingStateInvalidError(
                    "checkpoint model module registries must remain canonical dicts"
                )

            for name, value in parameters.items():
                if type(name) is not str:
                    raise TrainingStateInvalidError(
                        "checkpoint model parameter name is not canonical"
                    )
                if value is None or id(value) in seen_parameters:
                    continue
                if not isinstance(value, Tensor):
                    raise TrainingStateInvalidError(
                        "checkpoint model parameter is not a tensor"
                    )
                seen_parameters.add(id(value))
                hash_member("parameter", f"{prefix}{name}", value)
                gradient = value.grad
                if gradient is None:
                    digest.update(b"gradient:none\\0")
                else:
                    if type(gradient) is not Tensor:
                        raise TrainingStateInvalidError(
                            "checkpoint model gradient is not a canonical tensor"
                        )
                    hash_member("gradient", f"{prefix}{name}.grad", gradient)

            for name, value in buffers.items():
                if type(name) is not str:
                    raise TrainingStateInvalidError(
                        "checkpoint model buffer name is not canonical"
                    )
                if value is None or id(value) in seen_buffers:
                    continue
                if not isinstance(value, Tensor):
                    raise TrainingStateInvalidError(
                        "checkpoint model buffer is not a tensor"
                    )
                seen_buffers.add(id(value))
                hash_member("buffer", f"{prefix}{name}", value)

            for name, child in modules.items():
                if type(name) is not str:
                    raise TrainingStateInvalidError(
                        "checkpoint model child-module name is not canonical"
                    )
                if child is None:
                    continue
                if not isinstance(child, nn.Module):
                    raise TrainingStateInvalidError(
                        "checkpoint model child is not a torch module"
                    )
                walk(child, f"{prefix}{name}.")

        try:
            trainer_attrs = object.__getattribute__(self, "__dict__")
            model = trainer_attrs["model"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise TrainingStateInvalidError(
                "checkpoint trainer model binding is unavailable"
            ) from exc
        if not isinstance(model, nn.Module):
            raise TrainingStateInvalidError(
                "checkpoint trainer model binding is not a torch module"
            )
        walk(model, "")
        return digest.hexdigest()

    def _optimizer_live_fingerprint(self) -> str | None:
        """Bind first-party Torch optimizer groups and moments without export hooks."""
        if not self.optimizer.__class__.__module__.startswith("torch.optim"):
            return None
        if not isinstance(self.optimizer.state, Mapping):
            raise TrainingStateInvalidError("optimizer live state is not a mapping")

        digest = hashlib.sha256()

        def update(value: Any) -> None:
            kind = f"{type(value).__module__}.{type(value).__qualname__}"
            digest.update(kind.encode("utf-8") + b"\0")
            if isinstance(value, Tensor):
                if value.layout != torch.strided:
                    raise TrainingStateInvalidError(
                        "optimizer live state contains unsupported non-strided tensor"
                    )
                digest.update(
                    repr(
                        (
                            str(value.dtype),
                            str(value.device),
                            tuple(value.shape),
                            tuple(value.stride()),
                            value.storage_offset(),
                            value.data_ptr(),
                            value.requires_grad,
                        )
                    ).encode("utf-8")
                )
                detached = value.detach()
                if detached.is_contiguous():
                    flat = detached.reshape(-1)
                    for index in range(0, flat.numel(), 262_144):
                        raw = (
                            flat[index:index + 262_144]
                            .to(device="cpu")
                            .contiguous()
                            .view(torch.uint8)
                        )
                        digest.update(raw.numpy().tobytes())
                elif detached.numel() <= 262_144:
                    raw = detached.to(device="cpu").contiguous().reshape(-1)
                    digest.update(raw.view(torch.uint8).numpy().tobytes())
                elif detached.ndim == 1:
                    for index in range(0, detached.numel(), 262_144):
                        update(detached[index:index + 262_144])
                else:
                    for child in detached.unbind(0):
                        update(child)
                return
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise TrainingStateInvalidError(
                        "optimizer live state contains object NumPy array"
                    )
                digest.update(
                    repr((value.dtype.str, value.shape, value.strides)).encode("utf-8")
                )
                for block in np.nditer(
                    value,
                    flags=["external_loop", "buffered", "zerosize_ok"],
                    op_flags=[["readonly"]],
                    order="C",
                    buffersize=262_144,
                ):
                    digest.update(block.tobytes(order="C"))
                return
            if isinstance(value, np.generic):
                digest.update(value.dtype.str.encode("ascii") + b"\0" + value.tobytes())
                return
            if type(value) is float:
                digest.update(struct.pack("!d", value))
                return
            if type(value) is complex:
                digest.update(struct.pack("!dd", value.real, value.imag))
                return
            if value is None or type(value) in {bool, int, str, bytes}:
                digest.update(repr(value).encode("utf-8"))
                return
            if isinstance(value, Mapping):
                digest.update(str(len(value)).encode("ascii") + b"\0")
                for key, child in value.items():
                    if type(key) not in {str, int}:
                        raise TrainingStateInvalidError(
                            "optimizer live state contains unsupported mapping key"
                        )
                    update(key)
                    update(child)
                return
            if isinstance(value, (list, tuple)):
                digest.update(str(len(value)).encode("ascii") + b"\0")
                for child in value:
                    update(child)
                return
            raise TrainingStateInvalidError(
                "optimizer live state contains unsupported exact-resume value"
            )

        name_groups = self._optimizer_parameter_name_groups()
        digest.update(
            (
                self.optimizer.__class__.__module__
                + "."
                + self.optimizer.__class__.__qualname__
            ).encode("utf-8")
        )
        live_parameter_ids: set[int] = set()
        for group_index, (group, names) in enumerate(
            zip(self.optimizer.param_groups, name_groups, strict=True)
        ):
            if not isinstance(group, Mapping):
                raise TrainingStateInvalidError(
                    "optimizer live parameter group is not a mapping"
                )
            parameters = group.get("params")
            if not isinstance(parameters, (list, tuple)):
                raise TrainingStateInvalidError(
                    "optimizer live parameter group params are not a sequence"
                )
            update(group_index)
            update(names)
            update(
                {
                    key: value
                    for key, value in group.items()
                    if key not in ("params", "param_names")
                }
            )
            for parameter, name in zip(parameters, names, strict=True):
                live_parameter_ids.add(id(parameter))
                update(name)
                if parameter in self.optimizer.state:
                    digest.update(b"state-present\0")
                    update(self.optimizer.state[parameter])
                else:
                    digest.update(b"state-absent\0")
        if any(id(parameter) not in live_parameter_ids for parameter in self.optimizer.state):
            raise TrainingStateInvalidError(
                "optimizer live state contains foreign parameter state"
            )
        return digest.hexdigest()

    def _checkpoint_auxiliary_fingerprint(self) -> str:
        """Hash live optimizer/scheduler/scaler state without export or subclass hooks."""

        digest = hashlib.sha256()
        active_objects: set[int] = set()

        def label(value: Any) -> None:
            kind = f"{type(value).__module__}.{type(value).__qualname__}"
            digest.update(kind.encode("utf-8") + b"\0")

        def update_tensor(value: Tensor) -> None:
            if type(value) not in {Tensor, nn.Parameter}:
                raise TrainingStateInvalidError(
                    "checkpoint auxiliary state contains a tensor subclass"
                )
            if value.layout != torch.strided:
                raise TrainingStateInvalidError(
                    "checkpoint auxiliary state contains non-strided tensor"
                )
            digest.update(
                repr(
                    (
                        str(value.dtype),
                        str(value.device),
                        tuple(value.shape),
                        tuple(value.stride()),
                        value.storage_offset(),
                        value.data_ptr(),
                        value.requires_grad,
                    )
                ).encode("utf-8")
            )
            detached = value.detach()
            if detached.is_contiguous():
                flat = detached.reshape(-1)
                for index in range(0, flat.numel(), 262_144):
                    raw = (
                        flat[index:index + 262_144]
                        .to(device="cpu")
                        .contiguous()
                        .view(torch.uint8)
                    )
                    digest.update(raw.numpy().tobytes())
            elif detached.numel() <= 262_144:
                raw = detached.to(device="cpu").contiguous().reshape(-1)
                digest.update(raw.view(torch.uint8).numpy().tobytes())
            elif detached.ndim == 1:
                for index in range(0, detached.numel(), 262_144):
                    update_tensor(detached[index:index + 262_144])
            else:
                for child in detached.unbind(0):
                    update_tensor(child)

        def update(value: Any) -> None:
            label(value)
            if isinstance(value, Tensor):
                update_tensor(value)
                return
            if isinstance(value, np.ndarray):
                if type(value) is not np.ndarray or value.dtype.hasobject:
                    raise TrainingStateInvalidError(
                        "checkpoint auxiliary NumPy state is not canonical"
                    )
                digest.update(
                    repr((value.dtype.str, value.shape, value.strides)).encode("utf-8")
                )
                for block in np.nditer(
                    value,
                    flags=["external_loop", "buffered", "zerosize_ok"],
                    op_flags=[["readonly"]],
                    order="C",
                    buffersize=262_144,
                ):
                    digest.update(block.tobytes(order="C"))
                return
            if isinstance(value, np.generic):
                digest.update(value.dtype.str.encode("ascii") + b"\0" + value.tobytes())
                return
            if type(value) is float:
                digest.update(struct.pack("!d", value))
                return
            if type(value) is complex:
                digest.update(struct.pack("!dd", value.real, value.imag))
                return
            if value is None or type(value) in {bool, int, str, bytes}:
                digest.update(repr(value).encode("utf-8"))
                return
            if isinstance(value, Enum):
                digest.update(
                    f"{type(value).__module__}.{type(value).__qualname__}:{value.name}".encode(
                        "utf-8"
                    )
                )
                return
            if type(value).__module__ == "torch" and type(value).__name__ in {
                "device",
                "dtype",
                "layout",
                "memory_format",
            }:
                digest.update(str(value).encode("utf-8"))
                return
            if isinstance(value, FunctionType):
                digest.update(
                    f"{value.__module__}.{value.__qualname__}:{id(value)}".encode("utf-8")
                )
                return
            if type(value) in {dict, defaultdict, OrderedDict}:
                object_id = id(value)
                if object_id in active_objects:
                    raise TrainingStateInvalidError(
                        "checkpoint auxiliary state contains a container cycle"
                    )
                active_objects.add(object_id)
                try:
                    digest.update(str(len(value)).encode("ascii") + b"\0")
                    for key, child in value.items():
                        if type(key) not in {str, int, bool}:
                            raise TrainingStateInvalidError(
                                "checkpoint auxiliary mapping key is not canonical"
                            )
                        update(key)
                        update(child)
                finally:
                    active_objects.remove(object_id)
                return
            if type(value) in {list, tuple}:
                object_id = id(value)
                if object_id in active_objects:
                    raise TrainingStateInvalidError(
                        "checkpoint auxiliary state contains a container cycle"
                    )
                active_objects.add(object_id)
                try:
                    digest.update(str(len(value)).encode("ascii") + b"\0")
                    for child in value:
                        update(child)
                finally:
                    active_objects.remove(object_id)
                return
            try:
                raw_attrs = object.__getattribute__(value, "__dict__")
            except (AttributeError, TypeError):
                raw_attrs = None
            if type(raw_attrs) is dict:
                object_id = id(value)
                if object_id in active_objects:
                    raise TrainingStateInvalidError(
                        "checkpoint auxiliary state contains an object cycle"
                    )
                active_objects.add(object_id)
                try:
                    digest.update(str(id(value)).encode("ascii") + b"\0")
                    update(raw_attrs)
                finally:
                    active_objects.remove(object_id)
                return
            raise TrainingStateInvalidError(
                "checkpoint auxiliary state contains unsupported exact-resume value"
            )

        try:
            trainer_attrs = object.__getattribute__(self, "__dict__")
            optimizer = trainer_attrs["optimizer"]
            scheduler = trainer_attrs["scheduler"]
            scaler = trainer_attrs["scaler"]
            optimizer_attrs = object.__getattribute__(optimizer, "__dict__")
        except (AttributeError, KeyError, TypeError) as exc:
            raise TrainingStateInvalidError(
                "checkpoint auxiliary bindings are unavailable"
            ) from exc
        if type(optimizer_attrs) is not dict:
            raise TrainingStateInvalidError(
                "checkpoint optimizer storage is not canonical"
            )
        state = optimizer_attrs.get("state")
        groups = optimizer_attrs.get("param_groups")
        if type(state) not in {dict, defaultdict} or type(groups) is not list:
            raise TrainingStateInvalidError(
                "checkpoint optimizer live state is not canonical"
            )

        digest.update(b"optimizer\0")
        digest.update(
            (
                type(optimizer).__module__
                + "."
                + type(optimizer).__qualname__
            ).encode("utf-8")
        )
        digest.update(str(len(groups)).encode("ascii") + b"\0")
        live_parameters: set[int] = set()
        for group in groups:
            if type(group) is not dict:
                raise TrainingStateInvalidError(
                    "checkpoint optimizer parameter group is not canonical"
                )
            parameters = group.get("params")
            if type(parameters) not in {list, tuple}:
                raise TrainingStateInvalidError(
                    "checkpoint optimizer parameters are not a canonical sequence"
                )
            digest.update(str(len(parameters)).encode("ascii") + b"\0")
            for parameter in parameters:
                if type(parameter) is not nn.Parameter:
                    raise TrainingStateInvalidError(
                        "checkpoint optimizer parameter binding is not canonical"
                    )
                live_parameters.add(id(parameter))
                digest.update(str(id(parameter)).encode("ascii") + b"\0")
            update(
                {
                    key: value
                    for key, value in group.items()
                    if key not in ("params", "param_names")
                }
            )

        digest.update(str(len(state)).encode("ascii") + b"\0")
        for parameter, slot in state.items():
            if type(parameter) is not nn.Parameter or id(parameter) not in live_parameters:
                raise TrainingStateInvalidError(
                    "checkpoint optimizer state has a foreign parameter key"
                )
            digest.update(str(id(parameter)).encode("ascii") + b"\0")
            update(slot)

        digest.update(b"scheduler\0")
        if scheduler is None:
            digest.update(b"none\0")
        else:
            try:
                scheduler_attrs = object.__getattribute__(scheduler, "__dict__")
            except (AttributeError, TypeError) as exc:
                raise TrainingStateInvalidError(
                    "checkpoint scheduler storage is unavailable"
                ) from exc
            if type(scheduler_attrs) is not dict:
                raise TrainingStateInvalidError(
                    "checkpoint scheduler storage is not canonical"
                )
            digest.update(
                (
                    type(scheduler).__module__
                    + "."
                    + type(scheduler).__qualname__
                ).encode("utf-8")
            )
            update(
                {
                    key: value
                    for key, value in scheduler_attrs.items()
                    if key != "optimizer"
                }
            )

        digest.update(b"scaler\0")
        if scaler is None:
            digest.update(b"none\0")
        else:
            try:
                scaler_attrs = object.__getattribute__(scaler, "__dict__")
            except (AttributeError, TypeError) as exc:
                raise TrainingStateInvalidError(
                    "checkpoint scaler storage is unavailable"
                ) from exc
            if type(scaler_attrs) is not dict:
                raise TrainingStateInvalidError(
                    "checkpoint scaler storage is not canonical"
                )
            digest.update(
                (
                    type(scaler).__module__
                    + "."
                    + type(scaler).__qualname__
                ).encode("utf-8")
            )
            update(scaler_attrs)

        return digest.hexdigest()

    @staticmethod
    def _exact_export_leaf_equal(saved: Any, live: Any) -> bool:
        """Compare exact stored bits; numerical equality loses signed-zero identity."""
        if isinstance(saved, Tensor) or isinstance(live, Tensor):
            if not (
                isinstance(saved, Tensor)
                and isinstance(live, Tensor)
                and saved.dtype == live.dtype
                and saved.device == live.device
                and saved.shape == live.shape
                and saved.layout == live.layout == torch.strided
                and saved.is_contiguous()
                and live.is_contiguous()
            ):
                # Never create an unbounded contiguous copy of model-scale state.
                return False
            return bool(torch.equal(
                saved.reshape(-1).view(torch.uint8),
                live.reshape(-1).view(torch.uint8),
            ))
        if isinstance(saved, np.ndarray) or isinstance(live, np.ndarray):
            if not (
                isinstance(saved, np.ndarray)
                and isinstance(live, np.ndarray)
                and saved.dtype == live.dtype
                and saved.shape == live.shape
                and not saved.dtype.hasobject
            ):
                return False
            # Buffered external loops bound temporary memory for strided arrays.
            for left, right in np.nditer(
                [saved, live],
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"], ["readonly"]],
                order="C",
                buffersize=262_144,
            ):
                if left.tobytes(order="C") != right.tobytes(order="C"):
                    return False
            return True
        if isinstance(saved, np.generic) or isinstance(live, np.generic):
            return (
                isinstance(saved, np.generic)
                and isinstance(live, np.generic)
                and saved.dtype == live.dtype
                and saved.tobytes() == live.tobytes()
            )
        if type(saved) is float and type(live) is float:
            return struct.pack("!d", saved) == struct.pack("!d", live)
        if type(saved) is complex and type(live) is complex:
            return struct.pack("!dd", saved.real, saved.imag) == struct.pack(
                "!dd", live.real, live.imag,
            )
        if isinstance(saved, Mapping) and isinstance(live, Mapping):
            return (
                {(type(k), k) for k in saved} == {(type(k), k) for k in live}
                and all(Trainer._exact_export_leaf_equal(v, live[k]) for k, v in saved.items())
            )
        if isinstance(saved, (list, tuple)) and type(saved) is type(live):
            return len(saved) == len(live) and all(
                Trainer._exact_export_leaf_equal(a, b)
                for a, b in zip(saved, live, strict=True)
            )
        return type(saved) is type(live) and bool(saved == live)

    def _require_exported_optimizer_matches_live(self, exported: Any) -> None:
        """Do not publish finite but forged moments or optimizer hyperparameters."""
        saved_state = exported.get("state") if isinstance(exported, Mapping) else None
        saved_groups = exported.get("param_groups") if isinstance(exported, Mapping) else None
        if not isinstance(saved_state, Mapping) or not isinstance(saved_groups, list):
            raise TrainingStateInvalidError("optimizer export is not canonical")
        if len(saved_groups) != len(self.optimizer.param_groups):
            raise TrainingStateInvalidError("optimizer export group count differs")
        present: set[int] = set()
        ordinal = 0
        for saved_group, live_group in zip(
            saved_groups, self.optimizer.param_groups, strict=True,
        ):
            live_params = live_group["params"]
            expected_ids = list(range(ordinal, ordinal + len(live_params)))
            if (
                not isinstance(saved_group, Mapping)
                or not isinstance(saved_group.get("params"), list)
                or any(type(i) is not int for i in saved_group["params"])
                or saved_group["params"] != expected_ids
            ):
                raise TrainingStateInvalidError("optimizer export parameter IDs differ")
            saved_options = {
                k: v for k, v in saved_group.items()
                if k not in ("params", "param_names")
            }
            live_options = {
                k: v for k, v in live_group.items()
                if k not in ("params", "param_names")
            }
            if not Trainer._exact_export_leaf_equal(saved_options, live_options):
                raise TrainingStateInvalidError("optimizer export hyperparameters differ")
            for parameter in live_params:
                live_slot = self.optimizer.state.get(parameter)
                saved_slot = saved_state.get(ordinal)
                if live_slot is None:
                    if ordinal in saved_state:
                        raise TrainingStateInvalidError("optimizer export has foreign state")
                elif type(ordinal) is not int or ordinal not in saved_state or not (
                    Trainer._exact_export_leaf_equal(saved_slot, live_slot)
                ):
                    raise TrainingStateInvalidError("optimizer export moments differ")
                else:
                    present.add(ordinal)
                ordinal += 1
        if any(type(k) is not int for k in saved_state) or set(saved_state) != present:
            raise TrainingStateInvalidError("optimizer export contains noncanonical state IDs")

    def _require_checkpoint_scaler_state(self, scaler_state: Any) -> None:
        """Pure scaler authority shared by direct D02 and pre-model-apply D05.

        Native GradScaler.load_state_dict accepts invalid finite statistics, so
        schema matching and a detached native load probe are not sufficient.
        This check must run before either loader touches model or optimizer.
        """
        if self.scaler.is_enabled() and not scaler_state:
            raise ValueError("enabled gradient scaler checkpoint state missing")
        if self.scaler.is_enabled():
            expected_fields = {
                "scale", "growth_factor", "backoff_factor",
                "growth_interval", "_growth_tracker",
            }
            if not isinstance(scaler_state, Mapping) or set(scaler_state) != expected_fields:
                raise ValueError("enabled gradient scaler checkpoint schema invalid")
            scale = scaler_state["scale"]
            growth = scaler_state["growth_factor"]
            backoff = scaler_state["backoff_factor"]
            interval = scaler_state["growth_interval"]
            tracker = scaler_state["_growth_tracker"]
            if (
                any(type(value) is not float or not math.isfinite(value)
                    for value in (scale, growth, backoff))
                or scale <= 0.0
                or growth <= 1.0
                or not 0.0 < backoff < 1.0
                or type(interval) is not int or interval < 1
                or type(tracker) is not int or not 0 <= tracker < interval
            ):
                raise ValueError("enabled gradient scaler checkpoint statistics invalid")
            # GradScaler materializes these statistics in float32. A finite
            # Python float may underflow to zero or overflow on first use.
            # Reject before either D05 loader can apply model/RNG state.
            try:
                scale32, growth32, backoff32 = (
                    struct.unpack("!f", struct.pack("!f", value))[0]
                    for value in (scale, growth, backoff)
                )
            except (OverflowError, struct.error) as exc:
                raise ValueError(
                    "enabled gradient scaler checkpoint statistics invalid in float32"
                ) from exc
            if (
                not math.isfinite(scale32) or scale32 <= 0.0
                or not math.isfinite(growth32) or growth32 <= 1.0
                or not math.isfinite(backoff32) or not 0.0 < backoff32 < 1.0
            ):
                raise ValueError(
                    "enabled gradient scaler checkpoint statistics invalid in float32"
                )
            # GradScaler unscales with the float32 reciprocal. A subnormal,
            # positive scale can be representable while its inverse becomes
            # infinity, corrupting an otherwise finite optimizer update.
            try:
                inverse32 = struct.unpack("!f", struct.pack("!f", 1.0 / scale32))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError(
                    "enabled gradient scaler checkpoint statistics invalid in float32"
                ) from exc
            if not math.isfinite(inverse32):
                raise ValueError(
                    "enabled gradient scaler checkpoint statistics invalid in float32"
                )
        if (
            not self.scaler.is_enabled()
            and scaler_state is not None
            and (not isinstance(scaler_state, Mapping) or bool(scaler_state))
        ):
            # Disabled GradScaler.load_state_dict silently ignores a payload.
            raise ValueError("disabled gradient scaler checkpoint state must be empty")

    def _require_checkpoint_scheduler_chronology(
        self, scheduler_state: Any, optimizer_step: int, optimizer_state: Any,
    ) -> None:
        """Pure D02 chronology and configured-rate authority for direct/D05 preflight."""
        if type(self.scheduler) is LambdaLR and (
            not isinstance(scheduler_state, Mapping)
            or type(scheduler_state.get("last_epoch")) is not int
            or scheduler_state["last_epoch"] != optimizer_step
        ):
            # Refuse a known impossible committed history before optimizer or
            # scheduler load. A fresh target can retry a verified checkpoint.
            raise ValueError(
                "checkpoint scheduler chronology differs from committed optimizer step"
            )
        if type(self.scheduler) is LambdaLR and (
            type(scheduler_state.get("_step_count")) is not int
            or scheduler_state["_step_count"] != optimizer_step + 1
        ):
            # PyTorch LambdaLR starts at internal scheduler step 1 and moves
            # exactly once per successful optimizer update.
            raise ValueError(
                "checkpoint scheduler step count differs from committed optimizer step"
            )
        if type(self.scheduler) is LambdaLR:
            saved_groups = (
                optimizer_state.get("param_groups")
                if isinstance(optimizer_state, Mapping) else None
            )
            last_rates = scheduler_state.get("_last_lr")
            if (
                not isinstance(saved_groups, list)
                or not isinstance(last_rates, list)
                or len(saved_groups) != len(last_rates)
                or any(
                    not isinstance(group, Mapping)
                    or "lr" not in group
                    or not Trainer._exact_export_leaf_equal(rate, group["lr"])
                    for rate, group in zip(last_rates, saved_groups, strict=True)
                )
            ):
                raise ValueError(
                    "checkpoint scheduler last LR differs from checkpoint optimizer"
                )
        self._require_default_schedule_rates(
            scheduler_state, optimizer_step, optimizer_state,
        )
        self._require_constant_default_rate(optimizer_state)
        self._require_default_optimizer_options(optimizer_state)

    def _require_default_schedule_rates(
        self, scheduler_state: Any, optimizer_step: int, optimizer_state: Any,
    ) -> None:
        """Bind default LambdaLR rates to immutable config and committed step.

        A pair of forged finite optimizer/scheduler rates can agree with each
        other while changing the next update. Only the first-party optimizer
        and its first-party LambdaLR have a safe configuration-derived oracle.
        Injected optimizers/schedulers retain their existing authority.
        """
        if not self._canonical_default_schedule:
            return
        groups = (
            optimizer_state.get("param_groups")
            if isinstance(optimizer_state, Mapping) else None
        )
        if (
            not isinstance(scheduler_state, Mapping)
            or not isinstance(groups, list)
            or not isinstance(scheduler_state.get("base_lrs"), list)
            or not isinstance(scheduler_state.get("_last_lr"), list)
            or type(optimizer_step) is not int
            or optimizer_step < 0
            or len(groups) != len(scheduler_state["base_lrs"])
            or len(groups) != len(scheduler_state["_last_lr"])
        ):
            raise TrainingStateInvalidError("default scheduler rate authority is malformed")
        base = self.config.learning_rate
        rate = base * _lr_lambda(self.config)(optimizer_step)

        def bits_equal(actual: Any, expected: Any) -> bool:
            if type(actual) not in (int, float):
                return False
            try:
                observed = float(actual)
            except (OverflowError, ValueError):
                return False
            return (
                math.isfinite(observed)
                and struct.pack("!d", observed) == struct.pack("!d", float(expected))
            )

        for group, base_rate, last_rate in zip(
            groups, scheduler_state["base_lrs"], scheduler_state["_last_lr"],
            strict=True,
        ):
            if (
                not isinstance(group, Mapping)
                or not bits_equal(group.get("initial_lr"), base)
                or not bits_equal(base_rate, base)
                or not bits_equal(group.get("lr"), rate)
                or not bits_equal(last_rate, rate)
            ):
                raise TrainingStateInvalidError(
                    "default scheduler rate differs from configured committed schedule"
                )

    def _require_default_optimizer_options(self, optimizer_state: Any) -> None:
        """Pin first-party AdamW non-LR options before export or state application.

        LambdaLR owns the dynamic LR, but cannot justify a saved change to
        AdamW's betas, eps, decay or constructor feature switches. Exclude
        injected optimizers and schedulers from this first-party contract.
        """
        expected = self._canonical_default_optimizer_options
        if expected is None:
            return
        groups = (
            optimizer_state.get("param_groups")
            if isinstance(optimizer_state, Mapping) else None
        )
        if (
            not isinstance(groups, list)
            or len(groups) != 1
            or not isinstance(groups[0], Mapping)
        ):
            raise TrainingStateInvalidError("default AdamW option groups are malformed")
        actual = {
            key: value for key, value in groups[0].items()
            if key not in ("params", "lr", "initial_lr", "param_names")
        }

        def equal(left: Any, right: Any) -> bool:
            if type(left) is not type(right):
                return False
            if type(left) is float:
                return struct.pack("!d", left) == struct.pack("!d", right)
            if isinstance(left, (list, tuple)):
                return len(left) == len(right) and all(
                    equal(a, b) for a, b in zip(left, right, strict=True)
                )
            if isinstance(left, Mapping):
                return left.keys() == right.keys() and all(
                    equal(value, right[key]) for key, value in left.items()
                )
            return bool(left == right)

        if not equal(actual, expected):
            raise TrainingStateInvalidError(
                "default AdamW options differ from configured constructor"
            )

    def _require_first_party_optimizer_contract(self) -> None:
        """Check the live first-party step contract without invoking state_dict hooks."""
        groups = {"param_groups": self.optimizer.param_groups}
        self._require_default_optimizer_options(groups)
        self._require_constant_default_rate(groups)
        if self._canonical_default_schedule:
            # Read live LambdaLR attributes; do not re-enter state_dict hooks.
            # The direct checkpoint preflight uses ValueError for an invalid
            # saved payload. During training this is an invalid live state.
            try:
                self._require_checkpoint_scheduler_chronology(
                    vars(self.scheduler), self.optimizer_step, groups,
                )
            except ValueError as exc:
                raise TrainingStateInvalidError(
                    "live default scheduler chronology or rate is invalid"
                ) from exc

    def _require_constant_default_rate(self, optimizer_state: Any) -> None:
        """Reject forged finite LR on the default AdamW path without a scheduler.

        This first-party constant/no-warmup path has no LambdaLR state for
        the configured-rate oracle to inspect. Check both live publication
        and saved state before direct or D05 model/optimizer application.
        Custom/injected optimizers keep their existing rate policy.
        """
        if not self._canonical_unscheduled_default_optimizer:
            return
        groups = (
            optimizer_state.get("param_groups")
            if isinstance(optimizer_state, Mapping) else None
        )
        if (
            not isinstance(groups, list)
            or len(groups) != 1
            or not isinstance(groups[0], Mapping)
            or type(groups[0].get("lr")) not in (int, float)
        ):
            raise TrainingStateInvalidError("default constant optimizer rate is malformed")
        try:
            observed = float(groups[0]["lr"])
        except (OverflowError, ValueError) as exc:
            raise TrainingStateInvalidError(
                "default constant optimizer rate is malformed"
            ) from exc
        expected = float(self.config.learning_rate)
        if (
            not math.isfinite(observed)
            or struct.pack("!d", observed) != struct.pack("!d", expected)
        ):
            raise TrainingStateInvalidError(
                "default constant optimizer rate differs from configured learning rate"
            )

    def _require_exported_scaler_matches_live(self, exported: Any) -> None:
        """Refuse finite, detached GradScaler statistics that cannot replay."""
        scaler = self.scaler
        if scaler is None:
            if exported is not None:
                raise TrainingStateInvalidError("gradient scaler export is not canonical")
            return
        if not isinstance(exported, Mapping):
            raise TrainingStateInvalidError("gradient scaler export is not canonical")
        if not scaler.is_enabled():
            expected: dict[str, Any] = {}
        else:
            # Match GradScaler's own five-field state_dict schema using live
            # getters, never a second potentially effectful state_dict call.
            expected = {
                "scale": scaler.get_scale(),
                "growth_factor": scaler.get_growth_factor(),
                "backoff_factor": scaler.get_backoff_factor(),
                "growth_interval": scaler.get_growth_interval(),
                "_growth_tracker": scaler._get_growth_tracker(),
            }
        if not Trainer._exact_export_leaf_equal(exported, expected):
            raise TrainingStateInvalidError(
                "gradient scaler export differs from live state"
            )

    def _require_exported_scheduler_matches_live(self, exported: Any) -> None:
        """Bind finite scheduler snapshot to live epoch/rates without re-exporting.

        LambdaLR stores callable descriptions separately from its live scalar
        fields. Compare both parts directly; another state_dict() call could
        itself be effectful, so it cannot be used as the reference.
        """
        scheduler = self.scheduler
        if scheduler is None:
            if exported is not None:
                raise TrainingStateInvalidError("scheduler export is not canonical")
            return
        if not isinstance(exported, Mapping):
            raise TrainingStateInvalidError("scheduler export is not canonical")
        live = {
            key: value for key, value in vars(scheduler).items()
            if key != "optimizer"
        }
        if isinstance(scheduler, LambdaLR):
            live.pop("lr_lambdas", None)
            if set(exported) != set(live) | {"lr_lambdas"}:
                raise TrainingStateInvalidError("scheduler export fields differ")
            exported_lambdas = exported["lr_lambdas"]
            if (
                not isinstance(exported_lambdas, list)
                or len(exported_lambdas) != len(scheduler.lr_lambdas)
            ):
                raise TrainingStateInvalidError("scheduler export lambda count differs")
            for fn, saved in zip(
                scheduler.lr_lambdas, exported_lambdas, strict=True,
            ):
                # Match PyTorch LambdaLR's own function-vs-callable-object
                # serialization, but inspect the live object directly.
                expected = None if isinstance(fn, FunctionType) else vars(fn)
                if not Trainer._exact_export_leaf_equal(saved, expected):
                    raise TrainingStateInvalidError("scheduler export lambda differs")
            exported = {
                key: value for key, value in exported.items()
                if key != "lr_lambdas"
            }
        if not Trainer._exact_export_leaf_equal(exported, live):
            raise TrainingStateInvalidError("scheduler export differs from live state")

    def state_dict(self) -> TrainerState:
        """Return checkpoint-safe trainer state only after committed optimizer steps."""
        committed_before = (self.micro_step, self.optimizer_step, self.tokens_seen)
        model_before = self._model_export_fingerprint()
        optimizer_before = self._optimizer_live_fingerprint()
        scheduler_before = self._canonical_lambda_lr_live_state()
        self.assert_checkpoint_safe()
        if not _typed_state_equal(
            committed_before, (self.micro_step, self.optimizer_step, self.tokens_seen)
        ):
            self._mark_failed("checkpoint preflight changed committed counters")
            raise TrainingStateInvalidError("checkpoint export changed committed counters")
        if self._model_export_fingerprint() != model_before:
            self._mark_failed("checkpoint preflight changed model weights or buffers")
            raise TrainingStateInvalidError(
                "checkpoint export changed model weights or buffers"
            )
        if (
            optimizer_before is not None
            and self._optimizer_live_fingerprint() != optimizer_before
        ):
            self._mark_failed("checkpoint preflight changed live optimizer state")
            raise TrainingStateInvalidError(
                "checkpoint preflight changed optimizer state"
            )
        if scheduler_before is not None and not Trainer._exact_export_leaf_equal(
            scheduler_before, self._canonical_lambda_lr_live_state()
        ):
            self._mark_failed("checkpoint preflight changed live scheduler")
            raise TrainingStateInvalidError("checkpoint export changed live scheduler")
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
            self._require_exported_optimizer_matches_live(snapshot.optimizer)
            if snapshot.scheduler is not None:
                self._require_finite_state_tree(snapshot.scheduler, "checkpoint scheduler")
            # A hook may suppress the second export entirely. Even a missing
            # snapshot must agree with whether a live component exists.
            self._require_exported_scheduler_matches_live(snapshot.scheduler)
            if snapshot.scaler is not None:
                self._require_finite_state_tree(snapshot.scaler, "checkpoint gradient scaler")
            self._require_exported_scaler_matches_live(snapshot.scaler)
            if (
                optimizer_before is not None
                and self._optimizer_live_fingerprint() != optimizer_before
            ):
                raise TrainingStateInvalidError(
                    "checkpoint export changed optimizer state"
                )
            # Freeze the accepted count/weights across ALL effectful serializers;
            # a valid-looking detached optimizer snapshot is not enough.
            if not _typed_state_equal(
                committed_before, (self.micro_step, self.optimizer_step, self.tokens_seen)
            ):
                raise TrainingStateInvalidError("checkpoint export changed committed counters")
            if self._model_export_fingerprint() != model_before:
                raise TrainingStateInvalidError(
                    "checkpoint export changed model weights or buffers"
                )
            if scheduler_before is not None and not Trainer._exact_export_leaf_equal(
                scheduler_before, self._canonical_lambda_lr_live_state()
            ):
                raise TrainingStateInvalidError(
                    "checkpoint scheduler export differs from live committed state"
                )
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
        self._require_checkpoint_scaler_state(state.scaler)
        # PyTorch maps optimizer slot IDs by group position, ignoring shape-equal
        # parameter identity. Reject missing/reordered names before mutation.
        self._require_optimizer_state_parameter_order(state.optimizer)
        self._require_safe_optimizer_hyperparameters(state.optimizer)
        self._require_checkpoint_scheduler_chronology(
            state.scheduler, state.optimizer_step, state.optimizer,
        )

        # From the first component load onward a failure may leave optimizer,
        # scheduler, scaler or counters partially applied. No same-instance
        # retry is safe without also restoring the verified model/RNG state.
        # param_names is checkpoint-only transport metadata used above to
        # authenticate positional optimizer slots. Do not install it into the
        # live PyTorch optimizer: uninterrupted training does not carry this key,
        # and retaining it would make a resumed raw optimizer state differ from
        # the exact uninterrupted state despite identical numerical dynamics.
        optimizer_state = dict(state.optimizer)
        optimizer_state["param_groups"] = [
            {
                key: value
                for key, value in group.items()
                if key != "param_names"
            }
            for group in state.optimizer["param_groups"]
        ]

        self._update_incomplete = True
        try:
            self.optimizer.load_state_dict(optimizer_state)
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
