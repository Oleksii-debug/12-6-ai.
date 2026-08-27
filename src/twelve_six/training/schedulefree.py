"""Optional Schedule-Free AdamW integration for matched D02 experiments.

This module is deliberately additive. The canonical Trainer/AdamW path remains
unchanged; callers must opt into :class:`ScheduleFreeTrainer` explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .config import TrainerConfig
from .trainer import Trainer, TrainerState

SCHEDULEFREE_PACKAGE = "schedulefree"
SCHEDULEFREE_VERSION = "1.4.1"
SCHEDULEFREE_SOURCE_REPO = "facebookresearch/schedule_free"
SCHEDULEFREE_SOURCE_COMMIT = "70785b53e778d0e872c0bbb75ff4ee54ee10c291"
SCHEDULEFREE_LICENSE = "Apache-2.0"
SCHEDULEFREE_ADAPTER_SCHEMA = 1
SCHEDULEFREE_OPTIMIZER_KIND = "schedulefree_adamw"


class ScheduleFreeDependencyError(RuntimeError):
    """The exact optional Schedule-Free dependency is unavailable or stale."""


class ScheduleFreeConfigError(ValueError):
    """The trainer configuration would invalidate the matched-arm contract."""


class ScheduleFreeStateError(ValueError):
    """Serialized optimizer state violates Schedule-Free checkpoint semantics."""


def schedulefree_optimizer_binding() -> dict[str, Any]:
    """Return the exact data-only identity embedded in Schedule-Free trainer state."""

    return {
        "schema_version": SCHEDULEFREE_ADAPTER_SCHEMA,
        "optimizer_kind": SCHEDULEFREE_OPTIMIZER_KIND,
        "package": SCHEDULEFREE_PACKAGE,
        "package_version": SCHEDULEFREE_VERSION,
        "source_repo": SCHEDULEFREE_SOURCE_REPO,
        "source_commit": SCHEDULEFREE_SOURCE_COMMIT,
        "license": SCHEDULEFREE_LICENSE,
        "inner_momentum": 0.0,
        "foreach": False,
    }


def _load_schedulefree_optimizer_class() -> type[Optimizer]:
    try:
        installed = importlib_metadata.version(SCHEDULEFREE_PACKAGE)
    except importlib_metadata.PackageNotFoundError as exc:
        raise ScheduleFreeDependencyError(
            "Schedule-Free candidate requires the optional dependency "
            f"{SCHEDULEFREE_PACKAGE}=={SCHEDULEFREE_VERSION}; canonical AdamW remains available"
        ) from exc

    if installed != SCHEDULEFREE_VERSION:
        raise ScheduleFreeDependencyError(
            "Schedule-Free candidate dependency version mismatch: "
            f"expected {SCHEDULEFREE_VERSION}, found {installed}"
        )

    try:
        from schedulefree import AdamWScheduleFree
    except Exception as exc:  # pragma: no cover - exact import failure is environment-specific
        raise ScheduleFreeDependencyError(
            "schedulefree==1.4.1 is installed but AdamWScheduleFree could not be imported"
        ) from exc

    if not isinstance(AdamWScheduleFree, type) or not issubclass(AdamWScheduleFree, Optimizer):
        raise ScheduleFreeDependencyError(
            "schedulefree.AdamWScheduleFree is not a torch.optim.Optimizer subclass"
        )
    return AdamWScheduleFree


def _assert_matched_config(config: TrainerConfig) -> None:
    # The frozen AdamW control in PR #583 uses constant/no-warmup semantics.
    # Keep this candidate package one-variable-at-a-time: optimizer only.
    if config.scheduler != "constant":
        raise ScheduleFreeConfigError(
            "matched Schedule-Free experiment requires scheduler='constant'"
        )
    if config.warmup_steps != 0:
        raise ScheduleFreeConfigError(
            "matched Schedule-Free experiment requires warmup_steps=0"
        )


def build_schedulefree_adamw(model: nn.Module, config: TrainerConfig) -> Optimizer:
    """Build the exact optional Schedule-Free AdamW candidate optimizer."""

    _assert_matched_config(config)
    optimizer_class = _load_schedulefree_optimizer_class()
    optimizer = optimizer_class(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        warmup_steps=0,
        inner_momentum=0.0,
        foreach=False,
    )
    _assert_mode_api(optimizer)
    _assert_optimizer_eval_mode(optimizer)
    return optimizer


def _assert_mode_api(optimizer: Optimizer) -> None:
    if not callable(getattr(optimizer, "train", None)):
        raise ScheduleFreeDependencyError("Schedule-Free optimizer is missing train()")
    if not callable(getattr(optimizer, "eval", None)):
        raise ScheduleFreeDependencyError("Schedule-Free optimizer is missing eval()")


def _assert_optimizer_eval_mode(optimizer: Optimizer) -> None:
    if not optimizer.param_groups:
        raise ScheduleFreeStateError("Schedule-Free optimizer has no parameter groups")
    for index, group in enumerate(optimizer.param_groups):
        if group.get("train_mode") is not False:
            raise ScheduleFreeStateError(
                f"Schedule-Free optimizer param_group[{index}] must be in eval mode for checkpointing"
            )


def _validate_serialized_eval_mode(optimizer_state: Mapping[str, Any]) -> None:
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ScheduleFreeStateError(
            "Schedule-Free checkpoint optimizer state must contain non-empty param_groups"
        )
    for index, group in enumerate(param_groups):
        if not isinstance(group, Mapping) or group.get("train_mode") is not False:
            raise ScheduleFreeStateError(
                f"Schedule-Free checkpoint param_group[{index}] is not in eval mode"
            )


class ScheduleFreeTrainer(Trainer):
    """Trainer variant that owns Schedule-Free mode/checkpoint transitions.

    Upstream Schedule-Free keeps distinct training and evaluation parameter
    points. The optimizer is therefore switched to train mode before every
    training microbatch and to eval mode before trainer state is exposed for a
    checkpoint. The serialized state carries an exact optimizer provenance
    binding so a stale or non-Schedule-Free state cannot be resumed silently.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        optimizer = build_schedulefree_adamw(model, config)
        super().__init__(
            model,
            config,
            device=device,
            optimizer=optimizer,
            scheduler=None,
        )
        if self.scheduler is not None:
            raise ScheduleFreeConfigError(
                "matched Schedule-Free trainer must not attach an external LR scheduler"
            )
        _assert_optimizer_eval_mode(self.optimizer)
        self.model.eval()

    def enter_train_mode(self) -> None:
        self.optimizer.train()
        self.model.train()

    def enter_eval_mode(self) -> None:
        self.optimizer.eval()
        self.model.eval()
        _assert_optimizer_eval_mode(self.optimizer)

    def train_microbatch(self, batch: Mapping[str, torch.Tensor]):
        self.enter_train_mode()
        return super().train_microbatch(batch)

    def state_dict(self) -> TrainerState:
        # Refuse to mutate parameter representation if the current state is not
        # already a complete committed boundary.
        self.assert_checkpoint_safe()
        self.enter_eval_mode()
        state = super().state_dict()
        config = dict(state.config)
        config["_optimizer_binding"] = schedulefree_optimizer_binding()
        return TrainerState(
            micro_step=state.micro_step,
            optimizer_step=state.optimizer_step,
            tokens_seen=state.tokens_seen,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=state.scaler,
            config=config,
        )

    def load_state_dict(self, state: TrainerState | Mapping[str, Any]) -> None:
        if isinstance(state, Mapping):
            state = TrainerState(**state)

        config = dict(state.config)
        binding = config.pop("_optimizer_binding", None)
        expected_binding = schedulefree_optimizer_binding()
        if binding != expected_binding:
            raise ScheduleFreeStateError(
                "Schedule-Free checkpoint optimizer binding mismatch; refusing unsafe resume"
            )
        _validate_serialized_eval_mode(state.optimizer)

        plain_state = TrainerState(
            micro_step=state.micro_step,
            optimizer_step=state.optimizer_step,
            tokens_seen=state.tokens_seen,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=state.scaler,
            config=config,
        )
        super().load_state_dict(plain_state)
        _assert_optimizer_eval_mode(self.optimizer)
        self.model.eval()
