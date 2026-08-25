"""Activation-checkpointing policy for the maintained PyTorch scale path.

This module deliberately uses PyTorch's checkpoint wrapper rather than a custom
recomputation implementation. Apply checkpointing before FSDP2 ``fully_shard`` so
checkpoint wrappers become part of the module graph that FSDP2 shards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from twelve_six.model import TwelveSixDecoder

ActivationCheckpointPolicy = Literal["none", "every_other_block", "per_block"]


@dataclass(frozen=True, slots=True)
class ActivationCheckpointPlan:
    policy: ActivationCheckpointPolicy
    checkpointed_block_indices: tuple[int, ...]
    library: str = "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
    implementation: str = "CheckpointImpl.NO_REENTRANT"
    preserve_rng_state: bool = True

    @property
    def checkpointed_blocks(self) -> int:
        return len(self.checkpointed_block_indices)


def checkpoint_block_indices(
    n_layers: int,
    policy: ActivationCheckpointPolicy,
) -> tuple[int, ...]:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if policy == "none":
        return ()
    if policy == "every_other_block":
        return tuple(range(0, n_layers, 2))
    if policy == "per_block":
        return tuple(range(n_layers))
    raise ValueError(f"unsupported activation checkpoint policy: {policy}")


def apply_activation_checkpointing(
    model: TwelveSixDecoder,
    policy: ActivationCheckpointPolicy,
) -> ActivationCheckpointPlan:
    """Mutate ``model.blocks`` using PyTorch's maintained non-reentrant wrapper.

    The non-reentrant implementation is used because it is the maintained path
    compatible with modern autograd/FSDP usage. ``preserve_rng_state`` stays on so
    future stochastic blocks retain ordinary checkpointing semantics even though the
    current canonical Base model has zero attention dropout.
    """

    indices = checkpoint_block_indices(len(model.blocks), policy)
    if not indices:
        return ActivationCheckpointPlan(policy=policy, checkpointed_block_indices=indices)

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        CheckpointWrapper,
        checkpoint_wrapper,
    )

    for index in indices:
        block = model.blocks[index]
        if isinstance(block, CheckpointWrapper):
            raise RuntimeError(
                "activation checkpointing must be applied exactly once before FSDP2"
            )
        model.blocks[index] = checkpoint_wrapper(
            block,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=True,
        )

    return ActivationCheckpointPlan(
        policy=policy,
        checkpointed_block_indices=indices,
    )
