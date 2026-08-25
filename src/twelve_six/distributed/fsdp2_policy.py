"""Maintained PyTorch FSDP2 reshard-policy binding for the 12-6 decoder.

This module does not implement sharding or recomputation. It only selects among
PyTorch FSDP2's maintained ``reshard_after_forward`` behaviors while reusing the
canonical ``apply_fsdp2`` model grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from twelve_six.distributed.fsdp2_training import apply_fsdp2
from twelve_six.model import TwelveSixDecoder


class FSDP2ReshardPolicy(str, Enum):
    """Small policy set relevant to the current layer-wise decoder grouping."""

    FULL_SHARD = "full_shard"
    ROOT_KEEP_UNSHARDED = "root_keep_unsharded"
    SHARD_GRAD_OP = "shard_grad_op"


@dataclass(frozen=True, slots=True)
class FSDP2ReshardPolicySpec:
    name: FSDP2ReshardPolicy
    non_root_reshard_after_forward: bool
    root_reshard_after_forward: bool
    communication_note: str
    memory_note: str


_POLICY_SPECS = {
    FSDP2ReshardPolicy.FULL_SHARD: FSDP2ReshardPolicySpec(
        name=FSDP2ReshardPolicy.FULL_SHARD,
        non_root_reshard_after_forward=True,
        root_reshard_after_forward=True,
        communication_note="forward all-gather plus backward re-all-gather for every FSDP group",
        memory_note="lowest post-forward parameter residency of the compared policies",
    ),
    FSDP2ReshardPolicy.ROOT_KEEP_UNSHARDED: FSDP2ReshardPolicySpec(
        name=FSDP2ReshardPolicy.ROOT_KEEP_UNSHARDED,
        non_root_reshard_after_forward=True,
        root_reshard_after_forward=False,
        communication_note="layer groups re-all-gather in backward; root group stays unsharded",
        memory_note="retains only the root FSDP group's unsharded parameters after forward",
    ),
    FSDP2ReshardPolicy.SHARD_GRAD_OP: FSDP2ReshardPolicySpec(
        name=FSDP2ReshardPolicy.SHARD_GRAD_OP,
        non_root_reshard_after_forward=False,
        root_reshard_after_forward=False,
        communication_note="avoids backward parameter re-all-gathers by retaining unsharded parameters",
        memory_note="highest post-forward parameter residency of the compared policies",
    ),
}


def fsdp2_reshard_policy_spec(
    policy: FSDP2ReshardPolicy | str,
) -> FSDP2ReshardPolicySpec:
    """Resolve a stable project name to maintained FSDP2 API behavior."""

    resolved = policy if isinstance(policy, FSDP2ReshardPolicy) else FSDP2ReshardPolicy(policy)
    return _POLICY_SPECS[resolved]


def apply_fsdp2_reshard_policy(
    model: TwelveSixDecoder,
    mesh: Any,
    *,
    policy: FSDP2ReshardPolicy | str,
) -> TwelveSixDecoder:
    """Apply the canonical 12-6 FSDP2 grouping with one maintained reshard policy.

    ``apply_fsdp2`` remains the sole implementation of decoder grouping, including
    the explicit tied embedding/head FSDP group. The root-only variant uses
    ``FSDPModule.set_reshard_after_forward`` after canonical sharding so it does not
    fork the runtime or duplicate grouping logic.
    """

    spec = fsdp2_reshard_policy_spec(policy)
    model = apply_fsdp2(
        model,
        mesh,
        reshard_after_forward=spec.non_root_reshard_after_forward,
    )
    if spec.root_reshard_after_forward != spec.non_root_reshard_after_forward:
        setter = getattr(model, "set_reshard_after_forward", None)
        if not callable(setter):
            raise RuntimeError("current PyTorch FSDP2 root reshard setter is unavailable")
        setter(spec.root_reshard_after_forward, recurse=False)
    return model
