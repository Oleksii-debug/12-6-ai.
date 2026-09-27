"""Causal language-model objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_logits_targets(logits: Tensor, targets: Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, time, vocab], got {tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape [batch, time], got {tuple(targets.shape)}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "logits batch/time dimensions must match targets: "
            f"{tuple(logits.shape[:2])} != {tuple(targets.shape)}"
        )


def causal_lm_loss(logits: Tensor, labels: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Return shifted next-token cross entropy for unshifted decoder-only labels.

    ``logits[:, t]`` predicts ``labels[:, t + 1]``. The caller owns tokenization,
    packing, and dataset semantics.
    """
    _validate_logits_targets(logits, labels)
    if logits.shape[1] < 2:
        raise ValueError("causal loss requires sequence length >= 2")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )


def causal_pair_loss(
    logits: Tensor,
    target_ids: Tensor,
    *,
    loss_mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Return CE for already-aligned causal pairs such as D04 packed examples.

    Here ``logits[:, t]`` predicts ``target_ids[:, t]`` directly. ``loss_mask`` may
    mark padded tail positions with zero. This function deliberately does not shift.
    """
    _validate_logits_targets(logits, target_ids)
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(target_ids)

    valid = target_ids.ne(ignore_index)
    if loss_mask is not None:
        if loss_mask.shape != target_ids.shape:
            raise ValueError("loss_mask must match target_ids shape")
        if not torch.all((loss_mask == 0) | (loss_mask == 1)).item():
            raise ValueError("loss_mask values must be binary 0/1")
        valid = valid & loss_mask.bool()

    count = valid.sum()
    if count.item() == 0:
        raise ValueError("causal_pair_loss requires at least one unmasked target")
    return per_token.masked_select(valid).sum() / count
