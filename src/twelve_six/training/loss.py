"""Causal language-model objective."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def causal_lm_loss(logits: Tensor, labels: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Return next-token cross entropy for decoder-only logits.

    ``logits[:, t]`` predicts ``labels[:, t + 1]``. The caller owns tokenization,
    packing, and dataset semantics.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, time, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [batch, time], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits batch/time dimensions must match labels: "
            f"{tuple(logits.shape[:2])} != {tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("causal loss requires sequence length >= 2")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )
