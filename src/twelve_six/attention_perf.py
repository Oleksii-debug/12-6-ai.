from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AttentionGeometry:
    """Runtime-only SDPA geometry; it does not participate in ModelSpec identity."""

    query_heads: int
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name, value in {
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")

    @property
    def uses_gqa(self) -> bool:
        return self.query_heads != self.kv_heads

    @property
    def kv_repeat_factor(self) -> int:
        return self.query_heads // self.kv_heads


def infer_geometry(q: Tensor, k: Tensor, v: Tensor) -> AttentionGeometry:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape [batch, heads, sequence, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if q.shape[0] != k.shape[0]:
        raise ValueError("q, k and v batch sizes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q, k and v head dimensions must match")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must have the same dtype")
    return AttentionGeometry(
        query_heads=int(q.shape[-3]),
        kv_heads=int(k.shape[-3]),
        head_dim=int(q.shape[-1]),
    )


def expand_kv_for_reference(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Materialize the current canonical GQA behavior for parity/benchmark reference."""
    geometry = infer_geometry(q, k, v)
    if not geometry.uses_gqa:
        return k, v
    repeats = geometry.kv_repeat_factor
    return k.repeat_interleave(repeats, dim=-3), v.repeat_interleave(repeats, dim=-3)


def sdpa_expanded_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool,
) -> Tensor:
    """Existing 12-6 behavior: expand grouped K/V before PyTorch SDPA."""
    expanded_k, expanded_v = expand_kv_for_reference(q, k, v)
    return F.scaled_dot_product_attention(
        q,
        expanded_k,
        expanded_v,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


def sdpa_native_gqa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool,
) -> Tensor:
    """Performance candidate: preserve grouped K/V and delegate GQA to SDPA."""
    geometry = infer_geometry(q, k, v)
    try:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=geometry.uses_gqa,
        )
    except TypeError as exc:
        if geometry.uses_gqa and "enable_gqa" in str(exc):
            raise RuntimeError(
                "native SDPA GQA is unavailable in this PyTorch runtime; "
                "keep expanded reference semantics or raise the locked torch floor"
            ) from exc
        raise


def kv_tensor_bytes(
    *,
    batch_size: int,
    kv_heads: int,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Bytes occupied by one unexpanded K+V pair for the requested geometry."""
    for name, value in {
        "batch_size": batch_size,
        "kv_heads": kv_heads,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    item_size = torch.empty((), dtype=dtype).element_size()
    return 2 * batch_size * kv_heads * sequence_length * head_dim * item_size


def expanded_kv_tensor_bytes(
    *,
    batch_size: int,
    query_heads: int,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Bytes in the materialized K+V tensors after the current repeat_interleave path."""
    return kv_tensor_bytes(
        batch_size=batch_size,
        kv_heads=query_heads,
        sequence_length=sequence_length,
        head_dim=head_dim,
        dtype=dtype,
    )
