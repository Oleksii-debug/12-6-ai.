"""Research-only Q/K normalization for MODEL-119.

Canonical ``ModelSpec`` and ``TwelveSixDecoder`` semantics are not modified. The
research spec is a strict ModelSpec subclass whose disabled serialization is
identity-compatible with the incumbent; enabling the flag adds research fields
to the serialized ModelSpec and therefore creates a distinct checkpoint identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MethodType

import torch
import torch.nn.functional as F
from torch import Tensor

from twelve_six.model import (
    CausalSelfAttention,
    InitSpec,
    ModelSpec,
    TwelveSixDecoder,
    apply_rope,
)


@dataclass(frozen=True, slots=True)
class ResearchModelSpec(ModelSpec):
    """ModelSpec research extension; disabled form preserves incumbent identity."""

    research_qk_norm: bool = False
    research_qk_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        ModelSpec.__post_init__(self)
        if not isinstance(self.research_qk_norm, bool):
            raise ValueError("research_qk_norm must be boolean")
        if self.research_qk_norm_eps <= 0:
            raise ValueError("research_qk_norm_eps must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if not self.research_qk_norm:
            payload.pop("research_qk_norm")
            payload.pop("research_qk_norm_eps")
        return payload

    @classmethod
    def from_base(
        cls,
        spec: ModelSpec,
        *,
        research_qk_norm: bool,
        research_qk_norm_eps: float = 1e-6,
    ) -> "ResearchModelSpec":
        return cls(
            **spec.to_dict(),
            research_qk_norm=research_qk_norm,
            research_qk_norm_eps=research_qk_norm_eps,
        )


def qk_rms_normalize(x: Tensor, eps: float) -> Tensor:
    """Parameter-free per-head RMS normalization."""
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps).to(dtype=x.dtype)


def _qk_norm_forward(self: CausalSelfAttention, x: Tensor) -> Tensor:
    spec = getattr(self, "_research_spec", None)
    if not isinstance(spec, ResearchModelSpec) or not spec.research_qk_norm:
        raise RuntimeError("research QK attention invoked without enabled ResearchModelSpec")

    batch, seq_len, _ = x.shape
    q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rope.cos_sin(seq_len, device=x.device, dtype=q.dtype)
    q = apply_rope(q, cos, sin, self.rotary_dim)
    k = apply_rope(k, cos, sin, self.rotary_dim)
    q = qk_rms_normalize(q, spec.research_qk_norm_eps)
    k = qk_rms_normalize(k, spec.research_qk_norm_eps)

    if self.n_kv_heads != self.n_heads:
        repeats = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    attended = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True,
    )
    attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.q_dim)
    return self.out_proj(attended)


def build_research_decoder(
    spec: ResearchModelSpec,
    init_spec: InitSpec | None = None,
) -> TwelveSixDecoder:
    """Build incumbent decoder weights and patch only enabled research attention calls."""
    model = TwelveSixDecoder(spec, init_spec)
    if spec.research_qk_norm:
        for block in model.blocks:
            block.attn._research_spec = spec
            block.attn.forward = MethodType(_qk_norm_forward, block.attn)
    return model
