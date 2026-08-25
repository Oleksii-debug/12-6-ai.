"""Fail-closed 12-6 -> Transformers Llama interoperability planning and conversion.

The planning/conversion surface is dependency-light so canonical 12-6 environments do
not need Transformers installed. Runtime execution lives in transformers_llama_runtime
and consumes this exact mapping under the dedicated locked Transformers overlay.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from twelve_six.model import ModelSpec

_SCHEMA = "12-6.transformers-llama-interop-plan.v2"
_TARGET_ARCHITECTURE = "LlamaForCausalLM"
_RUNTIME_STATUS = "LOCKED_RUNTIME_REQUIRED"
_ROPE_TRANSFORM = "PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT"


class TransformersInteropError(ValueError):
    """Raised when 12-6 semantics cannot be represented without approximation."""


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_llama_representable(spec: ModelSpec) -> None:
    errors: list[str] = []
    if spec.schema_version != 1:
        errors.append("ModelSpec schema_version must be 1")
    if spec.activation != "swiglu":
        errors.append("activation must be swiglu")
    if spec.norm_kind != "rmsnorm" or spec.norm_placement != "pre":
        errors.append("normalization must be pre-RMSNorm")
    if spec.position_embedding != "rope":
        errors.append("position embedding must be RoPE")
    if spec.rope_rotary_dim != spec.head_dim:
        errors.append("partial RoPE is not accepted by the conservative Llama bridge")
    if spec.q_dim != spec.d_model:
        errors.append(
            "q_dim must equal d_model so hidden_size is divisible by num_attention_heads"
        )
    if spec.attention_bias:
        errors.append("attention projection bias is not accepted by bridge v2")
    if spec.mlp_bias:
        errors.append("MLP bias is not accepted by bridge v2")
    if spec.lm_head_bias:
        errors.append("LlamaForCausalLM has no compatible lm_head bias in bridge v2")
    if not spec.final_norm:
        errors.append("final RMSNorm is required by bridge v2")
    if errors:
        raise TransformersInteropError("; ".join(errors))


def llama_config_dict(spec: ModelSpec) -> dict[str, object]:
    """Return the exact raw-Base-safe Transformers 5.15 LlamaConfig payload.

    Transformers 5.15 moved Llama's serialized RoPE parameters under
    ``rope_parameters``. Special-token IDs remain explicit ``None`` values so the
    bridge does not invent BOS/EOS/PAD or chat semantics for the raw-byte Base model.
    """

    _require_llama_representable(spec)
    return {
        "model_type": "llama",
        "architectures": [_TARGET_ARCHITECTURE],
        "vocab_size": spec.vocab_size,
        "hidden_size": spec.d_model,
        "intermediate_size": spec.d_ff,
        "num_hidden_layers": spec.n_layers,
        "num_attention_heads": spec.n_heads,
        "num_key_value_heads": spec.n_kv_heads,
        "head_dim": spec.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": spec.max_seq_len,
        "rms_norm_eps": spec.norm_eps,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": spec.rope_theta,
        },
        "attention_bias": False,
        "attention_dropout": spec.attention_dropout,
        "mlp_bias": False,
        "tie_word_embeddings": spec.tie_word_embeddings,
        "bos_token_id": None,
        "eos_token_id": None,
        "pad_token_id": None,
        "use_cache": True,
    }


def _source_tensor_shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (spec.vocab_size, spec.d_model),
        "final_norm.weight": (spec.d_model,),
        "lm_head.weight": (spec.vocab_size, spec.d_model),
    }
    for layer in range(spec.n_layers):
        prefix = f"blocks.{layer}"
        shapes.update(
            {
                f"{prefix}.attn_norm.weight": (spec.d_model,),
                f"{prefix}.attn.q_proj.weight": (spec.q_dim, spec.d_model),
                f"{prefix}.attn.k_proj.weight": (spec.kv_dim, spec.d_model),
                f"{prefix}.attn.v_proj.weight": (spec.kv_dim, spec.d_model),
                f"{prefix}.attn.out_proj.weight": (spec.d_model, spec.q_dim),
                f"{prefix}.mlp_norm.weight": (spec.d_model,),
                f"{prefix}.mlp.gate_proj.weight": (spec.d_ff, spec.d_model),
                f"{prefix}.mlp.up_proj.weight": (spec.d_ff, spec.d_model),
                f"{prefix}.mlp.down_proj.weight": (spec.d_model, spec.d_ff),
            }
        )
    return shapes


def _target_name(source_name: str) -> str:
    if source_name == "token_embedding.weight":
        return "model.embed_tokens.weight"
    if source_name == "final_norm.weight":
        return "model.norm.weight"
    if source_name == "lm_head.weight":
        return "lm_head.weight"

    parts = source_name.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        raise TransformersInteropError(f"unsupported source tensor name: {source_name}")
    layer = parts[1]
    suffix = ".".join(parts[2:])
    mapping = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn.q_proj.weight": "self_attn.q_proj.weight",
        "attn.k_proj.weight": "self_attn.k_proj.weight",
        "attn.v_proj.weight": "self_attn.v_proj.weight",
        "attn.out_proj.weight": "self_attn.o_proj.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
    }
    target_suffix = mapping.get(suffix)
    if target_suffix is None:
        raise TransformersInteropError(f"unsupported source tensor name: {source_name}")
    return f"model.layers.{layer}.{target_suffix}"


def rope_pairwise_to_llama_permutation(*, heads: int, head_dim: int) -> tuple[int, ...]:
    """Return row indices converting adjacent-pair RoPE to Llama half-split basis."""

    if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0:
        raise ValueError("heads must be a positive integer")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE basis conversion")
    result: list[int] = []
    for head in range(heads):
        offset = head * head_dim
        result.extend(offset + index for index in range(0, head_dim, 2))
        result.extend(offset + index for index in range(1, head_dim, 2))
    return tuple(result)


def _convert_qk_rows(weight: Tensor, *, heads: int, head_dim: int) -> Tensor:
    permutation = torch.tensor(
        rope_pairwise_to_llama_permutation(heads=heads, head_dim=head_dim),
        dtype=torch.long,
        device=weight.device,
    )
    if weight.ndim != 2 or weight.shape[0] != len(permutation):
        raise TransformersInteropError("Q/K weight shape does not match declared head geometry")
    return weight.index_select(0, permutation).detach().clone()


def convert_state_dict_to_llama(
    spec: ModelSpec,
    source_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Convert canonical 12-6 tensors to Transformers Llama names and RoPE basis."""

    _require_llama_representable(spec)
    expected_shapes = _source_tensor_shapes(spec)
    actual_names = set(source_state)
    expected_names = set(expected_shapes)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise TransformersInteropError(
            f"source state inventory mismatch: missing={missing}, extra={extra}"
        )

    converted: dict[str, Tensor] = {}
    for source_name, expected_shape in expected_shapes.items():
        tensor = source_state[source_name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"source state tensor {source_name!r} must be torch.Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise TransformersInteropError(
                f"source tensor {source_name!r} shape mismatch: "
                f"expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        if source_name.endswith("attn.q_proj.weight"):
            target = _convert_qk_rows(tensor, heads=spec.n_heads, head_dim=spec.head_dim)
        elif source_name.endswith("attn.k_proj.weight"):
            target = _convert_qk_rows(tensor, heads=spec.n_kv_heads, head_dim=spec.head_dim)
        else:
            target = tensor.detach().clone()
        converted[_target_name(source_name)] = target
    return converted


@dataclass(frozen=True, slots=True)
class LlamaInteropPlan:
    schema: str
    source_model_spec_sha256: str
    source_parameter_count: int
    target_architecture: str
    target_config: dict[str, object]
    tensor_map: tuple[dict[str, str], ...]
    rope_transform: str
    runtime_status: str
    runtime_parity_required: bool

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_model_spec_sha256": self.source_model_spec_sha256,
            "source_parameter_count": self.source_parameter_count,
            "target_architecture": self.target_architecture,
            "target_config": self.target_config,
            "tensor_map": list(self.tensor_map),
            "rope_transform": self.rope_transform,
            "runtime_status": self.runtime_status,
            "runtime_parity_required": self.runtime_parity_required,
        }

    def identity_sha256(self) -> str:
        return _sha256_json(self.payload())


def build_llama_interop_plan(spec: ModelSpec) -> LlamaInteropPlan:
    """Build deterministic Llama conversion metadata for the runtime acceptance gate."""

    _require_llama_representable(spec)
    tensor_map = []
    for source_name in sorted(_source_tensor_shapes(spec)):
        transform = "COPY"
        if source_name.endswith(("attn.q_proj.weight", "attn.k_proj.weight")):
            transform = _ROPE_TRANSFORM
        tensor_map.append(
            {
                "source": source_name,
                "target": _target_name(source_name),
                "transform": transform,
            }
        )
    return LlamaInteropPlan(
        schema=_SCHEMA,
        source_model_spec_sha256=spec.identity_sha256(),
        source_parameter_count=spec.parameter_count(),
        target_architecture=_TARGET_ARCHITECTURE,
        target_config=llama_config_dict(spec),
        tensor_map=tuple(tensor_map),
        rope_transform=_ROPE_TRANSFORM,
        runtime_status=_RUNTIME_STATUS,
        runtime_parity_required=True,
    )
