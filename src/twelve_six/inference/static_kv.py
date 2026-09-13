from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from twelve_six.model import CausalLMOutput, TwelveSixDecoder


@dataclass(slots=True)
class StaticAttentionKVCache:
    """Fixed-capacity unexpanded K/V backing storage for one decoder layer."""

    key: Tensor
    value: Tensor

    @property
    def capacity(self) -> int:
        if self.key.ndim != 4:
            raise ValueError("static cached key must have shape [batch, kv_heads, capacity, head_dim]")
        return int(self.key.shape[2])

    @property
    def allocated_bytes(self) -> int:
        return (
            self.key.numel() * self.key.element_size()
            + self.value.numel() * self.value.element_size()
        )

    @property
    def storage_signature(self) -> tuple[int, int]:
        """Stable backing-storage identity used to prove decode does not reallocate K/V."""
        return self.key.data_ptr(), self.value.data_ptr()


@dataclass(slots=True)
class StaticDecoderKVCache:
    """Reusable fixed-capacity inference cache bound to one exact ModelSpec identity.

    K/V tensors are allocated once at [batch, n_kv_heads, capacity, head_dim].
    ``valid_lengths`` is explicit per row. The accepted fixed-row batching scheduler
    keeps those lengths equal during coalesced decode, including retired rows that
    receive filler tokens. Resetting only changes validity metadata; stale bytes are
    never read after reset and are overwritten before becoming valid again.
    """

    model_spec_sha256: str
    batch_size: int
    capacity: int
    valid_lengths: list[int]
    layers: tuple[StaticAttentionKVCache, ...]

    @property
    def sequence_length(self) -> int:
        if not self.valid_lengths:
            raise ValueError("static KV cache must contain at least one sequence")
        first = self.valid_lengths[0]
        if any(length != first for length in self.valid_lengths):
            raise ValueError("static KV cache rows have heterogeneous valid lengths")
        return first

    @property
    def allocated_bytes(self) -> int:
        return sum(layer.allocated_bytes for layer in self.layers)

    @property
    def logical_bytes(self) -> int:
        if not self.layers:
            return 0
        layer = self.layers[0]
        if layer.key.ndim != 4:
            raise ValueError("static cached key must be rank 4")
        kv_heads = int(layer.key.shape[1])
        head_dim = int(layer.key.shape[3])
        element_size = layer.key.element_size()
        return (
            2
            * len(self.layers)
            * kv_heads
            * head_dim
            * sum(self.valid_lengths)
            * element_size
        )

    @property
    def storage_signature(self) -> tuple[tuple[int, int], ...]:
        return tuple(layer.storage_signature for layer in self.layers)

    def reset(self, row_indices: Sequence[int] | None = None) -> None:
        """Invalidate rows without reallocating or mutating any model parameter."""
        if row_indices is None:
            for index in range(self.batch_size):
                self.valid_lengths[index] = 0
            return
        seen: set[int] = set()
        for row_index in row_indices:
            if not isinstance(row_index, int) or isinstance(row_index, bool):
                raise TypeError("static KV cache row indices must be integers")
            if not 0 <= row_index < self.batch_size:
                raise ValueError("static KV cache row index is out of range")
            if row_index in seen:
                raise ValueError("static KV cache row indices must be unique")
            seen.add(row_index)
            self.valid_lengths[row_index] = 0


def allocate_static_kv_cache(
    model: TwelveSixDecoder,
    *,
    batch_size: int,
    capacity: int | None = None,
) -> StaticDecoderKVCache:
    """Allocate one reusable unexpanded K/V arena for all decoder layers."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("static KV cache batch_size must be a positive integer")
    if capacity is None:
        capacity = model.spec.max_seq_len
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("static KV cache capacity must be a positive integer")
    if capacity > model.spec.max_seq_len:
        raise ValueError("static KV cache capacity exceeds ModelSpec max_seq_len")

    parameter = next(model.parameters())
    shape = (batch_size, model.spec.n_kv_heads, capacity, model.spec.head_dim)
    layers = tuple(
        StaticAttentionKVCache(
            key=torch.empty(shape, dtype=parameter.dtype, device=parameter.device),
            value=torch.empty(shape, dtype=parameter.dtype, device=parameter.device),
        )
        for _ in range(model.spec.n_layers)
    )
    return StaticDecoderKVCache(
        model_spec_sha256=model.spec.identity_sha256(),
        batch_size=batch_size,
        capacity=capacity,
        valid_lengths=[0] * batch_size,
        layers=layers,
    )


def _validate_cache(
    model: TwelveSixDecoder,
    cache: StaticDecoderKVCache,
    input_ids: Tensor,
) -> None:
    if not isinstance(cache, StaticDecoderKVCache):
        raise TypeError("cache must be a StaticDecoderKVCache")
    if cache.model_spec_sha256 != model.spec.identity_sha256():
        raise ValueError("static KV cache ModelSpec identity does not match decoder")
    if cache.batch_size != input_ids.shape[0]:
        raise ValueError("static KV cache batch size does not match decoder input")
    if cache.capacity <= 0 or cache.capacity > model.spec.max_seq_len:
        raise ValueError("static KV cache capacity is incompatible with ModelSpec")
    if len(cache.valid_lengths) != cache.batch_size:
        raise ValueError("static KV cache valid-length metadata does not match batch size")
    if len(cache.layers) != len(model.blocks):
        raise ValueError("static KV cache layer count does not match decoder")

    parameter = next(model.parameters())
    expected_shape = (
        cache.batch_size,
        model.spec.n_kv_heads,
        cache.capacity,
        model.spec.head_dim,
    )
    for layer in cache.layers:
        if layer.key.ndim != 4 or layer.value.ndim != 4 or layer.key.shape != layer.value.shape:
            raise ValueError("static K/V tensors must have identical rank-4 shapes")
        if tuple(layer.key.shape) != expected_shape:
            raise ValueError("static K/V tensor geometry is incompatible with decoder")
        if layer.key.device != parameter.device or layer.value.device != parameter.device:
            raise ValueError("static K/V tensors must be on the decoder device")
        if layer.key.dtype != parameter.dtype or layer.value.dtype != parameter.dtype:
            raise ValueError("static K/V tensors must use the decoder dtype")

    for length in cache.valid_lengths:
        if not isinstance(length, int) or isinstance(length, bool) or length < 0:
            raise ValueError("static KV cache valid lengths must be non-negative integers")
        if length > cache.capacity or length > model.spec.max_seq_len:
            raise ValueError("static KV cache valid length exceeds capacity")


def _require_equal_valid_length(cache: StaticDecoderKVCache) -> int:
    length = cache.sequence_length
    if length <= 0:
        raise ValueError("static KV cache sequence length must be positive")
    return length


def _prefill_block(
    x: Tensor,
    block: torch.nn.Module,
    layer_cache: StaticAttentionKVCache,
    *,
    sequence_length: int,
) -> Tensor:
    normed = block.attn_norm(x)
    q, k, v = block.attn._project_qkv(normed, position_offset=0)
    layer_cache.key[:, :, :sequence_length, :].copy_(k)
    layer_cache.value[:, :, :sequence_length, :].copy_(v)
    attention = block.attn._attend(
        q,
        layer_cache.key[:, :, :sequence_length, :],
        layer_cache.value[:, :, :sequence_length, :],
        is_causal=True,
    )
    x = x + attention
    return x + block.mlp(block.mlp_norm(x))


def _decode_block_one(
    x: Tensor,
    block: torch.nn.Module,
    layer_cache: StaticAttentionKVCache,
    *,
    position_offset: int,
) -> Tensor:
    normed = block.attn_norm(x)
    q, new_k, new_v = block.attn._project_qkv(normed, position_offset=position_offset)
    write_slice = slice(position_offset, position_offset + 1)
    layer_cache.key[:, :, write_slice, :].copy_(new_k)
    layer_cache.value[:, :, write_slice, :].copy_(new_v)
    valid_end = position_offset + 1
    attention = block.attn._attend(
        q,
        layer_cache.key[:, :, :valid_end, :],
        layer_cache.value[:, :, :valid_end, :],
        is_causal=False,
    )
    x = x + attention
    return x + block.mlp(block.mlp_norm(x))


@torch.no_grad()
def prefill_static_kv_cache(
    model: TwelveSixDecoder,
    input_ids: Tensor,
    cache: StaticDecoderKVCache,
) -> CausalLMOutput:
    """Prefill a reusable cache with one equal-length batch using in-place K/V writes."""
    if model.training:
        raise RuntimeError("static KV-cache inference requires model.eval()")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    sequence_length = int(input_ids.shape[1])
    if sequence_length <= 0:
        raise ValueError("input_ids sequence must be non-empty")
    _validate_cache(model, cache, input_ids)
    if sequence_length > cache.capacity:
        raise ValueError("prompt exceeds static KV cache capacity")
    if sequence_length > model.spec.max_seq_len:
        raise ValueError("prompt exceeds ModelSpec max_seq_len")

    cache.reset()
    try:
        x = model.token_embedding(input_ids)
        for block, layer_cache in zip(model.blocks, cache.layers, strict=True):
            x = _prefill_block(
                x,
                block,
                layer_cache,
                sequence_length=sequence_length,
            )
        x = model.final_norm(x)
    except Exception:
        cache.reset()
        raise

    for row_index in range(cache.batch_size):
        cache.valid_lengths[row_index] = sequence_length
    return CausalLMOutput(logits=model.lm_head(x))


@torch.no_grad()
def decode_one_with_static_kv_cache(
    model: TwelveSixDecoder,
    input_ids: Tensor,
    cache: StaticDecoderKVCache,
) -> CausalLMOutput:
    """Decode one position into existing cache storage without K/V concatenation."""
    if model.training:
        raise RuntimeError("static KV-cache inference requires model.eval()")
    if input_ids.ndim != 2 or input_ids.shape[1] != 1:
        raise ValueError("static cached decoder input_ids must have shape [batch, 1]")
    _validate_cache(model, cache, input_ids)
    position_offset = _require_equal_valid_length(cache)
    if position_offset >= cache.capacity:
        raise ValueError("static KV cache is already at fixed capacity")
    if position_offset >= model.spec.max_seq_len:
        raise ValueError("static KV cache is already at model max_seq_len")

    x = model.token_embedding(input_ids)
    for block, layer_cache in zip(model.blocks, cache.layers, strict=True):
        x = _decode_block_one(
            x,
            block,
            layer_cache,
            position_offset=position_offset,
        )
    x = model.final_norm(x)
    for row_index in range(cache.batch_size):
        cache.valid_lengths[row_index] += 1
    return CausalLMOutput(logits=model.lm_head(x))
