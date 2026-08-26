from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import tempfile
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 24
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 32
    ffn_hidden: int = 864
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    rotary_dim: int = 32
    max_seq_len: int = 1024
    default_context: int = 256
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    def validate(self):
        assert self.d_model == self.n_heads * self.head_dim
        assert self.n_heads % self.n_kv_heads == 0
        assert self.rotary_dim == self.head_dim
        assert self.default_context <= self.max_seq_len
        assert self.vocab_size == 256
        assert self.tie_word_embeddings
        assert self.dropout == 0.0

    def digest(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    cos = freqs.cos().to(x.dtype)[None, None, :, :]
    sin = freqs.sin().to(x.dtype)[None, None, :, :]
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


class StaticKVCache:
    def __init__(self, spec: ModelSpec, batch_size: int, dtype: torch.dtype, device: torch.device):
        shape = (spec.n_layers, batch_size, spec.n_kv_heads, spec.max_seq_len, spec.head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)
        self.valid_len = 0
        self.capacity = spec.max_seq_len

    @property
    def bytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()

    def reset(self):
        self.valid_len = 0


class Attention(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        qdim = spec.n_heads * spec.head_dim
        kvdim = spec.n_kv_heads * spec.head_dim
        self.q_proj = nn.Linear(spec.d_model, qdim, bias=False)
        self.k_proj = nn.Linear(spec.d_model, kvdim, bias=False)
        self.v_proj = nn.Linear(spec.d_model, kvdim, bias=False)
        self.o_proj = nn.Linear(qdim, spec.d_model, bias=False)

    def forward(self, x, layer_idx: int, start_pos: int = 0, cache: StaticKVCache | None = None):
        b, t, _ = x.shape
        s = self.spec
        q = self.q_proj(x).view(b, t, s.n_heads, s.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, s.n_kv_heads, s.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, s.n_kv_heads, s.head_dim).transpose(1, 2)
        pos = torch.arange(start_pos, start_pos + t, device=x.device)
        q = apply_rope(q, pos, s.rope_theta)
        k = apply_rope(k, pos, s.rope_theta)

        if cache is None:
            kr = k.repeat_interleave(s.n_heads // s.n_kv_heads, dim=1)
            vr = v.repeat_interleave(s.n_heads // s.n_kv_heads, dim=1)
            y = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, dropout_p=0.0)
        else:
            end = start_pos + t
            cache.k[layer_idx, :, :, start_pos:end, :].copy_(k)
            cache.v[layer_idx, :, :, start_pos:end, :].copy_(v)
            ka = cache.k[layer_idx, :, :, :end, :]
            va = cache.v[layer_idx, :, :, :end, :]
            kr = ka.repeat_interleave(s.n_heads // s.n_kv_heads, dim=1)
            vr = va.repeat_interleave(s.n_heads // s.n_kv_heads, dim=1)
            scores = torch.matmul(q, kr.transpose(-2, -1)) / math.sqrt(s.head_dim)
            qpos = torch.arange(start_pos, end, device=x.device)[:, None]
            kpos = torch.arange(0, end, device=x.device)[None, :]
            allowed = kpos <= qpos
            scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            y = torch.matmul(probs, vr)
        y = y.transpose(1, 2).contiguous().view(b, t, s.n_heads * s.head_dim)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.gate = nn.Linear(spec.d_model, spec.ffn_hidden, bias=False)
        self.up = nn.Linear(spec.d_model, spec.ffn_hidden, bias=False)
        self.down = nn.Linear(spec.ffn_hidden, spec.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.norm1 = RMSNorm(spec.d_model, spec.rms_eps)
        self.attn = Attention(spec)
        self.norm2 = RMSNorm(spec.d_model, spec.rms_eps)
        self.mlp = MLP(spec)

    def forward(self, x, layer_idx: int, start_pos: int = 0, cache: StaticKVCache | None = None):
        x = x + self.attn(self.norm1(x), layer_idx, start_pos, cache)
        x = x + self.mlp(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.tok_embeddings = nn.Embedding(spec.vocab_size, spec.d_model)
        self.blocks = nn.ModuleList([Block(spec) for _ in range(spec.n_layers)])
        self.final_norm = RMSNorm(spec.d_model, spec.rms_eps)

    def forward(self, ids: torch.Tensor, start_pos: int = 0, cache: StaticKVCache | None = None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("input_ids must be [batch, seq>=1]")
        end = start_pos + ids.shape[1]
        if start_pos < 0 or end > self.spec.max_seq_len:
            raise ValueError(f"context overflow: start={start_pos} len={ids.shape[1]} max={self.spec.max_seq_len}")
        x = self.tok_embeddings(ids)
        for i, block in enumerate(self.blocks):
            x = block(x, i, start_pos, cache)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_embeddings.weight)
        if cache is not None:
            cache.valid_len = max(cache.valid_len, end)
        return logits


def bytes_mib(n):
    return n / (1024**2)


def main():
    torch.manual_seed(342)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    s = ModelSpec()
    t0 = time.perf_counter()
    model = Model(s)
    build_s = time.perf_counter() - t0

    n = sum(p.numel() for p in model.parameters())
    expected = 19_935_488
    assert n == expected, (n, expected)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    named_shapes = {k: list(v.shape) for k, v in model.state_dict().items()}

    ids = torch.randint(0, 256, (1, 17), dtype=torch.long)
    q_before = model.blocks[0].attn.q_proj.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    logits = model(ids)
    assert logits.shape == (1, 17, 256)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, 256), ids[:, 1:].reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    assert grad_bytes == param_bytes
    assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    opt.step()
    delta = (model.blocks[0].attn.q_proj.weight.detach() - q_before).abs().max().item()
    assert delta > 0.0
    opt_state_bytes = 0
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                opt_state_bytes += value.numel() * value.element_size()
    opt.zero_grad(set_to_none=True)

    tokenizer_manifest = {
        "type": "byte",
        "vocab_size": 256,
        "id_range": [0, 255],
        "special_tokens": {},
    }
    probe = torch.tensor([[0, 1, 2, 127, 128, 254, 255]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        ref = model(probe)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model342.pt")
        torch.save({"model_spec": asdict(s), "model_spec_sha256": s.digest(), "tokenizer": tokenizer_manifest, "model": model.state_dict()}, path)
        checkpoint_bytes = os.path.getsize(path)
        loaded_obj = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded_obj["model_spec_sha256"] == s.digest()
        assert loaded_obj["tokenizer"] == tokenizer_manifest
        reloaded = Model(ModelSpec(**loaded_obj["model_spec"]))
        reloaded.load_state_dict(loaded_obj["model"], strict=True)
        reloaded.eval()
        with torch.no_grad():
            reload_logits = reloaded(probe)
        reload_max_abs = (ref - reload_logits).abs().max().item()
        assert reload_max_abs == 0.0

    cache = StaticKVCache(s, batch_size=1, dtype=torch.float32, device=torch.device("cpu"))
    k_ptr, v_ptr = cache.k.data_ptr(), cache.v.data_ptr()
    prompt = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long)
    with torch.no_grad():
        full_prompt = model(prompt)
        cached_prompt = model(prompt, start_pos=0, cache=cache)
    prefill_max_abs = (full_prompt - cached_prompt).abs().max().item()
    assert prefill_max_abs < 5e-5, prefill_max_abs
    assert cache.k.data_ptr() == k_ptr and cache.v.data_ptr() == v_ptr
    assert cache.k.shape == (24, 1, 2, 1024, 32)
    assert cache.valid_len == 8

    next_id = torch.tensor([[90]], dtype=torch.long)
    with torch.no_grad():
        full_next = model(torch.cat([prompt, next_id], dim=1))[:, -1:, :]
        cached_next = model(next_id, start_pos=8, cache=cache)
    decode_max_abs = (full_next - cached_next).abs().max().item()
    assert decode_max_abs < 5e-5, decode_max_abs
    assert cache.valid_len == 9
    assert cache.k.data_ptr() == k_ptr and cache.v.data_ptr() == v_ptr

    cache.reset()
    assert cache.valid_len == 0
    assert cache.k.data_ptr() == k_ptr and cache.v.data_ptr() == v_ptr

    cached_boundary_finite = True
    with torch.no_grad():
        for chunk_start in range(0, s.max_seq_len, 256):
            chunk = torch.randint(0, 256, (1, 256), dtype=torch.long)
            chunk_logits = model(chunk, start_pos=chunk_start, cache=cache)
            cached_boundary_finite = cached_boundary_finite and bool(torch.isfinite(chunk_logits).all())
    assert cache.valid_len == 1024
    assert cached_boundary_finite
    assert cache.k.data_ptr() == k_ptr and cache.v.data_ptr() == v_ptr
    cache_overflow_rejected = False
    try:
        model(torch.zeros((1, 1), dtype=torch.long), start_pos=1024, cache=cache)
    except ValueError:
        cache_overflow_rejected = True
    assert cache_overflow_rejected

    boundary_ids = torch.randint(0, 256, (1, s.max_seq_len), dtype=torch.long)
    tb = time.perf_counter()
    with torch.no_grad():
        boundary_logits = model(boundary_ids)
    boundary_s = time.perf_counter() - tb
    assert boundary_logits.shape == (1, 1024, 256)
    assert torch.isfinite(boundary_logits).all()
    overflow_rejected = False
    try:
        model(torch.zeros((1, 1025), dtype=torch.long))
    except ValueError:
        overflow_rejected = True
    assert overflow_rejected

    bf16_param_bytes = n * 2
    fp32_cache_bytes = cache.bytes
    bf16_cache_bytes = fp32_cache_bytes // 2
    result = {
        "worker_id": "MODEL-342-20M-CONTROL-B",
        "geometry": asdict(s),
        "model_spec_sha256": s.digest(),
        "parameter_count": n,
        "parameter_tensor_count": len(list(model.parameters())),
        "parameter_bytes_fp32": param_bytes,
        "parameter_mib_fp32": bytes_mib(param_bytes),
        "parameter_bytes_bf16": bf16_param_bytes,
        "parameter_mib_bf16": bytes_mib(bf16_param_bytes),
        "gradient_bytes_fp32_after_backward": grad_bytes,
        "optimizer_state_bytes_fp32_adamw_after_first_step": opt_state_bytes,
        "optimizer_state_mib": bytes_mib(opt_state_bytes),
        "update_q_proj_max_abs_delta": delta,
        "smoke_loss": float(loss.detach()),
        "checkpoint_file_bytes_torch_save": checkpoint_bytes,
        "reload_max_abs_logit_diff": reload_max_abs,
        "static_kv_shape_each_kv": list(cache.k.shape),
        "static_kv_bytes_fp32": fp32_cache_bytes,
        "static_kv_mib_fp32": bytes_mib(fp32_cache_bytes),
        "static_kv_bytes_bf16": bf16_cache_bytes,
        "static_kv_mib_bf16": bytes_mib(bf16_cache_bytes),
        "cache_prefill_max_abs_diff_vs_full": prefill_max_abs,
        "cache_decode_max_abs_diff_vs_full": decode_max_abs,
        "cache_pointer_stable": True,
        "cache_unexpanded_kv_heads": 2,
        "cache_boundary_1024_finite": cached_boundary_finite,
        "cache_overflow_rejected": cache_overflow_rejected,
        "boundary_1024_finite": True,
        "overflow_1025_rejected": overflow_rejected,
        "build_seconds": build_s,
        "boundary_forward_seconds_cpu": boundary_s,
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "state_dict_tensor_shapes_count": len(named_shapes),
        "random_init_seed": 342,
        "long_training": False,
        "paid_compute": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
