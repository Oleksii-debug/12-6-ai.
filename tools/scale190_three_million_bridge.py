#!/usr/bin/env python3
"""SCALE-190 LOCAL_FREE fixed-control ~3.2M bridge experiment.

Self-contained research harness that reproduces the RESEARCH41 fixed-control
architecture, tokenizer, S0 cyclic byte packing, optimizer, evaluation, and
token-accounting semantics. It deliberately does not alter the frozen
RESEARCH138 prediction after observing SCALE190 results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SCHEMA = "12-6.scale190.bridge.v1"
AUTHORITY = "LOCAL_FREE_RESEARCH_ONLY_NOT_PROMOTION"
LN2 = math.log(2.0)
TOKENIZER_ID = "s0-byte-v1"
TOKENIZER_CONFIG_SHA256 = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
TOKENIZER_VOCAB_SHA256 = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
CORPUS_DATASET_ID = "s0-tiny-controlled-v1"
CORPUS_IDENTITY_SHA256 = "bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89"
CORPUS_MANIFEST_SHA256 = "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"
TRAIN_JSONL_SHA256 = "61d24b7138df56527d201cea405d11c9f607684b4a9593dfa20c599cc2ee6998"
VALIDATION_JSONL_SHA256 = "57f18a846dcca75955a82612382d4635ba9583965aa6628e77626cd2a3eb19c5"
PACKING_ID = "research41-byte-stream-cyclic-v1"
PREDICTION_REPORT_SHA256 = "05c17878b770a1c621101986c4e66c3265967df647e7bc087e43e3752e53162a"
PREDICTION_INPUT_SHA256 = "af831fd461861ab1dc35b9fd8e4347d70071b071216fa36ee8536e5dcc5af66c"
PREDICTION_MODEL = "log_power"
PREDICTION_COEFFICIENTS = (1.2204377537331623, -0.11797847077904858, -0.2668976014889022)
PREDICTION_N0 = 333895.53593930375
PREDICTION_T0 = 16734.4584362899
CHECKPOINT_TOKENS = (16_632, 65_772, 131_292)
CHECKPOINT_STEPS = (66, 261, 521)
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
TOKEN_QUANTUM = BATCH_SIZE * (SEQUENCE_LENGTH - 1)
RESUME_TOKENS = 65_772
RESUME_STEP = 261

BRIDGE_SPEC_DICT = {
    "schema_version": 1,
    "vocab_size": 256,
    "max_seq_len": 256,
    "d_model": 192,
    "n_layers": 7,
    "n_heads": 8,
    "n_kv_heads": 8,
    "head_dim": 24,
    "d_ff": 530,
    "activation": "swiglu",
    "norm_kind": "rmsnorm",
    "norm_placement": "pre",
    "norm_eps": 1e-5,
    "position_embedding": "rope",
    "rope_theta": 10_000.0,
    "rope_rotary_dim": 24,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "final_norm": True,
    "tie_word_embeddings": True,
    "lm_head_bias": False,
}
BRIDGE_MODEL_SHA256 = "37b7fdd44b35280c121f9300022bfd69b23efbf0abbcfe62fbb0eb465470b693"
INIT_DICT = {
    "schema_version": 1,
    "family": "normal",
    "std": 0.02,
    "residual_branch_scale": "sqrt_2_layers",
}
INIT_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"

REFERENCE_SPEC_DICT = {
    **BRIDGE_SPEC_DICT,
    "d_model": 48,
    "n_layers": 3,
    "n_heads": 4,
    "n_kv_heads": 4,
    "head_dim": 12,
    "d_ff": 128,
    "rope_rotary_dim": 12,
}
REFERENCE_EXPECTED_BPB_65772 = 3.875612846985032

def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

@dataclass(frozen=True)
class ModelSpec:
    schema_version: int
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    activation: str = "swiglu"
    norm_kind: str = "rmsnorm"
    norm_placement: str = "pre"
    norm_eps: float = 1e-5
    position_embedding: str = "rope"
    rope_theta: float = 10_000.0
    rope_rotary_dim: int = 0
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    final_norm: bool = True
    tie_word_embeddings: bool = True
    lm_head_bias: bool = False

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def parameter_count(self) -> int:
        embedding = self.vocab_size * self.d_model
        attention_weights_per_layer = 2 * self.d_model * (self.q_dim + self.kv_dim)
        attention_biases_per_layer = self.q_dim + 2 * self.kv_dim + self.d_model if self.attention_bias else 0
        mlp_weights_per_layer = 3 * self.d_model * self.d_ff
        mlp_biases_per_layer = 2 * self.d_ff + self.d_model if self.mlp_bias else 0
        norms_per_layer = 2 * self.d_model
        block = attention_weights_per_layer + attention_biases_per_layer + mlp_weights_per_layer + mlp_biases_per_layer + norms_per_layer
        final_norm = self.d_model if self.final_norm else 0
        lm_head_extra = 0 if self.tie_word_embeddings else self.vocab_size * self.d_model
        if self.lm_head_bias:
            lm_head_extra += self.vocab_size
        return embedding + self.n_layers * block + final_norm + lm_head_extra

    def identity_sha256(self) -> str:
        return canonical_hash(asdict(self))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def cos_sin(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.repeat_interleave(freqs, 2, dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)

def rotate_pairs(x: Tensor) -> Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, rotary_dim: int) -> Tensor:
    rotary = x[..., :rotary_dim]
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1])
    rotated = rotary * cos + rotate_pairs(rotary) * sin
    return rotated if rotary_dim == x.shape[-1] else torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.n_heads = spec.n_heads
        self.n_kv_heads = spec.n_kv_heads
        self.head_dim = spec.head_dim
        self.q_dim = spec.q_dim
        self.kv_dim = spec.kv_dim
        self.rotary_dim = spec.rope_rotary_dim
        self.dropout = spec.attention_dropout
        self.q_proj = nn.Linear(spec.d_model, spec.q_dim, bias=spec.attention_bias)
        self.k_proj = nn.Linear(spec.d_model, spec.kv_dim, bias=spec.attention_bias)
        self.v_proj = nn.Linear(spec.d_model, spec.kv_dim, bias=spec.attention_bias)
        self.out_proj = nn.Linear(spec.q_dim, spec.d_model, bias=spec.attention_bias)
        self.rope = RotaryEmbedding(spec.rope_rotary_dim, spec.rope_theta)
    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope.cos_sin(seq_len, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin, self.rotary_dim)
        k = apply_rope(k, cos, sin, self.rotary_dim)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.q_dim)
        return self.out_proj(attended)

class SwiGLU(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(spec.d_model, spec.d_ff, bias=spec.mlp_bias)
        self.up_proj = nn.Linear(spec.d_model, spec.d_ff, bias=spec.mlp_bias)
        self.down_proj = nn.Linear(spec.d_ff, spec.d_model, bias=spec.mlp_bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(spec.d_model, spec.norm_eps)
        self.attn = CausalSelfAttention(spec)
        self.mlp_norm = RMSNorm(spec.d_model, spec.norm_eps)
        self.mlp = SwiGLU(spec)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class Decoder(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(spec) for _ in range(spec.n_layers))
        self.final_norm = RMSNorm(spec.d_model, spec.norm_eps) if spec.final_norm else nn.Identity()
        self.lm_head = nn.Linear(spec.d_model, spec.vocab_size, bias=spec.lm_head_bias)
        self.apply(self._init_module)
        residual_std = INIT_DICT["std"] / math.sqrt(2.0 * spec.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)
        if spec.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        actual = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if actual != spec.parameter_count():
            raise RuntimeError(f"parameter count drift: model={actual}, spec={spec.parameter_count()}")
    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_DICT["std"])
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def assert_data(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes]:
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    val_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    observed = {"train_jsonl_sha256": file_sha256(train_path), "evaluation_jsonl_sha256": file_sha256(val_path), "corpus_manifest_sha256": file_sha256(manifest_path)}
    expected = {"train_jsonl_sha256": TRAIN_JSONL_SHA256, "evaluation_jsonl_sha256": VALIDATION_JSONL_SHA256, "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256}
    if observed != expected:
        raise RuntimeError(f"data identity mismatch: observed={observed!r}, expected={expected!r}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != CORPUS_DATASET_ID or manifest.get("dataset_identity_sha256") != CORPUS_IDENTITY_SHA256:
        raise RuntimeError("dataset identity drift")
    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    if {str(r["id"]) for r in train_records} & {str(r["id"]) for r in val_records}:
        raise RuntimeError("train/eval record overlap")
    stream = b"\n".join(str(r["text"]).encode("utf-8") for r in train_records) + b"\n"
    if len(stream) != 1930:
        raise RuntimeError(f"unexpected cyclic stream bytes: {len(stream)}")
    return train_records, val_records, stream

def make_batch(stream: bytes, step: int) -> Tensor:
    width = BATCH_SIZE * SEQUENCE_LENGTH
    base = (step * width) % len(stream)
    rows = []
    for batch_index in range(BATCH_SIZE):
        start = (base + batch_index * SEQUENCE_LENGTH) % len(stream)
        rows.append([stream[(start + offset) % len(stream)] for offset in range(SEQUENCE_LENGTH)])
    return torch.tensor(rows, dtype=torch.long)

@torch.no_grad()
def validation_loss(model: Decoder, records: list[dict[str, Any]]) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for record in records:
        ids = list(str(record["text"]).encode("utf-8"))
        start = 0
        while start < len(ids) - 1:
            chunk = ids[start:start + model.spec.max_seq_len]
            if len(chunk) < 2:
                break
            x = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, model.spec.vocab_size), x[:, 1:].reshape(-1), reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(x.shape[1] - 1)
            start += model.spec.max_seq_len - 1
    model.train(was_training)
    return total_loss / total_tokens, total_tokens

def tensor_state_sha(state: Mapping[str, Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(json.dumps(list(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()

def recursive_state_sha(value: Any) -> str:
    h = hashlib.sha256()
    def walk(v: Any) -> None:
        if isinstance(v, Tensor):
            t = v.detach().cpu().contiguous(); h.update(b"T"); h.update(str(t.dtype).encode()); h.update(json.dumps(list(t.shape)).encode()); h.update(t.numpy().tobytes())
        elif isinstance(v, Mapping):
            h.update(b"M")
            for k in sorted(v, key=lambda x: repr(x)):
                h.update(repr(k).encode()); walk(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(b"S")
            for item in v: walk(item)
        else:
            h.update(b"V"); h.update(repr(v).encode())
    walk(value)
    return h.hexdigest()

def tensor_bytes(value: Any) -> int:
    if isinstance(value, Tensor): return value.numel() * value.element_size()
    if isinstance(value, Mapping): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)): return sum(tensor_bytes(v) for v in value)
    return 0

def rss_hwm_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024

def stats(values: Sequence[float]) -> dict[str, float]:
    if not values: return {"min": math.nan, "max": math.nan, "mean": math.nan, "last": math.nan}
    return {"min": min(values), "max": max(values), "mean": fmean(values), "last": values[-1]}

def predict(spec: ModelSpec, optimized_tokens: int) -> tuple[float, float]:
    b0, bn, bt = PREDICTION_COEFFICIENTS
    loss = math.exp(b0 + bn * math.log(spec.parameter_count() / PREDICTION_N0) + bt * math.log(optimized_tokens / PREDICTION_T0))
    return loss, loss / LN2

def _activation_row(name: str, x: Tensor) -> dict[str, Any]:
    y = x.detach().float()
    return {"name": name, "finite": bool(torch.isfinite(y).all().item()), "rms": float(torch.sqrt(torch.mean(y * y)).item()), "max_abs": float(y.abs().max().item()), "mean": float(y.mean().item()), "std": float(y.std(unbiased=False).item())}

def activation_health(model: Decoder, records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = list(str(records[0]["text"]).encode("utf-8"))[:model.spec.max_seq_len]
    x_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    was_training = model.training; model.eval(); rows = []
    with torch.no_grad():
        x = model.token_embedding(x_ids); rows.append(_activation_row("embedding", x))
        for index, block in enumerate(model.blocks):
            x = block(x); row = _activation_row(f"block_{index}", x)
            joined = torch.cat([p.detach().float().reshape(-1) for p in block.parameters()])
            row["weight_rms"] = float(torch.sqrt(torch.mean(joined * joined)).item()); rows.append(row)
        x = model.final_norm(x); rows.append(_activation_row("final_norm", x))
    model.train(was_training)
    return {"layers": rows, "all_finite": all(row["finite"] for row in rows), "activation_rms_min": min(row["rms"] for row in rows), "activation_rms_max": max(row["rms"] for row in rows), "activation_max_abs_max": max(row["max_abs"] for row in rows)}

def per_layer_grad_norms(model: Decoder) -> dict[str, float]:
    groups = {}
    named_groups = [("embedding", model.token_embedding.parameters()), *[(f"block_{i}", block.parameters()) for i, block in enumerate(model.blocks)], ("final_norm", model.final_norm.parameters())]
    for name, params in named_groups:
        sq = 0.0
        for p in params:
            if p.grad is not None:
                g = p.grad.detach().float(); sq += float(torch.sum(g * g).item())
        groups[name] = math.sqrt(sq)
    return groups

def checkpoint_payload(*, model: Decoder, optimizer: torch.optim.Optimizer, seed: int, step: int, tokens: int, source_sha: str, before_eval_bpb: float) -> dict[str, Any]:
    model_state = model.state_dict(); optimizer_state = optimizer.state_dict()
    return {"schema":"12-6.scale190.research-checkpoint.v1","lineage":{"source_sha":source_sha,"model_spec":asdict(model.spec),"model_identity_sha256":model.spec.identity_sha256(),"parameter_count":model.spec.parameter_count(),"init_identity_sha256":INIT_SHA256,"tokenizer_id":TOKENIZER_ID,"tokenizer_config_sha256":TOKENIZER_CONFIG_SHA256,"tokenizer_vocab_sha256":TOKENIZER_VOCAB_SHA256,"corpus_identity_sha256":CORPUS_IDENTITY_SHA256,"corpus_manifest_sha256":CORPUS_MANIFEST_SHA256,"train_jsonl_sha256":TRAIN_JSONL_SHA256,"evaluation_jsonl_sha256":VALIDATION_JSONL_SHA256,"packing_id":PACKING_ID,"seed":seed,"precision":"fp32","optimizer":{"name":"AdamW","learning_rate":3e-4,"betas":[0.9,0.95],"eps":1e-8,"weight_decay":0.0,"gradient_clip_norm":1.0,"scheduler":"constant"},"step":step,"optimized_tokens":tokens},"model":model_state,"optimizer":optimizer_state,"rng":{"python":random.getstate(),"torch":torch.random.get_rng_state()},"state_hashes":{"model_state_sha256":tensor_state_sha(model_state),"optimizer_state_sha256":recursive_state_sha(optimizer_state)},"heldout_bpb_at_save":before_eval_bpb}

def save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)
    return {"path":str(path),"sha256":file_sha256(path),"bytes":path.stat().st_size,"model_state_sha256":payload["state_hashes"]["model_state_sha256"],"optimizer_state_sha256":payload["state_hashes"]["optimizer_state_sha256"]}

def load_checkpoint(path: Path, *, model: Decoder, optimizer: torch.optim.Optimizer, expected_source_sha: str, expected_seed: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False); lineage = payload["lineage"]
    expected = {"source_sha":expected_source_sha,"model_identity_sha256":model.spec.identity_sha256(),"parameter_count":model.spec.parameter_count(),"tokenizer_config_sha256":TOKENIZER_CONFIG_SHA256,"tokenizer_vocab_sha256":TOKENIZER_VOCAB_SHA256,"corpus_identity_sha256":CORPUS_IDENTITY_SHA256,"corpus_manifest_sha256":CORPUS_MANIFEST_SHA256,"train_jsonl_sha256":TRAIN_JSONL_SHA256,"evaluation_jsonl_sha256":VALIDATION_JSONL_SHA256,"seed":expected_seed}
    for key, value in expected.items():
        if lineage.get(key) != value: raise RuntimeError(f"checkpoint lineage mismatch for {key}: {lineage.get(key)!r} != {value!r}")
    model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); random.setstate(payload["rng"]["python"]); torch.random.set_rng_state(payload["rng"]["torch"])
    if tensor_state_sha(model.state_dict()) != payload["state_hashes"]["model_state_sha256"]: raise RuntimeError("fresh-process model state hash mismatch")
    if recursive_state_sha(optimizer.state_dict()) != payload["state_hashes"]["optimizer_state_sha256"]: raise RuntimeError("fresh-process optimizer state hash mismatch")
    return payload

def build_model_optimizer(spec: ModelSpec, seed: int) -> tuple[Decoder, torch.optim.Optimizer]:
    random.seed(seed); torch.manual_seed(seed); model = Decoder(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    return model, optimizer

def run_segment(*, repo_root: Path, outdir: Path, source_sha: str, seed: int, spec: ModelSpec, start_step: int, end_step: int, resume_checkpoint: Path | None, expected_resume_bpb: float | None) -> dict[str, Any]:
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); _, val_records, stream = assert_data(repo_root); model, optimizer = build_model_optimizer(spec, seed)
    resume_proof = None
    if resume_checkpoint is not None:
        loaded = load_checkpoint(resume_checkpoint, model=model, optimizer=optimizer, expected_source_sha=source_sha, expected_seed=seed)
        if int(loaded["lineage"]["step"]) != start_step or int(loaded["lineage"]["optimized_tokens"]) != start_step * TOKEN_QUANTUM: raise RuntimeError("resume counters do not match requested segment")
        loss, _ = validation_loss(model, val_records); loaded_bpb = loss / LN2
        if expected_resume_bpb is not None and loaded_bpb != expected_resume_bpb: raise RuntimeError(f"fresh-process resume eval drift: {loaded_bpb!r} != {expected_resume_bpb!r}")
        resume_proof = {"fresh_process_pid":os.getpid(),"checkpoint_sha256":file_sha256(resume_checkpoint),"model_state_sha256_verified":loaded["state_hashes"]["model_state_sha256"],"optimizer_state_sha256_verified":loaded["state_hashes"]["optimizer_state_sha256"],"heldout_bpb_before_save":loaded["heldout_bpb_at_save"],"heldout_bpb_after_fresh_load":loaded_bpb,"heldout_bpb_bit_equal":loaded_bpb == loaded["heldout_bpb_at_save"],"rng_restored":True}
    grad_norms=[]; update_ratios=[]; losses=[]; clip_events=0; optimization_seconds=0.0; checkpoints=[]; checkpoint_metrics=[]
    checkpoint_step_set = {s for s in CHECKPOINT_STEPS if start_step < s <= end_step}; segment_started = time.perf_counter()
    for step_zero in range(start_step, end_step):
        batch = make_batch(stream, step_zero); before = [p.detach().clone() for p in model.parameters()]; train_started = time.perf_counter(); model.train(); logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, spec.vocab_size), batch[:, 1:].reshape(-1))
        if not torch.isfinite(loss).item(): raise FloatingPointError(f"non-finite loss at step {step_zero + 1}")
        (loss * TOKEN_QUANTUM).backward(); sq = torch.zeros((), dtype=torch.float32)
        for p in model.parameters():
            if p.grad is None: continue
            if not torch.isfinite(p.grad).all().item(): raise FloatingPointError(f"non-finite gradient at step {step_zero + 1}")
            p.grad.div_(TOKEN_QUANTUM); g = p.grad.detach().float(); sq += torch.sum(g * g)
        raw_grad_norm = float(torch.sqrt(sq).item()); layer_grads = per_layer_grad_norms(model) if (step_zero + 1) in checkpoint_step_set else None
        if raw_grad_norm > 1.0: clip_events += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step(); optimizer.zero_grad(set_to_none=True); optimization_seconds += time.perf_counter() - train_started
        losses.append(float(loss.item())); grad_norms.append(raw_grad_norm); delta_sq=0.0; weight_sq=0.0
        for p0, p in zip(before, model.parameters(), strict=True):
            delta = p.detach() - p0; delta_sq += float(torch.sum(delta.double() * delta.double()).item()); weight_sq += float(torch.sum(p.detach().double() * p.detach().double()).item())
        update_ratios.append(math.sqrt(delta_sq) / math.sqrt(weight_sq)); step = step_zero + 1; tokens = step * TOKEN_QUANTUM
        if step in checkpoint_step_set:
            eval_started=time.perf_counter(); val_loss,val_tokens=validation_loss(model,val_records); eval_seconds=time.perf_counter()-eval_started; bpb=val_loss/LN2; pred_loss,pred_bpb=predict(spec,tokens); health=activation_health(model,val_records)
            ckpt_path=outdir/"checkpoints"/f"seed-{seed}-tokens-{tokens}.pt"; payload=checkpoint_payload(model=model,optimizer=optimizer,seed=seed,step=step,tokens=tokens,source_sha=source_sha,before_eval_bpb=bpb); checkpoint_info=save_checkpoint(ckpt_path,payload); checkpoints.append(checkpoint_info)
            row={"seed":seed,"optimizer_step":step,"optimized_tokens":tokens,"heldout_loss_nats":val_loss,"heldout_bpb":bpb,"validation_tokens":val_tokens,"eval_seconds":eval_seconds,"compute_proxy":6*spec.parameter_count()*tokens,"frozen_predicted_loss_nats":pred_loss,"frozen_predicted_bpb":pred_bpb,"observed_minus_predicted_loss_nats":val_loss-pred_loss,"observed_minus_predicted_bpb":bpb-pred_bpb,"raw_gradient_norm_at_checkpoint":raw_grad_norm,"per_layer_raw_gradient_norm_at_checkpoint":layer_grads,"activation_health":health,"clip_events_cumulative_in_segment":clip_events,"clip_fraction_cumulative_in_segment":clip_events/(step-start_step),"update_ratio_window":stats(update_ratios),"gradient_norm_window":stats(grad_norms),"train_loss_window":stats(losses),"optimization_seconds_segment_cumulative":optimization_seconds,"optimized_tokens_per_optimization_second_segment":((step-start_step)*TOKEN_QUANTUM/optimization_seconds),"peak_rss_bytes":rss_hwm_bytes(),"checkpoint":checkpoint_info}; checkpoint_metrics.append(row)
            print(f"CHECKPOINT seed={seed} tokens={tokens} bpb={bpb:.12f} pred={pred_bpb:.12f} delta={bpb-pred_bpb:+.12f} clip={clip_events}/{step-start_step}",flush=True)
    final_optimizer_state=optimizer.state_dict(); result={"schema":"12-6.scale190.segment.v1","pid":os.getpid(),"source_sha":source_sha,"seed":seed,"start_step":start_step,"end_step":end_step,"start_optimized_tokens":start_step*TOKEN_QUANTUM,"end_optimized_tokens":end_step*TOKEN_QUANTUM,"model_identity_sha256":spec.identity_sha256(),"parameter_count":spec.parameter_count(),"resume_proof":resume_proof,"checkpoint_metrics":checkpoint_metrics,"checkpoints":checkpoints,"optimization":{"raw_gradient_norm":stats(grad_norms),"update_to_weight_l2_ratio":stats(update_ratios),"train_loss":stats(losses),"clip_events":clip_events,"clip_fraction":clip_events/max(1,end_step-start_step),"optimization_wall_seconds":optimization_seconds,"optimized_tokens_per_optimization_second":((end_step-start_step)*TOKEN_QUANTUM/optimization_seconds)},"memory":{"peak_process_rss_bytes":rss_hwm_bytes(),"model_parameter_tensor_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),"optimizer_state_tensor_bytes":tensor_bytes(final_optimizer_state)},"state_hashes":{"final_model_state_sha256":tensor_state_sha(model.state_dict()),"final_optimizer_state_sha256":recursive_state_sha(final_optimizer_state)},"segment_wall_seconds":time.perf_counter()-segment_started}; return result

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")

def worker(args: argparse.Namespace) -> int:
    spec=ModelSpec(**(REFERENCE_SPEC_DICT if args.reference else BRIDGE_SPEC_DICT))
    if not args.reference and (spec.parameter_count()!=3_221_184 or spec.identity_sha256()!=BRIDGE_MODEL_SHA256): raise RuntimeError("bridge ModelSpec drift")
    result=run_segment(repo_root=Path(args.repo_root),outdir=Path(args.outdir),source_sha=args.source_sha,seed=args.seed,spec=spec,start_step=args.start_step,end_step=args.end_step,resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,expected_resume_bpb=args.expected_resume_bpb); write_json(Path(args.output),result); return 0

def run_child(cmd: list[str]) -> None:
    completed=subprocess.run(cmd,check=False)
    if completed.returncode: raise RuntimeError(f"child process failed with exit code {completed.returncode}: {cmd!r}")

def orchestrate(args: argparse.Namespace) -> int:
    repo_root=Path(args.repo_root).resolve(); outdir=Path(args.outdir).resolve(); outdir.mkdir(parents=True,exist_ok=True); spec=ModelSpec(**BRIDGE_SPEC_DICT); assert_data(repo_root)
    if spec.parameter_count()!=3_221_184 or spec.identity_sha256()!=BRIDGE_MODEL_SHA256: raise RuntimeError("preregistered bridge spec drift")
    if canonical_hash(INIT_DICT)!=INIT_SHA256: raise RuntimeError("InitSpec identity drift")
    prereg={"schema":"12-6.scale190.runtime-prereg.v1","source_sha":args.source_sha,"seed":args.seed,"model_spec":asdict(spec),"model_identity_sha256":spec.identity_sha256(),"parameter_count":spec.parameter_count(),"checkpoint_tokens":list(CHECKPOINT_TOKENS),"checkpoint_steps":list(CHECKPOINT_STEPS),"resume_tokens":RESUME_TOKENS,"prediction_report_sha256":PREDICTION_REPORT_SHA256,"prediction_input_sha256":PREDICTION_INPUT_SHA256,"frozen_predictions":[{"optimized_tokens":t,"loss_nats":predict(spec,t)[0],"bpb":predict(spec,t)[1]} for t in CHECKPOINT_TOKENS],"identities":{"tokenizer":TOKENIZER_CONFIG_SHA256,"tokenizer_vocab":TOKENIZER_VOCAB_SHA256,"corpus":CORPUS_IDENTITY_SHA256,"corpus_manifest":CORPUS_MANIFEST_SHA256,"train_jsonl":TRAIN_JSONL_SHA256,"evaluation_jsonl":VALIDATION_JSONL_SHA256,"packing":PACKING_ID},"optimizer":{"name":"AdamW","lr":3e-4,"betas":[0.9,0.95],"eps":1e-8,"weight_decay":0.0,"gradient_clip_norm":1.0,"scheduler":"constant","precision":"fp32"},"comparison_rule":"frozen RESEARCH138 log_power prediction; no refit after SCALE190 observation"}; write_json(outdir/f"seed-{args.seed}-runtime-prereg.json",prereg)
    phase1=outdir/f"seed-{args.seed}-phase1.json"; phase2=outdir/f"seed-{args.seed}-phase2.json"; base=[sys.executable,str(Path(__file__).resolve()),"worker","--repo-root",str(repo_root),"--outdir",str(outdir),"--source-sha",args.source_sha,"--seed",str(args.seed)]
    run_child(base+["--start-step","0","--end-step",str(RESUME_STEP),"--output",str(phase1)]); p1=json.loads(phase1.read_text(encoding="utf-8")); resume_metric=next(row for row in p1["checkpoint_metrics"] if row["optimized_tokens"]==RESUME_TOKENS); resume_ckpt=Path(resume_metric["checkpoint"]["path"]); resume_bpb=float(resume_metric["heldout_bpb"])
    run_child(base+["--start-step",str(RESUME_STEP),"--end-step",str(CHECKPOINT_STEPS[-1]),"--resume-checkpoint",str(resume_ckpt),"--expected-resume-bpb",repr(resume_bpb),"--output",str(phase2)]); p2=json.loads(phase2.read_text(encoding="utf-8")); rows=p1["checkpoint_metrics"]+p2["checkpoint_metrics"]
    if [row["optimized_tokens"] for row in rows]!=list(CHECKPOINT_TOKENS): raise RuntimeError("checkpoint trajectory drift")
    resume=p2["resume_proof"]
    if not resume or not resume["heldout_bpb_bit_equal"]: raise RuntimeError("fresh-process resume proof failed")
    report={"schema":SCHEMA,"authority":AUTHORITY,"source_sha":args.source_sha,"seed":args.seed,"model":{"spec":asdict(spec),"model_identity_sha256":spec.identity_sha256(),"parameters":spec.parameter_count(),"ideal_parameters":3_221_432,"delta_from_ideal_parameters":spec.parameter_count()-3_221_432,"relative_delta_from_ideal":(spec.parameter_count()-3_221_432)/3_221_432,"init_identity_sha256":INIT_SHA256},"shared_identity":prereg["identities"],"optimizer":prereg["optimizer"],"frozen_prediction":{"report_sha256":PREDICTION_REPORT_SHA256,"input_payload_sha256":PREDICTION_INPUT_SHA256,"model":PREDICTION_MODEL,"coefficients":list(PREDICTION_COEFFICIENTS),"n0":PREDICTION_N0,"t0":PREDICTION_T0,"retuned_after_observation":False},"trajectory":rows,"fresh_process_resume":resume,"timing":{"optimization_wall_seconds_total":p1["optimization"]["optimization_wall_seconds"]+p2["optimization"]["optimization_wall_seconds"],"segment_wall_seconds_total":p1["segment_wall_seconds"]+p2["segment_wall_seconds"],"optimized_tokens_per_optimization_second_total":CHECKPOINT_TOKENS[-1]/(p1["optimization"]["optimization_wall_seconds"]+p2["optimization"]["optimization_wall_seconds"])},"memory":{"peak_process_rss_bytes_max":max(p1["memory"]["peak_process_rss_bytes"],p2["memory"]["peak_process_rss_bytes"]),"model_parameter_tensor_bytes":p2["memory"]["model_parameter_tensor_bytes"],"optimizer_state_tensor_bytes":p2["memory"]["optimizer_state_tensor_bytes"]},"checkpoint_files":p1["checkpoints"]+p2["checkpoints"],"environment":{"python":sys.version,"torch":torch.__version__,"platform":platform.platform(),"torch_threads":2,"deterministic_algorithms":True,"device":"cpu","paid_compute":False},"truth_boundary":{"repeated_tiny_fixture":True,"broad_corpus_claim":False,"universal_scaling_law_claim":False,"stage_promotion":False,"paid_compute":False}}; report_path=outdir/f"seed-{args.seed}-report.json"; write_json(report_path,report); print(f"REPORT {report_path}",flush=True); return 0

def reference(args: argparse.Namespace) -> int:
    repo_root=Path(args.repo_root).resolve(); outdir=Path(args.outdir).resolve(); spec=ModelSpec(**REFERENCE_SPEC_DICT)
    if spec.parameter_count()!=95_568: raise RuntimeError(f"reference parameter drift: {spec.parameter_count()}")
    result=run_segment(repo_root=repo_root,outdir=outdir,source_sha=args.source_sha,seed=1337,spec=spec,start_step=0,end_step=261,resume_checkpoint=None,expected_resume_bpb=None); row=next(r for r in result["checkpoint_metrics"] if r["optimized_tokens"]==65_772); observed=float(row["heldout_bpb"]); delta=observed-REFERENCE_EXPECTED_BPB_65772
    proof={"schema":"12-6.scale190.reference-regression.v1","expected_research41_bpb":REFERENCE_EXPECTED_BPB_65772,"observed_bpb":observed,"delta_bpb":delta,"exact_equal":observed==REFERENCE_EXPECTED_BPB_65772,"abs_delta_bpb":abs(delta),"pass_tolerance_1e-6":abs(delta)<=1e-6,"environment":{"torch":torch.__version__,"python":sys.version,"platform":platform.platform()}}; write_json(outdir/"reference-regression.json",proof); print(json.dumps(proof,sort_keys=True),flush=True); return 0

def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("orchestrate"); p.add_argument("--repo-root",required=True); p.add_argument("--outdir",required=True); p.add_argument("--source-sha",required=True); p.add_argument("--seed",type=int,required=True); p.set_defaults(func=orchestrate)
    p=sub.add_parser("worker"); p.add_argument("--repo-root",required=True); p.add_argument("--outdir",required=True); p.add_argument("--source-sha",required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--start-step",type=int,required=True); p.add_argument("--end-step",type=int,required=True); p.add_argument("--resume-checkpoint"); p.add_argument("--expected-resume-bpb",type=float); p.add_argument("--output",required=True); p.add_argument("--reference",action="store_true"); p.set_defaults(func=worker)
    p=sub.add_parser("reference"); p.add_argument("--repo-root",required=True); p.add_argument("--outdir",required=True); p.add_argument("--source-sha",required=True); p.set_defaults(func=reference); return parser

def main() -> int:
    args=build_parser().parse_args(); return int(args.func(args))
if __name__=="__main__": raise SystemExit(main())
