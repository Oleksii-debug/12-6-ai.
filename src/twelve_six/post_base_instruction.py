"""Plan 6 Section 2: bounded offline instruction descendant training.

This owns post-Base data admission and training composition; the canonical Base
model and Plan-3 causal loss implementation remain untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .training.loss import causal_lm_loss

_RIGHTS = frozenset({"owned", "public_domain", "licensed"})


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: object) -> str:
    return sha_bytes(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ).encode()
    )


def sha_field(value: str) -> None:
    if (
        type(value) is not str or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected lowercase sha256")


def bounded_id(value: str) -> None:
    if (
        type(value) is not str or not 0 < len(value) <= 128
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("invalid bounded identity")


def validate_source(
    *, sample_id: str, provenance_sha256: str, rights: str,
    rights_evidence_sha256: str, split: str, quality: float,
) -> None:
    bounded_id(sample_id)
    sha_field(provenance_sha256)
    sha_field(rights_evidence_sha256)
    if rights not in _RIGHTS or split != "train":
        raise ValueError("source lacks verified rights or is not training-only")
    if (
        type(quality) not in (float, int) or not math.isfinite(quality)
        or quality < 0.8 or quality > 1
    ):
        raise ValueError("insufficient quality")


def tokens(seq: tuple[int, ...], vocab: int) -> None:
    if (
        type(seq) is not tuple or not seq
        or any(type(t) is not int or t < 0 or t >= vocab for t in seq)
    ):
        raise ValueError("invalid token ids")


def state_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        x = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(b"\x00")
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_logits(model: nn.Module, ids: Tensor) -> Tensor:
    output: Any = model(ids)
    logits = output if isinstance(output, Tensor) else getattr(output, "logits", None)
    if not isinstance(logits, Tensor) or logits.ndim != 3 or logits.shape[:2] != ids.shape:
        raise ValueError("model must return [batch,time,vocab] logits")
    if not torch.isfinite(logits).all().item():
        raise ValueError("nonfinite model logits")
    return logits


@dataclass(frozen=True, slots=True)
class InstructionExample:
    sample_id: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    provenance_sha256: str
    rights: str
    rights_evidence_sha256: str
    quality: float
    split: str = "train"
    schema_version: int = 1

    def validate(self, vocab: int, max_seq_len: int) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported example schema")
        validate_source(
            sample_id=self.sample_id, provenance_sha256=self.provenance_sha256,
            rights=self.rights, rights_evidence_sha256=self.rights_evidence_sha256,
            split=self.split, quality=self.quality,
        )
        tokens(self.prompt_ids, vocab)
        tokens(self.target_ids, vocab)
        if len(self.prompt_ids) + len(self.target_ids) > max_seq_len:
            raise ValueError("instruction example exceeds context")

    def identity(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id, "prompt_ids": self.prompt_ids,
            "target_ids": self.target_ids, "provenance_sha256": self.provenance_sha256,
            "rights": self.rights, "rights_evidence_sha256": self.rights_evidence_sha256,
            "quality": self.quality, "split": self.split, "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class InstructionRecipe:
    parent_base_sha256: str
    tokenizer_sha256: str
    max_seq_len: int
    learning_rate: float = 0.01
    max_steps: int = 2
    batch_size: int = 2
    max_examples: int = 32
    seed: int = 17
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported instruction recipe")
        sha_field(self.parent_base_sha256)
        sha_field(self.tokenizer_sha256)
        values = (
            self.max_seq_len, self.max_steps, self.batch_size,
            self.max_examples, self.seed,
        )
        if any(type(v) is not int for v in values):
            raise ValueError("integer limits required")
        if not 2 <= self.max_seq_len <= 4096 or not 1 <= self.max_steps <= 16:
            raise ValueError("unbounded training run")
        if (
            not 1 <= self.batch_size <= 32 or not 1 <= self.max_examples <= 1024
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("invalid recipe bounds")
        if (
            type(self.learning_rate) not in (float, int)
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 0.1
        ):
            raise ValueError("unsafe learning rate")


@dataclass(frozen=True, slots=True)
class InstructionReceipt:
    parent_base_sha256: str
    descendant_sha256: str
    tokenizer_sha256: str
    data_sha256: str
    recipe_sha256: str
    losses: tuple[float, ...]
    engine: str = "instruction-sft-v1"


def format_instruction_batch(
    examples: tuple[InstructionExample, ...], *, pad_id: int,
    vocab: int, max_seq_len: int,
) -> dict[str, Tensor]:
    if not examples or not 0 <= pad_id < vocab:
        raise ValueError("empty batch or invalid padding token")
    for ex in examples:
        ex.validate(vocab, max_seq_len)
    width = max(len(ex.prompt_ids) + len(ex.target_ids) for ex in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for row, ex in enumerate(examples):
        seq = ex.prompt_ids + ex.target_ids
        ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        labels[row, len(ex.prompt_ids):len(seq)] = torch.tensor(ex.target_ids)
    return {"input_ids": ids, "labels": labels}


def train_instruction_descendant(
    parent: nn.Module, examples: tuple[InstructionExample, ...],
    recipe: InstructionRecipe, *, vocab: int, pad_id: int = 0,
) -> tuple[nn.Module, InstructionReceipt]:
    """Return isolated SFT descendant and evidence; never modify the clean Base."""
    recipe.validate()
    if type(examples) is not tuple or not examples or len(examples) > recipe.max_examples:
        raise ValueError("unbounded instruction dataset")
    if type(vocab) is not int or vocab < 2:
        raise ValueError("invalid vocab")
    if not isinstance(parent, nn.Module) or state_sha256(parent) != recipe.parent_base_sha256:
        raise ValueError("Base identity mismatch")
    seen: set[str] = set()
    for ex in examples:
        if type(ex) is not InstructionExample:
            raise ValueError("invalid instruction example")
        ex.validate(vocab, recipe.max_seq_len)
        if ex.sample_id in seen:
            raise ValueError("duplicate sample identity")
        seen.add(ex.sample_id)
    data_sha = canonical_digest([x.identity() for x in sorted(examples, key=lambda x: x.sample_id)])
    candidate = copy.deepcopy(parent).cpu()
    candidate.train()
    torch.manual_seed(recipe.seed)
    opt = torch.optim.SGD(candidate.parameters(), lr=recipe.learning_rate)
    order = sorted(examples, key=lambda x: x.sample_id)
    losses: list[float] = []
    for step in range(recipe.max_steps):
        rng = random.Random(recipe.seed + step)
        sample = rng.sample(order, min(recipe.batch_size, len(order)))
        batch = format_instruction_batch(
            tuple(sample), pad_id=pad_id, vocab=vocab,
            max_seq_len=recipe.max_seq_len,
        )
        opt.zero_grad(set_to_none=True)
        logits = model_logits(candidate, batch["input_ids"])
        loss = causal_lm_loss(logits, batch["labels"])
        if not torch.isfinite(loss).item():
            raise ValueError("nonfinite instruction loss")
        loss.backward()
        if not all(
            p.grad is None or torch.isfinite(p.grad).all().item()
            for p in candidate.parameters()
        ):
            raise ValueError("nonfinite instruction gradients")
        torch.nn.utils.clip_grad_norm_(
            candidate.parameters(), max_norm=1.0, error_if_nonfinite=True,
        )
        opt.step()
        losses.append(float(loss.detach()))
    if state_sha256(parent) != recipe.parent_base_sha256:
        raise RuntimeError("clean Base mutated")
    desc_sha = state_sha256(candidate)
    if desc_sha == recipe.parent_base_sha256:
        raise RuntimeError("descendant has no update")
    receipt = InstructionReceipt(
        recipe.parent_base_sha256, desc_sha, recipe.tokenizer_sha256,
        data_sha, canonical_digest(asdict(recipe)), tuple(losses),
    )
    return candidate, receipt
