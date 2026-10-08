"""Plan 6 Section 3: offline DPO on bounded verified pair/ranking fixtures.

This component never performs online champion selection or Base mutation.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .post_base_instruction import (
    canonical_digest,
    model_logits,
    sha_field,
    state_sha256,
    tokens,
    validate_source,
)


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    sample_id: str
    prompt_ids: tuple[int, ...]
    ranked_targets: tuple[tuple[int, ...], ...]
    provenance_sha256: str
    rights: str
    rights_evidence_sha256: str
    quality: float
    split: str = "train"
    schema_version: int = 1

    def validate(self, vocab: int, context: int) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported preference version")
        validate_source(
            sample_id=self.sample_id, provenance_sha256=self.provenance_sha256,
            rights=self.rights, rights_evidence_sha256=self.rights_evidence_sha256,
            quality=self.quality, split=self.split,
        )
        tokens(self.prompt_ids, vocab)
        if type(self.ranked_targets) is not tuple or not 2 <= len(self.ranked_targets) <= 8:
            raise ValueError("pair/ranking must contain 2 to 8 candidates")
        for target in self.ranked_targets:
            tokens(target, vocab)
            if len(self.prompt_ids) + len(target) > context:
                raise ValueError("preference exceeds context")
        if len(set(self.ranked_targets)) != len(self.ranked_targets):
            raise ValueError("inconsistent ranking: duplicated target")

    def identity(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreferenceRecipe:
    policy_sha256: str
    reference_sha256: str
    tokenizer_sha256: str
    max_seq_len: int
    learning_rate: float = 0.01
    beta: float = 0.3
    max_steps: int = 2
    batch_size: int = 2
    max_records: int = 32
    seed: int = 37
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported preference recipe")
        for item in (self.policy_sha256, self.reference_sha256, self.tokenizer_sha256):
            sha_field(item)
        for n in (
            self.max_seq_len, self.max_steps, self.batch_size,
            self.max_records, self.seed,
        ):
            if type(n) is not int:
                raise ValueError("integer limits required")
        if not 2 <= self.max_seq_len <= 4096 or not 1 <= self.max_steps <= 16:
            raise ValueError("unbounded preference recipe")
        if not 1 <= self.batch_size <= 32 or not 1 <= self.max_records <= 1024:
            raise ValueError("unbounded preference batch/dataset")
        if not 0 <= self.seed < 2**32:
            raise ValueError("invalid seed")
        if (
            type(self.learning_rate) not in (int, float)
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 0.1
        ):
            raise ValueError("unsafe preference learning rate")
        if (
            type(self.beta) not in (int, float)
            or not math.isfinite(self.beta) or not 0 < self.beta <= 10
        ):
            raise ValueError("unsafe preference beta")


@dataclass(frozen=True, slots=True)
class PreferenceReceipt:
    parent_policy_sha256: str
    reference_sha256: str
    descendant_sha256: str
    tokenizer_sha256: str
    data_sha256: str
    recipe_sha256: str
    losses: tuple[float, ...]
    mode: str = "offline-dpo-v1"


def target_log_probability(
    model: nn.Module, prompt: tuple[int, ...], target: tuple[int, ...],
) -> torch.Tensor:
    ids = torch.tensor((prompt + target,), dtype=torch.long)
    logits = model_logits(model, ids)
    start = len(prompt) - 1
    selected = F.log_softmax(logits[0, start:-1, :], dim=-1)
    indexes = torch.tensor(target, dtype=torch.long).view(-1, 1)
    return selected.gather(1, indexes).sum()


def train_preference_descendant(
    policy: nn.Module, reference: nn.Module,
    records: tuple[PreferenceRecord, ...], recipe: PreferenceRecipe, *,
    vocab: int,
) -> tuple[nn.Module, PreferenceReceipt]:
    """Offline DPO returns an isolated candidate; promotion is not authorized."""
    recipe.validate()
    if (
        not isinstance(policy, nn.Module) or not isinstance(reference, nn.Module)
        or state_sha256(policy) != recipe.policy_sha256
        or state_sha256(reference) != recipe.reference_sha256
    ):
        raise ValueError("policy/reference identity mismatch")
    if type(records) is not tuple or not records or len(records) > recipe.max_records:
        raise ValueError("unbounded preference dataset")
    if type(vocab) is not int or vocab < 2:
        raise ValueError("invalid vocab")
    seen_ids: set[str] = set()
    graph: dict[
        tuple[int, ...], set[tuple[tuple[int, ...], tuple[int, ...]]]
    ] = {}
    for record in records:
        if type(record) is not PreferenceRecord:
            raise ValueError("invalid preference record")
        record.validate(vocab, recipe.max_seq_len)
        if record.sample_id in seen_ids:
            raise ValueError("duplicate preference identity")
        seen_ids.add(record.sample_id)
        directed = graph.setdefault(record.prompt_ids, set())
        for winner, loser in zip(record.ranked_targets, record.ranked_targets[1:]):
            if (winner, loser) in directed or (loser, winner) in directed:
                raise ValueError("duplicate or contradictory preference comparison")
            directed.add((winner, loser))
    # Disallow non-local ranking cycles across independently supplied records.
    for directed in graph.values():
        edges = {}
        for winner, loser in directed:
            edges.setdefault(winner, set()).add(loser)
        for root, descendants in edges.items():
            visited = set()
            pending = list(descendants)
            while pending:
                node = pending.pop()
                if node == root:
                    raise ValueError("cyclic preference ranking")
                if node not in visited:
                    visited.add(node)
                    pending.extend(edges.get(node, ()))
    order = sorted(records, key=lambda x: x.sample_id)
    data_sha = canonical_digest([x.identity() for x in order])
    candidate = copy.deepcopy(policy).cpu().train()
    frozen = copy.deepcopy(reference).cpu().eval()
    for param in frozen.parameters():
        param.requires_grad_(False)
    torch.manual_seed(recipe.seed)
    optimizer = torch.optim.SGD(candidate.parameters(), lr=recipe.learning_rate)
    losses: list[float] = []
    for step in range(recipe.max_steps):
        sample = random.Random(recipe.seed + step).sample(
            order, min(recipe.batch_size, len(order)),
        )
        terms = []
        for record in sample:
            prompt = record.prompt_ids
            for winner, loser in zip(record.ranked_targets, record.ranked_targets[1:]):
                policy_margin = (
                    target_log_probability(candidate, prompt, winner)
                    - target_log_probability(candidate, prompt, loser)
                )
                with torch.no_grad():
                    reference_margin = (
                        target_log_probability(frozen, prompt, winner)
                        - target_log_probability(frozen, prompt, loser)
                    )
                terms.append(
                    -F.logsigmoid(recipe.beta * (policy_margin - reference_margin))
                )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(terms).mean()
        if not torch.isfinite(loss).item():
            raise ValueError("nonfinite preference loss")
        loss.backward()
        if not all(
            p.grad is None or torch.isfinite(p.grad).all().item()
            for p in candidate.parameters()
        ):
            raise ValueError("nonfinite preference gradient")
        torch.nn.utils.clip_grad_norm_(
            candidate.parameters(), 1.0, error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    if (
        state_sha256(policy) != recipe.policy_sha256
        or state_sha256(reference) != recipe.reference_sha256
    ):
        raise RuntimeError("policy/reference unexpectedly mutated")
    descendant_sha = state_sha256(candidate)
    if descendant_sha == recipe.policy_sha256:
        raise RuntimeError("preference descendant did not update")
    return candidate, PreferenceReceipt(
        recipe.policy_sha256, recipe.reference_sha256, descendant_sha,
        recipe.tokenizer_sha256, data_sha,
        canonical_digest(asdict(recipe)), tuple(losses),
    )
