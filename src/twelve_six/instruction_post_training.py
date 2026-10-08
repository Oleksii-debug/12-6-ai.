"""Plan 6 instruction descendant adapter; reuse D02 Trainer, never mutate Base."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer

_SHA = re.compile(r"^[0-9a-f]{64}$")
_IGNORE = -100


def _sha(value: object) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError("identity must be lowercase SHA-256")
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tokens(value: object, vocab_size: int) -> tuple[int, ...]:
    if type(value) is not tuple or not value or len(value) > 512:
        raise ValueError("instruction tokens must be nonempty bounded tuple")
    if any(type(v) is not int or not 0 <= v < vocab_size for v in value):
        raise ValueError("instruction token out of vocabulary")
    return value


@dataclass(frozen=True, slots=True)
class InstructionExample:
    prompt: tuple[int, ...]
    target: tuple[int, ...]
    source_sha256: str
    rights_evidence_sha256: str
    quality_evidence_sha256: str
    quality: float
    split: str = "train"

    def validate(self, vocab_size: int) -> None:
        _tokens(self.prompt, vocab_size)
        _tokens(self.target, vocab_size)
        _sha(self.source_sha256)
        _sha(self.rights_evidence_sha256)
        _sha(self.quality_evidence_sha256)
        if self.split != "train" or type(self.split) is not str:
            raise ValueError("evaluation/test examples cannot enter instruction training")
        if type(self.quality) not in (float, int) or not math.isfinite(self.quality):
            raise ValueError("invalid instruction quality")
        if not 0.8 <= self.quality <= 1.0:
            raise ValueError("instruction quality gate failed")

    @property
    def content_sha256(self) -> str:
        if type(self.prompt) is not tuple or type(self.target) is not tuple:
            raise ValueError("invalid instruction content")
        return _digest({"prompt": self.prompt, "target": self.target})


@dataclass(frozen=True, slots=True)
class InstructionRecipe:
    base_checkpoint_sha256: str
    tokenizer_sha256: str
    seed: int
    vocab_size: int
    bos: int
    sep: int
    eos: int
    pad: int
    max_seq_len: int = 64
    batch_size: int = 2
    max_steps: int = 2
    learning_rate: float = 0.01
    schema_version: int = 1

    def validate(self) -> None:
        _sha(self.base_checkpoint_sha256)
        _sha(self.tokenizer_sha256)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported instruction recipe version")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("bad instruction seed")
        if type(self.vocab_size) is not int or not 8 <= self.vocab_size <= 4096:
            raise ValueError("bounded local fixture vocabulary required")
        specials = (self.bos, self.sep, self.eos, self.pad)
        if any(type(v) is not int or v not in range(self.vocab_size) for v in specials):
            raise ValueError("invalid special token")
        if len(set(specials)) != 4:
            raise ValueError("special tokens must be distinct")
        for value, limit, name in ((self.max_seq_len, 512, "length"),
                                   (self.batch_size, 8, "batch"),
                                   (self.max_steps, 8, "steps")):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"invalid bounded {name}")
        if self.max_seq_len < 5:
            raise ValueError("instruction context too short")
        if type(self.learning_rate) is not float or not math.isfinite(self.learning_rate):
            raise ValueError("invalid instruction learning rate")
        if not 0 < self.learning_rate <= 0.1:
            raise ValueError("instruction fixture learning rate is out of bounds")

    @property
    def identity_sha256(self) -> str:
        self.validate()
        return _digest({field: getattr(self, field) for field in self.__dataclass_fields__})


def model_sha256(model: nn.Module) -> str:
    """Content identity for a bounded CPU model snapshot; never use as production attestation."""
    if not isinstance(model, nn.Module):
        raise TypeError("expected torch model")
    state = model.state_dict()
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, Tensor) or tensor.device.type != "cpu":
            raise ValueError("only CPU fixture model permitted")
        block = tensor.detach().contiguous()
        h.update(name.encode())
        h.update(str(block.dtype).encode())
        h.update(str(tuple(block.shape)).encode())
        h.update(block.numpy().tobytes())
    return h.hexdigest()


def admit_instruction_examples(
    recipe: InstructionRecipe,
    examples: tuple[InstructionExample, ...],
    *,
    approved_rights: frozenset[str],
    approved_quality: frozenset[str],
    reserved_eval_content: frozenset[str],
) -> str:
    """Caller-controlled admission allowlists must come from an independent authority."""
    recipe.validate()
    if type(examples) is not tuple or not 1 <= len(examples) <= 64:
        raise ValueError("invalid bounded instruction dataset")
    for values in (approved_rights, approved_quality, reserved_eval_content):
        if type(values) is not frozenset or any(_sha(v) != v for v in values):
            raise ValueError("untrusted instruction admission manifest")
    if not approved_rights or not approved_quality or not reserved_eval_content:
        raise ValueError("independent rights, quality and eval isolation required")
    seen: set[str] = set()
    for item in examples:
        if type(item) is not InstructionExample:
            raise ValueError("invalid instruction example")
        item.validate(recipe.vocab_size)
        if len(item.prompt) + len(item.target) + 3 > recipe.max_seq_len:
            raise ValueError("instruction sequence exceeds context")
        if any(token in (recipe.bos, recipe.sep, recipe.eos, recipe.pad)
               for token in item.prompt + item.target):
            raise ValueError("special-token injection in instruction example")
        if item.rights_evidence_sha256 not in approved_rights:
            raise ValueError("instruction rights unapproved")
        if item.quality_evidence_sha256 not in approved_quality:
            raise ValueError("instruction quality not independently admitted")
        if item.content_sha256 in reserved_eval_content:
            raise ValueError("instruction evaluation leakage")
        if item.content_sha256 in seen:
            raise ValueError("duplicate instruction content")
        seen.add(item.content_sha256)
    return _digest({"recipe": recipe.identity_sha256,
                    "examples": sorted(seen),
                    "rights": sorted(approved_rights),
                    "quality": sorted(approved_quality),
                    "eval_blocklist": sorted(reserved_eval_content)})


def format_instruction(example: InstructionExample, recipe: InstructionRecipe) -> tuple[list[int], list[int]]:
    recipe.validate()
    example.validate(recipe.vocab_size)
    sequence = [recipe.bos, *example.prompt, recipe.sep, *example.target, recipe.eos]
    if len(sequence) > recipe.max_seq_len:
        raise ValueError("instruction sequence exceeds context")
    # Logits at prefix-last predict first target. Never predict prompt/prefix tokens.
    prefix_last = len(example.prompt) + 1
    targets = [_IGNORE] * len(sequence)
    for pos in range(prefix_last, len(sequence) - 1):
        targets[pos] = sequence[pos + 1]
    return sequence, targets


def instruction_batches(examples: tuple[InstructionExample, ...], recipe: InstructionRecipe):
    """Yield reproducible already-aligned D02 Trainer batches; admission is separate."""
    recipe.validate()
    order = list(range(len(examples)))
    random.Random(recipe.seed).shuffle(order)
    for start in range(0, len(order), recipe.batch_size):
        formatted = [format_instruction(examples[i], recipe)
                     for i in order[start:start + recipe.batch_size]]
        width = max(len(ids) for ids, _ in formatted)
        input_ids = torch.full((len(formatted), width), recipe.pad, dtype=torch.long)
        target_ids = torch.full((len(formatted), width), _IGNORE, dtype=torch.long)
        mask = torch.zeros((len(formatted), width), dtype=torch.bool)
        for row, (ids, labels) in enumerate(formatted):
            input_ids[row, :len(ids)] = torch.tensor(ids)
            target_ids[row, :len(labels)] = torch.tensor(labels)
            mask[row, :len(labels)] = target_ids[row, :len(labels)].ne(_IGNORE)
        yield {"input_ids": input_ids, "target_ids": target_ids, "loss_mask": mask}


@dataclass(frozen=True, slots=True)
class DescendantReceipt:
    parent_sha256: str
    recipe_sha256: str
    dataset_sha256: str
    trained_model_sha256: str
    optimizer_steps: int
    optimized_tokens: int
    descendant_sha256: str


def train_fixture_descendant(
    base_model: nn.Module,
    recipe: InstructionRecipe,
    examples: tuple[InstructionExample, ...],
    *,
    approved_rights: frozenset[str],
    approved_quality: frozenset[str],
    reserved_eval_content: frozenset[str],
) -> tuple[nn.Module, DescendantReceipt]:
    """No paid compute, no real campaign, no promotion: clone Base and reuse D02."""
    dataset_sha = admit_instruction_examples(
        recipe, examples, approved_rights=approved_rights,
        approved_quality=approved_quality, reserved_eval_content=reserved_eval_content,
    )
    before = model_sha256(base_model)
    if before != recipe.base_checkpoint_sha256:
        raise ValueError("base checkpoint identity mismatch")
    descendant = copy.deepcopy(base_model)
    trainer = Trainer(descendant, TrainerConfig(
        max_steps=recipe.max_steps, learning_rate=recipe.learning_rate, seed=recipe.seed,
        precision="fp32", scheduler="constant", gradient_accumulation_steps=1,
    ), device="cpu")
    def batches():
        for _ in range(recipe.max_steps):
            yield from instruction_batches(examples, recipe)
    result = trainer.run(batches())
    if model_sha256(base_model) != before:
        raise RuntimeError("canonical Base mutated during instruction training")
    after = model_sha256(descendant)
    receipt_payload = {"parent": before, "recipe": recipe.identity_sha256,
                       "dataset": dataset_sha, "result": after,
                       "steps": result.optimizer_steps_completed,
                       "tokens": result.tokens_consumed}
    return descendant, DescendantReceipt(
        before, recipe.identity_sha256, dataset_sha, after,
        result.optimizer_steps_completed, result.tokens_consumed, _digest(receipt_payload),
    )
