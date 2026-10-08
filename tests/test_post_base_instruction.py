"""Plan 6 Section 2 synthetic CPU tests; no production model/training."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from twelve_six.post_base_instruction import (
    InstructionExample, InstructionRecipe, format_instruction_batch,
    state_sha256, train_instruction_descendant,
)

H = "a" * 64
T = "b" * 64


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 8)
        self.head = nn.Linear(8, 12)

    def forward(self, ids):
        return type("Out", (), {"logits": self.head(self.embedding(ids))})()


def ex(sample_id="a"):
    return InstructionExample(sample_id, (1, 2), (3, 4), H, "owned", T, 0.99)


def fixture():
    torch.manual_seed(42)
    parent = TinyLM()
    recipe = InstructionRecipe(state_sha256(parent), H, max_seq_len=12)
    return parent, recipe


def test_masking_batches_and_base_immutability():
    parent, recipe = fixture()
    batch = format_instruction_batch(
        (ex(), replace(ex(), sample_id="b", target_ids=(4,))),
        pad_id=0, vocab=12, max_seq_len=12,
    )
    assert batch["labels"].tolist() == [[-100, -100, 3, 4], [-100, -100, 4, -100]]
    child, receipt = train_instruction_descendant(parent, (ex(),), recipe, vocab=12)
    assert state_sha256(parent) == receipt.parent_base_sha256
    assert state_sha256(child) == receipt.descendant_sha256
    assert receipt.parent_base_sha256 != receipt.descendant_sha256
    assert len(receipt.losses) == 2


def test_deterministic_restart_and_receipt_binding():
    parent, recipe = fixture()
    _, first = train_instruction_descendant(parent, (ex(),), recipe, vocab=12)
    _, replay = train_instruction_descendant(parent, (ex(),), recipe, vocab=12)
    assert first == replay
    changed = train_instruction_descendant(
        parent, (replace(ex(), target_ids=(5,)),), recipe, vocab=12,
    )[1]
    assert first.data_sha256 != changed.data_sha256


@pytest.mark.parametrize("patch", [
    {"rights": "unknown"}, {"split": "eval"}, {"quality": 0.4},
    {"provenance_sha256": "bad"}, {"rights_evidence_sha256": "bad"},
    {"prompt_ids": (-1,)}, {"target_ids": (99,)}, {"schema_version": 2},
    {"target_ids": (3,) * 30},
])
def test_fail_closed_source(patch):
    parent, recipe = fixture()
    with pytest.raises(ValueError):
        train_instruction_descendant(parent, (replace(ex(), **patch),), recipe, vocab=12)


def test_duplicates_and_recipe_limits():
    parent, recipe = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        train_instruction_descendant(parent, (ex(), ex()), recipe, vocab=12)
    with pytest.raises(ValueError, match="Base identity"):
        train_instruction_descendant(
            parent, (ex(),), replace(recipe, parent_base_sha256=H), vocab=12,
        )
    with pytest.raises(ValueError, match="unbounded"):
        train_instruction_descendant(
            parent, (ex(),), replace(recipe, max_steps=999), vocab=12,
        )
    assert state_sha256(parent) == recipe.parent_base_sha256
