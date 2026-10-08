from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch
from torch import nn

from twelve_six.instruction_post_training import (
    InstructionExample,
    InstructionRecipe,
    admit_instruction_examples,
    format_instruction,
    instruction_batches,
    model_sha256,
    train_fixture_descendant,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.output = nn.Linear(8, 32)

    def forward(self, input_ids):
        return self.output(self.embedding(input_ids))


def fixture():
    torch.manual_seed(91)
    model = Tiny()
    recipe = InstructionRecipe(model_sha256(model), sha("tiny-tokenizer"),
                               17, 32, 0, 1, 2, 3, max_seq_len=16)
    examples = (
        InstructionExample((4, 5), (6, 7), sha("source1"), sha("r1"), sha("q1"), 0.9),
        InstructionExample((8,), (9, 10), sha("source2"), sha("r2"), sha("q2"), 1.0),
    )
    allowed = {"approved_rights": frozenset((sha("r1"), sha("r2"))),
               "approved_quality": frozenset((sha("q1"), sha("q2"))),
               "reserved_eval_content": frozenset((sha("unseen-eval-item"),))}
    return model, recipe, examples, allowed


def test_prompt_mask_and_eos_labels_are_aligned():
    _, recipe, examples, _ = fixture()
    ids, labels = format_instruction(examples[0], recipe)
    assert ids == [0, 4, 5, 1, 6, 7, 2]
    assert labels == [-100, -100, -100, 6, 7, 2, -100]
    batch = next(instruction_batches(examples, recipe))
    assert batch["input_ids"].shape == batch["target_ids"].shape
    assert int(batch["loss_mask"].sum()) == 6
    assert torch.equal(batch["target_ids"].ne(-100), batch["loss_mask"])


def test_tiny_torch_descendant_preserves_base_and_replays():
    model, recipe, examples, policy = fixture()
    parent = model_sha256(model)
    trained, receipt = train_fixture_descendant(model, recipe, examples, **policy)
    assert parent == model_sha256(model)
    assert model_sha256(trained) != parent
    assert receipt.parent_sha256 == parent
    assert receipt.optimizer_steps == 2
    assert receipt.optimized_tokens == 12
    model2, recipe2, examples2, policy2 = fixture()
    trained2, receipt2 = train_fixture_descendant(model2, recipe2, examples2, **policy2)
    assert receipt == receipt2
    assert model_sha256(trained2) == model_sha256(trained)


@pytest.mark.parametrize("change", [
    {"split": "eval"}, {"split": "test"}, {"quality": 0.79},
    {"quality": float("nan")}, {"quality": True},
    {"source_sha256": "A" * 64}, {"prompt": (True,)},
    {"target": (999,)}, {"prompt": (0, 4)},
])
def test_bad_data_and_special_token_injection_fail_closed(change):
    _, recipe, examples, policy = fixture()
    bad = (replace(examples[0], **change), examples[1])
    with pytest.raises(ValueError):
        admit_instruction_examples(recipe, bad, **policy)


def test_untrusted_rights_quality_eval_leakage_and_duplicate_rejected():
    _, recipe, examples, policy = fixture()
    with pytest.raises(ValueError, match="rights"):
        admit_instruction_examples(recipe, examples, **(policy | {"approved_rights": frozenset((sha("r1"),))}))
    with pytest.raises(ValueError, match="quality"):
        admit_instruction_examples(recipe, examples, **(policy | {"approved_quality": frozenset((sha("q1"),))}))
    with pytest.raises(ValueError, match="leakage"):
        admit_instruction_examples(recipe, examples, **(policy | {"reserved_eval_content": frozenset((examples[0].content_sha256,))}))
    with pytest.raises(ValueError, match="duplicate"):
        admit_instruction_examples(recipe, (examples[0], examples[0]), **policy)
    with pytest.raises(ValueError, match="independent"):
        admit_instruction_examples(recipe, examples, **(policy | {"approved_rights": frozenset()}))


@pytest.mark.parametrize("change", [
    {"max_steps": 9}, {"batch_size": 0}, {"max_seq_len": 4},
    {"vocab_size": True}, {"seed": -1}, {"seed": True},
    {"learning_rate": float("inf")}, {"learning_rate": 1.0},
    {"bos": 1}, {"schema_version": 2},
])
def test_invalid_recipe_fails_closed(change):
    _, recipe, _, _ = fixture()
    with pytest.raises(ValueError):
        replace(recipe, **change).validate()


def test_wrong_parent_and_context_exhaustion_fail_before_training():
    model, recipe, examples, policy = fixture()
    with pytest.raises(ValueError, match="base checkpoint"):
        train_fixture_descendant(model, replace(recipe, base_checkpoint_sha256=sha("fake")),
                                 examples, **policy)
    assert model_sha256(model) == recipe.base_checkpoint_sha256
    with pytest.raises(ValueError, match="context"):
        admit_instruction_examples(replace(recipe, max_seq_len=5), examples, **policy)


def test_deterministic_batches_and_recipe_manifest_identity():
    _, recipe, examples, policy = fixture()
    sha_a = admit_instruction_examples(recipe, examples, **policy)
    sha_b = admit_instruction_examples(recipe, examples, **policy)
    assert sha_a == sha_b
    lhs = list(instruction_batches(examples, recipe))
    rhs = list(instruction_batches(examples, recipe))
    for a, b in zip(lhs, rhs, strict=True):
        for name in a:
            assert torch.equal(a[name], b[name])
    assert replace(recipe, seed=18).identity_sha256 != recipe.identity_sha256
    assert replace(recipe, base_checkpoint_sha256=sha("another")).identity_sha256 != recipe.identity_sha256
