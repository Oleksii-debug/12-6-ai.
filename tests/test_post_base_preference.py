"""Synthetic Plan 6 Section 3 DPO qualification; no paid compute."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from twelve_six.post_base_instruction import state_sha256
from twelve_six.post_base_preference import (
    PreferenceRecipe, PreferenceRecord, train_preference_descendant,
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


def fixture():
    torch.manual_seed(42)
    policy = TinyLM()
    reference = copy.deepcopy(policy)
    recipe = PreferenceRecipe(
        state_sha256(policy), state_sha256(reference), H, max_seq_len=12,
    )
    return policy, reference, recipe


def example(name="a"):
    return PreferenceRecord(
        name, (1, 2), ((3, 4), (4, 3)), H, "licensed", T, 1.0,
    )


def test_offline_dpo_immutable_parents_and_deterministic_restart(tmp_path):
    policy, reference, recipe = fixture()
    descendant, first = train_preference_descendant(
        policy, reference, (example(),), recipe, vocab=12,
    )
    assert first.mode == "offline-dpo-v1"
    assert state_sha256(descendant) == first.descendant_sha256
    assert state_sha256(policy) == first.parent_policy_sha256
    assert state_sha256(reference) == first.reference_sha256
    assert first.descendant_sha256 != first.parent_policy_sha256
    path = tmp_path / "policy.pt"
    torch.save(policy.state_dict(), path)
    loaded = TinyLM()
    loaded.load_state_dict(torch.load(path, weights_only=True))
    _, second = train_preference_descendant(
        loaded, reference, (example(),), recipe, vocab=12,
    )
    assert first == second


def test_rankings_and_dataset_binding():
    policy, reference, recipe = fixture()
    ranked = replace(example(), ranked_targets=((3, 4), (4, 3), (4, 5)))
    _, receipt = train_preference_descendant(
        policy, reference, (ranked,), recipe, vocab=12,
    )
    assert len(receipt.losses) == 2
    _, altered = train_preference_descendant(
        policy, reference, (example(),), recipe, vocab=12,
    )
    assert receipt.data_sha256 != altered.data_sha256


@pytest.mark.parametrize("patch", [
    {"rights": "unknown"}, {"quality": 0.1}, {"split": "eval"},
    {"provenance_sha256": "wrong"}, {"rights_evidence_sha256": "bad"},
    {"ranked_targets": ((3,), (3,))}, {"ranked_targets": ((3,),)},
    {"ranked_targets": ((3,), (55,))}, {"prompt_ids": (55,)},
    {"schema_version": 99},
])
def test_reject_invalid_record(patch):
    policy, reference, recipe = fixture()
    with pytest.raises(ValueError):
        train_preference_descendant(
            policy, reference, (replace(example(), **patch),), recipe, vocab=12,
        )


def test_reject_contradictory_reversed_and_duplicate_ranks():
    policy, reference, recipe = fixture()
    mirror = replace(
        example("b"), ranked_targets=((4, 3), (3, 4)),
    )
    with pytest.raises(ValueError, match="contradictory"):
        train_preference_descendant(
            policy, reference, (example(), mirror), recipe, vocab=12,
        )
    with pytest.raises(ValueError, match="duplicate"):
        train_preference_descendant(
            policy, reference, (example(), example()), recipe, vocab=12,
        )
    assert state_sha256(policy) == recipe.policy_sha256


def test_reject_mismatched_reference_and_unbounded_recipe():
    policy, reference, recipe = fixture()
    with pytest.raises(ValueError, match="identity"):
        train_preference_descendant(
            policy, reference, (example(),),
            replace(recipe, reference_sha256=T), vocab=12,
        )
    with pytest.raises(ValueError, match="unbounded"):
        train_preference_descendant(
            policy, reference, (example(),),
            replace(recipe, max_steps=17), vocab=12,
        )
    with pytest.raises(ValueError, match="beta"):
        train_preference_descendant(
            policy, reference, (example(),),
            replace(recipe, beta=float("nan")), vocab=12,
        )


def test_reject_long_preference_cycles():
    policy, reference, recipe = fixture()
    a, b, c = (3, 4), (4, 3), (5, 3)
    records = (
        replace(example("a"), ranked_targets=(a, b)),
        replace(example("b"), ranked_targets=(b, c)),
        replace(example("c"), ranked_targets=(c, a)),
    )
    with pytest.raises(ValueError, match="cyclic"):
        train_preference_descendant(
            policy, reference, records, recipe, vocab=12,
        )
