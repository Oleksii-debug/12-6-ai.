from __future__ import annotations

from twelve_six.checkpoint import hash_json
from twelve_six.milestone150_resume_bridge import json_semantic


def test_json_semantic_preserves_identity_while_normalizing_tuple() -> None:
    manifest = {
        "schema": "example",
        "trainer_config": {"betas": (0.9, 0.95), "learning_rate": 3e-4},
    }
    before = hash_json(manifest)
    normalized = json_semantic(manifest)
    assert normalized["trainer_config"]["betas"] == [0.9, 0.95]
    assert hash_json(normalized) == before


def test_json_semantic_is_idempotent() -> None:
    manifest = {"a": [1, 2], "b": {"c": True}}
    assert json_semantic(json_semantic(manifest)) == json_semantic(manifest)
