from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.checkpoint import hash_json
from twelve_six.verify218_learned_10m import (
    PRODUCER_ARTIFACT_ID,
    PRODUCER_ARTIFACT_ZIP_SHA256,
    PRODUCER_SHA,
    STATE,
    WORKER,
    Verify218Error,
    _compare_common_eval,
    _independent_common_eval,
    _independent_eval_batch,
    _tree_sha256,
    _verify_self_hash,
)


def _eval() -> dict:
    by = {
        key: {
            "loss": 1.0 + index / 10,
            "bits_per_byte": 1.5 + index / 10,
            "predicted_byte_tokens": 100 + index,
        }
        for index, key in enumerate(("uk", "en", "code"))
    }
    return {
        "loss": 1.1,
        "bits_per_byte": 1.6,
        "predicted_byte_tokens": 303,
        "by_stratum": by,
        "model_state_sha256_before": "a" * 64,
        "model_state_sha256_after": "a" * 64,
        "non_mutation_passed": True,
    }


def test_authority_constants_bind_exact_learn217_artifact() -> None:
    assert WORKER == "VERIFY-218-LEARNED-10M-INDEPENDENT"
    assert STATE == "VERIFIED_LEARNED_10M"
    assert PRODUCER_SHA == "c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef"
    assert PRODUCER_ARTIFACT_ID == 9602650341
    assert PRODUCER_ARTIFACT_ZIP_SHA256 == (
        "8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee"
    )


def test_self_hash_is_fail_closed() -> None:
    value = {"schema": "x", "payload": {"a": 1}}
    value["identity_sha256"] = hash_json(value)
    assert _verify_self_hash(value, "identity_sha256", "fixture") == value[
        "identity_sha256"
    ]

    corrupted = deepcopy(value)
    corrupted["payload"]["a"] = 2
    with pytest.raises(Verify218Error, match="self-hash mismatch"):
        _verify_self_hash(corrupted, "identity_sha256", "fixture")


def test_common_eval_accepts_exact_and_rejects_metric_drift() -> None:
    expected = _eval()
    _compare_common_eval(deepcopy(expected), expected, label="fixture")

    drifted = deepcopy(expected)
    drifted["by_stratum"]["en"]["bits_per_byte"] += 1e-4
    with pytest.raises(Verify218Error, match="fixture.en.bits_per_byte mismatch"):
        _compare_common_eval(drifted, expected, label="fixture")


def test_common_eval_rejects_state_mutation() -> None:
    expected = _eval()
    actual = deepcopy(expected)
    actual["model_state_sha256_after"] = "b" * 64
    with pytest.raises(Verify218Error, match="state hash changed"):
        _compare_common_eval(actual, expected, label="fixture")


def test_independent_common_eval_does_not_delegate_producer_evaluator() -> None:
    common_source = inspect.getsource(_independent_common_eval)
    batch_source = inspect.getsource(_independent_eval_batch)
    assert "m100._evaluate" not in common_source
    assert "_independent_eval_batch" in common_source
    assert "F.cross_entropy" in batch_source
    assert 'reduction="sum"' in batch_source


def test_checkpoint_tree_digest_is_content_and_path_sensitive(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "a.bin").write_bytes(b"abc")
    first = _tree_sha256(root)
    assert len(first) == 64

    (root / "a.bin").write_bytes(b"abd")
    second = _tree_sha256(root)
    assert second != first

    (root / "a.bin").rename(root / "b.bin")
    third = _tree_sha256(root)
    assert third != second
