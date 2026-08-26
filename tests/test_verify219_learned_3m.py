from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.checkpoint import hash_json
from twelve_six.verify219_learned_3m import (
    PRODUCER_ARTIFACT_ID,
    PRODUCER_ARTIFACT_ZIP_SHA256,
    PRODUCER_SHA,
    STATE,
    WORKER,
    Verify219Error,
    _compare_selection,
    _self_hash,
    _tree_hash,
)


def _evaluation() -> dict:
    by = {
        stratum: {
            "bits_per_byte": 2.0 + index / 10,
            "loss_nats_per_byte": 1.0 + index / 10,
            "predicted_byte_tokens": 100 + index,
        }
        for index, stratum in enumerate(("uk", "en", "code"))
    }
    return {
        "split": "validation",
        "bits_per_byte": 2.1,
        "loss_nats_per_byte": 1.1,
        "predicted_byte_tokens": 303,
        "by_stratum": by,
        "model_state_sha256_before": "a" * 64,
        "model_state_sha256_after": "a" * 64,
        "non_mutation_passed": True,
    }


def test_exact_producer_binding() -> None:
    assert WORKER == "VERIFY-219-LEARNED-3M-INDEPENDENT"
    assert STATE == "VERIFIED_LEARNED_3M"
    assert PRODUCER_SHA == "a75920cef8bde37a8c590e34095be83c97b75f1d"
    assert PRODUCER_ARTIFACT_ID == 9597788382
    assert PRODUCER_ARTIFACT_ZIP_SHA256 == (
        "f57bf36113a68fffd4bfcf877bf08762393479b9c09e6fd0fd613fbb91f044ee"
    )


def test_self_hash_fails_closed() -> None:
    value = {"schema": "fixture", "x": 1}
    value["identity_sha256"] = hash_json(value)
    assert _self_hash(value, "identity_sha256", "fixture") == value["identity_sha256"]
    corrupted = dict(value)
    corrupted["x"] = 2
    with pytest.raises(Verify219Error, match="self-hash mismatch"):
        _self_hash(corrupted, "identity_sha256", "fixture")


def test_selection_comparison_rejects_metric_drift() -> None:
    expected = _evaluation()
    _compare_selection(deepcopy(expected), expected, "fixture")
    bad = deepcopy(expected)
    bad["by_stratum"]["uk"]["bits_per_byte"] += 1e-4
    with pytest.raises(Verify219Error, match="fixture.uk.bits_per_byte mismatch"):
        _compare_selection(bad, expected, "fixture")


def test_selection_comparison_rejects_model_mutation() -> None:
    expected = _evaluation()
    bad = deepcopy(expected)
    bad["model_state_sha256_after"] = "b" * 64
    with pytest.raises(Verify219Error, match="state changed"):
        _compare_selection(bad, expected, "fixture")


def test_tree_hash_changes_with_bytes_and_path(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "one.bin").write_bytes(b"abc")
    first = _tree_hash(root)
    (root / "one.bin").write_bytes(b"abd")
    second = _tree_hash(root)
    assert first != second
    (root / "one.bin").rename(root / "two.bin")
    third = _tree_hash(root)
    assert second != third
