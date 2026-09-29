from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint.core import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    VerifiedCheckpoint,
    hash_json,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)


class NumpyModel:
    def __init__(self, values: list[float]) -> None:
        self.weights = np.asarray(values, dtype=np.float64).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "strict-json-regression", "parameters": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"lr": 0.25},
        seed=7,
        precision="float64",
        step=1,
        tokens_seen=3,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _save(checkpoint: Path, *, trainer_state: dict[str, object] | None = None) -> None:
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        identity=_identity(),
        trainer_state=trainer_state,
    )


def _write_manifest_bytes(checkpoint: Path, data: bytes) -> None:
    (checkpoint / "manifest.json").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n",
        encoding="ascii",
    )


def _rebind_manifest_for_payload(checkpoint: Path, payload_name: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (checkpoint / payload_name).read_bytes()
    manifest["files"][payload_name] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    raw = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    _write_manifest_bytes(checkpoint, raw)


def _write_nonfinite_manifest(checkpoint: Path, *, value: float, token: str) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_config = manifest["identity"]["training_config"]
    training_config["lr"] = value
    manifest["identity"]["training_config_hash"] = hash_json(training_config)
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    raw = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )
    emitted = "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    assert emitted in raw
    raw = raw.replace(emitted, token, 1)
    _write_manifest_bytes(checkpoint, raw.encode("utf-8"))


@pytest.mark.parametrize(
    "needle",
    [
        '"format":"12-6-checkpoint"',
        '"lr":0.25',
    ],
)
def test_checksum_consistent_manifest_duplicate_keys_fail_closed(
    tmp_path: Path,
    needle: str,
) -> None:
    checkpoint = tmp_path / "manifest-duplicate"
    _save(checkpoint)
    path = checkpoint / "manifest.json"
    raw = path.read_text(encoding="utf-8")
    assert needle in raw
    duplicated = raw.replace(needle, f"{needle},{needle}", 1)
    _write_manifest_bytes(checkpoint, duplicated.encode("utf-8"))

    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("value", "token"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
        (float("inf"), "1e400"),
    ],
)
def test_checksum_and_identity_consistent_manifest_nonfinite_fails_closed(
    tmp_path: Path,
    value: float,
    token: str,
) -> None:
    checkpoint = tmp_path / f"manifest-nonfinite-{token.replace('-', 'neg')}"
    _save(checkpoint)
    _write_nonfinite_manifest(checkpoint, value=value, token=token)

    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        verify_checkpoint(checkpoint)


def test_state_tree_duplicate_object_key_fails_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state-duplicate"
    _save(checkpoint, trainer_state={"loss": 0.25})

    path = checkpoint / "state.json"
    raw = path.read_text(encoding="utf-8")
    needle = '"__kind__":"mapping"'
    assert needle in raw
    path.write_text(raw.replace(needle, f"{needle},{needle}", 1), encoding="utf-8")
    _rebind_manifest_for_payload(checkpoint, "state.json")

    target = NumpyModel([9.0, 9.0, 9.0])
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_state_tree_nonfinite_fails_before_model_mutation(
    tmp_path: Path,
    token: str,
) -> None:
    checkpoint = tmp_path / f"state-nonfinite-{token.replace('-', 'neg')}"
    _save(checkpoint, trainer_state={"loss": 0.25})

    path = checkpoint / "state.json"
    raw = path.read_text(encoding="utf-8")
    assert "0.25" in raw
    path.write_text(raw.replace("0.25", token, 1), encoding="utf-8")
    _rebind_manifest_for_payload(checkpoint, "state.json")

    target = NumpyModel([9.0, 9.0, 9.0])
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


def test_verified_checkpoint_manifest_reparse_rejects_duplicate_members() -> None:
    verified = VerifiedCheckpoint(
        _manifest_bytes=b'{"format":"12-6-checkpoint","format":"12-6-checkpoint"}',
        _artifacts={},
    )

    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        _ = verified.manifest


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_save_rejects_nonfinite_state_before_publication(
    tmp_path: Path,
    value: float,
) -> None:
    checkpoint = tmp_path / "save-nonfinite"

    with pytest.raises(CheckpointIntegrityError, match="strict finite JSON"):
        _save(checkpoint, trainer_state={"loss": value})

    assert not checkpoint.exists()


def test_valid_checkpoint_still_verifies_and_loads(tmp_path: Path) -> None:
    checkpoint = tmp_path / "valid"
    _save(checkpoint, trainer_state={"loss": 0.25})

    manifest = verify_checkpoint(checkpoint)
    target = NumpyModel([9.0, 9.0, 9.0])
    result = load_checkpoint(checkpoint, model=target, restore_rng=False)

    assert result.manifest == manifest
    assert result.trainer_state == {"loss": 0.25}
    np.testing.assert_array_equal(target.weights, [1.0, 2.0, 3.0])
    assert target.loads == 1
