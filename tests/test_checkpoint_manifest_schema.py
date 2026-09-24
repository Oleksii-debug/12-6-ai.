from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointError,
    CheckpointIdentity,
    hash_json,
    load_checkpoint,
    save_checkpoint,
)


class Model:
    def __init__(self, value: list[float]) -> None:
        self.weights = np.asarray(value, dtype=np.float64).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(
        self,
        state: dict[str, np.ndarray],
        strict: bool = True,
    ) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "manifest-schema-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision="float64",
        step=1,
        tokens_seen=3,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest(
    checkpoint: Path,
    mutation: Callable[[dict[str, object]], None],
    *,
    rebind_checkpoint_id: bool,
) -> None:
    path = checkpoint / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    if rebind_checkpoint_id:
        manifest["checkpoint_id"] = hash_json(
            {
                "identity": manifest["identity"],
                "files": manifest["files"],
            }
        )
    encoded = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n",
        encoding="ascii",
    )


def _unknown_top_level(manifest: dict[str, object]) -> None:
    manifest["future_manifest_semantics"] = {"version": 2}


def _unknown_identity(manifest: dict[str, object]) -> None:
    identity = manifest["identity"]
    assert isinstance(identity, dict)
    identity["future_identity_semantics"] = "changed"


def _unknown_file_record(manifest: dict[str, object]) -> None:
    files = manifest["files"]
    assert isinstance(files, dict)
    record = files["weights.safetensors"]
    assert isinstance(record, dict)
    record["future_storage_semantics"] = "changed"


def _unknown_serialization(manifest: dict[str, object]) -> None:
    serialization = manifest["serialization"]
    assert isinstance(serialization, dict)
    serialization["future_codec"] = "new-codec"


def _changed_pickle_contract(manifest: dict[str, object]) -> None:
    serialization = manifest["serialization"]
    assert isinstance(serialization, dict)
    serialization["pickle"] = True


def _boolean_format_version(manifest: dict[str, object]) -> None:
    manifest["format_version"] = True


def _non_utc_created_at(manifest: dict[str, object]) -> None:
    manifest["created_at_utc"] = "2026-09-07T20:00:00+02:00"


def _malformed_created_at(manifest: dict[str, object]) -> None:
    manifest["created_at_utc"] = "not-a-timestampZ"


@pytest.mark.parametrize(
    ("mutation", "rebind_checkpoint_id", "message"),
    [
        (_unknown_top_level, False, "manifest fields differ"),
        (_unknown_identity, True, "identity fields differ"),
        (_unknown_file_record, True, "files.weights.safetensors fields differ"),
        (_unknown_serialization, False, "serialization fields differ"),
        (_changed_pickle_contract, False, "serialization contract mismatch"),
        (_boolean_format_version, False, "format_version must be an integer"),
        (_non_utc_created_at, False, "ISO-8601 UTC Z timestamp"),
        (_malformed_created_at, False, "ISO-8601 UTC Z timestamp"),
    ],
)
def test_self_consistent_v1_schema_drift_fails_before_live_model_mutation(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    rebind_checkpoint_id: bool,
    message: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=Model([1.0, 2.0, 3.0]),
        identity=_identity(),
    )
    _rewrite_manifest(
        checkpoint,
        mutation,
        rebind_checkpoint_id=rebind_checkpoint_id,
    )
    target = Model([9.0, 9.0, 9.0])
    before = target.weights.copy()

    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0
