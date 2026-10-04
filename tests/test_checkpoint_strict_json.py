from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import core as checkpoint_core
from twelve_six.checkpoint.core import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    MAX_CHECKPOINT_CHECKSUM_BYTES,
    MAX_CHECKPOINT_MANIFEST_BYTES,
    VerifiedCheckpoint,
    canonical_json_bytes,
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


def _identity(*, training_lr: float = 0.25) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "strict-json-regression", "parameters": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"lr": training_lr},
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
    ("name", "max_bytes"),
    [
        ("manifest.json", MAX_CHECKPOINT_MANIFEST_BYTES),
        ("MANIFEST.sha256", MAX_CHECKPOINT_CHECKSUM_BYTES),
    ],
)
def test_checkpoint_metadata_read_is_bounded_before_parsing(
    tmp_path: Path, name: str, max_bytes: int
) -> None:
    checkpoint = tmp_path / "bounded-metadata"
    _save(checkpoint)
    path = checkpoint / name
    if name == "manifest.json":
        valid_prefix = path.read_bytes()
        payload = valid_prefix + b" " * (max_bytes + 1 - len(valid_prefix))
        _write_manifest_bytes(checkpoint, payload)
    else:
        path.write_bytes(path.read_bytes() + b"x" * (max_bytes + 1))

    with pytest.raises(
        CheckpointIntegrityError,
        match=f"checkpoint artifact exceeds {max_bytes}-byte limit: {name}",
    ):
        verify_checkpoint(checkpoint)


def test_manifest_growth_after_fstat_still_has_bounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "post-stat-growth"
    _save(checkpoint)
    path = checkpoint / "manifest.json"
    prefix = path.read_bytes()
    _write_manifest_bytes(
        checkpoint,
        prefix + b" " * (MAX_CHECKPOINT_MANIFEST_BYTES + 1 - len(prefix)),
    )
    original_fstat = os.fstat
    intercepted = []

    def stale_fstat(fd: int) -> os.stat_result:
        actual = original_fstat(fd)
        if actual.st_size > MAX_CHECKPOINT_MANIFEST_BYTES:
            intercepted.append(True)
            values = list(actual)
            values[6] = MAX_CHECKPOINT_MANIFEST_BYTES
            return os.stat_result(values)
        return actual

    monkeypatch.setattr(checkpoint_core.os, "fstat", stale_fstat)
    with pytest.raises(
        CheckpointIntegrityError,
        match="checkpoint artifact exceeds .*byte limit: manifest.json",
    ):
        verify_checkpoint(checkpoint)
    assert intercepted


@pytest.mark.parametrize(
    ("max_bytes", "exact_bytes"),
    [(None, None), (1, 1), (-1, None), (None, -1), (True, None)],
)
def test_checkpoint_reader_rejects_unbounded_or_invalid_limits(
    tmp_path: Path, max_bytes: int | None, exact_bytes: int | None
) -> None:
    # Validate the API contract before touching even a missing file.
    with pytest.raises(ValueError, match="checkpoint read bound"):
        checkpoint_core._read_regular_bytes(
            tmp_path, "missing", max_bytes=max_bytes, exact_bytes=exact_bytes
        )


def test_checkpoint_round_trip_with_multichunk_state_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "multichunk-payload"
    _save(checkpoint, trainer_state={"loss": 0.25})
    state_path = checkpoint / "state.json"
    original = state_path.read_bytes()
    state_path.write_bytes(original + b" " * max(0, 1024 * 1024 + 129 - len(original)))
    assert state_path.stat().st_size > 1024 * 1024
    _rebind_manifest_for_payload(checkpoint, "state.json")

    manifest = verify_checkpoint(checkpoint)
    assert manifest["files"]["state.json"]["bytes"] == state_path.stat().st_size
    target = NumpyModel([0.0, 0.0, 0.0])
    restored = load_checkpoint(checkpoint, model=target, restore_rng=False)
    assert target.loads == 1
    np.testing.assert_array_equal(target.weights, [1.0, 2.0, 3.0])
    assert restored.trainer_state["loss"] == 0.25


def test_checkpoint_reads_use_unbuffered_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "unbuffered-reads"
    _save(checkpoint)
    original_fdopen = os.fdopen
    opened: list[int] = []

    def inspect_fdopen(fd: int, *args: object, **kwargs: object) -> object:
        # A buffered stream may read ahead of the explicit size + 1 probe.
        assert kwargs.get("buffering") == 0
        opened.append(fd)
        return original_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(checkpoint_core.os, "fdopen", inspect_fdopen)
    verify_checkpoint(checkpoint)
    assert len(opened) == 5


@pytest.mark.parametrize("name", ["weights.safetensors", "state.safetensors", "state.json"])
@pytest.mark.parametrize("mutation", ["grow", "shrink"])
def test_checkpoint_payload_size_rejected_before_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, mutation: str
) -> None:
    checkpoint = tmp_path / "preflight-payload"
    _save(checkpoint, trainer_state={"loss": 0.25})
    path = checkpoint / name
    original = path.read_bytes()
    assert len(original) > 1
    path.write_bytes(original + b"tamper" if mutation == "grow" else original[:-1])
    target_stat = path.stat()
    original_fdopen = os.fdopen

    def forbid_payload_read(fd: int, *args: object, **kwargs: object) -> object:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            raise AssertionError("payload bytes read before expected size validation")
        return original_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(checkpoint_core.os, "fdopen", forbid_payload_read)
    with pytest.raises(CheckpointIntegrityError, match=f"size mismatch for {name}"):
        verify_checkpoint(checkpoint)


def test_checkpoint_payload_growth_after_stale_fstat_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "payload-growth"
    _save(checkpoint)
    path = checkpoint / "state.json"
    original = path.read_bytes()
    path.write_bytes(original + b"tamper")
    target_stat = path.stat()
    original_fstat = os.fstat
    intercepted: list[bool] = []

    def stale_fstat(fd: int) -> os.stat_result:
        actual = original_fstat(fd)
        if (actual.st_dev, actual.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            intercepted.append(True)
            values = list(actual)
            values[6] = len(original)
            return os.stat_result(values)
        return actual

    monkeypatch.setattr(checkpoint_core.os, "fstat", stale_fstat)
    with pytest.raises(CheckpointIntegrityError, match="size mismatch for state.json"):
        verify_checkpoint(checkpoint)
    assert intercepted


@pytest.mark.parametrize("operation", ["lstat", "iterdir"])
def test_checkpoint_root_metadata_denial_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    checkpoint = tmp_path / "root-denial"
    _save(checkpoint)
    original = getattr(Path, operation)

    def denied(self: Path) -> object:
        if self == checkpoint:
            raise PermissionError("injected root metadata denial")
        return original(self)

    monkeypatch.setattr(Path, operation, denied)
    with pytest.raises(CheckpointIntegrityError, match="cannot inspect checkpoint directory"):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize("name", ["manifest.json", "MANIFEST.sha256", "weights.safetensors"])
def test_checkpoint_lstat_denial_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    checkpoint = tmp_path / "lstat-denial"
    _save(checkpoint)
    target = checkpoint / name
    original_lstat = Path.lstat

    def denied_lstat(self: Path) -> os.stat_result:
        if self == target:
            raise PermissionError("injected lstat denial")
        return original_lstat(self)

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    with pytest.raises(
        CheckpointIntegrityError, match="cannot inspect checkpoint artifact"
    ):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize("name", ["manifest.json", "MANIFEST.sha256", "weights.safetensors"])
def test_checkpoint_opened_fstat_denial_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    checkpoint = tmp_path / "fstat-denial"
    _save(checkpoint)
    target_stat = (checkpoint / name).stat()
    original_fstat = os.fstat
    intercepted: list[bool] = []

    def denied_fstat(fd: int) -> os.stat_result:
        actual = original_fstat(fd)
        if (actual.st_dev, actual.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            intercepted.append(True)
            raise PermissionError("injected fstat denial")
        return actual

    monkeypatch.setattr(checkpoint_core.os, "fstat", denied_fstat)
    with pytest.raises(
        CheckpointIntegrityError, match="cannot inspect opened checkpoint artifact"
    ):
        verify_checkpoint(checkpoint)
    assert intercepted


@pytest.mark.parametrize(
    "name",
    [
        "manifest.json",
        "MANIFEST.sha256",
        "weights.safetensors",
        "state.safetensors",
        "state.json",
    ],
)
def test_checkpoint_read_denial_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    checkpoint = tmp_path / "read-denial"
    _save(checkpoint, trainer_state={"loss": 0.25})
    target_stat = (checkpoint / name).stat()
    original_fdopen = os.fdopen
    intercepted: list[bool] = []

    class FailingRead:
        def __enter__(self) -> FailingRead:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            intercepted.append(True)
            raise OSError("injected read denial")

    def denied_fdopen(fd: int, *args: object, **kwargs: object) -> object:
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            return FailingRead()
        return original_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(checkpoint_core.os, "fdopen", denied_fdopen)
    with pytest.raises(
        CheckpointIntegrityError, match="cannot safely read checkpoint artifact"
    ):
        verify_checkpoint(checkpoint)
    assert intercepted


@pytest.mark.parametrize("name", ["manifest.json", "weights.safetensors"])
def test_checkpoint_fdopen_denial_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    checkpoint = tmp_path / "fdopen-denial"
    _save(checkpoint)
    target_stat = (checkpoint / name).stat()
    original_fdopen = os.fdopen
    rejected_fd: list[int] = []

    def denied_fdopen(fd: int, *args: object, **kwargs: object) -> object:
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            rejected_fd.append(fd)
            raise OSError("injected fdopen denial")
        return original_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(checkpoint_core.os, "fdopen", denied_fdopen)
    with pytest.raises(
        CheckpointIntegrityError, match="cannot safely read checkpoint artifact"
    ):
        verify_checkpoint(checkpoint)
    assert len(rejected_fd) == 1
    with pytest.raises(OSError):
        os.fstat(rejected_fd[0])


@pytest.mark.parametrize(
    ("max_bytes", "expected_message"),
    [
        (MAX_CHECKPOINT_MANIFEST_BYTES, "cannot close checkpoint artifact: manifest.json"),
        (0, "checkpoint artifact exceeds 0-byte limit: manifest.json"),
    ],
)
def test_checkpoint_close_failure_preserves_primary_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_bytes: int, expected_message: str
) -> None:
    checkpoint = tmp_path / "close-denial"
    _save(checkpoint)
    target_stat = (checkpoint / "manifest.json").stat()
    original_close = os.close
    closed: list[int] = []

    def failing_close(fd: int) -> None:
        opened = os.fstat(fd)
        original_close(fd)
        if (opened.st_dev, opened.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            closed.append(fd)
            raise OSError("injected close denial after underlying close")

    monkeypatch.setattr(checkpoint_core.os, "close", failing_close)
    with pytest.raises(CheckpointIntegrityError, match=expected_message) as caught:
        checkpoint_core._read_regular_bytes(
            checkpoint, "manifest.json", max_bytes=max_bytes
        )
    assert len(closed) == 1
    with pytest.raises(OSError):
        os.fstat(closed[0])
    if max_bytes == 0:
        assert caught.value.__cause__ is None
    else:
        assert isinstance(caught.value.__cause__, OSError)


def test_checkpoint_close_denial_ignores_unrelated_ambient_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "ambient-close-denial"
    _save(checkpoint)
    target_stat = (checkpoint / "manifest.json").stat()
    original_close = os.close

    def failing_close(fd: int) -> None:
        opened = os.fstat(fd)
        original_close(fd)
        if (opened.st_dev, opened.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            raise OSError("injected close denial after underlying close")

    monkeypatch.setattr(checkpoint_core.os, "close", failing_close)
    try:
        raise RuntimeError("unrelated caller exception")
    except RuntimeError:
        with pytest.raises(
            CheckpointIntegrityError, match="cannot close checkpoint artifact: manifest.json"
        ) as caught:
            checkpoint_core._read_regular_bytes(
                checkpoint, "manifest.json", max_bytes=MAX_CHECKPOINT_MANIFEST_BYTES
            )
        assert isinstance(caught.value.__cause__, OSError)


def test_checkpoint_payload_shrink_after_stale_fstat_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "payload-shrink"
    _save(checkpoint)
    path = checkpoint / "state.json"
    original = path.read_bytes()
    assert len(original) > 1
    path.write_bytes(original[:-1])
    target_stat = path.stat()
    original_fstat = os.fstat
    intercepted: list[bool] = []

    def stale_fstat(fd: int) -> os.stat_result:
        actual = original_fstat(fd)
        if (actual.st_dev, actual.st_ino) == (target_stat.st_dev, target_stat.st_ino):
            intercepted.append(True)
            values = list(actual)
            values[6] = len(original)
            return os.stat_result(values)
        return actual

    monkeypatch.setattr(checkpoint_core.os, "fstat", stale_fstat)
    with pytest.raises(CheckpointIntegrityError, match="size mismatch for state.json"):
        verify_checkpoint(checkpoint)
    assert intercepted


@pytest.mark.parametrize("token", ["1e-4000", "-1e-4000", "0.0001e-4000"])
def test_checkpoint_manifest_rejects_nonzero_numeric_underflow(
    tmp_path: Path, token: str
) -> None:
    checkpoint = tmp_path / "manifest-underflow"
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        identity=_identity(training_lr=0.0),
    )
    raw = (checkpoint / "manifest.json").read_text(encoding="utf-8")
    assert '"lr":0.0' in raw
    _write_manifest_bytes(checkpoint, raw.replace('"lr":0.0', f'"lr":{token}', 1).encode("utf-8"))

    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize("token", ["1e-4000", "-1e-4000", "0.0001e-4000"])
def test_checkpoint_state_rejects_nonzero_numeric_underflow_before_mutation(
    tmp_path: Path, token: str
) -> None:
    checkpoint = tmp_path / "state-underflow"
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


def test_checkpoint_allows_lexically_zero_finite_exponent(tmp_path: Path) -> None:
    checkpoint = tmp_path / "manifest-genuine-zero"
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1.0, 2.0, 3.0]),
        identity=_identity(training_lr=0.0),
    )
    raw = (checkpoint / "manifest.json").read_text(encoding="utf-8")
    assert '"lr":0.0' in raw
    _write_manifest_bytes(checkpoint, raw.replace('"lr":0.0', '"lr":0e-4000', 1).encode("utf-8"))
    assert verify_checkpoint(checkpoint)["identity"]["training_config"]["lr"] == 0.0


@pytest.mark.parametrize(
    ("needle", "duplicated"),
    [
        (
            '"format":"12-6-checkpoint"',
            '"format":"attacker-controlled","format":"12-6-checkpoint"',
        ),
        (
            '"lr":0.25',
            '"lr":999.0,"lr":0.25',
        ),
    ],
)
def test_checksum_consistent_manifest_duplicate_keys_fail_closed(
    tmp_path: Path,
    needle: str,
    duplicated: str,
) -> None:
    checkpoint = tmp_path / "manifest-duplicate"
    _save(checkpoint)
    path = checkpoint / "manifest.json"
    raw = path.read_text(encoding="utf-8")
    assert needle in raw
    raw = raw.replace(needle, duplicated, 1)
    _write_manifest_bytes(checkpoint, raw.encode("utf-8"))

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
    path.write_text(
        raw.replace(
            needle,
            '"__kind__":"attacker-controlled","__kind__":"mapping"',
            1,
        ),
        encoding="utf-8",
    )
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_save_rejects_nonfinite_identity_before_publication(
    tmp_path: Path,
    value: float,
) -> None:
    checkpoint = tmp_path / "save-nonfinite-identity"

    with pytest.raises(CheckpointIntegrityError, match="strict finite JSON"):
        save_checkpoint(
            checkpoint,
            model=NumpyModel([1.0, 2.0, 3.0]),
            identity=_identity(training_lr=value),
        )

    assert not checkpoint.exists()


def test_finite_checkpoint_json_bytes_remain_canonical_v1(tmp_path: Path) -> None:
    checkpoint = tmp_path / "canonical-v1"
    _save(checkpoint, trainer_state={"loss": 0.25})

    for name in ("manifest.json", "state.json"):
        raw = (checkpoint / name).read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
        assert raw == canonical_json_bytes(parsed) + b"\n"


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

# External checkpoint JSON depth is a recoverable integrity failure, not a raw
# Python interpreter exception. These fixtures reseal all outer hashes so the
# strict decoder itself must enforce the recovery contract.


def _deep_json(nesting: str) -> str:
    if nesting == "arrays":
        return "[" * 10_000 + "0" + "]" * 10_000
    if nesting == "objects":
        return '{"k":' * 10_000 + "0" + "}" * 10_000
    raise AssertionError(f"unknown nesting: {nesting}")


@pytest.mark.parametrize("nesting", ["arrays", "objects"])
def test_checksum_consistent_deep_manifest_is_checkpoint_integrity_error(
    tmp_path: Path,
    nesting: str,
) -> None:
    checkpoint = tmp_path / f"deep-manifest-{nesting}"
    _save(checkpoint)
    raw = (checkpoint / "manifest.json").read_text(encoding="utf-8").strip()
    assert raw.endswith("}")
    malformed = raw[:-1] + ',"excessive":' + _deep_json(nesting) + "}\n"
    _write_manifest_bytes(checkpoint, malformed.encode("utf-8"))

    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize("nesting", ["arrays", "objects"])
def test_checksum_consistent_deep_state_fails_before_model_mutation(
    tmp_path: Path,
    nesting: str,
) -> None:
    checkpoint = tmp_path / f"deep-state-{nesting}"
    _save(checkpoint, trainer_state={"loss": 0.25})
    path = checkpoint / "state.json"
    raw = path.read_text(encoding="utf-8").strip()
    assert raw.endswith("}")
    path.write_text(
        raw[:-1] + ',"excessive":' + _deep_json(nesting) + "}\n",
        encoding="utf-8",
    )
    _rebind_manifest_for_payload(checkpoint, "state.json")

    target = NumpyModel([9.0, 9.0, 9.0])
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


@pytest.mark.parametrize("nesting", ["arrays", "objects"])
def test_immutable_verified_manifest_rejects_excessive_json_depth(
    nesting: str,
) -> None:
    verified = VerifiedCheckpoint(
        _manifest_bytes=('{"root":' + _deep_json(nesting) + "}").encode("utf-8"),
        _artifacts={},
    )
    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        _ = verified.manifest


@pytest.mark.parametrize("nesting", ["arrays", "objects"])
def test_deep_producer_json_fails_before_output_publication(
    tmp_path: Path,
    nesting: str,
) -> None:
    import twelve_six.checkpoint.core as checkpoint_core

    value: object = 0
    for _ in range(2_000):
        value = [value] if nesting == "arrays" else {"k": value}
    output = tmp_path / "never-publish.json"
    with pytest.raises(CheckpointIntegrityError, match="strict finite JSON"):
        checkpoint_core._write_json(output, {"root": value})
    assert not output.exists()


def test_unexpected_postparse_checkpoint_recursion_is_not_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import twelve_six.checkpoint.core as checkpoint_core

    checkpoint = tmp_path / "valid-input-unexpected-recursion"
    _save(checkpoint)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise RecursionError("unexpected checkpoint Product recursion")

    monkeypatch.setattr(
        checkpoint_core, "_validate_manifest_identity", unexpected
    )
    with pytest.raises(RecursionError, match="unexpected checkpoint Product recursion"):
        verify_checkpoint(checkpoint)
