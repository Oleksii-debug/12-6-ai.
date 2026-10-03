from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import twelve_six.scale141_recovery as recovery
import twelve_six.scale141_resume_sidecar as sidecar
from twelve_six.scale141_strict_json import (
    Scale141StrictJsonError,
    strict_json_loads,
)


@pytest.mark.parametrize(
    "raw",
    [
        '{"key":1,"key":2}',
        '{"outer":{"key":1,"key":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":-1e400}',
    ],
)
def test_strict_json_rejects_duplicate_or_nonfinite_values(raw: str) -> None:
    with pytest.raises(Scale141StrictJsonError):
        strict_json_loads(raw)


def test_strict_json_preserves_valid_finite_json_semantics() -> None:
    assert strict_json_loads('{"a":[1,2.5,{"b":true}]}') == {
        "a": [1, 2.5, {"b": True}]
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema":"first","schema":"second"}',
        '{"nested":{"key":1,"key":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
    ],
)
def test_recovery_pointer_rejects_ambiguous_json_before_semantic_validation(
    tmp_path: Path, raw: str
) -> None:
    (tmp_path / recovery.CURRENT_NAME).write_text(raw, encoding="utf-8")

    with pytest.raises(recovery.RecoveryLifecycleError, match="pointer is unreadable") as exc:
        recovery._read_pointer(tmp_path)

    assert isinstance(exc.value.__cause__, Scale141StrictJsonError)


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema":"first","schema":"second"}',
        '{"nested":{"key":1,"key":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
    ],
)
def test_resume_sidecar_rejects_hash_consistent_ambiguous_json(
    tmp_path: Path, raw: str
) -> None:
    payload = raw.encode("utf-8")
    path = tmp_path / sidecar.SIDECAR_FILE
    path.write_bytes(payload)

    with pytest.raises(sidecar.ResumeSidecarError, match="sidecar is unreadable") as exc:
        sidecar._read_payload(
            path,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        )

    assert isinstance(exc.value.__cause__, Scale141StrictJsonError)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_recovery_pointer_writer_rejects_nonfinite_without_publication(
    tmp_path: Path, value: float
) -> None:
    with pytest.raises(
        recovery.RecoveryLifecycleError,
        match="standards-strict JSON",
    ):
        recovery._atomic_publish_pointer(tmp_path, {"value": value})

    assert not (tmp_path / recovery.CURRENT_NAME).exists()
    assert list(tmp_path.glob(".current.*.tmp")) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_resume_sidecar_writer_rejects_nonfinite_without_publication(
    tmp_path: Path, value: float
) -> None:
    with pytest.raises(
        sidecar.ResumeSidecarError,
        match="standards-strict JSON",
    ):
        sidecar._write_payload_directory(tmp_path, 1, {"value": value})

    assert not (tmp_path / "generation-00000001").exists()
    assert list(tmp_path.glob(".generation-00000001.tmp-*")) == []


def test_finite_recovery_and_sidecar_writer_bytes_remain_canonical(tmp_path: Path) -> None:
    pointer_root = tmp_path / "pointer"
    recovery._atomic_publish_pointer(pointer_root, {"b": 2, "a": 1.5})
    assert (pointer_root / recovery.CURRENT_NAME).read_bytes() == b'{"a":1.5,"b":2}\n'

    sidecar_root = tmp_path / "sidecar"
    sidecar_root.mkdir()
    directory = sidecar._write_payload_directory(
        sidecar_root,
        1,
        {"b": 2, "a": 1.5},
    )
    assert (directory / sidecar.SIDECAR_FILE).read_bytes() == b'{"a":1.5,"b":2}\n'
