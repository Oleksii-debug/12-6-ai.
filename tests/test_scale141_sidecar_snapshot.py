from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import twelve_six.scale141_resume_sidecar as sidecar
from twelve_six.scale141_resume_sidecar import ResumeSidecarError, _read_payload


def _write(path: Path, payload: dict[str, object]) -> str:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_read_payload_uses_exact_opened_snapshot_when_path_changes_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    expected_payload = {"generation": "generation-00000001", "value": "verified"}
    expected_hash = _write(path, expected_payload)
    replacement = tmp_path / "replacement.json"
    _write(replacement, {"generation": "generation-00000001", "value": "tampered"})
    real_open = sidecar.os.open
    swapped = False

    def open_then_swap(target: os.PathLike[str] | str, flags: int, *args: object) -> int:
        nonlocal swapped
        fd = real_open(target, flags, *args)
        if Path(target) == path and not swapped:
            swapped = True
            os.replace(replacement, path)
        return fd

    monkeypatch.setattr(sidecar.os, "open", open_then_swap)

    observed = _read_payload(path, expected_hash)

    assert swapped is True
    assert observed == expected_payload
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == "tampered"


def test_read_payload_rejects_path_swap_between_lstat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    expected_hash = _write(path, {"value": "verified"})
    replacement = tmp_path / "replacement.json"
    _write(replacement, {"value": "tampered"})
    real_open = sidecar.os.open
    swapped = False

    def swap_then_open(target: os.PathLike[str] | str, flags: int, *args: object) -> int:
        nonlocal swapped
        if Path(target) == path and not swapped:
            swapped = True
            os.replace(replacement, path)
        return real_open(target, flags, *args)

    monkeypatch.setattr(sidecar.os, "open", swap_then_open)

    with pytest.raises(ResumeSidecarError, match="changed while opening"):
        _read_payload(path, expected_hash)


def test_read_payload_rejects_symlink_even_when_target_hash_matches(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    expected_hash = _write(target, {"value": "verified"})
    link = tmp_path / "state.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(ResumeSidecarError, match="regular non-symlink"):
        _read_payload(link, expected_hash)
