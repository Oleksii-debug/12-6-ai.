"""Fail-closed JSON input acceptance for the learned-20M lease operator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from runpy import run_path

import pytest

# The operator is a repository tool, not an installed Python package.
_OPERATOR = run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "tools"
        / "operate_learned20m_global_training_lease.py"
    )
)
MAX_MANIFEST_BYTES = _OPERATOR["MAX_MANIFEST_BYTES"]
_load_mapping = _OPERATOR["_load_mapping"]
main = _OPERATOR["main"]


@pytest.mark.parametrize(
    ("payload", "blocker"),
    [
        (
            b'{"padding":"' + b"a" * MAX_MANIFEST_BYTES,
            "manifest_exceeds_byte_limit",
        ),
        (
            b'{"nested":' + b"[" * 10_000 + b"0" + b"]" * 10_000 + b"}",
            "manifest_json_invalid",
        ),
        (b"\xff", "manifest_utf8_invalid"),
        (b'{"a":1,"a":2}', "duplicate_json_key:a"),
        (b'{"n":NaN}', "non_finite_json_constant:NaN"),
    ],
)
def test_manifest_json_input_fails_closed(
    tmp_path: Path, payload: bytes, blocker: str
) -> None:
    path = tmp_path / "untrusted-manifest.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=blocker):
        _load_mapping(path)


def test_small_valid_manifest_json_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "valid.json"
    path.write_bytes(b'{"ok":true}')
    assert _load_mapping(path) == {"ok": True}


@pytest.mark.parametrize(
    ("payload", "blocker"),
    [
        (b'{"padding":"' + b"a" * MAX_MANIFEST_BYTES, "manifest_exceeds_byte_limit"),
        (
            b'{"nested":' + b"[" * 10_000 + b"0" + b"]" * 10_000 + b"}",
            "manifest_json_invalid",
        ),
    ],
)
def test_operator_emits_structured_failure_before_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: bytes,
    blocker: str,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "operate_learned20m_global_training_lease.py",
            "--manifest",
            str(path),
            "inspect",
        ],
    )
    assert main() == 2
    assert json.loads(capsys.readouterr().out) == {
        "operation": "inspect",
        "ok": False,
        "blockers": [blocker],
    }
