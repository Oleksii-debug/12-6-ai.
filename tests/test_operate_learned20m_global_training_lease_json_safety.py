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
MAX_JSON_INTEGER_DIGITS = _OPERATOR["MAX_JSON_INTEGER_DIGITS"]
_load_mapping = _OPERATOR["_load_mapping"]
_OPERATOR_OS = _load_mapping.__globals__["os"]
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
        (
            b'{"value":1e-9999}',
            "nonzero_json_number_underflowed_to_zero",
        ),
        (
            b'{"value":' + b"9" * 100_000 + b"}",
            "json_integer_exceeds_64_digits",
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




@pytest.mark.parametrize("literal", ["1e-9999", "-1e-9999", "5.4e-9999"])
def test_manifest_loader_rejects_nonzero_float_underflow(
    tmp_path: Path,
    literal: str,
) -> None:
    path = tmp_path / "underflow.json"
    path.write_text('{"value":' + literal + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="nonzero_json_number_underflowed_to_zero"):
        _load_mapping(path)


@pytest.mark.parametrize("sign", ["", "-"])
def test_manifest_loader_accepts_64_digit_integer_boundary(
    tmp_path: Path,
    sign: str,
) -> None:
    literal = sign + "9" * MAX_JSON_INTEGER_DIGITS
    path = tmp_path / "int-boundary.json"
    path.write_text('{"value":' + literal + "}", encoding="utf-8")
    assert _load_mapping(path) == {"value": int(literal)}


def test_manifest_loader_bounds_integer_before_python_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "huge-int.json"
    path.write_text('{"value":' + "9" * 100_000 + "}", encoding="utf-8")
    before = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        with pytest.raises(ValueError, match="json_integer_exceeds_64_digits"):
            _load_mapping(path)
    finally:
        sys.set_int_max_str_digits(before)


def test_manifest_loader_rejects_descriptor_stamp_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "drifting.json"
    path.write_bytes(b'{"ok":true}')
    actual_fstat = _OPERATOR_OS.fstat
    calls = 0

    def drifting_fstat(fd: int):
        nonlocal calls
        info = actual_fstat(fd)
        calls += 1
        if calls >= 2:
            values = list(info)
            values[6] = info.st_size + 1
            return _OPERATOR_OS.stat_result(values)
        return info

    monkeypatch.setattr(_OPERATOR_OS, "fstat", drifting_fstat)
    with pytest.raises(ValueError, match="manifest_changed_during_read"):
        _load_mapping(path)


def test_manifest_loader_rejects_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    mkfifo = getattr(_OPERATOR_OS, "mkfifo", None)
    if not callable(mkfifo):
        pytest.skip("FIFO creation unavailable on this platform")
    path = tmp_path / "manifest.fifo"
    try:
        mkfifo(path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"FIFO unavailable on this filesystem: {exc}")
    with pytest.raises(ValueError, match="manifest_not_regular_file"):
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
        (
            b'{"value":1e-9999}',
            "nonzero_json_number_underflowed_to_zero",
        ),
        (
            b'{"value":' + b"9" * 100_000 + b"}",
            "json_integer_exceeds_64_digits",
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
