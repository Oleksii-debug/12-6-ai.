"""Reject unbounded cost integers without widening Windows LOCAL_FREE authority."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from twelve_six import windows_operator_preflight as operator

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/research/r01_windows_local_free_operator_v1.json"
PACKET = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"


def _inputs() -> tuple[dict, dict]:
    return (
        json.loads(PROFILE.read_text(encoding="utf-8")),
        json.loads(PACKET.read_text(encoding="utf-8")),
    )


@pytest.mark.parametrize("zero", [0, 0.0, -0.0])
def test_zero_cost_stays_eligible_for_operator_preflight_only(zero: object) -> None:
    profile, packet = _inputs()
    packet["resource"]["maximum_cost_usd"] = zero

    assert operator.validate_operator_profile(profile, packet) == []


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(10**400, id="huge-int-overflow"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(True, id="bool-one"),
        pytest.param(False, id="bool-zero"),
        pytest.param(-1, id="negative-int"),
        pytest.param("0", id="string-zero"),
    ],
)
def test_invalid_cost_fails_closed_without_numeric_conversion(
    invalid: object,
) -> None:
    profile, original_packet = _inputs()
    packet = copy.deepcopy(original_packet)
    packet["resource"]["maximum_cost_usd"] = invalid

    errors = operator.validate_operator_profile(profile, packet)

    assert "packet_maximum_cost_must_be_zero" in errors
    assert "portable_packet_contract:maximum_cost_usd_must_be_zero" in errors


def test_cli_reports_huge_cost_as_one_line_json_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    profile, packet = _inputs()
    packet["resource"]["maximum_cost_usd"] = 10**400
    profile_path = tmp_path / "профіль оператора.json"
    packet_path = tmp_path / "пакет із пробілами.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    packet_path.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")

    code = operator.main([
        "--profile", str(profile_path),
        "--packet", str(packet_path),
        "--json", "verify", "--target", "20m",
    ])
    captured = capsys.readouterr()

    assert code == operator.EXIT_ERROR
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    result = json.loads(captured.out)
    assert result["status"] == "ERROR"
    assert "packet_maximum_cost_must_be_zero" in result["error"]
    assert result["launch_authorized"] is False
    assert result["training_authorized"] is False
    assert result["truth_boundary"] == operator._TRUTH_BOUNDARY


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            ('{"nested":' + "[" * 80 + "0" + "]" * 80 + "}").encode(),
            id="excessive-structure-depth",
        ),
        pytest.param(
            ('{"nested":' + "[" * 10000 + "0" + "]" * 10000 + "}").encode(),
            id="parser-recursion",
        ),
        pytest.param(
            (" " * (operator.MAX_OPERATOR_JSON_BYTES + 1)).encode(),
            id="oversized-input",
        ),
        pytest.param(r'{"nested":"\ud800"}'.encode(), id="unpaired-surrogate"),
        pytest.param(b'{"nested":"\xff"}', id="invalid-utf8"),
    ],
)
def test_profile_packet_reader_rejects_untrusted_json_limits(
    tmp_path: Path, raw: bytes,
) -> None:
    path = tmp_path / "неправильний файл.json"
    path.write_bytes(raw)
    with pytest.raises(operator.OperatorPreflightError, match="invalid_or_unreadable_json"):
        operator._read_json_file(path)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            ('{"nested":' + "[" * 80 + "0" + "]" * 80 + "}").encode(),
            id="excessive-structure-depth",
        ),
        pytest.param(
            ('{"nested":' + "[" * 10000 + "0" + "]" * 10000 + "}").encode(),
            id="parser-recursion",
        ),
        pytest.param(
            (" " * (operator.MAX_OPERATOR_JSON_BYTES + 1)).encode(),
            id="oversized-marker",
        ),
        pytest.param(r'{"marker_sha256":"0","nested":"\ud800"}'.encode(), id="surrogate-marker"),
        pytest.param(b'{"marker_sha256":"0","nested":"\xff"}', id="invalid-utf8"),
    ],
)
def test_safe_stop_status_fails_closed_on_malformed_marker(
    tmp_path: Path, raw: bytes,
) -> None:
    state = tmp_path / "стан із пробілами"
    state.mkdir()
    (state / "STOP_REQUEST.json").write_bytes(raw)

    assert operator.read_stop_status(
        state,
        profile_sha256="a" * 64,
        packet_sha256="b" * 64,
        target="20m",
    ) == {
        "status": "INVALID",
        "marker": None,
        "errors": ["safe_stop_marker_invalid_json"],
    }


@pytest.mark.parametrize(
    "token",
    [
        "1e-9999",
        "-1e-9999",
        "0.000000000000000000000001e-9999",
        "-0.000000000000000000000001e-9999",
    ],
)
def test_nonzero_json_cost_cannot_underflow_to_free(
    tmp_path: Path, token: str, capsys: pytest.CaptureFixture[str],
) -> None:
    profile, packet = _inputs()
    profile_path = tmp_path / "профіль.json"
    packet_path = tmp_path / "пакет.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    payload = json.dumps(packet)
    marker = '"maximum_cost_usd": 0'
    assert payload.count(marker) == 1
    packet_path.write_text(
        payload.replace(marker, f'"maximum_cost_usd": {token}'),
        encoding="utf-8",
    )

    code = operator.main([
        "--profile", str(profile_path),
        "--packet", str(packet_path),
        "--json", "verify", "--target", "20m",
    ])
    captured = capsys.readouterr()
    assert code == operator.EXIT_ERROR
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    result = json.loads(captured.out)
    assert result["status"] == "ERROR"
    assert "nonzero_number_underflowed_to_zero" in result["error"]
    assert result["launch_authorized"] is False
    assert result["training_authorized"] is False


@pytest.mark.parametrize("token", ["0e-9999", "-0.000e-9999", "0.0"])
def test_lexical_zero_json_float_remains_readable(
    tmp_path: Path, token: str,
) -> None:
    path = tmp_path / "нуль.json"
    path.write_text('{"cost": ' + token + '}', encoding="utf-8")
    parsed, _digest = operator._read_json_file(path)
    assert parsed == {"cost": 0.0}


@pytest.mark.parametrize("broken_input", ["profile", "packet"])
@pytest.mark.parametrize("json_mode", [True, False], ids=["json", "text"])
def test_duplicate_surrogate_key_never_breaks_operator_error_output(
    tmp_path: Path, broken_input: str, json_mode: bool,
) -> None:
    malformed = tmp_path / "некоректний Unicode.json"
    malformed.write_text(
        r'{"\ud800": 1, "\ud800": 2}',
        encoding="utf-8",
    )
    profile = malformed if broken_input == "profile" else PROFILE
    packet = malformed if broken_input == "packet" else PACKET
    args = [
        sys.executable,
        "-m", "twelve_six.windows_operator_preflight",
        "--profile", str(profile),
        "--packet", str(packet),
    ]
    if json_mode:
        args.append("--json")
    args.extend(["verify", "--target", "20m"])
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == operator.EXIT_ERROR
    assert proc.stderr == ""
    assert proc.stdout.count("\n") == 1
    assert r"\ud800" in proc.stdout
    if json_mode:
        result = json.loads(proc.stdout)
        assert result["status"] == "ERROR"
        assert "duplicate_object_key:" in result["error"]
        assert result["launch_authorized"] is False
        assert result["training_authorized"] is False
    else:
        assert "OPERATOR_STATUS: ERROR" in proc.stdout
        assert "LAUNCH_AUTHORIZED: false" in proc.stdout
        assert "TRAINING_AUTHORIZED: false" in proc.stdout


@pytest.mark.parametrize("json_mode", [True, False], ids=["json", "text"])
def test_non_utf8_stdout_preserves_operator_diagnostic_and_no_authority(
    tmp_path: Path, json_mode: bool,
) -> None:
    malformed = tmp_path / "некоректний профіль.json"
    malformed.write_text("{not json", encoding="utf-8")
    args = [
        sys.executable,
        "-m", "twelve_six.windows_operator_preflight",
        "--profile", str(malformed),
        "--packet", str(PACKET),
    ]
    if json_mode:
        args.append("--json")
    args.extend(["verify", "--target", "20m"])
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONIOENCODING": "ascii",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == operator.EXIT_ERROR
    assert proc.stderr == ""
    assert proc.stdout.isascii()
    assert proc.stdout.count("\n") == 1
    if json_mode:
        result = json.loads(proc.stdout)
        assert result["status"] == "ERROR"
        assert str(malformed) in result["error"]
        assert result["launch_authorized"] is False
        assert result["training_authorized"] is False
        assert r"\u" in proc.stdout
    else:
        assert "OPERATOR_STATUS: ERROR" in proc.stdout
        assert r"\u" in proc.stdout
        assert "LAUNCH_AUTHORIZED: false" in proc.stdout
        assert "TRAINING_AUTHORIZED: false" in proc.stdout
