"""Reject unbounded cost integers without widening Windows LOCAL_FREE authority."""

from __future__ import annotations

import copy
import json
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
