from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six import launch_gate
from twelve_six import scale141_10m_runtime_v3 as runtime


def test_scale141_historical_missing_pytest_stops_before_project_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema": launch_gate.REQUEST_SCHEMA,
                "required_modules": ["pytest"],
            }
        ),
        encoding="utf-8",
    )
    heavy_called = False

    monkeypatch.setattr(
        launch_gate.importlib.util,
        "find_spec",
        lambda name: None if name == "pytest" else object(),
    )

    def heavy(_: dict[str, object]):
        nonlocal heavy_called
        heavy_called = True
        raise AssertionError("project/model setup must not be reached")

    monkeypatch.setattr(launch_gate, "_verify_project_contracts", heavy)
    with pytest.raises(launch_gate.LaunchGateError, match="required module unavailable: pytest"):
        launch_gate.create_launch_envelope(tmp_path, request_path, tmp_path / "envelope.json")
    assert heavy_called is False
    assert not (tmp_path / "envelope.json").exists()


def test_scale141_long_phases_refuse_absent_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TWELVE_SIX_LAUNCH_REQUEST", raising=False)
    monkeypatch.delenv("TWELVE_SIX_LAUNCH_ENVELOPE", raising=False)
    with pytest.raises(launch_gate.LaunchGateError, match="mandatory"):
        runtime._require_launch_gate(tmp_path)


def test_scale141_binding_is_exact() -> None:
    assert runtime._LAUNCH_BINDING == {
        "workflow": "scale141-10m-learned-continuation",
        "scale": "10m",
    }
