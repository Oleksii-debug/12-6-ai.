from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.probe_d03_rada_bulk_source import DEFAULT_CONFIG, ProbeError, _load_config


def _base() -> dict:
    return json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "mutated-rada-probe.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("archive_md5", "0" * 32),
        ("archive_bytes", 1),
        ("portal_file_count", 1),
        ("status", "TERMINAL"),
    ],
)
def test_discovery_observation_drift_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = copy.deepcopy(_base())
    config["discovery_observation"][field] = value

    with pytest.raises(ProbeError, match=f"discovery_observation.{field} drifted"):
        _load_config(_write(tmp_path, config))


def test_downstream_gate_removal_fails_closed(tmp_path: Path) -> None:
    config = copy.deepcopy(_base())
    config["downstream_required"].remove("EVALUATION_DECONTAMINATION")

    with pytest.raises(ProbeError, match="downstream_required drifted"):
        _load_config(_write(tmp_path, config))


def test_source_title_drift_fails_closed(tmp_path: Path) -> None:
    config = copy.deepcopy(_base())
    config["source"]["title"] = "other dataset"

    with pytest.raises(ProbeError, match="source.title drifted"):
        _load_config(_write(tmp_path, config))
