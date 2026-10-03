from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "operate_learned20m_global_training_lease.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("global_lease_operator_cli", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, raw: str) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(raw, encoding="utf-8")
    return path


@pytest.mark.parametrize("raw", ['{"value":1e400}', '{"value":-1e400}'])
def test_load_mapping_rejects_float_overflow(tmp_path: Path, raw: str) -> None:
    tool = _load_tool()
    with pytest.raises(ValueError, match="json_number_not_finite"):
        tool._load_mapping(_write(tmp_path, raw))


@pytest.mark.parametrize("raw", ['{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}'])
def test_load_mapping_preserves_nonfinite_constant_rejection(
    tmp_path: Path, raw: str
) -> None:
    tool = _load_tool()
    with pytest.raises(ValueError, match="non_finite_json_constant"):
        tool._load_mapping(_write(tmp_path, raw))


def test_load_mapping_preserves_valid_finite_float(tmp_path: Path) -> None:
    tool = _load_tool()
    assert tool._load_mapping(_write(tmp_path, '{"value":1e20}')) == {"value": 1e20}


def test_load_mapping_preserves_duplicate_key_rejection(tmp_path: Path) -> None:
    tool = _load_tool()
    with pytest.raises(ValueError, match="duplicate_json_key:value"):
        tool._load_mapping(_write(tmp_path, '{"value":1,"value":2}'))
