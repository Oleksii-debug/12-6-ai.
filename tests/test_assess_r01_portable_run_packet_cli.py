from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "assess_r01_portable_run_packet.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assess_r01_portable_run_packet_cli", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, raw: str) -> Path:
    path = tmp_path / "packet.json"
    path.write_text(raw, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "raw",
    [
        '{"provider":"a","provider":"b"}',
        '{"runtime":{"kind":"a","kind":"b"}}',
    ],
)
def test_strict_loader_rejects_duplicate_members(tmp_path: Path, raw: str) -> None:
    cli = _load_cli()
    with pytest.raises(ValueError, match="duplicate object member"):
        cli._load_packet(_write(tmp_path, raw))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_loader_rejects_nonfinite_constants(
    tmp_path: Path, constant: str
) -> None:
    cli = _load_cli()
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        cli._load_packet(_write(tmp_path, f'{{"value":{constant}}}'))


@pytest.mark.parametrize("number", ["1e400", "-1e400"])
def test_strict_loader_rejects_float_overflow(
    tmp_path: Path, number: str
) -> None:
    cli = _load_cli()
    with pytest.raises(ValueError, match="JSON number is not finite"):
        cli._load_packet(_write(tmp_path, f'{{"value":{number}}}'))


@pytest.mark.parametrize("raw", ["[]", "null", "1", '"packet"'])
def test_strict_loader_requires_object_root(tmp_path: Path, raw: str) -> None:
    cli = _load_cli()
    with pytest.raises(ValueError, match="run packet root must be an object"):
        cli._load_packet(_write(tmp_path, raw))


def test_strict_loader_preserves_valid_finite_json(tmp_path: Path) -> None:
    cli = _load_cli()
    payload = cli._load_packet(
        _write(
            tmp_path,
            '{"provider":"LOCAL_FREE","limits":{"hours":2.5,"bytes":1e6}}',
        )
    )
    assert payload == {
        "provider": "LOCAL_FREE",
        "limits": {"hours": 2.5, "bytes": 1e6},
    }


def test_main_rejects_ambiguous_packet_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    path = _write(tmp_path, '{"provider":"a","provider":"b"}')
    assert cli.main(["assess_r01_portable_run_packet.py", str(path)]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["contract_valid"] is False
    assert "duplicate object member" in payload["error"]
    assert captured.err == ""


def test_main_reports_missing_file_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    missing = tmp_path / "missing.json"
    assert cli.main(["assess_r01_portable_run_packet.py", str(missing)]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["contract_valid"] is False
    assert captured.err == ""
