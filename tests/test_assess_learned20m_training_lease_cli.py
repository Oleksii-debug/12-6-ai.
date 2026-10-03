from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_cli() -> ModuleType:
    script = Path(__file__).parents[1] / "tools" / "assess_learned20m_training_lease.py"
    spec = importlib.util.spec_from_file_location(
        "assess_learned20m_training_lease_cli", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, raw: str) -> Path:
    path.write_text(raw, encoding="utf-8")
    return path


def test_cli_malformed_enum_returns_machine_readable_denial(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"resource": {"resource_class": []}}),
        encoding="utf-8",
    )

    result = _load_cli().main(
        ["assess_learned20m_training_lease.py", str(manifest_path)]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert result == 1
    assert payload["contract_valid"] is False
    assert payload["local_duplicate_guard_open"] is False
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "raw",
    [
        '{"manifest_id":"a","manifest_id":"b"}',
        '{"outer":{"lease_id":"a","lease_id":"b"}}',
    ],
)
def test_strict_reader_rejects_duplicate_members(tmp_path: Path, raw: str) -> None:
    cli = _load_cli()
    path = _write(tmp_path / "input.json", raw)
    with pytest.raises(ValueError, match="duplicate object member"):
        cli._read_object(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_reader_rejects_nonfinite_constants(
    tmp_path: Path, constant: str
) -> None:
    cli = _load_cli()
    path = _write(tmp_path / "input.json", f'{{"value":{constant}}}')
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        cli._read_object(path)


@pytest.mark.parametrize("number", ["1e400", "-1e400"])
def test_strict_reader_rejects_float_overflow(
    tmp_path: Path, number: str
) -> None:
    cli = _load_cli()
    path = _write(tmp_path / "input.json", f'{{"value":{number}}}')
    with pytest.raises(ValueError, match="JSON number is not finite"):
        cli._read_object(path)


def test_strict_reader_preserves_valid_finite_json(tmp_path: Path) -> None:
    cli = _load_cli()
    path = _write(
        tmp_path / "input.json",
        '{"nested":{"small":1.25,"large":1e20},"flag":false}',
    )
    assert cli._read_object(path) == {
        "nested": {"small": 1.25, "large": 1e20},
        "flag": False,
    }


@pytest.mark.parametrize("raw", ["[]", "null", "1", '"lease"'])
def test_strict_reader_requires_object_root(tmp_path: Path, raw: str) -> None:
    cli = _load_cli()
    path = _write(tmp_path / "input.json", raw)
    with pytest.raises(ValueError, match="root must be an object"):
        cli._read_object(path)


def test_cli_rejects_ambiguous_manifest_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    manifest = _write(
        tmp_path / "manifest.json",
        '{"manifest_id":"a","manifest_id":"b"}',
    )
    assert cli.main(["assess_learned20m_training_lease.py", str(manifest)]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["contract_valid"] is False
    assert "duplicate object member" in payload["error"]
    assert captured.err == ""


def test_cli_rejects_ambiguous_lease_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    manifest = _write(tmp_path / "manifest.json", "{}")
    lease = _write(
        tmp_path / "lease.json",
        '{"run_id":"a","run_id":"b"}',
    )
    assert (
        cli.main(
            [
                "assess_learned20m_training_lease.py",
                str(manifest),
                str(lease),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["contract_valid"] is False
    assert "duplicate object member" in payload["error"]
    assert captured.err == ""
