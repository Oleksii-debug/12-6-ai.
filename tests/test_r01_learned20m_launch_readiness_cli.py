from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "assess_r01_learned20m_launch_readiness.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("r01_readiness_cli", TOOL)
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
        '{"schema_version":1,"schema_version":1}',
        '{"outer":{"value":1,"value":2}}',
    ],
)
def test_strict_loader_rejects_duplicate_object_members(
    tmp_path: Path, raw: str
) -> None:
    tool = _load_tool()
    with pytest.raises(ValueError, match="duplicate object member"):
        tool._load_packet(_write(tmp_path, raw))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_loader_rejects_nonstandard_nonfinite_constants(
    tmp_path: Path, constant: str
) -> None:
    tool = _load_tool()
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        tool._load_packet(_write(tmp_path, f'{{"value":{constant}}}'))


@pytest.mark.parametrize("number", ["1e400", "-1e400"])
def test_strict_loader_rejects_float_overflow(
    tmp_path: Path, number: str
) -> None:
    tool = _load_tool()
    with pytest.raises(ValueError, match="JSON number is not finite"):
        tool._load_packet(_write(tmp_path, f'{{"value":{number}}}'))


def test_strict_loader_preserves_valid_finite_json(tmp_path: Path) -> None:
    tool = _load_tool()
    path = _write(
        tmp_path,
        '{"schema_version":1,"nested":{"small":1.25,"large":1e20},"flag":false}',
    )
    payload = tool._load_packet(path)
    assert payload == {
        "schema_version": 1,
        "nested": {"small": 1.25, "large": 1e20},
        "flag": False,
    }


@pytest.mark.parametrize("raw", ["[]", "null", "1", '"packet"'])
def test_strict_loader_requires_object_root(tmp_path: Path, raw: str) -> None:
    tool = _load_tool()
    with pytest.raises(ValueError, match="launch packet root must be an object"):
        tool._load_packet(_write(tmp_path, raw))


def test_main_reports_decode_failure_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    path = _write(tmp_path, '{"outer":{"value":1,"value":2}}')
    assert tool.main(["assess", str(path)]) == 2
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response["error"].startswith("invalid launch packet:")
    assert "duplicate object member" in response["error"]
    assert captured.err == ""


def test_main_reports_excessive_nesting_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    nested = '{"nested":' * 10000 + "0" + "}" * 10000
    path = _write(tmp_path, nested)
    assert tool.main(["assess", str(path)]) == 2
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response["error"].startswith("invalid launch packet:")
    assert captured.err == ""


def test_main_reports_missing_file_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    missing = tmp_path / "missing.json"
    assert tool.main(["assess", str(missing)]) == 2
    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response["error"].startswith("invalid launch packet:")
    assert captured.err == ""


@pytest.mark.parametrize("token", ["1e-9999", "-1e-9999", "0.001e-9999"])
def test_loader_rejects_nonzero_decimal_underflow(tmp_path: Path, token: str) -> None:
    tool = _load_tool()
    with pytest.raises(ValueError, match="nonzero JSON number underflowed to zero"):
        tool._load_packet(_write(tmp_path, '{"maximum_cost_usd":' + token + '}'))


@pytest.mark.parametrize("token", ["0e-9999", "-0.000e-9999", "0.0"])
def test_loader_preserves_lexical_decimal_zero(tmp_path: Path, token: str) -> None:
    tool = _load_tool()
    assert tool._load_packet(_write(tmp_path, '{"value":' + token + '}')) == {
        "value": 0.0
    }


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        pytest.param(b" " * 1_048_577, "input byte limit", id="oversized"),
        pytest.param(
            ('{"nested":' + "[" * 80 + "0" + "]" * 80 + "}").encode("utf-8"),
            "JSON structure limit",
            id="depth",
        ),
        pytest.param(
            ('{"items":[' + ",".join(["0"] * 10_010) + "]}").encode("utf-8"),
            "JSON structure limit",
            id="nodes",
        ),
        pytest.param(br'{"\\ud800":"bad"}', "surrogates not allowed", id="surrogate-key"),
        pytest.param(br'{"value":"\\ud800"}', "surrogates not allowed", id="surrogate-value"),
        pytest.param(b'{"value":"\\xff"}', "decode", id="invalid-utf8"),
    ],
)
def test_loader_bounds_untrusted_json(
    tmp_path: Path, raw: bytes, error: str,
) -> None:
    tool = _load_tool()
    path = tmp_path / "зовнішній пакет із пробілами.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=error):
        tool._load_packet(path)


def test_main_rejects_oversized_file_without_scientific_assessment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    path = tmp_path / "пакет із пробілами.json"
    path.write_bytes(b" " * (tool.MAX_INPUT_BYTES + 1))
    assert tool.main(["assess", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    assert "input byte limit" in result["error"]


def test_loader_preserves_checked_in_readiness_packet() -> None:
    tool = _load_tool()
    path = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
    assert tool._load_packet(path) == json.loads(path.read_text(encoding="utf-8"))
