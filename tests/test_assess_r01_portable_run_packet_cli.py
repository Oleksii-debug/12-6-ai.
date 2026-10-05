from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
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


def test_missing_secret_packet_path_is_redacted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    secret = "PRIVATE-PORTABLE-RUN-PATH-998877"
    missing = tmp_path / f"{secret}.json"
    assert cli.main(["assess_r01_portable_run_packet.py", str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "contract_valid": False,
        "error": "cannot read run packet",
    }
    assert secret not in captured.out


def test_fifo_packet_never_blocks(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("POSIX nonblocking FIFO support required")
    fifo = tmp_path / "portable FIFO із пробілами.pipe"
    os.mkfifo(fifo)
    result = subprocess.run(
        [sys.executable, str(TOOL), str(fifo)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "contract_valid": False,
        "error": "run packet must be a regular file",
    }


def test_loader_detects_regular_file_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    requested = tmp_path / "requested.json"
    substitute = tmp_path / "substitute.json"
    requested.write_text("{}", encoding="utf-8")
    substitute.write_text("{}", encoding="utf-8")
    actual_open = cli.os.open

    def open_substitute(_path: Path, flags: int) -> int:
        return actual_open(substitute, flags)

    monkeypatch.setattr(cli.os, "open", open_substitute)
    with pytest.raises(ValueError, match="changed between check and open"):
        cli._load_packet(requested)


def test_loader_supports_valid_regular_symlink(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows symlink creation may require additional privileges")
    source = tmp_path / "source.json"
    source.write_text('{"provider":"LOCAL_FREE"}', encoding="utf-8")
    linked = tmp_path / "packet link.json"
    linked.symlink_to(source.resolve())
    assert _load_cli()._load_packet(linked) == {"provider": "LOCAL_FREE"}


@pytest.mark.parametrize("negative", [False, True])
def test_loader_bounds_integer_before_conversion(
    tmp_path: Path,
    negative: bool,
) -> None:
    cli = _load_cli()
    literal = ("-" if negative else "") + "9" * 100_000
    path = _write(tmp_path, '{"value":' + literal + "}")
    before = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        with pytest.raises(ValueError, match="JSON integer exceeds 64 digits"):
            cli._load_packet(path)
    finally:
        sys.set_int_max_str_digits(before)


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


@pytest.mark.parametrize("kind", ("array", "object"))
def test_nesting_depth_is_a_controlled_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    cli = _load_cli()
    depth = 10_000
    raw = ("[" * depth + "0" + "]" * depth) if kind == "array" else (
        '{"item":' * depth + "0" + "}" * depth
    )
    path = _write(tmp_path, raw)
    with pytest.raises(ValueError, match="JSON nesting exceeds decoder limit"):
        cli._load_packet(path)
    assert cli.main(["assess_r01_portable_run_packet.py", str(path)]) == 2
    output = capsys.readouterr()
    assert output.err == ""
    report = json.loads(output.out)
    assert report["contract_valid"] is False
    assert report["error"] == "JSON nesting exceeds decoder limit"


def test_assessor_programmer_recursion_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    path = _write(tmp_path, "{}")

    def fail_assessor(_payload: object) -> None:
        raise RecursionError("programmer error inside assessment")

    monkeypatch.setattr(cli, "assess_portable_run_packet", fail_assessor)
    with pytest.raises(RecursionError, match="programmer error inside assessment"):
        cli.main(["assess_r01_portable_run_packet.py", str(path)])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "secret_key",
    [
        "api_token_sk_live_123456",
        "Authorization: Bearer private-value",
        "password=hunter2",
    ],
)
def test_duplicate_member_error_does_not_echo_untrusted_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    secret_key: str,
) -> None:
    cli = _load_cli()
    raw = json.dumps({secret_key: 1})[:-1] + "," + json.dumps(secret_key) + ":2}"
    path = _write(tmp_path, raw)
    assert cli.main(["assess_r01_portable_run_packet.py", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload == {
        "contract_valid": False,
        "error": "duplicate object member",
    }
    assert secret_key not in captured.out


def test_cli_rejects_oversized_packet_before_json_decode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    path = tmp_path / "oversized-packet.json"
    path.write_bytes(b" " * (cli.MAX_INPUT_BYTES + 1))
    assert cli.main(["assess_r01_portable_run_packet.py", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "contract_valid": False,
        "error": "run packet exceeds input byte limit",
    }


@pytest.mark.parametrize("extra", [["second.json"], ["--unknown"], ["x", "y"]])
def test_cli_rejects_extra_arguments_without_reading_packet(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
) -> None:
    cli = _load_cli()
    path = _write(tmp_path, "{}")
    assert cli.main(["assess_r01_portable_run_packet.py", str(path), *extra]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "contract_valid": False,
        "error": "invalid arguments: expected at most one packet path",
    }


def test_default_packet_works_outside_repository_cwd(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(TOOL)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode in {0, 1}
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert "contract_valid" in payload
