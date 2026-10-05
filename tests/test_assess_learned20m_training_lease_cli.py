from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
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


@pytest.mark.parametrize("role", ["manifest", "lease"])
def test_non_object_root_redacts_secret_path_by_role(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    role: str,
) -> None:
    cli = _load_cli()
    secret = "PRIVATE-LEASE-PATH-998877"
    manifest = _write(tmp_path / "manifest.json", "{}")
    lease = _write(tmp_path / "lease.json", "{}")
    target = tmp_path / f"{secret}.json"
    target.write_text("[]", encoding="utf-8")
    if role == "manifest":
        manifest = target
    else:
        lease = target
    argv = ["assess_learned20m_training_lease.py", str(manifest)]
    if role == "lease":
        argv.append(str(lease))
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"] == f"{role} root must be an object"
    assert secret not in captured.out


@pytest.mark.parametrize("role", ["manifest", "lease"])
def test_missing_input_path_is_redacted_by_role(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    role: str,
) -> None:
    cli = _load_cli()
    manifest = _write(tmp_path / "manifest.json", "{}")
    lease = _write(tmp_path / "lease.json", "{}")
    secret = tmp_path / f"PRIVATE-{role.upper()}-SOURCE-998877.json"
    if role == "manifest":
        manifest = secret
    else:
        lease = secret
    argv = ["assess_learned20m_training_lease.py", str(manifest)]
    if role == "lease":
        argv.append(str(lease))
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "contract_valid": False,
        "error": f"cannot read {role}",
    }
    assert "PRIVATE-" not in captured.out


@pytest.mark.parametrize("role", ["manifest", "lease"])
def test_fifo_input_never_blocks(
    tmp_path: Path,
    role: str,
) -> None:
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("POSIX nonblocking FIFO support required")
    manifest = _write(tmp_path / "manifest.json", "{}")
    lease = _write(tmp_path / "lease.json", "{}")
    fifo = tmp_path / f"{role} FIFO із пробілами.pipe"
    os.mkfifo(fifo)
    if role == "manifest":
        manifest = fifo
    else:
        lease = fifo
    tool = Path(__file__).parents[1] / "tools" / "assess_learned20m_training_lease.py"
    argv = [sys.executable, str(tool), str(manifest)]
    if role == "lease":
        argv.append(str(lease))
    result = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout)["error"] == f"{role} must be a regular file"


@pytest.mark.parametrize("role", ["manifest", "lease"])
def test_reader_detects_regular_file_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    cli = _load_cli()
    requested = _write(tmp_path / "requested.json", "{}")
    substitute = _write(tmp_path / "substitute.json", "{}")
    actual_open = cli.os.open

    def open_substitute(_path: Path, flags: int) -> int:
        return actual_open(substitute, flags)

    monkeypatch.setattr(cli.os, "open", open_substitute)
    with pytest.raises(ValueError, match=f"{role} changed between check and open"):
        cli._read_object(requested, label=role)


@pytest.mark.parametrize("negative", [False, True])
def test_reader_bounds_integer_before_conversion(
    tmp_path: Path,
    negative: bool,
) -> None:
    cli = _load_cli()
    literal = ("-" if negative else "") + "9" * 100_000
    path = _write(tmp_path / "huge-int.json", '{"value":' + literal + "}")
    before = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        with pytest.raises(ValueError, match="JSON integer exceeds 64 digits"):
            cli._read_object(path)
    finally:
        sys.set_int_max_str_digits(before)


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
    manifest = _write(tmp_path / "manifest.json", raw)
    assert cli.main(["assess_learned20m_training_lease.py", str(manifest)]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload == {
        "contract_valid": False,
        "error": "duplicate object member",
    }
    assert secret_key not in captured.out


@pytest.mark.parametrize("target", ["manifest", "lease"])
def test_cli_rejects_oversized_json_before_decode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    cli = _load_cli()
    manifest = tmp_path / "manifest.json"
    lease = tmp_path / "lease.json"
    manifest.write_text("{}", encoding="utf-8")
    lease.write_text("{}", encoding="utf-8")
    oversized = b" " * (cli.MAX_INPUT_BYTES + 1)
    (manifest if target == "manifest" else lease).write_bytes(oversized)
    argv = ["assess_learned20m_training_lease.py", str(manifest)]
    if target == "lease":
        argv.append(str(lease))

    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "contract_valid": False,
        "error": "training lease input exceeds byte limit",
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["assess_learned20m_training_lease.py"],
        [
            "assess_learned20m_training_lease.py",
            "manifest",
            "lease",
            "--bad",
            "value",
        ],
        ["assess_learned20m_training_lease.py", "manifest", "lease", "--now"],
    ],
)
def test_invalid_argument_shapes_are_machine_readable(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    cli = _load_cli()
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["contract_valid"] is False
    assert payload["error"].startswith("invalid arguments:")


def test_invalid_now_timestamp_is_redacted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    manifest = _write(tmp_path / "manifest.json", "{}")
    lease = _write(tmp_path / "lease.json", "{}")
    secret = "Bearer-private-timestamp-value"
    assert (
        cli.main(
            [
                "assess_learned20m_training_lease.py",
                str(manifest),
                str(lease),
                "--now",
                secret,
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "contract_valid": False,
        "error": "invalid --now timestamp; expected YYYY-MM-DDTHH:MM:SSZ",
    }
    assert secret not in captured.out
