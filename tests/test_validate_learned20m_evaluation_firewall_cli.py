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
TOOL = ROOT / "tools" / "validate_learned20m_evaluation_firewall.py"
POLICY = ROOT / "configs" / "evaluation" / "learned20m_evaluation_firewall_v1.json"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_learned20m_evaluation_firewall_cli",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cli(policy: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), "--policy", str(policy)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_strict_policy_loader_rejects_ambiguous_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    cases = (
        (
            "duplicate.json",
            '{"evaluation_protocol":{"phase":"a","phase":"b"}}',
            "duplicate object member",
            ValueError,
        ),
        ("nan.json", '{"value":NaN}', "non-finite JSON constant", ValueError),
        ("infinity.json", '{"value":Infinity}', "non-finite JSON constant", ValueError),
        (
            "negative-infinity.json",
            '{"value":-Infinity}',
            "non-finite JSON constant",
            ValueError,
        ),
        ("overflow.json", '{"value":1e400}', "JSON number is not finite", ValueError),
        (
            "negative-overflow.json",
            '{"value":-1e400}',
            "JSON number is not finite",
            ValueError,
        ),
        (
            "nonobject.json",
            "[]",
            "evaluation firewall policy root must be an object",
            TypeError,
        ),
    )
    for name, raw, expected, exception_type in cases:
        path = tmp_path / name
        path.write_text(raw, encoding="utf-8")
        try:
            cli._load_policy(path)
        except exception_type as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{name} unexpectedly passed strict JSON loading")


def test_strict_policy_loader_preserves_valid_finite_json(tmp_path: Path) -> None:
    cli = _load_cli()
    path = tmp_path / "finite.json"
    path.write_text(
        '{"threshold":2.5,"budget":1e6,"nested":{"enabled":false}}',
        encoding="utf-8",
    )
    assert cli._load_policy(path) == {
        "threshold": 2.5,
        "budget": 1e6,
        "nested": {"enabled": False},
    }


def test_cli_rejects_malformed_policy_without_traceback(tmp_path: Path) -> None:
    cases = (
        ('{"root":{"key":"a","key":"b"}}', "duplicate object member"),
        ('{"value":NaN}', "non-finite JSON constant"),
        ('{"value":1e400}', "JSON number is not finite"),
    )
    for index, (raw, expected) in enumerate(cases):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(raw, encoding="utf-8")
        result = _run_cli(path)

        assert result.returncode == 2
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert payload["status"] == "FAIL"
        assert expected in payload["error"]


def test_checked_in_policy_cli_still_succeeds() -> None:
    result = _run_cli(POLICY)
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"


@pytest.mark.parametrize(
    ("variant", "expected_error"),
    [
        ("empty", "policy"),
        ("extra_field", "policy"),
        ("selection_bypass", "selection boundary weakened"),
        ("final_test_bypass", "final-test boundary weakened"),
    ],
)
def test_cli_rejects_semantically_invalid_json_without_traceback(
    tmp_path: Path,
    variant: str,
    expected_error: str,
) -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if variant == "empty":
        policy = {}
    elif variant == "extra_field":
        policy["unexpected_authority"] = {"training_authorized": True}
    elif variant == "selection_bypass":
        policy["selection_validation"]["may_report_final_test"] = True
    else:
        policy["final_test_reservation"]["outcomes_access_before_selection_lock"] = True

    path = tmp_path / f"{variant}.json"
    path.write_text(
        json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    result = _run_cli(path)

    assert result.returncode == 2
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "FAIL"
    assert expected_error in payload["error"]


def test_cli_unexpected_programming_error_is_not_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()

    def unexpected(_policy: dict) -> dict:
        raise RuntimeError("unexpected programmer defect")

    monkeypatch.setattr(cli, "validate_policy", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(TOOL), "--policy", str(POLICY)],
    )
    with pytest.raises(RuntimeError, match="unexpected programmer defect"):
        cli.main()


@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_cli_rejects_excessive_json_nesting_without_traceback(
    tmp_path: Path,
    nesting: str,
) -> None:
    if nesting == "arrays":
        raw = '{"root":' + "[" * 10000 + "0" + "]" * 10000 + "}"
    else:
        raw = '{"root":' + '{"k":' * 10000 + "0" + "}" * 10000 + "}"
    path = tmp_path / f"deep-{nesting}.json"
    path.write_text(raw, encoding="utf-8")

    cli = _load_cli()
    with pytest.raises(ValueError, match="JSON nesting limit exceeded"):
        cli._load_policy(path)

    result = _run_cli(path)
    assert result.returncode == 2
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "FAIL"
    assert payload["error"] == "evaluation firewall policy JSON nesting limit exceeded"


def test_unexpected_product_recursion_remains_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()

    def unexpected(_policy: dict) -> dict:
        raise RecursionError("unexpected programmer recursion")

    monkeypatch.setattr(cli, "validate_policy", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(TOOL), "--policy", str(POLICY)],
    )
    with pytest.raises(RecursionError, match="unexpected programmer recursion"):
        cli.main()


@pytest.mark.parametrize(
    "secret_key",
    [
        "api_token_sk_live_123456",
        "Authorization: Bearer private-value",
        "password=hunter2",
    ],
)
def test_cli_duplicate_member_refusal_does_not_echo_untrusted_key(
    tmp_path: Path,
    secret_key: str,
) -> None:
    raw = json.dumps({secret_key: 1})[:-1] + "," + json.dumps(secret_key) + ":2}"
    path = tmp_path / "duplicate-secret.json"
    path.write_text(raw, encoding="utf-8")
    result = _run_cli(path)

    assert result.returncode == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert payload["error"] == "duplicate object member"
    assert secret_key not in result.stdout


def test_policy_loader_rejects_oversized_input_before_json_decode(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    path = tmp_path / "oversized-policy.json"
    path.write_bytes(b" " * (cli.MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="input byte limit"):
        cli._load_policy(path)

    result = _run_cli(path)
    assert result.returncode == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload == {
        "error": "evaluation firewall policy exceeds input byte limit",
        "status": "FAIL",
    }


def test_policy_loader_missing_path_redacts_secret(
    tmp_path: Path,
) -> None:
    secret = "PRIVATE-FINAL-TEST-PATH-998877"
    result = _run_cli(tmp_path / f"{secret}.json")

    assert result.returncode == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload == {
        "error": "cannot read evaluation firewall policy",
        "status": "FAIL",
    }
    assert secret not in result.stdout


def test_policy_loader_rejects_directory_without_path_echo(tmp_path: Path) -> None:
    result = _run_cli(tmp_path)

    assert result.returncode == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    expected = (
        "cannot read evaluation firewall policy"
        if os.name == "nt"
        else "evaluation firewall policy must be a regular file"
    )
    assert payload == {"error": expected, "status": "FAIL"}


def test_policy_loader_fifo_never_blocks(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("POSIX nonblocking FIFO support required")
    fifo = tmp_path / "evaluation FIFO із пробілами.pipe"
    os.mkfifo(fifo)
    result = subprocess.run(
        [sys.executable, str(TOOL), "--policy", str(fifo)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "error": "evaluation firewall policy must be a regular file",
        "status": "FAIL",
    }


def test_policy_loader_rejects_regular_file_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    requested = tmp_path / "requested.json"
    substitute = tmp_path / "substitute.json"
    requested.write_bytes(POLICY.read_bytes())
    substitute.write_bytes(POLICY.read_bytes())
    real_open = cli.os.open

    def open_substitute(_path: Path, flags: int) -> int:
        return real_open(substitute, flags)

    monkeypatch.setattr(cli.os, "open", open_substitute)
    with pytest.raises(ValueError, match="changed between check and open"):
        cli._load_policy(requested)


def test_policy_loader_valid_regular_symlink_still_works(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows symlink creation may require additional privileges")
    linked = tmp_path / "evaluation policy link.json"
    linked.symlink_to(POLICY.resolve())
    cli = _load_cli()
    assert cli._load_policy(linked) == json.loads(POLICY.read_text(encoding="utf-8"))


def test_default_policy_works_outside_repository_cwd(tmp_path: Path) -> None:
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
    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"


def test_relative_policy_still_respects_explicit_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "alternate-root"
    relative = Path("nested") / "policy.json"
    path = repo_root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(POLICY.read_bytes())
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo-root",
            str(repo_root),
            "--policy",
            str(relative),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "PASS"
