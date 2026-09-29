from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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
