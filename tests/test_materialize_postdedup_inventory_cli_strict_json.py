from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "materialize_postdedup_inventory_v1.py"
ZERO_SHA256 = "0" * 64


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "materialize_postdedup_inventory_v1_cli",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cli(
    *,
    action: str,
    v8_report: Path,
    survivor_authority: Path,
    inventory: Path,
    binding_evidence: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            action,
            "--v8-report",
            str(v8_report),
            "--expected-v8-report-sha256",
            ZERO_SHA256,
            "--survivor-authority",
            str(survivor_authority),
            "--expected-survivor-authority-sha256",
            ZERO_SHA256,
            "--inventory",
            str(inventory),
            "--binding-evidence",
            str(binding_evidence),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_strict_loader_rejects_ambiguous_and_nonfinite_json(tmp_path: Path) -> None:
    cli = _load_cli()
    cases = (
        (
            "duplicate-root.json",
            '{"status":"PASS","status":"BLOCKED"}',
            "duplicate object member",
            ValueError,
        ),
        (
            "duplicate-nested.json",
            '{"outer":{"value":1,"value":2}}',
            "duplicate object member",
            ValueError,
        ),
        ("nan.json", '{"value":NaN}', "non-finite JSON constant", ValueError),
        (
            "infinity.json",
            '{"value":Infinity}',
            "non-finite JSON constant",
            ValueError,
        ),
        (
            "negative-infinity.json",
            '{"value":-Infinity}',
            "non-finite JSON constant",
            ValueError,
        ),
        (
            "overflow.json",
            '{"value":1e400}',
            "JSON number is not finite",
            ValueError,
        ),
        (
            "negative-overflow.json",
            '{"value":-1e400}',
            "JSON number is not finite",
            ValueError,
        ),
        (
            "nonobject.json",
            "[]",
            "must contain a JSON object",
            TypeError,
        ),
    )
    for name, raw, expected, exception_type in cases:
        path = tmp_path / name
        path.write_text(raw, encoding="utf-8")
        try:
            cli._load(path)
        except exception_type as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{name} unexpectedly passed strict JSON loading")


def test_strict_loader_preserves_valid_finite_json(tmp_path: Path) -> None:
    cli = _load_cli()
    path = tmp_path / "finite.json"
    path.write_text(
        '{"threshold":2.5,"budget":1e6,"nested":{"enabled":false}}',
        encoding="utf-8",
    )

    assert cli._load(path) == {
        "threshold": 2.5,
        "budget": 1e6,
        "nested": {"enabled": False},
    }


def test_materialize_cli_rejects_bad_report_without_output_or_traceback(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"status":"PASS","status":"BLOCKED"}', encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    binding = tmp_path / "binding.json"

    result = _run_cli(
        action="materialize",
        v8_report=report,
        survivor_authority=tmp_path / "unused-survivor.json",
        inventory=inventory,
        binding_evidence=binding,
    )

    assert result.returncode == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert "duplicate object member" in payload["error"]
    assert not inventory.exists()
    assert not binding.exists()


def test_materialize_cli_rejects_bad_survivor_without_output_or_traceback(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    survivor = tmp_path / "survivor.json"
    survivor.write_text('{"value":NaN}', encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    binding = tmp_path / "binding.json"

    result = _run_cli(
        action="materialize",
        v8_report=report,
        survivor_authority=survivor,
        inventory=inventory,
        binding_evidence=binding,
    )

    assert result.returncode == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert "non-finite JSON constant" in payload["error"]
    assert not inventory.exists()
    assert not binding.exists()


def test_verify_cli_rejects_bad_inventory_before_product_semantics(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    survivor = tmp_path / "survivor.json"
    inventory = tmp_path / "inventory.json"
    binding = tmp_path / "binding.json"
    report.write_text("{}", encoding="utf-8")
    survivor.write_text("{}", encoding="utf-8")
    inventory.write_text('{"value":1e400}', encoding="utf-8")
    binding.write_text("{}", encoding="utf-8")

    result = _run_cli(
        action="verify",
        v8_report=report,
        survivor_authority=survivor,
        inventory=inventory,
        binding_evidence=binding,
    )

    assert result.returncode == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert "JSON number is not finite" in payload["error"]


def test_verify_cli_rejects_bad_binding_before_product_semantics(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    survivor = tmp_path / "survivor.json"
    inventory = tmp_path / "inventory.json"
    binding = tmp_path / "binding.json"
    report.write_text("{}", encoding="utf-8")
    survivor.write_text("{}", encoding="utf-8")
    inventory.write_text("{}", encoding="utf-8")
    binding.write_text('{"value":Infinity}', encoding="utf-8")

    result = _run_cli(
        action="verify",
        v8_report=report,
        survivor_authority=survivor,
        inventory=inventory,
        binding_evidence=binding,
    )

    assert result.returncode == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert "non-finite JSON constant" in payload["error"]


@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_strict_loader_rejects_excessive_json_nesting(
    tmp_path: Path,
    nesting: str,
) -> None:
    cli = _load_cli()
    if nesting == "arrays":
        raw = '{"root":' + "[" * 10000 + "0" + "]" * 10000 + "}"
    else:
        raw = '{"root":' + '{"k":' * 10000 + "0" + "}" * 10000 + "}"
    path = tmp_path / f"deep-{nesting}.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="JSON nesting limit exceeded"):
        cli._load(path)


@pytest.mark.parametrize(
    ("action", "target"),
    (
        ("materialize", "v8_report"),
        ("materialize", "survivor_authority"),
        ("verify", "v8_report"),
        ("verify", "survivor_authority"),
        ("verify", "inventory"),
        ("verify", "binding_evidence"),
    ),
)
@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_cli_rejects_deep_json_in_each_external_authority_input(
    tmp_path: Path,
    action: str,
    target: str,
    nesting: str,
) -> None:
    paths = {
        "v8_report": tmp_path / "report.json",
        "survivor_authority": tmp_path / "survivor.json",
        "inventory": tmp_path / "inventory.json",
        "binding_evidence": tmp_path / "binding.json",
    }
    paths["v8_report"].write_text("{}", encoding="utf-8")
    paths["survivor_authority"].write_text("{}", encoding="utf-8")
    if action == "verify":
        paths["inventory"].write_text("{}", encoding="utf-8")
        paths["binding_evidence"].write_text("{}", encoding="utf-8")
    if nesting == "arrays":
        raw = '{"root":' + "[" * 10000 + "0" + "]" * 10000 + "}"
    else:
        raw = '{"root":' + '{"k":' * 10000 + "0" + "}" * 10000 + "}"
    paths[target].write_text(raw, encoding="utf-8")
    existing = {p: p.read_bytes() for p in paths.values() if p.exists()}

    result = _run_cli(action=action, **paths)

    assert result.returncode == 2
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "FAIL"
    assert "JSON nesting limit exceeded" in payload["error"]
    for path, original in existing.items():
        assert path.read_bytes() == original
    if action == "materialize":
        assert not paths["inventory"].exists()
        assert not paths["binding_evidence"].exists()


def test_unexpected_materialize_recursion_is_not_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    report = tmp_path / "report.json"
    survivor = tmp_path / "survivor.json"
    report.write_text("{}", encoding="utf-8")
    survivor.write_text("{}", encoding="utf-8")

    def unexpected(*_args: object, **_kwargs: object) -> dict:
        raise RecursionError("unexpected Product recursion")

    monkeypatch.setattr(cli, "materialize_postdedup_inventory", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "materialize",
            "--v8-report", str(report),
            "--expected-v8-report-sha256", ZERO_SHA256,
            "--survivor-authority", str(survivor),
            "--expected-survivor-authority-sha256", ZERO_SHA256,
            "--inventory", str(tmp_path / "inventory.json"),
            "--binding-evidence", str(tmp_path / "binding.json"),
        ],
    )
    with pytest.raises(RecursionError, match="unexpected Product recursion"):
        cli.main()
