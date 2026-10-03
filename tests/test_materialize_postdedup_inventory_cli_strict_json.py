from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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
