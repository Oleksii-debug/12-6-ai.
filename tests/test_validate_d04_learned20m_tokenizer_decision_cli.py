from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "validate_d04_learned20m_tokenizer_decision.py"
HASH_ARGS = (
    "--expected-selection-identity-sha256",
    "1" * 64,
    "--expected-application-identity-sha256",
    "2" * 64,
    "--expected-retained-inventory-identity-sha256",
    "3" * 64,
    "--expected-decontamination-authority-sha256",
    "4" * 64,
    "--expected-dedup-authority-sha256",
    "5" * 64,
    "--expected-balance-policy-identity-sha256",
    "6" * 64,
    "--expected-balance-result-identity-sha256",
    "7" * 64,
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_d04_learned20m_tokenizer_decision_cli",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_accepts_finite_json_object(tmp_path: Path) -> None:
    path = tmp_path / "finite.json"
    path.write_text(
        '{"count":2,"ratio":1.25,"nested":{"value":-3.5}}',
        encoding="utf-8",
    )

    assert _module()._load(path) == {
        "count": 2,
        "ratio": 1.25,
        "nested": {"value": -3.5},
    }


def test_load_rejects_recursive_duplicate_members(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"outer":{"same":1,"same":2}}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate_json_key:same"):
        _module()._load(path)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_nonfinite_constants(tmp_path: Path, value: str) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value":{value}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non_finite_json_constant"):
        _module()._load(path)


@pytest.mark.parametrize("value", ["1e400", "-1e400"])
def test_load_rejects_float_overflow(tmp_path: Path, value: str) -> None:
    path = tmp_path / "overflow.json"
    path.write_text(f'{{"value":{value}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non_finite_json_number"):
        _module()._load(path)


def test_load_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1,2,3]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain one JSON object"):
        _module()._load(path)


@pytest.mark.parametrize("bad_target", ["selection", "application", "report"])
def test_cli_malformed_authority_fails_machine_readably_without_output(
    tmp_path: Path,
    bad_target: str,
) -> None:
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    report = tmp_path / "report.json"
    output = tmp_path / "out.json"
    selection.write_text("{}", encoding="utf-8")
    application.write_text("{}", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")

    malformed = '{"outer":{"same":1,"same":2}}'
    bad_path = {
        "selection": selection,
        "application": application,
        "report": report,
    }[bad_target]
    bad_path.write_text(malformed, encoding="utf-8")

    command = [
        sys.executable,
        str(TOOL),
        "--balanced-selection",
        str(selection),
        "--split-application",
        str(application),
        *HASH_ARGS,
        "--output",
        str(output),
    ]
    if bad_target == "report":
        command.extend(["--verify-report", str(report)])

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["contract_valid"] is False
    assert "duplicate_json_key:same" in payload["error"]
    assert not output.exists()
