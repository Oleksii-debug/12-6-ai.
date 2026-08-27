from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "configs" / "research" / "lingua_bootstrap_stress_v1.json"
VALIDATOR = ROOT / "tools" / "validate_lingua_bootstrap_stress.py"
PROBE = ROOT / "tools" / "probe_lingua_runtime.py"


def test_manifest_is_fail_closed():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["status"] == "RETEST_RUNTIME_REQUIRED"
    assert data["canonical_base_impact"]["canonical_base_modified"] is False
    assert data["runtime"]["real_import_executed"] is False


def test_validator_accepts_sealed_manifest():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(MANIFEST)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_validator_rejects_fabricated_adoptable_status(tmp_path):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["status"] = "ADOPTABLE_COMPONENT"
    out = tmp_path / "mutated.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_runtime_probe_fails_closed_without_exact_package(tmp_path):
    output = tmp_path / "runtime.json"
    result = subprocess.run(
        [sys.executable, str(PROBE), "--out", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in {2, 3}
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["executed"] is False
    assert data["status"] == "NOT_EXECUTED"
