from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_open_source_bootstrap_stress_v1 as v

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "configs/research/open_source_bootstrap_stress_v1.json").read_text())
EVIDENCE = json.loads((ROOT / "evidence/research/open_source_bootstrap_stress_v1.json").read_text())


def test_validator_passes_canonical_evidence() -> None:
    assert v.main() == 0


def test_contract_rejects_main_sha_drift() -> None:
    mutated = copy.deepcopy(CONTRACT)
    mutated["project_authority"]["main_sha"] = "0" * 40
    with pytest.raises(AssertionError):
        v.validate_contract(mutated)


def test_contract_rejects_cpu_lock_drift() -> None:
    mutated = copy.deepcopy(CONTRACT)
    mutated["env151"]["files"]["requirements/execution/linux-x86_64/cpu-runtime.lock.txt"]["content_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        v.validate_contract(mutated)


def test_evidence_rejects_false_runtime_success() -> None:
    mutated = copy.deepcopy(EVIDENCE)
    mutated["execution"]["real_bootstrap_executed"] = True
    with pytest.raises(AssertionError):
        v.validate_evidence(mutated)


def test_evidence_rejects_global_python_mutation_claim() -> None:
    mutated = copy.deepcopy(EVIDENCE)
    mutated["installation_attempt"]["global_python_modified"] = True
    with pytest.raises(AssertionError):
        v.validate_evidence(mutated)


def test_evidence_identity_changes_on_material_drift() -> None:
    original = copy.deepcopy(EVIDENCE)
    identity = original.pop("identity_sha256")
    assert identity == v.digest(original)
    original["environment"]["installed_exactness"]["torch"] = "2.13.0+cpu"
    assert identity != v.digest(original)


def test_validator_cli_is_deterministic() -> None:
    cmd = [sys.executable, str(ROOT / "tools/validate_open_source_bootstrap_stress_v1.py")]
    first = subprocess.check_output(cmd, text=True).strip()
    second = subprocess.check_output(cmd, text=True).strip()
    assert first == second


def test_real_runtime_is_not_falsely_promoted() -> None:
    assert EVIDENCE["verdict"] == "RETEST_RUNTIME_REQUIRED"
    assert EVIDENCE["execution"]["real_bootstrap_executed"] is False
