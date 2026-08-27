from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "audit" / "datatrove_bootstrap_stress_v1.json"
VALIDATOR = ROOT / "tools" / "validate_datatrove_bootstrap_stress_v1.py"


def load_validator_globals():
    return runpy.run_path(str(VALIDATOR), run_name="datatrove_validator_test")


def load_evidence():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_validator_accepts_terminal_mechanics_blocker_record():
    ns = load_validator_globals()
    assert ns["main"]() == 0


def test_exact_pin_is_not_mutable_latest():
    evidence = load_evidence()
    assert evidence["upstream"]["release"] == "v0.10.0"
    assert evidence["upstream"]["tag_commit"] == "7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664"
    assert evidence["install_attempt"]["requested_distribution"] == "datatrove==0.10.0"


def test_runtime_success_cannot_be_claimed_from_blocked_install():
    ns = load_validator_globals()
    evidence = load_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["runtime"]["real_datatrove_import_executed"] = True
    mutated["install_attempt"]["result"] = "EXECUTED_PASS"
    assert "install_blocked" in ns["validate"](mutated)
    assert "runtime_not_executed" in ns["validate"](mutated)


def test_upstream_identity_drift_fails_closed():
    ns = load_validator_globals()
    mutated = load_evidence()
    mutated["upstream"]["tag_commit"] = "0" * 40
    assert "tag_commit" in ns["validate"](mutated)


def test_canonical_base_boundary_is_explicitly_clean():
    evidence = load_evidence()
    for key, value in evidence["canonical_base_boundary"].items():
        assert value is False, key
