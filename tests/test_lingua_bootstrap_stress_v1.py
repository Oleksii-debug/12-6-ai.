import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from validate_lingua_bootstrap_stress_v1 import digest, validate

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/research/lingua_bootstrap_stress_v1.json").read_text(
        encoding="utf-8"
    )
)


def evidence():
    value = {
        "schema": "12-6.lingua-bootstrap-stress-evidence.v1",
        "environment": {
            "python": "3.13.5",
            "os": "Linux",
            "machine": "x86_64",
            "cpu_cores": 5,
            "gpu_visible": False,
            "network_pypi": False,
            "network_github": False,
            "lingua_cached": False,
        },
        "install": {
            "isolated_venv": True,
            "global_packages_modified": False,
            "attempts": [
                {"mode": "local_cache", "exit_code": 1, "runtime_executed": False},
                {"mode": "network_pypi", "exit_code": 1, "runtime_executed": False},
            ],
            "runtime_executed": False,
        },
        "runtime": {
            "status": "NOT_EXECUTED",
            "installed_version": None,
            "artifact_sha256": None,
        },
        "benchmark": {"status": "NOT_EXECUTED", "reason": "exact wheel unavailable locally"},
        "parity": {"status": "NOT_EXECUTED", "reason": "runtime unavailable"},
        "adversarial": {
            "checks": [
                "wrong_commit",
                "wrong_wheel_hash",
                "fabricated_runtime",
                "canonical_base_mutation",
                "foreign_weights",
                "evidence_tamper",
            ],
            "negative_checks": "PASS",
        },
        "safety": {
            "canonical_base_changed": False,
            "foreign_weights_used": False,
            "training_executed": False,
        },
        "verdict": "RETEST_RUNTIME_REQUIRED",
        "notes": [
            "Independent CPython 3.13 venv used because Lingua 2.2.0 requires >=3.12,<3.15 while ENV-151 pins 3.11.16."
        ],
    }
    value["identity_sha256"] = digest(value)
    return value


def test_clean_evidence_passes():
    assert validate(CONFIG, evidence()) == []


def test_tampered_identity_fails():
    value = evidence()
    value["notes"].append("tamper")
    assert "evidence_identity" in validate(CONFIG, value)


def test_wrong_upstream_commit_fails():
    config = copy.deepcopy(CONFIG)
    config["upstream"]["commit"] = "0" * 40
    assert "upstream_commit" in validate(config, evidence())


def test_fabricated_runtime_fails_without_benchmark_parity():
    value = evidence()
    value["install"]["runtime_executed"] = True
    value["runtime"]["installed_version"] = "2.2.0"
    value["runtime"]["artifact_sha256"] = CONFIG["upstream"]["target_wheel"]["sha256"]
    value["identity_sha256"] = digest(value)
    assert "benchmark_missing" in validate(CONFIG, value)


def test_adoption_cannot_bypass_runtime():
    value = evidence()
    value["verdict"] = "ADOPTABLE_COMPONENT"
    value["identity_sha256"] = digest(value)
    assert "adoption_gate" in validate(CONFIG, value)
