import copy

from tools.validate_liger_bootstrap_stress import EXPECTED, validate

BASE = {
    "upstream": EXPECTED.copy(),
    "rights": {
        "software_license": "BSD-2-Clause",
        "dataset_rights": "NOT_USED",
        "model_weight_rights": "NOT_USED",
    },
    "canonical_base": {"mutated": False},
    "installation": {"status": "NOT_EXECUTED"},
    "execution": {"status": "NOT_EXECUTED", "gpu_hardware_visible": False, "mock_used": False},
    "benchmark": {"status": "NOT_EXECUTED"},
    "parity": {"status": "NOT_EXECUTED"},
    "adversarial_tests": {
        "passed_cases": [
            "version_drift_rejected",
            "upstream_commit_drift_rejected",
            "license_drift_rejected",
            "false_runtime_success_rejected",
            "gpu_claim_without_hardware_rejected",
            "environment_contamination_rejected",
        ]
    },
    "final_verdict": "RETEST_RUNTIME_REQUIRED",
}


def test_clean_blocked_evidence_validates():
    assert validate(BASE) == []


def test_version_drift_blocks():
    doc = copy.deepcopy(BASE)
    doc["upstream"]["version"] = "0.8.1"
    assert any("upstream.version" in item for item in validate(doc))


def test_commit_drift_blocks():
    doc = copy.deepcopy(BASE)
    doc["upstream"]["commit"] = "deadbeef"
    assert any("upstream.commit" in item for item in validate(doc))


def test_license_drift_blocks():
    doc = copy.deepcopy(BASE)
    doc["rights"]["software_license"] = "MIT"
    assert any("software_license" in item for item in validate(doc))


def test_false_runtime_success_blocks():
    doc = copy.deepcopy(BASE)
    doc["execution"]["status"] = "PASS"
    assert any("execution cannot PASS" in item for item in validate(doc))


def test_gpu_benchmark_without_gpu_blocks():
    doc = copy.deepcopy(BASE)
    doc["installation"]["status"] = "PASS"
    doc["benchmark"]["status"] = "PASS"
    assert any("visible GPU" in item for item in validate(doc))


def test_mock_benchmark_blocks():
    doc = copy.deepcopy(BASE)
    doc["installation"]["status"] = "PASS"
    doc["execution"]["gpu_hardware_visible"] = True
    doc["execution"]["mock_used"] = True
    doc["benchmark"]["status"] = "PASS"
    assert any("mock" in item for item in validate(doc))


def test_adoptable_without_install_blocks():
    doc = copy.deepcopy(BASE)
    doc["final_verdict"] = "ADOPTABLE_COMPONENT"
    assert any("ADOPTABLE_COMPONENT" in item for item in validate(doc))
