from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "12-6.open-source-bootstrap-stress.v1"
RESULT_SCHEMA = "12-6.open-source-bootstrap-stress-evidence.v1"
EXPECTED_MAIN_SHA = "5020afd671a3885c1b738c8b4eafe7525f630546"
EXPECTED_ENV151_HEAD = "bbca2101ea9409b47d844dd8292cd7f2290e3ff0"
EXPECTED_CPU_LOCK_SHA256 = "03e08dd06ff446651dcc6950d0f433325bb32261d3e2406b34506cd00e1be52a"
EXPECTED_CPU_LOCK_GIT_BLOB = "ca52939b8c4cdd1c06189ff861e0ddf056de83a7"
EXPECTED_BOOTSTRAP_BLOB = "86881ea145cb5faa4545cc7156fbb660ae9a33f0"
EXPECTED_CAPABILITIES_BLOB = "cd6fc00cb23ddb19b58fecab5c32a1a68224b144"
EXPECTED_PYTHON = "3.11.16"
EXPECTED_TORCH = "2.13.0+cpu"
EXPECTED_TORCH_WHEEL_SHA256 = "6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(contract: dict[str, Any]) -> None:
    assert contract["schema"] == CONTRACT_SCHEMA
    assert contract["canonical_base"]["random_init_only"] is True
    assert contract["canonical_base"]["foreign_pretrained_weights_used"] is False
    assert contract["ownership"]["worker_issue"] == 773
    assert contract["ownership"]["lane_key"] == "D10|EXECUTION-BOOTSTRAP|INTEGRATION-CONVERGENCE|STRESS-V1"
    assert contract["project_authority"]["main_sha"] == EXPECTED_MAIN_SHA
    assert contract["env151"]["head_sha"] == EXPECTED_ENV151_HEAD
    assert contract["env151"]["files"]["tools/execution_bootstrap.py"]["git_blob_sha"] == EXPECTED_BOOTSTRAP_BLOB
    assert contract["env151"]["files"]["requirements/execution/capabilities.json"]["git_blob_sha"] == EXPECTED_CAPABILITIES_BLOB
    lock = contract["env151"]["files"]["requirements/execution/linux-x86_64/cpu-runtime.lock.txt"]
    assert lock["content_sha256"] == EXPECTED_CPU_LOCK_SHA256
    assert lock["git_blob_sha"] == EXPECTED_CPU_LOCK_GIT_BLOB
    assert lock["artifact"] == "torch==2.13.0+cpu"
    assert lock["artifact_sha256"] == EXPECTED_TORCH_WHEEL_SHA256
    assert contract["env151"]["required_python"] == EXPECTED_PYTHON
    assert contract["env151"]["forbidden_cpu_packages"] == ["nvidia-*", "cuda-*", "triton"]


def validate_evidence(evidence: dict[str, Any]) -> None:
    assert evidence["schema"] == RESULT_SCHEMA
    assert evidence["verdict"] == "RETEST_RUNTIME_REQUIRED"
    assert evidence["execution"]["real_bootstrap_executed"] is False
    assert evidence["execution"]["reason"] == "exact_dependency_unavailable_and_required_python_mismatch"
    assert evidence["installation_attempt"]["attempted"] is True
    assert evidence["installation_attempt"]["global_python_modified"] is False
    assert evidence["installation_attempt"]["exact_requirement"] == f"torch=={EXPECTED_TORCH}"
    assert evidence["installation_attempt"]["exact_requirement_sha256"] == EXPECTED_TORCH_WHEEL_SHA256
    assert evidence["environment"]["python"] == "3.13.5"
    assert evidence["environment"]["gpu"]["hardware_visible"] is False
    assert evidence["environment"]["package_managers"]["pip"] == "25.1.1"
    assert evidence["environment"]["package_managers"]["uv"] == "0.10.0"
    assert evidence["environment"]["package_managers"]["poetry"] is None
    assert evidence["environment"]["network"]["github_dns"] == "UNAVAILABLE"
    assert evidence["environment"]["network"]["pypi_dns"] == "UNAVAILABLE"
    assert evidence["environment"]["installed_exactness"]["torch"] == "2.10.0+cpu"
    assert evidence["environment"]["installed_exactness"]["pytest"] == "9.0.2"
    assert evidence["environment"]["installed_exactness"]["numpy"] == "2.3.5"
    assert evidence["environment"]["installed_exactness"]["tokenizers"] is None
    assert evidence["environment"]["installed_exactness"]["transformers"] is None
    assert evidence["environment"]["installed_exactness"]["ruff"] is None
    assert evidence["environment"]["installed_exactness"]["datatrove"] is None
    assert evidence["environment"]["cache"]["matching_exact_artifacts_found"] == []
    assert evidence["rights"]["project_bootstrap_license_status"] == "NOT_ASSERTED_NO_ROOT_LICENSE_FILE_FOUND"
    assert evidence["rights"]["third_party_training_authority"] == "NOT_ASSERTED"
    assert evidence["base_safety"]["canonical_base_changed"] is False
    assert evidence["base_safety"]["foreign_weights_used"] is False
    assert evidence["base_safety"]["tokenizer_replaced"] is False
    assert evidence["base_safety"]["training_executed"] is False
    assert evidence["base_safety"]["paid_compute_used"] is False
    payload = {k: v for k, v in evidence.items() if k != "identity_sha256"}
    assert evidence["identity_sha256"] == digest(payload)
    assert all(math.isfinite(float(x)) for x in evidence["benchmark"]["mechanics_only_ms"])


def validate_lock_text(lock_text: str) -> None:
    actual = hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert actual == EXPECTED_CPU_LOCK_SHA256
    lines = [x.strip() for x in lock_text.splitlines() if x.strip() and not x.lstrip().startswith("#")]
    pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==([^\s;@/\\]+) --hash=sha256:[0-9a-f]{64}$")
    assert len(lines) == 12
    for line in lines:
        assert pattern.fullmatch(line), line
    names = [line.split("==", 1)[0].lower() for line in lines]
    assert len(names) == len(set(names))
    assert any(line.startswith("torch==2.13.0+cpu ") for line in lines)
    assert not any(re.match(r"(?:nvidia-|cuda-|triton)", name) for name in names)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    contract = load(root / "configs/research/open_source_bootstrap_stress_v1.json")
    evidence = load(root / "evidence/research/open_source_bootstrap_stress_v1.json")
    validate_contract(contract)
    validate_evidence(evidence)
    print(json.dumps({"status": "PASS", "contract": digest(contract), "evidence": evidence["identity_sha256"], "python": platform.python_version(), "implementation": sys.implementation.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
