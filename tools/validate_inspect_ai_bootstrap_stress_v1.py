#!/usr/bin/env python3
"""Fail-closed validator for the Inspect AI bootstrap-stress evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

EXPECTED = {
    "qualification_id": "INSPECT-AI-BOOTSTRAP-STRESS-V1",
    "release": "0.3.260",
    "tag": "0.3.260",
    "commit": "3f294e61b823d6bad5fc16706fc5825ea980c8ee",
    "license": "MIT",
    "base_sha": "5020afd671a3885c1b738c8b4eafe7525f630546",
    "wheel_sha256": "3da1dd4e4cbaec248b507799beb71eb9917eee1062eab1d9aeb6a8b5a03a386a",
}


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("qualification_id") != EXPECTED["qualification_id"]:
        errors.append("qualification_id_mismatch")
    project = data.get("project", {})
    upstream = data.get("upstream", {})
    for key in ("release", "tag", "commit", "license"):
        if upstream.get(key) != EXPECTED[key]:
            errors.append(f"upstream_{key}_mismatch")
    if project.get("base_sha") != EXPECTED["base_sha"]:
        errors.append("base_sha_mismatch")
    if project.get("canonical_base_contamination") is not False:
        errors.append("canonical_base_contamination")
    if project.get("foreign_weights_used") is not False:
        errors.append("foreign_weights_used")
    if project.get("tokenizer_changed") is not False:
        errors.append("tokenizer_changed")
    if upstream.get("pypi_wheel_sha256") != EXPECTED["wheel_sha256"]:
        errors.append("pypi_wheel_sha256_mismatch")
    if data.get("status") == "ADOPTABLE_COMPONENT":
        checks = [
            data.get("installation_attempt", {}).get("result") == "SUCCESS",
            data.get("runtime", {}).get("execution") == "EXECUTED",
            data.get("benchmark", {}).get("execution") == "EXECUTED",
            data.get("parity", {}).get("execution") == "EXECUTED",
        ]
        if not all(checks):
            errors.append("adoptable_without_real_runtime_evidence")
    if data.get("status") == "RETEST_RUNTIME_REQUIRED" and data.get("runtime", {}).get("execution") != "NOT_EXECUTED":
        errors.append("retest_runtime_state_inconsistent")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/research/inspect_ai_bootstrap_stress_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"validation": "PASS" if not errors else "FAIL", "errors": errors, "manifest_sha256": digest}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
