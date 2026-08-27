"""Fail-closed validator for Liger Kernel bootstrap evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED = {
    "repository": "https://github.com/linkedin/Liger-Kernel",
    "tag": "v0.8.2",
    "commit": "000be60929938fd1358e03524c6ab398b6d421bd",
    "package": "liger_kernel",
    "version": "0.8.2",
    "license": "BSD-2-Clause",
    "license_blob_sha": "d2fcc2b1c4384d0bcd1424b7f83db8e48fa753f6",
    "notice_blob_sha": "ea2881754f5b3e0eb9926dd9dc6c9d772f962911",
    "release_artifact": "liger_kernel-0.8.2.tar.gz",
    "release_artifact_sha256": "387673ed6bf64fc8150cc8315fed578d2fc717ec3450f53489f480880223c1b8",
}
ALLOWED_FINAL = {
    "ADOPTABLE_COMPONENT",
    "EXPERIMENTAL_CANDIDATE",
    "RETEST_RUNTIME_REQUIRED",
    "BLOCKED_ENVIRONMENT",
    "BLOCKED_RIGHTS",
    "BLOCKED_PARITY",
    "REJECTED",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def validate(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    upstream = doc.get("upstream", {})
    for key, expected in EXPECTED.items():
        if upstream.get(key) != expected:
            errors.append(f"upstream.{key}: expected {expected!r}, got {upstream.get(key)!r}")

    rights = doc.get("rights", {})
    for field, expected in (
        ("software_license", "BSD-2-Clause"),
        ("dataset_rights", "NOT_USED"),
        ("model_weight_rights", "NOT_USED"),
    ):
        if rights.get(field) != expected:
            errors.append(f"rights.{field} must be {expected}")

    if doc.get("canonical_base", {}).get("mutated") is not False:
        errors.append("canonical_base.mutated must be false")

    install = doc.get("installation", {})
    execution = doc.get("execution", {})
    benchmark = doc.get("benchmark", {})
    parity = doc.get("parity", {})
    final = doc.get("final_verdict")
    if final not in ALLOWED_FINAL:
        errors.append(f"invalid final_verdict: {final!r}")

    if install.get("status") != "PASS":
        if execution.get("status") == "PASS":
            errors.append("execution cannot PASS when installation is not PASS")
        if benchmark.get("status") == "PASS":
            errors.append("benchmark cannot PASS when installation is not PASS")
        if parity.get("status") == "PASS":
            errors.append("parity cannot PASS when installation is not PASS")
        if final == "ADOPTABLE_COMPONENT":
            errors.append("ADOPTABLE_COMPONENT forbidden without successful installation")

    if benchmark.get("status") == "PASS":
        if not execution.get("gpu_hardware_visible"):
            errors.append("benchmark PASS requires visible GPU hardware")
        if execution.get("mock_used"):
            errors.append("benchmark PASS cannot use a mock")

    required_adversarial = {
        "version_drift_rejected",
        "upstream_commit_drift_rejected",
        "license_drift_rejected",
        "false_runtime_success_rejected",
        "gpu_claim_without_hardware_rejected",
        "environment_contamination_rejected",
    }
    missing = sorted(required_adversarial - set(doc.get("adversarial_tests", {}).get("passed_cases", [])))
    if missing:
        errors.append("missing adversarial cases: " + ", ".join(missing))

    declared_hash = doc.get("evidence_sha256")
    if declared_hash:
        payload = dict(doc)
        payload.pop("evidence_sha256", None)
        actual = canonical_sha256(payload)
        if declared_hash != actual:
            errors.append(f"evidence_sha256 mismatch: {declared_hash} != {actual}")

    return errors


def main(argv: list[str] | None = None) -> int:
    import sys

    path = argv[0] if argv else "evidence/opensource/liger_kernel_bootstrap_stress_v1.json"
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate(doc)
    print(json.dumps({"status": "FAIL" if errors else "PASS", "errors": errors, "final_verdict": doc.get("final_verdict")}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
