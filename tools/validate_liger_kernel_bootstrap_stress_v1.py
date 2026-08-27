from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "research" / "liger_kernel_bootstrap_stress_v1.json"
EVIDENCE = REPO_ROOT / "evidence" / "research" / "liger_kernel_bootstrap_stress_v1.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    upstream = manifest.get("upstream", {})
    project = manifest.get("project", {})
    rights = manifest.get("rights", {})
    qualification = manifest.get("qualification", {})
    forbidden = manifest.get("forbidden", {})

    exact = {
        "repository": "https://github.com/linkedin/Liger-Kernel",
        "tag": "v0.8.2",
        "commit": "000be60929938fd1358e03524c6ab398b6d421bd",
        "license": "BSD-2-Clause",
        "license_blob": "d2fcc2b1c4384d0bcd1424b7f83db8e48fa753f6",
        "notice_blob": "ea2881754f5b3e0eb9926dd9dc6c9d772f962911",
        "release_artifact_sha256": "387673ed6bf64fc8150cc8315fed578d2fc717ec3450f53489f480880223c1b8",
    }
    for key, expected in exact.items():
        if upstream.get(key) != expected:
            errors.append(f"upstream {key} drift")
    if project.get("base_sha") != "5020afd671a3885c1b738c8b4eafe7525f630546":
        errors.append("project base SHA drift")
    if project.get("canonical_base_random_init_only") is not True:
        errors.append("canonical Base boundary not enforced")
    if rights.get("software_license") != "BSD-2-Clause":
        errors.append("software license drift")
    if rights.get("dataset_rights_authorized") is not False:
        errors.append("dataset rights incorrectly authorized")
    if rights.get("third_party_notice_required") is not True:
        errors.append("third-party notice boundary missing")
    if qualification.get("required_ops") != ["RMSNorm", "RoPE", "SwiGLU", "cross_entropy"]:
        errors.append("qualification operator set drift")
    if qualification.get("requires_real_upstream_execution") is not True:
        errors.append("real runtime gate disabled")
    if qualification.get("requires_gpu_triton_runtime") is not True:
        errors.append("GPU/Triton gate disabled")
    if qualification.get("benchmark_runs_required") != 2:
        errors.append("benchmark repetition gate drift")
    for key, message in (
        ("foreign_pretrained_weights", "foreign weights boundary violated"),
        ("foreign_tokenizer", "foreign tokenizer boundary violated"),
        ("foreign_instruction_alignment", "foreign alignment boundary violated"),
        ("canonical_base_mutation", "canonical Base mutation boundary violated"),
        ("training", "training boundary violated"),
        ("paid_compute", "paid compute boundary violated"),
        ("final_test_payload", "final-test boundary violated"),
        ("dataset_ingestion", "dataset ingestion boundary violated"),
    ):
        if forbidden.get(key) is not False:
            errors.append(message)
    return errors


def validate_evidence(evidence: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if evidence.get("project_base_sha") != manifest["project"]["base_sha"]:
        errors.append("evidence project base mismatch")
    if evidence.get("upstream", {}).get("commit") != manifest["upstream"]["commit"]:
        errors.append("evidence upstream mismatch")
    if evidence.get("promotion_state") != "RETEST_RUNTIME_REQUIRED":
        errors.append("runtime-unproven evidence must remain RETEST_RUNTIME_REQUIRED")
    if evidence.get("parity", {}).get("proven") is not False:
        errors.append("parity cannot be proven without real runtime")
    if evidence.get("benchmark", {}).get("executed") is not False:
        errors.append("benchmark cannot be marked executed without real runtime")
    canonical = evidence.get("canonical_base_integrity", {})
    required_false = (
        "foreign_weights_used",
        "tokenizer_changed",
        "checkpoint_changed",
        "training_executed",
        "paid_compute_used",
    )
    if any(canonical.get(key) is not False for key in required_false):
        errors.append("canonical Base/compute boundary not false")
    if evidence.get("environment_hash") != sha256_json(evidence.get("environment", {})):
        errors.append("environment identity mismatch")
    expected_identity = sha256_json({k: v for k, v in evidence.items() if k != "evidence_identity"})
    if evidence.get("evidence_identity") != expected_identity:
        errors.append("evidence identity mismatch")
    return errors


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest) + validate_evidence(evidence, manifest)
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS manifest_sha256={sha256_json(manifest)} evidence_identity={evidence['evidence_identity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
