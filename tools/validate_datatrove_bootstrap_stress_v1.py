#!/usr/bin/env python3
"""Fail-closed validator for the DataTrove bootstrap qualification manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED = {
    "upstream_commit": "7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664",
    "upstream_tag": "v0.10.0",
    "wheel_sha256": "c7bb75deed2c3e88fb5138f8ea075a170ee98d6c94fc263829609091ea9c2b5d",
    "sdist_sha256": "e31f89bdccb30ef0796854f5842ff52b4b224c28b2d5b110088e84071ea05c40",
}
REQUIRED_TOP = {
    "schema_version", "component", "upstream", "rights", "environment",
    "bootstrap", "runtime", "parity", "benchmark", "canonical_base_safety", "decision",
}
VALID_DECISIONS = {
    "ADOPTABLE_COMPONENT", "EXPERIMENTAL_CANDIDATE", "RETEST_RUNTIME_REQUIRED",
    "BLOCKED_ENVIRONMENT", "BLOCKED_RIGHTS", "BLOCKED_PARITY", "REJECTED",
}


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/research/datatrove_bootstrap_stress_v1.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return fail(f"cannot read manifest: {exc}")

    missing = REQUIRED_TOP - set(data)
    if missing:
        return fail(f"missing top-level fields: {sorted(missing)}")
    if data["schema_version"] != 1:
        return fail("unsupported schema_version")
    if data["decision"] not in VALID_DECISIONS:
        return fail("invalid decision state")
    if data["component"] != {"id": "DATATROVE", "package": "datatrove", "version": "0.10.0"}:
        return fail("component identity drift")

    upstream = data["upstream"]
    exact = {
        "repository": "https://github.com/huggingface/datatrove",
        "tag": EXPECTED["upstream_tag"],
        "commit": EXPECTED["upstream_commit"],
        "wheel_sha256": EXPECTED["wheel_sha256"],
        "sdist_sha256": EXPECTED["sdist_sha256"],
    }
    for key, value in exact.items():
        if upstream.get(key) != value:
            return fail(f"upstream.{key} drift")

    rights = data["rights"]
    for key, expected in {
        "software_license": "Apache-2.0",
        "source_notice_present": False,
        "dataset_rights_inherited": False,
        "model_weight_rights_applicable": False,
        "training_authority_from_code_license": False,
    }.items():
        if rights.get(key) != expected:
            return fail(f"rights.{key} violates recorded boundary")

    bootstrap = data["bootstrap"]
    for key in ("isolated_env_created", "exact_install_attempted", "installed", "install_command", "result_code"):
        if key not in bootstrap:
            return fail(f"missing bootstrap.{key}")
    if bootstrap["exact_install_attempted"] is not True:
        return fail("exact installation was not attempted")
    if bootstrap["installed"]:
        if bootstrap.get("installed_version") != "0.10.0":
            return fail("installed DataTrove version is not exact 0.10.0")
        if bootstrap.get("dependency_lock_status") not in {"LOCKED", "LOCKED_REPRODUCIBLE", "LOCKED_RUNTIME_FREEZE"}:
            return fail("installed runtime lacks a recorded dependency lock")
        if bootstrap["dependency_lock_status"] == "LOCKED_RUNTIME_FREEZE" and not bootstrap.get("dependency_lock_sha256"):
            return fail("runtime freeze lock lacks SHA-256")

    runtime = data["runtime"]
    parity = data["parity"]
    if runtime.get("execution_mode") == "mock":
        return fail("mock execution cannot be runtime evidence")
    if not bootstrap["installed"]:
        if runtime.get("status") != "NOT_EXECUTED":
            return fail("runtime must be NOT_EXECUTED without exact installation")
        if parity.get("status") != "NOT_EXECUTED":
            return fail("parity must be NOT_EXECUTED without exact runtime")
        if data["decision"] == "ADOPTABLE_COMPONENT":
            return fail("cannot adopt without exact installed runtime")

    safety = data["canonical_base_safety"]
    for key in (
        "canonical_base_modified", "foreign_pretrained_weights_used",
        "foreign_instruction_or_alignment_behavior_used", "tokenizer_modified",
        "training_executed", "benchmark_final_test_data_accessed", "model_checkpoint_modified",
    ):
        if safety.get(key) is not False:
            return fail(f"canonical_base_safety.{key} is not false")

    print("PASS: machine-readable DataTrove qualification manifest is internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
