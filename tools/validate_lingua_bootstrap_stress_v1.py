"""Fail-closed validator for the Lingua 2.2.0 bootstrap-stress package."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

CONFIG_SCHEMA = "12-6.lingua-bootstrap-stress.v1"
VERDICTS = {"RETEST_RUNTIME_REQUIRED", "EXPERIMENTAL_CANDIDATE", "ADOPTABLE_COMPONENT"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate(config, evidence):
    errors = []
    if config.get("schema") != CONFIG_SCHEMA:
        errors.append("config_schema")
    upstream = config.get("upstream", {})
    if upstream.get("tag") != "v2.2.0":
        errors.append("upstream_tag")
    if upstream.get("commit") != "754ce21122c083a7200763015fdaf7cda8d85453":
        errors.append("upstream_commit")
    if upstream.get("license") != "Apache-2.0":
        errors.append("license")
    wheel = upstream.get("target_wheel", {})
    if wheel.get("sha256") != "4fbf936b47ef4fdd7043ebb4159d4a5f1c3648028e19d6e3c60464abc5f5e195":
        errors.append("wheel_sha256")
    if upstream.get("runtime_dependencies") != []:
        errors.append("runtime_dependencies")

    environment = evidence.get("environment", {})
    if not environment:
        errors.append("environment_missing")

    install = evidence.get("install", {})
    if install.get("runtime_executed"):
        runtime = evidence.get("runtime", {})
        if runtime.get("installed_version") != "2.2.0":
            errors.append("runtime_version")
        if runtime.get("artifact_sha256") != wheel.get("sha256"):
            errors.append("runtime_artifact_hash")
        if evidence.get("benchmark", {}).get("status") != "EXECUTED":
            errors.append("benchmark_missing")
        if evidence.get("parity", {}).get("status") != "EXECUTED":
            errors.append("parity_missing")

    verdict = evidence.get("verdict")
    if verdict not in VERDICTS:
        errors.append("invalid_verdict")
    if verdict == "ADOPTABLE_COMPONENT" and (errors or not install.get("runtime_executed")):
        errors.append("adoption_gate")

    safety = evidence.get("safety", {})
    if safety.get("canonical_base_changed") is not False:
        errors.append("canonical_base_changed")
    if safety.get("foreign_weights_used") is not False:
        errors.append("foreign_weights")
    if safety.get("training_executed") is not False:
        errors.append("training")

    payload = dict(evidence)
    claimed = payload.pop("identity_sha256", None)
    if claimed != digest(payload):
        errors.append("evidence_identity")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    errors = validate(config, evidence)
    output = {
        "validation": "PASS" if not errors else "FAIL",
        "errors": errors,
        "evidence_identity_sha256": evidence.get("identity_sha256"),
        "checked_python": platform.python_version(),
        "checked_implementation": sys.implementation.name,
    }
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
