"""Fail-closed validator for the OpenLM bootstrap qualification package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED = {
    "component": "OPEN_LM",
    "project_repository": "Oleksii-debug/12-6-ai.",
    "upstream_commit": "9bb92ef1689333534b7057942a20d18a46d1fa52",
    "license": "MIT",
    "base_sha": "5020afd671a3885c1b738c8b4eafe7525f630546",
}


def canonical_hash(payload: dict[str, Any], identity_field: str) -> str:
    body = dict(payload)
    body.pop(identity_field, None)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def validate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        fail("manifest must be an object")

    if payload.get("component") != EXPECTED["component"]:
        fail("component identity drift")
    if payload.get("project_repository") != EXPECTED["project_repository"]:
        fail("project repository drift")
    if payload.get("project_base_sha") != EXPECTED["base_sha"]:
        fail("project base SHA drift")

    upstream = payload.get("upstream")
    if not isinstance(upstream, dict):
        fail("missing upstream record")
    if upstream.get("immutable_commit") != EXPECTED["upstream_commit"]:
        fail("upstream commit drift")
    if upstream.get("license") != EXPECTED["license"]:
        fail("upstream license drift")
    if upstream.get("source_head_observed") != EXPECTED["upstream_commit"]:
        fail("upstream source head drift")
    if upstream.get("tag_or_release") is not None:
        fail("unexpected unverified tag/release binding")

    requirements = payload.get("upstream_requirements")
    if not isinstance(requirements, dict):
        fail("missing upstream requirements")
    entries = requirements.get("entries")
    if not isinstance(entries, list) or not entries:
        fail("missing upstream requirement entries")
    if "pandas==2.1.4" not in entries:
        fail("missing exact upstream pandas pin")
    if not requirements.get("floating_or_lower_bound_entries"):
        fail("floating dependency inventory unexpectedly empty")
    if requirements.get("bootstrap_policy") != (
        "refuse unresolved/floating runtime closure for promoted execution evidence"
    ):
        fail("bootstrap policy drift")

    installation = payload.get("installation_attempt")
    if not isinstance(installation, dict):
        fail("missing installation attempt")
    if installation.get("attempted") is not True:
        fail("installation was not attempted")
    if installation.get("attempted_exact_requirement") != "pandas==2.1.4":
        fail("exact installation target drift")
    if installation.get("return_code") != 1:
        fail("install return code must remain the observed failure")
    if installation.get("execution_status") != "NOT_EXECUTED":
        fail("install execution status must remain NOT_EXECUTED")
    if installation.get("artifact_sha256") is not None:
        fail("unavailable artifact must not have a fabricated hash")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        fail("missing runtime state")
    if runtime.get("execution_status") != "NOT_EXECUTED":
        fail("runtime cannot be promoted without execution")
    if runtime.get("benchmark_status") != "NOT_EXECUTED":
        fail("benchmark cannot be claimed without execution")
    if runtime.get("parity_proven") is not False:
        fail("parity cannot be true without runtime evidence")
    if runtime.get("adoptable") is not False:
        fail("adoptable cannot be true without all gates")

    safety = payload.get("canonical_base_safety")
    if not isinstance(safety, dict):
        fail("missing canonical Base safety record")
    for key in (
        "foreign_pretrained_weights_used",
        "foreign_instruction_or_alignment_behavior_imported",
        "canonical_tokenizer_changed",
        "canonical_base_checkpoint_touched",
        "training_launched",
        "paid_compute_used",
        "benchmark_or_final_test_payload_accessed",
    ):
        if safety.get(key) is not False:
            fail(f"canonical Base safety violation: {key}")

    identity = payload.get("evidence_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        fail("missing evidence identity")
    calculated = canonical_hash(payload, "evidence_identity_sha256")
    if identity != calculated:
        fail("evidence identity mismatch")

    return {"status": "PASS", "evidence_identity_sha256": calculated}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="configs/research/open_lm_bootstrap_stress_v1.json")
    args = parser.parse_args()
    result = validate(Path(args.manifest))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
