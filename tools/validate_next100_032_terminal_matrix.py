#!/usr/bin/env python3
"""Validate NEXT100-032 terminal ADMIT/REJECT/RETEST seal."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_MATRIX = {
    "bounded_scan_backed_pd_transcription": "ADMIT",
    "whole_english_wikisource_project_or_dump": "REJECT",
    "independent_family_credit_vs_other_wikimedia_projects": "RETEST",
}
EXPECTED_REGISTRY_HEAD = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_REGISTRY_ID = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
EXPECTED_PRIMARY_ID = "d16a5e53ad3940778278532ee0be0602d7b9bbca3b27e53ad51e2287b1fa32e9"


def canonical_hash_without_identity(obj: dict) -> str:
    payload = dict(obj)
    payload.pop("matrix_identity_sha256", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(repo: Path) -> list[str]:
    failures: list[str] = []
    matrix_path = repo / "evidence/next100_032/terminal_matrix.json"
    primary_path = repo / "evidence/next100_032/en_wikisource_qualification.json"
    registry_path = repo / "data/registry/external_snapshots.v2.json"
    for path in (matrix_path, primary_path, registry_path):
        if not path.is_file():
            return [f"missing required file: {path}"]

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    if matrix.get("terminal_matrix") != EXPECTED_MATRIX:
        failures.append("terminal matrix mismatch")
    if matrix.get("local_free_only") is not True:
        failures.append("LOCAL_FREE seal missing")
    if matrix.get("bounded_authority", {}).get("primary_evidence_identity_sha256") != EXPECTED_PRIMARY_ID:
        failures.append("primary evidence binding mismatch")
    if primary.get("evidence_identity_sha256") != EXPECTED_PRIMARY_ID:
        failures.append("bound primary evidence changed")

    live = matrix.get("live_registry_refresh", {})
    if live.get("registry_head_sha") != EXPECTED_REGISTRY_HEAD:
        failures.append("registry head binding mismatch")
    if live.get("registry_identity_sha256") != EXPECTED_REGISTRY_ID:
        failures.append("registry identity binding mismatch")
    if registry.get("registry_identity_sha256") != EXPECTED_REGISTRY_ID:
        failures.append("checked-out registry identity drift")
    if registry.get("source_count") != live.get("source_count"):
        failures.append("checked-out registry source count drift")
    if registry.get("independent_source_family_count") != live.get("independent_source_family_count"):
        failures.append("checked-out registry family count drift")

    family = matrix.get("family_seal", {})
    if family.get("automatic_independent_family_credit") is not False:
        failures.append("automatic Wikimedia family credit is forbidden")
    if family.get("current_diversity_credit") != 0:
        failures.append("current Wikimedia diversity credit must be zero")
    if family.get("family_credit_terminal_state") != "RETEST":
        failures.append("family-credit state must be RETEST")

    observed_prs = {row.get("pr") for row in matrix.get("concurrent_wikimedia_authorities_observed", [])}
    if not {450, 455}.issubset(observed_prs):
        failures.append("late Wikimedia concurrency observations missing")

    computed = canonical_hash_without_identity(matrix)
    if matrix.get("matrix_identity_sha256") != computed:
        failures.append(
            f"matrix identity mismatch: expected {matrix.get('matrix_identity_sha256')}, computed {computed}"
        )
    return failures


def main() -> int:
    failures = validate(Path("."))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS NEXT100-032 TERMINAL MATRIX")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
