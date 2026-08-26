#!/usr/bin/env python3
"""Validate NEXT100-032 English Wikisource source qualification evidence.

This validator is intentionally LOCAL_FREE and network-free. It checks the
sealed source authority, bounded revision set, rights separation, family
accounting, exact-hash dedup claims, and evidence self-identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_WORKER = "NEXT100-032-DATA-EN-WIKISOURCE"
EXPECTED_REVISIONS = [8450353, 8450364, 6931309]
EXPECTED_NORMALIZED_SHA256 = "1c4ec6e66b425e517a17fb865dabaf3aeddfb1a16cb7e40bca2984be56dce0e7"
EXPECTED_REGISTRY_PATH = "data/registry/external_snapshots.v2.json"


def canonical_sha256_without_identity(obj: dict) -> str:
    payload = dict(obj)
    payload.pop("evidence_identity_sha256", None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(repo: Path) -> list[str]:
    failures: list[str] = []
    evidence_path = repo / "evidence/next100_032/en_wikisource_qualification.json"
    if not evidence_path.is_file():
        return [f"missing evidence: {evidence_path}"]
    e = load(evidence_path)

    if e.get("worker_id") != EXPECTED_WORKER:
        failures.append("worker_id mismatch")
    if e.get("terminal_verdict") != "ADMIT":
        failures.append("terminal verdict is not ADMIT")
    if e.get("terminal_scope") != "BOUNDED_RIGHTS_CLEAR_SCAN_BACKED_SUBSET_ONLY":
        failures.append("scope is not bounded rights-clear subset")
    if e.get("whole_project_dump", {}).get("verdict") != "REJECT":
        failures.append("whole-project dump must fail closed")

    rights = e.get("rights", {})
    if rights.get("underlying_work", {}).get("status") != "ALLOWED":
        failures.append("underlying work not allowed")
    if rights.get("platform_contributions", {}).get("status") != "ALLOWED":
        failures.append("platform contribution layer not allowed")
    if rights.get("edition_specific", {}).get("status") != "ALLOWED_FOR_SELECTED_1902_SCAN_TRANSCRIPTION_ONLY":
        failures.append("edition-specific scope not pinned")
    pp = rights.get("project_policy_compatibility", {})
    for field in ("acquisition", "storage", "analysis", "model_training"):
        if pp.get(field) != "ALLOWED":
            failures.append(f"{field} is not ALLOWED")
    if pp.get("evaluation") != "NOT_SEPARATELY_ADMITTED":
        failures.append("evaluation must remain separately gated")

    selected = e.get("provenance", {}).get("selected_revisions", [])
    revision_ids = [row.get("revision_id") for row in selected]
    if revision_ids != EXPECTED_REVISIONS:
        failures.append(f"revision set mismatch: {revision_ids!r}")
    if any(row.get("proofread_state") != "VALIDATED" for row in selected):
        failures.append("not every selected page is VALIDATED")
    if len({row.get("normalized_extracted_sha256") for row in selected}) != len(selected):
        failures.append("duplicate per-page normalized hash")

    acq = e.get("bounded_acquisition", {})
    if acq.get("page_revision_count") != 3:
        failures.append("bounded page count mismatch")
    if acq.get("normalized_sha256") != EXPECTED_NORMALIZED_SHA256:
        failures.append("bounded normalized hash mismatch")
    if acq.get("normalized_bytes") != 5536:
        failures.append("bounded normalized byte count mismatch")

    q = e.get("quality", {})
    if q.get("selected_pages_validated") != q.get("selected_pages_total"):
        failures.append("quality validation count mismatch")
    if q.get("unicode_replacement_characters") != 0:
        failures.append("replacement character defect")
    if q.get("language") != "en":
        failures.append("language mismatch")

    dedup = e.get("dedup", {})
    if dedup.get("within_subset", {}).get("status") != "PASS":
        failures.append("within-subset dedup not PASS")
    if dedup.get("against_live_registry", {}).get("status") != "PASS_EXACT_HASH_ONLY":
        failures.append("registry exact-hash dedup state mismatch")
    if not dedup.get("corpus_convergence_requirement"):
        failures.append("missing corpus near-copy convergence requirement")

    fam = e.get("family_lineage", {})
    if fam.get("parent_ecosystem") != "wikimedia":
        failures.append("Wikimedia parent lineage missing")
    if fam.get("automatic_independent_family_credit") is not False:
        failures.append("Wikimedia family was automatically counted independent")
    if fam.get("current_diversity_credit") != 0:
        failures.append("source must carry zero automatic diversity credit")

    live = e.get("live_registry_binding", {})
    if live.get("registry_path") != EXPECTED_REGISTRY_PATH:
        failures.append("registry path mismatch")
    registry_path = repo / EXPECTED_REGISTRY_PATH
    if registry_path.is_file():
        registry = load(registry_path)
        if registry.get("registry_identity_sha256") != live.get("observed_registry_identity_sha256"):
            failures.append("live registry identity drifted from sealed observation")
        if registry.get("source_count") != dedup.get("against_live_registry", {}).get("observed_source_count"):
            failures.append("live registry source count drifted from sealed observation")
        current_hashes = {
            src.get("snapshot", {}).get("normalized_sha256")
            for src in registry.get("sources", [])
            if src.get("snapshot", {}).get("normalized_sha256")
        }
        candidate_hashes = {acq.get("normalized_sha256")} | {
            row.get("normalized_extracted_sha256") for row in selected
        }
        if current_hashes & candidate_hashes:
            failures.append("candidate exact normalized hash collides with live registry")
    else:
        failures.append(f"missing live registry file: {registry_path}")

    want = e.get("evidence_identity_sha256")
    got = canonical_sha256_without_identity(e)
    if want != got:
        failures.append(f"evidence identity mismatch: expected {want}, computed {got}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    failures = validate(Path(args.repo))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS NEXT100-032 EN WIKISOURCE QUALIFICATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
