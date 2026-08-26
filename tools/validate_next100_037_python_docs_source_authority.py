#!/usr/bin/env python3
"""Validate NEXT100-037 bounded CPython documentation source authority.

Stdlib-only and network-free. The authority terminalizes the exact DATA-228
CPython tutorial candidate while late-binding the live DATA-287 registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_AUTHORITY_ID = "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d"
EXPECTED_COMMIT = "7f0ccd6c0e3f85fbaeceb2f67b06ab3631db0480"
EXPECTED_SOURCE_RAW = "cf1674daf9568abeb5fc22f62a991e17751fea4deb06f598362ce6e7de264808"
EXPECTED_SOURCE_NORMALIZED = "64a4ec4fd7574ba4c22e615a032b157e446b9c7f5a7917cb7f10fa214a05bd1a"
EXPECTED_LICENSE_SHA = "b0e25a78cffb43f4d92de8b61ccfa1f1f98ecbc22330b54b5251e7b6ba010231"
EXPECTED_LIVE_REGISTRY_HEAD = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_LIVE_REGISTRY_ID = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
EXPECTED_LIVE_REGISTRY_RUN = 32968622282
EXPECTED_CURRENT_FAMILIES = {
    "ua.rada.open-data.laws-texts",
    "en.standardebooks.manual",
    "github:encode/httpx",
    "github:psf/requests",
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-037 validation failed: {message}")


def validate(path: Path) -> dict[str, object]:
    authority = json.loads(path.read_text(encoding="utf-8"))
    claimed = authority.pop("authority_identity_sha256", None)
    computed = hashlib.sha256(_canonical(authority)).hexdigest()
    _require(claimed == EXPECTED_AUTHORITY_ID, "unexpected authority identity")
    _require(computed == claimed, "authority self-hash drift")

    _require(
        authority["schema_version"]
        == "12-6.next100-037-python-docs-source-authority.v1",
        "schema drift",
    )
    _require(authority["worker_id"] == "NEXT100-037-DATA-EN-PYTHON-DOCS", "worker drift")
    _require(authority["local_free_only"] is True, "LOCAL_FREE boundary weakened")
    _require(authority["terminal_verdict"] == "ADMIT", "terminal verdict drift")

    concurrency = authority["concurrency"]
    live = concurrency["live_external_snapshot_registry"]
    _require(live["head"] == EXPECTED_LIVE_REGISTRY_HEAD, "live registry head drift")
    _require(
        live["registry_identity_sha256"] == EXPECTED_LIVE_REGISTRY_ID,
        "live registry identity drift",
    )
    _require(live["dedicated_workflow_run"] == EXPECTED_LIVE_REGISTRY_RUN, "registry run drift")
    _require(live["dedicated_workflow_conclusion"] == "success", "registry gate not green")
    _require(
        concurrency["same_lineage_conflicting_terminal_authority_prs"] == [],
        "conflicting CPython authority recorded",
    )
    _require(
        concurrency["result"] == "NO_CONFLICTING_CPYTHON_DOCUMENTATION_AUTHORITY_FOUND",
        "late concurrency result drift",
    )
    _require(concurrency["stale_predecessor_pr"] == 451, "superseded PR binding drift")

    source = authority["source"]
    _require(source["source_id"] == "en.python.docs.tutorial-introduction", "source id drift")
    _require(source["source_family"] == "python.cpython.documentation", "family drift")
    _require(source["language"] == "en", "language drift")
    _require(source["modality"] == "natural_language_documentation", "modality drift")
    _require(source["upstream_commit"] == EXPECTED_COMMIT, "upstream commit drift")
    _require(source["python_version_at_commit"] == "3.16.0a0", "version drift")
    _require(source["file_set"] == ["Doc/tutorial/introduction.rst"], "file set expanded")
    _require(
        source["file_set_rule"] == "EXACT_ENUMERATION_ONLY_NO_GLOB_EXPANSION",
        "file-set rule weakened",
    )
    _require(source["raw_sha256"] == EXPECTED_SOURCE_RAW, "raw hash drift")
    _require(source["raw_bytes"] == 19188, "raw byte count drift")
    _require(
        source["normalization"]["normalized_sha256"] == EXPECTED_SOURCE_NORMALIZED,
        "normalized hash drift",
    )
    _require(source["normalization"]["normalized_utf8_bytes"] == 17901, "normalized bytes drift")

    rights = authority["rights"]
    _require(rights["license_id"] == "PSF-2.0", "license id drift")
    _require(rights["license_sha256"] == EXPECTED_LICENSE_SHA, "license hash drift")
    uses = rights["uses"]
    for purpose in ("acquisition", "storage", "analysis", "model_training"):
        _require(uses[purpose] == "ALLOWED", f"{purpose} permission weakened")
    _require(
        uses["redistribution"] == "ALLOWED_WITH_CONDITIONS",
        "redistribution conditions lost",
    )
    _require(uses["evaluation"] == "NOT_SEPARATELY_ADMITTED", "evaluation purpose leaked")
    _require(len(rights["redistribution_conditions"]) == 3, "license obligations incomplete")

    quality = authority["quality_privacy"]
    _require(quality["chunk_count"] == 16, "chunk count drift")
    _require(quality["accepted_chunk_count"] == 14, "accepted chunk count drift")
    _require(quality["rejected_chunk_count"] == 2, "rejected chunk count drift")
    _require(quality["rejection_reasons"] == {"pii_phone": 2}, "rejection evidence drift")
    accepted = quality["accepted_normalized_sha256"]
    _require(len(accepted) == 14 and len(set(accepted)) == 14, "accepted chunk identities invalid")
    _require(quality["exact_duplicate_chunks"] == 0, "exact duplicate chunks present")
    _require(
        quality["training_eligibility"]
        == "ONLY_ACCEPTED_CHUNKS_ARE_ELIGIBLE_REJECTED_CHUNKS_REMAIN_EXCLUDED",
        "quality fail-closed rule weakened",
    )

    separation = authority["code_separation"]
    _require(separation["included_extensions"] == [".rst"], "non-document extension included")
    _require(separation["code_evaluation_reservation_eligible"] is False, "code-eval overlap allowed")
    for suffix in (".py", ".pyi", ".c", ".h"):
        _require(suffix in separation["excluded_source_code_extensions"], "source-code exclusion weakened")

    dedup = authority["dedup"]
    comparison = dedup["comparison_authority"]
    _require(
        comparison["live_external_snapshot_registry_head"] == EXPECTED_LIVE_REGISTRY_HEAD,
        "dedup registry head drift",
    )
    _require(
        comparison["live_external_snapshot_registry_identity_sha256"] == EXPECTED_LIVE_REGISTRY_ID,
        "dedup registry identity drift",
    )
    _require(set(dedup["current_training_families"]) == EXPECTED_CURRENT_FAMILIES, "family vector drift")
    _require(dedup["candidate_family_is_distinct"] is True, "family identity not distinct")
    _require(dedup["raw_sha256_exact_collision"] is False, "raw duplicate collision")
    _require(dedup["normalized_sha256_exact_collision"] is False, "normalized duplicate collision")
    _require(dedup["family_count_credit_if_later_composed"] == 1, "family credit inflated")
    _require(
        dedup["live_registry_explicitly_excludes_candidate_as_historical_nonterminal"] is True,
        "historical registry state not preserved",
    )

    boundary = authority["claim_boundary"]
    for key in (
        "representative_by_itself",
        "corpus_frozen",
        "evaluation_authorized",
        "final_test_material_consumed",
        "training_executed",
        "paid_compute_used",
    ):
        _require(boundary[key] is False, f"truth boundary weakened: {key}")

    return {
        "status": "PASS",
        "terminal_verdict": authority["terminal_verdict"],
        "authority_identity_sha256": claimed,
        "live_registry_identity_sha256": EXPECTED_LIVE_REGISTRY_ID,
        "source_family": source["source_family"],
        "source_id": source["source_id"],
        "accepted_chunk_count": quality["accepted_chunk_count"],
        "rejected_chunk_count": quality["rejected_chunk_count"],
        "evaluation_authorized": False,
        "code_evaluation_reservation_eligible": False,
        "local_free_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="configs/data/next100_037_python_docs_source_authority_v1.json",
    )
    args = parser.parse_args()
    print(json.dumps(validate(Path(args.path)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
