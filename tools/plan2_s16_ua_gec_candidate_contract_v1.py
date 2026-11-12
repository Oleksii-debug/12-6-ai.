"""Plan2 S16 W2: pinned UA-GEC candidate evidence guard, never a corpus grant.

This module validates one versioned, source-metadata-only fixture.  A valid
receipt means the candidate evidence has not drifted; it NEVER authorizes
training, corpus admission, tokenizer fit, production release or Plan9 binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "12-6.plan2-s16-w2-upstream-candidate-evidence.v1"
RECEIPT_SCHEMA = "12-6.plan2-s16-w2-upstream-candidate-readback.v1"
DECISION = "BLOCKED_CANDIDATE_ONLY"
CANDIDATE_ID = "ua-gec-2026-10-13-rights-and-privacy-triage"
SOURCE = {
    "repository": "grammarly/ua-gec",
    "upstream_commit": "4757f72f192c4a41e4c8fb1d9690a948f87cf6d6",
    "license_path": "LICENSE",
    "license_git_blob_sha1": "10fabd90118f7ce38bb2e4753e105e744442a429",
    "license_name": "CC-BY-4.0",
    "readme_path": "README.md",
    "readme_git_blob_sha1": "efa28e3ab384f5e97c8f9e66657243aa20c30b64",
    "metadata_path": "data/metadata.csv",
    "metadata_git_blob_sha1": "1d6f69c8f26c8db90a4f0502bba241363242fc18",
}
SCOPE = {
    "language": "uk",
    "family_count_credit": 1,
    "train_test_separation_required": True,
    "test_split_for_training": False,
    "gec_only_and_gec_fluency_are_independent_families": False,
    "annotated_source_target_are_independent_documents": False,
    "metadata_sensitive_fields": [
        "author_id", "region", "gender", "occupation", "submission_type",
        "source_language", "annotator_id", "is_sensitive",
    ],
}
EXTERNAL_EVIDENCE = [
    "ORIGINAL_CONTRIBUTOR_COPYRIGHT_AND_DONATION_SCOPE",
    "TRANSLATION_RIGHTS_AND_THIRD_PARTY_ORIGINALS",
    "EXPLICIT_OWNER_APPROVED_TRAINING_AND_RELEASE_SCOPE",
    "ATTRIBUTION_AND_DERIVATIVE_DISTRIBUTION_PLAN",
    "PII_REVIEW_AND_CORRECTION_REMOVAL_POLICY",
    "PHYSICAL_MEMBER_SHA256_AND_PROVENANCE",
    "INDEPENDENT_UKRAINIAN_EVALUATION_SOURCE",
    "CROSS_FAMILY_DECONTAMINATION_AND_BALANCE",
]
FALSE_FLAGS = (
    "physical_corpus_materialized", "original_grants_independently_verified",
    "privacy_gate_passed", "source_training_authorized",
    "corpus_training_authorized", "production_release_authorized",
    "plan9_binding_authorized", "terminal_done",
)
TOP_KEYS = frozenset((
    "schema_version", "candidate_id", "decision", "source", "scope",
    "required_external_evidence", *FALSE_FLAGS,
))


class CandidateEvidenceDenied(ValueError):
    """Candidate fixture drift or attempted authority escalation."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise CandidateEvidenceDenied(reason)


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _need(key not in result, "duplicate JSON evidence key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise CandidateEvidenceDenied(f"non-finite JSON value: {value}")


def _exact_dict(actual: Any, expected: dict[str, Any], name: str) -> None:
    _need(type(actual) is dict, f"{name} must be an object")
    _need(set(actual) == set(expected), f"{name} keys drifted")
    for key, value in expected.items():
        got = actual[key]
        _need(type(got) is type(value) and got == value,
              f"{name}.{key} drifted")


def inspect_bytes(raw: bytes) -> dict[str, Any]:
    """Return hash-only blocked receipt for exact v1 fixture, or deny drift."""
    _need(type(raw) is bytes and 0 < len(raw) <= 65536,
          "invalid or unbounded candidate evidence")
    try:
        candidate = json.loads(raw.decode("utf-8", "strict"),
                               object_pairs_hook=_no_duplicate_pairs,
                               parse_constant=_invalid_constant)
    except (UnicodeError, ValueError) as exc:
        raise CandidateEvidenceDenied("invalid candidate JSON authority") from exc
    _need(type(candidate) is dict, "candidate evidence must be an object")
    _need(set(candidate) == TOP_KEYS, "candidate authority keys drifted")
    _need(type(candidate["schema_version"]) is str and
          candidate["schema_version"] == SCHEMA, "candidate schema drifted")
    _need(type(candidate["candidate_id"]) is str and
          candidate["candidate_id"] == CANDIDATE_ID, "candidate identity drifted")
    _need(type(candidate["decision"]) is str and
          candidate["decision"] == DECISION, "candidate decision promoted")
    _exact_dict(candidate["source"], SOURCE, "source")
    _exact_dict(candidate["scope"], SCOPE, "scope")
    _need(type(candidate["required_external_evidence"]) is list and
          candidate["required_external_evidence"] == EXTERNAL_EVIDENCE,
          "required external rights/privacy evidence drifted")
    for field in FALSE_FLAGS:
        _need(candidate[field] is False, f"unsafe candidate promotion: {field}")
    canonical = (json.dumps(candidate, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "decision": DECISION,
        "source_upstream_commit": SOURCE["upstream_commit"],
        "source_license_blob_sha1": SOURCE["license_git_blob_sha1"],
        "source_metadata_blob_sha1": SOURCE["metadata_git_blob_sha1"],
        "raw_input_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_candidate_sha256": hashlib.sha256(canonical).hexdigest(),
        "missing_external_evidence": EXTERNAL_EVIDENCE.copy(),
        "physical_corpus_admitted": False,
        "training_authorized": False,
        "production_release_authorized": False,
        "plan9_binding_authorized": False,
        "terminal_done": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        path = args.candidate
        _need(not path.is_symlink() and path.is_file(), "unsafe candidate file")
        _need(path.stat().st_size <= 65536, "unbounded candidate file")
        receipt = inspect_bytes(path.read_bytes())
    except (OSError, CandidateEvidenceDenied) as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
