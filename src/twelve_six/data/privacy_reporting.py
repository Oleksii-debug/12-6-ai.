"""Compact, secret-safe aggregate reporting for privacy corpus scans."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Iterable

from .privacy_filter import (
    COVERAGE_CLAIM,
    PrivacyFilterError,
    PrivacyRecordResult,
    assert_no_secret_values_in_manifest,
    privacy_policy_manifest,
)

PRIVACY_CORPUS_SUMMARY_SCHEMA = "12-6.pii-secrets-corpus-scan-summary.v1"
_ACTIONS = ("ALLOW", "REDACT", "QUARANTINE", "EXCLUDE")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def build_corpus_scan_summary(
    results: Iterable[PrivacyRecordResult],
    *,
    corpus_identity_sha256: str,
) -> dict[str, Any]:
    """Build manifest-bound aggregate evidence without record text or match values."""
    if not re.fullmatch(r"[0-9a-f]{64}", corpus_identity_sha256):
        raise PrivacyFilterError("corpus_identity_sha256 must be lowercase SHA-256")
    rows = list(results)
    policy = privacy_policy_manifest()
    expected_policy = policy["policy_sha256"]
    detector_names = tuple(sorted(policy["detector_actions"]))
    if any(row.policy_sha256 != expected_policy for row in rows):
        raise PrivacyFilterError("privacy policy identity drift within scan")

    detector_counts: Counter[str] = Counter({name: 0 for name in detector_names})
    action_counts: Counter[str] = Counter({name: 0 for name in _ACTIONS})
    modality_counts: dict[str, Counter[str]] = {
        modality: Counter(
            {
                **{name: 0 for name in _ACTIONS},
                **{f"detector:{name}": 0 for name in detector_names},
            }
        )
        for modality in ("natural", "code")
    }
    source_counts: dict[str, Counter[str]] = {}
    evidence_digest = hashlib.sha256()
    for row in rows:
        detector_counts.update(row.detector_counts)
        action_counts[row.action] += 1
        modality_counts[row.modality][row.action] += 1
        source_counter = source_counts.setdefault(
            row.source_id,
            Counter({name: 0 for name in _ACTIONS}),
        )
        source_counter[row.action] += 1
        for detector, count in row.detector_counts.items():
            modality_counts[row.modality][f"detector:{detector}"] += count
        evidence_digest.update(_canonical_json_bytes(row.evidence_record()))

    core = {
        "schema_version": PRIVACY_CORPUS_SUMMARY_SCHEMA,
        "privacy_policy_sha256": expected_policy,
        "corpus_identity_sha256": corpus_identity_sha256,
        "coverage_claim": COVERAGE_CLAIM,
        "records_total": len(rows),
        "records_train_eligible_after_privacy": sum(
            row.train_eligible_after_privacy for row in rows
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "detector_counts": dict(sorted(detector_counts.items())),
        "by_modality": {
            key: dict(sorted(value.items())) for key, value in modality_counts.items()
        },
        "by_source": {
            key: dict(sorted(value.items())) for key, value in sorted(source_counts.items())
        },
        "records_evidence_sha256": evidence_digest.hexdigest(),
    }
    manifest = {**core, "manifest_sha256": _sha256_json(core)}
    assert_no_secret_values_in_manifest(manifest)
    return manifest
