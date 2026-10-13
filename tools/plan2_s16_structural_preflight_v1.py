"""Section 16 physical-corpus structural preflight; never a rights or release authority.

This is intentionally downstream of *separately* verified S3--S9 stages. It
cannot admit a source, infer a license from a hash, certify PII clearance,
release final-test text, or authorize training. Records are text-free evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "12-6.plan2-s16-structural-candidate.v1"
RECORD_KEYS = frozenset({
    "record_id", "source_id", "source_family", "document_family",
    "source_revision", "language", "split", "payload_sha256",
    "normalized_sha256", "rights_receipt_sha256", "privacy_receipt_sha256",
})
SPLITS = ("train", "validation", "test")
LANGUAGES = ("uk", "en")
HEX = re.compile(r"[a-f0-9]{64}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9._:/-]{1,256}\Z")


class Section16PreflightDenied(ValueError):
    """Structural acceptance is impossible, even before real rights review."""


def _need(test: bool, reason: str) -> None:
    if not test:
        raise Section16PreflightDenied(reason)


def _id(value: object, label: str) -> str:
    _need(isinstance(value, str) and bool(SAFE_ID.fullmatch(value))
          and ".." not in value and value == value.strip(), f"invalid {label}")
    return value


def _hash(value: object, label: str) -> str:
    _need(isinstance(value, str) and bool(HEX.fullmatch(value)), f"invalid {label}")
    return value


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def inspect_structure(
    records: Sequence[Mapping[str, object]],
    *,
    reserved_payload_sha256: Sequence[str] = (),
    revoked_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Check text-free candidate membership, identity, cross-split leakage, balance.

    Neither supplied receipt SHA strings nor structural PASS prove independently
    verified licenses, privacy decisions, near-duplicates, or decontamination.
    The output always refuses production corpus/training/release authority.
    """
    _need(isinstance(records, (list, tuple)) and bool(records), "no physical records")
    _need(isinstance(reserved_payload_sha256, (list, tuple)), "invalid reserved index")
    reserved = [_hash(v, "reserved payload SHA") for v in reserved_payload_sha256]
    _need(len(reserved) == len(set(reserved)), "duplicate reserved payload")
    heldout = set(reserved)
    _need(isinstance(revoked_source_ids, (list, tuple)), "invalid revoked source index")
    revoked = [_id(value, "revoked source") for value in revoked_source_ids]
    _need(len(revoked) == len(set(revoked)), "duplicate revoked source")

    unique_records: set[str] = set()
    seen_payloads: set[str] = set()
    seen_normalized: set[str] = set()
    family_splits: dict[str, str] = {}
    family_languages: dict[str, str] = {}
    document_splits: dict[str, str] = {}
    source_identity: dict[str, tuple[str, str, str]] = {}
    counts: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    families: dict[str, dict[str, set[str]]] = {
        s: defaultdict(set) for s in SPLITS
    }
    for raw in records:
        _need(isinstance(raw, Mapping) and set(raw) == RECORD_KEYS,
              "record envelope incomplete or contains unsupported grant")
        row = dict(raw)
        rid = _id(row["record_id"], "record identity")
        sid = _id(row["source_id"], "source identity")
        _need(sid not in revoked, "revoked source cannot enter corpus")
        fam = _id(row["source_family"], "source family")
        docfam = _id(row["document_family"], "document family")
        revision = _id(row["source_revision"], "source revision")
        lang = row["language"]
        split = row["split"]
        _need(type(lang) is str and lang in LANGUAGES, "unqualified language")
        _need(type(split) is str and split in SPLITS, "invalid data split")
        payload = _hash(row["payload_sha256"], "payload identity")
        normalized = _hash(row["normalized_sha256"], "normalized identity")
        _hash(row["rights_receipt_sha256"], "rights receipt identity")
        _hash(row["privacy_receipt_sha256"], "privacy receipt identity")
        _need(rid not in unique_records, "duplicate record identity")
        unique_records.add(rid)
        # Raw and normalized digests are hashes of bytes in one namespace.
        # A later raw payload cannot alias an earlier normalized member, or
        # vice versa, even when the same-format duplicate sets are disjoint.
        # Comparing against *prior* records permits a legitimate record whose
        # own raw bytes already equal its normalized bytes.
        _need(payload not in seen_payloads and normalized not in seen_normalized
              and payload not in seen_normalized and normalized not in seen_payloads,
              "exact or normalized corpus duplicate")
        seen_payloads.add(payload)
        seen_normalized.add(normalized)
        _need(payload not in heldout and normalized not in heldout,
              "reserved evaluation collision")
        prior_source = source_identity.setdefault(sid, (fam, lang, revision))
        _need(prior_source == (fam, lang, revision), "source language/family/revision drift")
        prior_language = family_languages.setdefault(fam, lang)
        _need(prior_language == lang, "family language drift")
        prior_family = family_splits.setdefault(fam, split)
        _need(prior_family == split, "source family leaked across splits")
        prior_doc = document_splits.setdefault(docfam, split)
        _need(prior_doc == split, "document family leaked across splits")
        counts[split][lang] += 1
        families[split][lang].add(fam)

    for split in SPLITS:
        count = sum(counts[split].values())
        _need(count > 0, f"missing {split} physical cohort")
        for lang in LANGUAGES:
            _need(counts[split][lang] > 0 and bool(families[split][lang]),
                  f"missing {lang} {split} independent evaluation")
            # Conservative 20% candidate-screen heuristic; this is NOT an
            # approved scientific production mixture policy or release gate.
            _need(5 * counts[split][lang] >= count,
                  f"out-of-bound {lang} {split} distribution")
    # Bind the candidate receipt to the exact validated membership, not only
    # to aggregate counters. Canonical ordering prevents input permutation from
    # changing the receipt; no corpus text or authorizing grant is included.
    physical_membership_sha256 = hashlib.sha256(
        _canonical(sorted((dict(row) for row in records), key=lambda row: row["record_id"]))
    ).hexdigest()
    reserved_membership_sha256 = hashlib.sha256(
        _canonical(sorted(reserved))
    ).hexdigest()
    revoked_membership_sha256 = hashlib.sha256(
        _canonical(sorted(revoked))
    ).hexdigest()
    summary = {
        "physical_membership_sha256": physical_membership_sha256,
        "reserved_membership_sha256": reserved_membership_sha256,
        "revoked_membership_sha256": revoked_membership_sha256,
        "schema_version": SCHEMA,
        "status": "STRUCTURAL_CANDIDATE_ONLY_NO_PRODUCTION_ADMISSION",
        "physical_record_count": len(records),
        "distinct_source_count": len(source_identity),
        "distinct_family_count": len(family_splits),
        "split_language_record_counts": {
            split: {lang: counts[split][lang] for lang in LANGUAGES}
            for split in SPLITS
        },
        "split_language_family_counts": {
            split: {lang: len(families[split][lang]) for lang in LANGUAGES}
            for split in SPLITS
        },
        "reserved_reference_hash_count": len(heldout),
        "revoked_source_exclusion_count": len(revoked),
        "candidate_balance_floor": "0.20_PRELIMINARY_ONLY_NOT_POLICY",
        "rights_independently_verified": False,
        "privacy_independently_verified": False,
        "decontamination_independently_verified": False,
        "cross_source_near_dedup_verified": False,
        "training_corpus_authorized": False,
        "evaluation_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "required_next_gate": "PHYSICAL_S3_S9_INDEPENDENT_RIGHTS_PRIVACY_DEDUP_EVAL_ACCEPTANCE",
    }
    summary["structural_candidate_sha256"] = hashlib.sha256(_canonical(summary)).hexdigest()
    return summary
