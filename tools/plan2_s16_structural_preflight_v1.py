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
    _need(type(value) is str and bool(SAFE_ID.fullmatch(value))
          and ".." not in value and value == value.strip(), f"invalid {label}")
    return value


def _hash(value: object, label: str) -> str:
    _need(type(value) is str and bool(HEX.fullmatch(value)), f"invalid {label}")
    return value


def _physical_snapshot(values: list | tuple) -> list:
    """Snapshot built-in container storage, not an overrideable iterator.

    A list/tuple subclass may override __iter__ and silently omit a revoked
    source, reserved evaluation digest, or physical corpus record. Using the
    builtin container iterator prevents that caller-controlled omission.
    This is a candidate-only integrity check, not concurrent-write locking.
    """
    if isinstance(values, list):
        return list(list.__iter__(values))
    return list(tuple.__iter__(values))


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
    _need(isinstance(records, (list, tuple)), "no physical records")
    physical_rows = _physical_snapshot(records)
    _need(bool(physical_rows), "no physical records")
    _need(isinstance(reserved_payload_sha256, (list, tuple)), "invalid reserved index")
    reserved = [_hash(v, "reserved payload SHA") for v in _physical_snapshot(reserved_payload_sha256)]
    _need(len(reserved) == len(set(reserved)), "duplicate reserved payload")
    heldout = set(reserved)
    _need(isinstance(revoked_source_ids, (list, tuple)), "invalid revoked source index")
    revoked = [_id(value, "revoked source") for value in _physical_snapshot(revoked_source_ids)]
    _need(len(revoked) == len(set(revoked)), "duplicate revoked source")

    # Snapshot each Mapping once: receipt bytes must bind validated values,
    # never a later reread of caller-owned or concurrently mutable records.
    validated_rows: list[dict[str, object]] = []
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
    for raw in physical_rows:
        _need(isinstance(raw, Mapping),
              "record envelope incomplete or contains unsupported grant")
        try:
            # Snapshot physical dict storage before any caller-defined mapping
            # views can hide a grant or a revoked member. Reject non-exact str
            # keys *before* rehashing or comparing them to schema strings:
            # a str subclass may execute arbitrary code in __hash__/__eq__.
            if isinstance(raw, dict):
                physical_items = list(dict.items(raw))
                _need(all(type(key) is str for key, _ in physical_items),
                      "invalid record mapping keys")
                row = dict(physical_items)
            else:
                # dict(mapping) silently collapses repeated keys. Snapshot the
                # enumerated keys first, validate their exact type before any
                # hashing, and reject duplicates before fetching each value.
                # This is a structural candidate check, not a sandbox for
                # arbitrary executable Mapping implementations.
                # An arbitrary Mapping can yield an infinite key iterator or
                # conceal a member while advertising a larger physical size.
                # Enumerate no more than one key past the exact schema size;
                # never materialize an unbounded untrusted iterator. This
                # consistency check is a candidate guard, not a sandbox for
                # caller-executable Mapping methods or concurrent mutation.
                mapping_keys = [
                    key for _, key in zip(range(len(RECORD_KEYS) + 1), iter(raw))
                ]
                _need(all(type(key) is str for key in mapping_keys),
                      "invalid record mapping keys")
                _need(len(mapping_keys) == len(set(mapping_keys)),
                      "duplicate record mapping keys")
                _need(len(mapping_keys) <= len(RECORD_KEYS),
                      "record mapping key enumeration overflow")
                _need(len(raw) == len(mapping_keys),
                      "record mapping length mismatch")
                row = {key: raw[key] for key in mapping_keys}
        except Section16PreflightDenied:
            raise
        except Exception:  # noqa: BLE001
            # A caller-owned Mapping may raise or expose private material while
            # being enumerated. Fail closed with a payload-free public reason.
            raise Section16PreflightDenied("invalid record mapping") from None
        _need(set(row) == RECORD_KEYS,
              "record envelope incomplete or contains unsupported grant")
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
        validated_rows.append(row)

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
        _canonical(sorted(validated_rows, key=lambda row: row["record_id"]))
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
        "physical_record_count": len(validated_rows),
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
