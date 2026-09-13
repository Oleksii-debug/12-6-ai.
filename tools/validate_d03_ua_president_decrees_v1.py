#!/usr/bin/env python3
"""Validate body-free evidence for the bounded Ukrainian Presidential decree intake."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

SCHEMA: Final = "12-6.d03-ua-president-decrees-intake.v1"
SOURCE_ID: Final = "ua.president.official-decrees"
HEX64_RE: Final = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_RECORD_KEYS: Final = {
    "url",
    "document_number",
    "raw_sha256_a",
    "raw_sha256_b",
    "raw_byte_identical",
    "normalized_sha256",
    "normalized_bytes",
    "accepted",
    "reason",
}
FORBIDDEN_TEXT_KEYS: Final = {"text", "body", "subject", "content", "payload"}


def fail(message: str) -> None:
    raise ValueError(message)


def is_nonbool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_materialization_view(evidence: dict[str, Any]) -> dict[str, Any]:
    stable_records: list[dict[str, Any]] = []
    for record in evidence["records"]:
        stable_records.append(
            {
                key: record[key]
                for key in (
                    "url",
                    "document_number",
                    "normalized_sha256",
                    "normalized_bytes",
                    "accepted",
                    "reason",
                )
                if key in record
            }
        )
    return {
        "source_id": SOURCE_ID,
        "catalog_pages": evidence["catalog_pages"],
        "observed_yield": evidence["observed_yield"],
        "records": stable_records,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != "12-6.d03-ua-president-decrees-source-contract.v1":
        fail("wrong contract schema")
    if contract.get("status") != "BOUNDED_INTAKE_ZERO_CREDIT":
        fail("contract status drift")
    source = contract.get("source")
    if not isinstance(source, dict):
        fail("contract source missing")
    if source.get("source_id") != SOURCE_ID or source.get("family_id") != SOURCE_ID:
        fail("contract source identity drift")
    rights = contract.get("rights")
    if not isinstance(rights, dict):
        fail("contract rights missing")
    if rights.get("decision") != "OFFICIAL_DECREE_TEXT_ONLY_CANDIDATE":
        fail("contract rights widened")
    if rights.get("article") != "8(1)(3)":
        fail("contract legal article drift")
    if rights.get("site_wide_cc_by_nc_nd_is_not_training_authority") is not True:
        fail("contract must reject blanket site-license authority")
    execution = contract.get("bounded_execution")
    if not isinstance(execution, dict):
        fail("contract execution missing")
    if execution.get("class") != "LOCAL_FREE":
        fail("contract execution class drift")
    if execution.get("minimum_request_delay_seconds", 0) < 1.0:
        fail("contract request cadence weakened")
    claims = contract.get("claims")
    if not isinstance(claims, dict):
        fail("contract claims missing")
    forbidden_positive = (
        "canonical_capacity_credit_bytes",
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "optimizer_updates",
    )
    if any(claims.get(key) != 0 for key in forbidden_positive):
        fail("contract grants scientific credit")
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_accessed",
        "paid_compute_authorized",
        "learned_20m_claim",
    ):
        if claims.get(key) is not False:
            fail(f"contract forbidden claim: {key}")
    observed = contract.get("contract_identity_sha256")
    if not isinstance(observed, str) or not HEX64_RE.match(observed):
        fail("contract identity missing")
    view = dict(contract)
    view.pop("contract_identity_sha256", None)
    expected = sha256(
        json.dumps(
            view,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    if observed != expected:
        fail("contract identity mismatch")


def validate(evidence: dict[str, Any]) -> None:
    if evidence.get("schema") != SCHEMA:
        fail("wrong schema")
    if evidence.get("status") not in {"PASS_OBSERVED_YIELD", "PASS_ZERO_YIELD"}:
        fail("non-terminal bounded intake status")

    source = evidence.get("source")
    if not isinstance(source, dict):
        fail("missing source")
    if source.get("source_id") != SOURCE_ID or source.get("family_id") != SOURCE_ID:
        fail("source/family identity drift")
    if source.get("stratum") != "uk":
        fail("wrong stratum")
    if source.get("catalog_url") != "https://www.president.gov.ua/documents/decrees/":
        fail("catalog authority drift")
    if source.get("allowed_origin") != "https://www.president.gov.ua":
        fail("origin authority drift")
    if source.get("rights_scope") != "OFFICIAL_DECREE_TEXT_ONLY":
        fail("rights scope widened")
    if source.get("website_blanket_license_not_used_as_training_authority") is not True:
        fail("site-wide license must not become training authority")
    law = source.get("ukraine_copyright_law")
    if not isinstance(law, dict) or law.get("law") != "2811-IX":
        fail("copyright-law authority missing")
    if law.get("article") != "8(1)(3)":
        fail("copyright-law article drift")

    execution = evidence.get("execution")
    if not isinstance(execution, dict):
        fail("missing execution")
    if execution.get("class") != "LOCAL_FREE":
        fail("execution must remain LOCAL_FREE")
    if execution.get("double_fetch_required") is not True:
        fail("double-fetch gate missing")
    if execution.get("final_test_accessed") is not False:
        fail("final-test firewall violated")
    if execution.get("model_training_executed") is not False:
        fail("training must remain false")
    if execution.get("optimizer_updates") != 0:
        fail("optimizer updates must remain zero")
    if execution.get("paid_compute_used") is not False:
        fail("paid compute must remain false")
    for key in ("max_catalog_pages", "max_documents", "catalog_pages_observed", "documents_probed"):
        if not is_nonbool_int(execution.get(key)) or execution[key] < 0:
            fail(f"invalid execution integer: {key}")
    delay = execution.get("request_delay_seconds")
    if not isinstance(delay, (int, float)) or isinstance(delay, bool) or delay < 1.0:
        fail("request cadence is too aggressive")

    claims = evidence.get("claims")
    expected_claims = {
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "research_corpus_released": False,
        "learned_20m_claim": False,
    }
    if claims != expected_claims:
        fail("zero-credit scientific boundary changed")

    required = evidence.get("downstream_required")
    required_gates = {
        "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
        "RESERVED_EVALUATION_DECONTAMINATION",
        "POST_COMPOSITION_QUALITY_PRIVACY_BALANCE_FAMILY_CAPS",
        "CLUSTER_SAFE_SPLIT",
        "DETERMINISTIC_TOKENIZER_PACKING_DOUBLE_BUILD",
        "POSITIVE_EXACT_UNIQUE_CAUSAL_LOSS_LEDGER",
    }
    if not isinstance(required, list) or set(required) != required_gates:
        fail("downstream scientific gates changed")

    records = evidence.get("records")
    if not isinstance(records, list):
        fail("records must be a list")
    if execution["documents_probed"] != len(records):
        fail("documents_probed mismatch")

    accepted = 0
    accepted_bytes = 0
    digests: set[str] = set()
    rejected = Counter()
    for record in records:
        if not isinstance(record, dict):
            fail("record is not an object")
        if FORBIDDEN_TEXT_KEYS.intersection(record):
            fail("durable evidence contains raw text")
        unknown = set(record).difference(ALLOWED_RECORD_KEYS)
        if unknown:
            fail(f"unknown record keys: {sorted(unknown)}")
        url = record.get("url")
        if not isinstance(url, str):
            fail("record URL missing")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "www.president.gov.ua":
            fail("record left canonical origin")
        if not parsed.path.startswith("/documents/"):
            fail("record is not a document URL")
        accepted_flag = record.get("accepted")
        if not isinstance(accepted_flag, bool):
            fail("accepted flag must be boolean")
        if accepted_flag:
            for key in ("raw_sha256_a", "raw_sha256_b", "normalized_sha256"):
                if not isinstance(record.get(key), str) or not HEX64_RE.match(record[key]):
                    fail(f"accepted record missing digest: {key}")
            if record.get("reason") is not None:
                fail("accepted record has rejection reason")
            size = record.get("normalized_bytes")
            if not is_nonbool_int(size) or size < 1500:
                fail("accepted record size invalid")
            digest = record["normalized_sha256"]
            if digest in digests:
                fail("accepted normalized duplicate")
            digests.add(digest)
            accepted += 1
            accepted_bytes += size
        else:
            reason = record.get("reason")
            if not isinstance(reason, str) or not reason:
                fail("rejected record lacks reason")
            rejected[reason] += 1

    observed = evidence.get("observed_yield")
    if not isinstance(observed, dict):
        fail("observed_yield missing")
    if observed.get("accepted_documents") != accepted:
        fail("accepted document count mismatch")
    if observed.get("accepted_normalized_bytes") != accepted_bytes:
        fail("accepted byte count mismatch")
    if observed.get("rejected_documents") != len(records) - accepted:
        fail("rejected count mismatch")
    if observed.get("rejection_counts") != dict(sorted(rejected.items())):
        fail("rejection histogram mismatch")
    if observed.get("one_conservative_family") is not True:
        fail("family multiplication is forbidden")
    if evidence["status"] == "PASS_OBSERVED_YIELD" and accepted == 0:
        fail("positive status without yield")
    if evidence["status"] == "PASS_ZERO_YIELD" and accepted != 0:
        fail("zero-yield status with accepted records")

    expected_materialization = sha256(
        json.dumps(
            canonical_materialization_view(evidence),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    if evidence.get("materialization_identity_sha256") != expected_materialization:
        fail("materialization identity mismatch")

    evidence_view = dict(evidence)
    observed_evidence_identity = evidence_view.pop("evidence_identity_sha256", None)
    evidence_view.pop("generated_at_utc", None)
    evidence_view.pop("materialization_identity_sha256", None)
    # The producer computes evidence identity before adding either identity field.
    expected_evidence = sha256(
        json.dumps(
            evidence_view,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    if observed_evidence_identity != expected_evidence:
        fail("evidence identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence")
    parser.add_argument("--contract")
    args = parser.parse_args()
    if args.contract:
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        validate_contract(contract)
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    validate(evidence)
    print(
        "PASS",
        evidence["status"],
        evidence["observed_yield"]["accepted_documents"],
        evidence["observed_yield"]["accepted_normalized_bytes"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
