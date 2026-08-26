#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "configs/data/next100_034_nist_technical_series_authority_v1.json"

EXPECTED = {
    "NIST.SP.800-204": {
        "raw_bytes": 814054,
        "raw_sha256": "25412c860165e5ee1cfbf26ed47c56f4d213b1996a73365f5561be6403cf7588",
        "normalized_utf8_bytes": 19668,
        "normalized_sha256": "570e8d75b6dc6aefee1f089818b46765c0dd1965e06947bcc2fff0169d22274e",
        "doi": "10.6028/NIST.SP.800-204",
    },
    "NIST.SP.800-204C": {
        "raw_bytes": 717082,
        "raw_sha256": "d51133dc55a804990d80ba4b9c35e3fbb2d5acdf7b330b66edeaae59fc63d69b",
        "normalized_utf8_bytes": 19736,
        "normalized_sha256": "558da6a0886036a01a5139d635b1352b5cf5d74655d919c66a04e84f2d49c0fe",
        "doi": "10.6028/NIST.SP.800-204C",
    },
    "NIST.SP.800-215": {
        "raw_bytes": 1089318,
        "raw_sha256": "159e17820a0a337c4a7e9c7ee8b966823e81dc72f5c6229e7d7244c40b0b1645",
        "normalized_utf8_bytes": 19954,
        "normalized_sha256": "6c99c3b14ee3ea7fe915940e38c080dbf2a785f1abcee2fd73e7fd731424770d",
        "doi": "10.6028/NIST.SP.800-215",
    },
}


def fail(msg: str) -> None:
    raise SystemExit(f"NEXT100-034 validation failure: {msg}")


def main() -> None:
    data = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if data["schema_version"] != "12-6.next100-034-nist-technical-series-authority.v1":
        fail("schema drift")
    if data["worker_id"] != "NEXT100-034-DATA-EN-NIST":
        fail("worker drift")
    if data["authority_status"] != "ADMIT_BOUNDED_SUBSET_WITH_RETESTS":
        fail("terminal authority drift")
    if data["local_free_only"] is not True:
        fail("LOCAL_FREE boundary weakened")

    fam = data["family"]
    if fam["family_id"] != "en.usgov.nist.technical-series" or fam["independent_family_count"] != 1:
        fail("family identity/accounting drift")
    if fam["family_accounting"] != "ALL_THREE_OBJECTS_COUNT_AS_ONE_NIST_TECHNICAL_SERIES_FAMILY":
        fail("family inflation risk")

    evidence = ROOT / data["rights_evidence"]["repo_path"]
    actual_evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if actual_evidence_sha != data["rights_evidence"]["sha256"]:
        fail("rights evidence hash mismatch")

    docs = data["admit"]
    if {d["publication_id"] for d in docs} != set(EXPECTED):
        fail("admitted publication set drift")
    if len(docs) != 3:
        fail("bounded subset must contain exactly three publications")

    total_raw = 0
    total_norm = 0
    seen_raw = set()
    seen_norm = set()
    for doc in docs:
        pid = doc["publication_id"]
        expected = EXPECTED[pid]
        for key, value in expected.items():
            if doc[key] != value:
                fail(f"{pid}: {key} drift")
        if doc["author"] != "Ramaswamy Chandramouli (NIST)":
            fail(f"{pid}: sole NIST author binding drift")
        if doc["language"] != "en" or doc["modality"] != "text":
            fail(f"{pid}: language/modality drift")
        rights = doc["rights"]
        if rights["us_copyright"] != "NOT_SUBJECT_TO_COPYRIGHT_IN_US_DOCUMENT_SPECIFIC":
            fail(f"{pid}: document-specific copyright gate failed")
        if not rights["model_training"].startswith("ALLOWED"):
            fail(f"{pid}: training purpose not admitted")
        if not rights["redistribution"].startswith("ALLOWED"):
            fail(f"{pid}: redistribution not admitted")
        if rights["evaluation"] != "NOT_SEPARATELY_ADMITTED":
            fail(f"{pid}: evaluation boundary weakened")
        q = doc["quality"]
        if q["words"] < data["quality"]["minimum_words_per_document"]:
            fail(f"{pid}: word-count quality gate")
        if q["alphabetic_char_ratio"] < data["quality"]["minimum_alphabetic_char_ratio"]:
            fail(f"{pid}: alphabetic ratio gate")
        if q["english_stopword_ratio"] < data["quality"]["minimum_english_stopword_ratio"]:
            fail(f"{pid}: English ratio gate")
        if q["unicode_replacement_chars"] != 0 or q["unexpected_control_chars"] != 0:
            fail(f"{pid}: extraction corruption gate")
        if doc["privacy"]["private_or_user_generated_source"] is not False:
            fail(f"{pid}: privacy/source-kind drift")
        if doc["emails_redacted"] != 0 or doc["privacy"]["emails_redacted"] != 0:
            fail(f"{pid}: unexpected contact leakage result")
        total_raw += doc["raw_bytes"]
        total_norm += doc["normalized_utf8_bytes"]
        seen_raw.add(doc["raw_sha256"])
        seen_norm.add(doc["normalized_sha256"])

    if len(seen_raw) != 3 or len(seen_norm) != 3:
        fail("exact duplicate within NIST subset")
    if total_raw != 2620454 or total_norm != 59358:
        fail("bounded byte totals drift")
    if total_norm != data["subset"]["normalized_utf8_bytes"] or total_raw != data["subset"]["raw_bytes"]:
        fail("subset byte report mismatch")
    if any(d["normalized_utf8_bytes"] > data["normalization"]["max_normalized_utf8_bytes_per_document"] for d in docs):
        fail("normalization bound exceeded")

    dedup = data["dedup"]
    if dedup["exact_normalized_collision_with_terminal_baseline"] is not False:
        fail("baseline exact-collision status drift")
    baseline_hashes = set(dedup["terminal_registry_baseline"]["normalized_hashes"].values())
    if seen_norm & baseline_hashes:
        fail("NIST exact normalized hash collides with terminal baseline")
    if len(baseline_hashes) != 5:
        fail("terminal baseline hash inventory incomplete")
    if any(row["overlap"] >= dedup["intra_subset_fail_threshold"] for row in dedup["pairwise"]):
        fail("within-family near-duplicate gate")
    if dedup["canonical_cross_source_near_dedup"] != "REQUIRED_BEFORE_CORPUS_INTEGRATION":
        fail("canonical cross-source dedup requirement weakened")

    composition = data["composition_projection"]
    if composition["projected_nist_global_share"] > composition["data295_global_family_cap"]:
        fail("projected global family cap exceeded")
    if composition["projected_nist_english_stratum_share"] > composition["data295_within_stratum_family_cap"]:
        fail("projected English-stratum cap exceeded")
    if composition["claim_boundary"] != "PROJECTION_ONLY_NOT_CORPUS_ADMISSION":
        fail("composition projection overclaimed")

    retest = {row["scope"] for row in data["retest"]}
    required_retests = {"NIST.SP.800-204A", "NIST.SP.800-204B", "NIST.SP.800-204D", "NIST_STANDARD_REFERENCE_DATA"}
    if not required_retests.issubset(retest):
        fail("third-party/SRD RETEST boundary weakened")
    rejects = {row["scope"] for row in data["reject"]}
    if "ALL_NIST_MATERIAL_BLANKET_PUBLIC_DOMAIN_RULE" not in rejects:
        fail("blanket NIST public-domain rejection missing")
    if data["rights_policy"]["blanket_nist_public_domain_assumption"] != "REJECT":
        fail("blanket NIST public-domain rule reintroduced")
    if data["subset"]["corpus_integration_status"] != "NOT_INTEGRATED_REQUIRES_SUCCESSOR_CORPUS_CONTRACT_AND_CANONICAL_DEDUP":
        fail("frozen corpus boundary weakened")

    print("NEXT100-034 PASS")
    print(f"admit_publications={len(docs)} family_count=1 normalized_utf8_bytes={total_norm}")
    print("evaluation_admitted=0 corpus_integrated=0 local_free_only=true")


if __name__ == "__main__":
    main()
