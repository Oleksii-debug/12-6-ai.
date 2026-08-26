#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-031 English Wikipedia source authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "configs/data/next100_031_en_wikipedia_source_authority_v1.json"
CANDIDATES = ROOT / "configs/data/external_source_candidates_ua_en_v1.json"
REGISTRY = ROOT / "data/registry/external_snapshots.v2.json"


def canonical_sha256(value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    claimed = authority.pop("authority_identity_sha256")
    assert canonical_sha256(authority) == claimed, "authority identity drift"
    assert authority["schema_version"] == "12-6.en-wikipedia-source-authority.v1"
    assert authority["worker_id"] == "NEXT100-031-DATA-EN-WIKIPEDIA"
    assert authority["terminal"] is True
    assert authority["terminal_state"] == "REJECT"
    assert authority["local_free_only"] is True

    materialization = authority["bounded_materialization"]
    assert materialization["status"] == "NOT_RUN_RIGHTS_GATE_FAILED"
    assert materialization["network_bytes_downloaded"] == 0
    assert materialization["raw_sha256"] is None
    assert materialization["normalized_sha256"] is None

    decisions = authority["purpose_decisions"]
    assert decisions["model_training"]["status"] == "REJECT"
    assert decisions["redistribution"]["status"] == "NOT_ADMITTED"
    assert decisions["evaluation"]["status"] == "NOT_SEPARATELY_ADMITTED"

    dump = authority["source"]["authoritative_dump"]
    assert dump["dump_date"] == "20260801"
    assert dump["candidate_file"] == "enwiki-20260801-pages-articles1.xml-p1p41242.bz2"
    assert dump["candidate_file_sha1"] == "97ea1ad5a871e951ddadaaf199f69b1ddf121b34"

    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    enwiki = [item for item in candidates["sources"] if item["source_id"] == "en.wikipedia.enwiki"]
    assert len(enwiki) == 1
    candidate = enwiki[0]
    assert candidate["eligibility_status"] == "BLOCKED_BY_RIGHTS"
    assert candidate["rights"]["allows_model_training"] is False
    assert candidate["rights"]["license_id"] == "CC-BY-SA-4.0"
    assert "ShareAlike compliance policy" in candidate["block_reason"]
    assert candidate["acquisition_urls"] == []
    assert candidate["adapter"] is None

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    en_families = {
        item["independent_source_family"]["family_id"]
        for item in registry["sources"]
        if item["language"] == "en"
    }
    assert en_families == {"en.standardebooks.manual"}, en_families
    assert all(item["source_id"] != "en.wikipedia.enwiki" for item in registry["sources"])
    assert registry["claim_boundary"]["representative_corpus_claimed"] is False

    print(f"NEXT100-031 terminal REJECT verified: {claimed}")
    print("No Wikipedia bytes fetched; evaluation/final-test material not consumed.")


if __name__ == "__main__":
    main()
