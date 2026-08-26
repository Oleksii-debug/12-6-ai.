#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPECTED_WORKER = "LEARN-235-EXTERNAL-REAL-1M"
EXPECTED_STATUS = "BLOCKED_NO_TERMINAL_LEARN234_IDENTITY"
EXPECTED_ENV151_SHA = "bbca2101ea9409b47d844dd8292cd7f2290e3ff0"
EXPECTED_DATA25 = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_1M_SPEC = "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07"


def canonical_payload(report: dict) -> bytes:
    payload = dict(report)
    payload.pop("report_sha256", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def validate(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(canonical_payload(report)).hexdigest()

    assert report["schema"] == "12-6.learn235-external-real-1m.blocker.v1"
    assert report["worker_id"] == EXPECTED_WORKER
    assert report["status"] == EXPECTED_STATUS
    assert report["report_sha256"] == digest
    assert report["base_authority"]["env151_sha"] == EXPECTED_ENV151_SHA
    assert report["base_authority"]["universal_bootstrap_required"] is True
    assert report["training_executed"] is False
    assert report["optimizer_updates"] == 0
    assert report["paid_compute"] is False
    assert report["foreign_pretrained_weights"] is False

    missing = report["required_missing_authority"]
    assert missing["worker_id"] == "LEARN-234-EXTERNAL-REAL-500K"
    assert missing["observed_at_cutoff"] == "NOT_PUBLISHED"
    required = set(missing["required_fields"])
    assert {
        "terminal_success_source_sha",
        "corpus_identity",
        "tokenizer_identity",
        "evaluation_identity",
        "optimized_token_schedule",
        "500k_best_checkpoint_identity",
        "500k_final_checkpoint_identity",
    } <= required

    upstream = set(report["upstream_missing_or_nonterminal"])
    assert "DATA-230-CORPUS-V03-EXTERNAL-REAL" in upstream
    assert "LEARN-234-EXTERNAL-REAL-500K" in upstream

    old = report["non_substitutable_existing_evidence"]
    assert old["data183"]["status"] == "CANDIDATE_UA_EN_REAL_PROJECT_CODE"
    assert old["data25_1m"]["corpus_identity"] == EXPECTED_DATA25
    assert old["data25_1m"]["model_spec_sha256"] == EXPECTED_1M_SPEC
    assert old["data25_1m"]["role"] == "cross_corpus_diagnostic_only"

    forbidden = report["forbidden_actions_taken"]
    assert not any(forbidden.values())

    return report


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "evidence/learn235/blocker.json"
    )
    report = validate(path)
    print(
        json.dumps(
            {
                "validation": "PASS",
                "status": report["status"],
                "training_executed": report["training_executed"],
                "optimizer_updates": report["optimizer_updates"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
