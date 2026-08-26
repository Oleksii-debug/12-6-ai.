#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/next100_064/unique_loss_ledger_v2_authority.json"
SCHEMA = "12-6.next100-064-unique-loss-ledger-v2-authority.v1"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_obj(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    require(evidence.get("schema_version") == SCHEMA, "schema mismatch")
    require(
        evidence.get("worker_id") == "NEXT100-064-UNIQUE-LOSS-LEDGER-V2",
        "worker mismatch",
    )
    require(evidence.get("execution_profile") == "LOCAL_FREE", "profile mismatch")
    require(evidence.get("training_executed") is False, "training must be false")

    observed = evidence.get("evidence_identity_sha256")
    require(isinstance(observed, str) and len(observed) == 64, "missing self hash")
    unhashed = dict(evidence)
    unhashed.pop("evidence_identity_sha256", None)
    require(sha256_obj(unhashed) == observed, "evidence self hash mismatch")

    verdict = evidence.get("verdict")
    require(
        verdict == "BLOCKED_NO_TERMINAL_POSTPACK_CORPUS_MATERIALIZATION",
        "unexpected verdict",
    )
    postpack = evidence.get("exact_postpack_one_pass_maximum")
    require(isinstance(postpack, dict), "missing postpack report")
    require(postpack.get("status") == "NOT_MATERIALIZED", "postpack status drift")
    require(postpack.get("total") is None, "must not invent postpack total")
    require(postpack.get("by_language") is None, "must not invent language totals")
    require(postpack.get("by_modality") is None, "must not invent modality totals")
    require(postpack.get("by_family") is None, "must not invent family totals")

    safe = evidence.get("training_authorized_exposure_now")
    require(isinstance(safe, dict), "missing safe exposure report")
    require(safe.get("total") == 0, "safe authorized exposure must be zero")
    require(
        all(value == 0 for value in safe.get("by_language", {}).values()),
        "nonzero language exposure without terminal ledger",
    )
    require(
        all(value == 0 for value in safe.get("by_modality", {}).values()),
        "nonzero modality exposure without terminal ledger",
    )
    require(
        all(value == 0 for value in safe.get("by_family", {}).values()),
        "nonzero family exposure without terminal ledger",
    )

    diagnostic = evidence.get("historical_prebuild_diagnostic")
    require(isinstance(diagnostic, dict), "missing historical diagnostic")
    require(diagnostic.get("is_training_authority") is False, "diagnostic promoted")
    require(
        diagnostic.get("source_bytes_relabelled_as_loss_positions") is False,
        "source-byte relabeling prohibited",
    )
    require(diagnostic.get("total") == 183056, "historical diagnostic total drift")
    require(sum(diagnostic["by_language"].values()) == 183056, "language sum drift")
    require(sum(diagnostic["by_modality"].values()) == 183056, "modality sum drift")
    require(sum(diagnostic["by_family"].values()) == 183056, "family sum drift")

    terminal_sources = evidence.get("terminal_source_authority_vector")
    require(isinstance(terminal_sources, list) and terminal_sources, "missing sources")
    for item in terminal_sources:
        require(item.get("terminal") is True, "nonterminal source in terminal vector")
        require(item.get("training_source_admitted") is True, "source is not admitted")
        require(
            item.get("eligible_for_postpack_loss_ledger") is False,
            "uncomposed source promoted to loss ledger",
        )
        head = item.get("head_sha")
        require(isinstance(head, str) and len(head) == 40, "source head SHA invalid")

    blockers = evidence.get("blocking_authorities")
    require(isinstance(blockers, list) and len(blockers) >= 2, "missing blocker vector")
    require(
        any(item.get("worker_id") == "NEXT100-066-DECONTAMINATION-V4" for item in blockers),
        "latest decontamination blocker not consumed",
    )

    print("NEXT100-064 UNIQUE LOSS LEDGER V2 AUTHORITY: PASS")
    print(f"evidence_identity_sha256={observed}")
    print("authorized_training_exposure=0")
    print("exact_postpack_one_pass_maximum=NOT_MATERIALIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
