#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

EXPECTED = {
    "next100-028-php-doc-uk": {
        "historical_pr": 454,
        "historical_head": "a8fde2cb31db85b31a422e094f0c45ebd2658d39",
        "terminal_workflow_run": 33006932756,
        "terminal_artifact_id": 9623621921,
        "terminal_artifact_digest": (
            "sha256:f13ff28bd9bd998bbb8ebfb4a50893778f7edaea3f09ea052771bd1ba8b2c4e6"
        ),
        "terminal_evidence_identity_sha256": (
            "a591e8af0aa6aa9e3040c002087ea316485f38ed7d43ef30e127346498c02204"
        ),
        "source_commit": "c165db75cc6f81cfdabf754656e73a68940de46c",
        "family": "php.manual.documentation",
        "selected_objects": 10,
        "raw_bytes": 59986,
        "raw_bundle_sha256": "8b6e7235083095de65398eba4923aaf37a40808341ac04f8accfdd800e83a5d6",
        "normalized_bytes": 30510,
        "normalized_bundle_sha256": (
            "8802e3aa3d608305f8612efa9d90fdef65a76cdf518596fb52e3e7e5201db2e1"
        ),
    },
    "next100-030-rustbook-ua-oer": {
        "historical_pr": 478,
        "historical_head": "0bccb897ff72c577535d640e102009958a3d833d",
        "terminal_workflow_run": 33006780092,
        "terminal_artifact_id": 9623318910,
        "terminal_artifact_digest": (
            "sha256:87b9d3299f863450bff915f780de9f43f41428e6937147a2522cb7dfa7371e4d"
        ),
        "terminal_manifest_sha256": (
            "df9ab2943e140b8735c692eb4ee40c3f10dd03ca843ec9800e0bdf4d7ff664af"
        ),
        "source_commit": "ca2d2e4f4434c661836926017af23bdd40ad4e3d",
        "family": "rust-book.documentation.uk-translation",
        "selected_objects": 2,
        "raw_bytes": 20198,
        "raw_bundle_sha256": "a51faa914faae31a9c0254577667b1c5e62bf2ac2cde3d8db1115b5f563bc6bb",
        "normalized_bytes": 18165,
        "normalized_bundle_sha256": (
            "f2c8c39c2edef995e90e8c0be6a8abc1b278ce98b534adf6e912c0c717a9cb1c"
        ),
    },
}
REQUIRED_GATES = [
    "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
    "FRESH_RESERVED_EVALUATION_DECONTAMINATION",
    "POST_COMPOSITION_QUALITY_PRIVACY",
    "BALANCE_AND_FAMILY_CAPS",
    "CLUSTER_SAFE_SPLIT",
    "DETERMINISTIC_PACKING_TWO_CLEAN_BUILDS",
    "POSITIVE_EXACT_UNIQUE_CAUSAL_LOSS_LEDGER",
]


def canonical_without_identity(obj: dict[str, Any]) -> bytes:
    clone = dict(obj)
    clone.pop("authority_identity_sha256", None)
    rendered = json.dumps(
        clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (rendered + "\n").encode()


def validate(obj: dict[str, Any]) -> None:
    if obj.get("schema_version") != "12-6.d03-terminal-ua-source-seals.v1":
        raise ValueError("schema mismatch")
    if obj.get("worker_id") != "D03-UA-TERMINAL-SOURCE-SEALS-CURRENT-MAIN":
        raise ValueError("worker mismatch")
    if obj.get("local_free_only") is not True:
        raise ValueError("LOCAL_FREE boundary weakened")
    if obj.get("authority_state") != "TERMINAL_SOURCE_AUTHORITIES_ZERO_CANONICAL_CREDIT":
        raise ValueError("authority state mismatch")

    identity = obj.get("authority_identity_sha256")
    if not isinstance(identity, str) or not SHA64.fullmatch(identity):
        raise ValueError("invalid authority identity")
    if hashlib.sha256(canonical_without_identity(obj)).hexdigest() != identity:
        raise ValueError("authority self-identity mismatch")

    sources = obj.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("expected exactly two terminal sources")
    by_id = {s.get("source_id"): s for s in sources if isinstance(s, dict)}
    if set(by_id) != set(EXPECTED):
        raise ValueError("source set drift")

    families: set[str] = set()
    normalized_total = 0
    for sid, exp in EXPECTED.items():
        src = by_id[sid]
        for key, value in exp.items():
            if src.get(key) != value:
                raise ValueError(f"{sid}: {key} drift")
        if src.get("terminal_workflow_conclusion") != "success":
            raise ValueError(f"{sid}: nonterminal workflow")
        if (
            src.get("terminal_verdict") != "ADMIT"
            or src.get("historical_gates_all_passed") is not True
        ):
            raise ValueError(f"{sid}: terminal admission missing")
        if src.get("language") != "uk":
            raise ValueError(f"{sid}: language drift")
        if src.get("model_training_rights") != "ALLOWED":
            raise ValueError(f"{sid}: training rights weakened")
        if src.get("evaluation") != "NOT_SEPARATELY_ADMITTED":
            raise ValueError(f"{sid}: evaluation firewall weakened")
        if (
            not SHA40.fullmatch(src["historical_head"])
            or not SHA40.fullmatch(src["source_commit"])
        ):
            raise ValueError(f"{sid}: malformed git identity")
        if not DIGEST.fullmatch(src["terminal_artifact_digest"]):
            raise ValueError(f"{sid}: malformed artifact digest")
        for key in ("raw_bundle_sha256", "normalized_bundle_sha256"):
            if not SHA64.fullmatch(src[key]):
                raise ValueError(f"{sid}: malformed {key}")
        families.add(src["family"])
        normalized_total += src["normalized_bytes"]

    if len(families) != 2:
        raise ValueError("families are not independent at source-authority layer")

    agg = obj.get("aggregate", {})
    expected_agg = {
        "source_count": 2,
        "independent_family_count": 2,
        "normalized_source_bytes": normalized_total,
        "canonical_capacity_credit_bytes": 0,
        "canonical_family_credit": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
    }
    if agg != expected_agg:
        raise ValueError("aggregate or zero-credit boundary mismatch")

    truth = obj.get("current_main_truth_boundary", {})
    false_fields = (
        "canonical_corpus_admitted",
        "current_global_dedup_executed_for_these_sources",
        "fresh_reserved_eval_decontamination_executed",
        "post_composition_quality_privacy_executed",
        "split_packed_two_clean_builds",
        "positive_unique_loss_ledger",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_accessed",
        "paid_compute_used",
        "learned_20m_promoted",
    )
    if any(truth.get(k) is not False for k in false_fields):
        raise ValueError("current-main truth boundary promoted")
    if truth.get("optimizer_updates") != 0:
        raise ValueError("optimizer updates must remain zero")
    if obj.get("required_downstream_gates") != REQUIRED_GATES:
        raise ValueError("downstream gate order drift")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default="configs/data/d03_terminal_ua_source_seals_v1.json"
    )
    args = ap.parse_args()
    obj = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate(obj)
    print("D03_UA_TERMINAL_SOURCE_SEALS=PASS")
    print("D03_UA_TERMINAL_SOURCE_BYTES=48675")
    print("D03_UA_TERMINAL_SOURCE_FAMILIES=2")
    print("D03_CURRENT_CANONICAL_CREDIT_BYTES=0")
    print("D03_TRAINING_AUTHORIZED_BYTES=0")


if __name__ == "__main__":
    main()
