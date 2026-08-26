from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST = Path("configs/data/research_corpus_v1_predecontam_blocker_v1.json")
EXPECTED_SCHEMA = "12-6.research-corpus-v1-predecontam-blocker.v1"
EXPECTED_STATUS = "BLOCKED_WAIT_SOURCE_CONVERGENCE"
EXPECTED_DATA300_HEAD = "8ea7f830e50a23754d189dd4134f4afad76a7ee9"
EXPECTED_DATA300_ID = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
EXPECTED_DATA301_HEAD = "8820ba1b255f6bb95c7db0531fd846078a1aae01"
EXPECTED_DATA301_ID = "939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81"
EXPECTED_CONVERGENCE_ISSUE = 521
EXPECTED_CONVERGENCE_PR = 527
EXPECTED_CONVERGENCE_BASE = "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13"
EXPECTED_OBSERVED_HEAD = "481468a8cebcd82c96f4801062203d627e13ded4"


def canonical_sha256(doc: dict[str, Any]) -> str:
    payload = dict(doc)
    payload.pop("evidence_identity_sha256", None)
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_doc(doc: dict[str, Any]) -> None:
    _require(doc.get("schema_version") == EXPECTED_SCHEMA, "schema drift")
    _require(doc.get("repository") == "Oleksii-debug/12-6-ai.", "repository identity drift")
    _require(doc.get("execution_profile") == "LOCAL_FREE", "execution profile must remain LOCAL_FREE")
    _require(doc.get("status") == EXPECTED_STATUS, "blocked status weakened")
    _require(
        doc.get("evidence_identity_scope")
        == "SHA256(canonical JSON with evidence_identity_sha256 omitted)",
        "identity scope drift",
    )
    _require(
        doc.get("evidence_identity_sha256") == canonical_sha256(doc),
        "evidence self-hash mismatch",
    )

    d300 = doc["base_authorities"]["data300"]
    _require(d300.get("head_sha") == EXPECTED_DATA300_HEAD, "DATA-300 head drift")
    _require(
        d300.get("contract_identity_sha256") == EXPECTED_DATA300_ID,
        "DATA-300 contract identity drift",
    )
    _require(
        d300.get("corpus_state") == "NOT_BUILT_NOT_FROZEN_NOT_TERMINAL",
        "DATA-300 corpus truth weakened",
    )

    d301 = doc["base_authorities"]["data301"]
    _require(d301.get("head_sha") == EXPECTED_DATA301_HEAD, "DATA-301 head drift")
    _require(
        d301.get("evidence_identity_sha256") == EXPECTED_DATA301_ID,
        "DATA-301 evidence identity drift",
    )
    _require(d301.get("status") == "TERMINAL_BLOCKED", "DATA-301 state drift")
    _require(
        d301.get("corpus_identity") is None and d301.get("shard_identity") is None,
        "blocked DATA-301 cannot have corpus/shard identity",
    )
    _require(
        d301.get("authorized_balanced_no_replay_capacity") == 0,
        "DATA-301 capacity must remain zero",
    )

    conv = doc["required_source_convergence"]
    _require(conv.get("issue") == EXPECTED_CONVERGENCE_ISSUE, "source convergence issue drift")
    _require(conv.get("pull_request") == EXPECTED_CONVERGENCE_PR, "source convergence PR drift")
    _require(conv.get("base_head_sha") == EXPECTED_CONVERGENCE_BASE, "source convergence base drift")
    _require(
        conv.get("observed_head_sha") == EXPECTED_OBSERVED_HEAD,
        "source convergence observed-head drift",
    )
    _require(conv.get("pr_state") == "OPEN", "blocker snapshot expects open convergence PR")
    _require(
        conv.get("exact_head_ci_state") == "QUEUED_NONTERMINAL",
        "queue/nonterminal truth weakened",
    )
    _require(
        conv.get("terminal_authority_consumed") is False,
        "nonterminal convergence must not be consumed",
    )

    vector = conv[
        "reported_pre_successor_global_dedup_vector_non_authoritative_until_terminal"
    ]
    _require(
        vector.get("source_capacity_bytes") == 314140,
        "reported non-authoritative byte vector drift",
    )
    _require(
        vector.get("independent_families") == 10,
        "reported non-authoritative family vector drift",
    )
    _require(
        sum(item["bytes"] for item in vector["by_stratum"].values()) == 314140,
        "stratum byte vector does not sum",
    )
    _require(
        sum(item["families"] for item in vector["by_stratum"].values()) == 10,
        "stratum family vector does not sum",
    )

    freeze = doc["candidate_freeze"]
    _require(
        freeze.get("frozen") is False,
        "candidate must not be frozen before source convergence",
    )
    _require(
        freeze.get("candidate_set_digest_sha256") is None,
        "candidate digest must be null while blocked",
    )
    _require(
        freeze.get("record_inventory_digest_sha256") is None,
        "record inventory digest must be null while blocked",
    )
    _require(
        freeze.get("record_count") == 0 and freeze.get("records") == [],
        "no successor records may be hard-coded while blocked",
    )

    claims = doc["claim_boundary"]
    for key in (
        "corpus_frozen",
        "shards_published",
        "decontamination_executed",
        "final_test_payload_accessed",
        "final_test_outcomes_read",
        "tokenizer_fit_executed",
        "training_executed",
        "paid_compute_used",
        "long_training_authorized",
    ):
        _require(claims.get(key) is False, f"claim boundary weakened: {key}")
    _require(claims.get("optimizer_updates") == 0, "optimizer updates must remain zero")
    _require(
        claims.get("authorized_unique_optimized_targets") == 0,
        "pending source capacity must not become optimized-target authority",
    )

    gates = doc["downstream_gates"]
    _require(
        gates.get("reserved_evaluation_decontamination")
        == "NOT_PERMITTED_NO_FROZEN_CANDIDATE_IDENTITY",
        "decontamination gate weakened",
    )
    for key in ("corpus_materialization", "tokenizer_fit", "long_training", "paid_compute"):
        _require(gates.get(key) == "NOT_PERMITTED", f"downstream gate weakened: {key}")

    unblock = doc["unblock_rule"]
    _require(unblock.get("no_queue_as_pass") is True, "queue-as-pass guard missing")
    _require(
        unblock.get("no_pending_source_capacity_credit") is True,
        "pending capacity guard missing",
    )
    _require(
        unblock.get("no_replay_or_duplicate_quota_repair") is True,
        "replay guard missing",
    )


def main() -> int:
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_doc(doc)
    print(f"PASS {EXPECTED_STATUS} evidence={doc['evidence_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
