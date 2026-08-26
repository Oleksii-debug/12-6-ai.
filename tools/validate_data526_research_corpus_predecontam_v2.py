from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST = Path("configs/data/research_corpus_v1_predecontam_blocker_v2.json")
EXPECTED_SCHEMA = "12-6.research-corpus-v1-predecontam-blocker.v2"
EXPECTED_STATUS = "BLOCKED_WAIT_SOURCE_AND_GLOBAL_DEDUP_TERMINAL"
EXPECTED_DATA300_HEAD = "8ea7f830e50a23754d189dd4134f4afad76a7ee9"
EXPECTED_DATA300_ID = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
EXPECTED_DATA301_HEAD = "8820ba1b255f6bb95c7db0531fd846078a1aae01"
EXPECTED_DATA301_ID = "939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81"
EXPECTED_SOURCE_PR = 538
EXPECTED_SOURCE_HEAD = "226cbc26710a75af4a864576220b270089e7c52b"
EXPECTED_SOURCE_ID = "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58"
EXPECTED_DEDUP_PR = 632
EXPECTED_DEDUP_HEAD = "8181b247fc305f96f4be02d8630ce18cdcf63eae"
EXPECTED_DEDUP_RUN = 33008762043
EXPECTED_TOTAL = 2_045_180
EXPECTED_ENVELOPE = 2_047_541
EXPECTED_UNCREDITED = 2_361
EXPECTED_OBJECTS = 31
EXPECTED_FAMILIES = {"uk": 4, "en": 5, "code": 5}
EXPECTED_BYTES = {"uk": 100_856, "en": 1_838_293, "code": 106_031}


def canonical_sha256(doc: dict[str, Any]) -> str:
    payload = dict(doc)
    payload.pop("evidence_identity_sha256", None)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_doc(doc: dict[str, Any]) -> None:
    _require(doc.get("schema_version") == EXPECTED_SCHEMA, "schema drift")
    _require(doc.get("worker_id") == "DATA-526-RESEARCH-CORPUS-V1-PREDECONTAM-V2", "worker drift")
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

    source = doc["source_convergence_candidate"]
    _require(source.get("pull_request") == EXPECTED_SOURCE_PR, "source PR drift")
    _require(source.get("head_sha") == EXPECTED_SOURCE_HEAD, "source head drift")
    _require(
        source.get("registry_path")
        == "configs/data/next100_063_terminal_source_registry_v4.json",
        "source registry path drift",
    )
    _require(
        source.get("registry_schema_version")
        == "12-6.next100-063-terminal-source-registry.v4",
        "source registry schema drift",
    )
    _require(
        source.get("registry_identity_sha256") == EXPECTED_SOURCE_ID,
        "source registry identity drift",
    )
    _require(
        source.get("exact_head_ci_state") == "QUEUED_NONTERMINAL",
        "source queue/nonterminal truth weakened",
    )
    _require(
        source.get("terminal_authority_consumed") is False,
        "nonterminal source authority must not be consumed",
    )

    vector = source["pre_global_dedup_vector"]
    _require(vector.get("source_object_count") == EXPECTED_OBJECTS, "source object count drift")
    _require(
        vector.get("numeric_training_capacity_bytes") == EXPECTED_TOTAL,
        "source numeric capacity drift",
    )
    _require(
        vector.get("normalized_source_envelope_bytes") == EXPECTED_ENVELOPE,
        "source normalized envelope drift",
    )
    _require(
        vector.get("uncredited_normalized_bytes") == EXPECTED_UNCREDITED,
        "source uncredited byte drift",
    )
    _require(vector.get("independent_families") == 14, "source family count drift")
    _require(
        vector.get("authorized_balanced_no_replay_loss_positions") == 0,
        "source bytes illegally promoted to loss positions",
    )
    by = vector["by_stratum"]
    _require(set(by) == set(EXPECTED_BYTES), "source stratum set drift")
    for stratum in EXPECTED_BYTES:
        _require(
            by[stratum].get("numeric_training_capacity_bytes") == EXPECTED_BYTES[stratum],
            f"{stratum} source capacity drift",
        )
        _require(
            by[stratum].get("families") == EXPECTED_FAMILIES[stratum],
            f"{stratum} family count drift",
        )
    _require(
        sum(row["numeric_training_capacity_bytes"] for row in by.values()) == EXPECTED_TOTAL,
        "source stratum bytes do not sum",
    )
    _require(
        sum(row["families"] for row in by.values()) == 14,
        "source stratum family counts do not sum",
    )
    _require(
        EXPECTED_ENVELOPE - EXPECTED_TOTAL == EXPECTED_UNCREDITED,
        "source envelope/uncredited arithmetic drift",
    )

    dedup = doc["global_dedup_candidate"]
    _require(dedup.get("pull_request") == EXPECTED_DEDUP_PR, "dedup PR drift")
    _require(dedup.get("head_sha") == EXPECTED_DEDUP_HEAD, "dedup head drift")
    _require(
        dedup.get("worker_id") == "NEXT100-065D-CROSSSOURCE-DEDUP-V6",
        "dedup worker drift",
    )
    _require(
        dedup.get("config_path")
        == "configs/data/next100_065d_cross_source_dedup_v6.json",
        "dedup config path drift",
    )
    _require(dedup.get("dedicated_workflow_run") == EXPECTED_DEDUP_RUN, "dedup run drift")
    _require(
        dedup.get("exact_head_ci_state") == "PENDING_NONTERMINAL",
        "dedup pending/nonterminal truth weakened",
    )
    _require(
        dedup.get("terminal_authority_consumed") is False,
        "nonterminal dedup authority must not be consumed",
    )
    _require(
        dedup.get("expected_pre_dedup_source_object_count") == EXPECTED_OBJECTS,
        "dedup expected source count drift",
    )
    _require(
        dedup.get("expected_pre_dedup_numeric_capacity_bytes") == EXPECTED_TOTAL,
        "dedup expected capacity drift",
    )
    _require(
        dedup.get("expected_pre_dedup_family_counts") == EXPECTED_FAMILIES,
        "dedup expected family vector drift",
    )
    for key in (
        "post_dedup_report_sha256",
        "post_dedup_unique_capacity_bytes",
        "post_dedup_record_inventory_digest_sha256",
    ):
        _require(dedup.get(key) is None, f"nonterminal dedup cannot publish {key}")

    _require(
        dedup["expected_pre_dedup_numeric_capacity_bytes"]
        == source["pre_global_dedup_vector"]["numeric_training_capacity_bytes"],
        "source registry/dedup composed-byte vector mismatch",
    )
    _require(
        dedup["expected_pre_dedup_source_object_count"]
        == source["pre_global_dedup_vector"]["source_object_count"],
        "source registry/dedup source-object vector mismatch",
    )
    _require(
        dedup["expected_pre_dedup_family_counts"]
        == {key: value["families"] for key, value in by.items()},
        "source registry/dedup family vector mismatch",
    )

    freeze = doc["candidate_freeze"]
    _require(freeze.get("frozen") is False, "candidate must not be frozen before terminal global dedup")
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
        "no post-dedup records may be hard-coded while blocked",
    )

    claims = doc["claim_boundary"]
    for key in (
        "source_bytes_are_training_tokens",
        "corpus_frozen",
        "global_dedup_terminal",
        "decontamination_executed",
        "final_test_payload_accessed",
        "final_test_outcomes_read",
        "shards_published",
        "tokenizer_fit_executed",
        "training_executed",
        "long_training_authorized",
        "paid_compute_used",
    ):
        _require(claims.get(key) is False, f"claim boundary weakened: {key}")
    _require(claims.get("optimizer_updates") == 0, "optimizer updates must remain zero")
    _require(
        claims.get("authorized_unique_optimized_targets") == 0,
        "source bytes must not become optimized-target authority",
    )

    gates = doc["downstream_gates"]
    _require(
        gates.get("source_convergence") == "WAIT_EXACT_HEAD_TERMINAL_SUCCESS",
        "source convergence gate weakened",
    )
    _require(
        gates.get("global_exact_near_fragment_lineage_dedup")
        == "WAIT_EXACT_HEAD_TERMINAL_SUCCESS",
        "global dedup gate weakened",
    )
    _require(
        gates.get("record_inventory_freeze")
        == "NOT_PERMITTED_NO_TERMINAL_GLOBAL_DEDUP",
        "record freeze gate weakened",
    )
    _require(
        gates.get("reserved_evaluation_decontamination")
        == "NOT_PERMITTED_NO_FROZEN_POST_DEDUP_CANDIDATE_IDENTITY",
        "decontamination gate weakened",
    )
    for key in (
        "post_composition_quality_privacy",
        "balance_diversity_family_caps",
        "cluster_safe_split_pack",
        "two_clean_build_proof",
        "postpack_unique_loss_ledger",
        "tokenizer_fit",
        "long_training",
        "paid_compute",
    ):
        _require(gates.get(key) == "NOT_PERMITTED", f"downstream gate weakened: {key}")

    unblock = doc["unblock_rule"]
    _require(unblock.get("no_queue_or_pending_as_pass") is True, "queue/pending-as-pass guard missing")
    _require(
        unblock.get("no_source_bytes_as_token_or_loss_capacity") is True,
        "byte/token firewall missing",
    )
    _require(
        unblock.get("no_replay_or_duplicate_quota_repair") is True,
        "replay guard missing",
    )
    _require(
        unblock.get("source_registry_and_dedup_must_bind_same_composed_vector") is True,
        "source/dedup vector binding missing",
    )
    required = unblock.get("required_order")
    _require(isinstance(required, list) and len(required) == 5, "required order drift")
    _require("global dedup" in required[1].lower(), "global dedup must precede DATA-526 freeze")
    _require("freeze" in required[2].lower(), "record freeze step missing after global dedup")
    _require("decontamination" in required[3].lower(), "decontamination handoff missing")


def main() -> int:
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_doc(doc)
    print(f"PASS {EXPECTED_STATUS} evidence={doc['evidence_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
