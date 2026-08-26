from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST = Path("configs/data/research_corpus_v1_predecontam_blocker_v3.json")
EXPECTED_SCHEMA = "12-6.research-corpus-v1-predecontam-blocker.v3"
EXPECTED_STATUS = "BLOCKED_WAIT_SOURCE_AND_GLOBAL_DEDUP_TERMINAL"
EXPECTED_SOURCE_HEAD = "10342d590d91b6999c42515cdf87fe31e2355844"
EXPECTED_SOURCE_BLOB = "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf"
EXPECTED_SOURCE_RUN = 33010079393
EXPECTED_DEDUP_HEAD = "055d45ae08e055ecaeba638c7eaed5b41720e1bf"
EXPECTED_DEDUP_RUN = 33012342294
EXPECTED_TOTAL = 2_215_615
EXPECTED_ENVELOPE = 2_217_976
EXPECTED_UNCREDITED = 2_361
EXPECTED_OBJECTS = 35
EXPECTED_FAMILIES = {"uk": 4, "en": 5, "code": 6}
EXPECTED_BYTES = {"uk": 100_856, "en": 1_838_293, "code": 276_466}


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
    _require(
        doc.get("worker_id") == "DATA-526-RESEARCH-CORPUS-V1-PREDECONTAM-V3",
        "worker drift",
    )
    _require(doc.get("repository") == "Oleksii-debug/12-6-ai.", "repository drift")
    _require(doc.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
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

    source = doc["source_convergence_candidate"]
    _require(source.get("pull_request") == 538, "source PR drift")
    _require(source.get("head_sha") == EXPECTED_SOURCE_HEAD, "source head drift")
    _require(
        source.get("registry_path")
        == "configs/data/next100_063_terminal_source_registry_v5.json",
        "source registry path drift",
    )
    _require(
        source.get("registry_schema_version")
        == "12-6.next100-063-terminal-source-registry.v5",
        "source registry schema drift",
    )
    _require(
        source.get("registry_git_blob_sha1") == EXPECTED_SOURCE_BLOB,
        "source registry blob drift",
    )
    _require(
        source.get("base_v4_registry_identity_sha256")
        == "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58",
        "base V4 registry identity drift",
    )
    _require(source.get("exact_head_ci_run") == EXPECTED_SOURCE_RUN, "source run drift")
    _require(
        source.get("exact_head_ci_state") == "QUEUED_NONTERMINAL",
        "source nonterminal truth weakened",
    )
    _require(
        source.get("terminal_authority_consumed") is False,
        "nonterminal source authority consumed",
    )

    vector = source["pre_global_dedup_vector"]
    _require(vector.get("source_object_count") == EXPECTED_OBJECTS, "source object drift")
    _require(
        vector.get("numeric_training_capacity_bytes") == EXPECTED_TOTAL,
        "source capacity drift",
    )
    _require(
        vector.get("normalized_source_envelope_bytes") == EXPECTED_ENVELOPE,
        "source envelope drift",
    )
    _require(
        vector.get("uncredited_normalized_bytes") == EXPECTED_UNCREDITED,
        "source uncredited bytes drift",
    )
    _require(vector.get("independent_families") == 15, "source family total drift")
    _require(
        vector.get("authorized_balanced_no_replay_loss_positions") == 0,
        "source bytes promoted to loss positions",
    )
    by = vector["by_stratum"]
    _require(set(by) == set(EXPECTED_BYTES), "source stratum set drift")
    for stratum in EXPECTED_BYTES:
        _require(
            by[stratum].get("numeric_training_capacity_bytes") == EXPECTED_BYTES[stratum],
            f"{stratum} bytes drift",
        )
        _require(
            by[stratum].get("families") == EXPECTED_FAMILIES[stratum],
            f"{stratum} family drift",
        )
    _require(
        sum(row["numeric_training_capacity_bytes"] for row in by.values()) == EXPECTED_TOTAL,
        "source bytes do not sum",
    )
    _require(
        sum(row["families"] for row in by.values()) == 15,
        "source families do not sum",
    )
    _require(
        EXPECTED_ENVELOPE - EXPECTED_TOTAL == EXPECTED_UNCREDITED,
        "envelope arithmetic drift",
    )

    dedup = doc["global_dedup_candidate"]
    _require(dedup.get("pull_request") == 632, "dedup PR drift")
    _require(dedup.get("head_sha") == EXPECTED_DEDUP_HEAD, "dedup head drift")
    _require(
        dedup.get("worker_id") == "NEXT100-065E-CROSSSOURCE-DEDUP-V7",
        "dedup worker drift",
    )
    _require(
        dedup.get("config_path")
        == "configs/data/next100_065e_cross_source_dedup_v7.json",
        "dedup config drift",
    )
    _require(
        dedup.get("dedicated_workflow_name") == "NEXT100-065E Cross-Source Dedup V7",
        "dedup workflow name drift",
    )
    _require(dedup.get("dedicated_workflow_run") == EXPECTED_DEDUP_RUN, "dedup run drift")
    _require(
        dedup.get("exact_head_ci_state") == "QUEUED_NONTERMINAL",
        "dedup nonterminal truth weakened",
    )
    _require(
        dedup.get("terminal_authority_consumed") is False,
        "nonterminal dedup authority consumed",
    )
    _require(
        dedup.get("expected_pre_dedup_source_object_count") == EXPECTED_OBJECTS,
        "dedup source object drift",
    )
    _require(
        dedup.get("expected_pre_dedup_numeric_capacity_bytes") == EXPECTED_TOTAL,
        "dedup capacity drift",
    )
    _require(
        dedup.get("expected_pre_dedup_family_counts") == EXPECTED_FAMILIES,
        "dedup family vector drift",
    )
    for key in (
        "post_dedup_report_sha256",
        "post_dedup_unique_capacity_bytes",
        "post_dedup_record_inventory_digest_sha256",
    ):
        _require(dedup.get(key) is None, f"nonterminal dedup cannot publish {key}")

    _require(
        dedup["expected_pre_dedup_numeric_capacity_bytes"]
        == vector["numeric_training_capacity_bytes"],
        "source/dedup byte vector mismatch",
    )
    _require(
        dedup["expected_pre_dedup_source_object_count"] == vector["source_object_count"],
        "source/dedup object vector mismatch",
    )
    _require(
        dedup["expected_pre_dedup_family_counts"]
        == {key: value["families"] for key, value in by.items()},
        "source/dedup family vector mismatch",
    )

    freeze = doc["candidate_freeze"]
    _require(freeze.get("frozen") is False, "candidate frozen before terminal dedup")
    _require(
        freeze.get("record_count") == 0 and freeze.get("records") == [],
        "blocked candidate cannot publish records",
    )
    _require(
        freeze.get("record_inventory_digest_sha256") is None
        and freeze.get("candidate_set_digest_sha256") is None,
        "blocked candidate cannot publish identity",
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
    _require(claims.get("authorized_unique_optimized_targets") == 0, "loss capacity fabricated")
    _require(claims.get("optimizer_updates") == 0, "optimizer updates fabricated")

    gates = doc["downstream_gates"]
    _require(
        gates.get("source_convergence") == "WAIT_EXACT_HEAD_TERMINAL_SUCCESS",
        "source gate weakened",
    )
    _require(
        gates.get("global_exact_near_fragment_lineage_dedup")
        == "WAIT_EXACT_HEAD_TERMINAL_SUCCESS",
        "dedup gate weakened",
    )
    _require(
        gates.get("record_inventory_freeze") == "NOT_PERMITTED_NO_TERMINAL_GLOBAL_DEDUP",
        "record freeze gate weakened",
    )
    _require(
        gates.get("reserved_evaluation_decontamination")
        == "NOT_PERMITTED_NO_FROZEN_POST_DEDUP_CANDIDATE_IDENTITY",
        "decontam gate weakened",
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
    for key in (
        "no_queue_or_pending_as_pass",
        "no_source_bytes_as_token_or_loss_capacity",
        "no_replay_or_duplicate_quota_repair",
        "source_registry_and_dedup_must_bind_same_composed_vector",
    ):
        _require(unblock.get(key) is True, f"unblock guard missing: {key}")
    required = unblock.get("required_order")
    _require(isinstance(required, list) and len(required) == 5, "required order drift")
    _require("global dedup" in required[1].lower(), "global dedup ordering missing")
    _require("freeze" in required[2].lower(), "freeze ordering missing")
    _require("decontamination" in required[3].lower(), "decontam ordering missing")


def main() -> int:
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_doc(doc)
    print(f"PASS {EXPECTED_STATUS} evidence={doc['evidence_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
