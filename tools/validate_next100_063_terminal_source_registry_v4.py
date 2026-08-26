#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-063 terminal source registry V4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"
EXPECTED_IDENTITY = "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58"
EXPECTED_V3_IDENTITY = "66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c"
EXPECTED_PARENT_CAPACITY = 243_970
EXPECTED_CPYTHON_CAPACITY = 15_540
EXPECTED_CPYTHON_SOURCE = 17_901
EXPECTED_GUTENBERG_CAPACITY = 1_672_110
EXPECTED_TOTAL = 2_045_180
EXPECTED_EN = 1_838_293
EXPECTED_EN_ENVELOPE = 1_840_654
EXPECTED_CODE = 106_031
EXPECTED_UK = 100_856
EXPECTED_FAMILIES = 14
EXPECTED_GAP = 17_954_820


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-063 V4 FAIL: {message}")


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    claimed = data.get("registry_identity_sha256")
    body = dict(data)
    body.pop("registry_identity_sha256", None)
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    require(claimed == actual == EXPECTED_IDENTITY, "registry identity drift")
    require(data.get("schema_version") == "12-6.next100-063-terminal-source-registry.v4", "schema drift")
    require(data.get("worker_id") == "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE-V4", "worker drift")
    require(data.get("supersedes", {}).get("v3_registry_identity_sha256") == EXPECTED_V3_IDENTITY, "V3 binding drift")

    parent = data.get("dedup_parent", {})
    require(parent.get("head_sha") == "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13", "dedup parent head drift")
    require(parent.get("dedicated_workflow_run") == 32999969398, "dedup parent run drift")
    require(parent.get("dedicated_workflow_conclusion") == "success", "dedup parent not green")
    require(parent.get("numeric_training_capacity_bytes") == EXPECTED_PARENT_CAPACITY, "dedup parent capacity drift")
    require(parent.get("independent_family_count") == 7, "dedup parent family count drift")
    parent_families = set(parent.get("families", []))
    require(len(parent_families) == 7, "dedup parent family identity drift")

    rows = data.get("terminal_late_additions", [])
    by_pr = {row.get("pr"): row for row in rows}
    expected_prs = {445, 449, 462, 467, 468, 470, 472}
    require(set(by_pr) == expected_prs, "late authority set drift")

    late_families: set[str] = set()
    for pr, row in by_pr.items():
        require(row.get("dedicated_workflow_conclusion") == "success", f"PR {pr} exact authority not green")
        require(str(row.get("verdict", "")).startswith("ADMIT"), f"PR {pr} not admitted")
        require(str(row.get("training", "")).startswith("ALLOWED"), f"PR {pr} lacks training permission")
        family = row.get("family")
        require(isinstance(family, str) and family, f"PR {pr} missing family")
        require(family not in parent_families and family not in late_families, f"PR {pr} duplicate family credit")
        late_families.add(family)
        numeric = row.get("numeric_training_capacity_bytes")
        normalized = row.get("source_normalized_bytes")
        require(isinstance(numeric, int) and isinstance(normalized, int), f"PR {pr} capacity types invalid")
        require(normalized > 0 and 0 <= numeric <= normalized, f"PR {pr} invalid capacity relation")

    cp = by_pr[467]
    require(cp.get("head") == "5a6a495a24bce449334cbc5126d0114f61a9f57c", "CPython source head drift")
    require(cp.get("authority_identity") == "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d", "CPython source authority drift")
    require(cp.get("numeric_training_capacity_bytes") == EXPECTED_CPYTHON_CAPACITY, "CPython accepted capacity drift")
    require(cp.get("source_normalized_bytes") == EXPECTED_CPYTHON_SOURCE, "CPython source envelope drift")
    require(cp.get("accepted_chunk_count") == 14 and cp.get("rejected_chunk_count") == 2, "CPython chunk boundary drift")
    materialization = cp.get("accepted_materialization", {})
    require(materialization.get("head_sha") == "8f0cbc16f9a920ca9ab3e3061b53fbfec8838d77", "CPython materialization head drift")
    require(materialization.get("workflow_run") == 33005689174, "CPython materialization run drift")
    require(materialization.get("workflow_conclusion") == "success", "CPython materialization not green")
    require(materialization.get("artifact_id") == 9620571005, "CPython artifact id drift")
    require(materialization.get("artifact_digest") == "sha256:5c04e12f1100fd4012efc1cf693f213d1d7c9ababee2a16367897377cde60379", "CPython artifact digest drift")
    require(materialization.get("report_sha256") == "3d497b1fd4b7d11531ed4a389b98e9522886f4de27e3740f8e5ac1c07b662e92", "CPython report drift")
    require(materialization.get("accepted_capacity_bytes") == EXPECTED_CPYTHON_CAPACITY, "CPython accepted ledger drift")
    require(materialization.get("materialized_payload_bytes") == 15_566, "CPython materialized payload byte drift")
    require(materialization.get("materialized_payload_sha256") == "cc5dfb4dcaf15b492cb5ec4636aaae71edfd0263f387b12a499134ebc19dbd76", "CPython materialized payload identity drift")
    require(materialization.get("rejection_reasons") == {"pii_phone": 2}, "CPython rejection reasons drift")
    require(EXPECTED_CPYTHON_SOURCE - EXPECTED_CPYTHON_CAPACITY == 2_361, "CPython uncredited arithmetic drift")

    gutenberg = by_pr[470]
    require(gutenberg.get("head") == "3f4ad26e1e8f3406a1274418cf5f485814ce3032", "Gutenberg source head drift")
    require(gutenberg.get("authority_identity") == "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b", "Gutenberg authority drift")
    require(gutenberg.get("numeric_training_capacity_bytes") == EXPECTED_GUTENBERG_CAPACITY, "Gutenberg capacity drift")
    require(gutenberg.get("source_normalized_bytes") == EXPECTED_GUTENBERG_CAPACITY, "Gutenberg normalized capacity drift")
    require(gutenberg.get("source_record_count") == 3, "Gutenberg record count drift")
    require(gutenberg.get("family") == "en.project-gutenberg.public-domain-books", "Gutenberg family drift")
    require(gutenberg.get("dedicated_workflow_run") == 32998859164, "Gutenberg run drift")
    require(gutenberg.get("terminal_artifact_id") == 9618402768, "Gutenberg artifact id drift")
    require(gutenberg.get("terminal_artifact_digest") == "sha256:63fa5d9b403432074193e290beb0473b5a1f7b74de1ac30bad71b9ec8405e006", "Gutenberg artifact digest drift")
    require(gutenberg.get("terminal_report_sha256") == "2d1e99f0cb41a1b90ce995076d88bffa7435b151f6036dea0a54552d89556cd0", "Gutenberg report drift")
    require(gutenberg.get("terminal_report_evidence_identity_sha256") == "e454d9b1a94497aec95776c8b9b6318b73647bc3248f62da70e45a7c369f637e", "Gutenberg evidence identity drift")
    seal = gutenberg.get("terminal_seal", {})
    require(seal.get("pr") == 627 and seal.get("head_sha") == "c50b3f9cf871792c03886bdc1ccdc144812be88f", "Gutenberg terminal seal binding drift")
    require(seal.get("authority_identity_sha256") == gutenberg.get("authority_identity"), "Gutenberg seal identity mismatch")

    inv = data.get("pre_successor_global_dedup_inventory", {})
    late_numeric = sum(row["numeric_training_capacity_bytes"] for row in rows)
    source_envelope = EXPECTED_PARENT_CAPACITY + sum(row["source_normalized_bytes"] for row in rows)
    require(late_numeric == 1_801_210, "late numeric capacity arithmetic drift")
    require(EXPECTED_PARENT_CAPACITY + late_numeric == EXPECTED_TOTAL, "candidate numeric capacity arithmetic drift")
    require(source_envelope == 2_047_541, "source normalized envelope arithmetic drift")
    require(inv.get("candidate_numeric_training_capacity_bytes") == EXPECTED_TOTAL, "inventory candidate capacity drift")
    require(inv.get("candidate_source_normalized_envelope_bytes") == source_envelope, "inventory source envelope drift")
    require(inv.get("uncredited_source_normalized_bytes") == source_envelope - EXPECTED_TOTAL == 2_361, "inventory uncredited byte drift")
    by = inv.get("by_stratum", {})
    require(by.get("uk") == {"family_count": 4, "numeric_training_capacity_bytes": EXPECTED_UK, "source_normalized_envelope_bytes": EXPECTED_UK}, "UK stratum drift")
    require(by.get("en") == {"family_count": 5, "numeric_training_capacity_bytes": EXPECTED_EN, "source_normalized_envelope_bytes": EXPECTED_EN_ENVELOPE}, "EN stratum drift")
    require(by.get("code") == {"family_count": 5, "numeric_training_capacity_bytes": EXPECTED_CODE, "source_normalized_envelope_bytes": EXPECTED_CODE}, "code stratum drift")
    require(inv.get("candidate_independent_family_count") == EXPECTED_FAMILIES == len(parent_families | late_families), "candidate family count drift")
    require(inv.get("family_minimum_gate") == "PASS_AUTHORITY_LAYER_PRE_SUCCESSOR_DEDUP", "family gate drift")
    require(inv.get("research_corpus_v1_acquisition_planning_target_bytes") == 20_000_000, "research target drift")
    require(inv.get("target_gap_numeric_training_capacity_bytes") == EXPECTED_GAP, "research target gap drift")
    require(abs(float(inv.get("target_fraction_by_numeric_training_capacity")) - 0.102259) < 1e-12, "target fraction drift")

    held = {row.get("pr"): row for row in data.get("held_out_or_zero_credit", []) if isinstance(row, dict)}
    for pr, head, run in (
        (465, "ca1755886f052d272029d6d68b2f1b7f02187936", "32999061340"),
        (475, "78cada1d69b3f0c438012c4e6cf79143aae2f603", "32999511493"),
    ):
        require(held.get(pr, {}).get("head") == head, f"failed PR {pr} head drift")
        reason = held.get(pr, {}).get("reason", "")
        require(run in reason and "COMPLETED_FAILURE" in reason, f"failed PR {pr} gained unsafe credit")

    policy = data.get("composition_policy", {})
    require(policy.get("global_cross_source_dedup_required_for_late_additions_before_corpus_identity") is True, "global dedup gate weakened")
    require(policy.get("source_normalized_bytes_are_not_eligible_capacity_without_post_filter_materialization") is True, "source/capacity firewall weakened")
    require(policy.get("failed_queued_retest_or_pr_text_only_candidates_counted") is False, "failed/nonterminal credit enabled")
    require(policy.get("exact_artifact_success_may_resolve_stale_pr_prose_without_relaxing_downstream_gates") is True, "artifact/prose authority rule drift")

    gates = data.get("downstream_gate_vector", {})
    require(gates.get("authorized_balanced_no_replay_loss_positions") == 0, "training exposure must remain zero")
    require(gates.get("successor_global_cross_source_exact_near_dedup") == "REQUIRED_NEXT", "successor dedup gate weakened")
    require(gates.get("evaluation_decontamination") == "REQUIRED_AFTER_EXACT_CANDIDATE_INVENTORY", "decontamination gate weakened")
    require(gates.get("long_training") == "BLOCKED", "long training promoted")
    require(gates.get("paid_compute") == "NOT_AUTHORIZED", "paid compute promoted")

    print(
        "NEXT100-063 V4 PASS "
        f"identity={actual} numeric_capacity_bytes={EXPECTED_TOTAL} "
        f"source_envelope_bytes={source_envelope} en_numeric_bytes={EXPECTED_EN} "
        f"families={EXPECTED_FAMILIES} gap_bytes={EXPECTED_GAP}"
    )


if __name__ == "__main__":
    main()
