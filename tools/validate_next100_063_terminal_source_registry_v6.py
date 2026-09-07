"Fail-closed validator for NEXT100-063 terminal source registry V6."

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V6_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v6.json"
BUNDLE_PATH = ROOT / "configs/data/data_bulk_code1_permissive_python_bundle_v1.json"
EVIDENCE_PATH = ROOT / "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json"

EXPECTED_V5_HEAD = "991a0b6e939cddeff16c075922f7c407fa1e86cb"
EXPECTED_V5_BLOB = "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf"
EXPECTED_MAIN_MERGE = "4fa4839f2882984e7bf148dc38ce315d51b60957"
EXPECTED_EXECUTION_HEAD = "a045c602bfcead862f7852924fb78a5d78c992d6"
EXPECTED_RUN = 34155446113
EXPECTED_CONTRACT_BLOB = "05d2f5e6a83d4a8cf8159f422bd9a66a9dd3f393"
EXPECTED_EVIDENCE_BLOB = "b2b9b361dbf0e6c883dca4d7f006812c69cdd4e0"
EXPECTED_CONTRACT_ID = "7fd2228208f928859ebe68e947a72c977cda6952035a654d12923ce3a19a7dd6"
EXPECTED_REPORT_ID = "80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985"
EXPECTED_ARTIFACT_DIGEST = (
    "sha256:9febb6e4e900df63c58b1cb0ed2a7003df56d6e9bd593e4b4adbaa54496010ac"
)
EXPECTED_V6_ID = "57e26f58ee20498a702eea3e4feb36def8a74f8529165f23d5ad3a49248d027c"
EXPECTED_ADDITION_BYTES = 3_880_009
EXPECTED_TOTAL = 6_095_624
EXPECTED_ENVELOPE = 6_097_985
EXPECTED_GAP = 13_904_376

V5_CODE_FAMILIES = {
    "github:encode/httpx",
    "github:psf/requests",
    "github:django/django",
    "github:Kludex/starlette",
    "github:numpy/numpy",
    "github:python-attrs/attrs",
}


class RegistryV6Error(ValueError):
    "Raised when the V6 source-registry composition drifts."


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryV6Error(message)


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def canonical_identity(data: dict[str, Any]) -> str:
    body = deepcopy(data)
    body.pop("registry_identity_sha256", None)
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(
    v6: dict[str, Any],
    bundle: dict[str, Any],
    evidence: dict[str, Any],
    *,
    bundle_blob_sha1: str,
    evidence_blob_sha1: str,
) -> None:
    require(
        v6.get("schema_version") == "12-6.next100-063-terminal-source-registry.v6",
        "V6 schema drift",
    )
    require(
        v6.get("worker_id") == "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE-V6",
        "V6 worker drift",
    )
    require(v6.get("registry_identity_sha256") == EXPECTED_V6_ID, "V6 declared identity drift")
    require(canonical_identity(v6) == EXPECTED_V6_ID, "V6 content identity drift")

    supersedes = v6.get("supersedes", {})
    require(supersedes.get("pr") == 538, "V5 PR binding drift")
    require(supersedes.get("head_sha") == EXPECTED_V5_HEAD, "V5 head binding drift")
    require(supersedes.get("v5_git_blob_sha1") == EXPECTED_V5_BLOB, "V5 blob binding drift")

    base = v6.get("base_v5", {})
    require(base.get("numeric_training_capacity_bytes") == 2_215_615, "V5 capacity drift")
    require(base.get("source_normalized_envelope_bytes") == 2_217_976, "V5 envelope drift")
    require(base.get("uncredited_source_normalized_bytes") == 2_361, "V5 uncredited drift")
    require(base.get("independent_family_count") == 15, "V5 family-count drift")
    require(set(base.get("code_family_ids", [])) == V5_CODE_FAMILIES, "V5 code-family drift")
    require(
        base.get("authorized_balanced_no_replay_loss_positions") == 0,
        "V5 training exposure must remain zero",
    )

    require(bundle_blob_sha1 == EXPECTED_CONTRACT_BLOB, "bundle Git blob drift")
    require(evidence_blob_sha1 == EXPECTED_EVIDENCE_BLOB, "terminal evidence Git blob drift")
    require(bundle.get("contract_identity_sha256") == EXPECTED_CONTRACT_ID, "contract identity drift")
    require(
        evidence.get("contract_identity_sha256") == EXPECTED_CONTRACT_ID,
        "evidence contract binding drift",
    )
    require(evidence.get("execution_head_sha") == EXPECTED_EXECUTION_HEAD, "execution head drift")
    require(evidence.get("workflow_run_id") == EXPECTED_RUN, "workflow run drift")
    require(evidence.get("workflow_conclusion") == "success", "workflow not terminal success")

    artifact = evidence.get("artifact", {})
    require(artifact.get("id") == 10_030_825_509, "artifact id drift")
    require(artifact.get("digest") == EXPECTED_ARTIFACT_DIGEST, "artifact digest drift")
    require(
        artifact.get("retained_report_identity_sha256") == EXPECTED_REPORT_ID,
        "retained report identity drift",
    )

    execution = evidence.get("execution", {})
    require(execution.get("independent_materializations") == 2, "two-build evidence missing")
    require(execution.get("byte_identical_reports") is True, "materializations are not identical")
    require(execution.get("exact_source_checkout") is True, "exact source checkout missing")
    require(
        execution.get("exact_license_blob_validation") is True,
        "exact license validation missing",
    )
    require(
        execution.get("credential_scan") == "PASS_NO_HIGH_CONFIDENCE_HITS",
        "credential scan not terminal pass",
    )

    family_rows = evidence.get("families", [])
    family_ids = [row.get("family_id") for row in family_rows]
    require(len(family_rows) == 6, "bundle family count drift")
    require(len(set(family_ids)) == 6, "bundle family ids must be unique")
    require(not (set(family_ids) & V5_CODE_FAMILIES), "bundle overlaps V5 canonical families")
    require(
        sum(int(row.get("eligible_utf8_bytes", -1)) for row in family_rows)
        == EXPECTED_ADDITION_BYTES,
        "bundle byte arithmetic drift",
    )
    require(
        sum(int(row.get("eligible_file_count", -1)) for row in family_rows) == 229,
        "bundle file arithmetic drift",
    )

    aggregate = evidence.get("aggregate", {})
    require(aggregate.get("source_family_count") == 6, "aggregate family count drift")
    require(aggregate.get("eligible_file_count") == 229, "aggregate file count drift")
    require(
        aggregate.get("eligible_utf8_bytes") == EXPECTED_ADDITION_BYTES,
        "aggregate byte count drift",
    )

    boundary = evidence.get("claim_boundary", {})
    require(boundary.get("source_intake_evidence_terminal") is True, "intake not terminal")
    require(
        boundary.get("automatic_canonical_capacity_credit") is False,
        "bundle self-promoted canonical capacity",
    )
    require(
        boundary.get("post_global_dedup_capacity_claimed") is False,
        "bundle self-promoted post-dedup capacity",
    )
    require(boundary.get("authorized_training_exposure") == 0, "bundle training exposure drift")
    require(boundary.get("tokenizer_fit_authorized") is False, "bundle authorized tokenizer fit")
    require(boundary.get("model_training_executed") is False, "bundle executed training")
    require(boundary.get("final_test_accessed") is False, "bundle accessed final test")
    require(boundary.get("paid_compute_used") is False, "bundle used paid compute")

    addition = v6.get("terminal_addition", {})
    require(addition.get("merged_pr") == 818, "addition PR drift")
    require(addition.get("main_merge_sha") == EXPECTED_MAIN_MERGE, "main merge binding drift")
    require(addition.get("execution_head_sha") == EXPECTED_EXECUTION_HEAD, "addition head drift")
    require(addition.get("workflow_run_id") == EXPECTED_RUN, "addition run drift")
    require(addition.get("workflow_conclusion") == "success", "addition run not success")
    require(addition.get("contract_git_blob_sha1") == EXPECTED_CONTRACT_BLOB, "contract blob drift")
    require(
        addition.get("terminal_evidence_git_blob_sha1") == EXPECTED_EVIDENCE_BLOB,
        "evidence blob drift",
    )
    require(
        addition.get("numeric_training_capacity_bytes") == EXPECTED_ADDITION_BYTES,
        "addition capacity drift",
    )
    require(set(addition.get("families", [])) == set(family_ids), "addition family vector drift")

    inv = v6.get("derived_pre_successor_global_dedup_inventory", {})
    require(2_215_615 + EXPECTED_ADDITION_BYTES == EXPECTED_TOTAL, "V6 total arithmetic broken")
    require(2_217_976 + EXPECTED_ADDITION_BYTES == EXPECTED_ENVELOPE, "V6 envelope broken")
    require(inv.get("candidate_numeric_training_capacity_bytes") == EXPECTED_TOTAL, "V6 total drift")
    require(
        inv.get("candidate_source_normalized_envelope_bytes") == EXPECTED_ENVELOPE,
        "V6 envelope drift",
    )
    require(inv.get("candidate_independent_family_count") == 21, "V6 family-count drift")
    require(inv.get("target_gap_numeric_training_capacity_bytes") == EXPECTED_GAP, "V6 gap drift")
    require(
        abs(float(inv.get("target_fraction_by_numeric_training_capacity")) - 0.3047812) < 1e-12,
        "V6 target fraction drift",
    )
    by_stratum = inv.get("by_stratum", {})
    require(by_stratum["uk"]["numeric_training_capacity_bytes"] == 100_856, "UK drift")
    require(by_stratum["en"]["numeric_training_capacity_bytes"] == 1_838_293, "EN drift")
    require(by_stratum["code"]["numeric_training_capacity_bytes"] == 4_156_475, "code drift")
    require(by_stratum["code"]["family_count"] == 12, "code family-count drift")

    policy = v6.get("composition_policy", {})
    require(
        policy.get("historical_v5_is_bound_by_exact_pr_head_and_git_blob") is True,
        "historical authority binding weakened",
    )
    require(
        policy.get("global_cross_source_dedup_required_before_corpus_identity") is True,
        "global dedup gate weakened",
    )
    require(
        policy.get("source_intake_bytes_are_pre_dedup_planning_capacity_only") is True,
        "source bytes promoted beyond planning capacity",
    )
    require(
        policy.get("evaluation_permission_never_inferred_from_training_permission") is True,
        "evaluation firewall weakened",
    )
    require(policy.get("no_overlap_with_v5_family_ids_required") is True, "family overlap gate weakened")

    gates = v6.get("downstream_gate_vector", {})
    require(gates.get("authorized_balanced_no_replay_loss_positions") == 0, "training exposure drift")
    require(
        gates.get("successor_global_cross_source_exact_near_dedup") == "REQUIRED_NEXT",
        "global dedup no longer required next",
    )
    require(gates.get("tokenizer_fit") == "BLOCKED", "tokenizer fit promoted")
    require(gates.get("long_training") == "BLOCKED", "long training promoted")
    require(gates.get("paid_compute") == "NOT_AUTHORIZED", "paid compute promoted")


def main() -> None:
    raw_v6 = V6_PATH.read_bytes()
    raw_bundle = BUNDLE_PATH.read_bytes()
    raw_evidence = EVIDENCE_PATH.read_bytes()
    validate(
        json.loads(raw_v6),
        json.loads(raw_bundle),
        json.loads(raw_evidence),
        bundle_blob_sha1=git_blob_sha1(raw_bundle),
        evidence_blob_sha1=git_blob_sha1(raw_evidence),
    )
    print(
        "NEXT100-063 V6 PASS "
        f"numeric_capacity_bytes={EXPECTED_TOTAL} "
        f"code_numeric_bytes=4156475 families=21 gap_bytes={EXPECTED_GAP}"
    )


if __name__ == "__main__":
    main()
