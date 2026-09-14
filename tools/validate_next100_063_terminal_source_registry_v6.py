"""Fail-closed validator for NEXT100-063 source-registry convergence V6."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v6.json"
INTAKE_EVIDENCE_PATH = (
    ROOT / "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json"
)

EXPECTED_V5_BLOB = "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf"
EXPECTED_INTAKE_BLOB = "b2b9b361dbf0e6c883dca4d7f006812c69cdd4e0"
EXPECTED_INTAKE_BYTES = 3_880_009
EXPECTED_PRIOR_CREDITED = 2_215_615
EXPECTED_PRIOR_CODE = 276_466
EXPECTED_POTENTIAL_TOTAL = 6_095_624
EXPECTED_POTENTIAL_CODE = 4_156_475
EXPECTED_CREDITED_GAP = 17_784_385
EXPECTED_POTENTIAL_GAP = 13_904_376
EXPECTED_FAMILIES = [
    "github:pallets/flask",
    "github:pallets/click",
    "github:pallets/jinja",
    "github:pallets/werkzeug",
    "github:agronholm/anyio",
    "github:pytest-dev/pytest",
]
EXPECTED_PRIOR_CODE_FAMILIES = [
    "github:encode/httpx",
    "github:psf/requests",
    "github:django/django",
    "github:Kludex/starlette",
    "github:numpy/numpy",
    "github:python-attrs/attrs",
]


class RegistryV6Error(ValueError):
    """Raised when V6 composition would overclaim or drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryV6Error(message)


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def canonical_identity(data: dict[str, Any]) -> str:
    body = dict(data)
    body.pop("registry_identity_sha256", None)
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(config: dict[str, Any], intake: dict[str, Any], *, intake_blob_sha1: str) -> None:
    require(
        config.get("schema_version")
        == "12-6.next100-063-terminal-source-registry.v6",
        "V6 schema drift",
    )
    require(
        config.get("registry_identity_sha256") == canonical_identity(config),
        "V6 self identity drift",
    )

    prior = config.get("prior_registry_authority", {})
    require(prior.get("pr") == 538, "prior registry PR drift")
    require(prior.get("config_git_blob_sha1") == EXPECTED_V5_BLOB, "V5 blob drift")
    require(
        prior.get("candidate_numeric_training_capacity_bytes") == EXPECTED_PRIOR_CREDITED,
        "prior credited capacity drift",
    )
    require(
        prior.get("by_stratum", {}).get("code", {}).get("numeric_training_capacity_bytes")
        == EXPECTED_PRIOR_CODE,
        "prior code capacity drift",
    )
    require(
        prior.get("code_families") == EXPECTED_PRIOR_CODE_FAMILIES,
        "prior code-family vector drift",
    )

    require(intake_blob_sha1 == EXPECTED_INTAKE_BLOB, "terminal intake evidence blob drift")
    require(
        intake.get("schema_version")
        == "12-6.data-bulk-code1-permissive-python-bundle-terminal-evidence.v1",
        "terminal intake schema drift",
    )
    require(intake.get("workflow_conclusion") == "success", "terminal intake not successful")
    require(
        intake.get("execution", {}).get("independent_materializations") == 2,
        "two clean intake materializations required",
    )
    require(
        intake.get("execution", {}).get("byte_identical_reports") is True,
        "intake materializations are not byte-identical",
    )
    require(
        intake.get("execution", {}).get("credential_scan")
        == "PASS_NO_HIGH_CONFIDENCE_HITS",
        "credential scan not terminal pass",
    )
    require(
        intake.get("claim_boundary", {}).get("automatic_canonical_capacity_credit") is False,
        "intake evidence must forbid automatic capacity credit",
    )
    require(
        intake.get("claim_boundary", {}).get("authorized_training_exposure") == 0,
        "intake evidence must authorize zero training exposure",
    )

    addition = config.get("terminal_intake_addition", {})
    require(addition.get("source_pr") == 818, "intake PR drift")
    require(addition.get("evidence_git_blob_sha1") == EXPECTED_INTAKE_BLOB, "intake binding drift")
    require(
        addition.get("eligible_utf8_bytes") == EXPECTED_INTAKE_BYTES,
        "intake eligible bytes drift",
    )
    require(
        addition.get("eligible_utf8_bytes")
        == intake.get("aggregate", {}).get("eligible_utf8_bytes"),
        "config/evidence byte mismatch",
    )
    require(
        addition.get("eligible_file_count") == intake.get("aggregate", {}).get("eligible_file_count"),
        "config/evidence file-count mismatch",
    )
    evidence_families = [item["family_id"] for item in intake.get("families", [])]
    require(evidence_families == EXPECTED_FAMILIES, "terminal intake family vector drift")
    require(addition.get("families") == EXPECTED_FAMILIES, "V6 intake family vector drift")
    require(
        set(EXPECTED_PRIOR_CODE_FAMILIES).isdisjoint(EXPECTED_FAMILIES),
        "new intake overlaps prior code family",
    )
    require(addition.get("family_overlap_with_prior_code_families") == [], "overlap claim drift")
    require(addition.get("canonical_capacity_credit_bytes") == 0, "premature capacity promotion")

    state = config.get("live_pre_global_dedup_state", {})
    require(
        state.get("credited_candidate_numeric_training_capacity_bytes") == EXPECTED_PRIOR_CREDITED,
        "credited capacity changed before global dedup",
    )
    require(
        state.get("terminal_intake_eligible_bytes_pending_global_dedup") == EXPECTED_INTAKE_BYTES,
        "pending intake bytes drift",
    )
    require(
        EXPECTED_PRIOR_CREDITED + EXPECTED_INTAKE_BYTES == EXPECTED_POTENTIAL_TOTAL,
        "potential total arithmetic broken",
    )
    require(
        EXPECTED_PRIOR_CODE + EXPECTED_INTAKE_BYTES == EXPECTED_POTENTIAL_CODE,
        "potential code arithmetic broken",
    )
    require(
        state.get("potential_numeric_capacity_if_all_pending_intake_survives_dedup")
        == EXPECTED_POTENTIAL_TOTAL,
        "potential total drift",
    )
    require(
        state.get("potential_code_capacity_if_all_pending_intake_survives_dedup")
        == EXPECTED_POTENTIAL_CODE,
        "potential code drift",
    )
    require(
        state.get("credited_gap_to_planning_target_bytes") == EXPECTED_CREDITED_GAP,
        "credited planning gap drift",
    )
    require(
        state.get("potential_gap_if_all_pending_intake_survives_dedup") == EXPECTED_POTENTIAL_GAP,
        "potential planning gap drift",
    )
    require(
        state.get("code_target_state") == "POTENTIALLY_MET_PRE_DEDUP_NOT_CANONICAL",
        "code target overclaim or drift",
    )

    next_auth = config.get("required_next_authority", {})
    require(next_auth.get("incumbent_global_dedup_pr") == 632, "global dedup incumbent drift")
    require(
        next_auth.get("must_consume_prior_registry_v5_blob") == EXPECTED_V5_BLOB,
        "global dedup V5 binding drift",
    )
    require(
        next_auth.get("must_consume_terminal_intake_evidence_blob") == EXPECTED_INTAKE_BLOB,
        "global dedup intake binding drift",
    )
    require(
        next_auth.get("must_reconstruct_or_fetch_exact_229_file_payload_graph") is True,
        "payload reconstruction gate weakened",
    )
    require(
        next_auth.get("must_apply_global_exact_near_fragment_lineage_dedup") is True,
        "global dedup gate weakened",
    )

    truth = config.get("truth_boundary", {})
    require(truth.get("corpus_identity") is None, "corpus identity fabricated")
    require(truth.get("post_global_dedup_capacity_bytes") is None, "post-dedup capacity fabricated")
    require(
        truth.get("authorized_balanced_no_replay_loss_positions") == 0,
        "loss-position exposure promoted",
    )
    require(truth.get("tokenizer_fit") == "BLOCKED", "tokenizer fit promoted")
    require(truth.get("model_training_executed") is False, "training claim drift")
    require(truth.get("optimizer_updates") == 0, "optimizer update claim drift")
    require(truth.get("final_test_accessed") is False, "final-test firewall drift")
    require(truth.get("paid_compute_authorized") is False, "paid compute promoted")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_intake = INTAKE_EVIDENCE_PATH.read_bytes()
    intake = json.loads(raw_intake.decode("utf-8"))
    validate(config, intake, intake_blob_sha1=git_blob_sha1(raw_intake))
    state = config["live_pre_global_dedup_state"]
    print(
        "NEXT100-063 V6 PASS "
        f"credited={state['credited_candidate_numeric_training_capacity_bytes']} "
        f"pending_intake={state['terminal_intake_eligible_bytes_pending_global_dedup']} "
        f"potential={state['potential_numeric_capacity_if_all_pending_intake_survives_dedup']}"
    )


if __name__ == "__main__":
    main()
