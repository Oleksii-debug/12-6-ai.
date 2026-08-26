#!/usr/bin/env python3
"""Fail-closed validator for the live ~20M readiness controller."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/control/20m_readiness_controller_v1.json"

EXPECTED_SCHEMA = "12-6.autonomous-20m-readiness-controller.v1"
EXPECTED_WORKER = "AUTONOMOUS-20M-READINESS-CONTROLLER-20260826"
EXPECTED_MODEL_HEAD = "e4ff486fd90802fc123bebf60eed4e59196a98df"
EXPECTED_MODEL_SPEC = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
EXPECTED_MODEL_PARAMS = 20_613_440
EXPECTED_CORPUS_HEAD = "8820ba1b255f6bb95c7db0531fd846078a1aae01"
EXPECTED_UA_SOURCE_HEAD = "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2"
EXPECTED_EN_SOURCE_HEAD = "5a6a495a24bce449334cbc5126d0114f61a9f57c"
EXPECTED_DECONTAM_HEAD = "80e8fc9828214ce86e16b5c7f2fdec9107b4df43"
EXPECTED_D05_AUDIT_HEAD = "5c3d3d93cb035c05ad2045a243d722a4ad1dce60"
EXPECTED_D05_REMEDIATION_HEAD = "5c4bdf3faa31397fef65b800d00ff95483be3aac"
EXPECTED_D05_PREVIOUS_GREEN_HEAD = "397ba857f08b08eb1363a634597a757d25e8fd68"
EXPECTED_SOURCE_REGISTRY_HEAD = "94dce83cbe611144f961b9f93b3be273345a7f62"
EXPECTED_SOURCE_REGISTRY_ID = "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526"


class ReadinessValidationError(ValueError):
    """Raised when the 20M control-plane snapshot violates its contract."""


def _require(condition: bool, message: str) -> None:
    """Enforce an invariant even when the interpreter runs with ``-O``."""
    if not condition:
        raise ReadinessValidationError(message)


def canonical_identity(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_identity_sha256", None)
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(value: dict[str, Any]) -> None:
    _require(value["schema"] == EXPECTED_SCHEMA, "unexpected readiness schema")
    _require(value["worker_id"] == EXPECTED_WORKER, "unexpected readiness worker")
    _require(value["repository"] == "Oleksii-debug/12-6-ai.", "unexpected repository")
    _require(value["execution_profile"] == "LOCAL_FREE", "execution profile is not LOCAL_FREE")
    _require(value["training_executed"] is False, "snapshot claims training executed")
    _require(value["paid_compute_used"] is False, "snapshot claims paid compute")
    _require(value["long_training_authorized"] is False, "long training is unexpectedly authorized")

    authorities = value["authorities"]

    model = authorities["primary_20m_model"]
    _require(model["head_sha"] == EXPECTED_MODEL_HEAD, "unexpected primary 20M head")
    _require(model["qualification"] == "PASS", "primary 20M mechanics are not qualified")
    _require(model["random_init_only"] is True, "primary 20M authority is not random-init only")
    _require(model["long_training_performed"] is False, "primary 20M authority claims long training")
    _require(model["parameter_count"] == EXPECTED_MODEL_PARAMS, "unexpected primary parameter count")
    _require(model["model_spec_sha256"] == EXPECTED_MODEL_SPEC, "unexpected primary ModelSpec")
    _require(
        all(state == "PASS" for state in model["mechanics"].values()),
        "one or more primary 20M mechanics gates are not PASS",
    )

    # Nominal save/load mechanics passing is not a corruption-integrity PASS.
    # NEXT100-075 remains the independent fail-closed authority until its full
    # matrix is rerun against one exact remediated production head.
    d05 = authorities["checkpoint_corruption_audit"]
    _require(d05["head_sha"] == EXPECTED_D05_AUDIT_HEAD, "unexpected D05 audit head")
    _require(d05["base_model_sha"] == EXPECTED_MODEL_HEAD, "D05 audit targets wrong model head")
    _require(d05["required_cases"] == 11, "D05 audit case count changed")
    _require(d05["rejected_before_mutation"] == 8, "D05 rejected-case count changed")
    _require(len(d05["blocking_cases"]) == 3, "D05 blocker count changed")
    _require(d05["verdict"] == "RETEST_REQUIRED", "D05 audit is not RETEST_REQUIRED")
    _require(
        d05["direct_production_module_rerun_required"] is True,
        "D05 production rerun is not required",
    )

    remediation = authorities["d05_remediation_convergence"]
    _require(
        remediation["current_head_sha"] == EXPECTED_D05_REMEDIATION_HEAD,
        "unexpected current D05 remediation head",
    )
    _require(
        remediation["previous_green_head_sha"] == EXPECTED_D05_PREVIOUS_GREEN_HEAD,
        "unexpected prior green D05 head",
    )
    _require(
        remediation["previous_exact_head_green_run"] == 33005163523,
        "unexpected prior green D05 run",
    )
    _require(
        remediation["current_exact_head_d05_run"] == 33005737951,
        "unexpected current D05 run",
    )
    _require(
        remediation["current_exact_head_d05_status"] == "QUEUED_AT_REFRESH",
        "current D05 run was not queued at snapshot refresh",
    )
    _require(remediation["preferred_candidate_pr"] == 504, "unexpected D05 preferred PR")
    _require(remediation["terminal_authority"] is False, "D05 remediation is prematurely terminal")
    _require(
        remediation["state"].startswith("NONTERMINAL_"),
        "D05 remediation state is not explicitly nonterminal",
    )

    corpus = authorities["research_corpus_v03"]
    _require(corpus["head_sha"] == EXPECTED_CORPUS_HEAD, "unexpected corpus authority head")
    _require(corpus["state"] == "TERMINAL_BLOCKED", "corpus is not terminal-blocked")
    _require(corpus["corpus_identity"] is None, "blocked corpus unexpectedly has an identity")
    _require(corpus["shard_identity"] is None, "blocked corpus unexpectedly has a shard identity")
    _require(
        corpus["authorized_balanced_no_replay_capacity"] == 0,
        "blocked corpus unexpectedly authorizes no-replay capacity",
    )

    source_registry = authorities["source_registry_convergence_candidate"]
    _require(
        source_registry["head_sha"] == EXPECTED_SOURCE_REGISTRY_HEAD,
        "unexpected source-registry candidate head",
    )
    _require(
        source_registry["registry_identity_sha256"] == EXPECTED_SOURCE_REGISTRY_ID,
        "unexpected source-registry identity",
    )
    _require(
        source_registry["candidate_normalized_bytes"] == 565_743,
        "source-registry byte capacity changed",
    )
    _require(
        source_registry["candidate_independent_family_count"] == 13,
        "source-registry family count changed",
    )
    _require(
        source_registry["family_minimum_gate"] == "PASS_PRE_GLOBAL_DEDUP",
        "source-registry family gate changed",
    )
    _require(source_registry["draft"] is True, "source-registry candidate is no longer marked draft")
    _require(
        source_registry["terminal_authority"] is False,
        "source-registry candidate is prematurely terminal",
    )
    _require(
        source_registry["training_authorized_loss_positions"] == 0,
        "source-registry candidate unexpectedly authorizes training loss positions",
    )
    _require(
        source_registry["research_corpus_v1_target_gap_normalized_bytes"] == 19_434_257,
        "source-registry target gap changed",
    )
    _require(
        all(
            row["family_count"] >= 2
            for row in source_registry["pre_global_dedup_by_stratum"].values()
        ),
        "one or more source-registry strata are below the family minimum",
    )

    ua = authorities["ua_wikisource_source"]
    _require(ua["head_sha"] == EXPECTED_UA_SOURCE_HEAD, "unexpected UA source head")
    _require(
        ua["authority_state"] == "TERMINAL_RIGHTS_AND_IMMUTABLE_SNAPSHOT",
        "UA source lacks terminal rights/snapshot authority",
    )
    _require(
        ua["training_selection"] == "BLOCKED_UNTIL_STANDARD_NEAR_MATCH_DECONTAMINATION",
        "UA source decontamination boundary changed",
    )

    en = authorities["en_python_docs_source"]
    _require(en["head_sha"] == EXPECTED_EN_SOURCE_HEAD, "unexpected EN source head")
    _require(en["terminal_verdict"] == "ADMIT", "EN source is not admitted")
    _require(en["training_use"] == "ALLOWED", "EN source is not training-allowed")
    _require(en["accepted_chunks"] > 0, "EN source has no accepted chunks")

    decontam = authorities["decontamination_v4"]
    _require(decontam["head_sha"] == EXPECTED_DECONTAM_HEAD, "unexpected decontamination head")
    _require(
        decontam["state"] == "BLOCKED_NO_EXACT_CANDIDATE_CORPUS_IDENTITY",
        "decontamination blocker state changed",
    )
    _require(decontam["scan_executed"] is False, "blocked decontamination claims execution")

    selection = authorities["selection_validation_composite"]
    _require(
        selection["state"] == "NONEMPTY_SELECTION_AUTHORITY",
        "selection validation is not a nonempty authority",
    )
    _require(sum(selection["records"].values()) > 0, "selection validation has no records")

    code_selection = authorities["code_selection_validation_v2"]
    _require(code_selection["eligible_objects"] == 0, "code selection unexpectedly has eligible objects")
    _require(code_selection["eligible_families"] == 0, "code selection unexpectedly has eligible families")

    campaign = value["campaign"]
    _require(campaign["authorized_unique_optimized_targets"] == 0, "campaign unexpectedly has authorized targets")
    _require(campaign["campaign_runnable_now"] is False, "campaign unexpectedly marked runnable")
    _require(campaign["long_training_started"] is False, "campaign unexpectedly claims long training")
    _require(campaign["decision"] == "DO_NOT_START_LONG_TRAINING", "campaign decision is not fail-closed")

    gates = value["gates"]
    _require(gates["primary_20m_mechanics"] == "PASS", "primary 20M mechanics gate changed")
    _require(
        gates["checkpoint_recovery_20m"] == "BLOCKED_D05_CORRUPTION_RETEST_REQUIRED",
        "20M checkpoint recovery gate changed",
    )
    _require(
        gates["checkpoint_integrity_20m"]
        == "BLOCKED_PENDING_CURRENT_HEAD_GREEN_AND_11_OF_11_RERUN",
        "20M checkpoint integrity gate changed",
    )
    _require(
        gates["source_registry_convergence"]
        == "CANDIDATE_NONTERMINAL_565743_BYTES_13_FAMILIES",
        "source-registry convergence gate changed",
    )
    _require(
        gates["exact_candidate_corpus_identity"] == "BLOCKED_NOT_MATERIALIZED",
        "candidate corpus identity gate changed",
    )
    _require(gates["real_20m_training"] == "BLOCKED", "real 20M training is not blocked")
    _require(
        gates["unique_no_replay_loss_capacity"] == "BLOCKED_ZERO",
        "unique no-replay capacity is not blocked at zero",
    )

    parallel_tracks = value["parallel_local_free_tracks"]
    _require(
        "TERMINALIZE_NEXT100_063_AND_MATERIALIZE_EXACT_CANDIDATE_CORPUS_IDENTITY"
        in parallel_tracks,
        "source-registry/corpus convergence track is missing",
    )
    _require(
        "WAIT_FOR_PR504_CURRENT_EXACT_HEAD_D05_AND_THEN_RERUN_NEXT100_075_11_CASE_MATRIX"
        in parallel_tracks,
        "D05 convergence track is missing",
    )
    _require(len(parallel_tracks) == len(set(parallel_tracks)), "parallel tracks contain duplicates")

    next_campaign = value["ordered_next_campaign"]
    _require(
        next_campaign[0] == "TERMINALIZE_NEXT100_063_SOURCE_REGISTRY_CONVERGENCE",
        "first ordered next action changed",
    )
    _require(
        next_campaign[1]
        == "MATERIALIZE_EXACT_PRE_DECONTAMINATION_CANDIDATE_RECORD_INVENTORY_AND_IDENTITY",
        "second ordered next action changed",
    )
    _require(
        next_campaign[-1]
        == "REQUEST_EXPLICIT_COMPUTE_AUTHORIZATION_ONLY_AFTER_CAMPAIGN_IS_DATA_READY",
        "compute authorization is not the final ordered action",
    )
    _require(len(next_campaign) == len(set(next_campaign)), "ordered next actions contain duplicates")

    decision = value["terminal_decision"]
    _require(
        decision["status"] == "BLOCK_LONG_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING",
        "terminal decision is not fail-closed",
    )
    _require(
        decision["primary_blocker"]
        == "RESEARCH_CORPUS_V1_NOT_TERMINAL_AND_ZERO_AUTHORIZED_UNIQUE_LOSS_CAPACITY",
        "primary blocker changed unexpectedly",
    )
    _require(
        "D05_REMEDIATION_CURRENT_HEAD_NOT_YET_GREEN" in decision["secondary_blockers"],
        "D05 current-head blocker is missing",
    )
    _require(
        "NEXT100_075_FULL_11_CASE_PRODUCTION_RERUN_NOT_YET_GREEN"
        in decision["secondary_blockers"],
        "D05 11-case rerun blocker is missing",
    )

    _require(
        canonical_identity(value) == value["evidence_identity_sha256"],
        "readiness evidence identity mismatch",
    )


def load_and_validate(path: Path = CONTROL) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate(value)
    return value


def main() -> int:
    value = load_and_validate()
    print("20M_READINESS_CONTROLLER=PASS")
    print("20M_MODEL_PARAMETERS=" + str(value["authorities"]["primary_20m_model"]["parameter_count"]))
    print("20M_AUTHORIZED_UNIQUE_TARGETS=" + str(value["campaign"]["authorized_unique_optimized_targets"]))
    print("20M_CAMPAIGN_RUNNABLE=" + str(value["campaign"]["campaign_runnable_now"]).lower())
    print("20M_CHECKPOINT_INTEGRITY=" + value["gates"]["checkpoint_integrity_20m"])
    print("20M_SOURCE_REGISTRY=" + value["gates"]["source_registry_convergence"])
    print("20M_NEXT=" + value["ordered_next_campaign"][0])
    print("20M_EVIDENCE_SHA256=" + value["evidence_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
