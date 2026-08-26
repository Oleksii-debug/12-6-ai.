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
EXPECTED_D05_REMEDIATION_HEAD = "42296f6f228bbf866765d787e53004108fe7a39d"


def canonical_identity(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_identity_sha256", None)
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(value: dict[str, Any]) -> None:
    assert value["schema"] == EXPECTED_SCHEMA
    assert value["worker_id"] == EXPECTED_WORKER
    assert value["repository"] == "Oleksii-debug/12-6-ai."
    assert value["execution_profile"] == "LOCAL_FREE"
    assert value["training_executed"] is False
    assert value["paid_compute_used"] is False
    assert value["long_training_authorized"] is False

    authorities = value["authorities"]

    model = authorities["primary_20m_model"]
    assert model["head_sha"] == EXPECTED_MODEL_HEAD
    assert model["qualification"] == "PASS"
    assert model["random_init_only"] is True
    assert model["long_training_performed"] is False
    assert model["parameter_count"] == EXPECTED_MODEL_PARAMS
    assert model["model_spec_sha256"] == EXPECTED_MODEL_SPEC
    assert all(state == "PASS" for state in model["mechanics"].values())

    # Nominal save/load mechanics passing is not a corruption-integrity PASS.
    # NEXT100-075 is the independent fail-closed authority until its full matrix
    # is rerun against one exact remediated production head.
    d05 = authorities["checkpoint_corruption_audit"]
    assert d05["head_sha"] == EXPECTED_D05_AUDIT_HEAD
    assert d05["base_model_sha"] == EXPECTED_MODEL_HEAD
    assert d05["required_cases"] == 11
    assert d05["rejected_before_mutation"] == 8
    assert len(d05["blocking_cases"]) == 3
    assert d05["verdict"] == "RETEST_REQUIRED"
    assert d05["direct_production_module_rerun_required"] is True

    remediation = authorities["d05_remediation_convergence"]
    assert remediation["head_sha"] == EXPECTED_D05_REMEDIATION_HEAD
    assert remediation["preferred_candidate_pr"] == 507
    assert remediation["terminal_authority"] is False
    assert remediation["state"].startswith("NONTERMINAL_")

    corpus = authorities["research_corpus_v03"]
    assert corpus["head_sha"] == EXPECTED_CORPUS_HEAD
    assert corpus["state"] == "TERMINAL_BLOCKED"
    assert corpus["corpus_identity"] is None
    assert corpus["shard_identity"] is None
    assert corpus["authorized_balanced_no_replay_capacity"] == 0

    ua = authorities["ua_wikisource_source"]
    assert ua["head_sha"] == EXPECTED_UA_SOURCE_HEAD
    assert ua["authority_state"] == "TERMINAL_RIGHTS_AND_IMMUTABLE_SNAPSHOT"
    assert ua["training_selection"] == "BLOCKED_UNTIL_STANDARD_NEAR_MATCH_DECONTAMINATION"

    en = authorities["en_python_docs_source"]
    assert en["head_sha"] == EXPECTED_EN_SOURCE_HEAD
    assert en["terminal_verdict"] == "ADMIT"
    assert en["training_use"] == "ALLOWED"
    assert en["accepted_chunks"] > 0

    decontam = authorities["decontamination_v4"]
    assert decontam["head_sha"] == EXPECTED_DECONTAM_HEAD
    assert decontam["state"] == "BLOCKED_NO_EXACT_CANDIDATE_CORPUS_IDENTITY"
    assert decontam["scan_executed"] is False

    selection = authorities["selection_validation_composite"]
    assert selection["state"] == "NONEMPTY_SELECTION_AUTHORITY"
    assert sum(selection["records"].values()) > 0

    code_selection = authorities["code_selection_validation_v2"]
    assert code_selection["eligible_objects"] == 0
    assert code_selection["eligible_families"] == 0

    campaign = value["campaign"]
    assert campaign["authorized_unique_optimized_targets"] == 0
    assert campaign["campaign_runnable_now"] is False
    assert campaign["long_training_started"] is False
    assert campaign["decision"] == "DO_NOT_START_LONG_TRAINING"

    gates = value["gates"]
    assert gates["primary_20m_mechanics"] == "PASS"
    assert gates["checkpoint_recovery_20m"] == "BLOCKED_D05_CORRUPTION_RETEST_REQUIRED"
    assert gates["checkpoint_integrity_20m"] == "BLOCKED_3_OF_11_CORRUPTION_CLASSES"
    assert gates["real_20m_training"] == "BLOCKED"
    assert gates["unique_no_replay_loss_capacity"] == "BLOCKED_ZERO"

    parallel_tracks = value["parallel_local_free_tracks"]
    assert "CONVERGE_D05_REMEDIATION_AND_TRIAGE_LOCKED_ENVIRONMENT_CHECK_FAILURE" in parallel_tracks
    assert "COMPOSE_SUCCESSOR_RESEARCH_CORPUS_V1_INTAKE" in parallel_tracks
    assert len(parallel_tracks) == len(set(parallel_tracks))

    next_campaign = value["ordered_next_campaign"]
    assert next_campaign[0] == "COMPOSE_SUCCESSOR_RESEARCH_CORPUS_V1_INTAKE_FROM_TERMINAL_SOURCE_AUTHORITIES"
    assert next_campaign[-1] == "REQUEST_EXPLICIT_COMPUTE_AUTHORIZATION_ONLY_AFTER_CAMPAIGN_IS_DATA_READY"
    assert len(next_campaign) == len(set(next_campaign))

    decision = value["terminal_decision"]
    assert decision["status"] == "BLOCK_LONG_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING"
    assert decision["primary_blocker"] == (
        "RESEARCH_CORPUS_V1_NOT_TERMINAL_AND_ZERO_AUTHORIZED_UNIQUE_LOSS_CAPACITY"
    )
    assert "D05_CHECKPOINT_INTEGRITY_RETEST_REQUIRED_ON_3_OF_11_CORRUPTION_CLASSES" in decision["secondary_blockers"]

    assert canonical_identity(value) == value["evidence_identity_sha256"]


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
    print("20M_NEXT=" + value["ordered_next_campaign"][0])
    print("20M_EVIDENCE_SHA256=" + value["evidence_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
