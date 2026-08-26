#!/usr/bin/env python3
"""Fail-closed validator for the Research Corpus V1 source-authority intake."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/data/research_corpus_v1_intake_v1.json"

EXPECTED_SCHEMA = "12-6.research-corpus-v1-intake.v1"
EXPECTED_WORKER = "AUTONOMOUS-RESEARCH-CORPUS-V1-INTAKE-20260826"
EXPECTED_PARENT_HEAD = "83d92d50a4380636cc7f1cd41fa9ffd4445dd12e"
EXPECTED_REGISTRY_HEAD = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_REGISTRY_ID = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
EXPECTED_KMU_HEAD = "40950a950b60921fd856af2719e1ae2486d9e892"
EXPECTED_KMU_MANIFEST = "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9"
EXPECTED_CPY_HEAD = "5a6a495a24bce449334cbc5126d0114f61a9f57c"
EXPECTED_CPY_AUTHORITY = "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d"
EXPECTED_CPY_SOURCE_SHA = "64a4ec4fd7574ba4c22e615a032b157e446b9c7f5a7917cb7f10fa214a05bd1a"
EXPECTED_PREFILTER_ENVELOPE = 210_115
EXPECTED_KMU_WORKFLOW = 32997970539
EXPECTED_CPY_WORKFLOW = 32998356906


def canonical_identity(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_identity_sha256", None)
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _unique_sha256(values: object, expected_count: int) -> list[str]:
    assert isinstance(values, list)
    assert len(values) == expected_count
    assert len(set(values)) == expected_count
    assert all(_is_sha256(value) for value in values)
    return values


def validate(value: dict[str, Any]) -> None:
    assert value["schema"] == EXPECTED_SCHEMA
    assert value["worker_id"] == EXPECTED_WORKER
    assert value["repository"] == "Oleksii-debug/12-6-ai."
    assert value["execution_profile"] == "LOCAL_FREE"

    parent = value["parent_readiness_authority"]
    assert parent["head_sha"] == EXPECTED_PARENT_HEAD
    assert parent["primary_20m_parameter_count"] == 20_613_440
    assert parent["decision"] == "BLOCK_LONG_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING"

    incumbent = value["incumbent_registry"]
    assert incumbent["head_sha"] == EXPECTED_REGISTRY_HEAD
    assert incumbent["registry_identity_sha256"] == EXPECTED_REGISTRY_ID
    assert incumbent["snapshot_count"] == 5
    assert incumbent["independent_family_count"] == 4
    assert incumbent["normalized_source_bytes"] == 183_061
    assert incumbent["training_rights"] == "ALLOWED_FOR_ALL_FIVE_BOUND_OBJECTS"
    assert incumbent["evaluation_rights"] == "NOT_SEPARATELY_ADMITTED"

    authorities = value["successor_authorities"]
    kmu = authorities["ua_kmu_secretariat"]
    assert kmu["head_sha"] == EXPECTED_KMU_HEAD
    assert kmu["manifest_identity_sha256"] == EXPECTED_KMU_MANIFEST
    assert kmu["dedicated_workflow_run"] == EXPECTED_KMU_WORKFLOW
    assert kmu["dedicated_workflow_conclusion"] == "success"
    assert kmu["verdict"] == "ADMIT"
    assert kmu["family_id"] == "ua.kmu.portal.secretariat-news"
    assert kmu["record_count"] == 6
    assert kmu["normalized_source_bytes"] == 9_153
    assert kmu["training"] == "ALLOWED_PRETRAINING"
    assert kmu["evaluation"] == "NOT_SEPARATELY_ADMITTED"
    assert kmu["final_test"] == "PROHIBITED"
    _unique_sha256(kmu["normalized_record_sha256"], 6)

    cpy = authorities["en_cpython_docs"]
    assert cpy["head_sha"] == EXPECTED_CPY_HEAD
    assert cpy["authority_identity_sha256"] == EXPECTED_CPY_AUTHORITY
    assert cpy["source_normalized_sha256"] == EXPECTED_CPY_SOURCE_SHA
    assert cpy["dedicated_workflow_run"] == EXPECTED_CPY_WORKFLOW
    assert cpy["dedicated_workflow_conclusion"] == "success"
    assert cpy["verdict"] == "ADMIT"
    assert cpy["family_id"] == "python.cpython.documentation"
    assert cpy["source_normalized_bytes"] == 17_901
    assert cpy["accepted_chunk_count"] == 14
    assert cpy["rejected_chunk_count"] == 2
    assert cpy["rejection_reasons"] == {"pii_phone": 2}
    assert cpy["training"] == "ALLOWED_ONLY_FOR_ACCEPTED_CHUNKS"
    assert cpy["evaluation"] == "NOT_SEPARATELY_ADMITTED"
    assert cpy["code_evaluation_reservation_eligible"] is False
    _unique_sha256(cpy["accepted_normalized_chunk_sha256"], 14)

    diversity = value["candidate_family_diversity"]
    assert diversity["required_min_independent_families_per_stratum"] == 2
    expected_families = {
        "uk": ["ua.rada.open-data.laws-texts", "ua.kmu.portal.secretariat-news"],
        "en": ["en.standardebooks.manual", "python.cpython.documentation"],
        "code": ["github:encode/httpx", "github:psf/requests"],
    }
    assert diversity["families_by_stratum_after_authority_intake"] == expected_families
    assert diversity["counts"] == {"uk": 2, "en": 2, "code": 2}
    assert diversity["authority_level_gate"] == "PASS"
    assert diversity["corpus_level_gate"].startswith("NOT_YET_EVALUABLE")
    all_families = [family for group in expected_families.values() for family in group]
    assert len(all_families) == len(set(all_families)) == 6

    capacity = value["capacity_accounting"]
    assert capacity["pre_filter_source_envelope_bytes"] == (
        capacity["incumbent_registry_normalized_source_bytes"]
        + capacity["kmu_normalized_source_bytes"]
        + capacity["cpython_source_normalized_bytes_before_rejected_chunk_exclusion"]
    )
    assert capacity["pre_filter_source_envelope_bytes"] == EXPECTED_PREFILTER_ENVELOPE
    assert capacity["pre_filter_source_envelope_is_training_capacity"] is False
    assert capacity["exact_training_eligible_bytes"] is None
    assert capacity["exact_postpack_unique_nonignored_loss_positions"] is None
    assert capacity["authorized_unique_optimized_targets"] == 0

    gates = value["gates"]
    assert gates["source_authority_intake"] == "PASS"
    assert gates["family_diversity_authority_level"] == "PASS_2_2_2"
    assert gates["exact_candidate_record_inventory"] == "BLOCKED_NOT_MATERIALIZED_ON_ONE_BRANCH"
    assert gates["exact_candidate_corpus_identity"] == "BLOCKED"
    assert gates["evaluation_decontamination"] == "BLOCKED_UNTIL_EXACT_CANDIDATE_IDENTITY"
    assert gates["unique_loss_ledger"] == "BLOCKED"
    assert gates["real_20m_training"] == "BLOCKED"

    steps = value["ordered_next_steps"]
    assert len(steps) == 8
    assert len(steps) == len(set(steps))
    assert steps[0].startswith("MATERIALIZE_EXACT_ACCEPTED_KMU_AND_CPYTHON_OBJECTS")
    assert steps[-1].startswith("REFRESH_20M_CAMPAIGN_PREREGISTRATION")

    boundary = value["claim_boundary"]
    assert boundary == {
        "corpus_frozen": False,
        "final_test_outcomes_read": False,
        "learned_20m_checkpoint_created": False,
        "long_training_authorized": False,
        "optimizer_updates": 0,
        "paid_compute_used": False,
        "representative_corpus_claimed": False,
        "stage_promotion_claimed": False,
        "training_executed": False,
    }

    decision = value["terminal_decision"]
    assert decision["status"] == "SOURCE_AUTHORITY_INTAKE_READY_MATERIALIZATION_REQUIRED"
    assert canonical_identity(value) == value["evidence_identity_sha256"]


def load_and_validate(path: Path = MANIFEST) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate(value)
    return value


def main() -> int:
    value = load_and_validate()
    print("RESEARCH_CORPUS_V1_INTAKE=PASS")
    print("AUTHORITY_LEVEL_FAMILY_COUNTS=2/2/2")
    print("PREFILTER_SOURCE_ENVELOPE_BYTES=" + str(value["capacity_accounting"]["pre_filter_source_envelope_bytes"]))
    print("AUTHORIZED_UNIQUE_OPTIMIZED_TARGETS=0")
    print("REAL_20M_TRAINING=BLOCKED")
    print("EVIDENCE_SHA256=" + value["evidence_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
