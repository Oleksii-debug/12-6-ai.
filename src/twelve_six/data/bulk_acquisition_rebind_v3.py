"""Fail-closed Research Corpus V1 bulk-acquisition rebind against NEXT100-063 V5.

Integration/planning authority only: this module cannot materialize a corpus,
authorize tokenizer fitting/training, or convert source bytes into loss positions.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.research-corpus-v1-bulk-rebind.v3"
WORKER_ID = "D10-DATA-BULK-ACQ-V3-V5-REBIND"
PARENT_SCHEMA = "12-6.next100-063-terminal-source-registry.v5"
PARENT_DECISION = "V5_CONSUMES_TERMINAL_ATTRS_REQUIRES_SUCCESSOR_GLOBAL_DEDUP_NOT_CORPUS_FREEZE"
PARENT_PR = 538
PARENT_HEAD = "991a0b6e939cddeff16c075922f7c407fa1e86cb"
PARENT_RUN = 33046314943
BASE_V4_BLOB = "60924a7cb76dc76bbff26a340184f54a2c374c83"
BASE_V4_IDENTITY = "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58"
DEDUP_PR = 632
DEDUP_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
DEDUP_RUN = 33045763964
DEDUP_ARTIFACT_ID = 9635595510
DEDUP_ARTIFACT_DIGEST = "sha256:cca6921a2093d4e033976b23b0af180e9dc1945b624b82e218780f8d20bafd18"
STRATA = ("uk", "en", "code")


class BulkAcquisitionRebindV3Error(ValueError):
    """Raised when a planning or evidence binding drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BulkAcquisitionRebindV3Error(message)


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _vector(value: Any, field: str) -> dict[str, int]:
    _require(isinstance(value, Mapping), f"{field}: expected object")
    expected = {*STRATA, "total"}
    _require(set(value) == expected, f"{field}: unexpected keys")
    result: dict[str, int] = {}
    for key in (*STRATA, "total"):
        item = value[key]
        _require(type(item) is int and item >= 0, f"{field}.{key}: expected non-negative integer")
        result[key] = item
    _require(result["total"] == sum(result[key] for key in STRATA), f"{field}: total arithmetic drift")
    return result


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    return (value * denominator + numerator - 1) // numerator


def _parent_vectors(parent: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    inventory = parent.get("derived_pre_successor_global_dedup_inventory")
    _require(isinstance(inventory, Mapping), "parent V5 inventory missing")
    by_stratum = inventory.get("by_stratum")
    _require(isinstance(by_stratum, Mapping), "parent V5 by-stratum inventory missing")
    capacities: dict[str, int] = {}
    families: dict[str, int] = {}
    for stratum in STRATA:
        row = by_stratum.get(stratum)
        _require(isinstance(row, Mapping), f"parent V5 {stratum} row missing")
        capacity = row.get("numeric_training_capacity_bytes")
        family_count = row.get("family_count")
        _require(type(capacity) is int and capacity >= 0, f"parent V5 {stratum} capacity invalid")
        _require(type(family_count) is int and family_count >= 0, f"parent V5 {stratum} family count invalid")
        capacities[stratum] = capacity
        families[stratum] = family_count
    capacities["total"] = sum(capacities.values())
    families["total"] = sum(families.values())
    _require(capacities["total"] == inventory.get("candidate_numeric_training_capacity_bytes"), "parent V5 total capacity arithmetic drift")
    _require(families["total"] == inventory.get("candidate_independent_family_count"), "parent V5 total family arithmetic drift")
    return capacities, families


def validate_rebind(config: Mapping[str, Any], parent: Mapping[str, Any], parent_blob_sha1: str, base_v4: Mapping[str, Any], base_v4_blob_sha1: str) -> dict[str, Any]:
    _require(config.get("schema_version") == SCHEMA, "unsupported rebind schema")
    _require(config.get("worker_id") == WORKER_ID, "worker id drift")
    _require(config.get("local_free_only") is True, "LOCAL_FREE boundary weakened")
    for key in ("model_training_executed", "optimizer_update_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
        _require(config.get(key) is False, f"execution boundary weakened: {key}")

    binding = config.get("parent_authority")
    _require(isinstance(binding, Mapping), "parent_authority is required")
    _require(binding.get("pr") == PARENT_PR, "parent PR drift")
    _require(binding.get("observed_head_sha") == PARENT_HEAD, "parent exact head drift")
    _require(binding.get("dedicated_workflow_run") == PARENT_RUN, "parent exact workflow run drift")
    _require(binding.get("dedicated_workflow_conclusion") == "success", "parent workflow is not terminal success")
    _require(parent.get("schema_version") == PARENT_SCHEMA, "parent schema drift")
    _require(binding.get("schema_version") == PARENT_SCHEMA, "bound parent schema drift")
    _require(binding.get("config_blob_sha1") == parent_blob_sha1, "parent config blob drift")
    _require(parent.get("decision") == PARENT_DECISION, "parent decision drift")
    _require(binding.get("decision") == PARENT_DECISION, "bound parent decision drift")

    base_binding = parent.get("base_v4")
    _require(isinstance(base_binding, Mapping), "parent V5 base_v4 binding missing")
    _require(base_binding.get("path") == binding.get("base_v4_path"), "base V4 path drift")
    _require(base_binding.get("registry_identity_sha256") == BASE_V4_IDENTITY, "parent V5 base identity drift")
    _require(binding.get("base_v4_registry_identity_sha256") == BASE_V4_IDENTITY, "bound base V4 identity drift")
    _require(binding.get("base_v4_blob_sha1") == BASE_V4_BLOB, "bound base V4 blob drift")
    _require(base_v4_blob_sha1 == BASE_V4_BLOB, "base V4 checkout blob drift")
    _require(base_v4.get("registry_identity_sha256") == BASE_V4_IDENTITY, "base V4 registry identity drift")

    parent_bytes, parent_families = _parent_vectors(parent)
    credited = _vector(config.get("credited_post_successor_global_dedup_bytes"), "credited_post_successor_global_dedup_bytes")
    declared_families = _vector(config.get("credited_independent_families"), "credited_independent_families")
    _require(credited == parent_bytes, "planner credited vector drifted from parent V5")
    _require(declared_families == parent_families, "planner family vector drifted from parent V5")

    dedup = config.get("observed_successor_global_dedup")
    _require(isinstance(dedup, Mapping), "successor global-dedup authority missing")
    _require(dedup.get("pr") == DEDUP_PR, "dedup PR drift")
    _require(dedup.get("observed_head_sha") == DEDUP_HEAD, "dedup exact head drift")
    _require(dedup.get("dedicated_workflow_run") == DEDUP_RUN, "dedup exact workflow run drift")
    _require(dedup.get("dedicated_workflow_conclusion") == "success", "dedup workflow is not terminal success")
    _require(dedup.get("artifact_id") == DEDUP_ARTIFACT_ID, "dedup artifact id drift")
    _require(dedup.get("artifact_digest") == DEDUP_ARTIFACT_DIGEST, "dedup artifact digest drift")
    _require(dedup.get("conservative_unique_capacity_bytes") == credited["total"], "dedup unique capacity does not match parent V5 source vector")
    _require(dedup.get("duplicate_discount_bytes") == 0, "unexpected duplicate discount drift")
    _require(dedup.get("corpus_materialized") is False, "dedup evidence cannot masquerade as corpus materialization")
    _require(dedup.get("training_authorized") is False, "dedup evidence cannot authorize training")

    targets = _vector(config.get("frozen_targets_bytes"), "frozen_targets_bytes")
    gaps = _vector(config.get("remaining_gap_bytes"), "remaining_gap_bytes")
    expected_gaps = {key: targets[key] - credited[key] for key in STRATA}
    _require(all(value >= 0 for value in expected_gaps.values()), "credited unique capacity exceeds target")
    expected_gaps["total"] = sum(expected_gaps.values())
    _require(gaps == expected_gaps, "remaining acquisition gap arithmetic drift")
    inventory = parent["derived_pre_successor_global_dedup_inventory"]
    _require(inventory.get("research_corpus_v1_acquisition_planning_target_bytes") == targets["total"], "parent acquisition target drift")
    _require(inventory.get("target_gap_numeric_training_capacity_bytes") == gaps["total"], "parent acquisition gap drift")

    survival = config.get("planning_survival_ratio")
    _require(isinstance(survival, Mapping), "planning_survival_ratio missing")
    numerator, denominator = survival.get("numerator"), survival.get("denominator")
    _require(type(numerator) is int and type(denominator) is int, "planning survival ratio must be integer rational")
    _require(0 < numerator <= denominator, "planning survival ratio out of range")
    _require(survival.get("is_measured_retention_evidence") is False, "planning buffer cannot become measured evidence")
    required = {key: _ceil_ratio(gaps[key], numerator, denominator) for key in STRATA}
    required["total"] = sum(required.values())
    _require(_vector(config.get("buffered_gross_required_bytes"), "buffered_gross_required_bytes") == required, "buffered gross requirement arithmetic drift")

    planned = _vector(config.get("planned_gross_bytes"), "planned_gross_bytes")
    for stratum in STRATA:
        _require(planned[stratum] >= required[stratum], f"{stratum}: planned gross below buffered requirement")
    headroom = {key: planned[key] - required[key] for key in STRATA}
    headroom["total"] = sum(headroom.values())
    _require(_vector(config.get("planning_headroom_bytes"), "planning_headroom_bytes") == headroom, "planning headroom arithmetic drift")

    mixture = _vector(config.get("frozen_mixture_percent"), "frozen_mixture_percent")
    _require(mixture == {"uk": 45, "en": 35, "code": 20, "total": 100}, "frozen 45/35/20 mixture drift")
    ceilings = {key: credited[key] * 100 // mixture[key] for key in STRATA}
    limiter = min(ceilings.values())
    declared_ceiling = config.get("current_balanced_source_ceiling_bytes")
    _require(isinstance(declared_ceiling, Mapping), "balanced source ceiling missing")
    _require(dict(declared_ceiling) == {"uk": ceilings["uk"], "en": ceilings["en"], "code": ceilings["code"], "limiting_total": limiter}, "balanced source ceiling arithmetic drift")
    priority = sorted(STRATA, key=lambda item: (ceilings[item], item))
    _require(config.get("acquisition_priority") == priority, "acquisition priority no longer matches live limiter order")

    cap = config.get("family_cap_policy")
    _require(isinstance(cap, Mapping), "family_cap_policy missing")
    _require(cap.get("max_global_family_share_percent") == 25, "global family cap drift")
    _require(cap.get("max_within_stratum_family_share_percent") == 60, "stratum family cap drift")
    _require(cap.get("replay_or_duplication_quota_repair_allowed") is False, "replay quota repair illegally enabled")
    additions = base_v4.get("terminal_late_additions")
    _require(isinstance(additions, list), "base V4 terminal additions missing")
    gutenberg_family = cap.get("gutenberg_family")
    pg = next((item for item in additions if isinstance(item, Mapping) and item.get("family") == gutenberg_family), None)
    _require(isinstance(pg, Mapping), "bound Gutenberg family missing from base V4")
    pg_bytes = pg.get("numeric_training_capacity_bytes")
    _require(pg_bytes == cap.get("gutenberg_exact_bytes") == 1672110, "Gutenberg capacity drift")
    _require(pg.get("source_record_count") == 3, "Gutenberg record count drift")
    _require(pg_bytes * 100 > credited["en"] * 60, "Gutenberg no longer exceeds EN family cap")
    _require(pg_bytes * 100 > credited["total"] * 25, "Gutenberg no longer exceeds global family cap")
    _require(cap.get("gutenberg_requires_downselection_or_en_family_diversification") is True, "Gutenberg cap mitigation requirement weakened")

    execution = config.get("execution_policy")
    _require(isinstance(execution, Mapping), "execution_policy missing")
    _require(execution.get("prospective_bytes_receive_capacity_credit") is False, "prospective bytes cannot receive credit")
    _require(execution.get("terminal_authority_required_before_capacity_credit") is True, "terminal authority gate weakened")
    _require(execution.get("successor_global_dedup_required_before_corpus_identity") is True, "global dedup gate weakened")
    _require(execution.get("materialized_record_graph_required_before_corpus_identity") is True, "record materialization gate weakened")

    downstream = parent.get("downstream_gate_vector")
    _require(isinstance(downstream, Mapping), "parent downstream gate vector missing")
    _require(downstream.get("authorized_balanced_no_replay_loss_positions") == 0, "parent unexpectedly authorizes loss positions")
    _require(downstream.get("successor_global_cross_source_exact_near_dedup") == "REQUIRED_NEXT", "parent dedup handoff drift")

    boundary = config.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "claim_boundary missing")
    _require(boundary.get("successor_global_dedup_terminal_for_current_source_vector") is True, "terminal dedup evidence dropped")
    _require(boundary.get("corpus_materialized") is False, "corpus falsely materialized")
    _require(boundary.get("research_corpus_v1_released") is False, "corpus falsely released")
    _require(boundary.get("authorized_unique_loss_positions") == 0, "training exposure must remain zero")
    for key in ("tokenizer_fit_authorized", "model_training_authorized", "paid_compute_authorized", "learned_20m_checkpoint_claimed", "learned_100m_checkpoint_claimed"):
        _require(boundary.get(key) is False, f"claim boundary weakened: {key}")

    previous = config.get("supersedes_planning_snapshot")
    _require(isinstance(previous, Mapping), "superseded planning snapshot missing")
    _require(previous.get("branch") == "data/bulk-acq-v2-rebind-20260826", "predecessor branch drift")
    _require(previous.get("head_sha") == "9452e328356a382192ed9f6da294426389515fc8", "predecessor exact head drift")
    _require(credited["total"] > previous.get("credited_total_bytes", -1), "V3 does not improve predecessor unique baseline")
    _require(gaps["total"] < previous.get("remaining_gap_total_bytes", 10**30), "V3 does not reduce predecessor planning gap")

    return {"status": "PASS_V5_DEDUP_PLANNING_REBIND_ONLY", "parent_head": PARENT_HEAD, "parent_config_blob_sha1": parent_blob_sha1, "dedup_head": DEDUP_HEAD, "dedup_artifact_digest": DEDUP_ARTIFACT_DIGEST, "credited_post_successor_global_dedup_bytes": credited, "remaining_gap_bytes": gaps, "buffered_gross_required_bytes": required, "planned_gross_bytes": planned, "planning_headroom_bytes": headroom, "balanced_source_ceiling_bytes": {**ceilings, "limiting_total": limiter}, "acquisition_priority": priority, "corpus_materialized": False, "research_corpus_v1_released": False, "authorized_unique_loss_positions": 0, "training_authorized": False}


def load_and_validate(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    binding = config.get("parent_authority")
    _require(isinstance(binding, Mapping), "parent_authority is required")
    parent_relative = binding.get("config_path")
    _require(isinstance(parent_relative, str) and parent_relative, "parent config path missing")
    parent_path = repo_root / parent_relative
    _require(parent_path.is_file(), f"parent config missing: {parent_path}")
    parent_payload = parent_path.read_bytes()
    parent = json.loads(parent_payload.decode("utf-8"))
    base_relative = binding.get("base_v4_path")
    _require(isinstance(base_relative, str) and base_relative, "base V4 path missing")
    base_path = repo_root / base_relative
    _require(base_path.is_file(), f"base V4 config missing: {base_path}")
    base_payload = base_path.read_bytes()
    base_v4 = json.loads(base_payload.decode("utf-8"))
    return validate_rebind(config, parent, _git_blob_sha1(parent_payload), base_v4, _git_blob_sha1(base_payload))
