"""Fail-closed Research Corpus V1 bulk-acquisition rebind against NEXT100-063 V4."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.research-corpus-v1-bulk-rebind.v2"
PARENT_SCHEMA = "12-6.next100-063-terminal-source-registry.v4"
PARENT_DECISION = (
    "CONVERGED_TERMINAL_SOURCE_AUTHORITY_VECTOR_REQUIRES_SUCCESSOR_GLOBAL_DEDUP_NOT_CORPUS_FREEZE"
)
STRATA = ("uk", "en", "code")


class BulkAcquisitionRebindError(ValueError):
    """Raised when the acquisition snapshot no longer matches its source authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BulkAcquisitionRebindError(message)


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
    """Return ceil(value / (numerator / denominator)) using integer arithmetic."""
    return (value * denominator + numerator - 1) // numerator


def validate_rebind(config: Mapping[str, Any], parent: Mapping[str, Any], parent_blob_sha1: str) -> dict[str, Any]:
    _require(config.get("schema_version") == SCHEMA, "unsupported rebind schema")
    _require(config.get("worker_id") == "DATA-BULK-ACQ-V2-REBIND", "worker id drift")
    _require(config.get("local_free_only") is True, "LOCAL_FREE boundary weakened")
    for key in (
        "model_training_executed",
        "optimizer_update_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        _require(config.get(key) is False, f"execution boundary weakened: {key}")

    binding = config.get("parent_authority")
    _require(isinstance(binding, Mapping), "parent_authority is required")
    _require(parent.get("schema_version") == PARENT_SCHEMA, "parent schema drift")
    _require(binding.get("schema_version") == PARENT_SCHEMA, "bound parent schema drift")
    _require(binding.get("config_blob_sha1") == parent_blob_sha1, "parent config blob drift")
    _require(
        binding.get("registry_identity_sha256") == parent.get("registry_identity_sha256"),
        "parent registry identity drift",
    )
    _require(parent.get("decision") == PARENT_DECISION, "parent decision drift")
    _require(binding.get("decision") == PARENT_DECISION, "bound parent decision drift")

    inventory = parent.get("pre_successor_global_dedup_inventory")
    _require(isinstance(inventory, Mapping), "parent pre-dedup inventory missing")
    by_stratum = inventory.get("by_stratum")
    _require(isinstance(by_stratum, Mapping), "parent by-stratum inventory missing")

    parent_bytes: dict[str, int] = {}
    parent_families: dict[str, int] = {}
    for stratum in STRATA:
        row = by_stratum.get(stratum)
        _require(isinstance(row, Mapping), f"parent {stratum} row missing")
        capacity = row.get("numeric_training_capacity_bytes")
        families = row.get("family_count")
        _require(type(capacity) is int and capacity >= 0, f"parent {stratum} capacity invalid")
        _require(type(families) is int and families >= 0, f"parent {stratum} family count invalid")
        parent_bytes[stratum] = capacity
        parent_families[stratum] = families
    parent_bytes["total"] = sum(parent_bytes.values())
    parent_families["total"] = sum(parent_families.values())
    _require(
        parent_bytes["total"] == inventory.get("candidate_numeric_training_capacity_bytes"),
        "parent total capacity arithmetic drift",
    )
    _require(
        parent_families["total"] == inventory.get("candidate_independent_family_count"),
        "parent total family arithmetic drift",
    )

    credited = _vector(
        config.get("credited_pre_successor_global_dedup_bytes"),
        "credited_pre_successor_global_dedup_bytes",
    )
    _require(credited == parent_bytes, "planner credited vector drifted from parent V4")

    declared_families = config.get("credited_independent_families")
    _require(isinstance(declared_families, Mapping), "credited_independent_families missing")
    _require(dict(declared_families) == parent_families, "planner family vector drifted from parent V4")

    targets = _vector(config.get("frozen_targets_bytes"), "frozen_targets_bytes")
    gaps = _vector(config.get("remaining_gap_bytes"), "remaining_gap_bytes")
    expected_gaps = {key: targets[key] - credited[key] for key in STRATA}
    _require(all(value >= 0 for value in expected_gaps.values()), "credited source capacity exceeds target")
    expected_gaps["total"] = sum(expected_gaps.values())
    _require(gaps == expected_gaps, "remaining acquisition gap arithmetic drift")
    _require(
        inventory.get("research_corpus_v1_acquisition_planning_target_bytes") == targets["total"],
        "parent acquisition target drift",
    )
    _require(
        inventory.get("target_gap_numeric_training_capacity_bytes") == gaps["total"],
        "parent acquisition gap drift",
    )

    survival = config.get("planning_survival_ratio")
    _require(isinstance(survival, Mapping), "planning_survival_ratio missing")
    numerator = survival.get("numerator")
    denominator = survival.get("denominator")
    _require(type(numerator) is int and type(denominator) is int, "planning survival ratio must be integer rational")
    _require(0 < numerator <= denominator, "planning survival ratio out of range")
    _require(survival.get("is_measured_retention_evidence") is False, "planning buffer cannot become measured evidence")
    required = {key: _ceil_ratio(gaps[key], numerator, denominator) for key in STRATA}
    required["total"] = sum(required.values())
    declared_required = _vector(config.get("buffered_gross_required_bytes"), "buffered_gross_required_bytes")
    _require(declared_required == required, "buffered gross requirement arithmetic drift")

    planned = _vector(config.get("planned_gross_bytes"), "planned_gross_bytes")
    for stratum in STRATA:
        _require(planned[stratum] >= required[stratum], f"{stratum}: planned gross below buffered requirement")
    headroom = {key: planned[key] - required[key] for key in STRATA}
    headroom["total"] = sum(headroom.values())
    _require(
        _vector(config.get("planning_headroom_bytes"), "planning_headroom_bytes") == headroom,
        "planning headroom arithmetic drift",
    )

    mixture = _vector(config.get("frozen_mixture_percent"), "frozen_mixture_percent")
    _require(mixture == {"uk": 45, "en": 35, "code": 20, "total": 100}, "frozen 45/35/20 mixture drift")
    ceilings = {key: credited[key] * 100 // mixture[key] for key in STRATA}
    limiter = min(ceilings.values())
    declared_ceiling = config.get("current_balanced_source_ceiling_bytes")
    _require(isinstance(declared_ceiling, Mapping), "balanced source ceiling missing")
    _require(
        dict(declared_ceiling)
        == {"uk": ceilings["uk"], "en": ceilings["en"], "code": ceilings["code"], "limiting_total": limiter},
        "balanced source ceiling arithmetic drift",
    )
    priority = sorted(STRATA, key=lambda item: (ceilings[item], item))
    _require(config.get("acquisition_priority") == priority, "acquisition priority no longer matches live limiter order")

    cap = config.get("family_cap_policy")
    _require(isinstance(cap, Mapping), "family_cap_policy missing")
    _require(cap.get("max_global_family_share_percent") == 25, "global family cap drift")
    _require(cap.get("max_within_stratum_family_share_percent") == 60, "stratum family cap drift")
    _require(cap.get("replay_or_duplication_quota_repair_allowed") is False, "replay quota repair illegally enabled")
    gutenberg_family = cap.get("gutenberg_family")
    additions = parent.get("terminal_late_additions")
    _require(isinstance(additions, list), "parent terminal additions missing")
    pg = next((item for item in additions if isinstance(item, Mapping) and item.get("family") == gutenberg_family), None)
    _require(isinstance(pg, Mapping), "bound Gutenberg family missing from parent")
    pg_bytes = pg.get("numeric_training_capacity_bytes")
    _require(pg_bytes == cap.get("gutenberg_exact_bytes") == 1672110, "Gutenberg capacity drift")
    _require(pg.get("source_record_count") == 3, "Gutenberg record count drift")
    _require(pg_bytes * 100 > credited["en"] * 60, "Gutenberg no longer exceeds EN family cap")
    _require(pg_bytes * 100 > credited["total"] * 25, "Gutenberg no longer exceeds global family cap")
    _require(
        cap.get("gutenberg_requires_downselection_or_en_family_diversification") is True,
        "Gutenberg cap mitigation requirement weakened",
    )

    execution = config.get("execution_policy")
    _require(isinstance(execution, Mapping), "execution_policy missing")
    _require(execution.get("prospective_bytes_receive_capacity_credit") is False, "prospective bytes cannot receive credit")
    _require(execution.get("terminal_authority_required_before_capacity_credit") is True, "terminal authority gate weakened")
    _require(execution.get("successor_global_dedup_required_before_corpus_identity") is True, "global dedup gate weakened")

    downstream = parent.get("downstream_gate_vector")
    _require(isinstance(downstream, Mapping), "parent downstream gate vector missing")
    _require(downstream.get("authorized_balanced_no_replay_loss_positions") == 0, "parent unexpectedly authorizes loss positions")
    _require(downstream.get("successor_global_cross_source_exact_near_dedup") == "REQUIRED_NEXT", "parent dedup gate drift")

    boundary = config.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "claim_boundary missing")
    _require(boundary.get("successor_global_dedup_terminal") is False, "successor dedup falsely promoted")
    _require(boundary.get("research_corpus_v1_released") is False, "corpus falsely released")
    _require(boundary.get("authorized_unique_loss_positions") == 0, "training exposure must remain zero")
    for key in (
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "paid_compute_authorized",
        "learned_20m_checkpoint_claimed",
        "learned_100m_checkpoint_claimed",
    ):
        _require(boundary.get(key) is False, f"claim boundary weakened: {key}")

    previous = config.get("supersedes_planning_snapshot")
    _require(isinstance(previous, Mapping), "superseded planning snapshot missing")
    _require(previous.get("pr") == 594, "predecessor PR drift")
    _require(type(previous.get("credited_total_bytes")) is int, "predecessor credited bytes missing")
    _require(type(previous.get("remaining_gap_total_bytes")) is int, "predecessor gap missing")
    _require(credited["total"] > previous["credited_total_bytes"], "V2 does not improve the predecessor source baseline")
    _require(gaps["total"] < previous["remaining_gap_total_bytes"], "V2 does not reduce the predecessor planning gap")

    return {
        "status": "PASS_PLANNING_REBIND_ONLY",
        "parent_registry_identity_sha256": parent["registry_identity_sha256"],
        "credited_pre_successor_global_dedup_bytes": credited,
        "remaining_gap_bytes": gaps,
        "buffered_gross_required_bytes": required,
        "planned_gross_bytes": planned,
        "planning_headroom_bytes": headroom,
        "balanced_source_ceiling_bytes": {**ceilings, "limiting_total": limiter},
        "acquisition_priority": priority,
        "gutenberg_downselection_required": True,
        "research_corpus_v1_released": False,
        "training_authorized": False,
    }


def load_and_validate(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    binding = config.get("parent_authority")
    _require(isinstance(binding, Mapping), "parent_authority is required")
    relative = binding.get("config_path")
    _require(isinstance(relative, str) and relative, "parent config path missing")
    parent_path = repo_root / relative
    _require(parent_path.is_file(), f"parent config missing: {parent_path}")
    parent_payload = parent_path.read_bytes()
    parent = json.loads(parent_payload.decode("utf-8"))
    return validate_rebind(config, parent, _git_blob_sha1(parent_payload))
