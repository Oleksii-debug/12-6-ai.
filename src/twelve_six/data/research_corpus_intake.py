from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


class ResearchCorpusIntakeError(ValueError):
    """Raised when a Research Corpus V1 intake authority violates its truth boundary."""


_ALLOWED_STRATA = ("ua", "en", "code")
_ALLOWED_TRAINING_RIGHTS = {
    "ALLOWED",
    "ALLOWED_PRETRAINING",
    "ALLOWED_WITH_ATTRIBUTION",
    "ALLOWED_WITH_NIST_SOURCE_PROVENANCE",
    "ALLOWED_WITH_NOTICE",
    "CONDITIONED_ON_STANDARD_NEAR_MATCH_DECONTAMINATION",
}
_REQUIRED_FALSE_TRUTH_FLAGS = (
    "exact_predecontamination_inventory_frozen",
    "global_cross_source_dedup_passed",
    "evaluation_decontamination_passed",
    "privacy_quality_pipeline_passed",
    "train_val_split_frozen",
    "two_clean_builds_match",
    "unique_no_replay_loss_ledger_terminal",
    "final_corpus_ready",
    "training_authorized",
    "paid_compute_authorized",
)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchCorpusIntakeError(f"{field} must be an object")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchCorpusIntakeError(f"{field} must be a non-empty string")
    return value


def _require_sha(value: Any, field: str, length: int) -> str:
    text = _require_nonempty_string(value, field)
    if len(text) != length or any(ch not in "0123456789abcdef" for ch in text):
        raise ResearchCorpusIntakeError(
            f"{field} must be a lowercase {length}-hex digest"
        )
    return text


def _validate_source_record(
    record: Mapping[str, Any], field: str, *, candidate: bool
) -> None:
    _require_nonempty_string(record.get("source_id"), f"{field}.source_id")
    _require_nonempty_string(record.get("family_id"), f"{field}.family_id")

    stratum = record.get("stratum")
    if stratum not in _ALLOWED_STRATA:
        raise ResearchCorpusIntakeError(
            f"{field}.stratum must be one of {', '.join(_ALLOWED_STRATA)}"
        )

    payload_bytes = record.get("payload_bytes")
    if (
        not isinstance(payload_bytes, int)
        or isinstance(payload_bytes, bool)
        or payload_bytes <= 0
    ):
        raise ResearchCorpusIntakeError(
            f"{field}.payload_bytes must be a positive integer"
        )

    if record.get("terminal_evidence") is not True:
        raise ResearchCorpusIntakeError(f"{field}.terminal_evidence must be true")

    training_rights = record.get("training_rights")
    if training_rights not in _ALLOWED_TRAINING_RIGHTS:
        raise ResearchCorpusIntakeError(
            f"{field}.training_rights is not intake-eligible"
        )

    evaluation_rights = _require_nonempty_string(
        record.get("evaluation_rights"), f"{field}.evaluation_rights"
    )
    if not evaluation_rights.startswith("NOT_"):
        raise ResearchCorpusIntakeError(
            f"{field}.evaluation_rights must remain fail-closed for training intake"
        )

    if not candidate:
        return

    authority_pr = record.get("authority_pr")
    if (
        not isinstance(authority_pr, int)
        or isinstance(authority_pr, bool)
        or authority_pr <= 0
    ):
        raise ResearchCorpusIntakeError(
            f"{field}.authority_pr must be a positive integer"
        )
    _require_sha(record.get("authority_head_sha"), f"{field}.authority_head_sha", 40)

    authority_identity = record.get("authority_identity_sha256")
    if authority_identity is not None:
        _require_sha(authority_identity, f"{field}.authority_identity_sha256", 64)

    if record.get("admission_state") not in {
        "TERMINAL_ADMIT",
        "TERMINAL_RIGHTS_SNAPSHOT_DECONTAM_REQUIRED",
    }:
        raise ResearchCorpusIntakeError(f"{field}.admission_state is not terminal")

    if record.get("requires_standard_cross_source_dedup") is not True:
        raise ResearchCorpusIntakeError(
            f"{field}.requires_standard_cross_source_dedup must be true"
        )
    if record.get("requires_standard_eval_decontamination") is not True:
        raise ResearchCorpusIntakeError(
            f"{field}.requires_standard_eval_decontamination must be true"
        )


def validate_research_corpus_intake(authority: Mapping[str, Any]) -> None:
    """Validate an intake authority without upgrading any downstream corpus gate."""

    expected_schema = "12-6.research-corpus-v1-intake-convergence.v1"
    if authority.get("schema") != expected_schema:
        raise ResearchCorpusIntakeError("unexpected intake schema")
    if authority.get("execution_profile") != "LOCAL_FREE":
        raise ResearchCorpusIntakeError(
            "intake convergence must remain LOCAL_FREE"
        )

    controller = _require_mapping(
        authority.get("controller_binding"), "controller_binding"
    )
    _require_sha(
        controller.get("base_head_sha"), "controller_binding.base_head_sha", 40
    )
    expected_step = (
        "COMPOSE_SUCCESSOR_RESEARCH_CORPUS_V1_INTAKE_FROM_TERMINAL_SOURCE_AUTHORITIES"
    )
    if controller.get("ordered_campaign_step") != expected_step:
        raise ResearchCorpusIntakeError(
            "intake is not bound to the controller's first campaign step"
        )

    baseline = _require_mapping(
        authority.get("baseline_registry"), "baseline_registry"
    )
    _require_sha(baseline.get("head_sha"), "baseline_registry.head_sha", 40)
    _require_sha(
        baseline.get("registry_identity_sha256"),
        "baseline_registry.registry_identity_sha256",
        64,
    )

    baseline_sources = baseline.get("sources")
    candidates = authority.get("terminal_candidate_authorities")
    if not isinstance(baseline_sources, list) or not baseline_sources:
        raise ResearchCorpusIntakeError(
            "baseline_registry.sources must be a non-empty list"
        )
    if not isinstance(candidates, list) or not candidates:
        raise ResearchCorpusIntakeError(
            "terminal_candidate_authorities must be a non-empty list"
        )

    seen_source_ids: set[str] = set()
    for index, raw_record in enumerate(baseline_sources):
        field = f"baseline_registry.sources[{index}]"
        record = _require_mapping(raw_record, field)
        _validate_source_record(record, field, candidate=False)
        source_id = str(record["source_id"])
        if source_id in seen_source_ids:
            raise ResearchCorpusIntakeError(f"duplicate source_id: {source_id}")
        seen_source_ids.add(source_id)

    for index, raw_record in enumerate(candidates):
        field = f"terminal_candidate_authorities[{index}]"
        record = _require_mapping(raw_record, field)
        _validate_source_record(record, field, candidate=True)
        source_id = str(record["source_id"])
        if source_id in seen_source_ids:
            raise ResearchCorpusIntakeError(f"duplicate source_id: {source_id}")
        seen_source_ids.add(source_id)

    proxy = _require_mapping(authority.get("capacity_proxy"), "capacity_proxy")
    target_bytes = _require_mapping(
        proxy.get("target_bytes"), "capacity_proxy.target_bytes"
    )
    if set(target_bytes) != set(_ALLOWED_STRATA):
        raise ResearchCorpusIntakeError(
            "capacity_proxy.target_bytes must cover ua/en/code exactly"
        )
    for stratum in _ALLOWED_STRATA:
        value = target_bytes[stratum]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ResearchCorpusIntakeError(
                f"capacity_proxy.target_bytes.{stratum} must be a positive integer"
            )

    minimum_families = proxy.get("minimum_independent_families_per_stratum")
    if (
        not isinstance(minimum_families, int)
        or isinstance(minimum_families, bool)
    ):
        raise ResearchCorpusIntakeError(
            "capacity_proxy.minimum_independent_families_per_stratum must be an integer"
        )
    if minimum_families < 2:
        raise ResearchCorpusIntakeError(
            "minimum family diversity cannot be weakened below 2"
        )

    truth = _require_mapping(
        authority.get("hard_truth_boundary"), "hard_truth_boundary"
    )
    for flag in _REQUIRED_FALSE_TRUTH_FLAGS:
        if truth.get(flag) is not False:
            raise ResearchCorpusIntakeError(
                f"hard_truth_boundary.{flag} must remain false"
            )
    if truth.get("authorized_unique_optimized_targets") != 0:
        raise ResearchCorpusIntakeError(
            "authorized_unique_optimized_targets must remain zero at intake convergence"
        )


def build_research_corpus_intake_report(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic lower-bound readiness report from terminal intake metadata."""

    validate_research_corpus_intake(authority)

    baseline = authority["baseline_registry"]["sources"]
    candidates = authority["terminal_candidate_authorities"]
    all_records = [*baseline, *candidates]

    byte_totals: dict[str, int] = defaultdict(int)
    families: dict[str, set[str]] = defaultdict(set)
    for record in all_records:
        stratum = str(record["stratum"])
        byte_totals[stratum] += int(record["payload_bytes"])
        families[stratum].add(str(record["family_id"]))

    target_bytes = authority["capacity_proxy"]["target_bytes"]
    minimum_families = authority["capacity_proxy"][
        "minimum_independent_families_per_stratum"
    ]

    totals = {stratum: byte_totals[stratum] for stratum in _ALLOWED_STRATA}
    gaps = {
        stratum: max(0, int(target_bytes[stratum]) - totals[stratum])
        for stratum in _ALLOWED_STRATA
    }
    family_lists = {
        stratum: sorted(families[stratum]) for stratum in _ALLOWED_STRATA
    }
    family_counts = {
        stratum: len(family_lists[stratum]) for stratum in _ALLOWED_STRATA
    }
    projected_family_gate = {
        stratum: (
            "PROJECTED_PASS_NOT_TERMINAL"
            if family_counts[stratum] >= minimum_families
            else "PROJECTED_FAIL"
        )
        for stratum in _ALLOWED_STRATA
    }

    return {
        "schema": "12-6.research-corpus-v1-intake-report.v1",
        "intake_manifest_valid": True,
        "known_terminal_intake_lower_bound_bytes": totals,
        "known_terminal_intake_total_bytes": sum(totals.values()),
        "capacity_proxy_gap_bytes": gaps,
        "capacity_proxy_total_gap_bytes": sum(gaps.values()),
        "projected_independent_families_if_global_dedup_does_not_collapse_lineage": {
            "counts": family_counts,
            "families": family_lists,
            "gate": projected_family_gate,
        },
        "hard_gates": {
            "exact_predecontamination_candidate_identity": "BLOCKED_NOT_FROZEN",
            "global_cross_source_dedup": "BLOCKED_NOT_RUN_ON_COMPOSED_INVENTORY",
            "evaluation_decontamination": "BLOCKED_NOT_RUN",
            "privacy_quality_pipeline": "BLOCKED_NOT_RUN_ON_COMPOSED_INVENTORY",
            "train_val_split": "BLOCKED_NOT_FROZEN",
            "two_clean_builds": "BLOCKED_NOT_RUN",
            "unique_no_replay_loss_ledger": "BLOCKED_ZERO_AUTHORIZED_TARGETS",
            "final_corpus": "BLOCKED",
            "real_20m_training": "BLOCKED",
        },
        "authorized_unique_optimized_targets": 0,
        "training_authorized": False,
        "paid_compute_authorized": False,
        "next_actions": [
            "FREEZE_EXACT_PRE_DECONTAMINATION_CANDIDATE_RECORD_INVENTORY_AND_IDENTITY",
            "RUN_STANDARD_EXACT_NEAR_MATCH_EVALUATION_DECONTAMINATION",
            "RUN_GLOBAL_CROSS_SOURCE_DEDUP_AND_FAMILY_COLLAPSE",
            "EXPAND_TERMINAL_SOURCE_CAPACITY_TO_CLOSE_BYTE_PROXY_GAPS",
            "RUN_QUALITY_PRIVACY_SPLIT_TWO_CLEAN_BUILDS_AND_UNIQUE_LOSS_LEDGER",
        ],
    }


def load_research_corpus_intake(path: str | Path) -> dict[str, Any]:
    """Load and validate a Research Corpus V1 intake JSON authority."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ResearchCorpusIntakeError("intake JSON root must be an object")
    validate_research_corpus_intake(data)
    return data
