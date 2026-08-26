"""EVAL-237 source-family generalization V2 scientific contract.

This module intentionally does not guess the unpublished DATA-230 registry schema.
It defines the exact semantic projection EVAL-237 requires, diversity and
identifiability gates, matched exposure planning, and comparison definitions.
Numerical training is forbidden until a terminal DATA-230 registry can be
projected into this contract without ambiguity.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .source_generalization import (
    BATCH_SIZE,
    LOSS_TOKENS_PER_STEP,
    OPTIMIZED_TOKEN_BUDGET,
    PARAMETER_COUNT,
    SEQUENCE_LENGTH,
)

SCHEMA_VERSION = "12-6.eval237-source-family-generalization-v2.v1"
FAMILY_PROJECTION_SCHEMA = "12-6.eval237-data230-family-projection.v1"
WORKER_ID = "EVAL-237-SOURCE-FAMILY-GENERALIZATION-V2"
DATA230_WORKER_ID = "DATA-230-CORPUS-V03-EXTERNAL-REAL"
RECOVER179_HEAD = "7c83fd00a31d4237f7636a540148a89fef9526a1"

MIN_SELECTED_FAMILIES = 4
MIN_FAMILIES_PER_LANGUAGE = 2
MIN_HOLDOUT_RECORDS = 5
MIN_HOLDOUT_LOSS_TOKENS = 4096

MODEL_IDENTITY = {
    "parameter_count": PARAMETER_COUNT,
    "tokenizer": "byte-v1-vocab-256-no-specials",
    "sequence_length": SEQUENCE_LENGTH,
    "batch_size": BATCH_SIZE,
    "loss_tokens_per_full_step": LOSS_TOKENS_PER_STEP,
    "optimized_token_budget_per_arm": OPTIMIZED_TOKEN_BUDGET,
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "scheduler": "constant",
        "warmup_steps": 0,
        "gradient_clip_norm": 1.0,
        "precision": "fp32",
    },
    "seed": 1337,
}


class Eval237Error(ValueError):
    """Raised when the EVAL-237 preregistered contract is not satisfied."""


@dataclass(frozen=True)
class FamilyDescriptor:
    family_id: str
    language: str
    domain: str
    publisher_id: str
    origin: str
    training_authorized: bool
    independent_family: bool
    train_loss_tokens: int
    holdout_loss_tokens: int
    holdout_record_count: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FamilyDescriptor":
        required = {
            "family_id",
            "language",
            "domain",
            "publisher_id",
            "origin",
            "training_authorized",
            "independent_family",
            "train_loss_tokens",
            "holdout_loss_tokens",
            "holdout_record_count",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise Eval237Error(f"family descriptor missing fields: {missing}")
        descriptor = cls(
            family_id=str(value["family_id"]),
            language=str(value["language"]).lower(),
            domain=str(value["domain"]).lower(),
            publisher_id=str(value["publisher_id"]),
            origin=str(value["origin"]),
            training_authorized=bool(value["training_authorized"]),
            independent_family=bool(value["independent_family"]),
            train_loss_tokens=int(value["train_loss_tokens"]),
            holdout_loss_tokens=int(value["holdout_loss_tokens"]),
            holdout_record_count=int(value["holdout_record_count"]),
        )
        if not descriptor.family_id:
            raise Eval237Error("family_id must be non-empty")
        if not descriptor.language or not descriptor.domain:
            raise Eval237Error(f"{descriptor.family_id}: language/domain must be non-empty")
        if not descriptor.publisher_id:
            raise Eval237Error(f"{descriptor.family_id}: publisher_id must be non-empty")
        for field_name in (
            "train_loss_tokens",
            "holdout_loss_tokens",
            "holdout_record_count",
        ):
            if getattr(descriptor, field_name) < 0:
                raise Eval237Error(f"{descriptor.family_id}: {field_name} must be non-negative")
        return descriptor


def _report_sha(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_family_projection(value: Mapping[str, Any]) -> list[FamilyDescriptor]:
    if value.get("schema_version") != FAMILY_PROJECTION_SCHEMA:
        raise Eval237Error("wrong DATA-230 family projection schema")
    if value.get("producer_worker_id") != DATA230_WORKER_ID:
        raise Eval237Error("family projection is not bound to DATA-230")
    registry_identity = value.get("data230_registry_identity")
    if not isinstance(registry_identity, str) or not registry_identity:
        raise Eval237Error("DATA-230 registry identity missing")
    raw_families = value.get("families")
    if not isinstance(raw_families, list):
        raise Eval237Error("families must be a list")
    families = [FamilyDescriptor.from_mapping(item) for item in raw_families]
    ids = [family.family_id for family in families]
    if len(ids) != len(set(ids)):
        raise Eval237Error("duplicate family_id in DATA-230 projection")
    return families


def _eligible_holdout_family(family: FamilyDescriptor) -> bool:
    return (
        family.training_authorized
        and family.independent_family
        and family.train_loss_tokens > 0
        and family.holdout_loss_tokens >= MIN_HOLDOUT_LOSS_TOKENS
        and family.holdout_record_count >= MIN_HOLDOUT_RECORDS
    )


def _effect_identifiability(
    families: Sequence[FamilyDescriptor],
) -> dict[str, dict[str, Any]]:
    by_domain: dict[str, set[str]] = defaultdict(set)
    by_language: dict[str, set[str]] = defaultdict(set)
    by_language_domain: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_language_family: dict[str, set[str]] = defaultdict(set)

    for family in families:
        by_domain[family.domain].add(family.language)
        by_language[family.language].add(family.domain)
        by_language_domain[(family.language, family.domain)].add(family.publisher_id)
        by_language_family[family.language].add(family.family_id)

    language_matches = sorted(
        domain for domain, languages in by_domain.items() if len(languages) >= 2
    )
    domain_matches = sorted(
        language for language, domains in by_language.items() if len(domains) >= 2
    )
    publisher_matches = sorted(
        f"{language}::{domain}"
        for (language, domain), publishers in by_language_domain.items()
        if len(publishers) >= 2
    )
    replicated_languages = sorted(
        language
        for language, family_ids in by_language_family.items()
        if len(family_ids) >= MIN_FAMILIES_PER_LANGUAGE
    )

    return {
        "language_effect": {
            "identifiable": bool(language_matches),
            "matched_domains_crossing_languages": language_matches,
            "interpretation": (
                "Within-domain cross-language contrast is available."
                if language_matches
                else "No domain crosses languages; language is confounded with domain."
            ),
        },
        "domain_effect": {
            "identifiable": bool(domain_matches),
            "languages_crossing_domains": domain_matches,
            "interpretation": (
                "Within-language cross-domain contrast is available."
                if domain_matches
                else "No language crosses domains; domain is confounded with language."
            ),
        },
        "publisher_source_family_effect": {
            "identifiable": bool(publisher_matches),
            "replicated_language_domain_cells": publisher_matches,
            "interpretation": (
                "Multiple publishers/families exist within a matched language-domain cell."
                if publisher_matches
                else "No matched language-domain cell contains multiple publishers."
            ),
        },
        "per_language_replication": {
            "identifiable": len(replicated_languages) >= 2,
            "replicated_languages": replicated_languages,
            "interpretation": (
                "At least two languages have multiple independent source families."
                if len(replicated_languages) >= 2
                else "The design would still generalize from one source per language."
            ),
        },
    }


def assess_diversity(families: Sequence[FamilyDescriptor]) -> dict[str, Any]:
    eligible = [family for family in families if _eligible_holdout_family(family)]
    by_language: dict[str, list[FamilyDescriptor]] = defaultdict(list)
    for family in eligible:
        by_language[family.language].append(family)

    replicated_languages = {
        language
        for language, members in by_language.items()
        if len(members) >= MIN_FAMILIES_PER_LANGUAGE
    }
    selected = [
        family for family in eligible if family.language in replicated_languages
    ]
    identifiability = _effect_identifiability(selected)

    blockers: list[str] = []
    if len(selected) < MIN_SELECTED_FAMILIES:
        blockers.append("fewer_than_four_independent_meaningful_holdout_families")
    if not identifiability["per_language_replication"]["identifiable"]:
        blockers.append("one_source_per_language_regime_not_closed")
    for key in (
        "language_effect",
        "domain_effect",
        "publisher_source_family_effect",
    ):
        if not identifiability[key]["identifiable"]:
            blockers.append(f"{key}_not_identifiable")

    total_train_tokens = sum(family.train_loss_tokens for family in selected)
    if total_train_tokens < OPTIMIZED_TOKEN_BUDGET:
        blockers.append("mixed_arm_insufficient_unique_loss_tokens")
    for heldout in selected:
        loo_tokens = sum(
            family.train_loss_tokens
            for family in selected
            if family.family_id != heldout.family_id
        )
        if loo_tokens < OPTIMIZED_TOKEN_BUDGET:
            blockers.append(
                f"loo_arm_insufficient_unique_loss_tokens::{heldout.family_id}"
            )

    return {
        "ready": not blockers,
        "blockers": blockers,
        "eligible_family_ids": [family.family_id for family in eligible],
        "selected_family_ids": [family.family_id for family in selected],
        "family_counts_by_language": {
            language: len(members)
            for language, members in sorted(by_language.items())
        },
        "identifiability": identifiability,
        "minimums": {
            "selected_families": MIN_SELECTED_FAMILIES,
            "families_per_language": MIN_FAMILIES_PER_LANGUAGE,
            "holdout_records_per_family": MIN_HOLDOUT_RECORDS,
            "holdout_loss_tokens_per_family": MIN_HOLDOUT_LOSS_TOKENS,
            "unique_train_loss_tokens_per_arm": OPTIMIZED_TOKEN_BUDGET,
        },
    }


def build_matched_arm_plan(families: Sequence[FamilyDescriptor]) -> dict[str, Any]:
    assessment = assess_diversity(families)
    if not assessment["ready"]:
        raise Eval237Error(
            "source diversity is insufficient: " + ", ".join(assessment["blockers"])
        )
    selected_ids = set(assessment["selected_family_ids"])
    selected = [family for family in families if family.family_id in selected_ids]

    common = {
        "model": MODEL_IDENTITY,
        "optimized_token_budget": OPTIMIZED_TOKEN_BUDGET,
        "exposure_accounting": "actual_source_loss_tokens_only",
        "allow_padded_tensor_tokens_in_budget": False,
        "allow_example_or_loss_token_repetition": False,
        "evaluation_holdout_exposure": "zero_optimizer_exposure",
        "universal_locked_bootstrap_required": True,
    }
    arms: list[dict[str, Any]] = [
        {
            "arm_id": "mixed_source_control",
            "train_family_ids": [family.family_id for family in selected],
            "omitted_family_id": None,
            **common,
        }
    ]
    for heldout in selected:
        arms.append(
            {
                "arm_id": f"leave_one_family_out::{heldout.family_id}",
                "train_family_ids": [
                    family.family_id
                    for family in selected
                    if family.family_id != heldout.family_id
                ],
                "omitted_family_id": heldout.family_id,
                **common,
            }
        )

    return {
        "study_worker_id": WORKER_ID,
        "assessment": assessment,
        "selected_families": [asdict(family) for family in selected],
        "arms": arms,
    }


def family_comparison(
    *,
    random_init_bpb: float,
    mixed_direct_exposure_bpb: float,
    leave_one_family_out_bpb: float,
) -> dict[str, float]:
    return {
        "random_init_bpb": float(random_init_bpb),
        "mixed_direct_exposure_bpb": float(mixed_direct_exposure_bpb),
        "leave_one_family_out_bpb": float(leave_one_family_out_bpb),
        "random_init_improvement_bpb": (
            float(random_init_bpb) - float(leave_one_family_out_bpb)
        ),
        "direct_exposure_advantage_bpb": (
            float(leave_one_family_out_bpb) - float(mixed_direct_exposure_bpb)
        ),
    }


def blocked_missing_data230_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "worker_id": WORKER_ID,
        "status": "BLOCKED_MISSING_DATA230",
        "numerical_training_executed": False,
        "numerical_result_claimed": False,
        "required_predecessor": {
            "worker_id": DATA230_WORKER_ID,
            "terminal_registry_required": True,
            "observed_in_repository": False,
        },
        "inherited_predecessor": {
            "worker_id": "RECOVER-179-EVAL137-SOURCE-GENERALIZATION",
            "head_sha": RECOVER179_HEAD,
        },
        "frozen_controls": MODEL_IDENTITY,
        "scientific_boundary": {
            "old_two_family_regime_is_not_reused_as_v2_evidence": True,
            "one_source_per_language_generalization_forbidden": True,
            "language_domain_publisher_decomposition_requires_identifiability": True,
            "matched_exposure_uses_actual_source_loss_tokens": True,
            "padded_tensor_tokens_do_not_count_as_exposure": True,
            "example_repetition_to_fill_budget_forbidden": True,
        },
        "next_executable_gate": {
            "projection_schema": FAMILY_PROJECTION_SCHEMA,
            "required_functions": [
                "assess_diversity",
                "build_matched_arm_plan",
                "family_comparison",
            ],
        },
    }
    report["report_sha256"] = _report_sha(report)
    return report


def validate_blocked_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise Eval237Error("wrong report schema")
    if report.get("worker_id") != WORKER_ID:
        raise Eval237Error("wrong worker id")
    if report.get("status") != "BLOCKED_MISSING_DATA230":
        raise Eval237Error("blocked report status drift")
    if report.get("numerical_training_executed") is not False:
        raise Eval237Error("blocked report must not claim training")
    if report.get("numerical_result_claimed") is not False:
        raise Eval237Error("blocked report must not claim numerical results")
    claimed = report.get("report_sha256")
    if not isinstance(claimed, str) or claimed != _report_sha(report):
        raise Eval237Error("report hash mismatch")
