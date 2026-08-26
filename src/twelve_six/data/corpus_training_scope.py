"""Scientific firewall between corpus source milestones and learned-model token budgets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.research-corpus-v1-training-scope.v1"
MODEL_341_PARAMETERS = 20_613_440
REFERENCE_20X_TOKENS = MODEL_341_PARAMETERS * 20


class CorpusTrainingScopeError(RuntimeError):
    """Raised when source capacity is relabelled as learned-training sufficiency."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusTrainingScopeError(message)


def _zero_int(value: Any, *, field: str) -> None:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value == 0,
        f"{field} must remain zero until terminal materialized evidence exists",
    )


def validate_training_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    _require(scope.get("schema_version") == SCHEMA, "unsupported training-scope schema")
    _require(scope.get("local_free_only") is True, "training-scope policy must be LOCAL_FREE")
    _require(scope.get("model_training_executed") is False, "policy validation must not train")
    _require(
        scope.get("research_corpus_v1_source_milestone_bytes") == 20_000_000,
        "Research Corpus V1 source milestone drift",
    )
    _require(scope.get("source_bytes_are_training_tokens") is False, "source bytes are not tokens")
    _require(
        scope.get("source_milestone_proves_training_sufficiency") is False,
        "20 MB source milestone must not prove training sufficiency",
    )

    model = scope.get("exact_model")
    _require(isinstance(model, Mapping), "exact_model must be a mapping")
    _require(model.get("authority") == "MODEL-341", "exact model authority must remain MODEL-341")
    _require(model.get("parameters") == MODEL_341_PARAMETERS, "MODEL-341 parameter count drift")
    _require(
        model.get("mechanics_status") == "QUALIFIED_CANDIDATE_NOT_LEARNED_MODEL",
        "mechanics must not be relabelled as learned-model evidence",
    )

    dependency = scope.get("scaling_policy_dependency")
    _require(isinstance(dependency, Mapping), "scaling_policy_dependency must be a mapping")
    _require(
        dependency.get("status") == "CANDIDATE_ONLY_NOT_TERMINAL_AUTHORITY",
        "nonterminal scaling policy must remain candidate-only",
    )
    _require(
        dependency.get("reference_unique_loss_tokens_20x") == REFERENCE_20X_TOKENS,
        "20x planning reference drift",
    )
    _require(
        dependency.get("reference_is_authorized_budget") is False,
        "planning reference cannot authorize a training budget",
    )

    authority = scope.get("current_authority")
    _require(isinstance(authority, Mapping), "current_authority must be a mapping")
    _zero_int(authority.get("materialized_unique_loss_tokens"), field="materialized_unique_loss_tokens")
    _zero_int(
        authority.get("training_authorized_unique_loss_tokens"),
        field="training_authorized_unique_loss_tokens",
    )
    _require(
        authority.get("long_learned_20m_training_authorized") is False,
        "long learned 20M training must remain blocked",
    )

    required = scope.get("required_before_long_learned_20m_training")
    required_set = {
        "terminal_research_corpus_identity",
        "terminal_tokenizer_identity",
        "train_selection_final_eval_decontamination",
        "post_pack_unique_nonignored_causal_loss_token_count",
        "terminal_scaling_budget_policy",
        "preregistered_budget_and_stopping_rule",
        "checkpoint_recovery_terminal_pass",
        "compute_authorization",
    }
    _require(isinstance(required, list) and required_set.issubset(set(required)), "training gates incomplete")
    _require(
        scope.get("truth_boundary")
        == "RESEARCH_CORPUS_V1_20MB_IS_A_SOURCE_ACQUISITION_MILESTONE_ONLY; IT MUST NOT BE RELABELED AS 20M_MODEL_TRAINING_SUFFICIENCY",
        "truth boundary drift",
    )
    return {
        "status": "PASS_SOURCE_TOKEN_FIREWALL",
        "model_parameters": MODEL_341_PARAMETERS,
        "source_milestone_bytes": 20_000_000,
        "candidate_20x_reference_unique_loss_tokens": REFERENCE_20X_TOKENS,
        "training_authorized_unique_loss_tokens": 0,
        "long_learned_20m_training_authorized": False,
    }


def load_and_validate_training_scope(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusTrainingScopeError("training-scope artifact is not valid JSON") from exc
    _require(isinstance(value, Mapping), "training-scope root must be an object")
    return validate_training_scope(value)
