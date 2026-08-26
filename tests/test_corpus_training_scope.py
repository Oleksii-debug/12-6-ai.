from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.corpus_training_scope import (
    CorpusTrainingScopeError,
    validate_training_scope,
)

SCOPE_PATH = Path("configs/data/research_corpus_v1_training_scope_v1.json")


def _scope() -> dict:
    return json.loads(SCOPE_PATH.read_text(encoding="utf-8"))


def test_current_scope_keeps_long_training_blocked() -> None:
    report = validate_training_scope(_scope())
    assert report["status"] == "PASS_SOURCE_TOKEN_FIREWALL"
    assert report["model_parameters"] == 20_613_440
    assert report["source_milestone_bytes"] == 20_000_000
    assert report["candidate_20x_reference_unique_loss_tokens"] == 412_268_800
    assert report["training_authorized_unique_loss_tokens"] == 0
    assert report["long_learned_20m_training_authorized"] is False


def test_20mb_cannot_be_relabelled_as_training_sufficiency() -> None:
    scope = _scope()
    scope["source_milestone_proves_training_sufficiency"] = True
    with pytest.raises(CorpusTrainingScopeError, match="must not prove training sufficiency"):
        validate_training_scope(scope)


def test_source_bytes_cannot_be_relabelled_as_tokens() -> None:
    scope = _scope()
    scope["source_bytes_are_training_tokens"] = True
    with pytest.raises(CorpusTrainingScopeError, match="source bytes are not tokens"):
        validate_training_scope(scope)


def test_candidate_scaling_policy_cannot_authorize_budget() -> None:
    scope = _scope()
    scope["scaling_policy_dependency"]["reference_is_authorized_budget"] = True
    with pytest.raises(CorpusTrainingScopeError, match="cannot authorize a training budget"):
        validate_training_scope(scope)


def test_candidate_reference_drift_is_rejected() -> None:
    scope = _scope()
    scope["scaling_policy_dependency"]["reference_unique_loss_tokens_20x"] += 1
    with pytest.raises(CorpusTrainingScopeError, match="planning reference drift"):
        validate_training_scope(scope)


def test_nonzero_materialized_tokens_require_real_terminal_evidence() -> None:
    scope = copy.deepcopy(_scope())
    scope["current_authority"]["materialized_unique_loss_tokens"] = 20_000_000
    with pytest.raises(CorpusTrainingScopeError, match="must remain zero"):
        validate_training_scope(scope)
