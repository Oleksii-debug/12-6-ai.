from __future__ import annotations

import pytest

from twelve_six.learned20m_recipe import (
    REPOSITORY,
    SESSION_SCHEMA,
    RecipeValidationError,
    identity_sha256,
    readiness_fragment,
)


def _authority() -> dict[str, object]:
    return {
        "repository": REPOSITORY,
        "git_sha": "a" * 40,
        "evidence_sha256": "b" * 64,
        "terminal": True,
        "workflow_run_id": 1,
        "workflow_conclusion": "success",
    }


def _session(capacity: int = 12_345_678) -> dict[str, object]:
    core: dict[str, object] = {
        "schema": SESSION_SCHEMA,
        "status": "QUALIFIED_RECIPE_ONLY",
        "policy_identity_sha256": "1" * 64,
        "bindings_identity_sha256": "2" * 64,
        "trusted_authorities_identity_sha256": "3" * 64,
        "qualified_runtime_unique_loss_positions": capacity,
        "training_recipe_status": "QUALIFIED",
        "training_authorized": False,
        "compute_authorized": False,
        "authorized_optimized_targets": 0,
        "optimizer_updates_executed": 0,
    }
    return {**core, "session_identity_sha256": identity_sha256(core)}


def test_readiness_accepts_unchanged_identity_bound_session() -> None:
    session = _session()

    fragment = readiness_fragment(session, _authority())

    assert fragment["config_sha256"] == session["session_identity_sha256"]
    assert fragment["requested_unique_loss_positions"] == 12_345_678
    assert fragment["requested_total_training_exposures"] == 12_345_678


def test_stale_session_identity_cannot_manufacture_runtime_capacity() -> None:
    session = _session()
    session["qualified_runtime_unique_loss_positions"] = 20_000_000

    with pytest.raises(RecipeValidationError, match="session identity drift"):
        readiness_fragment(session, _authority())


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("bindings_identity_sha256", "4" * 64),
        ("trusted_authorities_identity_sha256", "5" * 64),
    ],
)
def test_stale_session_identity_rejects_authority_identity_drift(
    field: str,
    replacement: str,
) -> None:
    session = _session()
    session[field] = replacement

    with pytest.raises(RecipeValidationError, match="session identity drift"):
        readiness_fragment(session, _authority())
