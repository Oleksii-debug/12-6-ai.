from __future__ import annotations

from copy import deepcopy

import pytest

from twelve_six.verify218_runtime225_authority import (
    PRODUCER_SHA,
    RUNTIME_ARTIFACT_DIGEST,
    RUNTIME_ARTIFACT_ID,
    RUNTIME_ARTIFACT_NAME,
    RUNTIME_RUN_ID,
    SCIENTIFIC_ARTIFACT_DIGEST,
    SCIENTIFIC_ARTIFACT_ID,
    SCIENTIFIC_ARTIFACT_NAME,
    SCIENTIFIC_RUN_ID,
    STATUS,
    WORKER_ID,
    Verify218BridgeError,
    _canonical_sha256,
    _validate_transport,
)


def _transport(*, artifact_id: int, name: str, digest: str, run_id: int):
    artifact = {
        "id": artifact_id,
        "name": name,
        "digest": digest,
        "expired": False,
        "workflow_run": {"id": run_id, "head_sha": PRODUCER_SHA},
    }
    run = {
        "id": run_id,
        "head_sha": PRODUCER_SHA,
        "status": "completed",
        "conclusion": "success",
    }
    return artifact, run


def test_exact_authority_constants_match_downstream_contract() -> None:
    assert WORKER_ID == "VERIFY-218-LEARNED-10M-INDEPENDENT"
    assert STATUS == "VERIFIED_LEARNED_10M"
    assert SCIENTIFIC_ARTIFACT_ID == 9602650341
    assert SCIENTIFIC_RUN_ID == 32952787070
    assert RUNTIME_ARTIFACT_ID == 9602907196
    assert RUNTIME_RUN_ID == 32952786715
    assert RUNTIME_ARTIFACT_NAME == "scale141-10m-learned-fallback"


def test_transport_accepts_exact_scientific_and_runtime_artifacts() -> None:
    for artifact_id, name, digest, run_id in (
        (
            SCIENTIFIC_ARTIFACT_ID,
            SCIENTIFIC_ARTIFACT_NAME,
            SCIENTIFIC_ARTIFACT_DIGEST,
            SCIENTIFIC_RUN_ID,
        ),
        (
            RUNTIME_ARTIFACT_ID,
            RUNTIME_ARTIFACT_NAME,
            RUNTIME_ARTIFACT_DIGEST,
            RUNTIME_RUN_ID,
        ),
    ):
        artifact, run = _transport(
            artifact_id=artifact_id,
            name=name,
            digest=digest,
            run_id=run_id,
        )
        _validate_transport(
            artifact,
            run,
            artifact_id=artifact_id,
            artifact_name=name,
            artifact_digest=digest,
            run_id=run_id,
        )


def test_transport_fails_closed_on_name_digest_and_terminal_state() -> None:
    artifact, run = _transport(
        artifact_id=RUNTIME_ARTIFACT_ID,
        name=RUNTIME_ARTIFACT_NAME,
        digest=RUNTIME_ARTIFACT_DIGEST,
        run_id=RUNTIME_RUN_ID,
    )
    for field, value, pattern in (
        ("name", "wrong-artifact", "artifact name mismatch"),
        ("digest", "sha256:" + "0" * 64, "artifact digest mismatch"),
        ("expired", True, "artifact is expired"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(Verify218BridgeError, match=pattern):
            _validate_transport(
                bad,
                run,
                artifact_id=RUNTIME_ARTIFACT_ID,
                artifact_name=RUNTIME_ARTIFACT_NAME,
                artifact_digest=RUNTIME_ARTIFACT_DIGEST,
                run_id=RUNTIME_RUN_ID,
            )

    bad_run = deepcopy(run)
    bad_run["conclusion"] = "failure"
    with pytest.raises(Verify218BridgeError, match="workflow run is not SUCCESS"):
        _validate_transport(
            artifact,
            bad_run,
            artifact_id=RUNTIME_ARTIFACT_ID,
            artifact_name=RUNTIME_ARTIFACT_NAME,
            artifact_digest=RUNTIME_ARTIFACT_DIGEST,
            run_id=RUNTIME_RUN_ID,
        )


def test_canonical_identity_is_stable_and_content_sensitive() -> None:
    first = _canonical_sha256({"b": 2, "a": {"x": 1}})
    reordered = _canonical_sha256({"a": {"x": 1}, "b": 2})
    changed = _canonical_sha256({"a": {"x": 2}, "b": 2})
    assert first == reordered
    assert first != changed
    assert len(first) == 64
