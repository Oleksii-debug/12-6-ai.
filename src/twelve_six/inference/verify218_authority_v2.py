"""Exact-source VERIFY-218 authority gate for the learned 10M runtime consumer.

RUNTIME-225 V1 had the right scientific gate shape but retained the historical
SCALE-141 artifact name and accepted arbitrary positive learned-source transport
coordinates. V2 binds the consumer to the one terminal LEARN-217 artifact that
VERIFY-218 independently verifies, then delegates all existing structural and
verifier-transport checks to the maintained V1 validator.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from twelve_six.inference import verify218_authority as v1

EXPECTED_SOURCE_ARTIFACT_ID = 9602650341
EXPECTED_SOURCE_ARTIFACT_NAME = "learn217-terminal-10m-learned-base"
EXPECTED_SOURCE_ARTIFACT_DIGEST = (
    "sha256:8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee"
)
EXPECTED_SOURCE_WORKFLOW_RUN_ID = 32952787070
EXPECTED_SOURCE_SHA = "c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef"
GATE_SCHEMA = "12-6.runtime225-verify218-consumer-gate.v2"


class Verify218AuthorityV2Error(v1.Verify218AuthorityError):
    """Raised when VERIFY-218 does not bind the exact terminal LEARN-217 source."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Verify218AuthorityV2Error(f"{label} must be an object")
    return value


def _require_exact_source(manifest: Mapping[str, Any]) -> None:
    source = _mapping(manifest.get("source"), "VERIFY-218 learned source")
    expected = {
        "artifact_id": EXPECTED_SOURCE_ARTIFACT_ID,
        "artifact_name": EXPECTED_SOURCE_ARTIFACT_NAME,
        "artifact_digest": EXPECTED_SOURCE_ARTIFACT_DIGEST,
        "workflow_run_id": EXPECTED_SOURCE_WORKFLOW_RUN_ID,
        "source_sha": EXPECTED_SOURCE_SHA,
    }
    mismatches = {
        field: {"expected": value, "actual": source.get(field)}
        for field, value in expected.items()
        if source.get(field) != value
    }
    if mismatches:
        raise Verify218AuthorityV2Error(
            f"VERIFY-218 learned source is not exact terminal LEARN-217: {mismatches}"
        )


def validate_verify218_authority_v2(
    manifest: Mapping[str, Any],
    verifier_artifact: Mapping[str, Any],
    verifier_run: Mapping[str, Any],
    *,
    verifier_artifact_id: int,
    verifier_artifact_digest: str,
    verifier_run_id: int,
    verifier_source_sha: str,
) -> dict[str, Any]:
    """Validate VERIFY-218 and return immutable exact LEARN-217 coordinates."""

    _require_exact_source(manifest)

    # V1's artifact-name constant predates terminal LEARN-217. Normalize only
    # that already independently prevalidated field to reuse every incumbent
    # structural, model/tokenizer/corpus, checkpoint and verifier-transport gate.
    normalized = deepcopy(dict(manifest))
    source = dict(_mapping(normalized.get("source"), "VERIFY-218 learned source"))
    source["artifact_name"] = v1.EXPECTED_SOURCE_ARTIFACT_NAME
    normalized["source"] = source

    gate = v1.validate_verify218_authority(
        normalized,
        verifier_artifact,
        verifier_run,
        verifier_artifact_id=verifier_artifact_id,
        verifier_artifact_digest=verifier_artifact_digest,
        verifier_run_id=verifier_run_id,
        verifier_source_sha=verifier_source_sha,
    )

    gate["schema"] = GATE_SCHEMA
    gate["learned_source"].update(
        {
            "artifact_id": EXPECTED_SOURCE_ARTIFACT_ID,
            "artifact_name": EXPECTED_SOURCE_ARTIFACT_NAME,
            "artifact_digest": EXPECTED_SOURCE_ARTIFACT_DIGEST,
            "workflow_run_id": EXPECTED_SOURCE_WORKFLOW_RUN_ID,
            "source_sha": EXPECTED_SOURCE_SHA,
        }
    )
    gate["truth_boundary"]["exact_terminal_learn217_source_bound"] = True
    gate.pop("identity_sha256", None)
    gate["identity_sha256"] = v1._canonical_sha256(gate)
    return gate


load_json_object = v1.load_json_object
