from __future__ import annotations

import copy

from twelve_six.readiness_trust_root import (
    authenticated_trusted_readiness_inputs,
    trusted_readiness_bundle_sha256,
)

SHA40 = "a" * 40
SHA64 = "b" * 64


def _bundle() -> dict:
    return {
        "schema_version": 1,
        "scientific_authorities": {
            "code": {
                "authority": {
                    "repository": "Oleksii-debug/12-6-ai.",
                    "git_sha": SHA40,
                    "evidence_sha256": SHA64,
                    "terminal": True,
                    "workflow_run_id": 123,
                    "workflow_conclusion": "success",
                },
                "metadata": {"git_sha": SHA40},
            }
        },
        "verified_authorization_refs": [],
    }


def test_trusted_bundle_requires_exact_external_root() -> None:
    bundle = _bundle()
    expected = trusted_readiness_bundle_sha256(bundle)
    assert expected is not None
    resolved = authenticated_trusted_readiness_inputs(
        bundle,
        expected_identity_sha256=expected,
    )
    assert resolved is not None
    scientific, refs = resolved
    assert len(scientific) == 1
    assert refs == set()

    assert (
        authenticated_trusted_readiness_inputs(
            bundle,
            expected_identity_sha256="c" * 64,
        )
        is None
    )


def test_coherent_bundle_reseal_cannot_move_under_pinned_root() -> None:
    bundle = _bundle()
    expected = trusted_readiness_bundle_sha256(bundle)
    assert expected is not None

    forged = copy.deepcopy(bundle)
    forged["scientific_authorities"]["code"]["authority"]["git_sha"] = "c" * 40
    forged["scientific_authorities"]["code"]["metadata"]["git_sha"] = "c" * 40
    assert trusted_readiness_bundle_sha256(forged) != expected
    assert (
        authenticated_trusted_readiness_inputs(
            forged,
            expected_identity_sha256=expected,
        )
        is None
    )


def test_forged_success_workflow_cannot_move_under_pinned_root() -> None:
    bundle = _bundle()
    expected = trusted_readiness_bundle_sha256(bundle)
    assert expected is not None

    forged = copy.deepcopy(bundle)
    authority = forged["scientific_authorities"]["code"]["authority"]
    authority["workflow_run_id"] = 999999
    authority["workflow_conclusion"] = "success"
    assert (
        authenticated_trusted_readiness_inputs(
            forged,
            expected_identity_sha256=expected,
        )
        is None
    )


def test_malformed_or_noncanonical_expected_identity_fails_closed() -> None:
    bundle = _bundle()
    for bad in (None, "", "B" * 64, "b" * 63, 1, True):
        assert (
            authenticated_trusted_readiness_inputs(
                bundle,
                expected_identity_sha256=bad,
            )
            is None
        )
