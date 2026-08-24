from __future__ import annotations

import copy
from pathlib import Path

import pytest

from twelve_six.training.s0_evidence_contract import (
    LOCK_INDEX_FILE_SHA256,
    LOCK_INDEX_PATH,
    LOCK_INDEX_SEMANTIC_SHA256,
    LOCK_PROFILE_FILE_SHA256,
    LOCK_PROFILE_ID,
    LOCK_PROFILE_MANIFEST_SHA256,
    PYTHON_VERSION,
)
from twelve_six.training.s0_repeatability import (
    S0RepeatabilityError,
    _canonical_hash,
    _stable_probe_payload,
    build_s0_repeatability_evidence,
    run_s0_determinism_probe,
    validate_determinism_probe,
    validate_s0_repeatability_evidence,
)

SOURCE_SHA = "a" * 40


def _locked_environment(source_sha: str) -> dict:
    evidence = {
        "schema_version": "12-6.locked-environment-evidence.v1",
        "source_sha": source_sha,
        "profile_id": LOCK_PROFILE_ID,
        "python": {"version": PYTHON_VERSION},
        "lock_index": {
            "path": LOCK_INDEX_PATH,
            "file_sha256": LOCK_INDEX_FILE_SHA256,
            "index_sha256": LOCK_INDEX_SEMANTIC_SHA256,
        },
        "lock_profile": {
            "manifest_sha256": LOCK_PROFILE_MANIFEST_SHA256,
            "file_sha256": LOCK_PROFILE_FILE_SHA256,
        },
        "verification": {
            "committed_lock_validation": "PASS",
            "editable_install_import_cli": "PASS",
            "wheel_install_import_cli": "PASS",
            "repo_checks": "PASS",
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    return evidence


def _rehash_probe(probe: dict) -> None:
    probe["stable_result_sha256"] = _canonical_hash(_stable_probe_payload(probe))
    probe.pop("probe_sha256", None)
    probe["probe_sha256"] = _canonical_hash(probe)


@pytest.fixture(scope="module")
def probes() -> tuple[dict, dict, dict, dict]:
    root = Path(__file__).resolve().parents[1]
    same_seed_a = run_s0_determinism_probe(
        root, source_sha=SOURCE_SHA, seed=1337, max_steps=12, batch_size=3
    )
    same_seed_b = run_s0_determinism_probe(
        root, source_sha=SOURCE_SHA, seed=1337, max_steps=12, batch_size=3
    )
    different_seed = run_s0_determinism_probe(
        root, source_sha=SOURCE_SHA, seed=1338, max_steps=12, batch_size=3
    )
    return same_seed_a, same_seed_b, different_seed, _locked_environment(SOURCE_SHA)


def test_real_s0_same_seed_is_exact_and_different_seed_is_causal(probes) -> None:
    same_seed_a, same_seed_b, different_seed, environment = probes
    evidence = build_s0_repeatability_evidence(
        same_seed_a, same_seed_b, different_seed, environment
    )

    assert same_seed_a["stable_result_sha256"] == same_seed_b["stable_result_sha256"]
    assert same_seed_a["state_fingerprints"] == same_seed_b["state_fingerprints"]
    assert same_seed_a["step_trace"] == same_seed_b["step_trace"]
    assert (
        same_seed_a["state_fingerprints"]["initial_model_sha256"]
        != different_seed["state_fingerprints"]["initial_model_sha256"]
    )
    assert (
        same_seed_a["state_fingerprints"]["final_model_sha256"]
        != different_seed["state_fingerprints"]["final_model_sha256"]
    )
    assert evidence["proof"]["validation_optimized_tokens"] == 0
    assert evidence["claims"]["candidate_or_stable_promotion"] is False
    validate_s0_repeatability_evidence(evidence)


def test_same_seed_state_drift_fails_closed(probes) -> None:
    same_seed_a, same_seed_b, different_seed, environment = probes
    tampered = copy.deepcopy(same_seed_b)
    tampered["state_fingerprints"]["final_model_sha256"] = "0" * 64
    _rehash_probe(tampered)

    with pytest.raises(S0RepeatabilityError, match="same-seed stable results differ"):
        build_s0_repeatability_evidence(
            same_seed_a, tampered, different_seed, environment
        )


def test_validation_optimization_fails_closed(probes) -> None:
    same_seed_a, _, _, _ = probes
    tampered = copy.deepcopy(same_seed_a)
    tampered["split_isolation"]["validation_optimized_tokens"] = 1
    _rehash_probe(tampered)

    with pytest.raises(S0RepeatabilityError, match="optimized validation tokens"):
        validate_determinism_probe(tampered)


def test_different_seed_must_change_initialization(probes) -> None:
    same_seed_a, same_seed_b, different_seed, environment = probes
    tampered = copy.deepcopy(different_seed)
    tampered["state_fingerprints"]["initial_model_sha256"] = same_seed_a[
        "state_fingerprints"
    ]["initial_model_sha256"]
    _rehash_probe(tampered)

    with pytest.raises(
        S0RepeatabilityError,
        match="different seed did not change initial model fingerprint",
    ):
        build_s0_repeatability_evidence(
            same_seed_a, same_seed_b, tampered, environment
        )


def test_probe_source_drift_fails_before_environment_binding(probes) -> None:
    same_seed_a, same_seed_b, different_seed, environment = probes
    tampered = copy.deepcopy(different_seed)
    tampered["identity"]["source_sha"] = "b" * 40
    tampered["identity_sha256"] = _canonical_hash(tampered["identity"])
    _rehash_probe(tampered)

    with pytest.raises(S0RepeatabilityError, match="different-seed source mismatch"):
        build_s0_repeatability_evidence(
            same_seed_a, same_seed_b, tampered, environment
        )


def test_repeatability_manifest_self_hash_tamper_fails(probes) -> None:
    same_seed_a, same_seed_b, different_seed, environment = probes
    evidence = build_s0_repeatability_evidence(
        same_seed_a, same_seed_b, different_seed, environment
    )
    evidence["proof"]["same_seed_exact_equivalence"] = False

    with pytest.raises(S0RepeatabilityError, match="proof summary mismatch"):
        validate_s0_repeatability_evidence(evidence)
