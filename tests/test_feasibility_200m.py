from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six import accelerated_scaling as r01
from twelve_six.feasibility_200m import (
    FeasibilityPacketError,
    build_200m_feasibility_packet,
    canonical_sha256,
    compute_packet_sha256,
    expected_external_identities,
    validate_200m_feasibility_packet,
)

GIT_A = "a" * 40
GIT_B = "b" * 40
SHA_A = "a" * 64
SHA_B = "b" * 64
ROADMAP_PATH = Path("configs/research/r01_accelerated_scaling_roadmap_v2.json")


def authority(
    *, git_sha: str = GIT_A, evidence_sha256: str = SHA_A, run: int = 1
) -> dict:
    return {
        "repository": r01.REPOSITORY,
        "git_sha": git_sha,
        "evidence_sha256": evidence_sha256,
        "workflow_run_id": run,
        "workflow_conclusion": "success",
        "terminal": True,
    }


def roadmap() -> dict:
    value = json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))
    value["evidence_state"]["learned_20m"] = {
        "status": "PASS",
        "evidence_manifest_sha256": "1" * 64,
        "requirements_satisfied": sorted(r01.REQUIRED_TERMINAL_20M_EVIDENCE),
        "terminal_authority": authority(run=11),
        "independent_audit_authority": authority(
            git_sha=GIT_B, evidence_sha256=SHA_B, run=12
        ),
    }
    return value


def candidate() -> dict:
    return {
        "candidate_id": "product-200m-a",
        "architecture_sha256": "2" * 64,
        "parameter_count": 203_000_000,
        "target_unique_loss_positions": 700_000_000,
    }


def measurements() -> dict:
    return {
        "tokens_per_second": 321.5,
        "step_time_seconds": 1.25,
        "peak_ram_bytes": 8_000_000_000,
        "peak_vram_bytes_or_not_applicable": "NOT_APPLICABLE",
        "checkpoint_size_bytes": 2_500_000_000,
        "checkpoint_save_seconds": 8.5,
        "checkpoint_load_seconds": 7.5,
        "estimated_wall_clock_to_target": 86_400.0,
        "energy_or_thermal_observation_if_available": None,
    }


def measurement_authority(values: dict | None = None) -> dict:
    if values is None:
        values = measurements()
    return authority(evidence_sha256=canonical_sha256(values), run=20)


def evidence(candidate_value: dict | None = None) -> dict:
    if candidate_value is None:
        candidate_value = candidate()
    refs = {
        name: authority(
            git_sha=f"{index % 10}" * 40,
            evidence_sha256=f"{index % 10}" * 64,
            run=100 + index,
        )
        for index, name in enumerate(
            sorted(r01.REQUIRED_200M_FEASIBILITY), start=1
        )
    }
    refs["candidate_architecture_and_parameter_count"]["evidence_sha256"] = (
        canonical_sha256(candidate_value)
    )
    return refs


def build() -> dict:
    return build_200m_feasibility_packet(
        roadmap_snapshot=roadmap(),
        source_git_sha=GIT_A,
        candidate=candidate(),
        measurements_20m=measurements(),
        measurement_authority=measurement_authority(),
        requirement_evidence=evidence(),
        decision="GO",
    )


def validate(
    packet: dict, roadmap_snapshot: dict | None = None, expected: dict | None = None
) -> list[str]:
    if roadmap_snapshot is None:
        roadmap_snapshot = roadmap()
    if expected is None:
        expected = expected_external_identities(packet)
    return validate_200m_feasibility_packet(
        packet,
        roadmap_snapshot=roadmap_snapshot,
        expected_packet_sha256=expected["packet_sha256"],
        expected_roadmap_snapshot_sha256=expected["roadmap_snapshot_sha256"],
        expected_source_git_sha=expected["source_git_sha"],
        expected_measurements_20m_sha256=expected["measurements_20m_sha256"],
        expected_requirement_evidence_sha256=expected[
            "requirement_evidence_sha256"
        ],
    )


def test_build_is_deterministic_and_external_validation_passes() -> None:
    first = build()
    second = build()
    assert first == second
    assert first["packet_sha256"] == compute_packet_sha256(first)
    assert first["roadmap_snapshot_sha256"] == canonical_sha256(roadmap())
    assert first["requirements_covered"] == sorted(r01.REQUIRED_200M_FEASIBILITY)
    assert validate(first) == []


def test_builder_requires_canonical_terminal_20m_transition() -> None:
    blocked = json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))
    with pytest.raises(FeasibilityPacketError, match="roadmap_not_at_200m"):
        build_200m_feasibility_packet(
            roadmap_snapshot=blocked,
            source_git_sha=GIT_A,
            candidate=candidate(),
            measurements_20m=measurements(),
            measurement_authority=measurement_authority(),
            requirement_evidence=evidence(),
            decision="GO",
        )


def test_bool_aliases_and_nonfinite_measurements_fail_closed() -> None:
    for field, bad in (
        ("tokens_per_second", True),
        ("tokens_per_second", float("nan")),
        ("step_time_seconds", float("inf")),
        ("peak_ram_bytes", True),
        ("checkpoint_size_bytes", 0),
        ("checkpoint_save_seconds", -1),
    ):
        values = measurements()
        values[field] = bad
        with pytest.raises(FeasibilityPacketError):
            build_200m_feasibility_packet(
                roadmap_snapshot=roadmap(),
                source_git_sha=GIT_A,
                candidate=candidate(),
                measurements_20m=values,
                measurement_authority=measurement_authority(values),
                requirement_evidence=evidence(),
                decision="GO",
            )


def test_missing_extra_or_nonterminal_requirement_evidence_fails() -> None:
    missing = evidence()
    missing.pop(next(iter(missing)))
    with pytest.raises(FeasibilityPacketError, match="fields_mismatch"):
        build_200m_feasibility_packet(
            roadmap_snapshot=roadmap(),
            source_git_sha=GIT_A,
            candidate=candidate(),
            measurements_20m=measurements(),
            measurement_authority=measurement_authority(),
            requirement_evidence=missing,
            decision="GO",
        )

    not_terminal = evidence()
    first = next(iter(not_terminal))
    not_terminal[first]["terminal"] = False
    with pytest.raises(FeasibilityPacketError, match="requirement_evidence"):
        build_200m_feasibility_packet(
            roadmap_snapshot=roadmap(),
            source_git_sha=GIT_A,
            candidate=candidate(),
            measurements_20m=measurements(),
            measurement_authority=measurement_authority(),
            requirement_evidence=not_terminal,
            decision="GO",
        )


def test_candidate_and_measurement_authorities_bind_exact_payloads() -> None:
    wrong_candidate = evidence()
    wrong_candidate["candidate_architecture_and_parameter_count"][
        "evidence_sha256"
    ] = "f" * 64
    with pytest.raises(
        FeasibilityPacketError,
        match="candidate_architecture_evidence_payload_mismatch",
    ):
        build_200m_feasibility_packet(
            roadmap_snapshot=roadmap(),
            source_git_sha=GIT_A,
            candidate=candidate(),
            measurements_20m=measurements(),
            measurement_authority=measurement_authority(),
            requirement_evidence=wrong_candidate,
            decision="GO",
        )

    with pytest.raises(
        FeasibilityPacketError, match="measurement_authority_payload_mismatch"
    ):
        build_200m_feasibility_packet(
            roadmap_snapshot=roadmap(),
            source_git_sha=GIT_A,
            candidate=candidate(),
            measurements_20m=measurements(),
            measurement_authority=authority(evidence_sha256="f" * 64, run=20),
            requirement_evidence=evidence(),
            decision="GO",
        )


def test_coherent_reseal_still_fails_stale_external_packet_identity() -> None:
    packet = build()
    expected = expected_external_identities(packet)
    packet["candidate"]["parameter_count"] += 1
    packet["packet_sha256"] = compute_packet_sha256(packet)
    errors = validate(packet, expected=expected)
    assert "candidate_architecture_evidence_payload_mismatch" in errors
    assert "packet_sha256_external_mismatch" in errors


def test_resealed_requirement_evidence_fails_external_identity_map() -> None:
    packet = build()
    expected = expected_external_identities(packet)
    name = "evaluation_capacity"
    packet["requirement_evidence"][name]["evidence_sha256"] = "f" * 64
    packet["packet_sha256"] = compute_packet_sha256(packet)
    errors = validate(packet, expected=expected)
    assert f"requirement_evidence_{name}_external_mismatch" in errors


def test_backend_training_compute_or_stage_self_promotion_is_rejected() -> None:
    for field, bad in (
        ("backend_promotion_authorized_by_packet", True),
        ("stage_promotion_authorized_by_packet", True),
        ("compute_authorized_by_packet", True),
        ("paid_compute_authorized_by_packet", True),
        ("material_training_authorized_by_packet", True),
        ("training_executed_by_packet", True),
        ("optimizer_updates_executed_by_packet", 1),
    ):
        packet = build()
        packet["authority_boundaries"][field] = bad
        packet["packet_sha256"] = compute_packet_sha256(packet)
        expected = expected_external_identities(packet)
        assert "authority_boundaries_must_be_non_authorizing" in validate(
            packet, expected=expected
        )


def test_roadmap_snapshot_drift_fails_even_when_packet_is_unchanged() -> None:
    packet = build()
    expected = expected_external_identities(packet)
    drifted = roadmap()
    drifted["scale_route"][3]["approximate_target_parameters"] = 210_000_000
    errors = validate(packet, roadmap_snapshot=drifted, expected=expected)
    assert "roadmap_snapshot_sha256_recompute_mismatch" in errors
    assert "roadmap_target_parameters_mismatch" in errors


def test_unknown_fields_and_bool_candidate_counts_fail_closed() -> None:
    packet = build()
    packet["surprise"] = "not allowed"
    packet["packet_sha256"] = compute_packet_sha256(packet)
    expected = expected_external_identities(packet)
    assert "packet_fields_mismatch" in validate(packet, expected=expected)

    bad_candidate = candidate()
    bad_candidate["parameter_count"] = True
    with pytest.raises(
        FeasibilityPacketError, match="candidate_parameter_count_invalid"
    ):
        build_200m_feasibility_packet(
            roadmap_snapshot=roadmap(),
            source_git_sha=GIT_A,
            candidate=bad_candidate,
            measurements_20m=measurements(),
            measurement_authority=measurement_authority(),
            requirement_evidence=evidence(bad_candidate),
            decision="GO",
        )


def test_invalid_decision_and_stale_source_git_identity_fail_closed() -> None:
    with pytest.raises(FeasibilityPacketError, match="decision_invalid"):
        build_200m_feasibility_packet(
            roadmap_snapshot=roadmap(),
            source_git_sha=GIT_A,
            candidate=candidate(),
            measurements_20m=measurements(),
            measurement_authority=measurement_authority(),
            requirement_evidence=evidence(),
            decision="MAYBE",
        )

    packet = build()
    expected = expected_external_identities(packet)
    expected["source_git_sha"] = GIT_B
    assert "source_git_sha_external_mismatch" in validate(packet, expected=expected)


def test_expected_identity_map_must_cover_every_requirement() -> None:
    packet = build()
    expected = expected_external_identities(packet)
    expected["requirement_evidence_sha256"] = deepcopy(
        expected["requirement_evidence_sha256"]
    )
    expected["requirement_evidence_sha256"].pop("evaluation_capacity")
    assert "expected_requirement_evidence_sha256_fields_mismatch" in validate(
        packet, expected=expected
    )
