from __future__ import annotations

import importlib.util
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
    retained_identities_for_built_packet,
    validate_200m_feasibility_packet,
)

GIT_A = "a" * 40
GIT_B = "b" * 40
SHA_A = "a" * 64
SHA_B = "b" * 64
ROADMAP_PATH = Path("configs/research/r01_accelerated_scaling_roadmap_v2.json")
CLI_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "build_200m_feasibility_packet.py"
)


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "build_200m_feasibility_packet",
        CLI_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authority(
    *,
    git_sha: str = GIT_A,
    evidence_sha256: str = SHA_A,
    run: int = 1,
    attested_evidence_manifest_sha256: str | None = None,
    audited_producer_authority_sha256: str | None = None,
) -> dict:
    result = {
        "repository": r01.REPOSITORY,
        "git_sha": git_sha,
        "evidence_sha256": evidence_sha256,
        "workflow_run_id": run,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    if attested_evidence_manifest_sha256 is not None:
        result["attested_evidence_manifest_sha256"] = (
            attested_evidence_manifest_sha256
        )
    if audited_producer_authority_sha256 is not None:
        result["audited_producer_authority_sha256"] = (
            audited_producer_authority_sha256
        )
    return result


def roadmap() -> dict:
    value = json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))
    manifest_sha256 = "1" * 64
    producer = authority(
        run=11,
        attested_evidence_manifest_sha256=manifest_sha256,
    )
    audit = authority(
        git_sha=GIT_B,
        evidence_sha256=SHA_B,
        run=12,
        attested_evidence_manifest_sha256=manifest_sha256,
        audited_producer_authority_sha256=canonical_sha256(producer),
    )
    value["evidence_state"]["learned_20m"] = {
        "status": "PASS",
        "evidence_manifest_sha256": manifest_sha256,
        "requirements_satisfied": sorted(r01.REQUIRED_TERMINAL_20M_EVIDENCE),
        "terminal_authority": producer,
        "independent_audit_authority": audit,
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
        expected = retained_identities_for_built_packet(packet)
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


def test_learned_binding_consumes_canonical_crossbound_authorities() -> None:
    packet = build()
    binding = packet["learned_20m_binding"]
    producer = binding["terminal_authority"]
    audit = binding["independent_audit_authority"]
    manifest_sha256 = binding["evidence_manifest_sha256"]

    assert producer["attested_evidence_manifest_sha256"] == manifest_sha256
    assert audit["attested_evidence_manifest_sha256"] == manifest_sha256
    assert audit["audited_producer_authority_sha256"] == canonical_sha256(producer)
    assert validate(packet) == []


def test_tampered_learned_crossbinding_fails_against_roadmap_snapshot() -> None:
    packet = build()
    expected = retained_identities_for_built_packet(packet)
    packet["learned_20m_binding"]["independent_audit_authority"][
        "audited_producer_authority_sha256"
    ] = "0" * 64
    packet["packet_sha256"] = compute_packet_sha256(packet)

    errors = validate(packet, expected=expected)
    assert "learned_20m_binding_roadmap_mismatch" in errors


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
    expected = retained_identities_for_built_packet(packet)
    packet["candidate"]["parameter_count"] += 1
    packet["packet_sha256"] = compute_packet_sha256(packet)
    errors = validate(packet, expected=expected)
    assert "candidate_architecture_evidence_payload_mismatch" in errors
    assert "packet_sha256_external_mismatch" in errors


def test_resealed_requirement_evidence_fails_external_identity_map() -> None:
    packet = build()
    expected = retained_identities_for_built_packet(packet)
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
        expected = retained_identities_for_built_packet(packet)
        packet["authority_boundaries"][field] = bad
        packet["packet_sha256"] = compute_packet_sha256(packet)
        assert "authority_boundaries_must_be_non_authorizing" in validate(
            packet, expected=expected
        )


def test_roadmap_snapshot_drift_fails_even_when_packet_is_unchanged() -> None:
    packet = build()
    expected = retained_identities_for_built_packet(packet)
    drifted = roadmap()
    drifted["scale_route"][3]["approximate_target_parameters"] = 210_000_000
    errors = validate(packet, roadmap_snapshot=drifted, expected=expected)
    assert "roadmap_snapshot_sha256_recompute_mismatch" in errors
    assert "roadmap_target_parameters_mismatch" in errors


def test_unknown_fields_and_bool_candidate_counts_fail_closed() -> None:
    packet = build()
    expected = retained_identities_for_built_packet(packet)
    packet["surprise"] = "not allowed"
    packet["packet_sha256"] = compute_packet_sha256(packet)
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
    expected = retained_identities_for_built_packet(packet)
    expected["source_git_sha"] = GIT_B
    assert "source_git_sha_external_mismatch" in validate(packet, expected=expected)


def test_expected_identity_map_must_cover_every_requirement() -> None:
    packet = build()
    expected = retained_identities_for_built_packet(packet)
    expected["requirement_evidence_sha256"] = deepcopy(
        expected["requirement_evidence_sha256"]
    )
    expected["requirement_evidence_sha256"].pop("evaluation_capacity")
    assert "expected_requirement_evidence_sha256_fields_mismatch" in validate(
        packet, expected=expected
    )


def test_cli_json_reader_accepts_small_regular_utf8_json(tmp_path: Path) -> None:
    cli = _load_cli()
    path = tmp_path / "input.json"
    path.write_text('{"value":1}', encoding="utf-8")
    assert cli._read_json(path, label="input") == {"value": 1}


def test_cli_json_reader_rejects_oversized_input_before_decode(tmp_path: Path) -> None:
    cli = _load_cli()
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (cli.MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="input_exceeds_byte_limit"):
        cli._read_json(path, label="input")


def test_cli_json_reader_redacts_duplicate_member_name(tmp_path: Path) -> None:
    cli = _load_cli()
    path = tmp_path / "duplicate.json"
    secret = "PRIVATE-KEY-NAME"
    path.write_text(
        '{"' + secret + '":1,"' + secret + '":2}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate_json_key") as exc_info:
        cli._read_json(path, label="input")
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ('{"value":1e400}', "json_number_not_finite"),
        ('{"value":1e-9999}', "json_number_underflow"),
        ('{"value":' + "9" * 65 + "}", "json_integer_too_large"),
    ],
)
def test_cli_json_reader_rejects_unsafe_numbers(
    tmp_path: Path,
    payload: str,
    error: str,
) -> None:
    cli = _load_cli()
    path = tmp_path / "number.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        cli._read_json(path, label="input")


def test_cli_json_reader_converts_excessive_nesting_to_bounded_error(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    path = tmp_path / "deep.json"
    path.write_text("[" * 10_000 + "]" * 10_000, encoding="utf-8")
    with pytest.raises(ValueError, match="input_json_too_deep"):
        cli._read_json(path, label="input")



def test_cli_path_preflight_rejects_canonical_collision(tmp_path: Path) -> None:
    cli = _load_cli()
    target = tmp_path / "target.json"
    alias = tmp_path / "nested" / ".." / "target.json"
    with pytest.raises(ValueError, match="output_path_collides_with_roadmap"):
        cli._require_distinct_paths(
            [
                ("roadmap", target),
                ("output", alias),
            ]
        )


def test_cli_writer_rejects_nonregular_destination(tmp_path: Path) -> None:
    cli = _load_cli()
    target = tmp_path / "output.json"
    target.mkdir()
    with pytest.raises(ValueError, match="output_destination_not_regular_file"):
        cli._write_json(target, {"value": 1}, label="output")
    assert target.is_dir()


def test_cli_writer_replace_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    target = tmp_path / "output.json"
    target.write_text("previous\n", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(cli.os, "replace", fail_replace)
    with pytest.raises(ValueError, match="output_write_failed"):
        cli._write_json(target, {"value": 1}, label="output")

    assert target.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob(".output.json.*.tmp")) == []


def test_cli_writer_rejects_nonfinite_before_touching_target(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    target = tmp_path / "output.json"
    target.write_text("previous\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cli._write_json(target, {"value": float("nan")}, label="output")
    assert target.read_text(encoding="utf-8") == "previous\n"


def test_cli_writer_round_trips_utf8_without_temp_residue(tmp_path: Path) -> None:
    cli = _load_cli()
    target = tmp_path / "output.json"
    value = {"label": "перевірка", "value": 1}
    cli._write_json(target, value, label="output")
    assert json.loads(target.read_text(encoding="utf-8")) == value
    assert list(tmp_path.glob(".output.json.*.tmp")) == []



def test_builder_wrong_requirement_evidence_type_fails_closed() -> None:
    with pytest.raises(
        FeasibilityPacketError,
        match="requirement_evidence_not_object",
    ):
        build_200m_feasibility_packet(
            roadmap_snapshot=roadmap(),
            source_git_sha=GIT_A,
            candidate=candidate(),
            measurements_20m=measurements(),
            measurement_authority=measurement_authority(),
            requirement_evidence=[],
            decision="GO",
        )


def test_builder_huge_numeric_measurement_fails_without_overflow() -> None:
    values = measurements()
    values["tokens_per_second"] = 10**10_000
    with pytest.raises(
        FeasibilityPacketError,
        match="measurement_tokens_per_second_invalid",
    ):
        build_200m_feasibility_packet(
            roadmap_snapshot=roadmap(),
            source_git_sha=GIT_A,
            candidate=candidate(),
            measurements_20m=values,
            measurement_authority=measurement_authority(values),
            requirement_evidence=evidence(),
            decision="GO",
        )


@pytest.mark.parametrize(
    "bad_roadmap",
    [
        [],
        {"evidence_state": []},
        {"scale_route": []},
    ],
)
def test_validator_malformed_roadmap_shapes_fail_closed(bad_roadmap: object) -> None:
    packet = build()
    expected = retained_identities_for_built_packet(packet)
    errors = validate_200m_feasibility_packet(
        packet,
        roadmap_snapshot=bad_roadmap,
        expected_packet_sha256=expected["packet_sha256"],
        expected_roadmap_snapshot_sha256=expected["roadmap_snapshot_sha256"],
        expected_source_git_sha=expected["source_git_sha"],
        expected_measurements_20m_sha256=expected["measurements_20m_sha256"],
        expected_requirement_evidence_sha256=expected[
            "requirement_evidence_sha256"
        ],
    )
    assert "roadmap_not_at_200m_feasibility_boundary" in errors


def test_validator_rejects_malformed_external_scalar_identities() -> None:
    packet = build()
    expected = retained_identities_for_built_packet(packet)
    errors = validate_200m_feasibility_packet(
        packet,
        roadmap_snapshot=roadmap(),
        expected_packet_sha256=expected["packet_sha256"],
        expected_roadmap_snapshot_sha256=expected["roadmap_snapshot_sha256"],
        expected_source_git_sha="not-a-git-sha",
        expected_measurements_20m_sha256="not-a-sha256",
        expected_requirement_evidence_sha256=expected[
            "requirement_evidence_sha256"
        ],
    )
    assert "expected_source_git_sha_invalid" in errors
    assert "expected_measurements_20m_sha256_invalid" in errors



def _cli_request() -> dict:
    return {
        "source_git_sha": GIT_A,
        "candidate": candidate(),
        "measurements_20m": measurements(),
        "measurement_authority": measurement_authority(),
        "requirement_evidence": evidence(),
        "decision": "GO",
    }


def test_cli_build_then_verify_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    roadmap_path = tmp_path / "roadmap.json"
    input_path = tmp_path / "input.json"
    packet_path = tmp_path / "packet.json"
    expected_path = tmp_path / "expected.json"
    roadmap_path.write_text(
        json.dumps(roadmap(), sort_keys=True),
        encoding="utf-8",
    )
    input_path.write_text(
        json.dumps(_cli_request(), sort_keys=True),
        encoding="utf-8",
    )

    build_args = cli._parser().parse_args(
        [
            "build",
            "--roadmap",
            str(roadmap_path),
            "--input",
            str(input_path),
            "--output",
            str(packet_path),
            "--external-identities",
            str(expected_path),
        ]
    )
    assert build_args.run(build_args) == 0
    build_stdout = capsys.readouterr().out.strip()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert build_stdout == packet["packet_sha256"]

    verify_args = cli._parser().parse_args(
        [
            "verify",
            "--roadmap",
            str(roadmap_path),
            "--packet",
            str(packet_path),
            "--expected-identities",
            str(expected_path),
        ]
    )
    assert verify_args.run(verify_args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["authority_granted"] is False


def test_cli_main_wrong_requirement_type_returns_bounded_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    roadmap_path = tmp_path / "roadmap.json"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "packet.json"
    request = _cli_request()
    request["requirement_evidence"] = []
    roadmap_path.write_text(json.dumps(roadmap()), encoding="utf-8")
    input_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "build_200m_feasibility_packet.py",
            "build",
            "--roadmap",
            str(roadmap_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert "requirement_evidence_not_object" in captured.err
    assert not output_path.exists()


def test_cli_build_path_collision_preserves_authority_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    roadmap_path = tmp_path / "roadmap.json"
    input_path = tmp_path / "input.json"
    roadmap_path.write_text(json.dumps(roadmap()), encoding="utf-8")
    original = json.dumps(_cli_request(), sort_keys=True)
    input_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "build_200m_feasibility_packet.py",
            "build",
            "--roadmap",
            str(roadmap_path),
            "--input",
            str(input_path),
            "--output",
            str(input_path),
        ],
    )
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert "output_path_collides_with_build_input" in captured.err
    assert input_path.read_text(encoding="utf-8") == original


def test_cli_verify_malformed_roadmap_reports_invalid_not_crash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    good_roadmap_path = tmp_path / "good-roadmap.json"
    bad_roadmap_path = tmp_path / "bad-roadmap.json"
    input_path = tmp_path / "input.json"
    packet_path = tmp_path / "packet.json"
    expected_path = tmp_path / "expected.json"
    good_roadmap_path.write_text(json.dumps(roadmap()), encoding="utf-8")
    bad_roadmap_path.write_text(
        json.dumps({"evidence_state": []}),
        encoding="utf-8",
    )
    input_path.write_text(json.dumps(_cli_request()), encoding="utf-8")

    build_args = cli._parser().parse_args(
        [
            "build",
            "--roadmap",
            str(good_roadmap_path),
            "--input",
            str(input_path),
            "--output",
            str(packet_path),
            "--external-identities",
            str(expected_path),
        ]
    )
    assert build_args.run(build_args) == 0
    capsys.readouterr()

    verify_args = cli._parser().parse_args(
        [
            "verify",
            "--roadmap",
            str(bad_roadmap_path),
            "--packet",
            str(packet_path),
            "--expected-identities",
            str(expected_path),
        ]
    )
    assert verify_args.run(verify_args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False
    assert "roadmap_not_at_200m_feasibility_boundary" in report["errors"]



def test_retained_identity_helper_rejects_resealed_non_authorizing_drift() -> None:
    packet = build()
    packet["authority_boundaries"]["training_executed_by_packet"] = True
    packet["packet_sha256"] = compute_packet_sha256(packet)
    with pytest.raises(
        FeasibilityPacketError,
        match="authority_boundaries_must_be_non_authorizing",
    ):
        retained_identities_for_built_packet(packet)


def test_retained_identity_helper_rejects_resealed_candidate_drift() -> None:
    packet = build()
    packet["candidate"]["parameter_count"] += 1
    packet["packet_sha256"] = compute_packet_sha256(packet)
    with pytest.raises(
        FeasibilityPacketError,
        match="candidate_architecture_evidence_payload_mismatch",
    ):
        retained_identities_for_built_packet(packet)


def test_retained_identity_helper_rejects_unknown_fields() -> None:
    packet = build()
    packet["unexpected"] = True
    packet["packet_sha256"] = compute_packet_sha256(packet)
    with pytest.raises(
        FeasibilityPacketError,
        match="packet_fields_mismatch",
    ):
        retained_identities_for_built_packet(packet)
