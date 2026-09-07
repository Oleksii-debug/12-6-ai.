from __future__ import annotations

import copy
import hashlib

import pytest

from twelve_six.packing import two_clean_build as two_clean
from twelve_six.packing.loss_materialization import LossMaterializationDocument


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bindings(seed: str = "a") -> dict[str, str]:
    return {
        "normalization": seed * 64,
        "evaluation_reservations": "b" * 64,
        "dedup": "c" * 64,
        "split": "d" * 64,
        "packing": "e" * 64,
    }


def _documents(text: str = "fresh-process fixture\n" * 20) -> tuple[LossMaterializationDocument, ...]:
    return (
        LossMaterializationDocument(
            document_id="doc-train",
            text=text,
            source_id="source-a",
            language="en",
            modality="text",
            family_id="family-a",
            normalized_payload_sha256=_sha(text),
            source_bytes=len(text.encode("utf-8")),
            split="train",
            dedup_cluster_id="cluster-a",
        ),
        LossMaterializationDocument(
            document_id="doc-heldout",
            text="reserved validation fixture",
            source_id="source-b",
            language="en",
            modality="text",
            family_id="family-b",
            normalized_payload_sha256=_sha("reserved validation fixture"),
            source_bytes=len(b"reserved validation fixture"),
            split="validation",
            dedup_cluster_id="cluster-b",
            evaluation_reserved=True,
        ),
    )


def _packet(text: str = "fresh-process fixture\n" * 20) -> dict:
    return two_clean.make_input_packet(
        _documents(text),
        terminal_corpus_authority_identity_sha256="f" * 64,
        stage_bindings=_bindings(),
    )


def _materialization_bytes(packet: dict) -> bytes:
    verified = two_clean._verify_input_packet(
        packet,
        expected_identity_sha256=packet["input_packet_identity_sha256"],
    )
    return two_clean._canonical_json_bytes(two_clean._build_one(verified))


def test_two_fresh_processes_produce_literal_byte_identity() -> None:
    packet = _packet()
    proof = two_clean.prove_two_clean_build(
        packet,
        expected_input_packet_identity_sha256=packet["input_packet_identity_sha256"],
    )

    assert proof["schema_version"] == two_clean.PROOF_SCHEMA
    assert proof["fresh_process_count"] == 2
    assert proof["byte_identical"] is True
    assert proof["build_a_sha256"] == proof["build_b_sha256"]
    assert len(proof["materialization_identity_sha256"]) == 64
    assert proof["claim_boundary"] == {
        "contains_source_text": False,
        "authorizes_training": False,
        "authorizes_paid_compute": False,
        "creates_positive_unique_loss_authority": False,
    }
    assert "fresh-process fixture" not in str(proof)
    body = dict(proof)
    identity = body.pop("proof_identity_sha256")
    assert identity == two_clean._sha256_obj(body)


def test_proof_is_deterministic_across_independent_pairs() -> None:
    packet = _packet()
    expected = packet["input_packet_identity_sha256"]
    first = two_clean.prove_two_clean_build(
        packet, expected_input_packet_identity_sha256=expected
    )
    second = two_clean.prove_two_clean_build(
        packet, expected_input_packet_identity_sha256=expected
    )
    assert first == second


def test_self_consistent_input_substitution_fails_external_identity_binding() -> None:
    packet = _packet()
    expected = packet["input_packet_identity_sha256"]
    substituted = copy.deepcopy(packet)
    substituted["terminal_corpus_authority_identity_sha256"] = "0" * 64
    substituted.pop("input_packet_identity_sha256")
    substituted["input_packet_identity_sha256"] = two_clean._sha256_obj(substituted)

    with pytest.raises(two_clean.TwoCleanBuildError, match="independently expected"):
        two_clean.prove_two_clean_build(
            substituted,
            expected_input_packet_identity_sha256=expected,
        )


def test_different_but_individually_valid_builds_fail_literal_comparison() -> None:
    first_packet = _packet("first deterministic fixture\n" * 20)
    second_packet = _packet("second deterministic fixture\n" * 20)
    first = _materialization_bytes(first_packet)
    second = _materialization_bytes(second_packet)

    with pytest.raises(two_clean.TwoCleanBuildError, match="bytes differ"):
        two_clean.compare_clean_build_bytes(
            first,
            second,
            expected_corpus_identity_sha256="f" * 64,
            expected_stage_bindings=_bindings(),
        )


def test_noncanonical_materialization_serialization_is_rejected() -> None:
    packet = _packet()
    canonical = _materialization_bytes(packet)
    noncanonical = canonical.rstrip(b"\n") + b"  \n"

    with pytest.raises(two_clean.TwoCleanBuildError, match="not canonical"):
        two_clean.compare_clean_build_bytes(
            noncanonical,
            noncanonical,
            expected_corpus_identity_sha256="f" * 64,
            expected_stage_bindings=_bindings(),
        )


def test_stage_binding_drift_is_rejected_even_when_build_is_self_consistent() -> None:
    packet = _packet()
    payload = _materialization_bytes(packet)
    drifted = _bindings()
    drifted["split"] = "1" * 64

    with pytest.raises(two_clean.TwoCleanBuildError, match="stage binding drifted"):
        two_clean.compare_clean_build_bytes(
            payload,
            payload,
            expected_corpus_identity_sha256="f" * 64,
            expected_stage_bindings=drifted,
        )


def test_ephemeral_packet_contains_text_but_claims_zero_training_authority() -> None:
    packet = _packet()
    assert any("text" in row for row in packet["documents"])
    assert packet["claim_boundary"]["ephemeral_input_contains_source_text"] is True
    assert packet["claim_boundary"]["durable_proof_contains_source_text"] is False
    assert packet["claim_boundary"]["authorizes_training"] is False


def test_invalid_timeout_fails_before_spawning_children() -> None:
    packet = _packet()
    with pytest.raises(two_clean.TwoCleanBuildError, match="positive integer"):
        two_clean.prove_two_clean_build(
            packet,
            expected_input_packet_identity_sha256=packet["input_packet_identity_sha256"],
            timeout_seconds=True,
        )
