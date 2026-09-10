from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import pytest

from twelve_six.packing import two_clean_build as two_clean
from twelve_six.packing.core import PACKING_CONFIG_HASH
from twelve_six.packing.loss_materialization import (
    LossMaterializationDocument,
    tokenizer_identity_sha256,
)
from twelve_six.tokenization import ByteTokenizer


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


def _tokenizer_identity() -> str:
    return tokenizer_identity_sha256(ByteTokenizer())


def _runtime_identity() -> str:
    return two_clean.current_runtime_identity_sha256()


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
        expected_tokenizer_identity_sha256=_tokenizer_identity(),
        expected_packing_identity_sha256=PACKING_CONFIG_HASH,
        expected_runtime_identity_sha256=_runtime_identity(),
    )


def _rehash_packet(packet: dict) -> dict:
    value = copy.deepcopy(packet)
    value.pop("input_packet_identity_sha256", None)
    value["input_packet_identity_sha256"] = two_clean._sha256_obj(value)
    return value


def _materialization_bytes(packet: dict) -> bytes:
    verified = two_clean._verify_input_packet(
        packet,
        expected_identity_sha256=packet["input_packet_identity_sha256"],
    )
    return two_clean._canonical_json_bytes(two_clean._build_one(verified))


def _compare(first: bytes, second: bytes, **overrides: object) -> dict:
    kwargs: dict[str, object] = {
        "expected_corpus_identity_sha256": "f" * 64,
        "expected_stage_bindings": _bindings(),
        "expected_tokenizer_identity_sha256": _tokenizer_identity(),
        "expected_packing_identity_sha256": PACKING_CONFIG_HASH,
    }
    kwargs.update(overrides)
    return two_clean.compare_clean_build_bytes(first, second, **kwargs)


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
    assert proof["tokenizer_identity_sha256"] == _tokenizer_identity()
    assert proof["packing_identity_sha256"] == PACKING_CONFIG_HASH
    assert proof["runtime_identity_sha256"] == _runtime_identity()
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
        packet,
        expected_input_packet_identity_sha256=expected,
    )
    second = two_clean.prove_two_clean_build(
        packet,
        expected_input_packet_identity_sha256=expected,
    )
    assert first == second


def test_self_consistent_input_substitution_fails_external_identity_binding() -> None:
    packet = _packet()
    expected = packet["input_packet_identity_sha256"]
    substituted = copy.deepcopy(packet)
    substituted["terminal_corpus_authority_identity_sha256"] = "0" * 64
    substituted = _rehash_packet(substituted)

    with pytest.raises(two_clean.TwoCleanBuildError, match="independently expected"):
        two_clean.prove_two_clean_build(
            substituted,
            expected_input_packet_identity_sha256=expected,
        )


def test_different_but_individually_valid_builds_fail_literal_comparison() -> None:
    first = _materialization_bytes(_packet("first deterministic fixture\n" * 20))
    second = _materialization_bytes(_packet("second deterministic fixture\n" * 20))

    with pytest.raises(two_clean.TwoCleanBuildError, match="bytes differ"):
        _compare(first, second)


def test_noncanonical_materialization_serialization_is_rejected() -> None:
    canonical = _materialization_bytes(_packet())
    noncanonical = canonical.rstrip(b"\n") + b"  \n"

    with pytest.raises(two_clean.TwoCleanBuildError, match="not canonical"):
        _compare(noncanonical, noncanonical)


def test_stage_binding_drift_is_rejected_even_when_build_is_self_consistent() -> None:
    payload = _materialization_bytes(_packet())
    drifted = _bindings()
    drifted["split"] = "1" * 64

    with pytest.raises(two_clean.TwoCleanBuildError, match="stage binding drifted"):
        _compare(payload, payload, expected_stage_bindings=drifted)


def test_tokenizer_identity_is_externally_bound() -> None:
    payload = _materialization_bytes(_packet())
    with pytest.raises(two_clean.TwoCleanBuildError, match="tokenizer identity drifted"):
        _compare(
            payload,
            payload,
            expected_tokenizer_identity_sha256="1" * 64,
        )


def test_packing_identity_is_externally_bound() -> None:
    payload = _materialization_bytes(_packet())
    with pytest.raises(two_clean.TwoCleanBuildError, match="packing identity drifted"):
        _compare(
            payload,
            payload,
            expected_packing_identity_sha256="1" * 64,
        )


def test_runtime_identity_substitution_fails_before_child_spawn() -> None:
    packet = _packet()
    packet["expected_runtime_identity_sha256"] = "0" * 64
    substituted = _rehash_packet(packet)

    with pytest.raises(two_clean.TwoCleanBuildError, match="trusted current runtime"):
        two_clean.prove_two_clean_build(
            substituted,
            expected_input_packet_identity_sha256=(
                substituted["input_packet_identity_sha256"]
            ),
        )


def test_alternate_python_executable_is_rejected() -> None:
    packet = _packet()
    alternate = str(Path(os.sys.executable).resolve().parent)

    with pytest.raises(two_clean.TwoCleanBuildError, match="trusted current runtime"):
        two_clean.prove_two_clean_build(
            packet,
            expected_input_packet_identity_sha256=packet["input_packet_identity_sha256"],
            python_executable=alternate,
        )


def test_parent_python_injection_environment_is_not_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    packet = _packet()
    monkeypatch.setenv("PYTHONHOME", "/attacker/python-home")
    monkeypatch.setenv("PYTHONSTARTUP", "/attacker/startup.py")
    monkeypatch.setenv("PYTHONINSPECT", "1")
    monkeypatch.setenv("PYTHONCASEOK", "1")
    monkeypatch.setenv("PYTHONPATH", "/attacker/imports")

    proof = two_clean.prove_two_clean_build(
        packet,
        expected_input_packet_identity_sha256=packet["input_packet_identity_sha256"],
    )
    assert proof["runtime_identity_sha256"] == _runtime_identity()


def test_clean_child_environment_has_only_trusted_python_controls() -> None:
    source_root = two_clean._trusted_source_root()
    env = two_clean._clean_child_env(source_root)

    assert env == {
        "PYTHONPATH": str(source_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env


def test_durable_proof_is_independently_verifiable_and_tamper_fails() -> None:
    packet = _packet()
    proof = two_clean.prove_two_clean_build(
        packet,
        expected_input_packet_identity_sha256=packet["input_packet_identity_sha256"],
    )
    verified = two_clean.verify_proof(
        proof,
        expected_proof_identity_sha256=proof["proof_identity_sha256"],
        expected_input_packet_identity_sha256=packet["input_packet_identity_sha256"],
        expected_terminal_corpus_identity_sha256="f" * 64,
        expected_stage_bindings=_bindings(),
        expected_tokenizer_identity_sha256=_tokenizer_identity(),
        expected_packing_identity_sha256=PACKING_CONFIG_HASH,
        expected_runtime_identity_sha256=_runtime_identity(),
    )
    assert verified == proof

    tampered = copy.deepcopy(proof)
    tampered["byte_identical"] = False
    with pytest.raises(two_clean.TwoCleanBuildError, match="self-identity"):
        two_clean.verify_proof(
            tampered,
            expected_proof_identity_sha256=proof["proof_identity_sha256"],
            expected_input_packet_identity_sha256=packet["input_packet_identity_sha256"],
            expected_terminal_corpus_identity_sha256="f" * 64,
            expected_stage_bindings=_bindings(),
            expected_tokenizer_identity_sha256=_tokenizer_identity(),
            expected_packing_identity_sha256=PACKING_CONFIG_HASH,
            expected_runtime_identity_sha256=_runtime_identity(),
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
