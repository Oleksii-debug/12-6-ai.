from __future__ import annotations

import copy
from pathlib import Path

import pytest

import twelve_six.checkpoint.portability as portability
from twelve_six.checkpoint.portability import (
    CheckpointPortabilityError,
    consume_checkpoint_portability_bundle,
    produce_checkpoint_portability_bundle,
    validate_consumer_report,
    validate_producer_report,
)


def _other_architecture(current: str) -> str:
    return "aarch64" if current != "aarch64" else "x86_64"


def test_checkpoint_portability_bundle_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    producer = produce_checkpoint_portability_bundle(
        Path("."),
        source_sha="a" * 40,
        output_dir=bundle,
        verify_checkout=False,
        require_architecture=None,
    )
    producer_result = validate_producer_report(producer)
    assert producer_result["status"] == "PASS"
    assert producer["identity"]["optimizer_step"] == 1
    assert producer["checkpoint"]["verified_on_producer"] is True
    assert len(producer["checkpoint"]["artifact_sha256"]) == 5

    consumer_arch = _other_architecture(producer["source"]["architecture"])
    monkeypatch.setattr(
        portability,
        "_normalized_architecture",
        lambda value=None: consumer_arch,
    )
    consumer = consume_checkpoint_portability_bundle(
        Path("."),
        source_sha="a" * 40,
        bundle_dir=bundle,
        output=tmp_path / "consumer.json",
        verify_checkout=False,
        require_architecture=None,
    )
    consumer_result = validate_consumer_report(consumer, producer=producer)
    assert consumer_result["status"] == "PASS"
    assert consumer["source"]["architecture"] == consumer_arch
    assert consumer["producer"]["architecture"] != consumer_arch
    assert consumer["checkpoint"]["verify_checkpoint_pass"] is True
    assert consumer["checkpoint"]["trainer_restore_pass"] is True
    assert consumer["checkpoint"]["first_party_load_pass"] is True
    assert consumer["checkpoint"]["rng_restored"] is False
    assert (
        consumer["checkpoint"]["artifact_sha256"]
        == producer["checkpoint"]["artifact_sha256"]
    )
    assert (
        consumer["restored_state"]["optimizer_step"]
        == producer["identity"]["optimizer_step"]
    )
    assert consumer["claims"]["serialization_portability_proven"] is True
    assert consumer["claims"]["cross_arch_training_bitwise_reproducibility"] is False
    assert consumer["claims"]["cross_arch_inference_bitwise_reproducibility"] is False
    assert consumer["claims"]["rng_cross_arch_equivalence"] is False


def test_checkpoint_portability_reports_fail_closed_on_tamper(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    producer = produce_checkpoint_portability_bundle(
        Path("."),
        source_sha="b" * 40,
        output_dir=bundle,
        verify_checkout=False,
        require_architecture=None,
    )
    tampered = copy.deepcopy(producer)
    tampered["claims"]["cross_arch_training_bitwise_reproducibility"] = True

    with pytest.raises(CheckpointPortabilityError, match="self-hash"):
        validate_producer_report(tampered)


def test_checkpoint_portability_rejects_abbreviated_source(tmp_path: Path) -> None:
    with pytest.raises(CheckpointPortabilityError, match="full lowercase"):
        produce_checkpoint_portability_bundle(
            Path("."),
            source_sha="abc123",
            output_dir=tmp_path / "bundle",
            verify_checkout=False,
            require_architecture=None,
        )
