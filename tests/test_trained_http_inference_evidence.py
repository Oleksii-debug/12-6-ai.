from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from twelve_six.inference.http_evidence import (
    SCHEMA_VERSION,
    collect_trained_http_inference_evidence,
    validate_trained_http_inference_evidence,
)


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def test_real_trained_checkpoint_round_trips_over_loopback_http(tmp_path: Path) -> None:
    head = _head()
    payload = collect_trained_http_inference_evidence(
        Path("."),
        source_sha=head,
        output_dir=tmp_path / "evidence",
        train_steps=4,
        seed=1337,
        max_tokens=4,
    )
    validate_trained_http_inference_evidence(payload, expected_source_sha=head)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["candidate"]["source_sha"] == head
    assert payload["checkpoint"]["checkpoint_id"] == payload["backend"]["checkpoint_id"]
    assert payload["checkpoint"]["git_sha"] == payload["backend"]["git_sha"] == head
    assert payload["checkpoint"]["step"] == payload["backend"]["step"] == 4
    assert payload["parity"]["passed"] is True
    assert payload["parity"]["max_abs_error"] == 0.0
    assert payload["http"]["transport"] == "real_loopback_tcp_http11"
    assert payload["http"]["greedy"]["matches_direct_completion"] is True
    assert payload["http"]["seeded_sampling"]["same_seed_repeatable"] is True
    assert payload["http"]["stop_string"]["stop_and_strip_verified"] is True
    assert payload["http"]["context"]["exact_limit_verified"] is True
    assert payload["http"]["context"]["over_limit_rejected"] is True
    assert payload["http"]["raw_base_boundary"]["messages_rejected"] is True
    assert payload["http"]["raw_base_boundary"]["chat_endpoint_rejected"] is True
    assert (tmp_path / "evidence" / "trained-checkpoint" / "manifest.json").is_file()
    assert (tmp_path / "evidence" / "s0-trained-http-inference-evidence.json").is_file()


def test_evidence_validator_rejects_tamper(tmp_path: Path) -> None:
    head = _head()
    payload = collect_trained_http_inference_evidence(
        Path("."),
        source_sha=head,
        output_dir=tmp_path / "evidence",
        train_steps=2,
        max_tokens=2,
    )
    tampered = copy.deepcopy(payload)
    tampered["http"]["raw_base_boundary"]["messages_rejected"] = False
    with pytest.raises(ValueError, match="self-hash mismatch"):
        validate_trained_http_inference_evidence(tampered, expected_source_sha=head)


def test_source_sha_must_equal_checkout_head(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not equal checkout HEAD"):
        collect_trained_http_inference_evidence(
            Path("."),
            source_sha="0" * 40,
            output_dir=tmp_path / "evidence",
            train_steps=1,
            max_tokens=1,
        )
