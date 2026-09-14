from __future__ import annotations

import copy
import hashlib

import pytest

from twelve_six.data._data232_decontamination_matching import DecontaminationError
from twelve_six.data.eval647_current_corpus_overlap_v1 import build_report, verify_report


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest(payloads: list[bytes]) -> dict:
    return {
        "objects": [
            {
                "source_family": "github:example/one",
                "repository": "example/one",
                "revision": "1" * 40,
                "path": "one.py",
                "expected_raw_bytes": len(payloads[0]),
                "raw_sha256": _sha(payloads[0]),
                "training_allowed": False,
                "tokenizer_fit_allowed": False,
                "permanent_future_training_exclusion": True,
            },
            {
                "source_family": "github:example/two",
                "repository": "example/two",
                "revision": "2" * 40,
                "path": "two.py",
                "expected_raw_bytes": len(payloads[1]),
                "raw_sha256": _sha(payloads[1]),
                "training_allowed": False,
                "tokenizer_fit_allowed": False,
                "permanent_future_training_exclusion": True,
            },
        ],
        "materialization_evidence": {"identity_sha256": "3" * 64},
    }


def _materialization(payload_sha: str, count: int) -> dict:
    return {
        "schema_version": "12-6.d03-post-g05-g06-materialization.v1",
        "status": "MATERIALIZED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "materialization_identity_sha256": "4" * 64,
        "result": {
            "record_payload_jsonl_sha256": payload_sha,
            "record_count": count,
        },
        "truth_boundary": {
            "final_test_outcomes_read": False,
            "training_executed": False,
            "paid_compute_used": False,
        },
    }


def _training(text: str = "def train_only(value):\n    return value + 17\n") -> list[dict[str, str]]:
    return [
        {
            "record_id": "train:1",
            "source_id": "local:one",
            "source_family": "local:training",
            "modality": "code",
            "text": text,
        }
    ]


def _build(training: list[dict[str, str]], payloads: list[bytes]) -> dict:
    payload_sha = "5" * 64
    return build_report(
        training,
        _manifest(payloads),
        payloads,
        _materialization(payload_sha, len(training)),
        actual_training_jsonl_sha256=payload_sha,
        expected_materialization_identity_sha256="4" * 64,
        expected_training_jsonl_sha256=payload_sha,
    )


def test_clean_current_corpus_completes_only_overlap_gate() -> None:
    payloads = [
        b"def reserved_alpha(x):\n    return x * 101\n",
        b"def reserved_beta(y):\n    return y - 303\n",
    ]
    report = _build(_training(), payloads)
    verify_report(report)
    assert report["status"] == "PASS_CURRENT_CORPUS_OVERLAP_ONLY"
    assert report["current_corpus_overlap_zero"] is True
    assert report["match_evidence_count"] == 0
    assert report["completed_gate"] == "GLOBAL_CORPUS_EXACT_AND_NEAR_OVERLAP_ZERO_PROVEN"
    assert "GLOBAL_CORPUS_EXACT_AND_NEAR_OVERLAP_ZERO_PROVEN" not in report["remaining_successor_gates"]
    assert report["selection_validation_records_authorized"] == 0


def test_exact_reserved_payload_in_training_fails_closed() -> None:
    payloads = [
        b"def reserved_alpha(x):\n    return x * 101\n",
        b"def reserved_beta(y):\n    return y - 303\n",
    ]
    training = _training(payloads[0].decode("utf-8"))
    report = _build(training, payloads)
    verify_report(report)
    assert report["status"] == "FAIL_CURRENT_CORPUS_OVERLAP"
    assert report["current_corpus_overlap_zero"] is False
    assert report["match_evidence_count"] >= 1
    assert report["completed_gate"] is None
    assert "GLOBAL_CORPUS_EXACT_AND_NEAR_OVERLAP_ZERO_PROVEN" in report["remaining_successor_gates"]
    assert report["selection_validation_records_authorized"] == 0


def test_training_payload_identity_must_be_independently_bound() -> None:
    payloads = [b"def a():\n    return 1\n", b"def b():\n    return 2\n"]
    with pytest.raises(DecontaminationError, match="independent current-corpus authority"):
        build_report(
            _training(),
            _manifest(payloads),
            payloads,
            _materialization("5" * 64, 1),
            actual_training_jsonl_sha256="6" * 64,
            expected_materialization_identity_sha256="4" * 64,
            expected_training_jsonl_sha256="5" * 64,
        )


def test_reserved_payload_sha_drift_fails_closed() -> None:
    payloads = [b"def a():\n    return 1\n", b"def b():\n    return 2\n"]
    manifest = _manifest(payloads)
    manifest["objects"][0]["raw_sha256"] = "0" * 64
    with pytest.raises(DecontaminationError, match="raw SHA-256 drift"):
        build_report(
            _training(),
            manifest,
            payloads,
            _materialization("5" * 64, 1),
            actual_training_jsonl_sha256="5" * 64,
            expected_materialization_identity_sha256="4" * 64,
            expected_training_jsonl_sha256="5" * 64,
        )


def test_report_tamper_is_detected() -> None:
    payloads = [b"def a():\n    return 1\n", b"def b():\n    return 2\n"]
    report = _build(_training(), payloads)
    tampered = copy.deepcopy(report)
    tampered["selection_validation_records_authorized"] = 2
    with pytest.raises(DecontaminationError, match="self-hash"):
        verify_report(tampered)
