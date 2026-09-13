from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.d03_pdr_attributable_subset import (
    LICENSE,
    LICENSE_URL,
    PUBLISHER,
    REUSE_TERMS_URL,
    SOURCE_DATASET,
    SOURCE_FAMILY,
    SOURCE_REVISION,
    PdrSubsetError,
    derive_exact_replay,
    derive_subset,
    validate_blocker,
)


def _candidate(record_id: str, text: str) -> dict:
    encoded = text.encode("utf-8")
    return {
        "artifact_role": "SOURCE_CANDIDATE_ONLY",
        "record_id": record_id,
        "source_id": SOURCE_FAMILY,
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "origin_url": f"https://publicdomainreview.org/{record_id}",
        "license": LICENSE,
        "attribution_required": True,
        "language": "en",
        "normalized_sha256": hashlib.sha256(encoded).hexdigest(),
        "normalized_utf8_bytes": len(encoded),
        "text": text,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _sidecar(candidate: dict, author: str = "Named Author") -> dict:
    return {
        "schema_version": "12-6.d03-pdr-attribution-sidecar.v1",
        "record_id": candidate["record_id"],
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "normalized_sha256": candidate["normalized_sha256"],
        "normalized_utf8_bytes": candidate["normalized_utf8_bytes"],
        "origin_url": candidate["origin_url"],
        "license": LICENSE,
        "attribution": {
            "author": author,
            "publisher": PUBLISHER,
            "source_url": candidate["origin_url"],
            "license_url": LICENSE_URL,
            "reuse_terms_url": REUSE_TERMS_URL,
        },
        "attribution_metadata_in_training_text": False,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def test_subset_counts_only_attributable_bytes_and_stays_zero_credit() -> None:
    first = _candidate("a", "Attributable alpha text.")
    second = _candidate("b", "Excluded beta text.")
    receipt = derive_subset([first, second], [_sidecar(first)])
    assert receipt["attributable_record_count"] == 1
    assert receipt["attributable_normalized_utf8_bytes"] == first["normalized_utf8_bytes"]
    assert receipt["excluded_record_count"] == 1
    assert receipt["truth_boundary"]["training_authorized_bytes"] == 0
    assert receipt["truth_boundary"]["model_training_executed"] is False


def test_subset_identity_is_deterministic_and_text_free() -> None:
    first = _candidate("a", "Attributable alpha text.")
    receipt_a = derive_subset([first], [_sidecar(first)])
    receipt_b = derive_subset([copy.deepcopy(first)], [_sidecar(copy.deepcopy(first))])
    assert receipt_a == receipt_b
    rendered = json.dumps(receipt_a)
    assert first["text"] not in rendered
    assert "Named Author" not in rendered


def test_rejects_sidecar_not_in_candidate() -> None:
    candidate = _candidate("a", "Attributable alpha text.")
    foreign = _candidate("b", "Foreign beta text.")
    with pytest.raises(PdrSubsetError, match="absent from candidate"):
        derive_subset([candidate], [_sidecar(foreign)])


def test_rejects_sidecar_hash_or_byte_drift() -> None:
    candidate = _candidate("a", "Attributable alpha text.")
    sidecar = _sidecar(candidate)
    sidecar["normalized_utf8_bytes"] += 1
    with pytest.raises(PdrSubsetError, match="sidecar byte drift"):
        derive_subset([candidate], [sidecar])


def test_rejects_blank_author_and_bool_as_int() -> None:
    candidate = _candidate("a", "Attributable alpha text.")
    with pytest.raises(PdrSubsetError, match="author missing"):
        derive_subset([candidate], [_sidecar(candidate, " ")])
    bad = copy.deepcopy(candidate)
    bad["normalized_utf8_bytes"] = True
    with pytest.raises(PdrSubsetError, match="positive int"):
        derive_subset([bad], [])


def test_rejects_unknown_candidate_field_and_sidecar_reordering() -> None:
    first = _candidate("a", "Attributable alpha text.")
    bad = copy.deepcopy(first)
    bad["training_ready"] = True
    with pytest.raises(PdrSubsetError, match="candidate schema drift"):
        derive_subset([bad], [])
    second = _candidate("b", "Attributable beta text.")
    with pytest.raises(PdrSubsetError, match="sidecar order drift"):
        derive_subset([first, second], [_sidecar(second), _sidecar(first)])


def test_checked_in_blocker_is_zero_credit_and_rejects_promotion() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "evidence/d03_pdr_attributable_subset_replay_blocker_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_blocker(payload)
    promoted = copy.deepcopy(payload)
    promoted["derived_source_admission"]["source_admitted_records"] = 498
    with pytest.raises(PdrSubsetError, match="blocked admission boundary drift"):
        validate_blocker(promoted)


def test_exact_wrapper_rejects_unbound_files(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    report = tmp_path / "report.json"
    candidate.write_text("{}\n", encoding="utf-8")
    sidecar.write_text("", encoding="utf-8")
    report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PdrSubsetError, match="candidate file size drift"):
        derive_exact_replay(candidate, sidecar, report)
