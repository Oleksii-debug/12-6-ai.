from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "pdr_subset_authority",
    TOOLS / "derive_d03_pdr_attributable_subset_authority_v1.py",
)
assert SPEC and SPEC.loader
pdr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pdr
SPEC.loader.exec_module(pdr)


def _candidate(index: int) -> dict:
    record_id = f"record-{index:04d}"
    text = f"A stable attributable candidate text number {index}."
    encoded = text.encode("utf-8")
    return {
        "artifact_role": "SOURCE_CANDIDATE_ONLY",
        "record_id": record_id,
        "source_id": pdr.SOURCE_FAMILY,
        "source_dataset": pdr.SOURCE_DATASET,
        "source_revision": pdr.SOURCE_REVISION,
        "origin_url": f"https://publicdomainreview.org/essay/{record_id}/",
        "license": pdr.EXPECTED_LICENSE,
        "attribution_required": True,
        "language": "en",
        "normalized_sha256": hashlib.sha256(encoded).hexdigest(),
        "normalized_utf8_bytes": len(encoded),
        "text": text,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _sidecar(candidate: dict, *, author: str = "Ada Example") -> dict:
    return {
        "schema_version": pdr.SIDECAR_SCHEMA,
        "record_id": candidate["record_id"],
        "source_dataset": pdr.SOURCE_DATASET,
        "source_revision": pdr.SOURCE_REVISION,
        "normalized_sha256": candidate["normalized_sha256"],
        "normalized_utf8_bytes": candidate["normalized_utf8_bytes"],
        "origin_url": candidate["origin_url"],
        "license": pdr.EXPECTED_LICENSE,
        "attribution": {
            "author": author,
            "publisher": pdr.PUBLISHER,
            "source_url": candidate["origin_url"],
            "license_url": pdr.LICENSE_URL,
            "reuse_terms_url": pdr.REUSE_TERMS_URL,
        },
        "attribution_metadata_in_training_text": False,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


@pytest.fixture
def replay() -> tuple[list[dict], list[dict]]:
    candidates = [_candidate(index) for index in range(pdr.EXPECTED_CANDIDATE_RECORDS)]
    sidecar = [_sidecar(row) for row in candidates[: pdr.EXPECTED_ATTRIBUTABLE_RECORDS]]
    return candidates, sidecar


def test_exact_subset_accounting_is_text_free_and_zero_credit(
    replay: tuple[list[dict], list[dict]],
) -> None:
    candidates, sidecar = replay
    authority = pdr.derive_subset(candidates, sidecar)
    expected_bytes = sum(
        row["normalized_utf8_bytes"]
        for row in candidates[: pdr.EXPECTED_ATTRIBUTABLE_RECORDS]
    )

    assert authority["attributable_record_count"] == 498
    assert authority["excluded_record_count"] == 668
    assert authority["attributable_normalized_utf8_bytes"] == expected_bytes
    assert authority["pdr_source_admitted_records"] == 0
    assert authority["pdr_source_admitted_bytes"] == 0
    assert authority["canonical_capacity_credited"] == 0
    assert authority["training_authorized_bytes"] == 0
    assert authority["authorized_optimized_target_exposure"] == 0
    assert authority["tokenizer_fit_authorized"] is False
    assert authority["training_executed"] is False

    durable = json.dumps(authority, ensure_ascii=False, sort_keys=True)
    assert "record-0000" not in durable
    assert "Ada Example" not in durable
    assert "publicdomainreview.org/essay/record-" not in durable


def test_same_inputs_produce_same_authority(
    replay: tuple[list[dict], list[dict]],
) -> None:
    candidates, sidecar = replay
    assert pdr.derive_subset(copy.deepcopy(candidates), copy.deepcopy(sidecar)) == pdr.derive_subset(
        copy.deepcopy(candidates), copy.deepcopy(sidecar)
    )


def test_sidecar_candidate_byte_identity_substitution_fails_closed(
    replay: tuple[list[dict], list[dict]],
) -> None:
    candidates, sidecar = replay
    sidecar[0]["normalized_utf8_bytes"] += 1
    with pytest.raises(pdr.SubsetAccountingError, match="sidecar/candidate identity drift"):
        pdr.derive_subset(candidates, sidecar)


def test_sidecar_candidate_digest_substitution_fails_closed(
    replay: tuple[list[dict], list[dict]],
) -> None:
    candidates, sidecar = replay
    sidecar[0]["normalized_sha256"] = "0" * 64
    with pytest.raises(pdr.SubsetAccountingError, match="sidecar/candidate identity drift"):
        pdr.derive_subset(candidates, sidecar)


def test_duplicate_attributable_identity_fails_closed(
    replay: tuple[list[dict], list[dict]],
) -> None:
    candidates, sidecar = replay
    sidecar[1] = copy.deepcopy(sidecar[0])
    with pytest.raises(pdr.SubsetAccountingError, match="duplicate sidecar identity"):
        pdr.derive_subset(candidates, sidecar)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), "wrong", "sidecar version drift"),
        (("training_eligible",), True, "sidecar training drift"),
        (("normalized_utf8_bytes",), True, "sidecar bytes invalid"),
        (("attribution", "publisher"), "Other", "attribution publisher drift"),
        (("attribution", "license_url"), "https://example.invalid/", "attribution license URL drift"),
        (("attribution", "reuse_terms_url"), "https://example.invalid/", "attribution reuse terms drift"),
    ],
)
def test_sidecar_policy_substitution_fails_closed(
    replay: tuple[list[dict], list[dict]],
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    candidates, sidecar = replay
    target = sidecar[0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(pdr.SubsetAccountingError, match=message):
        pdr.derive_subset(candidates, sidecar)


def test_unknown_sidecar_member_fails_closed(
    replay: tuple[list[dict], list[dict]],
) -> None:
    candidates, sidecar = replay
    sidecar[0]["unexpected"] = "value"
    with pytest.raises(pdr.SubsetAccountingError, match="sidecar schema drift"):
        pdr.derive_subset(candidates, sidecar)


def test_candidate_text_tamper_fails_closed(
    replay: tuple[list[dict], list[dict]],
) -> None:
    candidates, sidecar = replay
    candidates[0]["text"] += " tampered"
    with pytest.raises(pdr.SubsetAccountingError, match="candidate byte count mismatch"):
        pdr.derive_subset(candidates, sidecar)


def test_strict_json_rejects_duplicate_members_and_nonfinite_numbers() -> None:
    with pytest.raises(pdr.SubsetAccountingError, match="duplicate JSON member"):
        pdr._loads_strict(b'{"a":1,"a":2}', label="probe")
    with pytest.raises(pdr.SubsetAccountingError, match="non-finite JSON number"):
        pdr._loads_strict(b'{"a":NaN}', label="probe")


def test_historical_report_rejects_bool_as_zero() -> None:
    report = {
        "selected_record_count": pdr.EXPECTED_CANDIDATE_RECORDS,
        "attributable_record_count": pdr.EXPECTED_ATTRIBUTABLE_RECORDS,
        "excluded_record_count": pdr.EXPECTED_EXCLUDED_RECORDS,
        "excluded_reason_counts": {"missing": pdr.EXPECTED_EXCLUDED_RECORDS},
        "excluded_candidate_identity_sha256": pdr.EXPECTED_EXCLUSION_IDENTITY,
        "attribution_sidecar_jsonl_sha256": pdr.EXPECTED_SIDECAR_SHA256,
        "candidate_projection_identity_sha256": pdr.EXPECTED_CANDIDATE_PROJECTION,
        "report_identity_sha256": pdr.EXPECTED_REPORT_IDENTITY,
        "source_rights_review_status": "REVIEW_REQUIRED",
        "training_authorized_bytes": False,
        "authorized_optimized_target_exposure": 0,
        "canonical_corpus_admitted": False,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
    }
    with pytest.raises(pdr.SubsetAccountingError, match="training credit drift"):
        pdr._validate_historical_report(report)
