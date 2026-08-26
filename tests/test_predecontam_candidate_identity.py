from __future__ import annotations

import copy

import pytest

from tools.build_predecontam_candidate_identity import (
    INPUT_SCHEMA,
    STRATA,
    _synthetic_record,
    build_candidate,
)
from tools.validate_predecontam_candidate_identity import validate_candidate


def _candidate() -> dict:
    records = []
    index = 0
    for stratum in STRATA:
        for suffix in ("a", "b"):
            index += 1
            records.append(
                _synthetic_record(index, stratum, f"{stratum}.family.{suffix}")
            )
    return build_candidate({"schema": INPUT_SCHEMA, "records": records})


def test_validator_accepts_exact_builder_output() -> None:
    report = validate_candidate(_candidate())
    assert report["verdict"] == "PASS"
    assert report["record_count"] == 6
    assert report["final_training_authorized"] is False
    assert report["decontamination_executed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_count", 999),
        ("total_normalized_utf8_bytes", 1),
        ("source_authority_bundle_identity_sha256", "0" * 64),
        ("candidate_record_inventory_identity_sha256", "0" * 64),
        ("candidate_identity_sha256", "0" * 64),
        ("final_training_authorized", True),
        ("decontamination_executed", True),
        ("replay_authorized", True),
    ],
)
def test_validator_rejects_tampered_top_level_fields(field: str, value: object) -> None:
    candidate = _candidate()
    candidate[field] = value
    with pytest.raises(ValueError):
        validate_candidate(candidate)


def test_validator_rejects_noncanonical_record_order() -> None:
    candidate = _candidate()
    candidate["records"] = list(reversed(candidate["records"]))
    with pytest.raises(ValueError, match="canonical builder order"):
        validate_candidate(candidate)


def test_validator_rejects_duplicate_source_identity_even_if_candidate_hash_is_stale() -> None:
    candidate = _candidate()
    tampered = copy.deepcopy(candidate["records"][0])
    tampered["normalized_sha256"] = "f" * 64
    candidate["records"].append(tampered)
    with pytest.raises(ValueError):
        validate_candidate(candidate)


def test_validator_rejects_record_contract_extension() -> None:
    candidate = _candidate()
    candidate["records"][0]["unexpected_authority"] = "should-not-be-silently-accepted"
    with pytest.raises(ValueError, match="field mismatch"):
        validate_candidate(candidate)
