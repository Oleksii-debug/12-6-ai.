from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.external_llm_provenance_quarantine_v1 import (
    BLOCKED_PAYLOAD_SHA256,
    EXPECTED_AUTHORITY_IDENTITY_SHA256,
    ExternalLLMProvenanceQuarantineError,
    reject_quarantined_inventory_rows,
    validate_authority,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = json.loads(
    (ROOT / "configs/data/d03_external_llm_provenance_quarantine_v1.json").read_text(
        encoding="utf-8"
    )
)
CURRENT_V8_INVENTORY = json.loads(
    (ROOT / "evidence/data526/v8/record_inventory.json").read_text(encoding="utf-8")
)


def test_exact_quarantine_authority_self_authenticates() -> None:
    assert validate_authority(AUTHORITY) == EXPECTED_AUTHORITY_IDENTITY_SHA256


def test_current_v8_inventory_is_stop_the_line() -> None:
    with pytest.raises(
        ExternalLLMProvenanceQuarantineError,
        match="known external-LLM payload quarantined",
    ):
        reject_quarantined_inventory_rows(CURRENT_V8_INVENTORY["records"], AUTHORITY)


def test_renamed_resealed_copy_of_blocked_payload_is_rejected_by_hash() -> None:
    renamed = [
        {
            "record_id": "totally-new-record-name",
            "source_id": "totally-new-source-name",
            "family": "totally-new-family-name",
            "payload_sha256": BLOCKED_PAYLOAD_SHA256,
            "payload_bytes": 1659,
        }
    ]
    with pytest.raises(
        ExternalLLMProvenanceQuarantineError,
        match="known external-LLM payload quarantined",
    ):
        reject_quarantined_inventory_rows(renamed, AUTHORITY)


def test_known_nomis_identity_is_rejected_even_if_payload_hash_changes() -> None:
    resealed = [
        {
            "record_id": "ua.verba.nomis1864.bounded24",
            "source_id": "resealed-source",
            "family": "resealed-family",
            "payload_sha256": "0" * 64,
            "payload_bytes": 1,
        }
    ]
    with pytest.raises(
        ExternalLLMProvenanceQuarantineError,
        match="known Nomis1864 authority identity quarantined",
    ):
        reject_quarantined_inventory_rows(resealed, AUTHORITY)


def test_unrelated_clean_inventory_is_not_overclaimed_or_blocked() -> None:
    clean = [
        {
            "record_id": "clean-record",
            "source_id": "clean-source",
            "family": "clean-family",
            "payload_sha256": "1" * 64,
            "payload_bytes": 12,
        }
    ]
    assert (
        reject_quarantined_inventory_rows(clean, AUTHORITY)
        == EXPECTED_AUTHORITY_IDENTITY_SHA256
    )
    assert (
        AUTHORITY["enforcement"]["all_other_corpus_bytes_declared_external_llm_clean"]
        is False
    )


def test_self_resealed_authority_cannot_widen_cleanliness_claim() -> None:
    authority = copy.deepcopy(AUTHORITY)
    authority["enforcement"]["all_other_corpus_bytes_declared_external_llm_clean"] = True
    with pytest.raises(
        ExternalLLMProvenanceQuarantineError,
        match="quarantine self-hash mismatch|quarantine enforcement drift",
    ):
        validate_authority(authority)
