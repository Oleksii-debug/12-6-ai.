from __future__ import annotations

import copy
import hashlib

import pytest

from twelve_six.data import (
    privacy_execution_authority as g06,
)
from twelve_six.data import (
    quality_execution_authority as g05,
)

_G05_INPUT_MANIFEST_SHA256 = "a" * 64
_SCOPED_KEY = "whole_corpus_external_llm_cleanliness_claimed"
_LEGACY_KEY = "external_llm_or_api_used_for_data_or_intelligence"


def _g05_records() -> list[dict[str, str]]:
    return [
        {
            "id": "quality-1",
            "mode": "en",
            "text": "A deterministic public documentation sentence for quality execution.\n",
        }
    ]


def _g05_input_rows_sha256(records: list[dict[str, str]]) -> str:
    rows = [
        {
            "record_id": row["id"],
            "mode": row["mode"],
            "payload_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
            "utf8_bytes": len(row["text"].encode("utf-8")),
        }
        for row in sorted(records, key=lambda row: row["id"])
    ]
    return hashlib.sha256(g05._cjson(rows)).hexdigest()


def _reseal_g05(authority: dict[str, object]) -> None:
    core = dict(authority)
    core.pop("execution_identity_sha256", None)
    authority["execution_identity_sha256"] = hashlib.sha256(g05._cjson(core)).hexdigest()


def _g06_records() -> list[dict[str, str]]:
    return [
        {
            "id": "privacy-1",
            "mode": "en",
            "text": "A deterministic public documentation sentence for privacy execution.",
        }
    ]


def _reseal_g06(authority: dict[str, object]) -> None:
    core = dict(authority)
    core.pop("execution_identity_sha256", None)
    authority["execution_identity_sha256"] = hashlib.sha256(g06._cjson(core)).hexdigest()


def test_g05_emits_scoped_nonclaim_and_rejects_legacy_global_negative() -> None:
    records = _g05_records()
    input_rows_sha256 = _g05_input_rows_sha256(records)
    authority = g05.build_quality_execution_authority(
        records,
        input_manifest_sha256=_G05_INPUT_MANIFEST_SHA256,
        expected_input_rows_sha256=input_rows_sha256,
    )
    truth = authority["truth_boundary"]

    assert truth[_SCOPED_KEY] is False
    assert _LEGACY_KEY not in truth

    legacy = copy.deepcopy(authority)
    legacy_truth = legacy["truth_boundary"]
    del legacy_truth[_SCOPED_KEY]
    legacy_truth[_LEGACY_KEY] = False
    _reseal_g05(legacy)
    with pytest.raises(g05.QualityExecutionAuthorityError):
        g05.verify_quality_execution_root(
            legacy,
            expected_input_manifest_sha256=_G05_INPUT_MANIFEST_SHA256,
            expected_input_rows_sha256=input_rows_sha256,
            expected_execution_identity_sha256=legacy["execution_identity_sha256"],
        )


@pytest.mark.parametrize("widened", [0, 0.0, True])
def test_g05_scoped_nonclaim_is_type_sensitive_and_cannot_widen(widened: object) -> None:
    records = _g05_records()
    input_rows_sha256 = _g05_input_rows_sha256(records)
    authority = g05.build_quality_execution_authority(
        records,
        input_manifest_sha256=_G05_INPUT_MANIFEST_SHA256,
        expected_input_rows_sha256=input_rows_sha256,
    )
    tampered = copy.deepcopy(authority)
    tampered["truth_boundary"][_SCOPED_KEY] = widened
    _reseal_g05(tampered)

    with pytest.raises(
        g05.QualityExecutionAuthorityError,
        match="truth boundary drift",
    ):
        g05.verify_quality_execution_root(
            tampered,
            expected_input_manifest_sha256=_G05_INPUT_MANIFEST_SHA256,
            expected_input_rows_sha256=input_rows_sha256,
            expected_execution_identity_sha256=tampered["execution_identity_sha256"],
        )


def test_g06_emits_scoped_nonclaim_and_rejects_legacy_global_negative() -> None:
    records = _g06_records()
    input_rows_sha256 = g06.input_rows_sha256(records)
    authority = g06.build_privacy_execution_authority(
        records,
        expected_input_rows_sha256=input_rows_sha256,
    )
    truth = authority["truth_boundary"]

    assert truth[_SCOPED_KEY] is False
    assert _LEGACY_KEY not in truth

    legacy = copy.deepcopy(authority)
    legacy_truth = legacy["truth_boundary"]
    del legacy_truth[_SCOPED_KEY]
    legacy_truth[_LEGACY_KEY] = False
    _reseal_g06(legacy)
    with pytest.raises(
        g06.PrivacyExecutionAuthorityError,
        match="truth boundary drift",
    ):
        g06.verify_privacy_execution_root(
            legacy,
            expected_input_rows_sha256=input_rows_sha256,
            expected_execution_identity_sha256=legacy["execution_identity_sha256"],
        )


@pytest.mark.parametrize("widened", [0, 0.0, True])
def test_g06_scoped_nonclaim_is_type_sensitive_and_cannot_widen(widened: object) -> None:
    records = _g06_records()
    input_rows_sha256 = g06.input_rows_sha256(records)
    authority = g06.build_privacy_execution_authority(
        records,
        expected_input_rows_sha256=input_rows_sha256,
    )
    tampered = copy.deepcopy(authority)
    tampered["truth_boundary"][_SCOPED_KEY] = widened
    _reseal_g06(tampered)

    with pytest.raises(
        g06.PrivacyExecutionAuthorityError,
        match="truth boundary drift",
    ):
        g06.verify_privacy_execution_root(
            tampered,
            expected_input_rows_sha256=input_rows_sha256,
            expected_execution_identity_sha256=tampered["execution_identity_sha256"],
        )
