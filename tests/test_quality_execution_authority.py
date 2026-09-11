from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.document_quality import default_quality_policy
from twelve_six.data.quality_execution_authority import (
    QUALITY_EXECUTION_AUTHORITY_CLASS,
    QUALITY_EXECUTION_SCHEMA,
    QualityExecutionAuthorityError,
    build_quality_execution_authority,
    verify_quality_execution_authority,
    verify_quality_execution_root,
)
from twelve_six.data.quality_granularity import frozen_granularity_policy

INPUT_MANIFEST_SHA256 = "a" * 64


def _alpha_suffix(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    base = len(alphabet)
    value = index
    chars: list[str] = []
    while True:
        chars.append(alphabet[value % base])
        value = value // base - 1
        if value < 0:
            return "".join(reversed(chars))


def _accepted_en(record_id: str = "en-1") -> dict[str, str]:
    text = " ".join(f"section{_alpha_suffix(index)}" for index in range(80))
    return {"id": record_id, "mode": "en", "text": text + ".\n"}


def _accepted_code(record_id: str = "code-1") -> dict[str, str]:
    text = "\n".join(
        [
            "def add(left, right):",
            "    result = left + right",
            "    if result:",
            "        return result",
            "    return 0",
        ]
    )
    return {"id": record_id, "mode": "code", "text": text + "\n"}


def _partial_en(record_id: str = "en-partial") -> dict[str, str]:
    repeated = ".. note:: documentation documentation documentation\n" * 350
    valid = " ".join(
        f"section{_alpha_suffix(index)}" for index in range(3000)
    )
    return {"id": record_id, "mode": "en", "text": repeated + valid + ".\n"}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_authority_binds_external_manifest_policies_rows_and_zero_credit() -> None:
    records = [_accepted_en(), _accepted_code()]
    authority = build_quality_execution_authority(
        records,
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    assert authority["schema_version"] == QUALITY_EXECUTION_SCHEMA
    assert authority["authority_class"] == QUALITY_EXECUTION_AUTHORITY_CLASS
    assert authority["input_manifest_sha256"] == INPUT_MANIFEST_SHA256
    assert authority["quality_threshold_policy"] == {
        "policy_id": default_quality_policy().policy_id,
        "policy_sha256": default_quality_policy().manifest()["policy_sha256"],
    }
    granularity = frozen_granularity_policy()
    assert authority["quality_granularity_policy"] == {
        "policy_id": granularity.policy_id,
        "policy_sha256": granularity.manifest()["policy_sha256"],
    }
    assert authority["truth_boundary"]["current_corpus_eligible"] is False
    assert authority["truth_boundary"]["training_authorized_bytes"] == 0
    assert (
        authority["truth_boundary"]["authorized_optimized_target_exposure"]
        == 0
    )
    assert authority["truth_boundary"]["training_executed"] is False
    assert authority["truth_boundary"]["learned_weights_created"] is False

    by_id = {row["record_id"]: row for row in authority["records"]}
    for source in records:
        row = by_id[source["id"]]
        assert row["payload_sha256"] == _hash_text(source["text"])
        assert row["utf8_bytes"] == len(source["text"].encode("utf-8"))
        assert row["retained_utf8_bytes"] + row["rejected_utf8_bytes"] == row[
            "utf8_bytes"
        ]


def test_execution_is_order_independent_and_canonical_reexecution_verifies() -> None:
    records = [
        _accepted_en(),
        _accepted_code(),
        {"id": "reject", "mode": "en", "text": "tiny"},
    ]
    first = build_quality_execution_authority(
        records,
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )
    second = build_quality_execution_authority(
        list(reversed(records)),
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    assert first == second
    identity = first["execution_identity_sha256"]
    assert verify_quality_execution_authority(
        first,
        records,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        expected_execution_identity_sha256=identity,
    ) == identity
    assert verify_quality_execution_root(
        first,
        expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        expected_execution_identity_sha256=identity,
    ) == identity


def test_partial_quality_units_bind_exact_payload_spans_and_bytes() -> None:
    source = _partial_en()
    authority = build_quality_execution_authority(
        [source],
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )
    row = authority["records"][0]

    assert row["status"] == "RETAIN_PARTIAL"
    assert len(row["units"]) > 1
    accepted = rejected = 0
    for unit in row["units"]:
        payload = source["text"][unit["start_char"] : unit["end_char"]]
        assert unit["payload_sha256"] == _hash_text(payload)
        assert unit["utf8_bytes"] == len(payload.encode("utf-8"))
        if unit["accepted"]:
            accepted += unit["utf8_bytes"]
        else:
            rejected += unit["utf8_bytes"]
    assert accepted == row["retained_utf8_bytes"]
    assert rejected == row["rejected_utf8_bytes"]
    assert accepted + rejected == row["utf8_bytes"]


def test_authority_never_persists_source_text_or_excerpt() -> None:
    secret = "NEVER_PERSIST_THIS_SOURCE_SENTINEL"
    source = _accepted_en()
    source["text"] += " " + secret
    authority = build_quality_execution_authority(
        [source],
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )
    serialized = json.dumps(authority, sort_keys=True)

    assert secret not in serialized
    assert '"text"' not in serialized
    assert '"preview"' not in serialized


def test_payload_or_decision_substitution_fails_canonical_reexecution() -> None:
    records = [_accepted_en(), _accepted_code()]
    authority = build_quality_execution_authority(
        records,
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    tampered = copy.deepcopy(authority)
    tampered["records"][0]["payload_sha256"] = "b" * 64
    with pytest.raises(
        QualityExecutionAuthorityError,
        match="canonical re-execution",
    ):
        verify_quality_execution_authority(
            tampered,
            records,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )

    tampered = copy.deepcopy(authority)
    tampered["records"][0]["status"] = "REJECT_DOCUMENT"
    with pytest.raises(
        QualityExecutionAuthorityError,
        match="canonical re-execution",
    ):
        verify_quality_execution_authority(
            tampered,
            records,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )


def test_self_consistent_reseal_cannot_replace_independently_pinned_root() -> None:
    records = [_accepted_en()]
    authority = build_quality_execution_authority(
        records,
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )
    pinned = authority["execution_identity_sha256"]

    tampered = copy.deepcopy(authority)
    tampered["truth_boundary"]["training_authorized_bytes"] = 1
    core = dict(tampered)
    del core["execution_identity_sha256"]
    tampered["execution_identity_sha256"] = hashlib.sha256(
        (
            json.dumps(
                core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(
        QualityExecutionAuthorityError,
        match="truth boundary drift",
    ):
        verify_quality_execution_root(
            tampered,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
            expected_execution_identity_sha256=pinned,
        )
    with pytest.raises(
        QualityExecutionAuthorityError,
        match="canonical re-execution",
    ):
        verify_quality_execution_authority(
            tampered,
            records,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"id": "x", "mode": "en", "text": "valid", "extra": "forbidden"}],
        [
            {"id": "x", "mode": "en", "text": "one"},
            {"id": "x", "mode": "en", "text": "two"},
        ],
        [{"id": "x", "mode": "other", "text": "invalid mode"}],
    ],
)
def test_input_schema_is_closed_and_duplicate_ids_fail(
    records: list[dict[str, str]],
) -> None:
    with pytest.raises(QualityExecutionAuthorityError):
        build_quality_execution_authority(
            records,
            input_manifest_sha256=INPUT_MANIFEST_SHA256,
        )


def test_wrong_manifest_or_root_fails_closed() -> None:
    records = [_accepted_en()]
    authority = build_quality_execution_authority(
        records,
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
    )

    with pytest.raises(
        QualityExecutionAuthorityError,
        match="input manifest authority mismatch",
    ):
        verify_quality_execution_root(
            authority,
            expected_input_manifest_sha256="b" * 64,
            expected_execution_identity_sha256=authority[
                "execution_identity_sha256"
            ],
        )
    with pytest.raises(
        QualityExecutionAuthorityError,
        match="authority root mismatch",
    ):
        verify_quality_execution_root(
            authority,
            expected_input_manifest_sha256=INPUT_MANIFEST_SHA256,
            expected_execution_identity_sha256="c" * 64,
        )
