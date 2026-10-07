from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data import privacy_execution_authority as g06


def _records() -> list[dict[str, str]]:
    return [
        {"id": "r-en-allow", "text": "A short public documentation line.", "mode": "en"},
        {
            "id": "r-en-email",
            "text": "Reach Alice at alice@real-domain.dev for the fixture.",
            "mode": "en",
        },
        {
            "id": "r-code-secret",
            "text": "token=AbCd1234!fixture",
            "mode": "code",
        },
    ]


def _inventory(records: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        payload = record["text"].encode("utf-8")
        rows.append(
            {
                "record_id": record["id"],
                "source_id": f"source:{record['id']}",
                "family": "fixture",
                "modality": record["mode"],
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_bytes": len(payload),
            }
        )
    return list(reversed(rows))


def _authority() -> tuple[list[dict[str, str]], str, dict[str, object]]:
    records = _records()
    expected_root = g06.input_rows_sha256_from_text_free_inventory(
        _inventory(records)
    )
    authority = g06.build_privacy_execution_authority(
        records,
        expected_input_rows_sha256=expected_root,
    )
    return records, expected_root, authority


def _resign(authority: dict[str, object]) -> None:
    authority["execution_rows_sha256"] = hashlib.sha256(
        g06._cjson(authority["records"])
    ).hexdigest()
    core = dict(authority)
    core.pop("execution_identity_sha256", None)
    authority["execution_identity_sha256"] = hashlib.sha256(
        g06._cjson(core)
    ).hexdigest()


def test_text_free_inventory_bridge_matches_raw_payload_root() -> None:
    records = _records()
    assert g06.input_rows_sha256_from_text_free_inventory(
        _inventory(records)
    ) == g06.input_rows_sha256(records)


def test_build_and_verify_are_deterministic_and_text_free() -> None:
    records, expected_root, authority = _authority()
    repeated = g06.build_privacy_execution_authority(
        list(reversed(records)),
        expected_input_rows_sha256=expected_root,
    )
    assert authority == repeated
    identity = authority["execution_identity_sha256"]
    assert g06.verify_privacy_execution_root(
        authority,
        expected_input_rows_sha256=expected_root,
        expected_execution_identity_sha256=identity,
    ) == identity
    assert g06.verify_privacy_execution_authority(
        authority,
        records,
        expected_input_rows_sha256=expected_root,
        expected_execution_identity_sha256=identity,
    ) == identity

    serialized = json.dumps(authority, sort_keys=True)
    for record in records:
        assert record["text"] not in serialized
    assert authority["truth_boundary"]["source_text_retained_in_authority"] is False
    assert authority["truth_boundary"]["matched_values_retained_in_authority"] is False


def test_raw_payload_or_mode_substitution_cannot_match_external_root() -> None:
    records = _records()
    expected_root = g06.input_rows_sha256_from_text_free_inventory(
        _inventory(records)
    )
    substituted = copy.deepcopy(records)
    substituted[0]["text"] += " changed"
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="expected input-row root"):
        g06.build_privacy_execution_authority(
            substituted,
            expected_input_rows_sha256=expected_root,
        )

    substituted = copy.deepcopy(records)
    substituted[0]["mode"] = "uk"
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="expected input-row root"):
        g06.build_privacy_execution_authority(
            substituted,
            expected_input_rows_sha256=expected_root,
        )


def test_inventory_bridge_rejects_duplicates_schema_and_bool_aliases() -> None:
    rows = _inventory(_records())
    duplicate = copy.deepcopy(rows)
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="duplicate inventory"):
        g06.input_rows_sha256_from_text_free_inventory(duplicate)

    malformed = copy.deepcopy(rows)
    malformed[0]["unexpected"] = "x"
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="schema drift"):
        g06.input_rows_sha256_from_text_free_inventory(malformed)

    bool_alias = copy.deepcopy(rows)
    bool_alias[0]["payload_bytes"] = True
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="must be an integer"):
        g06.input_rows_sha256_from_text_free_inventory(bool_alias)


def test_root_verifier_rejects_unknown_fields_and_bool_aliases() -> None:
    _, expected_root, authority = _authority()
    identity = authority["execution_identity_sha256"]

    unknown = copy.deepcopy(authority)
    unknown["unexpected"] = False
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="schema is not closed"):
        g06.verify_privacy_execution_root(
            unknown,
            expected_input_rows_sha256=expected_root,
            expected_execution_identity_sha256=identity,
        )

    bool_alias = copy.deepcopy(authority)
    bool_alias["records"][0]["utf8_bytes"] = True
    _resign(bool_alias)
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="must be an integer"):
        g06.verify_privacy_execution_root(
            bool_alias,
            expected_input_rows_sha256=expected_root,
            expected_execution_identity_sha256=identity,
        )


def test_self_consistent_action_substitution_still_fails_external_identity() -> None:
    _, expected_root, authority = _authority()
    identity = authority["execution_identity_sha256"]
    tampered = copy.deepcopy(authority)
    row = tampered["records"][0]
    old_action = row["action"]
    new_action = next(action for action in sorted(g06._ACTIONS) if action != old_action)
    row["action"] = new_action
    result_core = {
        "input_sha256": row["payload_sha256"],
        "input_bytes": row["utf8_bytes"],
        "action": row["action"],
        "detector_counts": row["detector_counts"],
    }
    row["privacy_result_sha256"] = hashlib.sha256(
        g06._cjson(result_core)
    ).hexdigest()
    tampered["counts"][old_action.lower()] -= 1
    tampered["counts"][new_action.lower()] += 1
    _resign(tampered)

    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="authority root mismatch"):
        g06.verify_privacy_execution_root(
            tampered,
            expected_input_rows_sha256=expected_root,
            expected_execution_identity_sha256=identity,
        )


def test_reexecution_rejects_self_consistent_action_substitution() -> None:
    records, expected_root, authority = _authority()
    tampered = copy.deepcopy(authority)
    row = tampered["records"][0]
    old_action = row["action"]
    new_action = next(action for action in sorted(g06._ACTIONS) if action != old_action)
    row["action"] = new_action
    result_core = {
        "input_sha256": row["payload_sha256"],
        "input_bytes": row["utf8_bytes"],
        "action": row["action"],
        "detector_counts": row["detector_counts"],
    }
    row["privacy_result_sha256"] = hashlib.sha256(
        g06._cjson(result_core)
    ).hexdigest()
    tampered["counts"][old_action.lower()] -= 1
    tampered["counts"][new_action.lower()] += 1
    _resign(tampered)

    with pytest.raises(
        g06.PrivacyExecutionAuthorityError,
        match="canonical re-execution",
    ):
        g06.verify_privacy_execution_authority(
            tampered,
            records,
            expected_input_rows_sha256=expected_root,
        )


def test_exact_privacy_implementation_blob_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _records()
    expected_root = g06.input_rows_sha256_from_text_free_inventory(
        _inventory(records)
    )
    monkeypatch.setattr(
        g06,
        "EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA1",
        "0" * 40,
    )
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="Git blob drift"):
        g06.build_privacy_execution_authority(
            records,
            expected_input_rows_sha256=expected_root,
        )


def test_callable_provenance_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = __import__(g06.PRIVACY_MODULE, fromlist=["hash_safe_scan"])
    original = module.hash_safe_scan

    def wrapper(text: str):
        return original(text)

    monkeypatch.setattr(module, "hash_safe_scan", wrapper)
    records = _records()
    expected_root = g06.input_rows_sha256_from_text_free_inventory(
        _inventory(records)
    )
    with pytest.raises(g06.PrivacyExecutionAuthorityError, match="callable provenance"):
        g06.build_privacy_execution_authority(
            records,
            expected_input_rows_sha256=expected_root,
        )
