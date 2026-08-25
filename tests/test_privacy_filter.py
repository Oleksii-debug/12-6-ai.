from __future__ import annotations

import hashlib
import json

import pytest

from twelve_six.data.corpus_foundation import PolicyHookEvidence, RecordPolicyMetadata
from twelve_six.data.privacy_filter import (
    COVERAGE_CLAIM,
    PrivacyFilterError,
    assert_no_secret_values_in_manifest,
    build_scan_manifest,
    detect_privacy_findings,
    privacy_policy_manifest,
    scan_record,
)

H = "a" * 64


def _pass_hook(hook_id: str) -> PolicyHookEvidence:
    return PolicyHookEvidence(hook_id, "PASS", "v1", "tool@1", "2026-08-25T00:00:00Z", H)


def test_contact_pii_is_redacted_and_privacy_hook_passes() -> None:
    result = scan_record(
        record_id="n1",
        source_id="project/s0",
        source_version="v1",
        modality="natural",
        text="Contact Alice at alice@example.com or +421 905 123 456 for the dataset review.",
    )
    assert result.action == "REDACT"
    assert result.status == "PASS"
    assert result.detector_counts == {"email": 1, "phone": 1}
    assert "alice@example.com" not in result.sanitized_text
    assert "+421 905 123 456" not in result.sanitized_text

    metadata = RecordPolicyMetadata(
        quality=_pass_hook("quality"),
        language=_pass_hook("language"),
        pii=result.policy_hook_evidence(executed_at="2026-08-25T14:31:00Z"),
        copyright=_pass_hook("copyright"),
    )
    metadata.assert_passed()


def test_private_keys_and_vendor_tokens_are_excluded_without_text_output() -> None:
    key_doc = scan_record(
        record_id="c1",
        source_id="licensed-code/repo",
        source_version="abc123",
        modality="code",
        text="-----BEGIN " + "PRIVATE KEY-----\nFAKEFIXTUREONLY\n-----END PRIVATE KEY-----",
    )
    assert key_doc.action == "EXCLUDE"
    assert key_doc.status == "REJECT"
    assert key_doc.sanitized_text is None
    assert key_doc.output_sha256 is None
    assert key_doc.detector_counts["private_key"] == 1

    token = "ghp_" + "A" * 36
    token_doc = scan_record(
        record_id="c2",
        source_id="licensed-code/repo",
        source_version="abc123",
        modality="code",
        text=f'auth = "{token}"',
    )
    assert token_doc.status == "REJECT"
    assert token_doc.detector_counts == {"github_token": 1}


def test_generic_password_assignment_is_quarantined_but_placeholder_is_not() -> None:
    suspicious = scan_record(
        record_id="c3",
        source_id="licensed-code/repo",
        source_version="abc123",
        modality="code",
        text='pass' + 'word = "' + 'M0re' + 'ThanTen!"',
    )
    assert suspicious.action == "QUARANTINE"
    assert suspicious.status == "REVIEW_REQUIRED"
    assert suspicious.sanitized_text is None

    placeholder = scan_record(
        record_id="c4",
        source_id="licensed-code/repo",
        source_version="abc123",
        modality="code",
        text='password = "changeme"',
    )
    assert placeholder.action == "ALLOW"
    assert placeholder.status == "PASS"


def test_structured_identifiers_use_validators_to_limit_false_positives() -> None:
    valid = scan_record(
        record_id="n2",
        source_id="fixture",
        source_version="v1",
        modality="natural",
        text="Sensitive fixtures: 4111 1111 1111 1111 and GB82 WEST 1234 5698 7654 32.",
    )
    assert valid.action == "REDACT"
    assert valid.detector_counts["payment_card_luhn"] == 1
    assert valid.detector_counts["iban"] == 1
    assert "4111 1111 1111 1111" not in valid.sanitized_text
    assert "GB82 WEST 1234 5698 7654 32" not in valid.sanitized_text

    invalid = detect_privacy_findings("Build number 1234-56-78 and card-like 1234 5678 9012 3456.")
    assert not invalid


def test_scan_manifest_contains_counts_hashes_and_no_matched_values() -> None:
    secret = "ghp_" + "B" * 36
    rows = [
        scan_record(
            record_id="n1",
            source_id="s",
            source_version="1",
            modality="natural",
            text="Write to bob@example.org for details.",
        ),
        scan_record(
            record_id="c1",
            source_id="s",
            source_version="1",
            modality="code",
            text=f'token = "{secret}"',
        ),
    ]
    manifest = build_scan_manifest(
        rows,
        input_content_sha256=hashlib.sha256(b"fixture").hexdigest(),
        source_registry_sha256=hashlib.sha256(b"registry").hexdigest(),
    )
    assert manifest["records_total"] == 2
    assert manifest["detector_counts"]["email"] == 1
    assert manifest["detector_counts"]["github_token"] == 1
    assert manifest["detector_counts"]["private_key"] == 0
    assert manifest["by_modality"]["natural"]["detector:email"] == 1
    assert manifest["by_modality"]["code"]["detector:github_token"] == 1
    serialized = json.dumps(manifest)
    assert "bob@example.org" not in serialized
    assert secret not in serialized
    assert COVERAGE_CLAIM in serialized
    assert_no_secret_values_in_manifest(manifest)


@pytest.mark.parametrize(
    ("detector", "payload"),
    [
        ("aws_access_key_id", "AK" + "IA" + "A" * 16),
        ("aws_secret_access_key", "aws_secret_access_key=" + "A1b2/" * 8),
        ("slack_token", "xox" + "b-" + "Ab12-" * 6),
        ("stripe_live_secret", "sk_" + "live_" + "Ab12" * 5),
        ("google_api_key", "AI" + "za" + "A" * 35),
        ("azure_storage_account_key", "AccountKey=" + "QUJD" * 12),
    ],
)
def test_cloud_and_vendor_credentials_are_high_confidence_exclusions(
    detector: str, payload: str
) -> None:
    result = scan_record(
        record_id="credential-fixture",
        source_id="fixture",
        source_version="v1",
        modality="code",
        text=payload,
    )
    assert result.action == "EXCLUDE"
    assert result.status == "REJECT"
    assert result.detector_counts[detector] >= 1
    assert result.sanitized_text is None


def test_ssn_validator_redacts_valid_shape_but_rejects_invalid_areas() -> None:
    result = scan_record(
        record_id="ssn-fixture",
        source_id="fixture",
        source_version="v1",
        modality="natural",
        text="Identifiers 123-45-6789 and invalid 000-12-3456.",
    )
    assert result.detector_counts["us_ssn"] == 1
    assert "123-45-6789" not in result.sanitized_text
    assert "000-12-3456" in result.sanitized_text


def test_privacy_policy_identity_changes_with_policy_payload() -> None:
    policy = privacy_policy_manifest()
    assert len(policy["policy_sha256"]) == 64
    assert policy["detector_actions"]["private_key"] == "EXCLUDE"
    assert policy["detector_actions"]["generic_password_assignment"] == "QUARANTINE"


def test_manifest_structure_guard_rejects_future_value_bearing_fields() -> None:
    with pytest.raises(PrivacyFilterError, match="forbidden"):
        assert_no_secret_values_in_manifest({"records": [{"match_value": "do-not-log"}]})


def test_bearer_jwt_and_generic_secret_assignment_policy() -> None:
    jwt = "eyJ" + "A" * 12 + "." + "B" * 12 + "." + "C" * 12
    excluded = scan_record(
        record_id="jwt-fixture",
        source_id="fixture",
        source_version="v1",
        modality="code",
        text="Authorization: Bearer " + jwt,
    )
    assert excluded.action == "EXCLUDE"
    assert excluded.detector_counts["bearer_jwt"] == 1

    quarantined = scan_record(
        record_id="generic-secret-fixture",
        source_id="fixture",
        source_version="v1",
        modality="code",
        text='client_' + 'secret = "' + 'AbCd' + '1234!fixture"',
    )
    assert quarantined.action == "QUARANTINE"
    assert quarantined.detector_counts["generic_secret_assignment"] == 1


def test_retained_policy_matches_runtime_policy() -> None:
    from pathlib import Path

    retained = json.loads(
        Path("configs/data/pii_secrets_policy_v1.json").read_text(encoding="utf-8")
    )
    assert retained == privacy_policy_manifest()


def test_current_s0_privacy_report_is_reproducible_and_manifest_bound() -> None:
    from pathlib import Path

    raw_path = Path("data/s0/raw/project_authored.jsonl")
    registry_path = Path("data/s0/source_registry.json")
    retained_path = Path("reports/d03/pii_secrets_scan_s0_project_authored_20260825.json")

    source_registry_sha256 = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    input_content_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    results = []
    for raw_line in raw_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        results.append(
            scan_record(
                record_id=row["document_id"],
                source_id="project-authored-s0-fixture-v1",
                source_version="v1",
                modality="natural",
                text=row["text"],
            )
        )

    actual = build_scan_manifest(
        results,
        input_content_sha256=input_content_sha256,
        source_registry_sha256=source_registry_sha256,
    )
    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    assert_no_secret_values_in_manifest(actual)
    assert actual == retained
