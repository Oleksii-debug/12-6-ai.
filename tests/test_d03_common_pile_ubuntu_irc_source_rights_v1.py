from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.ubuntu_irc_rights import (
    STATUS,
    UbuntuIrcRightsError,
    validate_authority,
    validate_candidate_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_PATH = ROOT / "configs/data/d03_common_pile_ubuntu_irc_source_rights_v1.json"
PARENT_PATH = ROOT / "configs/data/common_pile_source_rights_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid() -> tuple[dict, dict]:
    return _load(AUTHORITY_PATH), _load(PARENT_PATH)


def test_exact_authority_validates_and_stays_zero_credit() -> None:
    authority, parent = _valid()
    assert validate_authority(authority, parent) == STATUS
    assert authority["decision"]["source_scope_qualified"] is True
    assert authority["decision"]["canonical_training_authorized"] is False
    assert authority["decision"]["corpus_credit_bytes"] == 0
    assert authority["decision"]["authorized_loss_positions"] == 0
    assert authority["truth_boundary"]["training_executed"] is False
    assert authority["truth_boundary"]["learned_weights_created"] is False


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("source_scope", "required_source_label", "irc-chat"),
        ("source_scope", "required_metadata_license", "CC-BY"),
        ("source_scope", "required_metadata_url_prefix", "https://example.invalid/"),
        ("source_scope", "non_ubuntu_irc_allowed", True),
        ("source_scope", "website_wide_license_inference_allowed", True),
        ("source_scope", "dataset_metadata_alone_is_sufficient", True),
        ("decision", "legal_conclusion_claimed", True),
        ("decision", "general_irc_public_domain_claimed", True),
        ("decision", "canonical_training_authorized", True),
        ("decision", "corpus_credit_bytes", 1),
        ("decision", "authorized_loss_positions", 1),
        ("decision", "tokenizer_fit_authorized", True),
        ("decision", "evaluation_eligible", True),
        ("decision", "final_test_accessed", True),
        ("truth_boundary", "real_data_executed_by_this_package", True),
        ("truth_boundary", "training_executed", True),
        ("truth_boundary", "learned_weights_created", True),
        ("truth_boundary", "paid_compute_used", True),
        ("truth_boundary", "foreign_pretrained_weights_used", True),
    ],
)
def test_truth_widening_fails_closed(section: str, field: str, replacement: object) -> None:
    authority, parent = _valid()
    authority[section][field] = replacement
    with pytest.raises(UbuntuIrcRightsError):
        validate_authority(authority, parent)


def test_bool_int_aliases_fail_closed() -> None:
    authority, parent = _valid()
    authority["decision"]["corpus_credit_bytes"] = False
    with pytest.raises(UbuntuIrcRightsError):
        validate_authority(authority, parent)

    authority, parent = _valid()
    authority["project_authority"]["worker_issue"] = True
    with pytest.raises(UbuntuIrcRightsError):
        validate_authority(authority, parent)


def test_parent_registry_byte_or_source_drift_fails_closed() -> None:
    authority, parent = _valid()
    parent["audit_date"] = "2099-01-01"
    with pytest.raises(UbuntuIrcRightsError, match="parent registry bytes drifted"):
        validate_authority(authority, parent)

    authority, parent = _valid()
    ubuntu = next(row for row in parent["sources"] if row["key"] == "ubuntu_irc")
    ubuntu["credited_bytes"] = 1
    with pytest.raises(UbuntuIrcRightsError):
        validate_authority(authority, parent)


def test_upstream_and_policy_anchor_substitution_fails_closed() -> None:
    authority, parent = _valid()
    authority["upstream_collector"]["audited_commit"] = "0" * 40
    with pytest.raises(UbuntuIrcRightsError):
        validate_authority(authority, parent)

    authority, parent = _valid()
    authority["ubuntu_policy_evidence"]["irc_guidelines"]["url"] = "https://example.invalid/"
    with pytest.raises(UbuntuIrcRightsError):
        validate_authority(authority, parent)


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        (
            "irc_guidelines",
            "All IRC networks are public domain without restriction.",
        ),
        (
            "community_help_irc",
            "This policy applies to every IRC network and website.",
        ),
        (
            "irc_terms_of_service",
            "The Ubuntu IRC Terms expressly waive all copyrights and apply to all IRC networks.",
        ),
    ],
)
def test_policy_observed_fact_semantic_substitution_fails_closed(
    key: str,
    replacement: str,
) -> None:
    authority, parent = _valid()
    authority["ubuntu_policy_evidence"][key]["observed_fact"] = replacement
    with pytest.raises(UbuntuIrcRightsError, match=f"{key} observed fact drifted"):
        validate_authority(authority, parent)


def test_execution_reference_is_separate_and_exact_bound() -> None:
    authority, parent = _valid()
    authority["real_execution_reference"]["role"] = "TRAINING_AUTHORITY"
    with pytest.raises(UbuntuIrcRightsError):
        validate_authority(authority, parent)

    authority, parent = _valid()
    authority["real_execution_reference"]["source_sha256"] = "0" * 64
    with pytest.raises(UbuntuIrcRightsError):
        validate_authority(authority, parent)


def test_extra_fields_fail_closed() -> None:
    authority, parent = _valid()
    authority["decision"]["magic_training_override"] = True
    with pytest.raises(UbuntuIrcRightsError, match="keys drifted"):
        validate_authority(authority, parent)


def test_candidate_metadata_accepts_only_exact_ubuntu_envelope() -> None:
    validate_candidate_metadata(
        "ubuntu-chat",
        {
            "license": "Public Domain",
            "url": "https://irclogs.ubuntu.com/2024/01/01/%23ubuntu.txt",
            "channel": "#ubuntu",
        },
    )


@pytest.mark.parametrize(
    ("source", "metadata"),
    [
        (
            "other-chat",
            {
                "license": "Public Domain",
                "url": "https://irclogs.ubuntu.com/2024/01/01/%23ubuntu.txt",
                "channel": "#ubuntu",
            },
        ),
        (
            "ubuntu-chat",
            {
                "license": "CC-BY",
                "url": "https://irclogs.ubuntu.com/2024/01/01/%23ubuntu.txt",
                "channel": "#ubuntu",
            },
        ),
        (
            "ubuntu-chat",
            {
                "license": "Public Domain",
                "url": "https://example.invalid/log",
                "channel": "#ubuntu",
            },
        ),
        (
            "ubuntu-chat",
            {
                "license": "Public Domain",
                "url": "https://irclogs.ubuntu.com/2024/01/01/%23ubuntu.txt",
                "channel": "ubuntu",
            },
        ),
    ],
)
def test_candidate_metadata_substitutions_fail_closed(source: object, metadata: object) -> None:
    with pytest.raises(UbuntuIrcRightsError):
        validate_candidate_metadata(source, metadata)


def test_authority_fixture_is_not_mutated_by_validation() -> None:
    authority, parent = _valid()
    before_authority = copy.deepcopy(authority)
    before_parent = copy.deepcopy(parent)
    validate_authority(authority, parent)
    assert authority == before_authority
    assert parent == before_parent
