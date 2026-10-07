from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six import ubuntu_irc_execution_rights_crossbind as mod
from twelve_six.ubuntu_irc_rights import UbuntuIrcRightsError

REPO_ROOT = Path(__file__).parents[1]
CROSSBIND_PATH = (
    REPO_ROOT / "configs/data/d03_ubuntu_irc_repaired_execution_rights_crossbind_v1.json"
)
RIGHTS_PATH = REPO_ROOT / "configs/data/d03_common_pile_ubuntu_irc_source_rights_v1.json"
PARENT_PATH = REPO_ROOT / "configs/data/common_pile_source_rights_v1.json"


def load_inputs() -> tuple[dict, dict, dict, bytes]:
    crossbind, _ = mod.load_json(CROSSBIND_PATH)
    rights, rights_bytes = mod.load_json(RIGHTS_PATH)
    parent, _ = mod.load_json(PARENT_PATH)
    return crossbind, rights, parent, rights_bytes


def validate(crossbind: dict, rights: dict, parent: dict, rights_bytes: bytes) -> str:
    return mod.validate_crossbind(crossbind, rights, parent, rights_bytes)


def test_exact_repaired_execution_is_crossbound_zero_credit() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    assert validate(crossbind, rights, parent, rights_bytes) == mod.STATUS
    assert crossbind["repaired_execution_authority"]["product_head_sha"] == mod.REPAIRED_PRODUCT_HEAD
    assert crossbind["decision"]["downstream_global_dedup_candidate_input_allowed"] is True
    assert crossbind["decision"]["payload_source_admission_executed"] is False
    assert crossbind["decision"]["training_authorized_bytes"] == 0
    assert crossbind["decision"]["unique_causal_loss_positions_authorized"] == 0


def test_rights_blob_is_exact_current_merged_authority() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    assert mod.git_blob_sha1(rights_bytes) == mod.RIGHTS_BLOB_SHA1
    validate(crossbind, rights, parent, rights_bytes)
    drifted = rights_bytes + b"\n"
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="rights bytes drifted"):
        validate(crossbind, rights, parent, drifted)


def test_stale_predecessor_is_preserved_as_superseded_not_promoted() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    stale = crossbind["superseded_execution_reference"]
    assert stale["product_head_sha"] == mod.OLD_PRODUCT_HEAD
    assert stale["evidence_blob_sha1"] == mod.OLD_EVIDENCE_BLOB
    assert rights["real_execution_reference"]["product_head_sha"] == mod.OLD_PRODUCT_HEAD
    assert crossbind["repaired_execution_authority"]["product_head_sha"] != mod.OLD_PRODUCT_HEAD
    validate(crossbind, rights, parent, rights_bytes)


def test_predecessor_head_cannot_masquerade_as_repaired_authority() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    crossbind["repaired_execution_authority"]["product_head_sha"] = mod.OLD_PRODUCT_HEAD
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="repaired head drifted"):
        validate(crossbind, rights, parent, rights_bytes)


def test_predecessor_evidence_blob_cannot_masquerade_as_repaired_authority() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    crossbind["repaired_execution_authority"]["evidence_blob_sha1"] = mod.OLD_EVIDENCE_BLOB
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="repaired evidence blob drifted"):
        validate(crossbind, rights, parent, rights_bytes)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_revision", "0" * 40, "source revision drifted"),
        ("source_file", "other.jsonl.gz", "source file drifted"),
        ("source_sha256", "0" * 64, "source SHA-256 drifted"),
        ("candidate_payload_sha256", "0" * 64, "candidate SHA-256 drifted"),
    ],
)
def test_repaired_source_identity_drift_fails_closed(
    field: str, value: str, message: str
) -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    crossbind["repaired_execution_authority"][field] = value
    with pytest.raises(mod.UbuntuIrcCrossbindError, match=message):
        validate(crossbind, rights, parent, rights_bytes)


def test_rights_and_repaired_source_identity_must_cross_match() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    rights = copy.deepcopy(rights)
    rights["real_execution_reference"]["candidate_payload_sha256"] = "0" * 64
    with pytest.raises(UbuntuIrcRightsError, match="candidate payload hash drifted"):
        validate(crossbind, rights, parent, rights_bytes)


def test_bool_cannot_alias_integer_execution_count() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    crossbind["repaired_execution_authority"]["retained_records"] = True
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="retained records"):
        validate(crossbind, rights, parent, rights_bytes)


def test_ci_and_independent_audit_are_exact_authorities() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    mutated = copy.deepcopy(crossbind)
    mutated["repaired_execution_authority"]["independent_audit_issue"] = 1274
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="independent audit issue"):
        validate(mutated, rights, parent, rights_bytes)

    mutated = copy.deepcopy(crossbind)
    mutated["repaired_execution_authority"]["shared_ci_conclusion"] = "queued"
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="shared CI conclusion"):
        validate(mutated, rights, parent, rights_bytes)


def test_unknown_crossbind_field_fails_closed() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    crossbind["override"] = {"training": True}
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="crossbind keys drifted"):
        validate(crossbind, rights, parent, rights_bytes)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_training_authorized", True),
        ("canonical_capacity_credited", 1),
        ("family_credit_added", 1),
        ("training_authorized_bytes", 1),
        ("unique_causal_loss_positions_authorized", 1),
        ("tokenizer_fit_authorized", True),
        ("optimizer_updates_authorized", True),
        ("evaluation_eligible", True),
        ("final_test_accessed", True),
    ],
)
def test_scientific_authority_widening_fails_closed(field: str, value: object) -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    crossbind["decision"][field] = value
    with pytest.raises(mod.UbuntuIrcCrossbindError):
        validate(crossbind, rights, parent, rights_bytes)


def test_false_claim_of_payload_execution_fails_closed() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    crossbind["decision"]["payload_source_admission_executed"] = True
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="payload_source_admission_executed"):
        validate(crossbind, rights, parent, rights_bytes)


def test_truth_boundary_promotion_fails_closed() -> None:
    crossbind, rights, parent, rights_bytes = load_inputs()
    crossbind["truth_boundary"]["training_executed"] = True
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="training_executed"):
        validate(crossbind, rights, parent, rights_bytes)


def test_duplicate_json_keys_fail_closed() -> None:
    payload = b'{"schema_version":"x","schema_version":"y"}'
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="duplicate JSON key"):
        mod.load_json_bytes(payload, "duplicate")


def test_serialized_unknown_field_remains_rejected() -> None:
    crossbind, _, _, _ = load_inputs()
    crossbind["decision"]["secret_override"] = False
    payload = json.dumps(crossbind).encode("utf-8")
    parsed = mod.load_json_bytes(payload, "mutated crossbind")
    _, rights, parent, rights_bytes = load_inputs()
    with pytest.raises(mod.UbuntuIrcCrossbindError, match="decision keys drifted"):
        validate(parsed, rights, parent, rights_bytes)
