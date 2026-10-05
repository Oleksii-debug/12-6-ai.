from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data import trusted_family_authority_franko as parent
from twelve_six.data import trusted_family_authority_franko_ubuntu as successor


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_non_ubuntu_projection_and_root_remain_parent_exact() -> None:
    families = (
        parent.FRANKO_FAMILY,
        "ua.rada.open-data.laws-texts",
        "en.standardebooks.manual",
        "github:pydantic/pydantic",
        "github:fastapi/fastapi",
    )
    assert successor.trusted_family_projection(families) == (
        parent.trusted_family_projection(families)
    )
    assert successor.trusted_family_authority_root_sha256(families) == (
        parent.trusted_family_authority_root_sha256(families)
    )


def test_successor_adds_exactly_one_en_text_family() -> None:
    assert len(parent.TRUSTED_FAMILY_SEMANTICS) == 28
    assert len(successor.TRUSTED_FAMILY_SEMANTICS) == 29
    assert (
        set(successor.TRUSTED_FAMILY_SEMANTICS)
        - set(parent.TRUSTED_FAMILY_SEMANTICS)
    ) == {successor.UBUNTU_FAMILY}

    row = successor.TRUSTED_FAMILY_SEMANTICS[successor.UBUNTU_FAMILY]
    assert row == {
        "family": successor.UBUNTU_FAMILY,
        "source_family_identity_sha256": row[
            "source_family_identity_sha256"
        ],
        "language": "en",
        "modalities": ["text"],
        "stratum": "en",
    }


def test_ubuntu_identity_is_rights_authority_blob_bound() -> None:
    expected = hashlib.sha256(
        _canonical(
            {
                "authority_git_blob_sha1": successor.UBUNTU_AUTHORITY[
                    "blob_sha1"
                ],
                "source_family": successor.UBUNTU_FAMILY,
                "canonical_stratum": "en",
            }
        )
    ).hexdigest()
    assert successor.TRUSTED_FAMILY_SEMANTICS[successor.UBUNTU_FAMILY][
        "source_family_identity_sha256"
    ] == expected


def test_ubuntu_only_projection_is_source_authority_authenticated() -> None:
    projection = successor.trusted_family_projection([successor.UBUNTU_FAMILY])
    assert projection == [
        copy.deepcopy(
            successor.TRUSTED_FAMILY_SEMANTICS[successor.UBUNTU_FAMILY]
        )
    ]
    root = successor.trusted_family_authority_root_sha256(
        [successor.UBUNTU_FAMILY]
    )
    assert len(root) == 64
    assert set(root) <= set("0123456789abcdef")


def test_mixed_parent_franko_and_ubuntu_projection_is_sorted() -> None:
    parent_families = {
        parent.FRANKO_FAMILY,
        "ua.verba.public-domain.nomis1864",
        "github:pydantic/pydantic",
        "en.standardebooks.manual",
    }
    requested = parent_families | {successor.UBUNTU_FAMILY}
    projection = successor.trusted_family_projection(requested)

    assert [row["family"] for row in projection] == sorted(requested)
    parent_projection = {
        row["family"]: row
        for row in parent.trusted_family_projection(parent_families)
    }
    for row in projection:
        if row["family"] in parent_projection:
            assert row == parent_projection[row["family"]]


def test_ubuntu_source_authority_blob_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered = copy.deepcopy(successor.UBUNTU_AUTHORITY)
    altered["blob_sha1"] = "0" * 40
    monkeypatch.setattr(successor, "UBUNTU_AUTHORITY", altered)
    with pytest.raises(ValueError, match="Git blob drift"):
        successor.trusted_family_projection([successor.UBUNTU_FAMILY])


def test_ubuntu_family_declaration_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered = copy.deepcopy(successor.UBUNTU_AUTHORITY)
    altered["family_field"] = "schema_version"
    monkeypatch.setattr(successor, "UBUNTU_AUTHORITY", altered)
    with pytest.raises(ValueError, match="family declaration drift"):
        successor.trusted_family_projection([successor.UBUNTU_FAMILY])


def test_parent_franko_module_blob_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        successor,
        "PARENT_FRANKO_MODULE_BLOB_SHA1",
        "0" * 40,
    )
    with pytest.raises(ValueError, match="Git blob drift"):
        successor.trusted_family_projection([successor.UBUNTU_FAMILY])


def test_unknown_family_still_fails_closed_with_ubuntu() -> None:
    with pytest.raises(ValueError, match="absent from trusted Ubuntu authority"):
        successor.trusted_family_projection(
            [successor.UBUNTU_FAMILY, "en.example.not-authorized"]
        )


def test_authority_chain_is_explicitly_zero_credit_and_pre_post_qp() -> None:
    truth = successor.UBUNTU_AUTHORITY_CHAIN["truth_boundary"]
    assert truth == {
        "ubuntu_post_qp_complete": False,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    assert successor.UBUNTU_AUTHORITY_CHAIN["family_count_total"] == 29
