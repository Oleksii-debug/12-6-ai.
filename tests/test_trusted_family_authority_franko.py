from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data import trusted_family_authority_current as parent
from twelve_six.data import trusted_family_authority_franko as franko


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_non_franko_projection_and_root_remain_parent_exact() -> None:
    families = (
        "ua.rada.open-data.laws-texts",
        "ua.verba.public-domain.nomis1864",
        "en.standardebooks.manual",
        "github:pydantic/pydantic",
        "github:fastapi/fastapi",
    )
    assert franko.trusted_family_projection(families) == (
        parent.trusted_family_projection(families)
    )
    assert franko.trusted_family_authority_root_sha256(families) == (
        parent.trusted_family_authority_root_sha256(families)
    )


def test_successor_adds_exactly_one_uk_text_family() -> None:
    assert len(parent.TRUSTED_FAMILY_SEMANTICS) == 27
    assert len(franko.TRUSTED_FAMILY_SEMANTICS) == 28
    assert (
        set(franko.TRUSTED_FAMILY_SEMANTICS)
        - set(parent.TRUSTED_FAMILY_SEMANTICS)
    ) == {franko.FRANKO_FAMILY}

    row = franko.TRUSTED_FAMILY_SEMANTICS[franko.FRANKO_FAMILY]
    assert row == {
        "family": franko.FRANKO_FAMILY,
        "source_family_identity_sha256": row[
            "source_family_identity_sha256"
        ],
        "language": "uk",
        "modalities": ["text"],
        "stratum": "uk",
    }


def test_franko_identity_is_terminal_authority_blob_bound() -> None:
    expected = hashlib.sha256(
        _canonical(
            {
                "authority_git_blob_sha1": franko.FRANKO_AUTHORITY["blob_sha1"],
                "source_family": franko.FRANKO_FAMILY,
                "canonical_stratum": "uk",
            }
        )
    ).hexdigest()
    assert franko.TRUSTED_FAMILY_SEMANTICS[franko.FRANKO_FAMILY][
        "source_family_identity_sha256"
    ] == expected


def test_franko_only_projection_is_physically_authorized() -> None:
    projection = franko.trusted_family_projection([franko.FRANKO_FAMILY])
    assert projection == [
        copy.deepcopy(franko.TRUSTED_FAMILY_SEMANTICS[franko.FRANKO_FAMILY])
    ]
    root = franko.trusted_family_authority_root_sha256([franko.FRANKO_FAMILY])
    assert len(root) == 64
    assert set(root) <= set("0123456789abcdef")


def test_mixed_parent_and_franko_projection_is_sorted_and_preserved() -> None:
    parent_families = {
        "ua.verba.public-domain.nomis1864",
        "github:pydantic/pydantic",
        "en.standardebooks.manual",
    }
    requested = parent_families | {franko.FRANKO_FAMILY}
    projection = franko.trusted_family_projection(requested)

    assert [row["family"] for row in projection] == sorted(requested)
    parent_projection = {
        row["family"]: row
        for row in parent.trusted_family_projection(parent_families)
    }
    for row in projection:
        if row["family"] in parent_projection:
            assert row == parent_projection[row["family"]]


def test_franko_source_authority_blob_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered = copy.deepcopy(franko.FRANKO_AUTHORITY)
    altered["blob_sha1"] = "0" * 40
    monkeypatch.setattr(franko, "FRANKO_AUTHORITY", altered)
    with pytest.raises(ValueError, match="Git blob drift"):
        franko.trusted_family_projection([franko.FRANKO_FAMILY])


def test_franko_family_declaration_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered = copy.deepcopy(franko.FRANKO_AUTHORITY)
    altered["family_field"] = "schema_version"
    monkeypatch.setattr(franko, "FRANKO_AUTHORITY", altered)
    with pytest.raises(ValueError, match="family declaration drift"):
        franko.trusted_family_projection([franko.FRANKO_FAMILY])


def test_parent_current_module_blob_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(franko, "PARENT_CURRENT_MODULE_BLOB_SHA1", "0" * 40)
    with pytest.raises(ValueError, match="Git blob drift"):
        franko.trusted_family_projection([franko.FRANKO_FAMILY])


def test_unknown_family_still_fails_closed_with_franko() -> None:
    with pytest.raises(ValueError, match="absent from trusted Franko authority"):
        franko.trusted_family_projection(
            [franko.FRANKO_FAMILY, "ua.example.not-authorized"]
        )


def test_authority_chain_remains_zero_credit() -> None:
    truth = franko.FRANKO_AUTHORITY_CHAIN["truth_boundary"]
    assert truth == {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
    }
    assert franko.FRANKO_AUTHORITY_CHAIN["family_count_total"] == 28
