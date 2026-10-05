from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.data import trusted_family_authority_v1 as v1
from twelve_six.data import trusted_family_authority_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXTENSION_FAMILIES = {
    "github:pydantic/pydantic",
    "code.scipy.project",
    "github:pandas-dev/pandas",
    "github:fastapi/typer",
    "github:Textualize/rich",
    "github:fastapi/fastapi",
}


def test_v2_extension_is_exactly_six_code_families() -> None:
    assert set(v2.EXTENSION_AUTHORITY_SPECS) == EXPECTED_EXTENSION_FAMILIES
    assert (
        set(v2.TRUSTED_FAMILY_SEMANTICS)
        == set(v1.TRUSTED_FAMILY_SEMANTICS) | EXPECTED_EXTENSION_FAMILIES
    )
    for family in EXPECTED_EXTENSION_FAMILIES:
        row = v2.TRUSTED_FAMILY_SEMANTICS[family]
        assert row["family"] == family
        assert row["stratum"] == "code"
        assert row["language"] == "und"
        assert row["modalities"] == ["code"]
        assert len(row["source_family_identity_sha256"]) == 64


@pytest.mark.parametrize(
    "families",
    [
        [],
        ["github:encode/httpx"],
        ["github:encode/httpx", "github:pallets/flask"],
        sorted(v1.TRUSTED_FAMILY_SEMANTICS),
    ],
)
def test_v1_projection_and_root_are_byte_compatible(families: list[str]) -> None:
    assert v2.trusted_family_projection(families) == v1.trusted_family_projection(
        families
    )
    assert (
        v2.trusted_family_authority_root_sha256(families)
        == v1.trusted_family_authority_root_sha256(families)
    )


def test_extension_authority_blobs_are_exact_on_convergence_branch() -> None:
    observed = v2.verify_extension_authority_blobs(ROOT)
    assert observed == {
        family: spec["authority_git_blob_sha1"]
        for family, spec in sorted(v2.EXTENSION_AUTHORITY_SPECS.items())
    }


def test_mixed_projection_is_deterministic_and_preserves_old_rows() -> None:
    families = [
        "github:fastapi/fastapi",
        "github:encode/httpx",
        "github:pydantic/pydantic",
    ]
    projection = v2.trusted_family_projection(families)
    assert [row["family"] for row in projection] == sorted(families)
    assert (
        next(row for row in projection if row["family"] == "github:encode/httpx")
        == v1.TRUSTED_FAMILY_SEMANTICS["github:encode/httpx"]
    )
    first = v2.trusted_family_authority_root_sha256(families)
    second = v2.trusted_family_authority_root_sha256(reversed(families))
    assert first == second
    assert len(first) == 64


def test_unknown_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="absent from trusted V2 authority"):
        v2.trusted_family_projection(["github:attacker/unknown"])
    with pytest.raises(ValueError, match="absent from trusted V2 authority"):
        v2.trusted_family_authority_root_sha256(["github:attacker/unknown"])


def test_extension_blob_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "authority.json"
    path.write_text("{}\n", encoding="utf-8")
    family = "github:example/project"
    monkeypatch.setattr(
        v2,
        "EXTENSION_AUTHORITY_SPECS",
        {
            family: {
                "authority_path": "authority.json",
                "authority_git_blob_sha1": "0" * 40,
            }
        },
    )
    with pytest.raises(ValueError, match="authority blob drift"):
        v2.verify_extension_authority_blobs(tmp_path)


def test_extension_identity_binds_family_path_and_blob() -> None:
    family = "github:pydantic/pydantic"
    spec = v2.EXTENSION_AUTHORITY_SPECS[family]
    baseline = v2._extension_family_identity(family, spec)
    changed_blob = dict(spec)
    changed_blob["authority_git_blob_sha1"] = "0" * 40
    changed_path = dict(spec)
    changed_path["authority_path"] = "configs/data/other.json"
    assert baseline != v2._extension_family_identity(family, changed_blob)
    assert baseline != v2._extension_family_identity(family, changed_path)
    assert baseline != v2._extension_family_identity("github:other/project", spec)
