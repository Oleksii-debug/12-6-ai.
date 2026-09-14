from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType

import pytest

from twelve_six.data import attrs_dedup_intake as intake


def _authority(path: str, payload: bytes) -> intake.FileAuthority:
    return intake.FileAuthority(
        path=path,
        git_blob_sha1=intake._git_blob_sha1(payload),
        size_bytes=len(payload),
    )


def _tiny_fixture() -> tuple[dict[str, bytes], tuple[intake.FileAuthority, ...]]:
    first = b"def first():\n    return 'alpha'\n"
    second = b"def second():\n    return 'beta'\n"
    authorities = (
        _authority("src/attr/a.py", first),
        _authority("src/attr/b.py", second),
    )
    return {
        "src/attr/b.py": second,
        "src/attr/a.py": first,
    }, authorities


def test_live_authority_constants_lock_terminal_pr474() -> None:
    assert intake.UPSTREAM_PRODUCT_PR == 474
    assert intake.UPSTREAM_PRODUCT_HEAD == "cda0232d5574ef91eae0d7e0b7fa5efddcbe218b"
    assert intake.TERMINAL_AUTHORITY_SHA256 == (
        "151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4"
    )
    assert intake.UPSTREAM_COMMIT == "7bfc49e9b22d5ba25b6e429524c3d49fee27cb36"
    assert intake.UPSTREAM_TREE == "31beb3550ee7198eba22b862471ad6ea7bfb16d2"
    assert intake.UPSTREAM_WORKFLOW_RUN_ID == 33006080831
    assert intake.UPSTREAM_ARTIFACT_ID == 9621650719
    assert intake.ADMITTED_OBJECT_COUNT == 4
    assert len(intake.FILE_AUTHORITIES) == 4
    assert sum(item.size_bytes for item in intake.FILE_AUTHORITIES) == 170435
    assert intake.ADMITTED_SOURCE_BYTES == 170435
    assert tuple(item.path for item in intake.FILE_AUTHORITIES) == (
        "src/attr/_make.py",
        "src/attr/_funcs.py",
        "src/attr/validators.py",
        "src/attr/_next_gen.py",
    )


def test_tiny_projection_is_deterministic_and_v3_compatible() -> None:
    payloads, authorities = _tiny_fixture()
    projection = intake._project_payloads_for_authority(payloads, authorities)

    assert [row["source_id"] for row in projection.sources] == [
        "attrs-26.1.0:src/attr/a.py",
        "attrs-26.1.0:src/attr/b.py",
    ]
    for authority, row in zip(authorities, projection.sources, strict=True):
        payload = payloads[authority.path]
        payload_sha = intake._sha256(payload)
        assert row["source_family"] == intake.SOURCE_FAMILY
        assert row["modality"] == "code"
        assert row["evidence_status"] == "DEDICATED_TERMINAL"
        assert row["authority_ref"] == (
            f"PR474:{intake.UPSTREAM_PRODUCT_HEAD}:{intake.TERMINAL_AUTHORITY_SHA256}"
        )
        assert row["stable_origin_id"] == (
            f"{intake.SOURCE_FAMILY}@{intake.UPSTREAM_COMMIT}:{authority.path}"
        )
        assert row["stable_object_id"] == f"sha256:{payload_sha}"
        assert row["declared_capacity_bytes"] == len(payload)
        assert row["expected_raw_bytes"] == len(payload)
        assert row["expected_raw_sha256"] == payload_sha
        assert row["origin_key"] == f"{intake.SOURCE_FAMILY}:{authority.path}"
        assert row["acquisition_url"].endswith(
            f"/{intake.UPSTREAM_COMMIT}/{authority.path}"
        )
        assert projection.payloads[row["source_id"]] == payload

    assert projection.receipt["projection"]["object_count"] == 2
    assert projection.receipt["projection"]["payload_bytes"] == sum(
        len(value) for value in payloads.values()
    )
    assert projection.receipt["projection"]["matching_science_owned_here"] is False
    assert projection.receipt["projection"]["base_inventory_composition_owned_here"] is False


def test_projection_receipt_self_hash_and_truth_boundary_are_fail_closed() -> None:
    payloads, authorities = _tiny_fixture()
    projection = intake._project_payloads_for_authority(payloads, authorities)
    receipt = copy.deepcopy(projection.receipt)
    identity = receipt.pop("receipt_identity_sha256")
    assert intake._sha256(intake._canonical(receipt)) == identity

    truth = projection.receipt["truth_boundary"]
    assert truth == {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "external_llm_api_data_or_intelligence": False,
    }
    assert set(projection.receipt["downstream_gates"].values()) == {"NOT_RUN"}


def test_projection_order_comes_from_authority_not_mapping_insertion() -> None:
    payloads, authorities = _tiny_fixture()
    forward = intake._project_payloads_for_authority(payloads, authorities)
    reversed_mapping = dict(reversed(tuple(payloads.items())))
    reverse = intake._project_payloads_for_authority(reversed_mapping, authorities)
    assert intake._canonical(forward.sources) == intake._canonical(reverse.sources)
    assert intake._canonical(forward.receipt) == intake._canonical(reverse.receipt)


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_projection_rejects_path_set_drift(mode: str) -> None:
    payloads, authorities = _tiny_fixture()
    if mode == "missing":
        payloads.pop(authorities[0].path)
    else:
        payloads["src/attr/extra.py"] = b"extra\n"
    with pytest.raises(intake.AttrsDedupIntakeError, match="payload path set"):
        intake._project_payloads_for_authority(payloads, authorities)


def test_projection_rejects_non_dict_mapping_alias() -> None:
    payloads, authorities = _tiny_fixture()
    with pytest.raises(intake.AttrsDedupIntakeError, match="exact dict"):
        intake._project_payloads_for_authority(MappingProxyType(payloads), authorities)


def test_projection_rejects_same_size_git_blob_substitution() -> None:
    first = b"abc\n"
    substituted = b"xyz\n"
    authority = _authority("src/attr/a.py", first)
    with pytest.raises(intake.AttrsDedupIntakeError, match="Git blob identity drift"):
        intake._project_payloads_for_authority(
            {authority.path: substituted},
            (authority,),
        )


def test_projection_rejects_byte_size_drift_before_hash() -> None:
    payload = b"abc\n"
    authority = _authority("src/attr/a.py", payload)
    with pytest.raises(intake.AttrsDedupIntakeError, match="byte size drift"):
        intake._project_payloads_for_authority(
            {authority.path: payload + b"x"},
            (authority,),
        )


def test_projection_rejects_non_utf8_even_when_blob_identity_matches() -> None:
    payload = b"\xff"
    authority = _authority("src/attr/a.py", payload)
    with pytest.raises(intake.AttrsDedupIntakeError, match="strict UTF-8"):
        intake._project_payloads_for_authority({authority.path: payload}, (authority,))


def test_duplicate_authority_paths_fail_closed() -> None:
    payload = b"abc\n"
    authority = _authority("src/attr/a.py", payload)
    with pytest.raises(intake.AttrsDedupIntakeError, match="paths are not unique"):
        intake._project_payloads_for_authority(
            {authority.path: payload},
            (authority, authority),
        )


def test_public_projector_cannot_accept_unbound_tiny_payloads() -> None:
    payloads = {authority.path: b"x" for authority in intake.FILE_AUTHORITIES}
    with pytest.raises(intake.AttrsDedupIntakeError, match="byte size drift"):
        intake.project_attrs_payloads(payloads)


def test_root_projector_rejects_symlinked_authority_file(tmp_path: Path) -> None:
    authority = intake.FILE_AUTHORITIES[0]
    target = tmp_path / "outside.py"
    target.write_bytes(b"not the authority payload")
    candidate = tmp_path / authority.path
    candidate.parent.mkdir(parents=True)
    candidate.symlink_to(target)
    with pytest.raises(intake.AttrsDedupIntakeError, match="source file is symlink"):
        intake.project_attrs_root(tmp_path)


def test_validate_projection_rejects_wrong_type() -> None:
    with pytest.raises(intake.AttrsDedupIntakeError, match="projection type drift"):
        intake.validate_projection({})  # type: ignore[arg-type]
