from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from twelve_six.data import eval303_selection_payload_resolver_v1 as module


def _canonical_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_zip(path: Path, members: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
    return _sha(path.read_bytes())


def _build_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ua_rows = []
    membership = []
    for index in range(8):
        text = f"український резервований текст номер {index} для перевірки"
        raw = text.encode("utf-8")
        record_id = f"ua-{index}"
        source_family = "ua-family-a" if index < 4 else "ua-family-b"
        source_id = "ua-source-a" if index < 4 else "ua-source-b"
        ua_rows.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "source_family": source_family,
                "modality": "ua",
                "text": text,
                "content_sha256": _sha(raw),
                "utf8_bytes": len(raw),
                "purpose": "selection-validation",
                "selection_eligible": True,
                "training_eligible": False,
                "tokenizer_fit_eligible": False,
                "final_test_eligible": False,
                "future_training_prohibited": True,
            }
        )
        membership.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "source_family": source_family,
                "modality": "ua",
                "content_sha256": _sha(raw),
                "utf8_bytes": len(raw),
                "purpose": "selection-validation",
                "selection_eligible": True,
                "training_eligible": False,
                "tokenizer_fit_eligible": False,
                "final_test_eligible": False,
            }
        )

    en_rows = []
    en_sources = []
    for index in range(2):
        text = f"English reserved selection payload number {index}."
        raw = text.encode("utf-8")
        record_id = f"en-{index}"
        family = f"github:example/repo{index}"
        path = f"docs/example-{index}.md"
        en_rows.append(
            {
                "document_id": record_id,
                "source_family": family,
                "source_path": path,
                "purpose": "selection_validation",
                "text": text,
                "content_sha256": _sha(raw),
            }
        )
        en_sources.append(
            {
                "document_id": record_id,
                "source_family": family,
                "path": path,
                "raw_sha256": _sha(raw),
                "raw_bytes": len(raw),
                "project_reservation": {
                    "selection_validation": True,
                    "training": False,
                    "tokenizer_fit": False,
                    "final_test": False,
                },
            }
        )
        membership.append(
            {
                "record_id": record_id,
                "source_id": f"{family}:{path}",
                "source_family": family,
                "modality": "text",
                "content_sha256": _sha(raw),
                "utf8_bytes": len(raw),
                "purpose": "selection-validation",
                "selection_eligible": True,
                "training_eligible": False,
                "tokenizer_fit_eligible": False,
                "final_test_eligible": False,
            }
        )

    membership_raw = b"".join(_canonical_line(row) for row in membership)
    membership_path = tmp_path / "membership.jsonl"
    membership_path.write_bytes(membership_raw)

    ua_data = b"".join(_canonical_line(row) for row in ua_rows)
    ua_manifest = {
        "set_identity_sha256": "a" * 64,
        "documents": 8,
        "eligibility": {
            "training": False,
            "tokenizer_fit": False,
            "final_test": False,
            "model_selection": True,
        },
    }
    ua_manifest_raw = json.dumps(ua_manifest, sort_keys=True).encode("utf-8")
    ua_zip = tmp_path / "ua.zip"
    ua_zip_sha = _write_zip(
        ua_zip,
        {
            module.EVAL290_DATA_PATH: ua_data,
            module.EVAL290_MANIFEST_PATH: ua_manifest_raw,
        },
    )

    en_data = b"".join(_canonical_line(row) for row in en_rows)
    en_authority = {
        "authority_identity_sha256": "b" * 64,
        "documents": 2,
        "purpose": "selection_validation",
        "firewalls": {
            "selection_bytes_are_training_eligible": False,
            "selection_bytes_are_tokenizer_fit_eligible": False,
            "selection_bytes_are_final_test_eligible": False,
        },
        "final_test_boundary": {
            "outcomes_read_for_construction": False,
            "payload_read_for_construction": False,
        },
        "sources": en_sources,
    }
    en_authority_raw = json.dumps(en_authority, sort_keys=True).encode("utf-8")
    en_zip = tmp_path / "en.zip"
    en_zip_sha = _write_zip(
        en_zip,
        {
            module.EVAL291_DATA_PATH: en_data,
            module.EVAL291_AUTHORITY_PATH: en_authority_raw,
        },
    )

    replacements = {
        "EVAL303_MEMBERSHIP_SHA256": _sha(membership_raw),
        "EVAL290_ARTIFACT_SHA256": ua_zip_sha,
        "EVAL290_DATA_SHA256": _sha(ua_data),
        "EVAL290_MANIFEST_SHA256": _sha(ua_manifest_raw),
        "EVAL290_SET_ID": "a" * 64,
        "EVAL291_ARTIFACT_SHA256": en_zip_sha,
        "EVAL291_DATA_SHA256": _sha(en_data),
        "EVAL291_AUTHORITY_SHA256": _sha(en_authority_raw),
        "EVAL291_AUTHORITY_ID": "b" * 64,
    }
    for key, value in replacements.items():
        monkeypatch.setattr(module, key, value)
    return ua_zip, en_zip, membership_path, ua_rows, en_rows


def test_resolver_builds_exact_ten_ephemeral_rows_and_text_free_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ua_zip, en_zip, membership, ua_rows, en_rows = _build_fixture(tmp_path, monkeypatch)
    rows, reserved_set, evidence = module.resolve_eval303_selection_payloads(
        ua_zip,
        en_zip,
        membership,
    )
    assert len(rows) == 10
    assert sum(row["modality"] == "ua" for row in rows) == 8
    assert sum(row["modality"] == "text" for row in rows) == 2
    assert reserved_set["role"] == "selection_validation"
    assert len(reserved_set["members"]) == 10
    assert evidence["documents"] == 10
    assert evidence["raw_text_persisted_in_evidence"] is False
    assert evidence["selection_payload_accessed_for_decontamination"] is True
    assert evidence["final_test_payload_accessed"] is False
    assert evidence["final_test_outcomes_read"] is False
    assert evidence["authorized_training_exposure"] == 0
    durable = json.dumps({"reserved": reserved_set, "evidence": evidence})
    for source in [*ua_rows, *en_rows]:
        assert source["text"] not in durable


def test_en_payload_uses_eval303_text_modality_not_component_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ua_zip, en_zip, membership, _, _ = _build_fixture(tmp_path, monkeypatch)
    rows, _, _ = module.resolve_eval303_selection_payloads(ua_zip, en_zip, membership)
    assert {row["modality"] for row in rows if row["record_id"].startswith("en-")} == {
        "text"
    }


def test_artifact_zip_mutation_fails_exact_outer_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ua_zip, en_zip, membership, _, _ = _build_fixture(tmp_path, monkeypatch)
    raw = bytearray(ua_zip.read_bytes())
    raw[-1] ^= 1
    ua_zip.write_bytes(raw)
    with pytest.raises(module.SelectionPayloadResolverError, match="ZIP identity drift"):
        module.resolve_eval303_selection_payloads(ua_zip, en_zip, membership)


def test_membership_mutation_fails_before_payload_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ua_zip, en_zip, membership, _, _ = _build_fixture(tmp_path, monkeypatch)
    original = membership.read_bytes()
    membership.write_bytes(original + b"\n")
    with pytest.raises(module.SelectionPayloadResolverError, match="membership SHA-256 drift"):
        module.resolve_eval303_selection_payloads(ua_zip, en_zip, membership)


def test_component_content_substitution_fails_even_with_valid_zip_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ua_zip, en_zip, membership, _, _ = _build_fixture(tmp_path, monkeypatch)
    with zipfile.ZipFile(en_zip) as archive:
        data_raw = archive.read(module.EVAL291_DATA_PATH)
        authority_raw = archive.read(module.EVAL291_AUTHORITY_PATH)
    rows = [json.loads(line) for line in data_raw.splitlines()]
    mutated = copy.deepcopy(rows)
    mutated[0]["text"] += " mutation"
    mutated_raw = b"".join(_canonical_line(row) for row in mutated)
    new_zip_sha = _write_zip(
        en_zip,
        {
            module.EVAL291_DATA_PATH: mutated_raw,
            module.EVAL291_AUTHORITY_PATH: authority_raw,
        },
    )
    monkeypatch.setattr(module, "EVAL291_ARTIFACT_SHA256", new_zip_sha)
    monkeypatch.setattr(module, "EVAL291_DATA_SHA256", _sha(mutated_raw))
    with pytest.raises(module.SelectionPayloadResolverError, match="content hash drift"):
        module.resolve_eval303_selection_payloads(ua_zip, en_zip, membership)
