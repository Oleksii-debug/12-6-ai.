from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from tools import qualify_next100_048_pydantic as qualifier

ROOT = Path(__file__).resolve().parents[1]


def test_committed_pydantic_terminal_evidence_is_self_consistent() -> None:
    path = ROOT / "evidence/next100-048/pydantic-source-admission-v1.json"
    value = qualifier._load_json_bytes(path.read_bytes(), context=str(path))

    qualifier.verify_historical_evidence(value)

    predecessor = value["predecessor_code_authority"]
    assert predecessor["data227_head_sha"] == qualifier.DATA227_HEAD
    assert predecessor["rights_policy_git_blob_sha1"] == qualifier.DATA227_POLICY_BLOB
    assert value["upstream"]["commit"] == qualifier.UPSTREAM_COMMIT
    assert value["license"]["git_blob_sha1"] == qualifier.LICENSE_BLOB
    assert value["source_family_accounting"]["new_source_family"] == "github:pydantic/pydantic"
    assert value["source_family_accounting"]["selected_authored_capacity_bytes"] == 235_204


def test_data227_policy_fallback_is_pinned_and_bounded(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    raw = b'{"decisions":[]}\n'
    expected_blob = qualifier.git_blob_sha1(raw)
    observed: list[tuple[str, int]] = []

    def fake_download(url: str, max_bytes: int = 300_000) -> bytes:
        observed.append((url, max_bytes))
        return raw

    monkeypatch.setattr(qualifier, "DATA227_POLICY_BLOB", expected_blob)
    monkeypatch.setattr(qualifier, "download", fake_download)

    value = qualifier.load_data227_policy(tmp_path)

    assert value == {"decisions": []}
    assert observed == [
        (
            "https://raw.githubusercontent.com/Oleksii-debug/12-6-ai./"
            + qualifier.DATA227_HEAD
            + "/configs/data/data227_code_rights_policy_v1.json",
            100_000,
        )
    ]



@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":"x","schema_version":"y"}',
        b'{"schema_version":NaN}',
        b'{"schema_version":Infinity}',
        b'{"schema_version":1e400}',
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_values(raw: bytes) -> None:
    with pytest.raises(
        qualifier.QualificationError,
        match="duplicate JSON key|non-finite JSON number",
    ):
        qualifier._load_json_bytes(raw, context="adversarial")


def test_generated_and_historical_evidence_use_distinct_schema_versions() -> None:
    assert qualifier.HISTORICAL_SCHEMA == (
        "12-6.next100-048-pydantic-source-admission.v1"
    )
    assert qualifier.SCHEMA == "12-6.next100-048-pydantic-source-admission.v2"



def test_historical_coherent_rehash_substitution_is_rejected() -> None:
    path = ROOT / "evidence/next100-048/pydantic-source-admission-v1.json"
    value = qualifier._load_json_bytes(path.read_bytes(), context=str(path))
    value["status"] = "RETEST"
    unsigned = dict(value)
    unsigned.pop("authority_identity_sha256")
    canonical = (
        qualifier.json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    value["authority_identity_sha256"] = qualifier.sha256(canonical)

    with pytest.raises(
        qualifier.QualificationError,
        match="historical authority identity drift",
    ):
        qualifier.verify_historical_evidence(value)


def test_generated_v2_requires_external_source_sha_binding() -> None:
    value = {
        "schema_version": qualifier.SCHEMA,
        "status": "ADMIT",
        "worker_source_sha": "1" * 40,
        "predecessor_code_authority": {},
        "source_family_accounting": {
            "selected_implementation_object_count": 4,
            "selected_authored_capacity_bytes": 235_204,
        },
        "execution": {
            "class": "LOCAL_FREE",
            "paid_compute_used": False,
            "model_training_executed": False,
        },
    }
    unsigned = dict(value)
    canonical = (
        qualifier.json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    value["authority_identity_sha256"] = qualifier.sha256(canonical)

    with pytest.raises(
        qualifier.QualificationError,
        match="generated worker source SHA drift",
    ):
        qualifier.verify_evidence(value, expected_source_sha="2" * 40)



def test_policy_bytes_are_pinned_before_semantic_use(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    policy = (
        ROOT / "configs/data/next100_048_pydantic_code_rights_v1.json"
    ).read_bytes()
    changed = policy.replace(b'"MIT"', b'"BSD"', 1)
    assert changed != policy
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(changed)

    with pytest.raises(
        qualifier.QualificationError,
        match="Pydantic rights policy blob drift",
    ):
        qualifier.qualify(
            repo_root=tmp_path,
            policy_path=Path("policy.json"),
            source_sha="1" * 40,
        )


def test_generated_v2_rejects_coherent_upstream_substitution() -> None:
    value = {
        "schema_version": qualifier.SCHEMA,
        "status": "ADMIT",
        "worker_source_sha": "1" * 40,
        "candidate_policy_authority": {
            "schema_version": "12-6.next100-048-pydantic-code-rights.v1",
            "git_blob_sha1": qualifier.PYDANTIC_POLICY_BLOB,
        },
        "predecessor_code_authority": {
            "data227_head_sha": qualifier.DATA227_HEAD,
            "rights_policy_git_blob_sha1": qualifier.DATA227_POLICY_BLOB,
            "source_family_count": 2,
            "source_families": [
                "github:encode/httpx",
                "github:psf/requests",
            ],
            "near_duplicate_policy": {
                "reject_at_or_above_jaccard": qualifier.NEAR_THRESHOLD,
                "shingle_tokens": qualifier.SHINGLE_SIZE,
            },
        },
        "upstream": {
            "repository": "https://github.com/attacker/pydantic",
            "commit": qualifier.UPSTREAM_COMMIT,
            "tag_object_sha1": qualifier.TAG_OBJECT,
        },
        "license": {
            "license_id": "MIT",
            "git_blob_sha1": qualifier.LICENSE_BLOB,
        },
        "source_family_accounting": {
            "selected_implementation_object_count": 4,
            "selected_authored_capacity_bytes": 235_204,
        },
        "execution": {
            "class": "LOCAL_FREE",
            "paid_compute_used": False,
            "model_training_executed": False,
        },
    }
    unsigned = dict(value)
    canonical = (
        qualifier.json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    value["authority_identity_sha256"] = qualifier.sha256(canonical)

    with pytest.raises(
        qualifier.QualificationError,
        match="generated upstream identity drift",
    ):
        qualifier.verify_evidence(value, expected_source_sha="1" * 40)
