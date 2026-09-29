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
    terminal_path = ROOT / qualifier.TERMINAL_AUTHORITY_PATH
    qualifier.verify_historical_terminal_authority_bytes(terminal_path.read_bytes())

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
    value = _minimal_valid_v2_packet()
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
    value = _minimal_valid_v2_packet()
    value["upstream"]["repository"] = "https://github.com/attacker/pydantic"
    _reseal_v2(value)

    with pytest.raises(
        qualifier.QualificationError,
        match="generated upstream identity drift",
    ):
        qualifier.verify_evidence(value, expected_source_sha="1" * 40)


def _reseal_v2(value: dict[str, object]) -> None:
    unsigned = dict(value)
    unsigned.pop("authority_identity_sha256", None)
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


def _minimal_valid_v2_packet() -> dict[str, object]:
    objects = []
    for source_id, path, blob, size, digest in qualifier.EXPECTED_OBJECTS:
        objects.append(
            {
                "source_id": source_id,
                "source_family": "github:pydantic/pydantic",
                "commit": qualifier.UPSTREAM_COMMIT,
                "path": path,
                "git_blob_sha1": blob,
                "size_bytes": size,
                "raw_sha256": digest,
                "normalized_sha256": digest,
                "normalization_policy": "STRICT_UTF8_IDENTITY_PRESERVE_V1",
                "parse_validity": "PASS_AST_PARSE_PY311",
                "secret_scan": "PASS_DATA227_SECRET_PATTERNS",
                "privacy_credential_scan": "PASS_NO_CREDENTIAL_BEARING_LITERAL_PATTERN",
                "capacity_counted": True,
                "generated_material": False,
                "authorship_class": "UPSTREAM_AUTHORED_IMPLEMENTATION",
                "training_purpose_decision": "ALLOWED",
                "redistribution_decision": "ALLOWED_WITH_MIT_NOTICE",
            }
        )
    value: dict[str, object] = {
        "schema_version": qualifier.SCHEMA,
        "worker_id": qualifier.WORKER,
        "status": "ADMIT",
        "authority": "EXTERNAL_REAL_CODE_SOURCE_TERMINAL_LOCAL_FREE",
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
            "repository": "https://github.com/pydantic/pydantic",
            "tag": "v2.13.4",
            "tag_object_sha1": qualifier.TAG_OBJECT,
            "commit": qualifier.UPSTREAM_COMMIT,
            "tag_signature_status": "UNSIGNED",
            "repository_fork": False,
            "repository_mirror": False,
        },
        "license": {
            "license_id": "MIT",
            "path": "LICENSE",
            "git_blob_sha1": qualifier.LICENSE_BLOB,
            "sha256": qualifier.LICENSE_SHA256,
            "model_training": "ALLOWED_BY_REVIEWED_MIT_GRANT",
            "derivatives": "ALLOWED",
            "redistribution": "ALLOWED_WITH_NOTICE",
            "notice_required": True,
        },
        "objects": objects,
        "evaluation_boundary": {
            "eval289_head_sha": qualifier.EVAL289_HEAD,
            "active_reserved_objects": 0,
            "selected_objects_overlap_active_reservation": False,
        },
        "source_family_accounting": {
            "new_source_family": "github:pydantic/pydantic",
            "independent_new_family_count": 1,
            "predecessor_family_count": 2,
            "resulting_family_count_if_registered": 3,
            "selected_implementation_object_count": 4,
            "selected_authored_capacity_bytes": 235_204,
            "generated_capacity_bytes": 0,
            "generated_capacity_objects": 0,
        },
        "checks": {
            "strict_utf8_identity_normalization": "PASS",
            "parse_validity": "PASS_4_OF_4",
            "secret_privacy": "PASS_4_OF_4",
            "generated_selected_count": 0,
            "generated_selected_bytes": 0,
            "exact_duplicate_sha256": [],
            "near_duplicate_threshold": qualifier.NEAR_THRESHOLD,
            "near_duplicate_pairs": [],
            "max_observed_pair": qualifier.EXPECTED_MAX_PAIR,
            "max_observed_jaccard": qualifier.EXPECTED_MAX_JACCARD,
            "current_eval_reservation_active_at_eval289_head": False,
        },
        "excluded_capacity": dict(qualifier.EXPECTED_EXCLUDED_CAPACITY),
        "execution": {
            "class": "LOCAL_FREE",
            "paid_compute_used": False,
            "model_training_executed": False,
            "network_use": "bounded immutable-source qualification only",
            "github_api_authenticated": False,
        },
        "terminal_decision": qualifier.TERMINAL_DECISION,
    }
    _reseal_v2(value)
    return value


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["objects"][0].__setitem__("path", "pydantic/evil.py"),
            "generated object authority drift",
        ),
        (
            lambda value: value["objects"][0].__setitem__("raw_sha256", "0" * 64),
            "generated object authority drift",
        ),
        (
            lambda value: value["evaluation_boundary"].__setitem__(
                "selected_objects_overlap_active_reservation", True
            ),
            "generated evaluation boundary drift",
        ),
        (
            lambda value: value["source_family_accounting"].__setitem__(
                "resulting_family_count_if_registered", 4
            ),
            "generated source-family accounting drift",
        ),
    ],
)
def test_generated_v2_rejects_coherent_resealed_semantic_substitution(
    mutation, match: str
) -> None:
    value = _minimal_valid_v2_packet()
    mutation(value)
    _reseal_v2(value)
    with pytest.raises(qualifier.QualificationError, match=match):
        qualifier.verify_evidence(value, expected_source_sha="1" * 40)


def test_strict_json_rejects_python_integer_digit_overflow() -> None:
    raw = ('{"value":' + ('9' * 10000) + '}').encode("ascii")
    with pytest.raises(qualifier.QualificationError, match="invalid JSON"):
        qualifier._load_json_bytes(raw, context="huge-int")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["checks"].__setitem__(
                "near_duplicate_pairs",
                [{"left": "a", "right": "b", "jaccard": 0.9}],
            ),
            "generated checks boundary drift",
        ),
        (
            lambda value: value.__setitem__(
                "excluded_capacity",
                {"docs_tests_examples": "included"},
            ),
            "generated excluded-capacity boundary drift",
        ),
        (
            lambda value: value["objects"][0].__setitem__(
                "secret_scan",
                "NOT_RUN",
            ),
            "generated object authority drift",
        ),
    ],
)
def test_generated_v2_rejects_resealed_aggregate_boundary_substitution(
    mutation, match: str
) -> None:
    value = _minimal_valid_v2_packet()
    value["checks"] = {
        "strict_utf8_identity_normalization": "PASS",
        "parse_validity": "PASS_4_OF_4",
        "secret_privacy": "PASS_4_OF_4",
        "generated_selected_count": 0,
        "generated_selected_bytes": 0,
        "exact_duplicate_sha256": [],
        "near_duplicate_threshold": qualifier.NEAR_THRESHOLD,
        "near_duplicate_pairs": [],
        "max_observed_pair": qualifier.EXPECTED_MAX_PAIR,
        "max_observed_jaccard": qualifier.EXPECTED_MAX_JACCARD,
        "current_eval_reservation_active_at_eval289_head": False,
    }
    value["excluded_capacity"] = dict(qualifier.EXPECTED_EXCLUDED_CAPACITY)
    mutation(value)
    _reseal_v2(value)
    with pytest.raises(qualifier.QualificationError, match=match):
        qualifier.verify_evidence(value, expected_source_sha="1" * 40)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.__setitem__(
                "canonical_capacity_credit_bytes",
                235_204,
            ),
            "generated evidence schema is not closed",
        ),
        (
            lambda value: value["source_family_accounting"].__setitem__(
                "training_authorized_bytes",
                235_204,
            ),
            "generated source-family accounting schema is not closed",
        ),
        (
            lambda value: value["objects"][0].__setitem__(
                "evaluation_eligible",
                True,
            ),
            "generated object row schema is not closed",
        ),
    ],
)
def test_generated_v2_rejects_resealed_unknown_authority_fields(
    mutation, match: str
) -> None:
    value = _minimal_valid_v2_packet()
    value["execution"]["network_use"] = "bounded immutable-source qualification only"
    value["execution"]["github_api_authenticated"] = False
    mutation(value)
    _reseal_v2(value)
    with pytest.raises(qualifier.QualificationError, match=match):
        qualifier.verify_evidence(value, expected_source_sha="1" * 40)


def test_minimal_valid_v2_packet_passes_full_verifier() -> None:
    value = _minimal_valid_v2_packet()
    qualifier.verify_evidence(value, expected_source_sha="1" * 40)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["upstream"].__setitem__(
                "tag_signature_status",
                "VERIFIED",
            ),
            "generated upstream identity drift",
        ),
        (
            lambda value: value["license"].__setitem__(
                "redistribution",
                "ALLOWED",
            ),
            "generated license authority drift",
        ),
        (
            lambda value: value["execution"].__setitem__(
                "network_use",
                "unbounded",
            ),
            "execution network-use drift",
        ),
        (
            lambda value: value.__setitem__(
                "terminal_decision",
                "ADMIT EVERYTHING",
            ),
            "generated terminal decision drift",
        ),
    ],
)
def test_generated_v2_rejects_resealed_known_field_semantic_drift(
    mutation, match: str
) -> None:
    value = _minimal_valid_v2_packet()
    mutation(value)
    _reseal_v2(value)
    with pytest.raises(qualifier.QualificationError, match=match):
        qualifier.verify_evidence(value, expected_source_sha="1" * 40)


def test_terminal_authority_byte_or_semantic_substitution_fails_closed() -> None:
    path = ROOT / qualifier.TERMINAL_AUTHORITY_PATH
    raw = path.read_bytes()
    value = qualifier._load_json_bytes(raw, context=str(path))
    value["source_family_accounting"]["training_exposure_created_by_this_authority"] = True
    changed = (
        qualifier.json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with pytest.raises(
        qualifier.QualificationError,
        match="historical terminal authority blob drift",
    ):
        qualifier.verify_historical_terminal_authority_bytes(changed)


def test_terminal_authority_duplicate_json_key_fails_closed_after_blob_binding(
    monkeypatch: MonkeyPatch,
) -> None:
    raw = b'{"schema_version":"x","schema_version":"y"}'
    monkeypatch.setattr(
        qualifier,
        "TERMINAL_AUTHORITY_BLOB",
        qualifier.git_blob_sha1(raw),
    )
    with pytest.raises(
        qualifier.QualificationError,
        match="duplicate JSON key",
    ):
        qualifier.verify_historical_terminal_authority_bytes(raw)
