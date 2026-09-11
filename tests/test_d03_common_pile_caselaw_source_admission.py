from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "verify_d03_common_pile_caselaw_source_admission.py"
spec = importlib.util.spec_from_file_location("caselaw_source_admission", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def load() -> dict:
    return mod.load_policy()


def write_policy(tmp_path: Path, policy: dict) -> Path:
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path


def record(
    *,
    source: str = "Caselaw Access Project",
    license_value: str = "Public Domain",
    url: str = "https://static.case.law/",
    metadata_extra: dict | None = None,
) -> dict:
    metadata = {"author": "PER CURIAM", "license": license_value, "url": url}
    metadata.update(metadata_extra or {})
    return {
        "id": "fixture/case.html",
        "source": source,
        "added": "2024-08-24T03:29:51.129235",
        "created": "2024-08-24T03:29:51.129683",
        "metadata": metadata,
        "text": "Fixture text is intentionally not inspected by the source-level authority.",
    }


def test_policy_binds_exact_candidate_and_remains_zero_credit() -> None:
    policy = load()
    binding = policy["candidate_binding"]
    assert binding["product_pr"] == 904
    assert binding["product_semantic_commit_sha"] == mod.PRODUCT_SEMANTIC_COMMIT
    assert binding["product_config_path"] == mod.PRODUCT_CONFIG_PATH
    assert binding["product_config_git_blob_sha1"] == mod.PRODUCT_CONFIG_BLOB_SHA1
    assert binding["product_materializer_path"] == mod.PRODUCT_MATERIALIZER_PATH
    assert (
        binding["product_materializer_git_blob_sha1"]
        == mod.PRODUCT_MATERIALIZER_BLOB_SHA1
    )
    assert binding["dataset"] == mod.DATASET
    assert [item["file"] for item in binding["objects"]] == [
        "cap_00044.jsonl.gz",
        "cap_00043.jsonl.gz",
    ]
    assert policy["admission_policy"]["payload_source_admission_executed"] is False
    assert policy["truth_boundary"]["training_authorized_bytes"] == 0
    assert policy["truth_boundary"]["training_eligible"] is False


def test_generic_registry_path_and_blob_are_exactly_pinned() -> None:
    authority = load()["project_authority"]
    assert authority["generic_registry_path"] == str(mod.GENERIC_REGISTRY_PATH)
    assert authority["generic_registry_git_blob_sha1"] == mod.GENERIC_REGISTRY_BLOB_SHA1
    assert mod.git_blob_sha1(mod.GENERIC_REGISTRY_PATH.read_bytes()) == mod.GENERIC_REGISTRY_BLOB_SHA1


def test_exact_cap_record_is_only_conditionally_source_admitted() -> None:
    decision = mod.assess_record(record(), load())
    assert decision["decision"] == "CONDITIONAL_SOURCE_ADMISSION"
    assert decision["training_authorized_bytes"] == 0
    assert decision["canonical_capacity_credited"] == 0
    assert decision["legal_conclusion_claimed"] is False


@pytest.mark.parametrize("source", ["Court Listener", "CourtListener", "Other Source"])
def test_non_cap_source_labels_are_not_admitted(source: str) -> None:
    decision = mod.assess_record(record(source=source), load())
    assert decision["decision"] == "NOT_SOURCE_ADMITTED"


def test_license_drift_is_not_admitted() -> None:
    decision = mod.assess_record(record(license_value="CC-BY"), load())
    assert decision == {
        "decision": "NOT_SOURCE_ADMITTED",
        "reason": "license_not_exact_public_domain",
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://static.case.law/",
        "https://static.case.law.evil.example/",
        "https://case.law.evil.example/",
        "https://user@case.law/",
        "https://case.law:443/",
    ],
)
def test_cap_url_authority_spoofing_fails_closed(url: str) -> None:
    decision = mod.assess_record(record(url=url), load())
    assert decision["decision"] == "NOT_SOURCE_ADMITTED"
    assert decision["reason"] == "cap_url_authority_mismatch"


@pytest.mark.parametrize("key", mod.FORBIDDEN_EDITORIAL_KEYS)
def test_editorial_or_annotation_metadata_is_not_admitted(key: str) -> None:
    decision = mod.assess_record(record(metadata_extra={key: "fixture"}), load())
    assert decision == {
        "decision": "NOT_SOURCE_ADMITTED",
        "reason": "editorial_metadata_present",
    }


def test_candidate_identity_substitution_fails_closed() -> None:
    policy = load()
    binding = json.loads(json.dumps(policy["candidate_binding"]))
    binding["objects"][0]["bytes"] += 1
    with pytest.raises(mod.AdmissionError, match="candidate identity substitution"):
        mod.validate_candidate_identity(binding, policy)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("product_semantic_commit_sha", "8579edbc918b83254c237c10c9d7d250961c9b7f"),
        ("product_config_git_blob_sha1", "0" * 40),
        ("product_materializer_git_blob_sha1", "f" * 40),
    ],
)
def test_product_semantic_binding_substitution_fails_closed(
    tmp_path: Path, key: str, value: str
) -> None:
    policy = load()
    policy["candidate_binding"][key] = value
    with pytest.raises(mod.AdmissionError, match="candidate identity drift"):
        mod.load_policy(write_policy(tmp_path, policy))


def test_generic_registry_path_substitution_fails_closed(tmp_path: Path) -> None:
    policy = load()
    policy["project_authority"]["generic_registry_path"] = str(tmp_path / "forged.json")
    with pytest.raises(mod.AdmissionError, match="generic registry path drift"):
        mod.load_policy(write_policy(tmp_path, policy))


def test_same_shape_forged_registry_bytes_fail_blob_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes
    registry = json.loads(mod.GENERIC_REGISTRY_PATH.read_bytes())
    row = next(item for item in registry["sources"] if item["key"] == mod.GENERIC_SOURCE_KEY)
    row["provenance_summary"] = f"{row['provenance_summary']} forged"
    forged = json.dumps(registry, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def fake_read_bytes(path: Path) -> bytes:
        if path == mod.GENERIC_REGISTRY_PATH:
            return forged
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    with pytest.raises(mod.AdmissionError, match="generic registry blob identity drift"):
        load()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("admission_policy", "common_pile_license_metadata_sufficient_alone", True),
        ("admission_policy", "accepted_source_exact", "CourtListener"),
        ("admission_policy", "payload_source_admission_executed", True),
        ("admission_policy", "admitted_payload_bytes", 1),
        ("truth_boundary", "training_authorized_bytes", 1),
        ("truth_boundary", "training_eligible", True),
        ("truth_boundary", "source_policy_review_complete", False),
    ],
)
def test_policy_truth_or_admission_mutation_fails_closed(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    policy = load()
    policy[section][key] = value
    with pytest.raises(mod.AdmissionError):
        mod.load_policy(write_policy(tmp_path, policy))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("admission_policy", "admitted_payload_records"),
        ("admission_policy", "admitted_payload_bytes"),
        ("truth_boundary", "canonical_capacity_credited"),
        ("truth_boundary", "training_authorized_bytes"),
        ("truth_boundary", "optimizer_updates"),
    ],
)
def test_bool_cannot_alias_numeric_zero(tmp_path: Path, section: str, key: str) -> None:
    policy = load()
    policy[section][key] = False
    with pytest.raises(mod.AdmissionError, match="numeric zero boundary drift"):
        mod.load_policy(write_policy(tmp_path, policy))


@pytest.mark.parametrize(
    "section",
    [None, "project_authority", "admission_policy", "truth_boundary"],
)
def test_unknown_authority_keys_fail_closed(tmp_path: Path, section: str | None) -> None:
    policy = load()
    target = policy if section is None else policy[section]
    target["unexpected_authority_key"] = "drift"
    with pytest.raises(mod.AdmissionError, match="key drift"):
        mod.load_policy(write_policy(tmp_path, policy))


def test_missing_authority_key_fails_closed(tmp_path: Path) -> None:
    policy = load()
    policy["project_authority"].pop("generic_source_key")
    with pytest.raises(mod.AdmissionError, match="project authority key drift"):
        mod.load_policy(write_policy(tmp_path, policy))


def test_public_evidence_substitution_fails_closed(tmp_path: Path) -> None:
    policy = load()
    policy["primary_public_evidence"][0]["supported_fact"] = "substituted"
    with pytest.raises(mod.AdmissionError, match="public evidence drift"):
        mod.load_policy(write_policy(tmp_path, policy))


def test_record_field_drift_fails_closed() -> None:
    row = record()
    row["extra"] = "drift"
    with pytest.raises(mod.AdmissionError, match="record field drift"):
        mod.assess_record(row, load())
