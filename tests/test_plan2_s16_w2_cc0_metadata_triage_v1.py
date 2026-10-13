"""W2 independent CC0 source-metadata candidate denial gates (not S3-S9 admission)."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "configs/data/plan2_s16_w2_cc0_metadata_triage_v1.json"
SHA256 = "a833fef5cbf8f00aff7effb960d260af782acecf6ef1938c1690cf0fb41e3444"
IDS = {"wikidata-structured-cc0-candidate", "openalex-catalog-metadata-cc0-candidate"}
GATES = {
    "OWNER_APPROVAL_OF_SPECIFIC_SOURCE_AND_PURPOSE",
    "IMMUTABLE_SOURCE_REVISION_AND_ORIGINAL_LICENSE_BYTES",
    "PHYSICAL_MEMBER_PROVENANCE_AND_SHA256",
    "THIRD_PARTY_CONTENT_BOUNDARY_REVIEW",
    "RECORD_LEVEL_PRIVACY_PII_AND_REMOVAL_REVIEW",
    "CROSS_FAMILY_EXACT_NEAR_DEDUP_AND_EVAL_DECONTAMINATION",
    "INDEPENDENT_UA_EN_TRAIN_VALIDATION_TEST_FAMILIES",
    "APPROVED_LANGUAGE_SCIENCE_BALANCE_AND_S3_S9_ADMISSION",
}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ValueError("invalid bytes")
    try:
        obj = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc
    if type(obj) is not dict or obj.get("decision") != "CANDIDATE_METADATA_ONLY_NOT_ADMITTED":
        raise ValueError("not a candidate")
    if obj.get("schema_version") != "12-6.plan2-s16-w2-cc0-metadata-triage.v1":
        raise ValueError("schema drift")
    if obj.get("as_of_utc_date") != "2026-10-13":
        raise ValueError("unreviewed revision")
    if any(obj.get(key) is not False for key in (
        "physical_corpus_materialized", "corpus_training_authorized",
        "production_release_authorized", "plan9_binding_authorized", "terminal_done"
    )):
        raise ValueError("forged authorization")
    gates = obj.get("required_before_admission")
    if type(gates) is not list or len(gates) != len(GATES) or any(type(x) is not str for x in gates) or set(gates) != GATES:
        raise ValueError("missing rights gate")
    sources = obj.get("sources")
    if type(sources) is not list or len(sources) != 2 or any(type(x) is not dict for x in sources):
        raise ValueError("invalid sources")
    if {x.get("source_identity") for x in sources} != IDS:
        raise ValueError("source identity drift")
    for source in sources:
        if source.get("rights_status") != "SOURCE_LEVEL_CC0_ONLY":
            raise ValueError("rights promotion")
        if source.get("training_authorized") is not False or source.get("release_authorized") is not False:
            raise ValueError("source grant")
        if any(source.get(key) is not None for key in (
            "upstream_revision", "physical_source_sha256", "original_license_bytes_sha256"
        )):
            raise ValueError("fabricated physical provenance")
        if type(source.get("acquired_records")) is not int or source["acquired_records"] != 0:
            raise ValueError("unearned acquisition")
        if type(source.get("source_family_credit")) is not int or source["source_family_credit"] != 0:
            raise ValueError("unearned family")
        if source.get("observed_ua_en_science_coverage") != "UNKNOWN":
            raise ValueError("unverified balance")
        if "personal_data_without_review" not in source.get("excluded_scope", []):
            raise ValueError("privacy boundary")
        if source["source_identity"].startswith("openalex"):
            if "third_party_fulltext" not in source.get("excluded_scope", []):
                raise ValueError("third-party copyright")
        elif "external_linked_fulltext" not in source.get("excluded_scope", []):
            raise ValueError("linked content")
    return obj


def test_pinned_candidate_fixture_is_nonrelease():
    raw = PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    assert len(_read(raw)["sources"]) == 2


@pytest.mark.parametrize("field", [
    "physical_corpus_materialized", "corpus_training_authorized",
    "production_release_authorized", "plan9_binding_authorized", "terminal_done"
])
def test_global_promotion_rejected(field):
    obj = copy.deepcopy(_read(PATH.read_bytes()))
    obj[field] = True
    with pytest.raises(ValueError):
        _read(json.dumps(obj).encode())


@pytest.mark.parametrize("field,value", [
    ("rights_status", "ADMITTED"), ("training_authorized", True),
    ("release_authorized", True), ("upstream_revision", "latest"),
    ("physical_source_sha256", "a" * 64),
    ("original_license_bytes_sha256", "b" * 64),
    ("acquired_records", 1), ("acquired_records", True),
    ("source_family_credit", 1), ("observed_ua_en_science_coverage", "VERIFIED"),
])
def test_source_promotion_rejected(field, value):
    obj = copy.deepcopy(_read(PATH.read_bytes()))
    obj["sources"][0][field] = value
    with pytest.raises(ValueError):
        _read(json.dumps(obj).encode())


def test_privacy_and_fulltext_exclusions_cannot_be_removed():
    for source_index, boundary in ((0, "external_linked_fulltext"), (1, "third_party_fulltext"), (0, "personal_data_without_review")):
        obj = copy.deepcopy(_read(PATH.read_bytes()))
        obj["sources"][source_index]["excluded_scope"].remove(boundary)
        with pytest.raises(ValueError):
            _read(json.dumps(obj).encode())


def test_missing_owner_approval_rejected():
    obj = copy.deepcopy(_read(PATH.read_bytes()))
    obj["required_before_admission"].remove("OWNER_APPROVAL_OF_SPECIFIC_SOURCE_AND_PURPOSE")
    with pytest.raises(ValueError):
        _read(json.dumps(obj).encode())


@pytest.mark.parametrize("bad", [[], {}, ["unhashable"], {"fake": "rights"}])
def test_malformed_nested_gates_fail_closed(bad):
    obj = copy.deepcopy(_read(PATH.read_bytes()))
    obj["required_before_admission"][0] = bad
    with pytest.raises(ValueError):
        _read(json.dumps(obj).encode())


def test_duplicate_keys_fail_closed():
    raw = PATH.read_bytes().replace(b'"terminal_done": false', b'"terminal_done": false, "terminal_done": true')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _read(raw)


def test_restart_and_unpublished_failure(tmp_path):
    raw = PATH.read_bytes()
    assert _read(raw) == _read(raw)
    published = tmp_path / "candidate.json"
    published.write_bytes(raw)
    obj = copy.deepcopy(_read(raw))
    obj["corpus_training_authorized"] = True
    staged = tmp_path / "candidate.json.tmp"
    staged.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError):
        _read(staged.read_bytes())
    assert published.read_bytes() == raw
