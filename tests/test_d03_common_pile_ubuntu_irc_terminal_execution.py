from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
EVIDENCE_PATH = REPO_ROOT / "evidence/d03_common_pile_ubuntu_irc_real_execution_v1.json"
CONFIG_PATH = REPO_ROOT / "configs/data/d03_common_pile_ubuntu_irc_bounded_v1.json"
MATERIALIZER_PATH = REPO_ROOT / "tools/materialize_d03_common_pile_ubuntu_irc_v1.py"
FOCUSED_TEST_PATH = REPO_ROOT / "tests/test_d03_common_pile_ubuntu_irc_v1.py"
TEMP_WORKFLOW_PATH = (
    REPO_ROOT / ".github/workflows/temp-d03-ubuntu-irc-real-exec-1096.yml"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

EXECUTION_HEAD = "3400e2cd3c62a360b25282310390640a7f20fbc9"
EXECUTION_TREE_SHA1 = "6bb849adf3f480fcce4e48ddc4361f2bb8e29dd8"
WORKFLOW_BLOB_SHA1 = "f6268b728809850059eabf71d7389f322737f924"
CONFIG_BLOB_SHA1 = "5a92b605325d16572321ee1c04aed8dcc6e660ea"
MATERIALIZER_BLOB_SHA1 = "3c945c004a4a53ff6342dc6e08eda0eb1d9ceae2"
FOCUSED_TEST_BLOB_SHA1 = "4cbbf442d21ff7c7ac1762e7d1049433894a9e45"
ARTIFACT_SHA256 = "17a9da055c99f54d9bec3a7afcb1521a19c68c0fea26485e7cbc11a8512fbeb6"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def expect_exact_keys(value: object, expected: set[str]) -> dict:
    assert type(value) is dict
    assert set(value) == expected
    return value


def expect_int(value: object, expected: int) -> None:
    assert type(value) is int
    assert value == expected


def expect_bool(value: object, expected: bool) -> None:
    assert type(value) is bool
    assert value is expected


def expect_str(value: object, expected: str) -> None:
    assert type(value) is str
    assert value == expected


def validate_evidence(evidence: dict, config: dict) -> None:
    expect_exact_keys(
        evidence,
        {
            "schema_version",
            "worker_issue",
            "incumbent_pr",
            "execution_profile",
            "execution_head_sha",
            "execution_binding",
            "workflow_run_id",
            "workflow_job_id",
            "workflow_run_attempt",
            "workflow_conclusion",
            "runner",
            "focused_adversarial_tests",
            "materialization",
            "artifact",
            "truth_boundary",
            "next_required_gates",
        },
    )
    expect_str(
        evidence["schema_version"],
        "12-6.d03-common-pile-ubuntu-irc-real-execution.v1",
    )
    expect_int(evidence["worker_issue"], 1096)
    expect_int(evidence["incumbent_pr"], 913)
    expect_str(evidence["execution_profile"], "LOCAL_FREE_GITHUB_HOSTED_UBUNTU")
    expect_str(evidence["execution_head_sha"], EXECUTION_HEAD)
    expect_int(evidence["workflow_run_id"], 34549598850)
    expect_int(evidence["workflow_job_id"], 103109520811)
    expect_int(evidence["workflow_run_attempt"], 1)
    expect_str(evidence["workflow_conclusion"], "success")

    binding = expect_exact_keys(
        evidence["execution_binding"],
        {
            "repository_commit",
            "repository_tree_sha1",
            "workflow",
            "config",
            "materializer",
            "focused_test",
        },
    )
    expect_str(binding["repository_commit"], EXECUTION_HEAD)
    expect_str(binding["repository_tree_sha1"], EXECUTION_TREE_SHA1)
    for section_name, path, blob_sha1 in (
        (
            "workflow",
            ".github/workflows/temp-d03-ubuntu-irc-real-exec-1096.yml",
            WORKFLOW_BLOB_SHA1,
        ),
        (
            "config",
            "configs/data/d03_common_pile_ubuntu_irc_bounded_v1.json",
            CONFIG_BLOB_SHA1,
        ),
        (
            "materializer",
            "tools/materialize_d03_common_pile_ubuntu_irc_v1.py",
            MATERIALIZER_BLOB_SHA1,
        ),
        (
            "focused_test",
            "tests/test_d03_common_pile_ubuntu_irc_v1.py",
            FOCUSED_TEST_BLOB_SHA1,
        ),
    ):
        section = expect_exact_keys(binding[section_name], {"path", "git_blob_sha1"})
        expect_str(section["path"], path)
        expect_str(section["git_blob_sha1"], blob_sha1)

    runner = expect_exact_keys(
        evidence["runner"],
        {"os", "image", "image_version", "python_version"},
    )
    expect_str(runner["os"], "Ubuntu 24.04.5 LTS")
    expect_str(runner["image"], "ubuntu-24.04")
    expect_str(runner["image_version"], "20260907.300.1")
    expect_str(runner["python_version"], "3.11.16")

    focused = expect_exact_keys(
        evidence["focused_adversarial_tests"],
        {"conclusion", "passed"},
    )
    expect_str(focused["conclusion"], "success")
    expect_int(focused["passed"], 22)

    materialization = expect_exact_keys(
        evidence["materialization"],
        {
            "status",
            "independent_build_count",
            "independent_acquisition_count",
            "candidate_byte_identical",
            "report_byte_identical",
            "source_dataset",
            "source_revision",
            "source_file",
            "source_sha256",
            "observed_source_bytes",
            "scanned_records",
            "retained_records",
            "retained_normalized_utf8_bytes",
            "candidate_payload_bytes",
            "selection_reasons",
            "candidate_payload_sha256",
            "candidate_file_sha256",
            "report_file_sha256",
        },
    )
    expect_str(
        materialization["status"],
        "TWO_INDEPENDENT_MATERIALIZATIONS_BYTE_IDENTICAL",
    )
    expect_int(materialization["independent_build_count"], 2)
    expect_int(materialization["independent_acquisition_count"], 2)
    expect_bool(materialization["candidate_byte_identical"], True)
    expect_bool(materialization["report_byte_identical"], True)

    source = config["source"]
    selection = config["selection"]
    expect_str(materialization["source_dataset"], "common-pile/ubuntu_irc")
    expect_str(materialization["source_revision"], source["revision"])
    expect_str(materialization["source_file"], source["file"])
    expect_str(materialization["source_sha256"], source["sha256"])
    expect_int(materialization["observed_source_bytes"], 154409587)
    assert materialization["observed_source_bytes"] <= source["max_compressed_bytes"]
    expect_int(materialization["scanned_records"], selection["max_scanned_records"])
    expect_int(materialization["retained_records"], 972)
    expect_int(materialization["retained_normalized_utf8_bytes"], 4799981)
    assert materialization["retained_normalized_utf8_bytes"] <= (
        selection["max_retained_normalized_utf8_bytes"]
    )
    expect_int(materialization["candidate_payload_bytes"], 5133301)

    reasons = expect_exact_keys(
        materialization["selection_reasons"],
        {
            "accepted",
            "control_character",
            "email",
            "high_cyrillic_alpha_ratio",
            "ipv4",
            "low_latin_alpha_ratio",
            "phone",
            "retained_byte_cap_reached",
            "secret_marker",
            "too_long",
            "too_short",
        },
    )
    expected_reasons = {
        "accepted": 972,
        "control_character": 673,
        "email": 116,
        "high_cyrillic_alpha_ratio": 1,
        "ipv4": 229,
        "low_latin_alpha_ratio": 79,
        "phone": 433,
        "retained_byte_cap_reached": 745,
        "secret_marker": 1,
        "too_long": 29,
        "too_short": 818,
    }
    for key, expected in expected_reasons.items():
        expect_int(reasons[key], expected)
    assert sum(reasons.values()) == materialization["scanned_records"]

    expect_str(
        materialization["candidate_payload_sha256"],
        "d6a0eaea27313e0d5146bc2a0624104541923ef06b12f01c22b3ccf6956eb6c2",
    )
    expect_str(
        materialization["candidate_file_sha256"],
        "d6a0eaea27313e0d5146bc2a0624104541923ef06b12f01c22b3ccf6956eb6c2",
    )
    expect_str(
        materialization["report_file_sha256"],
        "0fac5af5f2b72af91a4b83cc0ec10ccd737903ea2d4983c26903b3e4d9d95253",
    )
    for key in (
        "candidate_payload_sha256",
        "candidate_file_sha256",
        "report_file_sha256",
    ):
        assert SHA256_RE.fullmatch(materialization[key]) is not None

    artifact = expect_exact_keys(
        evidence["artifact"],
        {
            "name",
            "artifact_id",
            "zip_size_bytes",
            "zip_sha256",
            "api_digest",
            "expires_at_utc",
        },
    )
    expect_str(artifact["name"], "d03-ubuntu-irc-real-execution-1096")
    expect_int(artifact["artifact_id"], 10180292836)
    expect_int(artifact["zip_size_bytes"], 1065)
    expect_str(artifact["zip_sha256"], ARTIFACT_SHA256)
    expect_str(artifact["api_digest"], f"sha256:{ARTIFACT_SHA256}")
    expect_str(artifact["expires_at_utc"], "2026-09-18T01:11:16Z")

    truth = expect_exact_keys(
        evidence["truth_boundary"],
        {
            "durable_evidence_contains_source_text",
            "source_rights_review_status",
            "canonical_corpus_admitted",
            "training_eligible",
            "evaluation_eligible",
            "training_authorized_bytes",
            "canonical_capacity_credited",
            "family_credit_added",
            "unique_causal_loss_positions_authorized",
            "tokenizer_fit_authorized",
            "optimizer_updates",
            "model_training_executed",
            "learned_weights_created",
            "final_test_accessed",
            "paid_compute_used",
            "foreign_pretrained_weights_used",
            "external_llm_or_api_used_for_data_or_intelligence",
        },
    )
    for key in (
        "durable_evidence_contains_source_text",
        "canonical_corpus_admitted",
        "training_eligible",
        "evaluation_eligible",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "learned_weights_created",
        "final_test_accessed",
        "paid_compute_used",
        "foreign_pretrained_weights_used",
        "external_llm_or_api_used_for_data_or_intelligence",
    ):
        expect_bool(truth[key], False)
    expect_str(truth["source_rights_review_status"], "REVIEW_REQUIRED")
    for key in (
        "training_authorized_bytes",
        "canonical_capacity_credited",
        "family_credit_added",
        "unique_causal_loss_positions_authorized",
        "optimizer_updates",
    ):
        expect_int(truth[key], 0)

    next_gates = evidence["next_required_gates"]
    assert type(next_gates) is list
    assert next_gates == [
        "final_source_rights_review",
        "current_global_cross_source_dedup",
        "fresh_reserved_evaluation_decontamination",
        "post_composition_quality_privacy",
        "balance_and_family_caps",
        "cluster_safe_split",
        "deterministic_packing_two_clean_builds",
        "positive_exact_unique_loss_ledger",
    ]


def test_execution_receipt_is_strictly_bound() -> None:
    validate_evidence(load_json(EVIDENCE_PATH), load_json(CONFIG_PATH))


def test_carried_scientific_bytes_match_executed_git_blobs() -> None:
    evidence = load_json(EVIDENCE_PATH)
    binding = evidence["execution_binding"]

    assert git_blob_sha1(CONFIG_PATH) == binding["config"]["git_blob_sha1"]
    assert git_blob_sha1(MATERIALIZER_PATH) == binding["materializer"]["git_blob_sha1"]
    assert git_blob_sha1(FOCUSED_TEST_PATH) == binding["focused_test"]["git_blob_sha1"]
    assert not TEMP_WORKFLOW_PATH.exists()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("execution_head_sha",), "0" * 40),
        (("execution_binding", "repository_tree_sha1"), "0" * 40),
        (("execution_binding", "config", "git_blob_sha1"), "0" * 40),
        (("workflow_run_id",), 1),
        (("workflow_job_id",), 1),
        (("artifact", "artifact_id"), 1),
        (("artifact", "api_digest"), "sha256:" + "0" * 64),
        (("truth_boundary", "training_authorized_bytes"), False),
        (("truth_boundary", "optimizer_updates"), False),
    ],
)
def test_provenance_and_numeric_alias_substitutions_fail_closed(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    evidence = load_json(EVIDENCE_PATH)
    config = load_json(CONFIG_PATH)
    tampered = copy.deepcopy(evidence)
    cursor = tampered
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement

    with pytest.raises(AssertionError):
        validate_evidence(tampered, config)


def test_unknown_fields_fail_closed() -> None:
    evidence = load_json(EVIDENCE_PATH)
    config = load_json(CONFIG_PATH)
    tampered = copy.deepcopy(evidence)
    tampered["execution_binding"]["unexpected"] = "reseal"

    with pytest.raises(AssertionError):
        validate_evidence(tampered, config)


def test_execution_evidence_remains_text_free() -> None:
    evidence = load_json(EVIDENCE_PATH)
    assert "text" not in evidence["materialization"]
    assert "rows" not in evidence["materialization"]
