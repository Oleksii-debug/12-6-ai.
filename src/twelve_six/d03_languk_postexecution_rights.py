"""Fail-closed post-execution rights bridge for the exact LangUK Supreme Court candidate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "12-6.d03-languk-postexecution-rights-admission.v1"
AUTHORITY_ID = "D03-LANGUK-SUPREME-COURT-POSTEXEC-RIGHTS-V1"
STATUS = "SOURCE_RIGHTS_POSTEXEC_ADMISSION_EXECUTED_ZERO_CREDIT"
CONFIG_PATH = "configs/data/d03_languk_supreme_court_postexecution_rights_v1.json"
SWARM_CONTROL_ISSUE = 723
SWARM_ISSUE = 1379
LANE_KEY = (
    "D03|LANGUK-SUPREME-COURT|POST-REAL-EXECUTION-RIGHTS-PROVENANCE-ADMISSION|V1"
)

RIGHTS_PR = 447
RIGHTS_HEAD = "0e682d8a63719063b7661600bf46e775b58ce823"
RIGHTS_PATH = "configs/data/next100_029_languk_rights_audit_v1.json"
RIGHTS_BLOB = "c88a2884f77f2c9747b56d7f08edfdf7b6478af2"
RIGHTS_EVIDENCE_ID = "96e1c85bdeaad908689b47252832ab9674897d3a3dff3e73804ca80b492ad9be"
RIGHTS_HISTORICAL_VERDICT = "RETEST_LANGUK_COURT_DECISIONS_ONLY"
RIGHTS_CANDIDATE_VERDICT = "RETEST_PRIVACY_AND_BYTE_MATERIALIZATION"
RIGHTS_TRAINING_STATUS = (
    "RIGHTS_LAYER_PLAUSIBLY_ALLOWED_BUT_NOT_ADMITTED_UNTIL_PRIVACY_AND_EXACT_LOCAL_"
    "MATERIALIZATION_PASS"
)
RIGHTS_LICENSE = "MIT"
RIGHTS_BASIS = (
    "Ukraine Law No. 2811-IX Article 8(1)(3): official documents of judicial character "
    "are not protected by copyright"
)

PRODUCT_PR = 833
PRODUCT_HEAD = "5d38a32b37490b11f6fa77b3bcd21855f3e28605"
EXECUTION_PATH = "evidence/d03-languk-supreme-court-real-execution-v2.json"
EXECUTION_BLOB = "a1f013c4ad9dfeacf3e92cea2378767b239647a2"
RETEST_CONFIG_PATH = "configs/data/d03_languk_supreme_court_retest_v1.json"
RETEST_CONFIG_BLOB = "b4ee7b0f698660145cb3c1db0f8ccb690c255a5e"
REPAIRED_PRODUCT_HEAD = "d8136e07e5bcdd81b889a3211c7f40521af97b5c"
EXECUTION_HEAD = "75121d8408bbb2413013063829ea6ad43451719a"
WORKFLOW_RUN = 34601996401
WORKFLOW_JOB = 103271308318
INDEPENDENT_AUDIT_ISSUE = 1269
INDEPENDENT_AUDIT_STATUS = "TERMINAL_PASS"
INDEPENDENT_AUDIT_VERDICT = "PASS_FOR_INTEGRATION_MECHANICS_AND_REAL_EXECUTION"
INDEPENDENT_AUDIT_SHARED_CI = 34602453161

SOURCE = {
    "dataset": "lang-uk/court-decisions-uk",
    "revision": "2dcac4c941b87bf9c242bdc919cef4b40f4a4813",
    "file": "2024-5K-supreme-court-decisions-deduplicated.parquet",
    "bytes": 20220778,
    "sha256": "9b8870d10695715e4a0540c6f8fdca381599c0e6cdbaf3ecdf3c0782207b6597",
    "excluded_file": "250-deanonymized-court-cases.parquet",
    "family": "ua.languk.supreme-court-decisions",
}

ADMITTED_RIGHTS_CANDIDATE = {
    "retained_records": 256,
    "retained_normalized_bytes": 2809632,
    "retained_jsonl_sha256": "03bf5089bb6ff4c304e3a299e2480b263a161aad8abe831db0b196af7c585db3",
    "report_identity_sha256": "91c83f200293695c6f28085712b8398246af2422e48d26ca43a62a50568ed479",
    "config_identity_sha256": "d9e826f3d316426807aa57f381edf200970086a5d688e5b94f6da3124648421b",
    "two_clean_materializations_equal": True,
    "privacy_quality_retest_passed_for_retained_rows": True,
    "universal_pii_absence_claimed": False,
    "payload_rematerialized_in_this_package": False,
}

TRUTH_BOUNDARY = {
    "source_rights_postexecution_admitted": True,
    "source_admitted_candidate_records": 256,
    "source_admitted_candidate_bytes": 2809632,
    "canonical_capacity_credited": 0,
    "training_authorized_bytes": 0,
    "family_credit_added": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "model_training_executed": False,
    "optimizer_updates": 0,
    "learned_weights_created": False,
    "evaluation_authorized_bytes": 0,
    "final_test_payload_accessed": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
}

DOWNSTREAM_REQUIRED = [
    "current_global_exact_near_lineage_dedup_including_rada_overlap",
    "reserved_evaluation_decontamination",
    "canonical_quality_privacy_consumer",
    "family_balance_caps",
    "cluster_safe_split",
    "deterministic_packing",
    "two_clean_build_reproducibility",
    "positive_unique_loss_accounting",
    "positive_training_authority",
]

ROOT_KEYS = {
    "schema_version",
    "authority_id",
    "status",
    "execution_profile",
    "swarm",
    "rights_authority",
    "execution_authority",
    "source_scope",
    "admitted_rights_candidate",
    "proof",
    "truth_boundary",
    "downstream_required",
}


class LangUKPostExecutionRightsError(ValueError):
    """Raised when the exact rights/execution bridge no longer validates."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LangUKPostExecutionRightsError(message)


def _exact_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        if set(actual) != set(expected):
            return False
        return all(_exact_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    _require(_exact_equal(actual, expected), f"{label} drift")


def git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def _git_show_bytes(root: Path, revision: str, relpath: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relpath}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LangUKPostExecutionRightsError(
            f"cannot resolve historical authority {revision}:{relpath}"
        ) from exc
    return result.stdout


def _historical_json(root: Path, revision: str, relpath: str, blob_sha1: str) -> Any:
    raw = _git_show_bytes(root, revision, relpath)
    _require(git_blob_sha1(raw) == blob_sha1, f"historical blob drift: {relpath}")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LangUKPostExecutionRightsError(f"invalid historical JSON: {relpath}") from exc


def _find_rights_candidate(rights: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = rights.get("candidates")
    _require(isinstance(candidates, list), "historical rights candidates missing")
    matches = [
        item
        for item in candidates
        if isinstance(item, Mapping)
        and item.get("candidate_id") == "languk.court-decisions-uk.supreme-2024-5k"
    ]
    _require(len(matches) == 1, "exact LangUK Supreme Court rights candidate missing/duplicated")
    return matches[0]


def _validate_historical_rights(root: Path) -> Mapping[str, Any]:
    rights = _historical_json(root, RIGHTS_HEAD, RIGHTS_PATH, RIGHTS_BLOB)
    _require(isinstance(rights, Mapping), "historical rights authority must be an object")
    _require(rights.get("evidence_identity_sha256") == RIGHTS_EVIDENCE_ID, "rights identity drift")
    terminal = rights.get("terminal_result")
    _require(isinstance(terminal, Mapping), "historical terminal result missing")
    _require(terminal.get("verdict") == RIGHTS_HISTORICAL_VERDICT, "rights verdict drift")
    _require(terminal.get("training_source_admitted") is False, "historical rights truth widened")
    _require(terminal.get("materialized_training_bytes") == 0, "historical payload truth widened")

    candidate = _find_rights_candidate(rights)
    _require(candidate.get("verdict") == RIGHTS_CANDIDATE_VERDICT, "candidate verdict drift")
    _require(candidate.get("selected_file") == SOURCE["file"], "rights source file drift")
    _require(candidate.get("selected_file_origin_commit") == SOURCE["revision"], "rights revision drift")
    _require(candidate.get("selected_file_raw_sha256") == SOURCE["sha256"], "rights source hash drift")
    _require(candidate.get("selected_file_bytes") == SOURCE["bytes"], "rights source bytes drift")
    _require(candidate.get("license") == RIGHTS_LICENSE, "rights license-label drift")
    _require(candidate.get("training_rights") == RIGHTS_TRAINING_STATUS, "rights training status drift")
    work_rights = candidate.get("underlying_work_rights")
    _require(isinstance(work_rights, Mapping), "underlying-work rights evidence missing")
    _require(work_rights.get("basis") == RIGHTS_BASIS, "underlying-work rights basis drift")
    privacy = candidate.get("privacy")
    _require(isinstance(privacy, Mapping), "historical privacy blocker missing")
    _require(privacy.get("state") == "BLOCKED_PENDING_DETERMINISTIC_SCAN", "privacy blocker drift")
    bounded = candidate.get("bounded_subset_rule")
    _require(isinstance(bounded, Mapping), "bounded subset rule missing")
    _require(bounded.get("allowed_source_file") == SOURCE["file"], "allowed file drift")
    _require(bounded.get("exclude_file") == SOURCE["excluded_file"], "excluded file drift")
    _require(bounded.get("max_records") == 256, "historical record cap drift")
    _require(bounded.get("materialized_records_now") == 0, "historical materialization truth drift")

    rejected = {
        "languk.ubertext": "REJECT_MIXED_RIGHTS",
        "brown-uk.corpus": "REJECT_CHAIN_OF_TITLE_UNPROVEN",
        "languk.malyuk": "REJECT_MIXED_UPSTREAM_RIGHTS",
    }
    observed = {
        item.get("candidate_id"): item.get("verdict")
        for item in rights["candidates"]
        if isinstance(item, Mapping) and item.get("candidate_id") in rejected
    }
    _require_exact(observed, rejected, "non-court LangUK rejection set")
    return candidate


def _validate_repaired_execution(root: Path) -> Mapping[str, Any]:
    evidence = _historical_json(root, PRODUCT_HEAD, EXECUTION_PATH, EXECUTION_BLOB)
    config = _historical_json(root, PRODUCT_HEAD, RETEST_CONFIG_PATH, RETEST_CONFIG_BLOB)
    _require(isinstance(evidence, Mapping), "execution evidence must be an object")
    _require(isinstance(config, Mapping), "retest config must be an object")

    _require(evidence.get("product_pr") == PRODUCT_PR, "execution product PR drift")
    _require(evidence.get("repaired_product_semantic_head") == REPAIRED_PRODUCT_HEAD, "repair head drift")
    _require(evidence.get("execution_head") == EXECUTION_HEAD, "execution head drift")
    _require(evidence.get("workflow_run") == WORKFLOW_RUN, "workflow run drift")
    _require(evidence.get("workflow_job") == WORKFLOW_JOB, "workflow job drift")
    _require(evidence.get("workflow_conclusion") == "success", "execution is not terminal-success")

    source = evidence.get("source")
    _require(isinstance(source, Mapping), "execution source missing")
    expected_execution_source = {
        key: SOURCE[key]
        for key in ("dataset", "revision", "file", "bytes", "sha256", "excluded_file")
    }
    _require_exact(source, expected_execution_source, "execution source")
    config_source = config.get("source")
    _require(isinstance(config_source, Mapping), "retest config source missing")
    _require(config_source.get("dataset") == SOURCE["dataset"], "config dataset drift")
    _require(config_source.get("revision") == SOURCE["revision"], "config revision drift")
    _require(config_source.get("file") == SOURCE["file"], "config file drift")
    _require(config_source.get("bytes") == SOURCE["bytes"], "config bytes drift")
    _require(config_source.get("sha256") == SOURCE["sha256"], "config source hash drift")
    _require(config_source.get("excluded_file") == SOURCE["excluded_file"], "config exclusion drift")
    _require(config_source.get("family") == SOURCE["family"], "config family drift")

    clean = evidence.get("two_clean_execution")
    _require(isinstance(clean, Mapping), "two-clean execution proof missing")
    for key in ("acquisition_a", "materialization_a", "acquisition_b", "materialization_b"):
        _require(clean.get(key) == "success", f"{key} not successful")
    _require(
        clean.get("report_a_equals_report_b_byte_for_byte") is True,
        "report reproducibility failed",
    )
    _require(
        clean.get("retained_jsonl_a_equals_b_byte_for_byte") is True,
        "payload reproducibility failed",
    )

    measured = evidence.get("measured_result")
    _require(isinstance(measured, Mapping), "measured result missing")
    _require(measured.get("retained_records") == 256, "retained records drift")
    _require(measured.get("retained_normalized_bytes") == 2809632, "retained bytes drift")
    _require(
        measured.get("retained_jsonl_sha256")
        == ADMITTED_RIGHTS_CANDIDATE["retained_jsonl_sha256"],
        "retained hash drift",
    )
    _require(
        measured.get("report_identity_sha256")
        == ADMITTED_RIGHTS_CANDIDATE["report_identity_sha256"],
        "report identity drift",
    )
    _require(
        measured.get("config_identity_sha256")
        == ADMITTED_RIGHTS_CANDIDATE["config_identity_sha256"],
        "config identity drift",
    )
    _require(
        measured.get("anonymization_marker_prefix_consistency_validated_for_retained_rows") is True,
        "privacy marker proof missing",
    )
    _require(measured.get("universal_pii_absence_claimed") is False, "privacy claim improperly widened")
    _require(measured.get("rejected_text_emitted") is False, "rejected text was emitted")
    _require(measured.get("rejected_hashes_emitted") is False, "rejected hashes were emitted")

    old = evidence.get("historical_pre_repair_comparison")
    _require(isinstance(old, Mapping), "pre-repair comparison missing")
    _require(old.get("historical_retained_normalized_bytes") == 2813635, "historical byte count drift")
    _require(old.get("repaired_retained_normalized_bytes") == 2809632, "repaired byte count drift")

    boundary = evidence.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "execution claim boundary missing")
    zero_fields = (
        "canonical_capacity_credited",
        "family_credit_added",
        "training_authorized_bytes",
        "authorized_optimized_target_exposure",
        "optimizer_updates",
    )
    for key in zero_fields:
        _require(
            type(boundary.get(key)) is int and boundary[key] == 0,
            f"execution {key} widened",
        )
    false_fields = (
        "current_corpus_eligible",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "learned_weights_created",
        "final_test_accessed",
        "paid_compute_used",
        "foreign_pretrained_weights_used",
        "external_llm_or_api_used_for_data_or_intelligence",
    )
    for key in false_fields:
        _require(boundary.get(key) is False, f"execution {key} widened")
    return measured


def _expected_rights_authority() -> dict[str, Any]:
    return {
        "historical_pr": RIGHTS_PR,
        "historical_head_sha": RIGHTS_HEAD,
        "manifest_path": RIGHTS_PATH,
        "manifest_blob_sha1": RIGHTS_BLOB,
        "evidence_identity_sha256": RIGHTS_EVIDENCE_ID,
        "historical_verdict": RIGHTS_HISTORICAL_VERDICT,
        "candidate_verdict": RIGHTS_CANDIDATE_VERDICT,
        "rights_layer_training_status": RIGHTS_TRAINING_STATUS,
        "license_label": RIGHTS_LICENSE,
        "underlying_work_basis": RIGHTS_BASIS,
    }


def _expected_execution_authority() -> dict[str, Any]:
    return {
        "product_pr": PRODUCT_PR,
        "product_final_head_sha": PRODUCT_HEAD,
        "execution_evidence_path": EXECUTION_PATH,
        "execution_evidence_blob_sha1": EXECUTION_BLOB,
        "retest_config_path": RETEST_CONFIG_PATH,
        "retest_config_blob_sha1": RETEST_CONFIG_BLOB,
        "repaired_product_semantic_head": REPAIRED_PRODUCT_HEAD,
        "execution_head_sha": EXECUTION_HEAD,
        "workflow_run": WORKFLOW_RUN,
        "workflow_job": WORKFLOW_JOB,
        "independent_audit_issue": INDEPENDENT_AUDIT_ISSUE,
        "independent_audit_status": INDEPENDENT_AUDIT_STATUS,
        "independent_audit_target_head_sha": PRODUCT_HEAD,
        "independent_audit_verdict": INDEPENDENT_AUDIT_VERDICT,
        "independent_audit_shared_ci_run": INDEPENDENT_AUDIT_SHARED_CI,
    }


def validate_languk_postexecution_rights(
    root: Path, config_path: str | Path = CONFIG_PATH
) -> dict[str, Any]:
    """Validate exact historical rights + repaired execution and the zero-credit bridge."""
    root = root.resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    try:
        admission = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LangUKPostExecutionRightsError("cannot load admission config") from exc
    _require(isinstance(admission, Mapping), "admission config must be an object")
    _require(set(admission) == ROOT_KEYS, "admission root keys drift")
    _require(admission.get("schema_version") == SCHEMA_VERSION, "schema drift")
    _require(admission.get("authority_id") == AUTHORITY_ID, "authority id drift")
    _require(admission.get("status") == STATUS, "status drift")
    _require(admission.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    _require_exact(
        admission.get("swarm"),
        {"control_issue": SWARM_CONTROL_ISSUE, "worker_issue": SWARM_ISSUE, "lane_key": LANE_KEY},
        "swarm binding",
    )
    _require_exact(
        admission.get("rights_authority"),
        _expected_rights_authority(),
        "rights authority",
    )
    _require_exact(
        admission.get("execution_authority"),
        _expected_execution_authority(),
        "execution authority",
    )
    _require_exact(admission.get("source_scope"), SOURCE, "source scope")
    _require_exact(
        admission.get("admitted_rights_candidate"),
        ADMITTED_RIGHTS_CANDIDATE,
        "admitted rights candidate",
    )
    _require_exact(
        admission.get("proof"),
        {
            "historical_rights_manifest_blob_verified": True,
            "historical_rights_candidate_scope_verified": True,
            "repaired_execution_evidence_blob_verified": True,
            "repaired_retest_config_blob_verified": True,
            "exact_source_identity_equal_across_rights_and_execution": True,
            "historical_privacy_materialization_blocker_consumed": True,
            "other_languk_families_admitted": False,
            "deanonymized_file_admitted": False,
            "rights_layer_admitted_for_exact_repaired_candidate": True,
            "training_source_admitted": False,
        },
        "proof vector",
    )
    _require_exact(admission.get("truth_boundary"), TRUTH_BOUNDARY, "truth boundary")
    _require_exact(admission.get("downstream_required"), DOWNSTREAM_REQUIRED, "downstream gates")

    rights_candidate = _validate_historical_rights(root)
    measured = _validate_repaired_execution(root)
    _require(
        rights_candidate["selected_file_origin_commit"] == SOURCE["revision"],
        "rights/execution revision mismatch",
    )
    _require(
        rights_candidate["selected_file_raw_sha256"] == SOURCE["sha256"],
        "rights/execution hash mismatch",
    )
    _require(
        measured["retained_normalized_bytes"]
        == ADMITTED_RIGHTS_CANDIDATE["retained_normalized_bytes"],
        "admitted/executed byte mismatch",
    )
    return dict(admission)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args(argv)
    admission = validate_languk_postexecution_rights(args.repo_root, args.config)
    print(
        json.dumps(
            {
                "status": admission["status"],
                "source_rights_postexecution_admitted": True,
                "source_admitted_candidate_records": 256,
                "source_admitted_candidate_bytes": 2809632,
                "training_authorized_bytes": 0,
                "authorized_optimized_target_exposure": 0,
                "optimizer_updates": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
