"""Validate NEXT100-058 Ukrainian selection-validation V2 fail-closed authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/evaluation/next100_058_ua_selection_validation_v2.json")
MEMBERSHIP_PATH = Path("data/evaluation/eval303/selection-validation/composite-membership.jsonl")
DATA300_PATH = Path("configs/data/data300_corpus_v03_frozen_build_contract_v2.json")
EXACT_PROOF_PATH = Path("evidence/eval303/data300-exact-exclusion-proof-v1.json")

SCHEMA = "12-6.next100-058-ua-selection-validation-expansion.v2"
WORKER_ID = "NEXT100-058-UA-SELECTION-V2"
BLOCKED_STATUS = "BLOCKED_NO_NEW_TERMINAL_PURPOSE_RESERVED_UA_SOURCE_AND_NO_NEAR_COPY_TRAINING_PROOF"
EVAL290_WORKER = "EVAL-290-UA-SELECTION-VALIDATION-V1"
EXPECTED_AUTHORITY_ID = "a52179e925f1261cf2d17ec3485dc8f6de19ffd000afb5ba15fd0e58ff2e4fca"
EXPECTED_EVAL290_HEAD = "029514654829cebc149cff6fc1fea2a8ba4fa566"
EXPECTED_EVAL290_SET = "c32320a706a283049e35eb537eb20a1e7f5865b86c24397c8b73d1e3d2014164"
EXPECTED_EVAL303_MEMBERSHIP = "e4bb39dd7aa6a20c7ed34e093f563b5f4896ac16828151c6b375a83cd8a068c6"
EXPECTED_DATA300_ID = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
EXPECTED_EXACT_PROOF_ID = "ac9a0e2c3beab26c0d664b0006b11ec9fd155fa78be9f46d56ecb3ed336f2621"
EXPECTED_FAMILIES = {"kubernetes.website.docs", "lang-uk.perestoroha-ocr"}


class AuthorityError(RuntimeError):
    """Raised when immutable NEXT100-058 authority evidence drifts."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"unable to read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuthorityError(f"expected JSON object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _authority_identity(config: dict[str, Any]) -> str:
    semantic = dict(config)
    semantic.pop("authority_identity_sha256", None)
    payload = json.dumps(
        semantic,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def _load_membership(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AuthorityError(f"unable to read membership: {path}") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuthorityError(f"invalid membership JSONL line {index}") from exc
        if not isinstance(row, dict):
            raise AuthorityError(f"membership line {index} is not an object")
        rows.append(row)
    return rows


def validate(repo_root: Path) -> dict[str, Any]:
    config = _read_json(repo_root / CONFIG_PATH)
    _require(config.get("schema_version") == SCHEMA, "schema drift")
    _require(config.get("worker_id") == WORKER_ID, "worker identity drift")
    _require(config.get("execution_profile") == "LOCAL_FREE", "non-LOCAL_FREE execution profile")
    _require(config.get("training_executed") is False, "training must remain unexecuted")
    _require(config.get("status") == BLOCKED_STATUS, "fail-closed status drift")
    _require(config.get("final_test_outcomes_exposed") is False, "final-test outcomes must remain unexposed")

    identity = _authority_identity(config)
    _require(identity == EXPECTED_AUTHORITY_ID, f"authority identity drift: {identity}")
    _require(config.get("authority_identity_sha256") == identity, "committed authority self-hash drift")

    hard = config.get("hard_requirements", {})
    _require(hard.get("minimum_independent_ua_families") == 2, "UA family minimum drift")
    _require(hard.get("selection_validation_rights_exact_object_required") is True, "rights gate weakened")
    _require(hard.get("pre_training_reservation_required") is True, "reservation gate weakened")
    _require(hard.get("future_training_prohibited") is True, "future-training prohibition weakened")
    _require(hard.get("tokenizer_fit_prohibited") is True, "tokenizer-fit prohibition weakened")
    _require(hard.get("final_test_prohibited") is True, "final-test prohibition weakened")
    _require(hard.get("exact_training_overlap_allowed") is False, "exact-overlap gate weakened")
    _require(hard.get("near_copy_training_cluster_overlap_allowed") is False, "near-copy gate weakened")
    _require(hard.get("deterministic_selection_required") is True, "determinism gate weakened")
    _require(hard.get("final_test_outcome_access_allowed") is False, "final-test outcome firewall weakened")

    preserved = config.get("preserved_eval290", {})
    _require(preserved.get("head_sha") == EXPECTED_EVAL290_HEAD, "EVAL-290 head drift")
    _require(preserved.get("dedicated_workflow_run") == 32968339064, "EVAL-290 run drift")
    _require(preserved.get("dedicated_workflow_conclusion") == "success", "EVAL-290 is not terminal-success")
    _require(preserved.get("set_identity_sha256") == EXPECTED_EVAL290_SET, "EVAL-290 set identity drift")
    _require(preserved.get("records") == 8, "EVAL-290 record count drift")
    _require(preserved.get("independent_family_count") == 2, "EVAL-290 family count drift")
    _require(set(preserved.get("source_families", [])) == EXPECTED_FAMILIES, "EVAL-290 family set drift")
    _require(preserved.get("deterministic_selector") is True, "EVAL-290 determinism lost")
    _require(preserved.get("selection_only") is True, "EVAL-290 purpose drift")
    _require(preserved.get("training_eligible") is False, "EVAL-290 training prohibition lost")
    _require(preserved.get("tokenizer_fit_eligible") is False, "EVAL-290 tokenizer-fit prohibition lost")
    _require(preserved.get("final_test_eligible") is False, "EVAL-290 final-test prohibition lost")

    membership_path = repo_root / MEMBERSHIP_PATH
    membership_bytes = membership_path.read_bytes()
    membership_sha = _sha256_bytes(membership_bytes)
    _require(membership_sha == EXPECTED_EVAL303_MEMBERSHIP, "EVAL-303 membership byte identity drift")
    _require(config["eval303_binding"].get("membership_jsonl_sha256") == membership_sha, "membership binding drift")
    _require(config["eval303_binding"].get("final_test_payload_read") is False, "V2 may not read final-test payload")
    _require(config["eval303_binding"].get("final_test_outcomes_read") is False, "V2 may not read final-test outcomes")

    rows = _load_membership(membership_path)
    ua_rows = [row for row in rows if row.get("component_worker_id") == EVAL290_WORKER]
    _require(len(ua_rows) == 8, "EVAL-290 membership count is not eight")
    families = {str(row.get("source_family")) for row in ua_rows}
    _require(families == EXPECTED_FAMILIES, "EVAL-290 membership family drift")
    for row in ua_rows:
        _require(row.get("purpose") == "selection-validation", "selected row purpose drift")
        _require(row.get("selection_eligible") is True, "selected row is not selection eligible")
        _require(row.get("model_selection_eligible") is True, "model-selection eligibility drift")
        _require(row.get("tokenizer_selection_eligible") is True, "tokenizer-selection eligibility drift")
        _require(row.get("hyperparameter_selection_eligible") is True, "hyperparameter eligibility drift")
        _require(row.get("training_eligible") is False, "selected row became training eligible")
        _require(row.get("tokenizer_fit_eligible") is False, "selected row became tokenizer-fit eligible")
        _require(row.get("final_test_eligible") is False, "selected row became final-test eligible")
        _require(row.get("final_reporting_eligible") is False, "selected row became final-reporting eligible")
        _require(row.get("future_training_prohibited") is True, "future-training prohibition missing")

    data300 = _read_json(repo_root / DATA300_PATH)
    _require(data300.get("contract_identity_sha256") == EXPECTED_DATA300_ID, "DATA-300 identity drift")
    boundary = config.get("training_exclusion_boundary", {})
    _require(boundary.get("data300_contract_identity_sha256") == EXPECTED_DATA300_ID, "DATA-300 binding drift")
    _require(boundary.get("exact_overlap_count") == 0, "exact training overlap is nonzero")
    _require(boundary.get("near_copy_cluster_proof") == "ABSENT", "unexpected near-copy proof status")

    exact_proof = _read_json(repo_root / EXACT_PROOF_PATH)
    _require(exact_proof.get("proof_identity_sha256") == EXPECTED_EXACT_PROOF_ID, "EVAL-303 proof identity drift")
    _require(boundary.get("eval303_exact_exclusion_proof_identity_sha256") == EXPECTED_EXACT_PROOF_ID, "proof binding drift")
    comparisons = exact_proof.get("comparisons", {})
    _require(not comparisons.get("selected_content_vs_training_raw_or_normalized_sha256_overlap"), "exact hash overlap detected")
    verdict = exact_proof.get("verdict", {})
    _require(verdict.get("exact_byte_overlap_count") == 0, "exact byte overlap detected")
    _require(verdict.get("near_copy_or_dedup_cluster_scan_claimed") is False, "near-copy scan unexpectedly claimed")
    _require(verdict.get("wave3_data300_g07_g08_still_required") is True, "G07/G08 requirement lost")

    candidates = config.get("late_bound_candidates")
    _require(isinstance(candidates, list) and candidates, "late-bound candidate vector missing")
    _require(all(item.get("admitted_to_v2") is False for item in candidates), "candidate admitted without full gate")
    _require(all(int(item.get("new_family_credit", -1)) == 0 for item in candidates), "candidate received family credit")
    _require(all(item.get("blockers") for item in candidates), "candidate blocker list missing")

    expansion = config.get("expansion_result", {})
    _require(expansion.get("new_records_admitted") == 0, "new records were fabricated or admitted")
    _require(expansion.get("new_families_admitted") == 0, "new family was fabricated or admitted")
    _require(expansion.get("total_preserved_records") == 8, "preserved record count drift")
    _require(expansion.get("total_preserved_families") == 2, "preserved family count drift")
    _require(expansion.get("immutable_payload_rewritten") is False, "immutable payload rewrite claimed")
    _require(expansion.get("fabricated_records") == 0, "fabricated record count is nonzero")

    return {
        "schema_version": "12-6.next100-058-validation-report.v1",
        "worker_id": WORKER_ID,
        "status": BLOCKED_STATUS,
        "authority_identity_sha256": identity,
        "preserved_eval290_records": len(ua_rows),
        "preserved_eval290_families": sorted(families),
        "new_records_admitted": 0,
        "new_families_admitted": 0,
        "exact_training_overlap_count": 0,
        "near_copy_training_cluster_proof_present": False,
        "final_test_payload_read": False,
        "final_test_outcomes_read": False,
        "final_test_outcomes_exposed": False,
        "training_executed": False,
        "execution_profile": "LOCAL_FREE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.repo_root.resolve())
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
