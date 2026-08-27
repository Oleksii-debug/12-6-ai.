#!/usr/bin/env python3
"""Independent red-team audit for the merged MODEL-341 learned-20M launch gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

AUDIT_ID = "SWARM-754-MODEL341-20M-LAUNCH-AUTHORITY-V1"
REPOSITORY = "Oleksii-debug/12-6-ai."
BASE_MAIN_SHA = "5020afd671a3885c1b738c8b4eafe7525f630546"
PR714_HEAD_SHA = "abd9a3771e30a17ed9a956430b4d5ea1f8df8521"
PR714_MERGE_SHA = "f13e657832953b59049aa6fcbfae4b7a3c684272"
MODEL341_SHA = "e4ff486fd90802fc123bebf60eed4e59196a98df"
MODEL341_MODELSPEC_SHA256 = (
    "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
)
SOURCE_BLOB_SHA1 = "9baaa2c201f4f28d3776908cf9939bf7f22eeab5"
CONFIG_BLOB_SHA1 = "753c906ef053b997f4518ad825688dc03037ea73"
UPSTREAM_TEST_BLOB_SHA1 = "6313af878a5cda42e0d4340e6b8abf253e139a3b"
SHA40 = "a" * 40
SHA64 = "b" * 64
COMPUTE_REF = "issue:754#verified-compute-decision"
TRAINING_REF = "issue:754#verified-training-decision"

AssessmentFn = Callable[..., Any]


def _authority(*, workflow: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {
        "repository": REPOSITORY,
        "git_sha": SHA40,
        "evidence_sha256": SHA64,
        "terminal": True,
    }
    if workflow:
        value.update({"workflow_run_id": 123, "workflow_conclusion": "success"})
    return value


def _make_local_ready(packet: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(packet)
    evidence = data["evidence"]
    evidence["code"]["git_sha"] = SHA40
    evidence["corpus"].update(
        {
            "manifest_sha256": SHA64,
            "split_sha256": SHA64,
            "packing_sha256": SHA64,
            "two_clean_builds_identical": True,
            "authority": _authority(),
        }
    )
    evidence["tokenizer"].update(
        {
            "identity_sha256": SHA64,
            "decision": "BYTE_BASELINE_RETAINED",
            "authority": _authority(),
        }
    )
    evidence["loss_ledger"].update(
        {
            "identity_sha256": SHA64,
            "unique_causal_loss_positions": 1000,
            "authority": _authority(),
            "data_budget_authority": _authority(),
            "data_budget_status": "QUALIFIED",
        }
    )
    evidence["checkpoint_integrity"].update(
        {"authority": _authority(), "status": "PASS"}
    )
    evidence["evaluation"].update(
        {
            "firewall_authority": _authority(),
            "selection_validation_authority": _authority(),
            "status": "PASS",
        }
    )
    evidence["training_recipe"].update(
        {
            "authority": _authority(),
            "status": "QUALIFIED",
            "seed_count": 2,
            "config_sha256": SHA64,
            "stopping_policy_sha256": SHA64,
            "requested_unique_loss_positions": 1000,
            "requested_total_training_exposures": 1000,
            "max_exposures_per_unique_position": 1,
        }
    )
    return data


def _make_compute_ready(packet: dict[str, Any]) -> dict[str, Any]:
    data = _make_local_ready(packet)
    evidence = data["evidence"]
    evidence["bounded_pilot"].update(
        {
            "authority": _authority(),
            "status": "PASS",
            "numerics_finite": True,
            "resume_equivalent": True,
            "loss_trajectory_acceptable": True,
        }
    )
    for label in ("learned_3m", "learned_10m"):
        evidence["learned_scale_evidence"][label].update(
            {"authority": _authority(), "status": "PASS"}
        )
    evidence["cost_envelope"].update(
        {
            "authority": _authority(workflow=False),
            "status": "ESTIMATED",
            "maximum_cost_usd": 50.0,
        }
    )
    evidence["independent_audit"].update(
        {"authority": _authority(), "status": "PASS"}
    )
    return data


def _add_material_authority(data: dict[str, Any]) -> None:
    evidence = data["evidence"]
    evidence["compute_authorization"].update(
        {
            "authority": _authority(workflow=False),
            "status": "COMPUTE_AUTHORIZED",
            "decision_ref": COMPUTE_REF,
            "maximum_cost_usd": 50.0,
        }
    )
    evidence["training_authorization"].update(
        {
            "authority": _authority(workflow=False),
            "status": "TRAINING_AUTHORIZED",
            "decision_ref": TRAINING_REF,
        }
    )


def _assess(
    assess_fn: AssessmentFn,
    data: dict[str, Any],
    verified_refs: Collection[str] = (),
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        result = assess_fn(data, verified_authorization_refs=verified_refs)
    except Exception as exc:  # audit must record Product crashes rather than hide them
        return None, f"{type(exc).__name__}: {exc}"
    return result.as_dict(), None


def _finding(
    finding_id: str,
    severity: str,
    title: str,
    evidence: str,
    repair: str,
    regression: str,
) -> dict[str, str]:
    return {
        "id": finding_id,
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "owner": "R01 learned-20M readiness gate / issue #654 successor",
        "repair_direction": repair,
        "required_regression": regression,
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(report)
    unsigned.pop("report_sha256", None)
    report["report_sha256"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return report


def run_audit(packet: dict[str, Any], assess_fn: AssessmentFn) -> dict[str, Any]:
    """Execute adversarial probes against the canonical evaluator contract."""
    cases: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    verified = {COMPUTE_REF, TRAINING_REF}

    current, error = _assess(assess_fn, copy.deepcopy(packet))
    cases.append({"id": "CURRENT_PACKET", "result": current, "error": error})
    if error is not None or current is None or current["material_training_authorized"]:
        findings.append(
            _finding(
                "AUDIT754-000",
                "CRITICAL",
                "Checked-in blocked packet did not remain fail-closed",
                f"current_packet_error={error!r}; result={current!r}",
                "Restore the checked-in packet to an unambiguously blocked state.",
                "Assert all three readiness phases are false for the committed packet.",
            )
        )

    forged = _make_compute_ready(packet)
    _add_material_authority(forged)
    result, error = _assess(assess_fn, forged, verified)
    promoted = bool(result and result["material_training_authorized"])
    cases.append(
        {
            "id": "SELF_ASSERTED_SCIENTIFIC_AUTHORITIES",
            "material_training_authorized": promoted,
            "error": error,
        }
    )
    if promoted:
        findings.append(
            _finding(
                "AUDIT754-001",
                "HIGH",
                "Self-asserted scientific authority objects can satisfy the machine gate",
                "All corpus/tokenizer/ledger/checkpoint/eval/recipe/pilot/scale/audit refs use "
                "synthetic exact-looking SHAs, terminal=true and workflow_conclusion=success; "
                "with only the two out-of-packet authorization refs verified, the evaluator "
                "returns material_training_authorized=true.",
                "Require an out-of-packet verified evidence set/resolver for every scientific "
                "authority and bind role, exact SHA, evidence digest and workflow result before "
                "any readiness phase can pass.",
                "Use syntactically valid but nonexistent/stale scientific refs and assert all "
                "readiness phases fail until every referenced authority is independently verified.",
            )
        )

    nan_estimate = _make_compute_ready(packet)
    nan_estimate["evidence"]["cost_envelope"]["maximum_cost_usd"] = math.nan
    _add_material_authority(nan_estimate)
    result, error = _assess(assess_fn, nan_estimate, verified)
    promoted = bool(result and result["material_training_authorized"])
    cases.append(
        {
            "id": "NAN_ESTIMATED_MAXIMUM_COST",
            "material_training_authorized": promoted,
            "error": error,
        }
    )
    if promoted:
        findings.append(
            _finding(
                "AUDIT754-002",
                "HIGH",
                "NaN estimated cost bypasses finite positive budget enforcement",
                "maximum_cost_usd=float('nan') is numeric; comparisons with NaN are false, so "
                "the non-positive check and authorized_limit < maximum_cost check do not block.",
                "Require math.isfinite(value) and value > 0 for estimated and authorized costs.",
                "Reject NaN, +Inf and -Inf for every cost field before any "
                "compute/training authority.",
            )
        )

    nan_limit = _make_compute_ready(packet)
    _add_material_authority(nan_limit)
    nan_limit["evidence"]["compute_authorization"]["maximum_cost_usd"] = math.nan
    result, error = _assess(assess_fn, nan_limit, verified)
    promoted = bool(result and result["material_training_authorized"])
    cases.append(
        {
            "id": "NAN_AUTHORIZED_COST_LIMIT",
            "material_training_authorized": promoted,
            "error": error,
        }
    )
    if promoted:
        findings.append(
            _finding(
                "AUDIT754-003",
                "HIGH",
                "NaN authorized cost limit can satisfy material-training authorization",
                "authorized maximum_cost_usd=float('nan') passes numeric typing and "
                "nan < estimated_maximum is false, so a non-budget becomes authorization.",
                "Require a finite positive authorized limit before comparing it with the estimate.",
                "Reject NaN/+Inf/-Inf authorized limits and require limit >= finite estimate.",
            )
        )

    bool_seed = _make_compute_ready(packet)
    bool_seed["evidence"]["training_recipe"]["seed_count"] = True
    _add_material_authority(bool_seed)
    result, error = _assess(assess_fn, bool_seed, verified)
    promoted = bool(result and result["material_training_authorized"])
    cases.append(
        {
            "id": "BOOLEAN_SEED_COUNT",
            "material_training_authorized": promoted,
            "error": error,
        }
    )
    if promoted:
        findings.append(
            _finding(
                "AUDIT754-004",
                "MEDIUM",
                "Boolean seed_count is accepted as a valid seed plan",
                "Python bool is an int subtype; True < 1 is false, so seed_count=True passes.",
                "Validate seed_count with an explicit positive-int helper that rejects bool.",
                "Reject True, False, floats, strings, zero and negative seed counts.",
            )
        )

    malformed_seed = _make_compute_ready(packet)
    malformed_seed["evidence"]["training_recipe"]["seed_count"] = "2"
    result, error = _assess(assess_fn, malformed_seed)
    cases.append({"id": "STRING_SEED_COUNT", "result": result, "error": error})
    if error is not None:
        findings.append(
            _finding(
                "AUDIT754-005",
                "MEDIUM",
                "Malformed seed_count crashes instead of returning fail-closed blockers",
                f"seed_count='2' produced {error}.",
                "Type-check seed_count before comparison and append a deterministic blocker.",
                "Malformed seed_count values must return readiness=false without raising.",
            )
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "authority": {
            "repository": REPOSITORY,
            "base_main_sha": BASE_MAIN_SHA,
            "pr714_head_sha": PR714_HEAD_SHA,
            "pr714_merge_sha": PR714_MERGE_SHA,
            "issue654": 654,
            "issue714": 714,
            "model341_git_sha": MODEL341_SHA,
            "model341_modelspec_sha256": MODEL341_MODELSPEC_SHA256,
            "model341_parameter_count": 20_613_440,
            "source_blob_sha1": SOURCE_BLOB_SHA1,
            "config_blob_sha1": CONFIG_BLOB_SHA1,
            "upstream_test_blob_sha1": UPSTREAM_TEST_BLOB_SHA1,
        },
        "execution_status": "COMPLETE",
        "verdict": "CHANGES_REQUIRED" if findings else "PASS_WITH_NOTES",
        "cases": cases,
        "findings": findings,
        "not_tested": [
            "No live GitHub resolver is implemented by this auditor; source bindings were read "
            "from connected live GitHub before the audit.",
            "No model training, optimizer update, GPU provisioning or paid compute was run.",
            "No Product source was modified by this audit package.",
        ],
        "truth_boundary": {
            "local_free_only": True,
            "product_code_modified": False,
            "paid_compute_used": False,
            "model_training_executed": False,
            "foreign_pretrained_weights_used": False,
            "final_test_payload_consumed": False,
        },
    }
    return _finalize(report)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_live_checkout_audit(root: Path | None = None) -> dict[str, Any]:
    """Run against a repository checkout containing the canonical merged gate."""
    if root is None:
        root = _repo_root()
    from twelve_six.learned20m_readiness import assess_learned20m_readiness

    packet_path = root / "configs/research/r01_learned20m_launch_readiness_v1.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    return run_audit(packet, assess_learned20m_readiness)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_live_checkout_audit()
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
