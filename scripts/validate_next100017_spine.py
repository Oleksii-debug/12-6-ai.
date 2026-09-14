#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-017 learned execution spine composition."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "next100017" / "learned-execution-spine.v1.json"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_CLASSES = {"terminal_authority", "candidate", "blocked", "historical"}


def fail(message: str) -> None:
    raise SystemExit(f"NEXT100-017 learned execution spine: FAIL: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_sha40(value: object, label: str) -> None:
    require(isinstance(value, str) and bool(SHA40.fullmatch(value)), f"{label} must be a 40-hex SHA")


def require_sha256(value: object, label: str) -> None:
    require(isinstance(value, str) and bool(SHA256.fullmatch(value)), f"{label} must be a 64-hex SHA256")


def validate_selected(node: dict, label: str, *, learned: bool, mechanics_only: bool) -> None:
    require(node.get("classification") == "terminal_authority", f"{label} is not terminal_authority")
    require(node.get("learned") is learned, f"{label} learned tag mismatch")
    require(node.get("mechanics_only") is mechanics_only, f"{label} mechanics-only tag mismatch")
    head = node.get("head_sha") or node.get("authority_head_sha")
    require_sha40(head, f"{label}.head")


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    require(data.get("schema_version") == "12-6.learned-execution-spine.next100017.v1", "schema mismatch")
    require(data.get("worker_id") == "NEXT100-017-INTEGRATE282-REFRESH", "worker mismatch")
    require(data.get("repository", {}).get("physical") == "Oleksii-debug/12-6-ai.", "physical repository mismatch")

    constraints = data.get("constraints", {})
    require(constraints.get("local_free_only") is True, "LOCAL_FREE must be true")
    require(constraints.get("long_training_performed") is False, "long training must remain false")
    require(constraints.get("training_authorized") is False, "training must remain unauthorized")
    require(constraints.get("mechanical_pr_merge") is False, "mechanical PR merge must remain false")
    require(constraints.get("late_bound_exact_sha") is True, "exact-SHA late binding must remain true")

    source = data.get("source_base", {})
    require(source.get("classification") == "historical", "INTEGRATE-282 carrier must remain historical")
    require(source.get("role") == "carrier_only", "INTEGRATE-282 must be carrier only")
    require_sha40(source.get("head_sha"), "source_base.head_sha")

    policy = data.get("composition_policy", {})
    require(set(policy.get("classification_enum", [])) == ALLOWED_CLASSES, "classification enum drift")
    require(policy.get("selected_classification") == "terminal_authority", "selected class drift")
    require(set(policy.get("orthogonal_tags", [])) == {"mechanics-only", "learned"}, "orthogonal tag drift")

    vector = data.get("composition_vector", {})
    require(set(vector) == {
        "environment",
        "trainer",
        "d05_recovery",
        "learned_evidence",
        "perf_static_kv",
        "primary_20m_mechanics",
        "evaluation_fixes",
        "post_base_communication_boundary",
    }, "composition vector slots changed")

    validate_selected(vector["environment"], "environment", learned=False, mechanics_only=True)
    validate_selected(vector["trainer"], "trainer", learned=False, mechanics_only=True)
    validate_selected(vector["d05_recovery"], "d05_recovery", learned=False, mechanics_only=True)
    validate_selected(vector["perf_static_kv"], "perf_static_kv", learned=False, mechanics_only=True)
    validate_selected(vector["post_base_communication_boundary"], "post_base_communication_boundary", learned=False, mechanics_only=True)

    learned = vector["learned_evidence"]
    require(set(learned) == {"3m", "10m"}, "learned evidence must contain exactly 3m and 10m")
    validate_selected(learned["3m"], "learned_evidence.3m", learned=True, mechanics_only=False)
    validate_selected(learned["10m"], "learned_evidence.10m", learned=True, mechanics_only=False)
    require(learned["3m"].get("canonical_ladder_admission") == "blocked", "3M admission must remain blocked")
    require(learned["3m"].get("missing_independent_verifier") == "VERIFY-219", "3M blocker mismatch")
    require(learned["10m"].get("canonical_ladder_admission") == "blocked", "10M admission must remain blocked")
    require(learned["10m"].get("missing_independent_verifier") == "VERIFY-218", "10M blocker mismatch")
    require(learned["3m"].get("parameters") == 3213120, "3M parameter count drift")
    require(learned["10m"].get("parameters") == 10000640, "10M parameter count drift")
    require_sha256(learned["3m"]["evidence"].get("artifact_sha256"), "3M artifact")
    require_sha256(learned["10m"]["evidence"].get("artifact_sha256"), "10M artifact")

    primary20 = vector["primary_20m_mechanics"]
    validate_selected(primary20, "primary_20m_mechanics", learned=False, mechanics_only=True)
    require(primary20.get("source", {}).get("parameters") == 20613440, "20M parameter count drift")
    require_sha40(primary20["source"].get("model341_head_sha"), "20M MODEL-341 head")
    require_sha40(primary20["source"].get("perf347_head_sha"), "20M PERF-347 head")
    require_sha256(primary20["source"].get("model_spec_sha256"), "20M ModelSpec")
    proof20 = primary20.get("evidence", {})
    require(proof20.get("terminal_status") == "PASS", "20M committed terminal artifact is not PASS")
    require(proof20.get("training_performed") is False, "20M mechanics authority must not train")
    require(proof20.get("device") == "cpu", "20M authority must remain CPU-scoped")
    require(proof20.get("exact_head_actions_runs") == 0, "20M transport fact changed; refresh authority instead of silently rewriting")
    require_sha40(proof20.get("authority_blob"), "20M authority blob")
    require_sha40(proof20.get("independent_run_blob"), "20M run blob")
    require_sha256(proof20.get("independent_run_file_sha256"), "20M run file")

    evaluations = vector["evaluation_fixes"]
    require(set(evaluations) == {"evaluator_execution", "selection_validation"}, "evaluation roles changed")
    validate_selected(evaluations["evaluator_execution"], "evaluation.evaluator_execution", learned=False, mechanics_only=True)
    validate_selected(evaluations["selection_validation"], "evaluation.selection_validation", learned=False, mechanics_only=True)

    postbase = vector["post_base_communication_boundary"]
    require(postbase.get("schema") == "12-6.post-base.communication-consumption.v1", "post-Base schema drift")
    require(postbase.get("evidence", {}).get("optimizer_updates") == 0, "post-Base boundary optimizer update violation")
    require(postbase.get("evidence", {}).get("training_authorized") is False, "post-Base boundary training authorization violation")

    blockers = data.get("blocked_admissions", [])
    require(len(blockers) == 3, "expected exactly three admission blockers")
    require(all(item.get("classification") == "blocked" for item in blockers), "all admission blockers must be blocked")
    require(any(item.get("subject") == "20M learned promotion" for item in blockers), "20M learned-promotion blocker missing")

    for item in data.get("observed_non_selected", []):
        require(item.get("classification") in ALLOWED_CLASSES, "invalid non-selected classification")
        require_sha40(item.get("head_sha"), f"non-selected {item.get('worker')}.head")

    refresh = data.get("concurrency_refresh", {})
    require(refresh.get("performed_immediately_before_vector_write") is True, "concurrency refresh marker missing")
    require(refresh.get("vector_write_count") == 1, "composition vector must be written exactly once")
    for name, sha in refresh.get("refreshed_heads", {}).items():
        require_sha40(sha, f"refreshed head {name}")
    require(refresh.get("refreshed_heads", {}).get("NEXT100-009") == "7e3fc17aa204f647e4493861ce0817a3e7a19e98", "stale NEXT100-009 head retained")

    decision = data.get("decision", "")
    require(decision.startswith("PASS:"), "manifest decision is not PASS")

    print("NEXT100-017 learned execution spine: PASS")
    print("selected slots: environment trainer D05 learned-3M learned-10M PERF-250 20M-mechanics evaluator selection-validator POSTBASE-253")
    print("canonical learned-ladder admission remains blocked for 3M/10M; 20M remains mechanics-only")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"malformed manifest: {exc}")
