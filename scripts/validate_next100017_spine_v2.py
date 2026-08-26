#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-017B learned execution spine V2."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "next100017" / "v2" / "learned-execution-spine.v2.json"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CLASSES = {"terminal_authority", "candidate", "blocked", "historical"}
SLOTS = {
    "environment", "trainer", "d05_recovery", "learned_evidence",
    "perf_static_kv", "primary_20m_mechanics", "evaluation_fixes",
    "post_base_communication_boundary",
}


def fail(message: str) -> None:
    raise SystemExit(f"NEXT100-017B spine V2: FAIL: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha40(value: object, label: str) -> None:
    require(isinstance(value, str) and bool(SHA40.fullmatch(value)), f"{label} must be 40-hex")


def sha256(value: object, label: str) -> None:
    require(isinstance(value, str) and bool(SHA256.fullmatch(value)), f"{label} must be 64-hex")


def selected(node: dict, label: str, *, learned: bool, mechanics: bool) -> None:
    require(node.get("classification") == "terminal_authority", f"{label} is not terminal")
    sha40(node.get("head_sha"), f"{label}.head_sha")
    require(node.get("learned") is learned, f"{label} learned tag mismatch")
    require(node.get("mechanics_only") is mechanics, f"{label} mechanics tag mismatch")


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(data.get("schema_version") == "12-6.learned-execution-spine.next100017.v2", "schema mismatch")
    require(data.get("worker_id") == "NEXT100-017B-INTEGRATE282-LIVE-REFRESH", "worker mismatch")
    require(data.get("repository", {}).get("physical") == "Oleksii-debug/12-6-ai.", "repo mismatch")

    c = data.get("constraints", {})
    require(c.get("local_free_only") is True, "LOCAL_FREE required")
    require(c.get("long_training_performed") is False, "long training forbidden")
    require(c.get("training_authorized") is False, "training authorization forbidden")
    require(c.get("mechanical_pr_merge") is False, "mechanical PR merge forbidden")
    require(c.get("late_bound_exact_sha") is True, "exact-SHA late binding required")

    pred = data.get("predecessor", {})
    require(pred.get("classification") == "historical", "V1 predecessor must be historical")
    require(pred.get("terminal_at_cutoff") is True, "V1 terminal fact lost")
    require(pred.get("head_sha") == "bc121b26cafdcfa1706e2939a535d4147f7b4d10", "wrong V1 predecessor")
    require(pred.get("dedicated_runs") == [32998224265, 32998248844], "V1 terminal runs drift")

    policy = data.get("composition_policy", {})
    require(set(policy.get("classification_enum", [])) == CLASSES, "classification enum drift")
    require(set(policy.get("orthogonal_tags", [])) == {"mechanics-only", "learned"}, "tag enum drift")
    require(policy.get("selected_classification") == "terminal_authority", "selection class drift")

    v = data.get("composition_vector", {})
    require(set(v) == SLOTS, "composition slots changed")
    selected(v["environment"], "environment", learned=False, mechanics=True)
    selected(v["trainer"], "trainer", learned=False, mechanics=True)
    selected(v["d05_recovery"], "d05_recovery", learned=False, mechanics=True)
    selected(v["perf_static_kv"], "perf_static_kv", learned=False, mechanics=True)
    selected(v["post_base_communication_boundary"], "post_base_boundary", learned=False, mechanics=True)

    d05c = v["d05_recovery"]["candidate_successor"]
    require(d05c.get("classification") == "candidate", "D05 successor illegally promoted")
    require(d05c.get("head_sha") == "826ce26aad580077d4a94963bcbdf4ccc3ef3954", "D05 candidate head drift")
    require(d05c.get("exact_head_d05_run_id") == 33010092169, "D05 candidate run drift")
    require(d05c.get("exact_head_d05_status") == "queued", "D05 candidate terminality changed inside sealed vector")

    learned = v["learned_evidence"]
    require(set(learned) == {"3m", "10m"}, "learned arms changed")
    selected(learned["3m"], "learned.3m", learned=True, mechanics=False)
    selected(learned["10m"], "learned.10m", learned=True, mechanics=False)
    require(learned["3m"].get("parameters") == 3213120, "3M params drift")
    require(learned["10m"].get("parameters") == 10000640, "10M params drift")
    require(learned["3m"].get("canonical_ladder_admission") == "blocked", "3M admission illegally promoted")
    require(learned["10m"].get("canonical_ladder_admission") == "blocked", "10M admission illegally promoted")
    sha256(learned["3m"]["evidence"].get("artifact_sha256"), "3M artifact")
    sha256(learned["10m"]["evidence"].get("artifact_sha256"), "10M artifact")
    v219 = learned["3m"]["independent_verifier_candidate"]
    v218 = learned["10m"]["independent_verifier_candidate"]
    require(v219.get("classification") == "candidate" and v219.get("terminal_independent_verification") is False, "VERIFY-219 illegally promoted")
    require(v219.get("head_sha") == "ac0279602fdbd0bab0c08fa53851ba7a85138ff9", "VERIFY-219 head drift")
    require(v218.get("classification") == "candidate" and v218.get("terminal_independent_verification") is False, "VERIFY-218 illegally promoted")
    require(v218.get("head_sha") == "2ed7773fb5e7638a4ab1062c2ccd47f152cf64b0", "VERIFY-218 head drift")
    require(v218.get("dedicated_run_conclusion") == "failure", "VERIFY-218 sealed failure fact drift")

    p20 = v["primary_20m_mechanics"]
    selected(p20, "primary_20m_mechanics", learned=False, mechanics=True)
    require(p20.get("worker") == "NEXT100-077-20M-STATICKV-BATCH-STRESS", "wrong 20M authority")
    require(p20.get("head_sha") == "43b33b340640d549edd76d7bb370897a9e3424d2", "NEXT100-077 head drift")
    source = p20["source"]
    require(source.get("model341_head_sha") == "e4ff486fd90802fc123bebf60eed4e59196a98df", "MODEL-341 drift")
    require(source.get("perf347_head_sha") == "a5b32313e401fc2ec38158cfc997d6633636bfa8", "PERF-347 drift")
    require(source.get("next100009_head_sha") == "7e3fc17aa204f647e4493861ce0817a3e7a19e98", "NEXT100-009 lineage drift")
    require(source.get("parameters") == 20613440, "20M parameter drift")
    sha256(source.get("model_spec_sha256"), "20M ModelSpec")
    e20 = p20["evidence"]
    require(e20.get("workflow_run_id") == 32999460150 and e20.get("conclusion") == "success", "NEXT100-077 terminal run missing")
    require(e20.get("artifact_id") == 9619235824, "NEXT100-077 artifact id drift")
    sha256(e20.get("artifact_sha256"), "NEXT100-077 artifact")
    require(e20.get("batch_sizes") == [1, 2, 4], "20M batch stress vector drift")
    require(e20.get("static_kv_bytes") == {"1": 8388608, "2": 16777216, "4": 33554432}, "20M static KV bytes drift")
    require(e20.get("local_free") is True and e20.get("device") == "cpu", "20M LOCAL_FREE CPU boundary drift")
    require(e20.get("training_performed") is False, "20M mechanics must not train")
    require(p20["supersedes_in_slot"].get("head_sha") == "7e3fc17aa204f647e4493861ce0817a3e7a19e98", "20M predecessor drift")

    ev = v["evaluation_fixes"]
    require(set(ev) == {"evaluator_execution", "selection_validation", "post_base_communication_suite"}, "evaluation role vector drift")
    selected(ev["evaluator_execution"], "evaluation.evaluator", learned=False, mechanics=True)
    selected(ev["selection_validation"], "evaluation.selection", learned=False, mechanics=True)
    selected(ev["post_base_communication_suite"], "evaluation.post_base", learned=False, mechanics=True)
    pb_eval = ev["post_base_communication_suite"]
    require(pb_eval.get("head_sha") == "a3b598fd6951267f4d5582b874ddc265278337a4", "EVAL-354 drift")
    require(pb_eval["evidence"].get("workflow_run_id") == 32997462539 and pb_eval["evidence"].get("conclusion") == "success", "EVAL-354 terminal proof drift")
    require(pb_eval["evidence"].get("training_eligible") is False, "EVAL-354 cannot train")
    require(pb_eval["evidence"].get("optimizer_updates") == 0 and pb_eval["evidence"].get("model_executed") is False, "EVAL-354 scope violation")

    boundary = v["post_base_communication_boundary"]
    require(boundary.get("head_sha") == "f6463424b5f53152fce6e6053b705f94e03f9f06", "POSTBASE-253 boundary drift")
    require(boundary.get("schema") == "12-6.post-base.communication-consumption.v1", "POSTBASE schema drift")
    require(boundary["evidence"].get("optimizer_updates") == 0 and boundary["evidence"].get("training_authorized") is False, "POSTBASE boundary violation")
    ext = boundary.get("downstream_terminal_extensions", [])
    require(len(ext) == 1 and ext[0].get("worker") == "POSTBASE-352", "POSTBASE-352 downstream binding missing")
    require(ext[0].get("head_sha") == "d83fe9f7227112615da1f8f6e7a10f56531dbb35", "POSTBASE-352 head drift")
    require(ext[0].get("classification") == "terminal_authority" and ext[0].get("selected_as_boundary") is False, "POSTBASE-352 boundary confusion")

    blockers = data.get("blocked_admissions", [])
    require(len(blockers) == 3 and all(x.get("classification") == "blocked" for x in blockers), "blocker vector drift")
    require({x.get("subject") for x in blockers} == {"3M canonical learned-ladder admission", "10M canonical learned-ladder admission", "20M learned promotion"}, "blocker subjects drift")

    for item in data.get("observed_non_selected", []):
        require(item.get("classification") in CLASSES, "invalid non-selected classification")
        sha40(item.get("head_sha"), f"non-selected {item.get('worker')}")

    refresh = data.get("concurrency_refresh", {})
    require(refresh.get("performed_immediately_before_vector_write") is True, "concurrency refresh missing")
    require(refresh.get("vector_write_count") == 1, "V2 composition must be written once")
    require(refresh.get("refreshed_heads", {}).get("NEXT100-077") == "43b33b340640d549edd76d7bb370897a9e3424d2", "stale NEXT100-077 refresh")
    for name, value in refresh.get("refreshed_heads", {}).items():
        sha40(value, f"refresh {name}")

    require(data.get("decision", "").startswith("PASS:"), "decision not PASS")
    print("NEXT100-017B learned execution spine V2: PASS")
    print("20M terminal mechanics: NEXT100-077 B1/B2/B4; learned=false")
    print("3M/10M producer evidence retained; independent admission remains blocked")
    print("POSTBASE-253 remains boundary; POSTBASE-352/EVAL-354 remain downstream scoped terminal authorities")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"malformed manifest: {exc}")
