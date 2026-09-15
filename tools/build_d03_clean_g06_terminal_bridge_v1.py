#!/usr/bin/env python3
"""Bind the audited clean G05/G06 physical replay to the native terminal G06 contract.

This tool is deliberately zero-credit.  It does not execute training or make the
corpus eligible; it only seals already-physical replay evidence into the exact
G06 envelope/terminal schemas consumed by the incumbent composition preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_ZERO_CREDIT = {
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
}
_REPLAY_SCHEMA = "12-6.d03-clean-g05-g06-physical-replay.v1"
_TERMINAL_SCHEMA = "12-6.d03-clean-g05-g06-two-replay-terminal.v1"
_G06_ENVELOPE_SCHEMA = "12-6.current-survivor-g06-dependency-bound-execution.v1"
_G06_QUALIFICATION_SCHEMA = "12-6.g06-exact-byte-terminal-qualification.v1"


class BridgeError(ValueError):
    """Raised when physical replay evidence cannot be terminally bound."""


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BridgeError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    _need(isinstance(document, dict), f"{path} must contain a JSON object")
    return document


def build_bridge(
    replay_a: dict[str, Any],
    replay_b: dict[str, Any],
    terminal_summary: dict[str, Any],
    *,
    target_pr: int,
    target_head: str,
    replay_head: str,
    replay_run: int,
    replay_job: int,
    final_run: int,
    final_job: int,
    artifact_id: int,
    artifact_zip_sha256: str,
    audit_issue: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a dependency-bound G06 envelope and zero-credit qualification."""
    for receipt, label in ((replay_a, "A"), (replay_b, "B")):
        _need(receipt.get("schema_version") == _REPLAY_SCHEMA, f"replay {label} schema drift")
        _need(
            receipt.get("status") == "PHYSICAL_REPLAY_EXECUTED_ZERO_CREDIT",
            f"replay {label} status drift",
        )
        _need(receipt.get("execution_profile") == "LOCAL_FREE", f"replay {label} profile drift")
        _need(
            receipt.get("truth_boundary", {}).get("authorized_optimized_target_exposure") == 0,
            f"replay {label} optimized exposure widened",
        )
        _need("normalized_payload" not in receipt, f"replay {label} retained raw payload")
        physical = receipt.get("physical_identity", {})
        _need(physical.get("execution_head_sha") == replay_head, f"replay {label} head drift")
        _need(physical.get("workflow_run_id") == replay_run, f"replay {label} run drift")

    _need(
        replay_a.get("replay_identity_sha256") != replay_b.get("replay_identity_sha256"),
        "replay identities must be physically distinct",
    )
    _need(
        replay_a.get("replay_projection_sha256") == replay_b.get("replay_projection_sha256"),
        "deterministic replay projection drift",
    )
    _need(
        replay_a["physical_identity"].get("workflow_job")
        != replay_b["physical_identity"].get("workflow_job"),
        "physical replay jobs must differ",
    )
    for key in (
        "g05",
        "g06",
        "physical_clean_reconstruction",
        "retained_clean_authority",
        "engine_bindings",
    ):
        _need(replay_a.get(key) == replay_b.get(key), f"{key} deterministic drift")

    _need(
        terminal_summary.get("schema_version") == _TERMINAL_SCHEMA,
        "terminal summary schema drift",
    )
    _need(
        terminal_summary.get("deterministic_scientific_projection_agrees") is True,
        "terminal summary does not prove deterministic agreement",
    )
    _need(
        terminal_summary.get("two_distinct_physical_replay_identities") is True,
        "terminal summary does not prove two physical identities",
    )
    _need(
        terminal_summary.get("deterministic_replay_projection_sha256")
        == replay_a["replay_projection_sha256"],
        "terminal replay projection drift",
    )
    _need(
        terminal_summary.get("product_execution_head_sha") == replay_head,
        "terminal replay head drift",
    )
    _need(
        terminal_summary.get("g05_execution_identity_sha256")
        == replay_a["g05"]["execution_identity_sha256"],
        "terminal G05 identity drift",
    )
    _need(
        terminal_summary.get("g06_execution_identity_sha256")
        == replay_a["g06"]["execution_identity_sha256"],
        "terminal G06 identity drift",
    )

    reconstruction = replay_a["physical_clean_reconstruction"]
    dependency = {
        "survivor_materialization_evidence_identity_sha256": reconstruction[
            "evidence_identity_sha256"
        ],
        "survivor_record_payload_jsonl_sha256": reconstruction[
            "record_payload_jsonl_sha256"
        ],
        "survivor_record_inventory_digest_sha256": reconstruction[
            "record_inventory_digest_sha256"
        ],
        "survivor_payload_inventory_digest_sha256": reconstruction[
            "payload_inventory_digest_sha256"
        ],
        "survivor_record_count": reconstruction["record_count"],
        "survivor_total_payload_bytes": reconstruction["total_payload_bytes"],
        "survivor_source_object_count": reconstruction["source_object_count"],
    }
    envelope_core = {
        "schema_version": _G06_ENVELOPE_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "dependency": dependency,
        "g06_input_rows_sha256": replay_a["g06"]["input_rows_sha256"],
        "privacy_execution_authority": replay_a["g06"]["authority"],
        "truth_boundary": dict(_ZERO_CREDIT),
    }
    envelope = {
        **envelope_core,
        "evidence_identity_sha256": _sha256(_cjson(envelope_core)),
    }

    qualification_core = {
        "schema": _G06_QUALIFICATION_SCHEMA,
        "status": "PASS_FOR_G06_TERMINAL_CONSUMPTION",
        "target_pr_number": target_pr,
        "target_head_git_sha": target_head,
        "real_replay_head_git_sha": replay_head,
        "real_replay_run_id": replay_run,
        "real_replay_job_id": replay_job,
        "final_head_ci_run_id": final_run,
        "final_head_ci_job_id": final_job,
        "g06_envelope_identity_sha256": envelope["evidence_identity_sha256"],
        "g06_execution_identity_sha256": replay_a["g06"]["execution_identity_sha256"],
        "input_rows_sha256": replay_a["g06"]["input_rows_sha256"],
        "repeated_execution_evidence_sha256": envelope["evidence_identity_sha256"],
        "artifact_id": artifact_id,
        "artifact_zip_sha256": artifact_zip_sha256,
        "replay_record_count": reconstruction["record_count"],
        "replay_utf8_bytes": reconstruction["total_payload_bytes"],
        "replay_count": 2,
        "independent_audit_issue_number": audit_issue,
        "independent_audit_status": "PASS_FOR_INTEGRATION_RELEASED",
        "local_free_only": True,
        "head_change_invalidates": True,
        "truth_boundary": dict(_ZERO_CREDIT),
    }
    qualification = {
        **qualification_core,
        "qualification_identity_sha256": _sha256(_cjson(qualification_core)),
    }
    return envelope, qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-a", type=Path, required=True)
    parser.add_argument("--replay-b", type=Path, required=True)
    parser.add_argument("--terminal-summary", type=Path, required=True)
    parser.add_argument("--target-pr", type=int, required=True)
    parser.add_argument("--target-head", required=True)
    parser.add_argument("--replay-head", required=True)
    parser.add_argument("--replay-run", type=int, required=True)
    parser.add_argument("--replay-job", type=int, required=True)
    parser.add_argument("--final-run", type=int, required=True)
    parser.add_argument("--final-job", type=int, required=True)
    parser.add_argument("--artifact-id", type=int, required=True)
    parser.add_argument("--artifact-zip-sha256", required=True)
    parser.add_argument("--audit-issue", type=int, required=True)
    parser.add_argument("--envelope-output", type=Path, required=True)
    parser.add_argument("--qualification-output", type=Path, required=True)
    args = parser.parse_args()

    envelope, qualification = build_bridge(
        _read_json(args.replay_a),
        _read_json(args.replay_b),
        _read_json(args.terminal_summary),
        target_pr=args.target_pr,
        target_head=args.target_head,
        replay_head=args.replay_head,
        replay_run=args.replay_run,
        replay_job=args.replay_job,
        final_run=args.final_run,
        final_job=args.final_job,
        artifact_id=args.artifact_id,
        artifact_zip_sha256=args.artifact_zip_sha256,
        audit_issue=args.audit_issue,
    )
    for path, document in (
        (args.envelope_output, envelope),
        (args.qualification_output, qualification),
    ):
        _need(not path.exists(), f"refusing overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(_cjson(document))
    print(qualification["qualification_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
