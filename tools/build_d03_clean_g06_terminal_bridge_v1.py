#!/usr/bin/env python3
"""Bind audited clean G05/G06 physical replay to the native terminal G06 contract.

This tool is deliberately zero-credit. It does not execute training or make the
corpus eligible; it only seals already-physical replay evidence into the exact
G06 envelope/terminal schemas consumed by the incumbent composition preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_OUTPUT_ZERO_CREDIT = {
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
_REPLAY_ZERO_CREDIT = {
    **_OUTPUT_ZERO_CREDIT,
    "foreign_pretrained_weights_used": False,
    "raw_payloads_retained_in_output": False,
    "whole_corpus_external_llm_cleanliness_claimed": False,
}
_REPLAY_SCHEMA = "12-6.d03-clean-g05-g06-physical-replay.v1"
_TERMINAL_SCHEMA = "12-6.d03-clean-g05-g06-two-replay-terminal.v1"
_G06_ENVELOPE_SCHEMA = "12-6.current-survivor-g06-dependency-bound-execution.v1"
_G06_QUALIFICATION_SCHEMA = "12-6.g06-exact-byte-terminal-qualification.v1"
_REPOSITORY = "Oleksii-debug/12-6-ai."
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REPLAY_KEYS = {
    "schema_version",
    "execution_profile",
    "status",
    "physical_identity",
    "replay_projection_sha256",
    "engine_bindings",
    "retained_clean_authority",
    "physical_clean_reconstruction",
    "g05",
    "g06",
    "truth_boundary",
    "scope_note",
    "replay_identity_sha256",
}
_PHYSICAL_IDENTITY_KEYS = {
    "repository",
    "workflow_run_id",
    "workflow_run_attempt",
    "workflow_job",
    "execution_head_sha",
}
_TERMINAL_KEYS = {
    "authorized_optimized_target_exposure",
    "corpus_materialization_authority_head_sha",
    "current_corpus_eligible",
    "data526_evidence_identity_sha256",
    "deterministic_replay_projection_sha256",
    "deterministic_scientific_projection_agrees",
    "execution_profile",
    "final_test_outcomes_read",
    "foreign_pretrained_weights_used",
    "g05_execution_identity_sha256",
    "g06_execution_identity_sha256",
    "learned_weights_created",
    "optimizer_updates_executed",
    "paid_compute_used",
    "physical_jobs",
    "product_execution_head_sha",
    "raw_payloads_retained_in_output",
    "receipt_a_sha256",
    "receipt_b_sha256",
    "replay_identity_a_sha256",
    "replay_identity_b_sha256",
    "schema_version",
    "tokenizer_fit_authorized",
    "training_authorized_bytes",
    "training_executed",
    "two_distinct_physical_replay_identities",
    "whole_corpus_external_llm_cleanliness_claimed",
}


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


def _hash40(value: Any, field: str) -> str:
    _need(type(value) is str and _HEX40.fullmatch(value) is not None, f"{field} must be Git SHA-1")
    return value


def _hash64(value: Any, field: str) -> str:
    _need(type(value) is str and _HEX64.fullmatch(value) is not None, f"{field} must be SHA-256")
    return value


def _positive_int(value: Any, field: str) -> int:
    _need(type(value) is int and value > 0, f"{field} must be an exact positive integer")
    return value


def _obj(value: Any, field: str) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden in value or any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _self_hash(document: Mapping[str, Any], field: str, label: str) -> str:
    claimed = _hash64(document.get(field), f"{label}.{field}")
    core = dict(document)
    core.pop(field, None)
    _need(claimed == _sha256(_cjson(core)), f"{label} {field} self-hash mismatch")
    return claimed


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


def _validate_replay(
    receipt: Mapping[str, Any],
    *,
    label: str,
    replay_head: str,
    replay_run: int,
) -> tuple[str, str, str]:
    _need(set(receipt) == _REPLAY_KEYS, f"replay {label} schema is not closed")
    _need(receipt.get("schema_version") == _REPLAY_SCHEMA, f"replay {label} schema drift")
    _need(
        receipt.get("status") == "PHYSICAL_REPLAY_EXECUTED_ZERO_CREDIT",
        f"replay {label} status drift",
    )
    _need(receipt.get("execution_profile") == "LOCAL_FREE", f"replay {label} profile drift")
    _need(receipt.get("truth_boundary") == _REPLAY_ZERO_CREDIT, f"replay {label} truth boundary drift")
    _need(not _contains_key(receipt, "normalized_payload"), f"replay {label} retained raw payload")
    identity = _self_hash(receipt, "replay_identity_sha256", f"replay {label}")
    projection = _hash64(receipt.get("replay_projection_sha256"), f"replay {label} projection")

    physical = _obj(receipt.get("physical_identity"), f"replay {label}.physical_identity")
    _need(set(physical) == _PHYSICAL_IDENTITY_KEYS, f"replay {label} physical identity schema drift")
    _need(physical.get("repository") == _REPOSITORY, f"replay {label} repository drift")
    _need(
        _hash40(physical.get("execution_head_sha"), f"replay {label} head") == replay_head,
        f"replay {label} head drift",
    )
    _need(
        _positive_int(physical.get("workflow_run_id"), f"replay {label} workflow run") == replay_run,
        f"replay {label} run drift",
    )
    _positive_int(physical.get("workflow_run_attempt"), f"replay {label} run attempt")
    job = physical.get("workflow_job")
    _need(type(job) is str and bool(job), f"replay {label} workflow job malformed")

    for name in ("g05", "g06", "physical_clean_reconstruction", "retained_clean_authority", "engine_bindings"):
        _obj(receipt.get(name), f"replay {label}.{name}")
    g05 = _obj(receipt["g05"], f"replay {label}.g05")
    g06 = _obj(receipt["g06"], f"replay {label}.g06")
    _hash64(g05.get("execution_identity_sha256"), f"replay {label} G05 identity")
    g06_identity = _hash64(g06.get("execution_identity_sha256"), f"replay {label} G06 identity")
    _hash64(g06.get("input_rows_sha256"), f"replay {label} G06 input root")
    g06_authority = _obj(g06.get("authority"), f"replay {label}.g06.authority")
    _need(
        g06_authority.get("execution_identity_sha256") == g06_identity,
        f"replay {label} G06 wrapper/authority identity drift",
    )
    return identity, projection, job


def _validate_terminal(
    terminal: Mapping[str, Any],
    *,
    replay_a: Mapping[str, Any],
    replay_b: Mapping[str, Any],
    replay_identity_a: str,
    replay_identity_b: str,
    projection: str,
    job_a: str,
    job_b: str,
    replay_head: str,
) -> None:
    _need(set(terminal) == _TERMINAL_KEYS, "terminal summary schema is not closed")
    _need(terminal.get("schema_version") == _TERMINAL_SCHEMA, "terminal summary schema drift")
    _need(terminal.get("execution_profile") == "LOCAL_FREE", "terminal execution profile drift")
    _need(
        terminal.get("deterministic_scientific_projection_agrees") is True,
        "terminal summary does not prove deterministic agreement",
    )
    _need(
        terminal.get("two_distinct_physical_replay_identities") is True,
        "terminal summary does not prove two physical identities",
    )
    _need(
        _hash64(terminal.get("deterministic_replay_projection_sha256"), "terminal projection") == projection,
        "terminal replay projection drift",
    )
    _need(
        _hash40(terminal.get("product_execution_head_sha"), "terminal replay head") == replay_head,
        "terminal replay head drift",
    )
    _need(
        terminal.get("replay_identity_a_sha256") == replay_identity_a
        and terminal.get("replay_identity_b_sha256") == replay_identity_b,
        "terminal replay identity binding drift",
    )
    _need(terminal.get("physical_jobs") == [job_a, job_b], "terminal physical job binding drift")
    _need(
        terminal.get("receipt_a_sha256") == _sha256(_cjson(replay_a))
        and terminal.get("receipt_b_sha256") == _sha256(_cjson(replay_b)),
        "terminal replay receipt hash drift",
    )

    for field, expected in _REPLAY_ZERO_CREDIT.items():
        _need(terminal.get(field) == expected, f"terminal zero-credit truth drift: {field}")

    g05 = _obj(replay_a["g05"], "replay A.g05")
    g06 = _obj(replay_a["g06"], "replay A.g06")
    reconstruction = _obj(replay_a["physical_clean_reconstruction"], "replay A reconstruction")
    _need(
        terminal.get("g05_execution_identity_sha256") == g05.get("execution_identity_sha256"),
        "terminal G05 identity drift",
    )
    _need(
        terminal.get("g06_execution_identity_sha256") == g06.get("execution_identity_sha256"),
        "terminal G06 identity drift",
    )
    _need(
        terminal.get("data526_evidence_identity_sha256") == reconstruction.get("evidence_identity_sha256"),
        "terminal DATA526 identity drift",
    )
    _need(
        terminal.get("corpus_materialization_authority_head_sha")
        == reconstruction.get("materialization_authority_head_sha"),
        "terminal materialization authority head drift",
    )


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
    _positive_int(target_pr, "target PR")
    target_head = _hash40(target_head, "target head")
    replay_head = _hash40(replay_head, "replay head")
    replay_run = _positive_int(replay_run, "replay run")
    replay_job = _positive_int(replay_job, "replay job")
    final_run = _positive_int(final_run, "final CI run")
    final_job = _positive_int(final_job, "final CI job")
    artifact_id = _positive_int(artifact_id, "artifact id")
    artifact_zip_sha256 = _hash64(artifact_zip_sha256, "artifact ZIP")
    audit_issue = _positive_int(audit_issue, "audit issue")

    identity_a, projection_a, job_a = _validate_replay(
        replay_a, label="A", replay_head=replay_head, replay_run=replay_run
    )
    identity_b, projection_b, job_b = _validate_replay(
        replay_b, label="B", replay_head=replay_head, replay_run=replay_run
    )
    _need(identity_a != identity_b, "replay identities must be physically distinct")
    _need(projection_a == projection_b, "deterministic replay projection drift")
    _need(job_a != job_b, "physical replay jobs must differ")
    for key in (
        "g05",
        "g06",
        "physical_clean_reconstruction",
        "retained_clean_authority",
        "engine_bindings",
    ):
        _need(replay_a.get(key) == replay_b.get(key), f"{key} deterministic drift")

    _validate_terminal(
        terminal_summary,
        replay_a=replay_a,
        replay_b=replay_b,
        replay_identity_a=identity_a,
        replay_identity_b=identity_b,
        projection=projection_a,
        job_a=job_a,
        job_b=job_b,
        replay_head=replay_head,
    )

    reconstruction = _obj(replay_a["physical_clean_reconstruction"], "physical clean reconstruction")
    dependency = {
        "survivor_materialization_evidence_identity_sha256": _hash64(
            reconstruction.get("evidence_identity_sha256"), "reconstruction evidence identity"
        ),
        "survivor_record_payload_jsonl_sha256": _hash64(
            reconstruction.get("record_payload_jsonl_sha256"), "reconstruction record payload root"
        ),
        "survivor_record_inventory_digest_sha256": _hash64(
            reconstruction.get("record_inventory_digest_sha256"), "reconstruction record inventory"
        ),
        "survivor_payload_inventory_digest_sha256": _hash64(
            reconstruction.get("payload_inventory_digest_sha256"), "reconstruction payload inventory"
        ),
        "survivor_record_count": _positive_int(reconstruction.get("record_count"), "reconstruction record count"),
        "survivor_total_payload_bytes": _positive_int(
            reconstruction.get("total_payload_bytes"), "reconstruction payload bytes"
        ),
        "survivor_source_object_count": _positive_int(
            reconstruction.get("source_object_count"), "reconstruction source object count"
        ),
    }
    g06 = _obj(replay_a["g06"], "G06 replay wrapper")
    envelope_core = {
        "schema_version": _G06_ENVELOPE_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "dependency": dependency,
        "g06_input_rows_sha256": _hash64(g06.get("input_rows_sha256"), "G06 input rows root"),
        "privacy_execution_authority": _obj(g06.get("authority"), "G06 execution authority"),
        "truth_boundary": dict(_OUTPUT_ZERO_CREDIT),
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
        "g06_execution_identity_sha256": _hash64(g06.get("execution_identity_sha256"), "G06 identity"),
        "input_rows_sha256": _hash64(g06.get("input_rows_sha256"), "G06 input rows root"),
        "repeated_execution_evidence_sha256": envelope["evidence_identity_sha256"],
        "artifact_id": artifact_id,
        "artifact_zip_sha256": artifact_zip_sha256,
        "replay_record_count": dependency["survivor_record_count"],
        "replay_utf8_bytes": dependency["survivor_total_payload_bytes"],
        "replay_count": 2,
        "independent_audit_issue_number": audit_issue,
        "independent_audit_status": "PASS_FOR_INTEGRATION_RELEASED",
        "local_free_only": True,
        "head_change_invalidates": True,
        "truth_boundary": dict(_OUTPUT_ZERO_CREDIT),
    }
    qualification = {
        **qualification_core,
        "qualification_identity_sha256": _sha256(_cjson(qualification_core)),
    }
    return envelope, qualification


def _validate_output_paths(envelope_output: Path, qualification_output: Path) -> None:
    _need(
        envelope_output.resolve(strict=False) != qualification_output.resolve(strict=False),
        "envelope and qualification outputs must be distinct",
    )
    for path in (envelope_output, qualification_output):
        _need(not path.exists(), f"refusing overwrite: {path}")


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

    _validate_output_paths(args.envelope_output, args.qualification_output)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(_cjson(document))
    print(qualification["qualification_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
