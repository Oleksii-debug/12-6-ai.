#!/usr/bin/env python3
"""Apply incumbent quality-granularity and privacy V3 to Rada_Trees candidates.

This is deliberately a thin orchestration seam. It does not implement quality or
privacy algorithms. Instead it late-binds the incumbent reusable mechanics from
`twelve_six.data.quality_granularity` (#840 lineage) and
`twelve_six.data.privacy_filter_v3` (#849 lineage), fails closed when either is
unavailable, preserves source-native record identity, and emits only zero-credit
candidate output plus hash-safe evidence.

Important: this gate does not own the independent Ukrainian language authority
from #917. Therefore it may mark quality/privacy complete for retained records,
but it must never promote the combined language+quality+privacy completion bit.
That combined bit can be set only by a later convergence step that binds both
independent authorities to the same immutable inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "12-6.d03-rada-trees-quality-privacy-gate.v2"
EXPECTED_FAMILY = "ua.rada.open-data.plenary-transcripts"
EXPECTED_DATASET = "uacorpus/Rada_Trees"
EXPECTED_REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
QUALITY_MODULE = "twelve_six.data.quality_granularity"
PRIVACY_MODULE = "twelve_six.data.privacy_filter_v3"


class GateError(RuntimeError):
    """Fail-closed gate error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_mechanics() -> tuple[Callable[..., Any], Callable[..., Any], Callable[[], Mapping[str, Any]]]:
    try:
        quality = importlib.import_module(QUALITY_MODULE)
        privacy = importlib.import_module(PRIVACY_MODULE)
    except ImportError as exc:
        raise GateError(
            "incumbent quality/privacy mechanics are not composed on this lineage"
        ) from exc
    apply_quality = getattr(quality, "apply_frozen_granularity", None)
    scan_privacy = getattr(privacy, "hash_safe_scan", None)
    privacy_policy = getattr(privacy, "policy_manifest", None)
    require(callable(apply_quality), "quality granularity API missing")
    require(callable(scan_privacy), "privacy hash-safe scan API missing")
    require(callable(privacy_policy), "privacy policy manifest API missing")
    return apply_quality, scan_privacy, privacy_policy


def _read_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    require(path.is_file() and not path.is_symlink(), "candidate JSONL must be a regular file")
    raw = path.read_bytes()
    require(raw, "candidate JSONL is empty")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(raw.splitlines(), 1):
        require(line.strip(), f"blank JSONL line at {line_no}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"invalid JSONL at line {line_no}") from exc
        require(isinstance(row, dict), f"record {line_no} must be an object")
        record_id = row.get("record_id")
        text = row.get("text")
        require(isinstance(record_id, str) and record_id, f"record_id missing at line {line_no}")
        require(record_id not in seen_ids, f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        require(isinstance(text, str), f"text missing at line {line_no}")
        require(row.get("source_family") == EXPECTED_FAMILY, f"family drift: {record_id}")
        require(row.get("source_dataset") == EXPECTED_DATASET, f"dataset drift: {record_id}")
        require(row.get("source_revision") == EXPECTED_REVISION, f"revision drift: {record_id}")
        require(row.get("training_eligible") is False, f"premature training credit: {record_id}")
        require(row.get("evaluation_eligible") is False, f"premature evaluation credit: {record_id}")
        require(
            row.get("language_quality_privacy_complete") is False,
            f"pre-completed combined gate flag is forbidden: {record_id}",
        )
        require(
            row.get("quality_privacy_complete", False) is False,
            f"pre-completed quality/privacy flag is forbidden: {record_id}",
        )
        records.append(row)
    return records, sha256_bytes(raw)


def _privacy_evidence(scan: Any) -> dict[str, Any]:
    evidence = getattr(scan, "evidence", None)
    require(callable(evidence), "privacy scan does not expose hash-safe evidence")
    value = evidence()
    require(isinstance(value, dict), "privacy evidence must be an object")
    forbidden = {"text", "preview", "matched_value", "matched_values", "match_hash"}
    require(not forbidden.intersection(value), "privacy evidence contains forbidden payload fields")
    return value


def run_gate(candidate_jsonl: Path, output_jsonl: Path, report_path: Path) -> dict[str, Any]:
    apply_quality, scan_privacy, privacy_policy = _load_mechanics()
    records, input_sha256 = _read_records(candidate_jsonl)
    policy = privacy_policy()
    require(isinstance(policy, Mapping), "privacy policy manifest must be a mapping")
    privacy_policy_sha = policy.get("policy_sha256")
    require(isinstance(privacy_policy_sha, str) and len(privacy_policy_sha) == 64, "privacy policy identity missing")

    kept: list[dict[str, Any]] = []
    quality_statuses: Counter[str] = Counter()
    privacy_actions: Counter[str] = Counter()
    quality_rejected_utf8_bytes = 0
    privacy_blocked_records = 0
    privacy_redact_records = 0

    for row in records:
        record_id = str(row["record_id"])
        text = str(row["text"])
        quality = apply_quality(record_id, text, "uk")
        require(isinstance(quality, Mapping), f"quality result malformed: {record_id}")
        status = quality.get("status")
        require(status in {"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"}, f"unknown quality status: {record_id}")
        quality_statuses[str(status)] += 1
        rejected_bytes = quality.get("rejected_utf8_bytes")
        require(isinstance(rejected_bytes, int) and rejected_bytes >= 0, f"quality byte accounting malformed: {record_id}")
        quality_rejected_utf8_bytes += rejected_bytes

        privacy = scan_privacy(text)
        privacy_ev = _privacy_evidence(privacy)
        action = privacy_ev.get("action")
        require(action in {"ALLOW", "REDACT", "QUARANTINE", "EXCLUDE"}, f"unknown privacy action: {record_id}")
        privacy_actions[str(action)] += 1

        # This seam is intentionally conservative. REDACT is not automatically
        # materialized because doing so would require payload mutation semantics
        # owned by the privacy lineage. Anything other than ALLOW is held.
        if action != "ALLOW":
            privacy_blocked_records += 1
            privacy_redact_records += int(action == "REDACT")
            continue
        if status != "RETAIN_ALL":
            # RETAIN_PARTIAL needs quality-window payload materialization. Until
            # that exact transformation is composed, hold it rather than silently
            # changing source-native text or granting bytes.
            continue

        clean = dict(row)
        clean["quality_privacy_complete"] = True
        # Language is independently terminalized by #917. This gate does not bind
        # that evidence, so the combined completion bit must remain false.
        clean["language_quality_privacy_complete"] = False
        clean["quality_gate_status"] = "RETAIN_ALL"
        clean["privacy_gate_action"] = "ALLOW"
        clean["training_eligible"] = False
        clean["evaluation_eligible"] = False
        kept.append(clean)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    partial = output_jsonl.with_suffix(output_jsonl.suffix + ".partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with partial.open("wb") as handle:
            for row in kept:
                line = canonical_bytes(row)
                handle.write(line)
                digest.update(line)
        partial.replace(output_jsonl)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "input_candidate_jsonl_sha256": input_sha256,
        "input_records": len(records),
        "output_records": len(kept),
        "output_jsonl_sha256": digest.hexdigest(),
        "quality": {
            "mechanics": QUALITY_MODULE,
            "statuses": dict(sorted(quality_statuses.items())),
            "rejected_utf8_bytes": quality_rejected_utf8_bytes,
            "partial_records_are_held_not_materialized": True,
        },
        "privacy": {
            "mechanics": PRIVACY_MODULE,
            "policy_sha256": privacy_policy_sha,
            "actions": dict(sorted(privacy_actions.items())),
            "blocked_records": privacy_blocked_records,
            "redact_records_held_not_materialized": privacy_redact_records,
            "evidence_is_hash_safe": True,
        },
        "claim_boundary": {
            "quality_privacy_complete_only_for_output_records": True,
            "language_authority_bound": False,
            "language_quality_privacy_complete": False,
            "combined_gate_requires_independent_language_authority": True,
            "candidate_jsonl_is_canonical_corpus": False,
            "global_dedup_complete": False,
            "reserved_evaluation_decontamination_complete": False,
            "family_caps_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
        "safe_result": "RADA_TREES_QUALITY_PRIVACY_ZERO_CREDIT",
    }
    report = {**core, "report_sha256": sha256_bytes(canonical_bytes(core))}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_gate(args.candidate_jsonl, args.output_jsonl, args.report)
        print("D03_RADA_TREES_QUALITY_PRIVACY=PASS_ZERO_CREDIT")
        print("OUTPUT_RECORDS=" + str(report["output_records"]))
        print("REPORT_SHA256=" + report["report_sha256"])
        return 0
    except (GateError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
