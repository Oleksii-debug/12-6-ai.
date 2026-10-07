#!/usr/bin/env python3
"""Materialize #840-accepted Rada_Trees quality windows, then apply privacy V3.

This is a zero-credit successor to #916's intentionally held RETAIN_PARTIAL path.
It reuses incumbent #840/#849 policy mechanics and adds no quality or privacy rules.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-rada-trees-quality-window-materialization.v1"
EXPECTED_FAMILY = "ua.rada.open-data.plenary-transcripts"
EXPECTED_DATASET = "uacorpus/Rada_Trees"
EXPECTED_REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
EXPECTED_ARCHIVE = "rada_xtag_texts.7z"
EXPECTED_RIGHTS_STATUS = "RIGHTS_SCOPE_SUPPORTED_DATED_PARLIAMENT_TRANSCRIPT_CANDIDATE"
QUALITY_MODULE = "twelve_six.data.quality_granularity"
PRIVACY_MODULE = "twelve_six.data.privacy_filter_v3"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class MaterializationError(RuntimeError):
    """Fail-closed quality-window materialization error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.casefold())
    )


def _load_mechanics() -> tuple[
    Callable[..., Any],
    Callable[[], Any],
    Callable[..., Any],
    Callable[[], Mapping[str, Any]],
]:
    try:
        quality = importlib.import_module(QUALITY_MODULE)
        privacy = importlib.import_module(PRIVACY_MODULE)
    except ImportError as exc:
        raise MaterializationError(
            "incumbent quality/privacy mechanics are not composed on this lineage"
        ) from exc
    apply_quality = getattr(quality, "apply_frozen_granularity", None)
    quality_policy = getattr(quality, "frozen_granularity_policy", None)
    scan_privacy = getattr(privacy, "hash_safe_scan", None)
    privacy_policy = getattr(privacy, "policy_manifest", None)
    require(callable(apply_quality), "quality granularity API missing")
    require(callable(quality_policy), "quality granularity policy API missing")
    require(callable(scan_privacy), "privacy hash-safe scan API missing")
    require(callable(privacy_policy), "privacy policy manifest API missing")
    return apply_quality, quality_policy, scan_privacy, privacy_policy


def _policy_sha(provider: Callable[[], Any], label: str) -> str:
    value = provider()
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        value = manifest()
    require(isinstance(value, Mapping), f"{label} policy manifest must be a mapping")
    policy_sha = value.get("policy_sha256")
    require(_is_sha256(policy_sha), f"{label} policy identity missing")
    return str(policy_sha).casefold()


def _privacy_evidence(scan: Any, payload: str) -> Mapping[str, Any]:
    evidence = getattr(scan, "evidence", None)
    require(callable(evidence), "privacy scan does not expose hash-safe evidence")
    value = evidence()
    require(isinstance(value, Mapping), "privacy evidence must be an object")
    forbidden = {
        "text",
        "raw_text",
        "preview",
        "match",
        "matched_value",
        "matched_values",
        "value",
        "secret_value",
        "match_hash",
    }

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                require(
                    str(key).casefold() not in forbidden,
                    f"privacy evidence contains forbidden payload field: {key}",
                )
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
    raw = payload.encode("utf-8")
    require(value.get("input_sha256") == sha256_bytes(raw), "privacy input SHA binding drift")
    require(value.get("input_bytes") == len(raw), "privacy input byte binding drift")
    return value


def _validate_paths(candidate: Path, output: Path, report: Path) -> None:
    require(candidate.is_file() and not candidate.is_symlink(), "candidate must be regular file")
    require(not output.is_symlink(), "output JSONL path must not be a symlink")
    require(not report.is_symlink(), "report path must not be a symlink")
    resolved = {path.resolve(strict=False) for path in (candidate, output, report)}
    require(len(resolved) == 3, "candidate, output, and report paths must be distinct")
    require(not output.exists(), "output JSONL already exists")
    require(not report.exists(), "report already exists")


def _candidate_sha(path: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            total += len(chunk)
    require(total > 0, "candidate JSONL is empty")
    return digest.hexdigest()


def _validate_row(row: Mapping[str, Any], line_no: int) -> tuple[str, str]:
    record_id = row.get("record_id")
    text = row.get("text")
    require(isinstance(record_id, str) and record_id, f"record_id missing at line {line_no}")
    require(isinstance(text, str), f"text missing at line {line_no}")
    require(row.get("source_family") == EXPECTED_FAMILY, f"family drift: {record_id}")
    require(row.get("source_dataset") == EXPECTED_DATASET, f"dataset drift: {record_id}")
    require(row.get("source_revision") == EXPECTED_REVISION, f"revision drift: {record_id}")
    require(row.get("source_archive") == EXPECTED_ARCHIVE, f"archive drift: {record_id}")
    require(
        row.get("rights_scope_status") == EXPECTED_RIGHTS_STATUS,
        f"rights-scope status drift: {record_id}",
    )
    require(row.get("attribution_required") is True, f"attribution boundary drift: {record_id}")
    require(row.get("training_eligible") is False, f"premature training credit: {record_id}")
    require(row.get("evaluation_eligible") is False, f"premature evaluation credit: {record_id}")
    require(row.get("global_dedup_complete") is False, f"pre-completed dedup flag: {record_id}")
    require(
        row.get("reserved_evaluation_decontamination_complete") is False,
        f"pre-completed decontamination flag: {record_id}",
    )
    require(
        row.get("language_quality_privacy_complete") is False,
        f"pre-completed combined gate flag is forbidden: {record_id}",
    )
    require(
        row.get("quality_privacy_complete", False) is False,
        f"pre-completed quality/privacy flag is forbidden: {record_id}",
    )
    raw = text.encode("utf-8")
    require(
        row.get("decoded_text_utf8_sha256") == sha256_bytes(raw),
        f"decoded text SHA drift: {record_id}",
    )
    decoded_bytes = row.get("decoded_text_utf8_bytes")
    require(
        type(decoded_bytes) is int and decoded_bytes == len(raw),
        f"decoded text byte drift: {record_id}",
    )
    return record_id, text


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_no, line in enumerate(handle, 1):
            require(line.strip(), f"blank JSONL line at {line_no}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializationError(f"invalid JSONL at line {line_no}") from exc
            require(isinstance(row, dict), f"record {line_no} must be an object")
            record_id, _ = _validate_row(row, line_no)
            require(record_id not in seen, f"duplicate record_id: {record_id}")
            seen.add(record_id)
            yield row


def _quality_units(
    record_id: str, text: str, quality: Mapping[str, Any]
) -> tuple[str, list[tuple[int | None, str]], int, int]:
    require(quality.get("record_id") == record_id, f"quality record binding drift: {record_id}")
    require(quality.get("mode") == "uk", f"quality mode binding drift: {record_id}")
    require(
        quality.get("family_eviction_authority") is False,
        f"quality family-eviction authority drift: {record_id}",
    )
    status = quality.get("status")
    require(
        status in {"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"},
        f"unknown quality status: {record_id}",
    )
    total = len(text.encode("utf-8"))
    retained = quality.get("retained_utf8_bytes")
    rejected = quality.get("rejected_utf8_bytes")
    require(type(retained) is int and retained >= 0, f"bad retained bytes: {record_id}")
    require(type(rejected) is int and rejected >= 0, f"bad rejected bytes: {record_id}")
    require(retained + rejected == total, f"quality byte accounting mismatch: {record_id}")
    windows = quality.get("windows")
    require(isinstance(windows, list), f"quality windows malformed: {record_id}")

    if status == "RETAIN_ALL":
        require(retained == total and rejected == 0, f"RETAIN_ALL byte drift: {record_id}")
        return status, [(None, text)], retained, rejected
    if status == "REJECT_DOCUMENT":
        require(retained == 0 and rejected == total, f"REJECT_DOCUMENT byte drift: {record_id}")
        return status, [], retained, rejected

    require(
        quality.get("authoritative_unit") == "BOUNDED_NATURAL_LANGUAGE_WINDOW",
        f"partial result is not authoritative quality windows: {record_id}",
    )
    require(windows, f"partial result has no windows: {record_id}")
    accepted: list[tuple[int | None, str]] = []
    reconstructed: list[str] = []
    observed_retained = 0
    observed_rejected = 0
    expected_start = 0
    for expected_index, window in enumerate(windows):
        require(isinstance(window, Mapping), f"window malformed: {record_id}")
        index = window.get("index")
        start = window.get("start_char")
        end = window.get("end_char")
        nbytes = window.get("utf8_bytes")
        decision = window.get("decision")
        require(type(index) is int and index == expected_index, f"window index drift: {record_id}")
        require(type(start) is int and type(end) is int, f"window span malformed: {record_id}")
        require(
            start == expected_start and start < end <= len(text),
            f"window partition drift: {record_id}",
        )
        require(type(nbytes) is int and nbytes >= 0, f"window bytes malformed: {record_id}")
        require(window.get("authoritative") is True, f"window is not authoritative: {record_id}")
        require(isinstance(decision, Mapping), f"window decision missing: {record_id}")
        accepted_flag = decision.get("accepted")
        require(type(accepted_flag) is bool, f"window decision malformed: {record_id}")
        payload = text[start:end]
        actual = len(payload.encode("utf-8"))
        require(actual == nbytes, f"window byte binding drift: {record_id}")
        reconstructed.append(payload)
        if accepted_flag:
            observed_retained += actual
            accepted.append((index, payload))
        else:
            observed_rejected += actual
        expected_start = end
    require(expected_start == len(text), f"quality windows do not cover document: {record_id}")
    require("".join(reconstructed) == text, f"quality reconstruction drift: {record_id}")
    require(observed_retained == retained, f"retained window byte drift: {record_id}")
    require(observed_rejected == rejected, f"rejected window byte drift: {record_id}")
    require(accepted and observed_rejected > 0, f"RETAIN_PARTIAL vector drift: {record_id}")
    return status, accepted, retained, rejected


def _survivor(
    parent: Mapping[str, Any],
    payload: str,
    *,
    status: str,
    window_index: int | None,
    quality_policy_sha: str,
    privacy_policy_sha: str,
) -> dict[str, Any]:
    raw = payload.encode("utf-8")
    payload_sha = sha256_bytes(raw)
    parent_id = str(parent["record_id"])
    if window_index is None:
        record_id = parent_id
        unit_kind = "SOURCE_NATIVE_DOCUMENT"
    else:
        record_id = f"{parent_id}#quality-window-{window_index:04d}:{payload_sha}"
        unit_kind = "BOUNDED_NATURAL_LANGUAGE_WINDOW"
    row = dict(parent)
    row.update(
        {
            "record_id": record_id,
            "quality_parent_record_id": parent_id,
            "quality_parent_decoded_text_utf8_sha256": parent["decoded_text_utf8_sha256"],
            "quality_parent_decoded_text_utf8_bytes": parent["decoded_text_utf8_bytes"],
            "quality_unit_kind": unit_kind,
            "quality_window_index": window_index,
            "quality_unit_utf8_sha256": payload_sha,
            "quality_unit_utf8_bytes": len(raw),
            "decoded_text_utf8_sha256": payload_sha,
            "decoded_text_utf8_bytes": len(raw),
            "quality_gate_status": status,
            "quality_granularity_policy_sha256": quality_policy_sha,
            "privacy_gate_action": "ALLOW",
            "privacy_policy_sha256": privacy_policy_sha,
            "privacy_scan_input_sha256": payload_sha,
            "privacy_scan_input_bytes": len(raw),
            "quality_privacy_complete": True,
            "language_quality_privacy_complete": False,
            "training_eligible": False,
            "evaluation_eligible": False,
            "text": payload,
        }
    )
    return row


def materialize(
    candidate: Path,
    output: Path,
    report_path: Path,
    *,
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    _validate_paths(candidate, output, report_path)
    require(_is_sha256(expected_candidate_sha256), "expected candidate SHA-256 is malformed")
    expected_candidate_sha256 = expected_candidate_sha256.casefold()
    input_sha = _candidate_sha(candidate)
    require(
        input_sha == expected_candidate_sha256,
        "candidate JSONL SHA-256 does not match expected upstream authority",
    )
    apply_quality, quality_policy, scan_privacy, privacy_policy = _load_mechanics()
    quality_policy_sha = _policy_sha(quality_policy, "quality granularity")
    privacy_policy_sha = _policy_sha(privacy_policy, "privacy")

    quality_statuses: Counter[str] = Counter()
    privacy_actions: Counter[str] = Counter()
    detector_counts: Counter[str] = Counter()
    input_records = input_text_bytes = 0
    retained_bytes = rejected_bytes = retained_units = 0
    partial_records = partial_units = 0
    held_units = held_bytes = 0
    output_records = output_jsonl_bytes = output_text_bytes = 0
    output_sha = hashlib.sha256()

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    report_partial = report_path.with_suffix(report_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    report_partial.unlink(missing_ok=True)
    published_output = False
    try:
        with partial.open("wb") as handle:
            for parent in _iter_records(candidate):
                input_records += 1
                text = str(parent["text"])
                input_text_bytes += int(parent["decoded_text_utf8_bytes"])
                parent_id = str(parent["record_id"])
                quality = apply_quality(parent_id, text, "uk")
                require(isinstance(quality, Mapping), f"quality result malformed: {parent_id}")
                status, units, retained, rejected = _quality_units(parent_id, text, quality)
                quality_statuses[status] += 1
                retained_bytes += retained
                rejected_bytes += rejected
                retained_units += len(units)
                if status == "RETAIN_PARTIAL":
                    partial_records += 1
                    partial_units += len(units)

                for window_index, payload in units:
                    evidence = _privacy_evidence(scan_privacy(payload), payload)
                    action = evidence.get("action")
                    require(
                        action in {"ALLOW", "REDACT", "QUARANTINE", "EXCLUDE"},
                        f"unknown privacy action: {parent_id}",
                    )
                    privacy_actions[str(action)] += 1
                    counts = evidence.get("detector_counts")
                    require(isinstance(counts, Mapping), f"bad detector counts: {parent_id}")
                    for detector, count in counts.items():
                        require(
                            isinstance(detector, str) and detector,
                            f"bad privacy detector id: {parent_id}",
                        )
                        require(
                            type(count) is int and count >= 0,
                            f"privacy detector count malformed: {parent_id}",
                        )
                        detector_counts[detector] += count
                    if action != "ALLOW":
                        held_units += 1
                        held_bytes += len(payload.encode("utf-8"))
                        continue
                    row = _survivor(
                        parent,
                        payload,
                        status=status,
                        window_index=window_index,
                        quality_policy_sha=quality_policy_sha,
                        privacy_policy_sha=privacy_policy_sha,
                    )
                    line = canonical_bytes(row)
                    handle.write(line)
                    output_sha.update(line)
                    output_records += 1
                    output_jsonl_bytes += len(line)
                    output_text_bytes += int(row["quality_unit_utf8_bytes"])

        require(input_records > 0, "candidate JSONL emitted zero records")
        require(
            retained_bytes + rejected_bytes == input_text_bytes,
            "quality byte conservation invariant failed",
        )
        require(
            output_text_bytes + held_bytes == retained_bytes,
            "privacy byte conservation invariant failed",
        )
        require(
            output_records + held_units == retained_units,
            "privacy unit conservation invariant failed",
        )
        require(
            sum(privacy_actions.values()) == retained_units,
            "privacy action accounting invariant failed",
        )

        core = {
            "schema_version": SCHEMA,
            "execution_profile": "LOCAL_FREE",
            "input_candidate_jsonl_sha256": input_sha,
            "expected_input_candidate_jsonl_sha256": expected_candidate_sha256,
            "input_records": input_records,
            "input_text_utf8_bytes": input_text_bytes,
            "output_records": output_records,
            "output_jsonl_sha256": output_sha.hexdigest(),
            "output_jsonl_bytes": output_jsonl_bytes,
            "output_text_utf8_bytes": output_text_bytes,
            "quality": {
                "mechanics": QUALITY_MODULE,
                "policy_sha256": quality_policy_sha,
                "statuses": dict(sorted(quality_statuses.items())),
                "retained_utf8_bytes": retained_bytes,
                "rejected_utf8_bytes": rejected_bytes,
                "retained_units_before_privacy": retained_units,
                "partial_records_materialized": partial_records,
                "partial_units_materialized_before_privacy": partial_units,
                "rejected_payload_text_retained_in_report": False,
            },
            "privacy": {
                "mechanics": PRIVACY_MODULE,
                "policy_sha256": privacy_policy_sha,
                "actions": dict(sorted(privacy_actions.items())),
                "detector_counts": dict(sorted(detector_counts.items())),
                "allowed_units": output_records,
                "allowed_utf8_bytes": output_text_bytes,
                "held_units": held_units,
                "held_utf8_bytes": held_bytes,
                "non_allow_payload_mutation_attempted": False,
                "privacy_match_payload_retained": False,
                "quality_retained_bytes_reconciled": True,
                "quality_retained_units_reconciled": True,
            },
            "claim_boundary": {
                "language_authority_bound": False,
                "language_quality_privacy_complete": False,
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
            "safe_result": "RADA_TREES_QUALITY_WINDOWS_PRIVACY_V3_ZERO_CREDIT",
        }
        report = {**core, "report_sha256": sha256_bytes(canonical_bytes(core))}
        report_partial.write_bytes(canonical_bytes(report))
        partial.replace(output)
        published_output = True
        report_partial.replace(report_path)
        return report
    except Exception:
        partial.unlink(missing_ok=True)
        report_partial.unlink(missing_ok=True)
        if published_output:
            output.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = materialize(
            args.candidate_jsonl,
            args.output_jsonl,
            args.report,
            expected_candidate_sha256=args.expected_candidate_sha256,
        )
    except (MaterializationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_RADA_TREES_QUALITY_WINDOWS=PASS_ZERO_CREDIT")
    print("OUTPUT_RECORDS=" + str(report["output_records"]))
    print("REPORT_SHA256=" + report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
