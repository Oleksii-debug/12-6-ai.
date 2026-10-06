#!/usr/bin/env python3
"""Execute current-Rada post-DATA232 survivors through canonical G05/G06.

This is an execution-only child. It consumes exact text-free DATA-232 evidence from
the frozen parent execution, rebinds freshly reconstructed current-Rada training
rows to that evidence, executes the merged G05/G06 authorities, physically
materializes accepted G05 windows and G06 redactions, and emits text-free evidence.

No quality threshold, quality granularity, privacy detector, DATA-232 relation,
balance, split, packing, tokenizer, optimizer, training, or scale policy is defined
here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA = "12-6.d03-rada-current-postdata232-g05-g06-execution.v1"
PARENT_EXECUTION_HEAD = "b66ac95e68f83641ff8134baabd55a29fd62b99f"
UPSTREAM_GLOBAL_DEDUP_HEAD = "a4663e87b010b190343caf1d42784f5dc7984601"
PRODUCT_DATA232_HEAD = "3bd56b62b318e1ecf5b1b5f16298a85350d31efb"
EXPECTED_FULL_SELECTION_SHA256 = (
    "601398c39769dabb930997f25506eaaf71bd8960aa82d8af9c697d1cc4075e22"
)
EXPECTED_RADA_SLICE_SHA256 = (
    "f103a3f18216519bd9228e586bd673d49f73cace03b3b0278f2dd0a33383bffb"
)
EXPECTED_DEPENDENCY_BLOBS = {
    "src/twelve_six/__init__.py":
        "5433166c507bc845bd12d8d5c4145f1fbedda204",
    "src/twelve_six/data/_data232_decontamination_matching.py":
        "afa70511f82dc81d9c9f85e3d0b67eba343004f9",
    "src/twelve_six/data/decontamination_authority_v2.py":
        "3ca8f21945c02f692c130a015e036673fa24e7af",
    "src/twelve_six/data/document_quality.py":
        "b1461263034b4fb9510479b20c9697e22faa5f97",
    "src/twelve_six/data/quality_granularity.py":
        "513523b86824c423cad97352b3abb3d1241531b9",
    "src/twelve_six/data/eval647_future_training_exclusion_v1.py":
        "5516577a0720150a7ec12c1bf8898972968e6970",
    "src/twelve_six/data/eval647_reserved_decontamination_v1.py":
        "ce33771c9fb4a6cc421e2f8e1f6f232c119bec71",
    "src/twelve_six/data/current_clean_execution_v1.py":
        "b82ed11626a267dfffa16acd79e45c0cf3e6750b",
    "src/twelve_six/data/current_reserved_decontamination_v1.py":
        "e5c555e3cd27844e98d4ae91af0b746e427f36c9",
    "src/twelve_six/data/quality_execution_authority.py":
        "4659a9d4aba49908f372250904a54361c8d8cf46",
    "src/twelve_six/data/privacy_execution_authority.py":
        "9215287e81c0a82f05ec8405dc4f34c60313c193",
    "src/twelve_six/data/post_g05_g06_materialization_v1.py":
        "830087f91d1fa24385c5cbc8d2f687e4cc46b419",
    "src/twelve_six/data/privacy_filter_v3.py":
        "bcc5938395724f6728ab212f98b39f2334b0f37d",
}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class RadaPostData232Error(RuntimeError):
    """Fail-closed current-Rada post-DATA232 execution error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaPostData232Error(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return canonical(value) + b"\n"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


_DURABLE_FORBIDDEN_TEXT_KEYS = frozenset(
    {
        "text",
        "raw_text",
        "source_text",
        "content",
        "payload_text",
        "normalized_payload",
        "preview",
        "matched_value",
        "secret_value",
    }
)


def assert_text_free_durable(value: Any, *, label: str) -> None:
    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).casefold()
                require(
                    normalized not in _DURABLE_FORBIDDEN_TEXT_KEYS,
                    f"{label}: raw-text-bearing durable key at {path}.{key}",
                )
                walk(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "$")


def write_immutable_bytes(path: Path, payload: bytes, *, label: str) -> None:
    """Atomically create deterministic evidence or resume an identical write."""
    require(not path.is_symlink(), f"{label}: output path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file(), f"{label}: output path is not a regular file")
        require(
            path.read_bytes() == payload,
            f"{label}: refusing to overwrite divergent durable evidence",
        )
        return

    temp = path.with_name(path.name + ".tmp")
    require(not temp.is_symlink(), f"{label}: temp path must not be a symlink")
    if temp.exists():
        require(temp.is_file(), f"{label}: temp path is not a regular file")
        require(
            temp.read_bytes() == payload,
            f"{label}: divergent interrupted temp evidence",
        )
        temp.replace(path)
        return

    created_temp = False
    try:
        with temp.open("xb") as handle:
            created_temp = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except OSError:
        if created_temp:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def commit_durable_bundle(
    *,
    evidence_path: Path,
    evidence_bytes: bytes,
    quality_path: Path,
    quality_bytes: bytes,
    privacy_path: Path,
    privacy_bytes: bytes,
    survivor_inventory_path: Path,
    survivor_inventory_bytes: bytes,
) -> None:
    """Commit child artifacts first and the bound evidence receipt last."""
    paths = (
        evidence_path,
        quality_path,
        privacy_path,
        survivor_inventory_path,
    )
    normalized_paths = {
        os.path.normcase(str(path.resolve(strict=False)))
        for path in paths
    }
    require(
        len(normalized_paths) == len(paths),
        "durable output paths must be pairwise distinct",
    )
    write_immutable_bytes(
        quality_path,
        quality_bytes,
        label="G05 authority",
    )
    write_immutable_bytes(
        privacy_path,
        privacy_bytes,
        label="G06 authority",
    )
    write_immutable_bytes(
        survivor_inventory_path,
        survivor_inventory_bytes,
        label="survivor inventory",
    )
    write_immutable_bytes(
        evidence_path,
        evidence_bytes,
        label="post-G05/G06 evidence",
    )


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or str(
            result.returncode
        )
        raise RadaPostData232Error(
            "git failed: " + " ".join(args) + ": " + detail
        )
    return result


def strict_loads(raw: bytes, label: str) -> Any:
    duplicate = False

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RadaPostData232Error(
                    f"{label}: non-finite JSON constant {item}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RadaPostData232Error(f"{label}: invalid strict JSON") from exc
    require(not duplicate, f"{label}: duplicate JSON member")
    return value


def load_json(
    path: Path,
    label: str,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    require(
        _SHA64.fullmatch(expected_file_sha256) is not None,
        f"{label}: expected file SHA-256 malformed",
    )
    require(
        sha256(raw) == expected_file_sha256,
        f"{label}: file SHA-256 drift",
    )
    value = strict_loads(raw, label)
    require(type(value) is dict, f"{label}: root must be exact object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            require(bool(raw.strip()), f"{label}: blank row {number}")
            value = strict_loads(raw, f"{label}:{number}")
            require(type(value) is dict, f"{label}:{number}: row must be object")
            rows.append(value)
    require(bool(rows), f"{label}: no rows")
    return rows


def verify_self_hash(
    value: Mapping[str, Any],
    field: str,
    *,
    label: str,
    expected: str | None = None,
) -> str:
    require(type(value) is dict, f"{label}: root must be exact object")
    claimed = value.get(field)
    require(
        isinstance(claimed, str) and _SHA64.fullmatch(claimed) is not None,
        f"{label}: {field} malformed",
    )
    if expected is not None:
        require(
            _SHA64.fullmatch(expected) is not None and claimed == expected,
            f"{label}: {field} is not independently expected",
        )
    body = copy.deepcopy(dict(value))
    body.pop(field, None)
    require(
        sha256(canonical(body)) == claimed,
        f"{label}: {field} self-hash mismatch",
    )
    return claimed


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        isinstance(expected_execution_head, str)
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be lowercase 40-hex",
    )
    require(
        git("rev-parse", "HEAD").stdout.strip() == expected_execution_head,
        "execution HEAD drift",
    )
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        PARENT_EXECUTION_HEAD,
        expected_execution_head,
        check=False,
    )
    require(
        ancestor.returncode == 0,
        "exact DATA-232 parent is not an ancestor",
    )
    observed: dict[str, str] = {}
    for path, expected_blob in EXPECTED_DEPENDENCY_BLOBS.items():
        head_blob = git("rev-parse", f"HEAD:{path}").stdout.strip()
        worktree_blob = git("hash-object", str(ROOT / path)).stdout.strip()
        require(head_blob == expected_blob, f"Git blob drift: {path}")
        require(worktree_blob == expected_blob, f"worktree blob drift: {path}")
        observed[path] = expected_blob
    carrier = "tools/run_d03_rada_current_postdata232_g05_g06_v1.py"
    carrier_blob = git("rev-parse", f"HEAD:{carrier}").stdout.strip()
    require(
        git("hash-object", str(ROOT / carrier)).stdout.strip() == carrier_blob,
        "executing carrier worktree drift",
    )
    observed[carrier] = carrier_blob
    return dict(sorted(observed.items()))


def load_bound_runtime() -> tuple[Any, Any, Any]:
    """Import behavior modules only after the exact worktree closure is pinned."""
    from twelve_six.data import current_clean_execution_v1 as clean
    from twelve_six.data import current_reserved_decontamination_v1 as reserved
    from twelve_six.data.decontamination_authority_v2 import verify_report

    return clean, reserved, verify_report


def verify_zero_credit_truth(
    value: Mapping[str, Any],
    *,
    label: str,
    require_model_selection_false: bool,
) -> None:
    zero_fields = (
        "canonical_capacity_credited",
        "authorized_optimized_target_exposure",
        "authorized_unique_loss_positions",
        "authorized_training_exposure",
    )
    false_fields = (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "scale_promotion_authorized",
    )
    for field in zero_fields:
        require(
            type(value.get(field)) is int and value.get(field) == 0,
            f"{label}: authority widened: {field}",
        )
    for field in false_fields:
        require(
            value.get(field) is False,
            f"{label}: truth boundary widened: {field}",
        )
    if require_model_selection_false:
        require(
            value.get("model_architecture_or_hyperparameters_selected") is False,
            f"{label}: truth boundary widened: "
            "model_architecture_or_hyperparameters_selected",
        )


def verify_parent(
    *,
    reserved_module: Any,
    verify_report_fn: Any,
    training_records: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    handoff: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
    proof: Mapping[str, Any],
    expected_inventory_identity_sha256: str,
    expected_handoff_identity_sha256: str,
    expected_result_identity_sha256: str,
    expected_proof_identity_sha256: str,
    report_file_sha256: str,
    evidence_file_sha256: str,
    result_file_sha256: str,
) -> tuple[str, str, str]:
    inventory_id = verify_self_hash(
        inventory,
        "inventory_identity_sha256",
        label="parent inventory",
        expected=expected_inventory_identity_sha256,
    )
    require(
        inventory.get("schema_version")
        == "12-6.d03-rada-current-postdedup-inventory.v1",
        "parent inventory schema drift",
    )
    require(
        inventory.get("full_selection_projection_sha256")
        == EXPECTED_FULL_SELECTION_SHA256,
        "parent inventory full-selection drift",
    )
    require(
        inventory.get("current_rada_slice_authority_sha256")
        == EXPECTED_RADA_SLICE_SHA256,
        "parent inventory Rada-slice drift",
    )
    require(
        inventory.get("raw_text_persisted") is False,
        "parent inventory persisted raw text",
    )

    handoff_id = verify_self_hash(
        handoff,
        "handoff_identity_sha256",
        label="parent handoff",
        expected=expected_handoff_identity_sha256,
    )
    reserved_module._verify_training_handoff(
        training_records,
        handoff,
        expected_inventory_identity_sha256=inventory_id,
        expected_survivor_authority_sha256=EXPECTED_FULL_SELECTION_SHA256,
        expected_handoff_identity_sha256=handoff_id,
    )

    verify_report_fn(report)
    reserved_module.verify_execution_evidence(evidence, report)
    report_id = report.get("report_sha256")
    evidence_id = evidence.get("execution_identity_sha256")
    require(
        isinstance(report_id, str) and _SHA64.fullmatch(report_id) is not None,
        "parent DATA-232 report identity malformed",
    )
    require(
        isinstance(evidence_id, str)
        and _SHA64.fullmatch(evidence_id) is not None,
        "parent DATA-232 execution identity malformed",
    )
    require(
        report.get("status") in {"PASS_CLEAN", "PASS_WITH_EXCLUSIONS"},
        "parent DATA-232 report is not terminal PASS",
    )
    require(
        evidence.get("status") == report.get("status"),
        "parent DATA-232 report/evidence status drift",
    )

    result_id = verify_self_hash(
        result,
        "result_identity_sha256",
        label="parent result",
        expected=expected_result_identity_sha256,
    )
    require(
        result.get("schema_version")
        == "12-6.d03-rada-current-data232-execution-result.v1",
        "parent result schema drift",
    )
    require(
        result.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "parent result execution HEAD drift",
    )
    require(
        result.get("parent_execution_head_sha") == UPSTREAM_GLOBAL_DEDUP_HEAD,
        "parent result upstream global-dedup HEAD drift",
    )
    require(
        result.get("postdedup_inventory_identity_sha256") == inventory_id,
        "parent result inventory drift",
    )
    require(
        result.get("full_selection_projection_sha256")
        == EXPECTED_FULL_SELECTION_SHA256,
        "parent result full-selection drift",
    )
    require(
        result.get("current_rada_slice_authority_sha256")
        == EXPECTED_RADA_SLICE_SHA256,
        "parent result Rada-slice drift",
    )
    require(
        result.get("training_handoff_identity_sha256") == handoff_id,
        "parent result handoff drift",
    )
    require(
        result.get("data232_report_sha256") == report_id,
        "parent result/report drift",
    )
    require(
        result.get("data232_execution_identity_sha256") == evidence_id,
        "parent result/execution-evidence drift",
    )
    require(
        result.get("selection_validation_identity_sha256")
        == report.get("selection_validation_identity")
        and result.get("final_test_identity_sha256")
        == report.get("final_test_identity"),
        "parent result reserved-evaluation identity drift",
    )
    require(
        result.get("counts") == evidence.get("counts")
        and result.get("counts") == report.get("counts"),
        "parent result/report/evidence count drift",
    )
    require(
        result.get("next_gate") == "CURRENT_RADA_POST_DATA232_QUALITY_PRIVACY",
        "parent result next-gate drift",
    )
    require(
        result.get("status") == report.get("status")
        and result.get("status") == evidence.get("status"),
        "parent result/report/evidence status drift",
    )
    require(
        result.get("durable_evidence_hash_only") is True,
        "parent result durable evidence is not hash-only",
    )
    require(
        result.get("final_test_payload_accessed_for_decontamination") is True,
        "parent result final-test decontamination access truth missing",
    )
    verify_zero_credit_truth(
        result,
        label="parent result",
        require_model_selection_false=True,
    )

    proof_id = verify_self_hash(
        proof,
        "proof_identity_sha256",
        label="parent two-clean proof",
        expected=expected_proof_identity_sha256,
    )
    require(
        proof.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "parent proof execution HEAD drift",
    )
    require(
        proof.get("parent_execution_head_sha") == UPSTREAM_GLOBAL_DEDUP_HEAD,
        "parent proof upstream global-dedup HEAD drift",
    )
    require(
        proof.get("data232_report_sha256") == report_id
        and proof.get("data232_execution_identity_sha256") == evidence_id
        and proof.get("result_identity_sha256") == result_id,
        "parent proof nested identity drift",
    )
    require(
        proof.get("data232_report_file_sha256") == report_file_sha256
        and proof.get("data232_execution_evidence_file_sha256")
        == evidence_file_sha256
        and proof.get("result_file_sha256") == result_file_sha256,
        "parent proof/file SHA-256 drift",
    )
    require(
        proof.get("schema_version")
        == "12-6.d03-rada-current-data232-two-clean.v1",
        "parent proof schema drift",
    )
    require(
        proof.get("full_selection_projection_sha256")
        == EXPECTED_FULL_SELECTION_SHA256
        and proof.get("current_rada_slice_authority_sha256")
        == EXPECTED_RADA_SLICE_SHA256
        and proof.get("postdedup_inventory_identity_sha256") == inventory_id
        and proof.get("training_handoff_identity_sha256") == handoff_id
        and proof.get("reserved_payload_binding_identity_sha256")
        == result.get("reserved_payload_binding_identity_sha256"),
        "parent proof corpus-lineage drift",
    )
    counts = result.get("counts")
    require(isinstance(counts, Mapping), "parent result counts missing")
    proof_counts = {
        "training_records": proof.get("training_records"),
        "evaluation_records": proof.get("evaluation_records"),
        "excluded_training_records": proof.get("excluded_training_records"),
        "quarantined_source_families": proof.get(
            "quarantined_source_families"
        ),
        "match_evidence_records": proof.get("match_evidence_records"),
    }
    require(
        proof_counts == {
            key: counts.get(key)
            for key in proof_counts
        },
        "parent proof count projection drift",
    )
    require(
        proof.get("two_fresh_processes_byte_identical") is True,
        "parent proof lacks two byte-identical fresh processes",
    )
    verify_zero_credit_truth(
        proof,
        label="parent two-clean proof",
        require_model_selection_false=False,
    )
    return report_id, evidence_id, proof_id


def _materialize_quality_survivors_with_partial(
    inputs: list[dict[str, str]],
    metadata: Mapping[str, Mapping[str, str]],
    quality: Mapping[str, Any],
) -> tuple[
    list[dict[str, str]],
    dict[str, int],
    dict[str, Any],
]:
    """Materialize authoritative G05 accepted units, including partial windows.

    This is the exact seam proven by the terminal Rada quality-window lineage:
    a RETAIN_PARTIAL natural-language document emits each accepted authoritative
    quality window independently; rejected siblings are not allowed to evict
    accepted siblings. The current G05 authority remains the sole source of
    spans, decisions, hashes, and byte counts.
    """
    raw_rows = quality.get("records")
    require(isinstance(raw_rows, list), "G05 records missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        require(isinstance(row, Mapping), "G05 record is not an object")
        record_id = row.get("record_id")
        require(
            isinstance(record_id, str) and bool(record_id),
            "G05 record_id missing",
        )
        require(record_id not in by_id, "duplicate G05 record_id")
        by_id[record_id] = row
    require(
        set(by_id) == {row["id"] for row in inputs},
        "G05/input record-set drift",
    )

    output: list[dict[str, str]] = []
    seen_output_ids: set[str] = set()
    partial_projection: list[dict[str, Any]] = []
    stats = {
        "g05_reject_documents": 0,
        "g05_partial_documents": 0,
        "g05_rejected_units": 0,
        "g05_rejected_utf8_bytes": 0,
    }
    detail = {
        "input_documents": len(inputs),
        "retain_all_documents": 0,
        "partial_documents": 0,
        "reject_documents": 0,
        "partial_emitted_units": 0,
        "partial_rejected_units": 0,
        "input_utf8_bytes": 0,
        "retained_utf8_bytes": 0,
        "rejected_utf8_bytes": 0,
    }

    for source in inputs:
        record_id = source["id"]
        text = source["text"]
        mode = source["mode"]
        row = by_id[record_id]
        payload = text.encode("utf-8")
        require(
            row.get("payload_sha256") == sha256(payload)
            and row.get("utf8_bytes") == len(payload)
            and row.get("mode") == mode,
            f"G05 payload binding drift: {record_id}",
        )
        status = row.get("status")
        units = row.get("units")
        authoritative_unit = row.get("authoritative_unit")
        require(
            status in {"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"},
            f"G05 status drift: {record_id}",
        )
        require(
            authoritative_unit
            in {"DOCUMENT", "BOUNDED_NATURAL_LANGUAGE_WINDOW"},
            f"G05 authoritative unit drift: {record_id}",
        )
        require(
            isinstance(units, list) and bool(units),
            f"G05 units missing: {record_id}",
        )

        detail["input_utf8_bytes"] += len(payload)
        accepted_bytes = 0
        rejected_bytes = 0
        expected_start = 0
        accepted_units: list[tuple[Mapping[str, Any], str]] = []
        seen_unit_ids: set[str] = set()
        for index, unit in enumerate(units):
            require(isinstance(unit, Mapping), "G05 unit must be an object")
            start = unit.get("start_char")
            end = unit.get("end_char")
            accepted = unit.get("accepted")
            unit_id = unit.get("unit_id")
            require(
                type(start) is int
                and type(end) is int
                and start == expected_start
                and start < end <= len(text),
                f"G05 unit partition drift: {record_id}",
            )
            require(
                type(accepted) is bool,
                f"G05 unit accepted type drift: {record_id}",
            )
            require(
                isinstance(unit_id, str)
                and bool(unit_id)
                and unit_id not in seen_unit_ids,
                f"G05 unit_id drift: {record_id}",
            )
            seen_unit_ids.add(unit_id)
            if authoritative_unit == "DOCUMENT":
                require(
                    len(units) == 1
                    and index == 0
                    and unit_id == record_id
                    and start == 0
                    and end == len(text),
                    f"G05 document unit drift: {record_id}",
                )
            else:
                require(
                    unit_id
                    == f"{record_id}#quality-window-{index:04d}",
                    f"G05 quality-window identity drift: {record_id}",
                )

            piece = text[start:end]
            piece_raw = piece.encode("utf-8")
            piece_sha = sha256(piece_raw)
            require(
                unit.get("payload_sha256") == piece_sha
                and unit.get("utf8_bytes") == len(piece_raw),
                f"G05 unit payload drift: {unit_id}",
            )
            decision_sha = unit.get("decision_sha256")
            require(
                isinstance(decision_sha, str)
                and _SHA64.fullmatch(decision_sha) is not None,
                f"G05 decision identity drift: {unit_id}",
            )
            if accepted:
                accepted_bytes += len(piece_raw)
                accepted_units.append((unit, piece))
            else:
                rejected_bytes += len(piece_raw)
                stats["g05_rejected_units"] += 1
                stats["g05_rejected_utf8_bytes"] += len(piece_raw)
            expected_start = end

        require(
            expected_start == len(text),
            f"G05 units do not reconstruct document: {record_id}",
        )
        require(
            accepted_bytes == row.get("retained_utf8_bytes")
            and rejected_bytes == row.get("rejected_utf8_bytes"),
            f"G05 retained/rejected byte accounting drift: {record_id}",
        )
        detail["retained_utf8_bytes"] += accepted_bytes
        detail["rejected_utf8_bytes"] += rejected_bytes

        meta = metadata[record_id]
        if status == "RETAIN_ALL":
            require(
                accepted_bytes == len(payload)
                and rejected_bytes == 0
                and len(accepted_units) == len(units),
                f"G05 RETAIN_ALL vector drift: {record_id}",
            )
            require(
                record_id not in seen_output_ids,
                f"G05 materialized record-id collision: {record_id}",
            )
            seen_output_ids.add(record_id)
            output.append(
                {
                    "record_id": record_id,
                    "source_id": meta["source_id"],
                    "family": meta["family"],
                    "modality": mode,
                    "normalized_payload": text,
                }
            )
            detail["retain_all_documents"] += 1
            continue

        if status == "REJECT_DOCUMENT":
            require(
                accepted_bytes == 0
                and rejected_bytes == len(payload)
                and not accepted_units,
                f"G05 rejected record retained bytes: {record_id}",
            )
            stats["g05_reject_documents"] += 1
            detail["reject_documents"] += 1
            continue

        require(
            authoritative_unit == "BOUNDED_NATURAL_LANGUAGE_WINDOW",
            f"G05 partial row is not authoritative windows: {record_id}",
        )
        require(
            accepted_bytes > 0
            and rejected_bytes > 0
            and accepted_units
            and len(accepted_units) < len(units),
            f"G05 partial vector drift: {record_id}",
        )
        stats["g05_partial_documents"] += 1
        detail["partial_documents"] += 1
        rejected_unit_count = len(units) - len(accepted_units)
        detail["partial_rejected_units"] += rejected_unit_count

        for unit, piece in accepted_units:
            piece_raw = piece.encode("utf-8")
            piece_sha = sha256(piece_raw)
            unit_id = str(unit["unit_id"])
            derived_id = f"{unit_id}:{piece_sha}"
            require(
                derived_id not in seen_output_ids,
                f"G05 derived record-id collision: {derived_id}",
            )
            seen_output_ids.add(derived_id)
            output.append(
                {
                    "record_id": derived_id,
                    "source_id": meta["source_id"],
                    "family": meta["family"],
                    "modality": mode,
                    "normalized_payload": piece,
                }
            )
            partial_projection.append(
                {
                    "parent_record_id": record_id,
                    "record_id": derived_id,
                    "source_id": meta["source_id"],
                    "family": meta["family"],
                    "modality": mode,
                    "unit_id": unit_id,
                    "start_char": unit["start_char"],
                    "end_char": unit["end_char"],
                    "payload_sha256": piece_sha,
                    "utf8_bytes": len(piece_raw),
                    "decision_sha256": unit["decision_sha256"],
                }
            )
            detail["partial_emitted_units"] += 1

    require(bool(output), "G05 removed every post-decontamination record")
    require(
        detail["input_utf8_bytes"]
        == detail["retained_utf8_bytes"] + detail["rejected_utf8_bytes"],
        "G05 byte conservation drift",
    )
    require(
        detail["partial_documents"] == stats["g05_partial_documents"],
        "G05 partial-document accounting drift",
    )
    require(
        detail["reject_documents"] == stats["g05_reject_documents"],
        "G05 rejected-document accounting drift",
    )
    require(
        detail["partial_rejected_units"]
        <= stats["g05_rejected_units"],
        "G05 partial rejected-unit accounting drift",
    )
    require(
        len(output)
        == detail["retain_all_documents"] + detail["partial_emitted_units"],
        "G05 materialized output-count drift",
    )
    output.sort(key=lambda row: row["record_id"])
    detail["partial_unit_projection_sha256"] = sha256(
        canonical(partial_projection)
    )
    detail["partial_unit_projection_count"] = len(partial_projection)
    return output, stats, detail



def execute(args: argparse.Namespace) -> dict[str, Any]:
    authority_blobs = verify_local_authority(args.expected_execution_head)
    clean, reserved, verify_report = load_bound_runtime()
    require(
        type(args.parent_artifact_id) is int and args.parent_artifact_id > 0,
        "parent artifact ID must be a positive exact integer",
    )
    require(
        isinstance(args.expected_parent_artifact_zip_sha256, str)
        and _SHA64.fullmatch(args.expected_parent_artifact_zip_sha256) is not None,
        "parent artifact ZIP SHA-256 malformed",
    )
    training = load_jsonl(args.training_records_jsonl, "training records")
    inventory = load_json(
        args.parent_inventory_json,
        "parent inventory",
        expected_file_sha256=args.expected_parent_inventory_file_sha256,
    )
    handoff = load_json(
        args.parent_handoff_json,
        "parent handoff",
        expected_file_sha256=args.expected_parent_handoff_file_sha256,
    )
    report = load_json(
        args.parent_report_json,
        "parent DATA-232 report",
        expected_file_sha256=args.expected_parent_report_file_sha256,
    )
    evidence = load_json(
        args.parent_execution_evidence_json,
        "parent DATA-232 execution evidence",
        expected_file_sha256=(
            args.expected_parent_execution_evidence_file_sha256
        ),
    )
    result = load_json(
        args.parent_result_json,
        "parent DATA-232 result",
        expected_file_sha256=args.expected_parent_result_file_sha256,
    )
    proof = load_json(
        args.parent_two_clean_proof_json,
        "parent DATA-232 two-clean proof",
        expected_file_sha256=args.expected_parent_proof_file_sha256,
    )

    report_id, data232_execution_id, proof_id = verify_parent(
        reserved_module=reserved,
        verify_report_fn=verify_report,
        training_records=training,
        inventory=inventory,
        handoff=handoff,
        report=report,
        evidence=evidence,
        result=result,
        proof=proof,
        expected_inventory_identity_sha256=(
            args.expected_parent_inventory_identity_sha256
        ),
        expected_handoff_identity_sha256=(
            args.expected_parent_handoff_identity_sha256
        ),
        expected_result_identity_sha256=args.expected_parent_result_identity_sha256,
        expected_proof_identity_sha256=args.expected_parent_proof_identity_sha256,
        report_file_sha256=args.expected_parent_report_file_sha256,
        evidence_file_sha256=(
            args.expected_parent_execution_evidence_file_sha256
        ),
        result_file_sha256=args.expected_parent_result_file_sha256,
    )

    quality_inputs, metadata, data232_excluded = (
        clean._post_decontamination_records(training, report)
    )
    quality_projection = clean._input_projection(quality_inputs)
    quality_rows_sha256 = sha256(canonical_line(quality_projection))
    quality = clean.build_quality_execution_authority(
        quality_inputs,
        input_manifest_sha256=data232_execution_id,
        expected_input_rows_sha256=quality_rows_sha256,
    )
    quality_id = quality.get("execution_identity_sha256")
    require(
        isinstance(quality_id, str) and _SHA64.fullmatch(quality_id) is not None,
        "G05 execution identity malformed",
    )
    clean.verify_quality_execution_authority(
        quality,
        quality_inputs,
        expected_input_manifest_sha256=data232_execution_id,
        expected_input_rows_sha256=quality_rows_sha256,
        expected_execution_identity_sha256=quality_id,
    )

    quality_survivors, quality_stats, partial_detail = (
        _materialize_quality_survivors_with_partial(
            quality_inputs,
            metadata,
            quality,
        )
    )
    privacy_inputs = clean._quality_records_for_privacy(quality_survivors)
    privacy_projection = clean._input_projection(privacy_inputs)
    privacy_rows_sha256 = sha256(canonical_line(privacy_projection))
    privacy = clean.build_privacy_execution_authority(
        privacy_inputs,
        expected_input_rows_sha256=privacy_rows_sha256,
    )
    privacy_id = privacy.get("execution_identity_sha256")
    require(
        isinstance(privacy_id, str)
        and _SHA64.fullmatch(privacy_id) is not None,
        "G06 execution identity malformed",
    )
    clean.verify_privacy_execution_authority(
        privacy,
        privacy_inputs,
        expected_input_rows_sha256=privacy_rows_sha256,
        expected_execution_identity_sha256=privacy_id,
    )

    final_survivors, privacy_stats = clean._materialize_privacy_survivors(
        quality_survivors,
        privacy,
    )
    survivor_inventory = clean.materialize_record_inventory(final_survivors)
    final_bytes = survivor_inventory["total_payload_bytes"]
    require(
        type(final_bytes) is int and final_bytes > 0,
        "post-G05/G06 survivor bytes invalid",
    )
    g05_retained_bytes = partial_detail["retained_utf8_bytes"]
    require(
        type(g05_retained_bytes) is int and g05_retained_bytes > 0,
        "G05 retained byte count invalid",
    )
    g06_payload_delta_bytes = final_bytes - g05_retained_bytes

    detector_counts = privacy.get("detector_counts")
    require(isinstance(detector_counts, Mapping), "G06 detector counts missing")
    normalized_detectors: dict[str, int] = {}
    for key, value in detector_counts.items():
        require(
            isinstance(key, str)
            and bool(key)
            and type(value) is int
            and value >= 0,
            "G06 detector count malformed",
        )
        normalized_detectors[key] = value

    quality_bytes = canonical_line(quality)
    privacy_bytes = canonical_line(privacy)
    survivor_inventory_bytes = canonical_line(survivor_inventory)

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": args.expected_execution_head,
        "parent": {
            "execution_head_sha": PARENT_EXECUTION_HEAD,
            "product_data232_head_sha": PRODUCT_DATA232_HEAD,
            "artifact_id": args.parent_artifact_id,
            "artifact_zip_sha256": args.expected_parent_artifact_zip_sha256,
            "inventory_identity_sha256": (
                args.expected_parent_inventory_identity_sha256
            ),
            "training_handoff_identity_sha256": (
                args.expected_parent_handoff_identity_sha256
            ),
            "data232_report_sha256": report_id,
            "data232_execution_identity_sha256": data232_execution_id,
            "data232_result_identity_sha256": (
                args.expected_parent_result_identity_sha256
            ),
            "data232_two_clean_proof_identity_sha256": proof_id,
            "full_selection_projection_sha256": (
                EXPECTED_FULL_SELECTION_SHA256
            ),
            "current_rada_slice_authority_sha256": (
                EXPECTED_RADA_SLICE_SHA256
            ),
            "two_fresh_data232_processes_byte_identical": True,
        },
        "g05": {
            "input_rows_sha256": quality_rows_sha256,
            "execution_identity_sha256": quality_id,
            "counts": quality["counts"],
            "bytes": quality["bytes"],
            "partial_materialization": partial_detail,
        },
        "g06": {
            "input_rows_sha256": privacy_rows_sha256,
            "execution_identity_sha256": privacy_id,
            "detector_counts": dict(sorted(normalized_detectors.items())),
            "materialization": privacy_stats,
            "payload_delta_from_g05_retained_bytes": g06_payload_delta_bytes,
        },
        "durable_artifacts": {
            "g05_authority_file_sha256": sha256(quality_bytes),
            "g06_authority_file_sha256": sha256(privacy_bytes),
            "survivor_inventory_file_sha256": sha256(
                survivor_inventory_bytes
            ),
            "completion_marker": "POST_G05_G06_EVIDENCE_WRITTEN_LAST",
        },
        "survivor_inventory": {
            "record_count": survivor_inventory["record_count"],
            "total_payload_bytes": survivor_inventory["total_payload_bytes"],
            "record_inventory_digest_sha256": survivor_inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": survivor_inventory[
                "payload_inventory_digest_sha256"
            ],
        },
        "counts": {
            "input_training_records": len(training),
            "data232_excluded_records": data232_excluded,
            "post_data232_records": len(quality_inputs),
            "post_g05_records": len(quality_survivors),
            "post_g06_records": len(final_survivors),
            "post_g06_source_objects": len(
                {row["source_id"] for row in final_survivors}
            ),
            "post_g06_payload_bytes": final_bytes,
        },
        "authority_blobs": authority_blobs,
        "content_boundary": {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "truth_boundary": {
            "current_rada_data232_parent_two_clean_complete": True,
            "canonical_quality_privacy_executed": True,
            "balance_diversity_retest_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "deterministic_pack_two_clean_complete": False,
            "positive_exact_unique_loss_ledger": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "scale_promotion_authorized": False,
        },
        "next_gate": "CURRENT_RADA_BALANCE_DIVERSITY_FAMILY_CAP_RETEST",
    }
    output = {
        **evidence_core,
        "evidence_identity_sha256": sha256(canonical(evidence_core)),
    }
    assert_text_free_durable(output, label="post-G05/G06 evidence")
    assert_text_free_durable(quality, label="G05 authority")
    assert_text_free_durable(privacy, label="G06 authority")
    assert_text_free_durable(
        survivor_inventory,
        label="survivor inventory",
    )
    # Child artifacts are committed first; the evidence receipt is the final
    # completion marker and binds their exact canonical file bytes.
    commit_durable_bundle(
        evidence_path=args.output_evidence,
        evidence_bytes=canonical_line(output),
        quality_path=args.output_quality,
        quality_bytes=quality_bytes,
        privacy_path=args.output_privacy,
        privacy_bytes=privacy_bytes,
        survivor_inventory_path=args.output_survivor_inventory,
        survivor_inventory_bytes=survivor_inventory_bytes,
    )
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--expected-execution-head", required=True)
    result.add_argument("--parent-artifact-id", type=int, required=True)
    result.add_argument("--expected-parent-artifact-zip-sha256", required=True)
    result.add_argument("--training-records-jsonl", type=Path, required=True)
    result.add_argument("--parent-inventory-json", type=Path, required=True)
    result.add_argument("--parent-handoff-json", type=Path, required=True)
    result.add_argument("--parent-report-json", type=Path, required=True)
    result.add_argument("--parent-execution-evidence-json", type=Path, required=True)
    result.add_argument("--parent-result-json", type=Path, required=True)
    result.add_argument("--parent-two-clean-proof-json", type=Path, required=True)
    result.add_argument("--expected-parent-inventory-file-sha256", required=True)
    result.add_argument("--expected-parent-handoff-file-sha256", required=True)
    result.add_argument("--expected-parent-report-file-sha256", required=True)
    result.add_argument(
        "--expected-parent-execution-evidence-file-sha256",
        required=True,
    )
    result.add_argument("--expected-parent-result-file-sha256", required=True)
    result.add_argument("--expected-parent-proof-file-sha256", required=True)
    result.add_argument("--expected-parent-inventory-identity-sha256", required=True)
    result.add_argument("--expected-parent-handoff-identity-sha256", required=True)
    result.add_argument("--expected-parent-result-identity-sha256", required=True)
    result.add_argument("--expected-parent-proof-identity-sha256", required=True)
    result.add_argument("--output-evidence", type=Path, required=True)
    result.add_argument("--output-quality", type=Path, required=True)
    result.add_argument("--output-privacy", type=Path, required=True)
    result.add_argument("--output-survivor-inventory", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        output = execute(args)
    except (
        ImportError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:500]
        print(f"D03_RADA_CURRENT_POST_DATA232_G05_G06=BLOCKED: {detail}")
        return 2
    print("D03_RADA_CURRENT_POST_DATA232_G05_G06=PASS_ZERO_CREDIT")
    print(
        "EVIDENCE_IDENTITY_SHA256="
        + output["evidence_identity_sha256"]
    )
    print(
        "POST_G06_RADA_BYTES="
        + str(output["counts"]["post_g06_payload_bytes"])
    )
    print(
        "POST_G06_RADA_RECORDS="
        + str(output["counts"]["post_g06_records"])
    )
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
