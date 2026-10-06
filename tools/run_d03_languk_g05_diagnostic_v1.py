"""Emit text-free aggregate G05 diagnostics for the exact Lang-UK court cohort.

This execution-only diagnostic replays the exact #2848 source/privacy/selection
lineage, verifies the incumbent G05 authority, then aggregates rejection reasons
and bounded feature statistics. It never changes quality policy and never emits
record identifiers or source text.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tools.run_d03_languk_court_privacy_retest_v1 as owner
from twelve_six.data.document_quality import (
    QualityDecision,
    assess_document,
    default_quality_policy,
)
from twelve_six.data.quality_granularity import frozen_granularity_policy

SCHEMA = "12-6.d03-languk-g05-aggregate-diagnostic.v1"
OWNER_HEAD = "7cda697ac4fc8330754b28f0a02e119e395dbd25"
OWNER_RUNNER_BLOB = "da2d904bf9ab67bf8b20acd8cdb6b09d33657736"
DOCUMENT_QUALITY_BLOB = "b1461263034b4fb9510479b20c9697e22faa5f97"
GRANULARITY_BLOB = "513523b86824c423cad97352b3abb3d1241531b9"
EXPECTED_SELECTED_MANIFEST = (
    "bbcbd9c2235234007737128c37f375ba0802f32a6dcc45bac02aff30fe4c3fc3"
)
EXPECTED_QUALITY_EXECUTION = (
    "537b302bfe32314fbdcfe284094c98c817fddd0a9910bb475b1c2dcdbba1a732"
)
EXPECTED_THRESHOLD_POLICY = (
    "97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d"
)
EXPECTED_GRANULARITY_POLICY = (
    "e8685c2c6b265b9b289ded7a5245888d8d16ae4d6e881f6229f3bc777601f857"
)
EXPECTED_RECORDS = 256

_TRUTH = {
    "canonical_capacity_credited": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "optimizer_updates_executed": 0,
    "tokenizer_fit_authorized": False,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "scale_promotion_authorized": False,
    "registry_change_authorized": False,
    "source_family_admitted": False,
}


class LangUkG05DiagnosticError(RuntimeError):
    """Fail-closed diagnostic error."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise LangUkG05DiagnosticError(message)


def canonical(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def self_hashed(core: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**copy.deepcopy(dict(core)), field: sha256(canonical(core))}


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(dict(value), newline=True))


def _rounded(value: int | float) -> int | float:
    if isinstance(value, int):
        return value
    return round(float(value), 6)


def numeric_summary(values: Sequence[int | float]) -> dict[str, int | float]:
    """Return deterministic bounded distribution statistics without exemplars."""
    need(bool(values), "cannot summarize an empty numeric sequence")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": _rounded(ordered[0]),
        "median": _rounded(statistics.median(ordered)),
        "max": _rounded(ordered[-1]),
    }


def _selected_rows(source_rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    privacy_inputs = [
        {"id": row["record_id"], "text": row["normalized_text"], "mode": "uk"}
        for row in source_rows
    ]
    privacy_input_root = owner.input_rows_sha256(privacy_inputs)
    privacy = owner.build_privacy_execution_authority(
        privacy_inputs,
        expected_input_rows_sha256=privacy_input_root,
    )
    owner.verify_privacy_execution_authority(
        privacy,
        privacy_inputs,
        expected_input_rows_sha256=privacy_input_root,
        expected_execution_identity_sha256=privacy["execution_identity_sha256"],
    )
    privacy_by_id = {row["record_id"]: row for row in privacy["records"]}
    allow_rows = [
        row
        for row in source_rows
        if privacy_by_id[row["record_id"]]["action"] == "ALLOW"
    ]
    selected = list(allow_rows[: owner.MAX_RECORDS])
    need(len(selected) == EXPECTED_RECORDS, "exact selected cohort size drift")
    return selected


def _selected_manifest_identity(selected: Sequence[Mapping[str, str]]) -> str:
    rows = [
        {
            "record_id": row["record_id"],
            "source_id_sha256": sha256(row["source_id"].encode("utf-8")),
            "raw_payload_sha256": sha256(row["raw_text"].encode("utf-8")),
            "normalized_payload_sha256": sha256(
                row["normalized_text"].encode("utf-8")
            ),
            "normalized_payload_bytes": len(
                row["normalized_text"].encode("utf-8")
            ),
        }
        for row in selected
    ]
    manifest = {
        "dataset": owner.DATASET,
        "revision": owner.REVISION,
        "source_file": owner.SOURCE_FILE,
        "source_family": owner.FAMILY,
        "family_admitted": False,
        "max_records": owner.MAX_RECORDS,
        "selection_rule": "privacy_action_ALLOW_then_ascending_stable_source_id",
        "records": rows,
    }
    return sha256(canonical(manifest))


def _quality_inputs(
    selected: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    return [
        {"id": row["record_id"], "text": row["normalized_text"], "mode": "uk"}
        for row in selected
    ]


def _verify_parent_quality(
    quality_inputs: Sequence[Mapping[str, str]],
    selected_manifest_identity: str,
) -> Mapping[str, Any]:
    quality_input_root = owner._quality_input_rows_sha(quality_inputs)
    authority = owner.build_quality_execution_authority(
        quality_inputs,
        input_manifest_sha256=selected_manifest_identity,
        expected_input_rows_sha256=quality_input_root,
    )
    owner.verify_quality_execution_authority(
        authority,
        quality_inputs,
        expected_input_manifest_sha256=selected_manifest_identity,
        expected_input_rows_sha256=quality_input_root,
        expected_execution_identity_sha256=EXPECTED_QUALITY_EXECUTION,
    )
    need(
        authority["execution_identity_sha256"] == EXPECTED_QUALITY_EXECUTION,
        "parent G05 execution identity drift",
    )
    return authority


def aggregate_decisions(
    decisions: Sequence[QualityDecision],
) -> dict[str, Any]:
    """Aggregate decisions without retaining record IDs, source text, or examples."""
    need(bool(decisions), "diagnostic requires at least one decision")
    reason_counts: Counter[str] = Counter()
    reason_set_counts: Counter[str] = Counter()
    accepted = 0
    for decision in decisions:
        accepted += int(decision.accepted)
        reason_counts.update(decision.reasons)
        reason_set_counts["|".join(decision.reasons) if decision.reasons else "<accepted>"] += 1

    fields = {
        "chars": [d.features.chars for d in decisions],
        "utf8_bytes": [d.features.utf8_bytes for d in decisions],
        "token_count": [d.features.token_count for d in decisions],
        "unique_token_count": [d.features.unique_token_count for d in decisions],
        "distinct_token_ratio": [d.features.distinct_token_ratio for d in decisions],
        "dominant_token_ratio": [d.features.dominant_token_ratio for d in decisions],
        "symbol_ratio": [d.features.symbol_ratio for d in decisions],
        "repeated_line_ratio": [d.features.repeated_line_ratio for d in decisions],
        "url_char_ratio": [d.features.url_char_ratio for d in decisions],
        "template_line_ratio": [d.features.template_line_ratio for d in decisions],
        "boilerplate_line_ratio": [d.features.boilerplate_line_ratio for d in decisions],
        "other_script_letter_ratio": [
            d.features.other_script_letter_ratio for d in decisions
        ],
        "edge_margin": [d.edge_margin for d in decisions],
    }
    policy = default_quality_policy()
    thresholds = policy.uk

    return {
        "documents": len(decisions),
        "accepted_documents": accepted,
        "rejected_documents": len(decisions) - accepted,
        "reason_counts": dict(sorted(reason_counts.items())),
        "reason_set_counts": dict(sorted(reason_set_counts.items())),
        "feature_summaries": {
            name: numeric_summary(values) for name, values in fields.items()
        },
        "threshold_crossings": {
            "below_min_chars": sum(
                d.features.chars < thresholds.min_chars for d in decisions
            ),
            "above_max_chars": sum(
                d.features.chars > thresholds.max_chars for d in decisions
            ),
            "above_symbol_ratio": sum(
                d.features.symbol_ratio > thresholds.max_symbol_ratio
                for d in decisions
            ),
            "above_repeated_line_ratio": sum(
                d.features.repeated_line_ratio > thresholds.max_repeated_line_ratio
                for d in decisions
            ),
            "above_url_char_ratio": sum(
                d.features.url_char_ratio > thresholds.max_url_char_ratio
                for d in decisions
            ),
            "above_template_line_ratio": sum(
                d.features.template_line_ratio > thresholds.max_template_line_ratio
                for d in decisions
            ),
            "above_boilerplate_line_ratio": sum(
                d.features.boilerplate_line_ratio
                > thresholds.max_boilerplate_line_ratio
                for d in decisions
            ),
            "above_other_script_letter_ratio": sum(
                d.features.other_script_letter_ratio
                > thresholds.max_other_script_letter_ratio
                for d in decisions
            ),
            "eligible_for_token_diversity_check": sum(
                d.features.token_count >= thresholds.diversity_min_tokens
                for d in decisions
            ),
            "below_distinct_token_ratio": sum(
                d.features.token_count >= thresholds.diversity_min_tokens
                and d.features.distinct_token_ratio
                < thresholds.min_distinct_token_ratio
                for d in decisions
            ),
            "above_dominant_token_ratio": sum(
                d.features.token_count >= thresholds.diversity_min_tokens
                and d.features.dominant_token_ratio
                > thresholds.max_dominant_token_ratio
                for d in decisions
            ),
        },
        "invalid_scalar_or_control_documents": sum(
            any(
                reason.startswith("invalid_")
                or reason == "disallowed_control_character"
                for reason in d.reasons
            )
            for d in decisions
        ),
    }


def run(args: argparse.Namespace) -> None:
    dependency_blobs = owner.bind_execution_head(args.expected_execution_head)
    need(
        owner.git(
            "rev-parse", "HEAD:tools/run_d03_languk_court_privacy_retest_v1.py"
        )
        == OWNER_RUNNER_BLOB,
        "owner runner blob drift",
    )
    need(
        owner.git("rev-parse", "HEAD:src/twelve_six/data/document_quality.py")
        == DOCUMENT_QUALITY_BLOB,
        "document-quality blob drift",
    )
    need(
        owner.git("rev-parse", "HEAD:src/twelve_six/data/quality_granularity.py")
        == GRANULARITY_BLOB,
        "granularity blob drift",
    )
    owner.verify_parent_audit(args.parent_audit)
    source_rows = owner.load_source(args.source_parquet)
    selected = _selected_rows(source_rows)
    selected_manifest = _selected_manifest_identity(selected)
    need(
        selected_manifest == EXPECTED_SELECTED_MANIFEST,
        "selected-manifest identity drift",
    )

    quality_inputs = _quality_inputs(selected)
    _verify_parent_quality(quality_inputs, selected_manifest)

    threshold_policy = default_quality_policy()
    threshold_manifest = threshold_policy.manifest()
    need(
        threshold_manifest["policy_sha256"] == EXPECTED_THRESHOLD_POLICY,
        "G05 threshold policy identity drift",
    )
    granularity = frozen_granularity_policy()
    need(
        granularity.manifest()["policy_sha256"] == EXPECTED_GRANULARITY_POLICY,
        "G05 granularity policy identity drift",
    )
    need(
        granularity.quality_threshold_policy_sha256 == EXPECTED_THRESHOLD_POLICY,
        "granularity/threshold policy cross-binding drift",
    )

    decisions = [
        assess_document(
            row["record_id"],
            row["normalized_text"],
            "uk",
            policy=threshold_policy,
        )
        for row in selected
    ]
    aggregate = aggregate_decisions(decisions)
    need(
        aggregate["documents"] == EXPECTED_RECORDS,
        "diagnostic document count drift",
    )

    core = {
        "schema_version": SCHEMA,
        "execution_head_sha": args.expected_execution_head,
        "owner_binding": {
            "pr": 2848,
            "head_sha": OWNER_HEAD,
            "runner_blob": OWNER_RUNNER_BLOB,
        },
        "source_binding": {
            "dataset": owner.DATASET,
            "revision": owner.REVISION,
            "file": owner.SOURCE_FILE,
            "sha256": owner.SOURCE_SHA256,
            "bytes": owner.SOURCE_BYTES,
            "family": owner.FAMILY,
        },
        "cohort_binding": {
            "selection_rule": "privacy_action_ALLOW_then_ascending_stable_source_id",
            "selected_records": EXPECTED_RECORDS,
            "selected_manifest_identity_sha256": selected_manifest,
            "parent_quality_execution_identity_sha256": EXPECTED_QUALITY_EXECUTION,
        },
        "quality_authority_binding": {
            "threshold_policy_id": threshold_policy.policy_id,
            "threshold_policy_sha256": EXPECTED_THRESHOLD_POLICY,
            "threshold_implementation_blob": DOCUMENT_QUALITY_BLOB,
            "granularity_policy_id": granularity.policy_id,
            "granularity_policy_sha256": EXPECTED_GRANULARITY_POLICY,
            "granularity_implementation_blob": GRANULARITY_BLOB,
            "natural_window_trigger_chars": granularity.natural_window_trigger_chars,
        },
        "diagnostic": {
            **aggregate,
            "documents_at_or_below_window_trigger": sum(
                d.features.chars <= granularity.natural_window_trigger_chars
                for d in decisions
            ),
            "documents_above_window_trigger": sum(
                d.features.chars > granularity.natural_window_trigger_chars
                for d in decisions
            ),
        },
        "dependency_git_blobs": dependency_blobs,
        "durable_evidence_text_free_aggregate_only": True,
        "product_policy_mutated": False,
        "capacity_or_training_authority_granted": False,
        "truth_boundary": dict(_TRUTH),
    }
    report = self_hashed(core, "diagnostic_identity_sha256")
    write_json(args.output_dir / "g05-aggregate-diagnostic.json", report)

    print(
        json.dumps(
            {
                "status": "LANGUK_G05_DIAGNOSTIC_ZERO_CREDIT",
                "documents": aggregate["documents"],
                "accepted_documents": aggregate["accepted_documents"],
                "rejected_documents": aggregate["rejected_documents"],
                "reason_counts": aggregate["reason_counts"],
                "diagnostic_identity_sha256": report[
                    "diagnostic_identity_sha256"
                ],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    value.add_argument("--expected-execution-head", required=True)
    value.add_argument("--parent-audit", type=Path, required=True)
    value.add_argument("--source-parquet", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
