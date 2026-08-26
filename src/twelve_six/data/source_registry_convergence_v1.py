"""NEXT100-063 scoped terminal-source convergence for the CPython docs family."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3
from twelve_six.data.pipeline import PipelineConfig, _quality_reason, normalize_text

CONFIG_SCHEMA = "12-6.next100-063-source-registry-convergence-cpython.v1"
REPORT_SCHEMA = "12-6.next100-063-source-registry-convergence-cpython-report.v1"
WORKER_ID = "NEXT100-063-SOURCE-REGISTRY-CONVERGENCE-CPYTHON"
MATERIALIZATION_POLICY = "DATA228_ACCEPTED_CHUNKS_ONLY_V1"


class SourceRegistryConvergenceError(RuntimeError):
    """Fail-closed NEXT100-063 contract error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceRegistryConvergenceError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceRegistryConvergenceError(f"{path}: JSON root must be an object")
    return value


def _chunk_text(text: str, *, max_chars: int, min_chars: int) -> tuple[str, ...]:
    """Reproduce the DATA-228 generic natural-text chunker exactly."""
    if max_chars < min_chars or min_chars < 20:
        raise SourceRegistryConvergenceError("invalid chunk limits")
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            value = "\n".join(current).strip()
            if len(value) >= min_chars:
                chunks.append(value)
            current, current_len = [], 0

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces = [paragraph]
        else:
            pieces: list[str] = []
            words = paragraph.split()
            part: list[str] = []
            part_len = 0
            for word in words:
                needed = len(word) if not part else len(word) + 1
                if part and part_len + needed > max_chars:
                    pieces.append(" ".join(part))
                    part = [word]
                    part_len = len(word)
                else:
                    part.append(word)
                    part_len += needed
            if part:
                pieces.append(" ".join(part))

        for piece in pieces:
            needed = len(piece) if not current else len(piece) + 1
            if current and current_len + needed > max_chars:
                flush()
            current.append(piece)
            current_len += needed
    flush()
    return tuple(chunks)


def _quality_config(spec: Mapping[str, Any]) -> PipelineConfig:
    policy = spec["quality_privacy"]["quality_policy"]
    _require(policy.get("reject_control_characters") is True, "control-character gate weakened")
    _require(policy.get("reject_email") is True, "email gate weakened")
    _require(policy.get("reject_phone") is True, "phone gate weakened")
    return PipelineConfig(
        split_seed="12-6-data228-probe-v1",
        validation_fraction=0.20,
        min_chars=int(policy["min_chars"]),
        max_chars=int(policy["max_chars"]),
        min_alpha_ratio=float(policy["min_alpha_ratio"]),
        near_duplicate_threshold=0.92,
        near_duplicate_shingle_words=5,
        tiny_near_dedup_max_documents=5000,
    )


def materialize_authorized_text(raw: bytes, addition: Mapping[str, Any]) -> dict[str, Any]:
    """Verify upstream bytes and materialize only DATA-228 accepted chunks."""
    source = addition["source"]
    quality = addition["quality_privacy"]
    _require(
        quality.get("materialization_policy") == MATERIALIZATION_POLICY,
        "unsupported accepted-chunk materialization policy",
    )
    _require(len(raw) == source["raw_bytes"], "CPython upstream byte count drift")
    _require(v1._sha256(raw) == source["raw_sha256"], "CPython upstream SHA-256 drift")
    _require(
        v1._git_blob_sha1(raw) == source["source_git_blob_sha1"],
        "CPython upstream Git blob drift",
    )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceRegistryConvergenceError("CPython source is not strict UTF-8") from exc

    normalization = source["normalization"]
    bounded = text[: int(normalization["truncate_chars"])]
    normalized = normalize_text(bounded)
    normalized_bytes = normalized.encode("utf-8")
    _require(
        len(normalized_bytes) == normalization["normalized_utf8_bytes"],
        "DATA-228 normalized byte count drift",
    )
    _require(
        v1._sha256(normalized_bytes) == normalization["normalized_sha256"],
        "DATA-228 normalized SHA-256 drift",
    )

    chunking = quality["chunking"]
    chunks = _chunk_text(
        normalized,
        max_chars=int(chunking["max_chars"]),
        min_chars=int(chunking["min_chars"]),
    )
    _require(len(chunks) == quality["chunk_count"], "DATA-228 chunk count drift")

    config = _quality_config(addition)
    accepted: list[str] = []
    accepted_hashes: list[str] = []
    rejection_reasons: Counter[str] = Counter()
    for chunk in chunks:
        reason = _quality_reason(chunk, config)
        if reason is not None:
            rejection_reasons[reason] += 1
            continue
        accepted_chunk = normalize_text(chunk)
        accepted.append(accepted_chunk)
        accepted_hashes.append(v1._sha256(accepted_chunk.encode("utf-8")))

    _require(
        len(accepted) == quality["accepted_chunk_count"],
        "accepted chunk count drift",
    )
    _require(
        len(chunks) - len(accepted) == quality["rejected_chunk_count"],
        "rejected chunk count drift",
    )
    _require(
        dict(sorted(rejection_reasons.items())) == quality["rejection_reasons"],
        "quality/privacy rejection reasons drift",
    )
    _require(
        accepted_hashes == quality["accepted_normalized_sha256"],
        "accepted chunk identity/order drift",
    )
    _require(
        len(accepted_hashes) == len(set(accepted_hashes)),
        "duplicate accepted chunk identity",
    )

    payload = "\n\n".join(accepted).encode("utf-8")
    capacity = sum(len(chunk.encode("utf-8")) for chunk in accepted)
    _require(capacity > 0 and payload, "accepted materialization is empty")
    return {
        "payload": payload,
        "declared_capacity_bytes": capacity,
        "materialized_payload_bytes": len(payload),
        "materialized_payload_sha256": v1._sha256(payload),
        "normalized_source_bytes": len(normalized_bytes),
        "normalized_source_sha256": v1._sha256(normalized_bytes),
        "accepted_chunk_count": len(accepted),
        "accepted_chunk_sha256": accepted_hashes,
        "rejected_chunk_count": len(chunks) - len(accepted),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "raw_bytes": len(raw),
        "raw_sha256": v1._sha256(raw),
        "raw_git_blob_sha1": v1._git_blob_sha1(raw),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "unsupported config schema")
    _require(config.get("worker_id") == WORKER_ID, "worker id drift")
    _require(config.get("local_free_only") is True, "LOCAL_FREE boundary weakened")
    _require(config.get("model_training_executed") is False, "training boundary weakened")
    base = config["base"]
    _require(base.get("head_sha") == "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13", "base head drift")
    _require(base.get("dedicated_workflow_conclusion") == "success", "base dedup gate not green")
    authority = config["addition"]["authority"]
    _require(authority.get("terminal_verdict") == "ADMIT", "source authority is not ADMIT")
    _require(authority.get("dedicated_workflow_conclusion") == "success", "source authority gate not green")
    rights = config["addition"]["rights"]
    _require(rights.get("model_training") == "ALLOWED", "model-training rights not allowed")
    _require(rights.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "evaluation-purpose leak")
    boundary = config["claim_boundary"]
    for key in (
        "canonical_registry_replaced",
        "corpus_materialized",
        "decontamination_pass_claimed",
        "tokenizer_fit_authorized",
        "training_authorized",
        "paid_compute_used",
        "twenty_megabyte_corpus_ready",
    ):
        _require(boundary.get(key) is False, f"claim boundary weakened: {key}")


def _base_inventory(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    base = config["base"]
    inventory = _load_json(repo_root / base["inventory_path"])
    _require(inventory.get("schema_version") == base["inventory_schema"], "base inventory schema drift")
    _require(len(inventory.get("sources", [])) == base["expected_source_count"], "base source count drift")
    rows, _ = v3._validate_inventory(inventory)
    family_counts = Counter()
    by_modality: dict[str, set[str]] = {}
    for row in rows:
        by_modality.setdefault(row["modality"], set()).add(row["source_family"])
    for modality, families in by_modality.items():
        family_counts[modality] = len(families)
    _require(dict(family_counts) == base["expected_family_counts"], "base family vector drift")
    return inventory


def _transient_inventory(
    base_inventory: Mapping[str, Any],
    config: Mapping[str, Any],
    materialized: Mapping[str, Any],
) -> dict[str, Any]:
    source = config["addition"]["source"]
    row = {
        "source_id": source["source_id"],
        "source_family": source["source_family"],
        "stable_origin_id": source["stable_origin_id"],
        "stable_object_id": source["stable_object_id"],
        "modality": source["modality"],
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": (
            f"NEXT100-037 exact head {config['addition']['authority']['head_sha']} "
            f"workflow {config['addition']['authority']['dedicated_workflow_run']}"
        ),
        "declared_capacity_bytes": materialized["declared_capacity_bytes"],
        "expected_raw_bytes": materialized["materialized_payload_bytes"],
        "expected_raw_sha256": materialized["materialized_payload_sha256"],
        "acquisition_url": "materialized://next100-063/cpython-data228-accepted-chunks",
        "origin_key": (
            f"github:python/cpython:{source['upstream_commit']}:{source['upstream_path']}"
        ),
    }
    combined = dict(base_inventory)
    combined["sources"] = [dict(item) for item in base_inventory["sources"]] + [row]
    combined["lineage_edges"] = [dict(item) for item in base_inventory.get("lineage_edges", [])]
    return combined


def _family_counts(report: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    by_modality = report["terminal_candidates"]["by_modality"]
    for modality, summary in by_modality.items():
        result[str(modality)] = int(summary["declared_source_family_count"])
    return result


def _assert_family_gate(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    added_source_id: str,
) -> dict[str, int]:
    counts = _family_counts(report)
    required = config["family_gate"]["required_family_counts_after"]
    _require(counts == required, f"post-convergence family vector drift: {counts}")
    minimum = int(config["family_gate"]["minimum_independent_families_per_stratum"])
    _require(all(count >= minimum for count in counts.values()), "minimum family gate remains blocked")
    if config["family_gate"]["forbid_cross_family_capacity_collapse_for_added_source"]:
        for match in report["matches"]:
            involves_added = added_source_id in {
                match["left_source_id"],
                match["right_source_id"],
            }
            if (
                involves_added
                and match.get("cross_source_family") is True
                and match.get("capacity_collapsing") is True
            ):
                raise SourceRegistryConvergenceError(
                    "added source has a cross-family capacity-collapsing match"
                )
    return counts


def audit_live(
    config: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    _validate_config(config)
    root = Path(repo_root)
    base_inventory = _base_inventory(root, config)
    base_rows, _ = v3._validate_inventory(base_inventory)
    base_payloads = {
        row["source_id"]: v1.fetch_exact_source(row["acquisition_url"])
        for row in base_rows
    }

    source = config["addition"]["source"]
    upstream_raw = v1.fetch_exact_source(source["acquisition_url"])
    materialized = materialize_authorized_text(upstream_raw, config["addition"])
    transient_inventory = _transient_inventory(base_inventory, config, materialized)
    payloads = dict(base_payloads)
    payloads[source["source_id"]] = materialized["payload"]
    dedup = v3.audit_payloads(transient_inventory, payloads)
    v3.verify_report(dedup)
    counts = _assert_family_gate(dedup, config, source["source_id"])

    core = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "base": {
            "head_sha": config["base"]["head_sha"],
            "dedicated_workflow_run": config["base"]["dedicated_workflow_run"],
            "source_count": config["base"]["expected_source_count"],
            "dedup_report_sha256": dedup["report_sha256"],
        },
        "terminal_addition": {
            "source_id": source["source_id"],
            "source_family": source["source_family"],
            "authority_head_sha": config["addition"]["authority"]["head_sha"],
            "authority_identity_sha256": config["addition"]["authority"]["authority_identity_sha256"],
            "authority_workflow_run": config["addition"]["authority"]["dedicated_workflow_run"],
            "upstream_raw_bytes": materialized["raw_bytes"],
            "upstream_raw_sha256": materialized["raw_sha256"],
            "upstream_git_blob_sha1": materialized["raw_git_blob_sha1"],
            "normalized_source_bytes": materialized["normalized_source_bytes"],
            "normalized_source_sha256": materialized["normalized_source_sha256"],
            "accepted_chunk_count": materialized["accepted_chunk_count"],
            "accepted_chunk_sha256": materialized["accepted_chunk_sha256"],
            "rejected_chunk_count": materialized["rejected_chunk_count"],
            "rejection_reasons": materialized["rejection_reasons"],
            "declared_training_capacity_bytes": materialized["declared_capacity_bytes"],
            "materialized_payload_bytes": materialized["materialized_payload_bytes"],
            "materialized_payload_sha256": materialized["materialized_payload_sha256"],
        },
        "source_registry_convergence": {
            "status": "PASS_SCOPED_TERMINAL_EN_FAMILY_CONVERGENCE",
            "source_count": dedup["source_count"],
            "family_counts": counts,
            "minimum_family_gate": "PASS",
            "en_hard_family_blocker_resolved_for_this_exact_vector": True,
            "conservative_unique_capacity_bytes_after": dedup["terminal_candidates"][
                "conservative_unique_capacity_bytes_after"
            ],
            "dedup_source_count": dedup["source_count"],
        },
        "dedup_report": dedup,
        "remaining_blockers": [
            "20M_SOURCE_BYTE_ACQUISITION_GAP_REMAINS",
            "GLOBAL_TERMINAL_SOURCE_REGISTRY_CONVERGENCE_BEYOND_THIS_SCOPED_ADDITION_REMAINS",
            "EVALUATION_RESERVATION_AND_CORPUS_MATERIALIZATION_REMAIN",
            "DECONTAMINATION_PASS_REMAINS",
            "POSTPACK_UNIQUE_LOSS_LEDGER_REMAINS",
            "TOKENIZER_FIT_NOT_AUTHORIZED",
            "MODEL_TRAINING_NOT_AUTHORIZED"
        ],
        "claim_boundary": dict(config["claim_boundary"]),
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": v1._sha256(v1._canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == REPORT_SCHEMA, "unsupported report schema")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected == v1._sha256(v1._canonical_bytes(core)), "report self-hash mismatch")
    _require(report.get("local_free_only") is True, "LOCAL_FREE invariant failed")
    _require(report.get("model_training_executed") is False, "training invariant failed")
    _require(report.get("raw_text_emitted") is False, "raw text emission is forbidden")
    convergence = report.get("source_registry_convergence", {})
    _require(
        convergence.get("status") == "PASS_SCOPED_TERMINAL_EN_FAMILY_CONVERGENCE",
        "scoped convergence not PASS",
    )
    _require(convergence.get("family_counts") == {"code": 4, "en": 2, "uk": 2}, "family gate drift")
    _require(
        convergence.get("en_hard_family_blocker_resolved_for_this_exact_vector") is True,
        "EN family blocker not resolved",
    )
    addition = report.get("terminal_addition", {})
    _require(addition.get("accepted_chunk_count") == 14, "accepted chunk count drift")
    _require(addition.get("rejected_chunk_count") == 2, "rejected chunk count drift")
    _require(addition.get("rejection_reasons") == {"pii_phone": 2}, "privacy rejection drift")
    boundary = report.get("claim_boundary", {})
    for key in (
        "canonical_registry_replaced",
        "corpus_materialized",
        "decontamination_pass_claimed",
        "tokenizer_fit_authorized",
        "training_authorized",
        "paid_compute_used",
        "twenty_megabyte_corpus_ready",
    ):
        _require(boundary.get(key) is False, f"truth boundary weakened: {key}")
    v3.verify_report(report["dedup_report"])


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    v1.write_report(report, path)
