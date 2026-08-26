"""NEXT100-065D canonical Research Corpus V1 cross-source dedup V6.

V6 reconciles the active V5 dedup graph with two exact-green authorities that
arrived after its inherited V4 cutoff: bounded NumPy code and the terminal
Project Gutenberg seal. It re-materializes accepted-only CPython rather than
crediting rejected chunks and runs the existing V3 exact/near/fragment/lineage
dedup engine over the complete composed object graph.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3
from twelve_six.data import cross_source_capacity_audit_v5 as v5

SCHEMA = "12-6.next100-065d-canonical-cross-source-dedup-report.v6"
INVENTORY_SCHEMA = "12-6.next100-065d-canonical-cross-source-dedup.v6"
WORKER_ID = "NEXT100-065D-CANONICAL-CROSSSOURCE-DEDUP-V6"
PG_NORMALIZER = "NEXT100_033_PG_BODY_NFC_LF_V1"
START_RE = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")
END_RE = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")


class CrossSourceV6Error(RuntimeError):
    """Fail-closed V6 composition, materialization, or authority error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossSourceV6Error(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrossSourceV6Error(f"{path}: JSON root must be an object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == INVENTORY_SCHEMA, "unsupported V6 config schema")
    _require(config.get("worker_id") == WORKER_ID, "V6 worker id drift")
    _require(config.get("local_free_only") is True, "V6 must remain LOCAL_FREE")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        _require(config.get(key) is False, f"V6 execution boundary weakened: {key}")

    parent = config["v5_parent"]
    _require(parent.get("pr") == 632, "V5 parent PR drift")
    _require(
        parent.get("head_sha") == "7fc6e3ec43ee7fb4361cd5d9b4e795bc3fd7c4b5",
        "V5 parent exact head drift",
    )
    _require(parent.get("source_object_count") == 23, "V5 source object count drift")
    _require(
        parent.get("source_family_counts") == {"uk": 4, "en": 4, "code": 4},
        "V5 source family vector drift",
    )
    _require(
        parent.get("fixed_capacity_without_cpython_accepted_chunks")
        == {"uk": 100856, "en": 150643, "code": 69133, "total": 320632},
        "V5 fixed capacity vector drift",
    )
    _require(
        parent.get("expected_cpython_accepted_capacity_bytes") == 15540,
        "accepted-only CPython capacity drift",
    )

    registry = config["canonical_registry_v3"]
    _require(
        registry.get("registry_identity_sha256")
        == "66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c",
        "canonical V3 registry identity drift",
    )
    _require(
        registry.get("numpy_authority_identity_sha256")
        == "e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70",
        "NumPy authority identity drift",
    )
    _require(registry.get("numpy_workflow_conclusion") == "success", "NumPy authority is not green")

    numpy = config["numpy"]
    _require(numpy.get("head_sha") == "bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8", "NumPy head drift")
    _require(numpy.get("source_family") == "github:numpy/numpy", "NumPy family drift")
    _require(numpy.get("training") == "ALLOWED", "NumPy training-right boundary drift")
    _require(numpy.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "NumPy evaluation boundary drift")
    selected = numpy.get("selected_files")
    _require(isinstance(selected, list) and len(selected) == 5, "NumPy exact five-file subset drift")
    _require(
        sum(int(row["raw_bytes"]) for row in selected) == numpy["expected_total_capacity_bytes"] == 36898,
        "NumPy byte-capacity arithmetic drift",
    )

    gutenberg = config["gutenberg"]
    _require(
        gutenberg.get("authority_identity_sha256")
        == "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b",
        "Gutenberg terminal seal identity drift",
    )
    _require(gutenberg.get("workflow_conclusion") == "success", "Gutenberg authority is not green")
    _require(
        gutenberg.get("training") == "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
        "Gutenberg training-right boundary drift",
    )
    _require(gutenberg.get("evaluation") == "NOT_AUTHORIZED", "Gutenberg evaluation boundary drift")
    records = gutenberg.get("records")
    _require(isinstance(records, list) and len(records) == 3, "Gutenberg exact three-record seal drift")
    _require(
        sum(int(row["normalized_utf8_bytes"]) for row in records)
        == gutenberg["expected_total_capacity_bytes"]
        == 1672110,
        "Gutenberg byte-capacity arithmetic drift",
    )

    expected = config["expected_pre_global_dedup_vector"]
    _require(expected.get("source_object_count") == 31, "V6 expected source object count drift")
    _require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "V6 expected family vector drift",
    )
    _require(
        expected.get("capacity_bytes")
        == {"uk": 100856, "en": 1838293, "code": 106031, "total": 2045180},
        "V6 expected capacity vector drift",
    )
    _require(expected.get("independent_family_count") == 14, "V6 family total drift")

    boundary = config["claim_boundary"]
    for key in (
        "canonical_registry_replaced",
        "corpus_materialized",
        "decontamination_pass_claimed",
        "balance_release_claimed",
        "postpack_unique_loss_ledger_complete",
        "tokenizer_fit_authorized",
        "training_authorized",
        "paid_compute_authorized",
        "research_corpus_v1_terminal",
    ):
        _require(boundary.get(key) is False, f"V6 truth boundary weakened: {key}")


def _normalize_gutenberg_body(raw: bytes, encoding: str) -> bytes:
    """Reproduce NEXT100-033 PG body extraction exactly."""
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise CrossSourceV6Error(f"Gutenberg decode failure under {encoding}: {exc}") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    starts = [index for index, line in enumerate(lines) if START_RE.match(line.strip())]
    ends = [index for index, line in enumerate(lines) if END_RE.match(line.strip())]
    _require(len(starts) == 1, f"Gutenberg requires exactly one START marker, got {len(starts)}")
    _require(len(ends) == 1, f"Gutenberg requires exactly one END marker, got {len(ends)}")
    _require(ends[0] > starts[0], "Gutenberg END marker must follow START marker")
    body_lines = lines[starts[0] + 1 : ends[0]]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    body = "\n".join(body_lines)
    if body.startswith("\ufeff"):
        body = body[1:]
    return (unicodedata.normalize("NFC", body) + "\n").encode("utf-8")


def _materialize_numpy(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    total = 0
    for item in spec["selected_files"]:
        path = str(item["path"])
        url = f"https://raw.githubusercontent.com/numpy/numpy/{spec['upstream_commit']}/{path}"
        raw = v1.fetch_exact_source(url)
        _require(len(raw) == int(item["raw_bytes"]), f"NumPy {path}: raw byte-count drift")
        _require(_git_blob_sha1(raw) == item["git_blob_sha1"], f"NumPy {path}: Git blob drift")
        try:
            raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CrossSourceV6Error(f"NumPy {path}: strict UTF-8 failure") from exc
        _require(b"\x00" not in raw, f"NumPy {path}: NUL payload rejected")
        source_id = f"code.numpy.{path.replace('/', '.')}"
        row = {
            "source_id": source_id,
            "source_family": spec["source_family"],
            "stable_origin_id": spec["stable_origin_id"],
            "stable_object_id": f"git-sha1:{item['git_blob_sha1']}",
            "modality": "code",
            "evidence_status": "DEDICATED_TERMINAL",
            "authority_ref": f"NEXT100-049@{spec['head_sha']} exact NumPy authority",
            "declared_capacity_bytes": len(raw),
            "expected_raw_bytes": len(raw),
            "expected_raw_sha256": _sha256(raw),
            "acquisition_url": url,
            "origin_key": f"github:numpy/numpy:{spec['upstream_commit']}:{path}",
        }
        rows.append(row)
        payloads[source_id] = raw
        evidence.append(
            {
                "source_id": source_id,
                "path": path,
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
                "git_blob_sha1": _git_blob_sha1(raw),
            }
        )
        total += len(raw)
    _require(total == int(spec["expected_total_capacity_bytes"]), "NumPy materialized total drift")
    return rows, payloads, evidence


def _materialize_gutenberg(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    total = 0
    for item in spec["records"]:
        url = (
            f"https://raw.githubusercontent.com/{item['transport_repo']}/"
            f"{item['transport_commit']}/{item['transport_path']}"
        )
        raw = v1.fetch_exact_source(url)
        source_id = str(item["source_id"])
        _require(len(raw) == int(item["raw_bytes"]), f"{source_id}: Gutenberg raw byte-count drift")
        _require(_sha256(raw) == item["raw_sha256"], f"{source_id}: Gutenberg raw SHA-256 drift")
        _require(
            _git_blob_sha1(raw) == item["transport_git_blob_sha1"],
            f"{source_id}: Gutenberg Git blob drift",
        )
        normalized = _normalize_gutenberg_body(raw, str(item["encoding"]))
        _require(
            len(normalized) == int(item["normalized_utf8_bytes"]),
            f"{source_id}: Gutenberg normalized byte-count drift",
        )
        _require(
            _sha256(normalized) == item["normalized_sha256"],
            f"{source_id}: Gutenberg normalized SHA-256 drift",
        )
        row = {
            "source_id": source_id,
            "source_family": spec["source_family"],
            "stable_origin_id": spec["stable_origin_id"],
            "stable_object_id": f"sha256:{item['normalized_sha256']}",
            "modality": "en",
            "evidence_status": "DEDICATED_TERMINAL",
            "authority_ref": f"NEXT100-107@{spec['head_sha']} terminal Gutenberg seal",
            "declared_capacity_bytes": len(normalized),
            "expected_raw_bytes": len(normalized),
            "expected_raw_sha256": _sha256(normalized),
            "acquisition_url": f"materialized-v6://{source_id}",
            "origin_key": f"gutenberg:{source_id}",
        }
        rows.append(row)
        payloads[source_id] = normalized
        evidence.append(
            {
                "source_id": source_id,
                "normalizer_id": PG_NORMALIZER,
                "transport_raw_bytes": len(raw),
                "transport_raw_sha256": _sha256(raw),
                "transport_git_blob_sha1": _git_blob_sha1(raw),
                "normalized_utf8_bytes": len(normalized),
                "normalized_sha256": _sha256(normalized),
            }
        )
        total += len(normalized)
    _require(total == int(spec["expected_total_capacity_bytes"]), "Gutenberg materialized total drift")
    return rows, payloads, evidence


def _materialize_v5(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]], int]:
    """Rebuild V5 inputs without trusting a prior V5 report."""
    v5._validate_config(v5_config)
    merged, payloads, inherited_evidence = v5._materialize_v4(base_inventory, v4_extension)
    mdn_payload, mdn_evidence = v5._materialize_mdn(v5_config["mdn"])
    cpython_payload, cpython_capacity, cpython_evidence = v5._materialize_cpython(v5_config["cpython"])

    mdn = v5_config["mdn"]
    cpython = v5_config["cpython"]
    mdn_row = {
        "source_id": mdn["source_id"],
        "source_family": mdn["source_family"],
        "stable_origin_id": mdn["stable_origin_id"],
        "stable_object_id": mdn["stable_object_id"],
        "modality": mdn["modality"],
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": f"NEXT100-038@{mdn['head_sha']} workflow {mdn['dedicated_workflow_run']}",
        "declared_capacity_bytes": mdn["normalized_bytes"],
        "expected_raw_bytes": mdn["normalized_bytes"],
        "expected_raw_sha256": mdn["normalized_sha256"],
        "acquisition_url": "materialized-v6://mdn-prose-only",
        "origin_key": f"github:mdn/content:{mdn['upstream_commit']}:{mdn['upstream_path']}",
    }
    cpython_row = {
        "source_id": cpython["source_id"],
        "source_family": cpython["source_family"],
        "stable_origin_id": cpython["stable_origin_id"],
        "stable_object_id": cpython["stable_object_id"],
        "modality": cpython["modality"],
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": f"NEXT100-037@{cpython['head_sha']} workflow {cpython['dedicated_workflow_run']} accepted-only",
        "declared_capacity_bytes": cpython_capacity,
        "expected_raw_bytes": len(cpython_payload),
        "expected_raw_sha256": _sha256(cpython_payload),
        "acquisition_url": "materialized-v6://cpython-data228-accepted-chunks",
        "origin_key": f"github:python/cpython:{cpython['upstream_commit']}:{cpython['upstream_path']}:accepted-only",
    }
    final_inventory = copy.deepcopy(merged)
    final_inventory["sources"] = [*copy.deepcopy(merged["sources"]), mdn_row, cpython_row]
    final_inventory["final_refresh_required"] = False
    final_inventory["terminal_refresh_rule"] = "V6 independently re-materializes the complete V5 graph before adding post-V5 authorities."
    payloads = dict(payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads[cpython["source_id"]] = cpython_payload
    evidence = [*inherited_evidence, mdn_evidence, cpython_evidence]
    _require(len(final_inventory["sources"]) == 23, "re-materialized V5 object count drift")
    return final_inventory, payloads, evidence, cpython_capacity


def audit_live(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    v6_config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_config(v6_config)
    inventory, payloads, inherited_evidence, cpython_capacity = _materialize_v5(
        base_inventory, v4_extension, v5_config
    )
    _require(
        cpython_capacity == int(v6_config["v5_parent"]["expected_cpython_accepted_capacity_bytes"]),
        f"accepted-only CPython exact capacity drift: {cpython_capacity}",
    )

    numpy_rows, numpy_payloads, numpy_evidence = _materialize_numpy(v6_config["numpy"])
    gutenberg_rows, gutenberg_payloads, gutenberg_evidence = _materialize_gutenberg(v6_config["gutenberg"])

    final_inventory = copy.deepcopy(inventory)
    final_inventory["sources"] = [
        *copy.deepcopy(inventory["sources"]),
        *numpy_rows,
        *gutenberg_rows,
    ]
    final_inventory["final_refresh_required"] = False
    final_inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T20:00:00Z"
    final_inventory["terminal_refresh_rule"] = (
        "V6 composes V5 accepted-only CPython/MDN with exact-green NumPy and the exact terminal Gutenberg seal; "
        "failed/nonterminal siblings remain zero-credit."
    )
    payloads = {**payloads, **numpy_payloads, **gutenberg_payloads}

    dedup = v3.audit_payloads(final_inventory, payloads)
    v3.verify_report(dedup)
    expected = v6_config["expected_pre_global_dedup_vector"]
    _require(dedup["source_count"] == expected["source_object_count"], "V6 source object count drift")
    scope = dedup["terminal_candidates"]
    counts = {
        modality: int(summary["declared_source_family_count"])
        for modality, summary in scope["by_modality"].items()
    }
    _require(counts == expected["source_family_counts"], f"V6 family vector drift: {counts}")
    capacity = expected["capacity_bytes"]
    _require(scope["declared_capacity_bytes_before"] == capacity["total"], "V6 total pre-dedup capacity drift")
    for modality in ("uk", "en", "code"):
        _require(
            scope["by_modality"][modality]["declared_capacity_bytes_before"] == capacity[modality],
            f"V6 {modality} pre-dedup capacity drift",
        )

    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "authority_reconciliation": {
            "v5_parent_head_sha": v6_config["v5_parent"]["head_sha"],
            "canonical_registry_v3_identity_sha256": v6_config["canonical_registry_v3"]["registry_identity_sha256"],
            "numpy_authority_identity_sha256": v6_config["canonical_registry_v3"]["numpy_authority_identity_sha256"],
            "gutenberg_authority_identity_sha256": v6_config["gutenberg"]["authority_identity_sha256"],
            "accepted_cpython_capacity_bytes": cpython_capacity,
        },
        "source_vector": {
            "source_object_count": dedup["source_count"],
            "source_family_counts": counts,
            "independent_family_count": sum(counts.values()),
            "pre_global_dedup_capacity_bytes": copy.deepcopy(capacity),
            "conservative_unique_capacity_bytes_after_global_dedup": scope[
                "conservative_unique_capacity_bytes_after"
            ],
            "duplicate_discount_bytes": scope["duplicate_discount_bytes"],
        },
        "new_materialization_evidence": {
            "numpy": numpy_evidence,
            "gutenberg": gutenberg_evidence,
            "cpython": next(
                item
                for item in inherited_evidence
                if item["source_id"] == "en.python.docs.tutorial-introduction"
            ),
        },
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(v6_config["claim_boundary"]),
        "remaining_blockers": [
            "SOURCE_CAPACITY_AND_BALANCE_FAMILY_CAP_CONVERGENCE",
            "FREEZE_EXACT_PREDECONTAM_RECORD_INVENTORY",
            "RESERVED_EVALUATION_DECONTAMINATION",
            "POST_COMPOSITION_QUALITY_PRIVACY_REVALIDATION",
            "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACKING",
            "TWO_CLEAN_BYTE_IDENTICAL_BUILDS",
            "POSTPACK_UNIQUE_NONIGNORED_CAUSAL_LOSS_LEDGER",
            "D05_CHECKPOINT_INTEGRITY_TERMINAL_PROOF",
            "BOUNDED_MODEL341_TRAINING_REQUALIFICATION",
            "EXPLICIT_MATERIAL_COMPUTE_AUTHORIZATION",
        ],
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA, "unsupported V6 report schema")
    expected_hash = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected_hash == _sha256(_canonical_bytes(core)), "V6 report self-hash mismatch")
    _require(report.get("local_free_only") is True, "V6 LOCAL_FREE invariant failed")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
        "raw_text_emitted",
    ):
        _require(report.get(key) is False, f"V6 execution/text boundary failed: {key}")
    vector = report["source_vector"]
    _require(vector.get("source_object_count") == 31, "V6 report object count drift")
    _require(
        vector.get("source_family_counts") == {"code": 5, "en": 5, "uk": 4},
        "V6 report family vector drift",
    )
    _require(vector.get("independent_family_count") == 14, "V6 report family total drift")
    _require(
        vector.get("pre_global_dedup_capacity_bytes")
        == {"uk": 100856, "en": 1838293, "code": 106031, "total": 2045180},
        "V6 report pre-dedup capacity drift",
    )
    _require(
        0 <= vector["conservative_unique_capacity_bytes_after_global_dedup"] <= 2045180,
        "V6 dedup capacity invariant failed",
    )
    _require(
        report["authority_reconciliation"]["accepted_cpython_capacity_bytes"] == 15540,
        "V6 accepted-only CPython capacity drift",
    )
    cp = report["new_materialization_evidence"]["cpython"]
    _require(cp.get("accepted_chunk_count") == 14, "V6 CPython accepted count drift")
    _require(cp.get("rejected_chunk_count") == 2, "V6 CPython rejected count drift")
    _require(cp.get("rejection_reasons") == {"pii_phone": 2}, "V6 CPython privacy rejection drift")
    boundary = report["claim_boundary"]
    for value in boundary.values():
        _require(value is False, "V6 truth boundary contains a promoted claim")
    v3.verify_report(report["dedup_v3"])


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    v1.write_report(report, path)


def load_inputs(
    base_inventory_path: str | Path,
    v4_extension_path: str | Path,
    v5_config_path: str | Path,
    v6_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        _load_json(base_inventory_path),
        _load_json(v4_extension_path),
        _load_json(v5_config_path),
        _load_json(v6_config_path),
    )
