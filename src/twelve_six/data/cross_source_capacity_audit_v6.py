"""NEXT100-065D successor global dedup over NumPy and Gutenberg terminal sources."""

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

SCHEMA = "12-6.next100-065d-cross-source-dedup-report.v6"
INVENTORY_SCHEMA = "12-6.next100-065d-cross-source-dedup.v6"
WORKER_ID = "NEXT100-065D-CROSSSOURCE-DEDUP-V6"
PG_NORMALIZER = "NEXT100_033_PG_BODY_NFC_LF_V1"
NUMPY_NORMALIZER = "STRICT_UTF8_IDENTITY_PRESERVE_V1"

START_RE = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")
END_RE = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")


class CrossSourceV6Error(RuntimeError):
    """Fail-closed V6 source-materialization or authority error."""


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
    _require(config.get("local_free_only") is True, "LOCAL_FREE boundary weakened")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        _require(config.get(key) is False, f"execution boundary weakened: {key}")

    base = config["base_v5"]
    _require(
        base.get("head_sha") == "7fc6e3ec43ee7fb4361cd5d9b4e795bc3fd7c4b5",
        "V5 base head drift",
    )
    _require(base.get("source_object_count") == 23, "V5 source count drift")
    _require(
        base.get("source_family_counts") == {"uk": 4, "en": 4, "code": 4},
        "V5 family vector drift",
    )
    _require(base.get("cpython_accepted_capacity_bytes") == 15540, "V5 CPython capacity drift")
    _require(base.get("expected_source_capacity_bytes") == 336172, "V5 capacity drift")

    numpy_cfg = config["numpy"]
    _require(numpy_cfg.get("dedicated_workflow_conclusion") == "success", "NumPy authority not green")
    _require(
        numpy_cfg.get("authority_identity_sha256")
        == "e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70",
        "NumPy authority identity drift",
    )
    _require(numpy_cfg.get("normalization_policy") == NUMPY_NORMALIZER, "NumPy normalization drift")
    _require(numpy_cfg.get("training") == "ALLOWED", "NumPy training-rights drift")
    _require(
        numpy_cfg.get("evaluation") == "NOT_SEPARATELY_ADMITTED",
        "NumPy evaluation boundary drift",
    )
    _require(sum(int(row["raw_bytes"]) for row in numpy_cfg["files"]) == 36898, "NumPy capacity drift")

    pg = config["gutenberg"]
    _require(pg.get("dedicated_workflow_conclusion") == "success", "Gutenberg authority not green")
    _require(
        pg.get("authority_identity_sha256")
        == "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b",
        "Gutenberg authority identity drift",
    )
    _require(pg.get("normalization_policy") == PG_NORMALIZER, "Gutenberg normalization drift")
    _require(
        pg.get("training") == "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
        "Gutenberg training-rights drift",
    )
    _require(pg.get("evaluation") == "NOT_AUTHORIZED", "Gutenberg evaluation boundary drift")
    _require(pg.get("worldwide_public_domain_claim") is False, "Gutenberg rights boundary broadened")
    _require(
        sum(int(row["normalized_utf8_bytes"]) for row in pg["records"]) == 1672110,
        "Gutenberg capacity drift",
    )

    expected = config["expected_vector"]
    _require(expected.get("source_object_count") == 31, "V6 expected source count drift")
    _require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "V6 expected family vector drift",
    )
    _require(
        expected.get("source_capacity_bytes_before_global_dedup") == 2045180,
        "V6 expected total capacity drift",
    )
    _require(
        expected.get("source_capacity_by_modality_before_global_dedup")
        == {"uk": 100856, "en": 1838293, "code": 106031},
        "V6 expected modality capacity drift",
    )

    for key, value in config["claim_boundary"].items():
        _require(value is False, f"claim boundary weakened: {key}")


def _normalize_pg_body(raw: bytes, encoding: str) -> bytes:
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise CrossSourceV6Error(
            f"Gutenberg decode failure under preregistered encoding {encoding}: {exc}"
        ) from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if START_RE.match(line.strip())]
    ends = [i for i, line in enumerate(lines) if END_RE.match(line.strip())]
    _require(len(starts) == 1, f"expected one Gutenberg START marker, got {len(starts)}")
    _require(len(ends) == 1, f"expected one Gutenberg END marker, got {len(ends)}")
    _require(ends[0] > starts[0], "Gutenberg END marker must follow START marker")

    body_lines = lines[starts[0] + 1 : ends[0]]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    body = "\n".join(body_lines)
    if body.startswith("\ufeff"):
        body = body[1:]
    body = unicodedata.normalize("NFC", body)
    return (body + "\n").encode("utf-8")


def _materialize_numpy(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    repo = "numpy/numpy"
    commit = str(spec["upstream_commit"])
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    for item in spec["files"]:
        path = str(item["path"])
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        raw = v1.fetch_exact_source(url)
        _require(len(raw) == int(item["raw_bytes"]), f"NumPy byte-count drift: {path}")
        _require(_git_blob_sha1(raw) == item["git_blob_sha1"], f"NumPy Git blob drift: {path}")
        raw.decode("utf-8", errors="strict")
        source_id = f"code.numpy.{path}"
        digest = _sha256(raw)
        rows.append(
            {
                "source_id": source_id,
                "source_family": spec["repository_family"],
                "stable_origin_id": spec["repository_family"],
                "stable_object_id": f"git-sha1:{item['git_blob_sha1']}",
                "modality": "code",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": (
                    f"NEXT100-049@{spec['head_sha']} workflow "
                    f"{spec['dedicated_workflow_run']}"
                ),
                "declared_capacity_bytes": len(raw),
                "expected_raw_bytes": len(raw),
                "expected_raw_sha256": digest,
                "acquisition_url": url,
                "origin_key": f"github:numpy/numpy:{commit}:{path}",
            }
        )
        payloads[source_id] = raw
        evidence.append(
            {
                "source_id": source_id,
                "path": path,
                "raw_bytes": len(raw),
                "raw_sha256": digest,
                "git_blob_sha1": _git_blob_sha1(raw),
                "normalization_policy": NUMPY_NORMALIZER,
            }
        )
    _require(sum(len(value) for value in payloads.values()) == 36898, "NumPy materialized capacity drift")
    return rows, payloads, evidence


def _materialize_gutenberg(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    for item in spec["records"]:
        url = (
            f"https://raw.githubusercontent.com/{item['transport_repo']}/"
            f"{item['transport_commit']}/{item['transport_path']}"
        )
        raw = v1.fetch_exact_source(url)
        source_id = str(item["source_id"])
        _require(len(raw) == int(item["raw_bytes"]), f"Gutenberg raw byte-count drift: {source_id}")
        _require(_sha256(raw) == item["raw_sha256"], f"Gutenberg raw SHA-256 drift: {source_id}")
        _require(
            _git_blob_sha1(raw) == item["transport_git_blob_sha1"],
            f"Gutenberg Git blob drift: {source_id}",
        )
        normalized = _normalize_pg_body(raw, str(item["encoding"]))
        _require(
            len(normalized) == int(item["normalized_utf8_bytes"]),
            f"Gutenberg normalized byte-count drift: {source_id}",
        )
        _require(
            _sha256(normalized) == item["normalized_sha256"],
            f"Gutenberg normalized SHA-256 drift: {source_id}",
        )
        rows.append(
            {
                "source_id": source_id,
                "source_family": spec["family_id"],
                "stable_origin_id": f"project-gutenberg:{item['ebook_id']}",
                "stable_object_id": f"sha256:{item['normalized_sha256']}",
                "modality": "en",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": (
                    f"NEXT100-107@{spec['head_sha']} parent {spec['parent_head_sha']} "
                    f"workflow {spec['dedicated_workflow_run']}"
                ),
                "declared_capacity_bytes": len(normalized),
                "expected_raw_bytes": len(normalized),
                "expected_raw_sha256": _sha256(normalized),
                "acquisition_url": f"materialized-v6://gutenberg/{item['ebook_id']}",
                "origin_key": (
                    f"github:{item['transport_repo']}:{item['transport_commit']}:"
                    f"{item['transport_path']}:pg-body"
                ),
            }
        )
        payloads[source_id] = normalized
        evidence.append(
            {
                "source_id": source_id,
                "ebook_id": int(item["ebook_id"]),
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
                "git_blob_sha1": _git_blob_sha1(raw),
                "normalized_utf8_bytes": len(normalized),
                "normalized_sha256": _sha256(normalized),
                "normalization_policy": PG_NORMALIZER,
            }
        )
    _require(sum(len(value) for value in payloads.values()) == 1672110, "Gutenberg capacity drift")
    return rows, payloads, evidence


def _materialize_v5_graph(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]], int]:
    v5._validate_config(v5_config)
    merged, payloads, inherited_evidence = v5._materialize_v4(base_inventory, v4_extension)
    mdn_payload, mdn_evidence = v5._materialize_mdn(v5_config["mdn"])
    cpython_payload, cpython_capacity, cpython_evidence = v5._materialize_cpython(
        v5_config["cpython"]
    )
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
        "authority_ref": (
            f"NEXT100-037@{cpython['head_sha']} workflow "
            f"{cpython['dedicated_workflow_run']} accepted-only"
        ),
        "declared_capacity_bytes": cpython_capacity,
        "expected_raw_bytes": len(cpython_payload),
        "expected_raw_sha256": _sha256(cpython_payload),
        "acquisition_url": "materialized-v6://cpython-data228-accepted-chunks",
        "origin_key": (
            f"github:python/cpython:{cpython['upstream_commit']}:"
            f"{cpython['upstream_path']}:accepted-only"
        ),
    }
    inventory = copy.deepcopy(merged)
    inventory["sources"] = [*copy.deepcopy(merged["sources"]), mdn_row, cpython_row]
    inventory["final_refresh_required"] = False
    payloads = dict(payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads[cpython["source_id"]] = cpython_payload
    report = v3.audit_payloads(inventory, payloads)
    v3.verify_report(report)
    _require(report["source_count"] == 23, "re-materialized V5 source count drift")
    return (
        inventory,
        payloads,
        [*inherited_evidence, mdn_evidence, cpython_evidence],
        cpython_capacity,
    )


def _family_counts(report: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(modality): int(summary["declared_source_family_count"])
        for modality, summary in report["terminal_candidates"]["by_modality"].items()
    }


def audit_live(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_config(config)
    inventory, payloads, inherited_evidence, cpython_capacity = _materialize_v5_graph(
        base_inventory, v4_extension, v5_config
    )
    _require(cpython_capacity == 15540, "terminal CPython accepted capacity drift")
    _require(len(inventory["sources"]) == 23, "V5 inherited object count drift")

    numpy_rows, numpy_payloads, numpy_evidence = _materialize_numpy(config["numpy"])
    pg_rows, pg_payloads, pg_evidence = _materialize_gutenberg(config["gutenberg"])

    final_inventory = copy.deepcopy(inventory)
    final_inventory["sources"] = [
        *copy.deepcopy(inventory["sources"]),
        *numpy_rows,
        *pg_rows,
    ]
    final_inventory["final_refresh_required"] = False
    final_inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T19:55:00Z"
    final_inventory["terminal_refresh_rule"] = (
        "V6 re-materializes V5, then adds exact-green NEXT100-049 NumPy and "
        "NEXT100-107 Gutenberg terminal source payloads. No sibling source is credited "
        "without exact scoped success authority."
    )
    payloads = {**payloads, **numpy_payloads, **pg_payloads}

    dedup = v3.audit_payloads(final_inventory, payloads)
    v3.verify_report(dedup)
    expected = config["expected_vector"]
    _require(dedup["source_count"] == expected["source_object_count"], "V6 source-count drift")
    counts = _family_counts(dedup)
    _require(counts == expected["source_family_counts"], f"V6 family-vector drift: {counts}")
    scope = dedup["terminal_candidates"]
    _require(
        scope["declared_capacity_bytes_before"]
        == expected["source_capacity_bytes_before_global_dedup"],
        "V6 total capacity arithmetic drift",
    )
    for modality, capacity in expected[
        "source_capacity_by_modality_before_global_dedup"
    ].items():
        _require(
            scope["by_modality"][modality]["declared_capacity_bytes_before"] == capacity,
            f"V6 {modality} capacity arithmetic drift",
        )

    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "base_v5_head_sha": config["base_v5"]["head_sha"],
        "authority_vector": {
            "numpy": {
                "head_sha": config["numpy"]["head_sha"],
                "workflow_run": config["numpy"]["dedicated_workflow_run"],
                "authority_identity_sha256": config["numpy"]["authority_identity_sha256"],
            },
            "gutenberg": {
                "head_sha": config["gutenberg"]["head_sha"],
                "workflow_run": config["gutenberg"]["dedicated_workflow_run"],
                "authority_identity_sha256": config["gutenberg"]["authority_identity_sha256"],
            },
        },
        "source_vector": {
            "source_object_count": dedup["source_count"],
            "source_family_counts": counts,
            "source_capacity_bytes_before_global_dedup": scope[
                "declared_capacity_bytes_before"
            ],
            "source_capacity_by_modality_before_global_dedup": {
                modality: scope["by_modality"][modality]["declared_capacity_bytes_before"]
                for modality in ("uk", "en", "code")
            },
            "conservative_unique_capacity_bytes_after_global_dedup": scope[
                "conservative_unique_capacity_bytes_after"
            ],
            "research_corpus_v1_acquisition_planning_target_bytes": expected[
                "research_corpus_v1_acquisition_planning_target_bytes"
            ],
            "planning_gap_before_global_dedup": expected["planning_gap_before_global_dedup"],
        },
        "materialization_evidence": sorted(
            [*inherited_evidence, *numpy_evidence, *pg_evidence],
            key=lambda item: item["source_id"],
        ),
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
        "remaining_blockers": [
            "SOURCE_CAPACITY_BELOW_20M_RESEARCH_TARGET",
            "FREEZE_EXACT_RECORD_LEVEL_CANDIDATE_INVENTORY",
            "RESERVED_EVALUATION_DECONTAMINATION",
            "FINAL_RECORD_GRANULARITY_QUALITY_PRIVACY_REVALIDATION",
            "BALANCE_DIVERSITY_AND_FAMILY_CAPS",
            "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACKING",
            "TWO_CLEAN_BYTE_IDENTICAL_BUILDS",
            "POSTPACK_UNIQUE_LOSS_LEDGER",
            "TOKENIZER_FLOP_CALIBRATION",
            "CHECKPOINT_D05_TERMINAL_INTEGRITY",
            "BOUNDED_20M_TRAINING_REQUALIFICATION",
            "MATERIAL_COMPUTE_AUTHORIZATION",
        ],
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA, "unsupported V6 report schema")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected == _sha256(_canonical_bytes(core)), "V6 report self-hash mismatch")
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
    _require(vector.get("source_object_count") == 31, "V6 report source count drift")
    _require(
        vector.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "V6 report family vector drift",
    )
    _require(
        vector.get("source_capacity_bytes_before_global_dedup") == 2045180,
        "V6 report capacity drift",
    )
    _require(
        vector.get("source_capacity_by_modality_before_global_dedup")
        == {"uk": 100856, "en": 1838293, "code": 106031},
        "V6 report modality vector drift",
    )
    for key, value in report["claim_boundary"].items():
        _require(value is False, f"V6 report claim boundary weakened: {key}")
    v3.verify_report(report["dedup_v3"])


def run_from_files(
    base_inventory_path: str | Path,
    v4_extension_path: str | Path,
    v5_config_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    return audit_live(
        _load_json(base_inventory_path),
        _load_json(v4_extension_path),
        _load_json(v5_config_path),
        _load_json(config_path),
    )
