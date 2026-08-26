"""NEXT100-065D global cross-source dedup V6.

V6 is a stacked successor to NEXT100-065C/V5. It reconstructs the exact V5
comparison graph and adds the three terminal-sealed Project Gutenberg bodies as
one English source family before rerunning the incumbent lineage-aware dedup
engine. It authorizes no corpus release, tokenizer fit, model training, or paid
compute.
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

SCHEMA = "12-6.next100-065d-cross-source-dedup-report.v6"
CONFIG_SCHEMA = "12-6.next100-065d-cross-source-dedup.v6"
WORKER_ID = "NEXT100-065D-GUTENBERG-GLOBAL-DEDUP-V6"
GUTENBERG_FAMILY = "en.project-gutenberg.public-domain-books"
GUTENBERG_AUTHORITY_ID = (
    "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b"
)
GUTENBERG_PARENT_HEAD = "3f4ad26e1e8f3406a1274418cf5f485814ce3032"
GUTENBERG_TERMINAL_RUN = 32998859164
GUTENBERG_TOTAL_BYTES = 1_672_110
CPYTHON_ACCEPTED_BYTES = 15_540

_START_RE = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")
_END_RE = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")


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


def normalize_gutenberg_body(raw: bytes, encoding: str) -> bytes:
    """Reproduce NEXT100_033_PG_BODY_NFC_LF_V1 exactly."""
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise CrossSourceV6Error(
            f"Gutenberg decode failure under preregistered encoding {encoding}: {exc}"
        ) from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _START_RE.match(line.strip())]
    ends = [i for i, line in enumerate(lines) if _END_RE.match(line.strip())]
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
    return (unicodedata.normalize("NFC", body) + "\n").encode("utf-8")


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "unsupported V6 config schema")
    _require(config.get("worker_id") == WORKER_ID, "V6 worker id drift")
    _require(
        config.get("base_v5_head_sha")
        == "8b67e6cfe0c0ae025d1e5d0d3647b70273e16946",
        "V5 head drift",
    )
    _require(config.get("local_free_only") is True, "LOCAL_FREE boundary weakened")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        _require(config.get(key) is False, f"execution boundary weakened: {key}")

    authority = config["gutenberg_authority"]
    _require(
        authority.get("authority_identity_sha256") == GUTENBERG_AUTHORITY_ID,
        "Gutenberg authority identity drift",
    )
    _require(
        authority.get("parent_head_sha") == GUTENBERG_PARENT_HEAD,
        "Gutenberg parent head drift",
    )
    _require(
        authority.get("workflow_run_id") == GUTENBERG_TERMINAL_RUN,
        "Gutenberg workflow run drift",
    )
    _require(
        authority.get("workflow_conclusion") == "success",
        "Gutenberg authority not green",
    )
    _require(
        authority.get("decision") == "TERMINAL_SOURCE_ADMIT",
        "Gutenberg terminal decision drift",
    )
    _require(authority.get("family_id") == GUTENBERG_FAMILY, "Gutenberg family drift")
    _require(
        authority.get("independent_family_credit") == 1,
        "Gutenberg family-credit inflation",
    )
    _require(
        authority.get("model_training")
        == "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
        "Gutenberg training-purpose drift",
    )
    _require(
        authority.get("evaluation") == "NOT_AUTHORIZED",
        "Gutenberg evaluation boundary drift",
    )
    _require(
        authority.get("worldwide_public_domain_claim") is False,
        "worldwide public-domain claim introduced",
    )
    records = authority.get("records")
    _require(isinstance(records, list) and len(records) == 3, "Gutenberg record count drift")
    _require(
        sum(int(row["normalized_utf8_bytes"]) for row in records)
        == GUTENBERG_TOTAL_BYTES,
        "Gutenberg normalized-byte total drift",
    )
    _require(
        len({row["source_id"] for row in records}) == 3,
        "duplicate Gutenberg source id",
    )

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
        _require(boundary.get(key) is False, f"claim boundary weakened: {key}")


def _reconstruct_v5(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]], int]:
    v5._validate_config(v5_config)
    merged, payloads, inherited_evidence = v5._materialize_v4(
        base_inventory, v4_extension
    )
    mdn_payload, mdn_evidence = v5._materialize_mdn(v5_config["mdn"])
    cpython_payload, cpython_capacity, cpython_evidence = v5._materialize_cpython(
        v5_config["cpython"]
    )
    _require(
        cpython_capacity == CPYTHON_ACCEPTED_BYTES,
        f"accepted-only CPython capacity drift: {cpython_capacity}",
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
        "authority_ref": (
            f"NEXT100-038@{mdn['head_sha']} workflow {mdn['dedicated_workflow_run']}"
        ),
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
    final_inventory = copy.deepcopy(merged)
    final_inventory["sources"] = [
        *copy.deepcopy(merged["sources"]),
        mdn_row,
        cpython_row,
    ]
    final_inventory["final_refresh_required"] = False
    final_inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T20:00:00Z"
    payloads = dict(payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads[cpython["source_id"]] = cpython_payload
    evidence = [*inherited_evidence, mdn_evidence, cpython_evidence]
    return final_inventory, payloads, evidence, cpython_capacity


def _materialize_gutenberg(
    authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    for source in authority["records"]:
        repo = str(source["transport_repo"])
        commit = str(source["transport_commit"])
        path = str(source["transport_path"])
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        raw = v1.fetch_exact_source(url)
        _require(
            len(raw) == int(source["raw_bytes"]),
            f"{source['source_id']} raw-byte drift",
        )
        _require(
            _sha256(raw) == source["raw_sha256"],
            f"{source['source_id']} raw SHA-256 drift",
        )
        _require(
            _git_blob_sha1(raw) == source["transport_git_blob_sha1"],
            f"{source['source_id']} Git blob drift",
        )
        normalized = normalize_gutenberg_body(raw, str(source["encoding"]))
        _require(
            len(normalized) == int(source["normalized_utf8_bytes"]),
            f"{source['source_id']} normalized-byte drift",
        )
        _require(
            _sha256(normalized) == source["normalized_sha256"],
            f"{source['source_id']} normalized SHA-256 drift",
        )
        source_id = str(source["source_id"])
        rows.append(
            {
                "source_id": source_id,
                "source_family": GUTENBERG_FAMILY,
                "stable_origin_id": f"gitenberg:{repo}",
                "stable_object_id": f"{repo}@{commit}:{path}",
                "modality": "en",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": (
                    f"NEXT100-107 authority {GUTENBERG_AUTHORITY_ID} "
                    f"parent {GUTENBERG_PARENT_HEAD} workflow {GUTENBERG_TERMINAL_RUN}"
                ),
                "declared_capacity_bytes": int(source["normalized_utf8_bytes"]),
                "expected_raw_bytes": len(normalized),
                "expected_raw_sha256": _sha256(normalized),
                "acquisition_url": f"materialized-v6://{source_id}",
                "origin_key": f"gitenberg:{repo}:{commit}:{path}",
            }
        )
        payloads[source_id] = normalized
        evidence.append(
            {
                "source_id": source_id,
                "family_id": GUTENBERG_FAMILY,
                "transport_repo": repo,
                "transport_commit": commit,
                "transport_path": path,
                "transport_git_blob_sha1": _git_blob_sha1(raw),
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
                "normalized_utf8_bytes": len(normalized),
                "normalized_sha256": _sha256(normalized),
                "normalizer_id": "NEXT100_033_PG_BODY_NFC_LF_V1",
            }
        )
    _require(
        sum(len(value) for value in payloads.values()) == GUTENBERG_TOTAL_BYTES,
        "materialized Gutenberg byte-total drift",
    )
    return rows, payloads, evidence


def _family_counts(report: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(modality): int(summary["declared_source_family_count"])
        for modality, summary in report["terminal_candidates"]["by_modality"].items()
    }


def audit_live(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    v6_config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_config(v6_config)
    inventory, payloads, evidence, cpython_capacity = _reconstruct_v5(
        base_inventory, v4_extension, v5_config
    )
    _require(len(inventory["sources"]) == 23, "reconstructed V5 source count drift")
    rows, gutenberg_payloads, gutenberg_evidence = _materialize_gutenberg(
        v6_config["gutenberg_authority"]
    )
    inventory = copy.deepcopy(inventory)
    inventory["sources"] = [*inventory["sources"], *rows]
    inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T20:00:00Z"
    inventory["terminal_refresh_rule"] = (
        "V6 reconstructs exact V5 and adds only the exact three-record "
        "NEXT100-107 Gutenberg terminal seal as one English family."
    )
    payloads = {**payloads, **gutenberg_payloads}
    dedup = v3.audit_payloads(inventory, payloads)
    v3.verify_report(dedup)

    _require(dedup["source_count"] == 26, "V6 source-count drift")
    counts = _family_counts(dedup)
    _require(
        counts == {"uk": 4, "en": 5, "code": 4},
        f"V6 family-vector drift: {counts}",
    )
    pre_dedup_total = 320_632 + cpython_capacity + GUTENBERG_TOTAL_BYTES
    pre_dedup_en = 150_643 + cpython_capacity + GUTENBERG_TOTAL_BYTES
    scope = dedup["terminal_candidates"]
    _require(
        scope["declared_capacity_bytes_before"] == pre_dedup_total,
        "V6 pre-dedup total drift",
    )
    _require(
        scope["by_modality"]["en"]["declared_capacity_bytes_before"] == pre_dedup_en,
        "V6 pre-dedup EN total drift",
    )
    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "base_v5_head_sha": v6_config["base_v5_head_sha"],
        "source_vector": {
            "source_object_count": dedup["source_count"],
            "source_family_counts": counts,
            "cpython_accepted_capacity_bytes": cpython_capacity,
            "gutenberg_normalized_capacity_bytes": GUTENBERG_TOTAL_BYTES,
            "source_capacity_bytes_before_global_dedup": pre_dedup_total,
            "source_capacity_by_modality_before_global_dedup": {
                "uk": 100_856,
                "en": pre_dedup_en,
                "code": 69_133,
            },
            "conservative_unique_capacity_bytes_after_global_dedup": scope[
                "conservative_unique_capacity_bytes_after"
            ],
        },
        "gutenberg_authority_identity_sha256": GUTENBERG_AUTHORITY_ID,
        "materialization_evidence": sorted(
            [*evidence, *gutenberg_evidence], key=lambda item: item["source_id"]
        ),
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(v6_config["claim_boundary"]),
        "remaining_blockers": [
            "SOURCE_CAPACITY_AND_BALANCE_STILL_REQUIRE_RETEST",
            "FINAL_RECORD_GRANULARITY_QUALITY_PRIVACY_REVALIDATION",
            "EVALUATION_SELECTION_RESERVATIONS_AND_DECONTAMINATION",
            "IMMUTABLE_SPLIT_AND_PACKING",
            "POSTPACK_UNIQUE_LOSS_LEDGER",
            "TOKENIZER_FIT_AUTHORIZATION",
            "CHECKPOINT_D05_TERMINAL_INTEGRITY",
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
    _require(vector.get("source_object_count") == 26, "V6 report source count drift")
    _require(
        vector.get("source_family_counts") == {"code": 4, "en": 5, "uk": 4},
        "V6 report family vector drift",
    )
    _require(
        vector.get("cpython_accepted_capacity_bytes") == CPYTHON_ACCEPTED_BYTES,
        "V6 CPython capacity drift",
    )
    _require(
        vector.get("gutenberg_normalized_capacity_bytes") == GUTENBERG_TOTAL_BYTES,
        "V6 Gutenberg capacity drift",
    )
    _require(
        vector.get("source_capacity_bytes_before_global_dedup")
        == 320_632 + CPYTHON_ACCEPTED_BYTES + GUTENBERG_TOTAL_BYTES,
        "V6 pre-dedup total arithmetic drift",
    )
    _require(
        report.get("gutenberg_authority_identity_sha256") == GUTENBERG_AUTHORITY_ID,
        "V6 report Gutenberg authority drift",
    )
    boundary = report["claim_boundary"]
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
        _require(boundary.get(key) is False, f"V6 report claim boundary failed: {key}")


def load_inputs(
    base_inventory_path: str | Path,
    v4_extension_path: str | Path,
    v5_config_path: str | Path,
    v6_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        v5._load_json(base_inventory_path),
        v5._load_json(v4_extension_path),
        v5._load_json(v5_config_path),
        v5._load_json(v6_config_path),
    )


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    verify_report(report)
    Path(path).write_bytes(_canonical_bytes(report))
