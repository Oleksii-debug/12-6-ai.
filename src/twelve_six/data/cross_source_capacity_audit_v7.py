"""NEXT100-065E global cross-source dedup V7.

V7 preserves the terminal-capable V6 implementation and adds the exact-green
attrs 26.1.0 source authority as four exact code objects / one source family.
The complete payload graph is globally deduplicated again; registry prose or
nonterminal registry CI is never used as source authority.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit_v6 as v6

SCHEMA = "12-6.next100-065e-cross-source-dedup-report.v7"
INVENTORY_SCHEMA = "12-6.next100-065e-cross-source-dedup.v7"
WORKER_ID = "NEXT100-065E-CROSSSOURCE-DEDUP-V7"
ATTRS_FAMILY = "github:python-attrs/attrs"
ATTRS_HEAD = "cda0232d5574ef91eae0d7e0b7fa5efddcbe218b"
ATTRS_RUN = 33006080831
ATTRS_AUTHORITY = "151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4"
ATTRS_ARTIFACT = 9621650719
ATTRS_DIGEST = "sha256:a8176b50a2254fcb50a6f80ca82b63459ba8e9cfddba904b16e5ac79f9c55ff2"
ATTRS_COMMIT = "7bfc49e9b22d5ba25b6e429524c3d49fee27cb36"


class CrossSourceV7Error(RuntimeError):
    """Fail-closed V7 convergence error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossSourceV7Error(message)


def _validate_config(
    config: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    v6_config: Mapping[str, Any],
) -> None:
    _require(config.get("schema_version") == INVENTORY_SCHEMA, "unsupported V7 config schema")
    _require(config.get("worker_id") == WORKER_ID, "V7 worker id drift")
    _require(config.get("local_free_only") is True, "LOCAL_FREE boundary weakened")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        _require(config.get(key) is False, f"execution boundary weakened: {key}")

    v6._validate_config(v6_config, v5_config)
    base = config["base_v6"]
    _require(base.get("source_object_count") == 31, "V6 source-count binding drift")
    _require(
        base.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "V6 family-vector binding drift",
    )
    _require(
        base.get("fixed_capacity_without_cpython_accepted_chunks")
        == {"uk": 100856, "en": 1822753, "code": 106031, "total": 2029640},
        "V6 capacity binding drift",
    )
    _require(
        base.get("expected_total_if_cpython_accepted_capacity_is_15540") == 2045180,
        "V6 total binding drift",
    )

    attrs = config["attrs"]
    _require(attrs.get("worker") == "NEXT100-053-CODE-ATTRS", "attrs worker drift")
    _require(attrs.get("pr") == 474, "attrs PR drift")
    _require(attrs.get("head_sha") == ATTRS_HEAD, "attrs head drift")
    _require(attrs.get("dedicated_workflow_run") == ATTRS_RUN, "attrs run drift")
    _require(attrs.get("dedicated_workflow_conclusion") == "success", "attrs nonterminal")
    _require(attrs.get("authority_identity_sha256") == ATTRS_AUTHORITY, "attrs authority drift")
    _require(attrs.get("terminal_artifact_id") == ATTRS_ARTIFACT, "attrs artifact drift")
    _require(attrs.get("terminal_artifact_digest") == ATTRS_DIGEST, "attrs digest drift")
    _require(attrs.get("source_family") == ATTRS_FAMILY, "attrs family drift")
    _require(attrs.get("upstream_commit") == ATTRS_COMMIT, "attrs commit drift")
    _require(
        attrs.get("normalization_policy") == "STRICT_UTF8_IDENTITY_PRESERVE_V1",
        "attrs normalization drift",
    )
    _require(attrs.get("training") == "ALLOWED", "attrs training boundary drift")
    _require(attrs.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "attrs evaluation boundary drift")
    _require(len(attrs.get("files", [])) == 4, "attrs file-count drift")
    _require(
        sum(int(item["raw_bytes"]) for item in attrs["files"]) == 170435,
        "attrs capacity arithmetic drift",
    )
    _require(attrs.get("exact_capacity_bytes") == 170435, "attrs exact capacity drift")

    reconciliation = config["registry_v5_reconciliation"]
    _require(reconciliation.get("registry_pr") == 538, "registry reconciliation PR drift")
    _require(
        reconciliation.get("config_git_blob_sha1")
        == "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf",
        "registry V5 blob binding drift",
    )
    _require(
        reconciliation.get("registry_workflow_terminal") is False,
        "nonterminal registry must not be promoted",
    )

    expected = config["expected_vector"]
    _require(expected.get("source_object_count") == 35, "V7 expected source count drift")
    _require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 6},
        "V7 expected family vector drift",
    )
    _require(
        expected.get("fixed_capacity_without_cpython_accepted_chunks")
        == {"uk": 100856, "en": 1822753, "code": 276466, "total": 2200075},
        "V7 fixed-capacity vector drift",
    )
    _require(
        expected.get("expected_total_if_cpython_accepted_capacity_is_15540") == 2215615,
        "V7 planning vector drift",
    )
    _require(
        expected.get("planning_gap_if_no_successor_global_dedup_collapse") == 17784385,
        "V7 planning gap drift",
    )
    _require(
        expected.get("full_cpython_normalized_bytes_must_not_be_credited") == 17901,
        "CPython full-source prohibition drift",
    )
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
        _require(config["claim_boundary"].get(key) is False, f"claim boundary weakened: {key}")


def _materialize_attrs(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    total = 0
    commit = str(spec["upstream_commit"])
    for item in spec["files"]:
        path = str(item["path"])
        url = f"https://raw.githubusercontent.com/python-attrs/attrs/{commit}/{path}"
        raw = v6.v5.v1.fetch_exact_source(url)
        _require(len(raw) == int(item["raw_bytes"]), f"attrs byte-count drift: {path}")
        _require(v6.v5._git_blob_sha1(raw) == item["git_blob_sha1"], f"attrs Git blob drift: {path}")
        try:
            raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CrossSourceV7Error(f"attrs strict UTF-8 drift: {path}") from exc
        source_id = f"code.attrs.{path.replace('/', '.').removesuffix('.py')}"
        rows.append(
            {
                "source_id": source_id,
                "source_family": ATTRS_FAMILY,
                "stable_origin_id": spec["stable_origin_id"],
                "stable_object_id": f"git-sha1:{item['git_blob_sha1']}",
                "modality": "code",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"NEXT100-053@{ATTRS_HEAD} workflow {ATTRS_RUN}",
                "declared_capacity_bytes": len(raw),
                "expected_raw_bytes": len(raw),
                "expected_git_blob_sha1": item["git_blob_sha1"],
                "acquisition_url": url,
                "origin_key": f"github:python-attrs/attrs:{commit}:{path}",
            }
        )
        payloads[source_id] = raw
        evidence.append(
            {
                "source_id": source_id,
                "raw_bytes": len(raw),
                "raw_sha256": v6.v5._sha256(raw),
                "git_blob_sha1": v6.v5._git_blob_sha1(raw),
                "normalization_policy": spec["normalization_policy"],
                "authority_identity_sha256": ATTRS_AUTHORITY,
            }
        )
        total += len(raw)
    _require(total == int(spec["exact_capacity_bytes"]), "attrs total capacity drift")
    return rows, payloads, evidence


def audit_live(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    v6_config: Mapping[str, Any],
    v7_config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_config(v7_config, v5_config, v6_config)

    merged, payloads, inherited_evidence = v6.v5._materialize_v4(base_inventory, v4_extension)
    mdn_payload, mdn_evidence = v6.v5._materialize_mdn(v5_config["mdn"])
    cpython_payload, cpython_capacity, cpython_evidence = v6.v5._materialize_cpython(v5_config["cpython"])
    mdn = v5_config["mdn"]
    cp = v5_config["cpython"]
    v5_rows = [
        {
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
            "acquisition_url": "materialized-v7://mdn-prose-only",
            "origin_key": f"github:mdn/content:{mdn['upstream_commit']}:{mdn['upstream_path']}",
        },
        {
            "source_id": cp["source_id"],
            "source_family": cp["source_family"],
            "stable_origin_id": cp["stable_origin_id"],
            "stable_object_id": cp["stable_object_id"],
            "modality": cp["modality"],
            "evidence_status": "DEDICATED_TERMINAL",
            "authority_ref": f"NEXT100-037@{cp['head_sha']} workflow {cp['dedicated_workflow_run']} accepted-only",
            "declared_capacity_bytes": cpython_capacity,
            "expected_raw_bytes": len(cpython_payload),
            "expected_raw_sha256": v6.v5._sha256(cpython_payload),
            "acquisition_url": "materialized-v7://cpython-data228-accepted-chunks",
            "origin_key": f"github:python/cpython:{cp['upstream_commit']}:{cp['upstream_path']}:accepted-only",
        },
    ]

    numpy_rows, numpy_payloads, numpy_evidence = v6._materialize_numpy(v6_config["numpy"])
    pg_rows, pg_payloads, pg_evidence = v6._materialize_gutenberg(v6_config["gutenberg"])
    attrs_rows, attrs_payloads, attrs_evidence = _materialize_attrs(v7_config["attrs"])

    final_inventory = copy.deepcopy(merged)
    final_inventory["sources"] = [
        *copy.deepcopy(merged["sources"]),
        *v5_rows,
        *numpy_rows,
        *pg_rows,
        *attrs_rows,
    ]
    final_inventory["final_refresh_required"] = False
    final_inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T20:16:31Z"
    final_inventory["terminal_refresh_rule"] = (
        "V7 composes exact source authorities for the complete V6 graph plus terminal attrs; "
        "nonterminal registry coordination is diagnostic only and grants no source credit."
    )

    payloads = dict(payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads[cp["source_id"]] = cpython_payload
    payloads.update(numpy_payloads)
    payloads.update(pg_payloads)
    payloads.update(attrs_payloads)

    dedup = v6.v3.audit_payloads(final_inventory, payloads)
    v6.v3.verify_report(dedup)
    _require(dedup["source_count"] == 35, "V7 source-count drift")
    counts = v6._family_counts(dedup)
    _require(counts == v7_config["expected_vector"]["source_family_counts"], f"V7 family-vector drift: {counts}")

    fixed = v7_config["expected_vector"]["fixed_capacity_without_cpython_accepted_chunks"]
    expected_total = int(fixed["total"]) + cpython_capacity
    expected_en = int(fixed["en"]) + cpython_capacity
    scope = dedup["terminal_candidates"]
    _require(scope["declared_capacity_bytes_before"] == expected_total, "V7 total capacity arithmetic drift")
    _require(
        scope["by_modality"]["en"]["declared_capacity_bytes_before"] == expected_en,
        "V7 EN capacity arithmetic drift",
    )
    _require(
        scope["by_modality"]["code"]["declared_capacity_bytes_before"] == int(fixed["code"]),
        "V7 code capacity arithmetic drift",
    )
    _require(
        cpython_capacity != v7_config["expected_vector"]["full_cpython_normalized_bytes_must_not_be_credited"],
        "full CPython normalized bytes were incorrectly credited",
    )

    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "source_vector": {
            "source_object_count": dedup["source_count"],
            "source_family_counts": counts,
            "fixed_capacity_without_cpython_accepted_chunks": copy.deepcopy(fixed),
            "cpython_accepted_capacity_bytes": cpython_capacity,
            "source_capacity_bytes_before_global_dedup": expected_total,
            "source_capacity_by_modality_before_global_dedup": {
                "uk": fixed["uk"],
                "en": expected_en,
                "code": fixed["code"],
            },
            "conservative_unique_capacity_bytes_after_global_dedup": scope[
                "conservative_unique_capacity_bytes_after"
            ],
            "research_corpus_v1_acquisition_target_bytes": v7_config["expected_vector"][
                "research_corpus_v1_acquisition_target_bytes"
            ],
            "pre_dedup_planning_gap_bytes": max(
                0,
                int(v7_config["expected_vector"]["research_corpus_v1_acquisition_target_bytes"])
                - expected_total,
            ),
        },
        "materialization_evidence": sorted(
            [
                *inherited_evidence,
                mdn_evidence,
                cpython_evidence,
                *numpy_evidence,
                *pg_evidence,
                *attrs_evidence,
            ],
            key=lambda item: item["source_id"],
        ),
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(v7_config["claim_boundary"]),
        "remaining_blockers": [
            "SOURCE_CAPACITY_FAR_BELOW_20M_RESEARCH_TARGET",
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
    return {**core, "report_sha256": v6.v5._sha256(v6.v5._canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA, "unsupported V7 report schema")
    expected_hash = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected_hash == v6.v5._sha256(v6.v5._canonical_bytes(core)), "V7 report self-hash mismatch")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
        "raw_text_emitted",
    ):
        _require(report.get(key) is False, f"V7 execution/text boundary failed: {key}")
    vector = report["source_vector"]
    _require(vector.get("source_object_count") == 35, "V7 report source count drift")
    _require(
        vector.get("source_family_counts") == {"code": 6, "en": 5, "uk": 4},
        "V7 report family vector drift",
    )
    cp_capacity = vector.get("cpython_accepted_capacity_bytes")
    _require(isinstance(cp_capacity, int) and 0 < cp_capacity < 17901, "V7 CPython accepted capacity invalid")
    _require(
        vector.get("source_capacity_bytes_before_global_dedup") == 2200075 + cp_capacity,
        "V7 report total arithmetic drift",
    )
    _require(
        vector.get("pre_dedup_planning_gap_bytes") == 20000000 - (2200075 + cp_capacity),
        "V7 planning gap arithmetic drift",
    )
    attrs_evidence = [
        item for item in report["materialization_evidence"] if str(item["source_id"]).startswith("code.attrs.")
    ]
    _require(len(attrs_evidence) == 4, "V7 attrs evidence count drift")
    _require(sum(int(item["raw_bytes"]) for item in attrs_evidence) == 170435, "V7 attrs evidence capacity drift")
    _require(
        {item["authority_identity_sha256"] for item in attrs_evidence} == {ATTRS_AUTHORITY},
        "V7 attrs authority evidence drift",
    )
    cp_evidence = next(
        item for item in report["materialization_evidence"] if item["source_id"] == "en.python.docs.tutorial-introduction"
    )
    _require(cp_evidence.get("accepted_chunk_count") == 14, "V7 CPython accepted count drift")
    _require(cp_evidence.get("rejected_chunk_count") == 2, "V7 CPython rejected count drift")
    _require(cp_evidence.get("rejection_reasons") == {"pii_phone": 2}, "V7 CPython privacy rejection drift")
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
        _require(report["claim_boundary"].get(key) is False, f"V7 truth boundary weakened: {key}")
    v6.v3.verify_report(report["dedup_v3"])


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    v6.v5.v1.write_report(report, path)


def load_inputs(
    base_inventory_path: str | Path,
    v4_extension_path: str | Path,
    v5_config_path: str | Path,
    v6_config_path: str | Path,
    v7_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        v6.v5._load_json(base_inventory_path),
        v6.v5._load_json(v4_extension_path),
        v6.v5._load_json(v5_config_path),
        v6.v5._load_json(v6_config_path),
        v6.v5._load_json(v7_config_path),
    )
