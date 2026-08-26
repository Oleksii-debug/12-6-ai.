"""NEXT100-065D convergence audit V6.

V6 keeps NEXT100-065C/V5 as an executable base and composes exact-green
NumPy and attrs code families plus the terminal-sealed Project Gutenberg
family. It performs one global dedup pass over every materialized payload.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit_v3 as v3
from twelve_six.data import cross_source_capacity_audit_v5 as v5

SCHEMA = "12-6.next100-065d-cross-source-dedup-report.v6"
INVENTORY_SCHEMA = "12-6.next100-065d-cross-source-dedup.v6"
WORKER_ID = "NEXT100-065D-CROSSSOURCE-DEDUP-V6"
GUTENBERG_POLICY = "NEXT100_033_PG_BODY_NFC_LF_V1"
IDENTITY_POLICY = "STRICT_UTF8_IDENTITY_PRESERVE_V1"

START_RE = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")
END_RE = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")


class CrossSourceV6Error(RuntimeError):
    """Fail-closed V6 convergence error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossSourceV6Error(message)


def _normalize_gutenberg_body(raw: bytes, encoding: str) -> bytes:
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
    _require(len(starts) == 1, f"Gutenberg START marker count drift: {len(starts)}")
    _require(len(ends) == 1, f"Gutenberg END marker count drift: {len(ends)}")
    _require(ends[0] > starts[0], "Gutenberg END marker ordering drift")
    body_lines = lines[starts[0] + 1 : ends[0]]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    body = "\n".join(body_lines)
    if body.startswith("\ufeff"):
        body = body[1:]
    return (unicodedata.normalize("NFC", body) + "\n").encode("utf-8")


def _validate_code_family(
    spec: Mapping[str, Any],
    *,
    label: str,
    head_sha: str,
    run_id: int,
    family: str,
    file_count: int,
    capacity: int,
) -> None:
    _require(spec.get("head_sha") == head_sha, f"{label} head drift")
    _require(spec.get("dedicated_workflow_run") == run_id, f"{label} run drift")
    _require(spec.get("dedicated_workflow_conclusion") == "success", f"{label} nonterminal")
    _require(spec.get("source_family") == family, f"{label} family drift")
    _require(spec.get("normalization_policy") == IDENTITY_POLICY, f"{label} normalization drift")
    _require(spec.get("training") == "ALLOWED", f"{label} training boundary drift")
    _require(
        spec.get("evaluation") == "NOT_SEPARATELY_ADMITTED",
        f"{label} evaluation boundary drift",
    )
    _require(len(spec.get("files", [])) == file_count, f"{label} file-count drift")
    _require(
        sum(int(item["raw_bytes"]) for item in spec["files"]) == capacity,
        f"{label} capacity drift",
    )
    _require(spec.get("exact_capacity_bytes") == capacity, f"{label} exact capacity drift")


def _validate_config(config: Mapping[str, Any], v5_config: Mapping[str, Any]) -> None:
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
    v5._validate_config(v5_config)

    base = config["base_v5"]
    _require(
        base.get("base_v4_head_sha") == "5738bb8bac8fda058d5ae9c1361c4a0c3756f360",
        "V5/V4 ancestry drift",
    )
    _require(base.get("source_object_count") == 23, "V5 source count drift")
    _require(
        base.get("source_family_counts") == {"uk": 4, "en": 4, "code": 4},
        "V5 family vector drift",
    )
    _require(
        base.get("fixed_capacity_without_cpython_accepted_chunks")
        == {"uk": 100856, "en": 150643, "code": 69133, "total": 320632},
        "V5 fixed-capacity vector drift",
    )

    _validate_code_family(
        config["numpy"],
        label="NumPy",
        head_sha="bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8",
        run_id=32998548535,
        family="github:numpy/numpy",
        file_count=5,
        capacity=36898,
    )
    _validate_code_family(
        config["attrs"],
        label="attrs",
        head_sha="cda0232d5574ef91eae0d7e0b7fa5efddcbe218b",
        run_id=33006080831,
        family="github:python-attrs/attrs",
        file_count=4,
        capacity=170435,
    )
    attrs = config["attrs"]
    _require(
        attrs.get("authority_identity_sha256")
        == "151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4",
        "attrs authority identity drift",
    )
    _require(attrs.get("artifact_id") == 9621650719, "attrs artifact id drift")
    _require(
        attrs.get("artifact_digest")
        == "sha256:a8176b50a2254fcb50a6f80ca82b63459ba8e9cfddba904b16e5ac79f9c55ff2",
        "attrs artifact digest drift",
    )

    pg = config["gutenberg"]
    _require(
        pg.get("head_sha") == "c50b3f9cf871792c03886bdc1ccdc144812be88f",
        "Gutenberg seal head drift",
    )
    _require(
        pg.get("authority_identity_sha256")
        == "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b",
        "Gutenberg authority identity drift",
    )
    _require(pg.get("dedicated_workflow_run") == 32998859164, "Gutenberg parent run drift")
    _require(pg.get("dedicated_workflow_conclusion") == "success", "Gutenberg parent nonterminal")
    _require(
        pg.get("source_family") == "en.project-gutenberg.public-domain-books",
        "Gutenberg family drift",
    )
    _require(pg.get("normalization_policy") == GUTENBERG_POLICY, "Gutenberg normalization drift")
    _require(
        pg.get("training") == "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
        "Gutenberg training boundary drift",
    )
    _require(pg.get("evaluation") == "NOT_AUTHORIZED", "Gutenberg evaluation boundary drift")
    _require(len(pg.get("records", [])) == 3, "Gutenberg record-count drift")
    _require(
        sum(int(item["normalized_bytes"]) for item in pg["records"]) == 1672110,
        "Gutenberg capacity drift",
    )
    _require(pg.get("exact_capacity_bytes") == 1672110, "Gutenberg exact capacity drift")

    expected = config["expected_vector"]
    _require(expected.get("source_object_count") == 35, "V6 expected source count drift")
    _require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 6},
        "V6 expected family vector drift",
    )
    _require(
        expected.get("fixed_capacity_without_cpython_accepted_chunks")
        == {"uk": 100856, "en": 1822753, "code": 276466, "total": 2200075},
        "V6 fixed-capacity vector drift",
    )
    _require(
        expected.get("expected_total_if_cpython_accepted_capacity_is_15540") == 2215615,
        "V6 planning vector drift",
    )
    _require(
        expected.get("planning_gap_if_no_successor_global_dedup_collapse") == 17784385,
        "V6 planning gap drift",
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


def _materialize_code_family(
    spec: Mapping[str, Any],
    *,
    repo: str,
    source_prefix: str,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    commit = str(spec["upstream_commit"])
    family = str(spec["source_family"])
    total = 0
    for item in spec["files"]:
        path = str(item["path"])
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        raw = v5.v1.fetch_exact_source(url)
        _require(len(raw) == int(item["raw_bytes"]), f"{label} byte-count drift: {path}")
        _require(
            v5._git_blob_sha1(raw) == item["git_blob_sha1"],
            f"{label} Git blob drift: {path}",
        )
        try:
            raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CrossSourceV6Error(f"{label} strict UTF-8 drift: {path}") from exc
        source_id = f"{source_prefix}.{path.replace('/', '.').removesuffix('.py')}"
        rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "stable_origin_id": spec["stable_origin_id"],
                "stable_object_id": f"git-sha1:{item['git_blob_sha1']}",
                "modality": "code",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": (
                    f"{spec['worker']}@{spec['head_sha']} workflow "
                    f"{spec['dedicated_workflow_run']}"
                ),
                "declared_capacity_bytes": len(raw),
                "expected_raw_bytes": len(raw),
                "expected_git_blob_sha1": item["git_blob_sha1"],
                "acquisition_url": url,
                "origin_key": f"github:{repo}:{commit}:{path}",
            }
        )
        payloads[source_id] = raw
        evidence.append(
            {
                "source_id": source_id,
                "source_family": family,
                "raw_bytes": len(raw),
                "raw_sha256": v5._sha256(raw),
                "git_blob_sha1": v5._git_blob_sha1(raw),
                "normalization_policy": spec["normalization_policy"],
            }
        )
        total += len(raw)
    _require(total == int(spec["exact_capacity_bytes"]), f"{label} total capacity drift")
    return rows, payloads, evidence


def _materialize_numpy(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    return _materialize_code_family(
        spec,
        repo="numpy/numpy",
        source_prefix="code.numpy",
        label="NumPy",
    )


def _materialize_attrs(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    return _materialize_code_family(
        spec,
        repo="python-attrs/attrs",
        source_prefix="code.attrs",
        label="attrs",
    )


def _materialize_gutenberg(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    total = 0
    for item in spec["records"]:
        repo = str(item["transport_repo"])
        commit = str(item["transport_commit"])
        path = str(item["transport_path"])
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        raw = v5.v1.fetch_exact_source(url)
        _require(
            len(raw) == int(item["raw_bytes"]),
            f"Gutenberg raw byte-count drift: {item['source_id']}",
        )
        _require(
            v5._git_blob_sha1(raw) == item["git_blob_sha1"],
            f"Gutenberg Git blob drift: {item['source_id']}",
        )
        normalized = _normalize_gutenberg_body(raw, str(item["encoding"]))
        _require(
            len(normalized) == int(item["normalized_bytes"]),
            f"Gutenberg normalized byte-count drift: {item['source_id']}",
        )
        _require(
            v5._sha256(normalized) == item["normalized_sha256"],
            f"Gutenberg normalized SHA-256 drift: {item['source_id']}",
        )
        source_id = str(item["source_id"])
        rows.append(
            {
                "source_id": source_id,
                "source_family": spec["source_family"],
                "stable_origin_id": "project-gutenberg",
                "stable_object_id": f"sha256:{item['normalized_sha256']}",
                "modality": "en",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": (
                    f"NEXT100-107@{spec['head_sha']} parent workflow "
                    f"{spec['dedicated_workflow_run']}"
                ),
                "declared_capacity_bytes": len(normalized),
                "expected_raw_bytes": len(normalized),
                "expected_raw_sha256": item["normalized_sha256"],
                "acquisition_url": f"materialized-v6://gutenberg/{item['ebook_id']}",
                "origin_key": f"project-gutenberg:{item['ebook_id']}:{commit}:{path}",
            }
        )
        payloads[source_id] = normalized
        evidence.append(
            {
                "source_id": source_id,
                "ebook_id": int(item["ebook_id"]),
                "raw_bytes": len(raw),
                "raw_sha256": v5._sha256(raw),
                "git_blob_sha1": v5._git_blob_sha1(raw),
                "normalized_bytes": len(normalized),
                "normalized_sha256": v5._sha256(normalized),
                "normalization_policy": GUTENBERG_POLICY,
            }
        )
        total += len(normalized)
    _require(total == int(spec["exact_capacity_bytes"]), "Gutenberg total capacity drift")
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
    _validate_config(v6_config, v5_config)

    merged, payloads, inherited_evidence = v5._materialize_v4(base_inventory, v4_extension)
    mdn_payload, mdn_evidence = v5._materialize_mdn(v5_config["mdn"])
    cpython_payload, cpython_capacity, cpython_evidence = v5._materialize_cpython(
        v5_config["cpython"]
    )

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
            "acquisition_url": "materialized-v6://mdn-prose-only",
            "origin_key": f"github:mdn/content:{mdn['upstream_commit']}:{mdn['upstream_path']}",
        },
        {
            "source_id": cp["source_id"],
            "source_family": cp["source_family"],
            "stable_origin_id": cp["stable_origin_id"],
            "stable_object_id": cp["stable_object_id"],
            "modality": cp["modality"],
            "evidence_status": "DEDICATED_TERMINAL",
            "authority_ref": (
                f"NEXT100-037@{cp['head_sha']} workflow {cp['dedicated_workflow_run']} "
                "accepted-only"
            ),
            "declared_capacity_bytes": cpython_capacity,
            "expected_raw_bytes": len(cpython_payload),
            "expected_raw_sha256": v5._sha256(cpython_payload),
            "acquisition_url": "materialized-v6://cpython-data228-accepted-chunks",
            "origin_key": (
                f"github:python/cpython:{cp['upstream_commit']}:"
                f"{cp['upstream_path']}:accepted-only"
            ),
        },
    ]

    numpy_rows, numpy_payloads, numpy_evidence = _materialize_numpy(v6_config["numpy"])
    pg_rows, pg_payloads, pg_evidence = _materialize_gutenberg(v6_config["gutenberg"])
    attrs_rows, attrs_payloads, attrs_evidence = _materialize_attrs(v6_config["attrs"])

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
        "V6 composes exact-green V4 materialization, MDN, accepted-only CPython, NumPy, "
        "terminal-sealed Gutenberg, and terminal attrs; failed/nonterminal sibling "
        "authorities receive zero credit."
    )

    payloads = dict(payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads[cp["source_id"]] = cpython_payload
    payloads.update(numpy_payloads)
    payloads.update(pg_payloads)
    payloads.update(attrs_payloads)

    dedup = v3.audit_payloads(final_inventory, payloads)
    v3.verify_report(dedup)
    _require(dedup["source_count"] == 35, "V6 source-count drift")
    counts = _family_counts(dedup)
    _require(
        counts == v6_config["expected_vector"]["source_family_counts"],
        f"V6 family-vector drift: {counts}",
    )

    fixed = v6_config["expected_vector"]["fixed_capacity_without_cpython_accepted_chunks"]
    expected_total = int(fixed["total"]) + cpython_capacity
    expected_en = int(fixed["en"]) + cpython_capacity
    scope = dedup["terminal_candidates"]
    _require(
        scope["declared_capacity_bytes_before"] == expected_total,
        "V6 total capacity arithmetic drift",
    )
    _require(
        scope["by_modality"]["en"]["declared_capacity_bytes_before"] == expected_en,
        "V6 EN capacity arithmetic drift",
    )
    _require(
        scope["by_modality"]["code"]["declared_capacity_bytes_before"] == int(fixed["code"]),
        "V6 code capacity arithmetic drift",
    )
    _require(
        cpython_capacity
        != v6_config["expected_vector"]["full_cpython_normalized_bytes_must_not_be_credited"],
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
            "research_corpus_v1_acquisition_target_bytes": v6_config["expected_vector"][
                "research_corpus_v1_acquisition_target_bytes"
            ],
            "pre_dedup_planning_gap_bytes": max(
                0,
                int(
                    v6_config["expected_vector"][
                        "research_corpus_v1_acquisition_target_bytes"
                    ]
                )
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
        "claim_boundary": copy.deepcopy(v6_config["claim_boundary"]),
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
    return {**core, "report_sha256": v5._sha256(v5._canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA, "unsupported V6 report schema")
    expected_hash = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(
        expected_hash == v5._sha256(v5._canonical_bytes(core)),
        "V6 report self-hash mismatch",
    )
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
        "raw_text_emitted",
    ):
        _require(report.get(key) is False, f"V6 execution/text boundary failed: {key}")
    vector = report["source_vector"]
    _require(vector.get("source_object_count") == 35, "V6 report source count drift")
    _require(
        vector.get("source_family_counts") == {"code": 6, "en": 5, "uk": 4},
        "V6 report family vector drift",
    )
    cp_capacity = vector.get("cpython_accepted_capacity_bytes")
    _require(
        isinstance(cp_capacity, int) and 0 < cp_capacity < 17901,
        "V6 CPython accepted capacity invalid",
    )
    _require(
        vector.get("source_capacity_bytes_before_global_dedup") == 2200075 + cp_capacity,
        "V6 report total arithmetic drift",
    )
    _require(
        vector.get("pre_dedup_planning_gap_bytes") == 20000000 - (2200075 + cp_capacity),
        "V6 planning gap arithmetic drift",
    )

    evidence = report["materialization_evidence"]
    numpy_evidence = [
        item for item in evidence if str(item["source_id"]).startswith("code.numpy.")
    ]
    _require(len(numpy_evidence) == 5, "V6 NumPy evidence count drift")
    attrs_evidence = [
        item for item in evidence if str(item["source_id"]).startswith("code.attrs.")
    ]
    _require(len(attrs_evidence) == 4, "V6 attrs evidence count drift")
    _require(
        sum(int(item["raw_bytes"]) for item in attrs_evidence) == 170435,
        "V6 attrs evidence capacity drift",
    )
    _require(
        {item.get("source_family") for item in attrs_evidence} == {"github:python-attrs/attrs"},
        "V6 attrs evidence family drift",
    )
    pg_evidence = [
        item
        for item in evidence
        if str(item["source_id"]).startswith("en.project-gutenberg.")
    ]
    _require(len(pg_evidence) == 3, "V6 Gutenberg evidence count drift")
    _require(
        sum(int(item["normalized_bytes"]) for item in pg_evidence) == 1672110,
        "V6 Gutenberg evidence capacity drift",
    )
    cp_evidence = next(
        item
        for item in evidence
        if item["source_id"] == "en.python.docs.tutorial-introduction"
    )
    _require(cp_evidence.get("accepted_chunk_count") == 14, "V6 CPython accepted count drift")
    _require(cp_evidence.get("rejected_chunk_count") == 2, "V6 CPython rejected count drift")
    _require(
        cp_evidence.get("rejection_reasons") == {"pii_phone": 2},
        "V6 CPython privacy rejection drift",
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
        _require(
            report["claim_boundary"].get(key) is False,
            f"V6 truth boundary weakened: {key}",
        )
    v3.verify_report(report["dedup_v3"])


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    v5.v1.write_report(report, path)


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
