"""NEXT100-065D global cross-source dedup V6.

V6 extends the executable NEXT100-065C/V5 graph with two already-qualified
source authorities: the bounded NumPy implementation subset and the terminal
three-record Project Gutenberg family. It re-materializes exact upstream bytes,
reproduces each authority's comparison payload, and reruns the incumbent
lineage-aware V3 global dedup engine over the resulting 31-object graph.

This module is LOCAL_FREE data engineering only. Source-capacity accounting is
not a corpus, tokenizer, causal-loss, or training authorization.
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
INVENTORY_SCHEMA = "12-6.next100-065d-cross-source-dedup.v6"
WORKER_ID = "NEXT100-065D-CROSSSOURCE-DEDUP-V6"
NUMPY_POLICY = "STRICT_UTF8_IDENTITY_PRESERVE_V1"
GUTENBERG_POLICY = "NEXT100_033_PG_BODY_NFC_LF_V1"

NUMPY_HEAD = "bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8"
NUMPY_RUN = 32998548535
NUMPY_COMMIT = "4f94a9ac128175d05992ce9946e5b066603c0d9d"
NUMPY_FAMILY = "github:numpy/numpy"
NUMPY_CAPACITY = 36898

GUTENBERG_HEAD = "c50b3f9cf871792c03886bdc1ccdc144812be88f"
GUTENBERG_PARENT_HEAD = "3f4ad26e1e8f3406a1274418cf5f485814ce3032"
GUTENBERG_RUN = 32998859164
GUTENBERG_AUTHORITY = "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b"
GUTENBERG_FAMILY = "en.project-gutenberg.public-domain-books"
GUTENBERG_CAPACITY = 1672110

_START_RE = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")
_END_RE = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")


class CrossSourceV6Error(RuntimeError):
    """Fail-closed V6 materialization, authority, or accounting error."""


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


def _normalize_pg_body(raw: bytes, encoding: str) -> bytes:
    """Reproduce NEXT100_033_PG_BODY_NFC_LF_V1 exactly."""
    try:
        text = raw.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise CrossSourceV6Error(
            f"Gutenberg decode failure under preregistered encoding {encoding}"
        ) from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _START_RE.match(line.strip())]
    ends = [i for i, line in enumerate(lines) if _END_RE.match(line.strip())]
    _require(len(starts) == 1, f"expected one Gutenberg START marker, got {len(starts)}")
    _require(len(ends) == 1, f"expected one Gutenberg END marker, got {len(ends)}")
    _require(ends[0] > starts[0], "Gutenberg END marker does not follow START")

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

    base = config.get("base_v5")
    _require(isinstance(base, Mapping), "base_v5 missing")
    _require(
        base.get("config_path") == "configs/data/next100_065c_cross_source_dedup_v5.json",
        "V5 config binding drift",
    )
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

    numpy = config.get("numpy")
    _require(isinstance(numpy, Mapping), "NumPy authority missing")
    _require(numpy.get("worker") == "NEXT100-049-CODE-NUMPY", "NumPy worker drift")
    _require(numpy.get("pr") == 468, "NumPy PR drift")
    _require(numpy.get("head_sha") == NUMPY_HEAD, "NumPy head drift")
    _require(numpy.get("dedicated_workflow_run") == NUMPY_RUN, "NumPy run drift")
    _require(numpy.get("dedicated_workflow_conclusion") == "success", "NumPy run not green")
    _require(numpy.get("upstream_commit") == NUMPY_COMMIT, "NumPy upstream drift")
    _require(numpy.get("source_family") == NUMPY_FAMILY, "NumPy family drift")
    _require(numpy.get("stable_origin_id") == NUMPY_FAMILY, "NumPy lineage drift")
    _require(numpy.get("normalization_policy") == NUMPY_POLICY, "NumPy policy drift")
    _require(numpy.get("training") == "ALLOWED", "NumPy training-rights drift")
    _require(
        numpy.get("evaluation") == "NOT_SEPARATELY_ADMITTED",
        "NumPy evaluation boundary drift",
    )
    files = numpy.get("files")
    _require(isinstance(files, list) and len(files) == 5, "NumPy file vector drift")
    _require(
        sum(int(row.get("raw_bytes", -1)) for row in files if isinstance(row, Mapping))
        == NUMPY_CAPACITY,
        "NumPy capacity sum drift",
    )
    _require(numpy.get("exact_capacity_bytes") == NUMPY_CAPACITY, "NumPy capacity drift")
    seen_paths: set[str] = set()
    for row in files:
        _require(isinstance(row, Mapping), "NumPy file entry must be an object")
        path = row.get("path")
        blob = row.get("git_blob_sha1")
        _require(isinstance(path, str) and path.startswith("numpy/_core/") and path.endswith(".py"), "NumPy path drift")
        _require(path not in seen_paths, f"duplicate NumPy path: {path}")
        seen_paths.add(path)
        _require(isinstance(blob, str) and re.fullmatch(r"[0-9a-f]{40}", blob) is not None, "NumPy blob identity invalid")
        _require(isinstance(row.get("raw_bytes"), int) and row["raw_bytes"] > 0, "NumPy raw byte count invalid")

    gutenberg = config.get("gutenberg")
    _require(isinstance(gutenberg, Mapping), "Gutenberg authority missing")
    _require(
        gutenberg.get("worker") == "NEXT100-107-DATA-EN-GUTENBERG-TERMINAL-SEAL",
        "Gutenberg worker drift",
    )
    _require(gutenberg.get("pr") == 627, "Gutenberg PR drift")
    _require(gutenberg.get("head_sha") == GUTENBERG_HEAD, "Gutenberg seal head drift")
    _require(gutenberg.get("parent_pr") == 470, "Gutenberg parent PR drift")
    _require(gutenberg.get("parent_head_sha") == GUTENBERG_PARENT_HEAD, "Gutenberg parent head drift")
    _require(gutenberg.get("dedicated_workflow_run") == GUTENBERG_RUN, "Gutenberg run drift")
    _require(
        gutenberg.get("dedicated_workflow_conclusion") == "success",
        "Gutenberg run not green",
    )
    _require(
        gutenberg.get("authority_identity_sha256") == GUTENBERG_AUTHORITY,
        "Gutenberg terminal authority drift",
    )
    _require(gutenberg.get("source_family") == GUTENBERG_FAMILY, "Gutenberg family drift")
    _require(gutenberg.get("normalization_policy") == GUTENBERG_POLICY, "Gutenberg policy drift")
    _require(
        gutenberg.get("training") == "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
        "Gutenberg training-rights drift",
    )
    _require(gutenberg.get("evaluation") == "NOT_AUTHORIZED", "Gutenberg evaluation boundary drift")
    records = gutenberg.get("records")
    _require(isinstance(records, list) and len(records) == 3, "Gutenberg record vector drift")
    _require(
        [row.get("ebook_id") for row in records if isinstance(row, Mapping)]
        == [37177, 37985, 40652],
        "Gutenberg ebook identity/order drift",
    )
    _require(
        sum(int(row.get("normalized_bytes", -1)) for row in records if isinstance(row, Mapping))
        == GUTENBERG_CAPACITY,
        "Gutenberg normalized capacity sum drift",
    )
    _require(
        gutenberg.get("exact_capacity_bytes") == GUTENBERG_CAPACITY,
        "Gutenberg capacity drift",
    )

    expected = config.get("expected_vector")
    _require(isinstance(expected, Mapping), "V6 expected vector missing")
    _require(expected.get("source_object_count") == 31, "V6 expected source count drift")
    _require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "V6 expected family vector drift",
    )
    _require(
        expected.get("fixed_capacity_without_cpython_accepted_chunks")
        == {"uk": 100856, "en": 1822753, "code": 106031, "total": 2029640},
        "V6 fixed capacity drift",
    )
    _require(
        expected.get("expected_total_if_cpython_accepted_capacity_is_15540") == 2045180,
        "V6 expected total drift",
    )
    _require(
        expected.get("full_cpython_normalized_bytes_must_not_be_credited") == 17901,
        "CPython full-byte prohibition drift",
    )

    boundary = config.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "V6 claim boundary missing")
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


def _v5_graph(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]], int]:
    """Materialize the exact V5 comparison graph while retaining payload access."""
    v5._validate_config(v5_config)
    merged, payloads, evidence = v5._materialize_v4(base_inventory, v4_extension)
    _require(len(merged["sources"]) == 21, "inherited V4 source count drift")

    mdn_payload, mdn_evidence = v5._materialize_mdn(v5_config["mdn"])
    cpython_payload, cpython_capacity, cpython_evidence = v5._materialize_cpython(
        v5_config["cpython"]
    )
    mdn = v5_config["mdn"]
    cp = v5_config["cpython"]
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
    cp_row = {
        "source_id": cp["source_id"],
        "source_family": cp["source_family"],
        "stable_origin_id": cp["stable_origin_id"],
        "stable_object_id": cp["stable_object_id"],
        "modality": cp["modality"],
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": f"NEXT100-037@{cp['head_sha']} workflow {cp['dedicated_workflow_run']} accepted-only",
        "declared_capacity_bytes": cpython_capacity,
        "expected_raw_bytes": len(cpython_payload),
        "expected_raw_sha256": _sha256(cpython_payload),
        "acquisition_url": "materialized-v6://cpython-data228-accepted-chunks",
        "origin_key": f"github:python/cpython:{cp['upstream_commit']}:{cp['upstream_path']}:accepted-only",
    }
    inventory = copy.deepcopy(merged)
    inventory["sources"] = [*copy.deepcopy(merged["sources"]), mdn_row, cp_row]
    payloads = dict(payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads[cp["source_id"]] = cpython_payload
    return inventory, payloads, [*evidence, mdn_evidence, cpython_evidence], cpython_capacity


def _materialize_numpy(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    total = 0
    for item in spec["files"]:
        path = str(item["path"])
        url = f"https://raw.githubusercontent.com/numpy/numpy/{spec['upstream_commit']}/{path}"
        raw = v1.fetch_exact_source(url)
        _require(len(raw) == int(item["raw_bytes"]), f"NumPy byte-count drift: {path}")
        _require(_git_blob_sha1(raw) == item["git_blob_sha1"], f"NumPy Git blob drift: {path}")
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CrossSourceV6Error(f"NumPy strict UTF-8 failure: {path}") from exc
        _require(decoded.encode("utf-8") == raw, f"NumPy identity normalization drift: {path}")
        source_id = f"code.numpy.numpy:{path}"
        sha = _sha256(raw)
        rows.append(
            {
                "source_id": source_id,
                "source_family": spec["source_family"],
                "stable_origin_id": spec["stable_origin_id"],
                "stable_object_id": f"gitblob:{item['git_blob_sha1']}",
                "modality": "code",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"NEXT100-049@{spec['head_sha']} workflow {spec['dedicated_workflow_run']}",
                "declared_capacity_bytes": len(raw),
                "expected_raw_bytes": len(raw),
                "expected_raw_sha256": sha,
                "acquisition_url": f"materialized-v6://numpy/{path}",
                "origin_key": f"github:numpy/numpy:{spec['upstream_commit']}:{path}",
            }
        )
        payloads[source_id] = raw
        evidence.append(
            {
                "source_id": source_id,
                "path": path,
                "raw_bytes": len(raw),
                "raw_sha256": sha,
                "git_blob_sha1": _git_blob_sha1(raw),
                "normalization_policy": NUMPY_POLICY,
            }
        )
        total += len(raw)
    _require(total == NUMPY_CAPACITY, "NumPy materialized capacity drift")
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
        _require(len(raw) == int(item["raw_bytes"]), f"Gutenberg raw byte-count drift: {source_id}")
        _require(_git_blob_sha1(raw) == item["git_blob_sha1"], f"Gutenberg Git blob drift: {source_id}")
        normalized = _normalize_pg_body(raw, str(item["encoding"]))
        _require(
            len(normalized) == int(item["normalized_bytes"]),
            f"Gutenberg normalized byte-count drift: {source_id}",
        )
        _require(
            _sha256(normalized) == item["normalized_sha256"],
            f"Gutenberg normalized SHA-256 drift: {source_id}",
        )
        rows.append(
            {
                "source_id": source_id,
                "source_family": spec["source_family"],
                "stable_origin_id": spec["source_family"],
                "stable_object_id": f"pg:{item['ebook_id']}:{item['normalized_sha256']}",
                "modality": "en",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"NEXT100-107@{spec['head_sha']} parent workflow {spec['dedicated_workflow_run']}",
                "declared_capacity_bytes": len(normalized),
                "expected_raw_bytes": len(normalized),
                "expected_raw_sha256": _sha256(normalized),
                "acquisition_url": f"materialized-v6://gutenberg/{item['ebook_id']}",
                "origin_key": f"project-gutenberg:{item['ebook_id']}:{item['transport_commit']}",
            }
        )
        payloads[source_id] = normalized
        evidence.append(
            {
                "source_id": source_id,
                "ebook_id": item["ebook_id"],
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
                "git_blob_sha1": _git_blob_sha1(raw),
                "comparison_bytes": len(normalized),
                "comparison_sha256": _sha256(normalized),
                "normalization_policy": GUTENBERG_POLICY,
            }
        )
        total += len(normalized)
    _require(total == GUTENBERG_CAPACITY, "Gutenberg materialized capacity drift")
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
    _validate_config(v6_config)
    inventory, payloads, inherited_evidence, cp_capacity = _v5_graph(
        base_inventory, v4_extension, v5_config
    )
    _require(len(inventory["sources"]) == 23, "V5 materialized source count drift")

    numpy_rows, numpy_payloads, numpy_evidence = _materialize_numpy(v6_config["numpy"])
    pg_rows, pg_payloads, pg_evidence = _materialize_gutenberg(v6_config["gutenberg"])

    final_inventory = copy.deepcopy(inventory)
    final_inventory["sources"] = [
        *copy.deepcopy(inventory["sources"]),
        *numpy_rows,
        *pg_rows,
    ]
    final_inventory["final_refresh_required"] = False
    final_inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T20:00:00Z"
    final_inventory["terminal_refresh_rule"] = (
        "V6 re-executes the V5 comparison graph and adds exact-green NEXT100-049 NumPy "
        "plus NEXT100-107 Gutenberg normalized bodies; no sibling source receives implicit credit."
    )
    payloads = {**payloads, **numpy_payloads, **pg_payloads}

    dedup = v3.audit_payloads(final_inventory, payloads)
    v3.verify_report(dedup)
    _require(dedup["source_count"] == 31, "V6 source count drift")
    counts = _family_counts(dedup)
    _require(
        counts == v6_config["expected_vector"]["source_family_counts"],
        f"V6 family vector drift: {counts}",
    )
    scope = dedup["terminal_candidates"]
    fixed = v6_config["expected_vector"]["fixed_capacity_without_cpython_accepted_chunks"]
    expected_total = int(fixed["total"]) + cp_capacity
    expected_en = int(fixed["en"]) + cp_capacity
    _require(
        scope["declared_capacity_bytes_before"] == expected_total,
        "V6 total pre-dedup capacity arithmetic drift",
    )
    _require(
        scope["by_modality"]["en"]["declared_capacity_bytes_before"] == expected_en,
        "V6 EN capacity arithmetic drift",
    )
    _require(
        scope["by_modality"]["code"]["declared_capacity_bytes_before"] == fixed["code"],
        "V6 code capacity arithmetic drift",
    )
    _require(
        scope["by_modality"]["uk"]["declared_capacity_bytes_before"] == fixed["uk"],
        "V6 UK capacity arithmetic drift",
    )
    _require(cp_capacity < 17901, "full CPython source bytes were incorrectly credited")

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
            "cpython_accepted_capacity_bytes": cp_capacity,
            "source_capacity_bytes_before_global_dedup": expected_total,
            "source_capacity_by_modality_before_global_dedup": {
                "uk": int(fixed["uk"]),
                "en": expected_en,
                "code": int(fixed["code"]),
            },
            "conservative_unique_capacity_bytes_after_global_dedup": scope[
                "conservative_unique_capacity_bytes_after"
            ],
            "research_corpus_v1_acquisition_target_bytes": v6_config["expected_vector"][
                "research_corpus_v1_acquisition_target_bytes"
            ],
        },
        "materialization_evidence": sorted(
            [*inherited_evidence, *numpy_evidence, *pg_evidence],
            key=lambda item: item["source_id"],
        ),
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(v6_config["claim_boundary"]),
        "remaining_blockers": [
            "SOURCE_CAPACITY_AND_BALANCED_UK_CODE_CAPACITY_STILL_INSUFFICIENT",
            "FREEZE_ONE_TERMINAL_SOURCE_VECTOR",
            "EVALUATION_SELECTION_RESERVATIONS_AND_DECONTAMINATION",
            "POST_COMPOSITION_RECORD_GRANULARITY_QUALITY_PRIVACY",
            "FAMILY_CAP_AND_45_35_20_BALANCE_SELECTION_WITHOUT_REPLAY",
            "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACKING",
            "TWO_CLEAN_BYTE_IDENTICAL_BUILDS",
            "POSTPACK_UNIQUE_NONIGNORED_CAUSAL_LOSS_LEDGER",
            "TOKENIZER_FIT_AUTHORIZATION",
            "CHECKPOINT_D05_TERMINAL_INTEGRITY",
            "MATERIAL_COMPUTE_AUTHORIZATION",
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
    _require(vector.get("source_object_count") == 31, "V6 report source count drift")
    _require(
        vector.get("source_family_counts") == {"code": 5, "en": 5, "uk": 4},
        "V6 report family vector drift",
    )
    cp_capacity = vector.get("cpython_accepted_capacity_bytes")
    _require(
        isinstance(cp_capacity, int) and 0 < cp_capacity < 17901,
        "V6 CPython accepted capacity invalid",
    )
    _require(
        vector.get("source_capacity_bytes_before_global_dedup") == 2029640 + cp_capacity,
        "V6 total arithmetic drift",
    )
    _require(
        vector.get("source_capacity_by_modality_before_global_dedup")
        == {"uk": 100856, "en": 1822753 + cp_capacity, "code": 106031},
        "V6 modality arithmetic drift",
    )
    _require(
        0
        <= vector.get("conservative_unique_capacity_bytes_after_global_dedup", -1)
        <= vector["source_capacity_bytes_before_global_dedup"],
        "V6 post-dedup conservative capacity invalid",
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
        _require(boundary.get(key) is False, f"V6 truth boundary weakened: {key}")
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
