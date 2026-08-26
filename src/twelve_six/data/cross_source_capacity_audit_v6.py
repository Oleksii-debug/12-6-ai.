"""NEXT100-109 global cross-source dedup V6.

V6 is a stacked successor to NEXT100-065C/V5. It re-materializes the V5
graph, then adds the two terminal authorities that the newer NEXT100-063 V3
convergence contract says must be reconciled before Research Corpus V1 can be
frozen: the bounded NumPy subset and the three-record Gutenberg seal.
"""
from __future__ import annotations

import ast
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

SCHEMA = "12-6.next100-109-cross-source-dedup-report.v6"
CONFIG_SCHEMA = "12-6.next100-109-cross-source-dedup.v6"
WORKER_ID = "AUTODEV-NEXT100-109-GLOBAL-DEDUP-V6"
EXPECTED_PARENT_V5_HEAD = "7fc6e3ec43ee7fb4361cd5d9b4e795bc3fd7c4b5"
EXPECTED_PARENT_V5_CONFIG_BLOB = "92f230e3480755742aa32125e6bf28f7f852f364"
EXPECTED_CPYTHON_CAPACITY = 15540
EXPECTED_NUMPY_CAPACITY = 36898
EXPECTED_GUTENBERG_CAPACITY = 1672110
EXPECTED_TOTAL_CAPACITY = 2045180

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
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrossSourceV6Error(f"{path}: JSON root must be an object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "unsupported V6 config schema")
    _require(config.get("worker_id") == WORKER_ID, "V6 worker id drift")
    _require(config.get("local_free_only") is True, "V6 LOCAL_FREE boundary weakened")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        _require(config.get(key) is False, f"V6 execution boundary weakened: {key}")

    parent = config["parent_v5"]
    _require(parent.get("pr") == 632, "V6 parent PR drift")
    _require(parent.get("head_sha") == EXPECTED_PARENT_V5_HEAD, "V6 parent V5 head drift")
    _require(
        parent.get("config_blob_sha1") == EXPECTED_PARENT_V5_CONFIG_BLOB,
        "V6 parent V5 config blob drift",
    )
    _require(
        parent.get("expected_cpython_accepted_capacity_bytes") == EXPECTED_CPYTHON_CAPACITY,
        "V6 CPython terminal capacity drift",
    )

    numpy = config["numpy"]
    _require(numpy.get("pr") == 468, "NumPy authority PR drift")
    _require(
        numpy.get("head_sha") == "bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8",
        "NumPy authority head drift",
    )
    _require(numpy.get("dedicated_workflow_run") == 32998548535, "NumPy run drift")
    _require(numpy.get("dedicated_workflow_conclusion") == "success", "NumPy not terminal green")
    _require(
        numpy.get("authority_identity_sha256")
        == "e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70",
        "NumPy authority identity drift",
    )
    _require(numpy.get("repository_family") == "github:numpy/numpy", "NumPy family drift")
    _require(numpy.get("normalization_policy") == "STRICT_UTF8_IDENTITY_PRESERVE_V1", "NumPy policy drift")
    _require(numpy.get("training") == "ALLOWED", "NumPy training-rights drift")
    _require(numpy.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "NumPy evaluation boundary drift")
    _require(numpy.get("expected_capacity_bytes") == EXPECTED_NUMPY_CAPACITY, "NumPy capacity drift")
    _require(len(numpy.get("selected_files", ())) == 5, "NumPy selected-file count drift")

    gutenberg = config["gutenberg"]
    _require(gutenberg.get("seal_pr") == 627, "Gutenberg seal PR drift")
    _require(
        gutenberg.get("seal_head_sha") == "c50b3f9cf871792c03886bdc1ccdc144812be88f",
        "Gutenberg seal head drift",
    )
    _require(
        gutenberg.get("authority_identity_sha256")
        == "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b",
        "Gutenberg authority identity drift",
    )
    _require(gutenberg.get("parent_pr") == 470, "Gutenberg parent PR drift")
    _require(
        gutenberg.get("parent_head_sha") == "3f4ad26e1e8f3406a1274418cf5f485814ce3032",
        "Gutenberg parent head drift",
    )
    _require(gutenberg.get("dedicated_workflow_run") == 32998859164, "Gutenberg run drift")
    _require(
        gutenberg.get("dedicated_workflow_conclusion") == "success",
        "Gutenberg parent execution not terminal green",
    )
    _require(
        gutenberg.get("family") == "en.project-gutenberg.public-domain-books",
        "Gutenberg family drift",
    )
    _require(
        gutenberg.get("normalization_policy") == "NEXT100_033_PG_BODY_NFC_LF_V1",
        "Gutenberg normalization drift",
    )
    _require(
        gutenberg.get("training") == "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
        "Gutenberg training-rights drift",
    )
    _require(gutenberg.get("evaluation") == "NOT_AUTHORIZED", "Gutenberg evaluation boundary drift")
    _require(
        gutenberg.get("expected_capacity_bytes") == EXPECTED_GUTENBERG_CAPACITY,
        "Gutenberg capacity drift",
    )
    _require(len(gutenberg.get("records", ())) == 3, "Gutenberg record-count drift")

    expected = config["expected_vector"]
    _require(expected.get("source_object_count") == 31, "V6 expected object-count drift")
    _require(
        expected.get("source_family_counts") == {"uk": 4, "en": 5, "code": 5},
        "V6 expected family-vector drift",
    )
    _require(
        expected.get("source_capacity_bytes_before_global_dedup") == EXPECTED_TOTAL_CAPACITY,
        "V6 expected total-capacity drift",
    )
    _require(
        expected.get("source_capacity_by_modality_before_global_dedup")
        == {"uk": 100856, "en": 1838293, "code": 106031},
        "V6 expected modality-capacity drift",
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
        _require(boundary.get(key) is False, f"V6 claim boundary weakened: {key}")


def _normalize_gutenberg_body(raw: bytes, encoding: str) -> bytes:
    """Reproduce NEXT100_033_PG_BODY_NFC_LF_V1 exactly."""
    try:
        text = raw.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise CrossSourceV6Error(f"Gutenberg decode failure under {encoding}: {exc}") from exc

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    starts = [index for index, line in enumerate(lines) if _START_RE.match(line.strip())]
    ends = [index for index, line in enumerate(lines) if _END_RE.match(line.strip())]
    _require(len(starts) == 1, f"expected one Gutenberg START marker, got {len(starts)}")
    _require(len(ends) == 1, f"expected one Gutenberg END marker, got {len(ends)}")
    _require(ends[0] > starts[0], "Gutenberg END marker does not follow START marker")

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


def _materialize_numpy(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    capacity = 0
    seen_blobs: set[str] = set()

    commit = str(spec["upstream_commit"])
    family = str(spec["repository_family"])
    for item in spec["selected_files"]:
        path = str(item["path"])
        source_id = f"code.numpy:{path}"
        url = f"https://raw.githubusercontent.com/numpy/numpy/{commit}/{path}"
        raw = v1.fetch_exact_source(url)
        _require(len(raw) == int(item["raw_bytes"]), f"NumPy raw byte-count drift: {path}")
        blob = _git_blob_sha1(raw)
        _require(blob == item["git_blob_sha1"], f"NumPy Git blob drift: {path}")
        _require(blob not in seen_blobs, f"duplicate NumPy blob: {path}")
        seen_blobs.add(blob)
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CrossSourceV6Error(f"NumPy UTF-8 drift: {path}") from exc
        _require("\x00" not in text, f"NumPy NUL byte drift: {path}")
        try:
            ast.parse(text, filename=path)
        except SyntaxError as exc:
            raise CrossSourceV6Error(f"NumPy AST parse drift: {path}: {exc}") from exc

        payloads[source_id] = raw
        capacity += len(raw)
        rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "stable_origin_id": "github:numpy/numpy",
                "stable_object_id": f"git-sha1:{blob}",
                "modality": "code",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"NEXT100-049@{spec['head_sha']} workflow {spec['dedicated_workflow_run']}",
                "declared_capacity_bytes": len(raw),
                "expected_raw_bytes": len(raw),
                "expected_raw_sha256": _sha256(raw),
                "acquisition_url": f"materialized-v6://numpy/{path}",
                "origin_key": f"github:numpy/numpy:{commit}:{path}",
            }
        )
        evidence.append(
            {
                "source_id": source_id,
                "path": path,
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
                "git_blob_sha1": blob,
                "normalization_policy": "STRICT_UTF8_IDENTITY_PRESERVE_V1",
            }
        )

    _require(capacity == int(spec["expected_capacity_bytes"]), "NumPy total capacity drift")
    return rows, payloads, evidence


def _materialize_gutenberg(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    capacity = 0
    family = str(spec["family"])

    for item in spec["records"]:
        source_id = str(item["source_id"])
        repo = str(item["transport_repo"])
        commit = str(item["transport_commit"])
        path = str(item["transport_path"])
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        raw = v1.fetch_exact_source(url)
        _require(len(raw) == int(item["raw_bytes"]), f"Gutenberg raw byte-count drift: {source_id}")
        _require(_sha256(raw) == item["raw_sha256"], f"Gutenberg raw SHA-256 drift: {source_id}")
        blob = _git_blob_sha1(raw)
        _require(blob == item["git_blob_sha1"], f"Gutenberg Git blob drift: {source_id}")
        normalized = _normalize_gutenberg_body(raw, str(item["encoding"]))
        _require(len(normalized) == int(item["normalized_bytes"]), f"Gutenberg normalized byte-count drift: {source_id}")
        _require(_sha256(normalized) == item["normalized_sha256"], f"Gutenberg normalized SHA-256 drift: {source_id}")

        payloads[source_id] = normalized
        capacity += len(normalized)
        rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "stable_origin_id": "project-gutenberg",
                "stable_object_id": f"git-sha1:{blob}",
                "modality": "en",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"NEXT100-107 seal {spec['seal_head_sha']} -> NEXT100-033 workflow {spec['dedicated_workflow_run']}",
                "declared_capacity_bytes": len(normalized),
                "expected_raw_bytes": len(normalized),
                "expected_raw_sha256": _sha256(normalized),
                "acquisition_url": f"materialized-v6://gutenberg/{source_id}",
                "origin_key": f"github:{repo}:{commit}:{path}",
            }
        )
        evidence.append(
            {
                "source_id": source_id,
                "transport_repo": repo,
                "transport_commit": commit,
                "transport_path": path,
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
                "git_blob_sha1": blob,
                "normalized_bytes": len(normalized),
                "normalized_sha256": _sha256(normalized),
                "normalization_policy": "NEXT100_033_PG_BODY_NFC_LF_V1",
            }
        )

    _require(capacity == int(spec["expected_capacity_bytes"]), "Gutenberg total capacity drift")
    return rows, payloads, evidence


def _materialize_v5_graph(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    *,
    expected_cpython_capacity: int,
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]], int]:
    """Rebuild V5's exact 23-object graph without treating V5 prose as evidence."""
    v5._validate_config(v5_config)
    merged, payloads, inherited_evidence = v5._materialize_v4(base_inventory, v4_extension)
    _require(len(merged["sources"]) == 21, "V5 inherited source count drift")

    mdn_payload, mdn_evidence = v5._materialize_mdn(v5_config["mdn"])
    cpython_payload, cpython_capacity, cpython_evidence = v5._materialize_cpython(v5_config["cpython"])
    _require(cpython_capacity == expected_cpython_capacity, f"CPython accepted-only terminal capacity drift: {cpython_capacity}")

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
        "authority_ref": f"NEXT100-037@{cpython['head_sha']} workflow {cpython['dedicated_workflow_run']} accepted-only; NEXT100-063 V3 terminal capacity 15540",
        "declared_capacity_bytes": cpython_capacity,
        "expected_raw_bytes": len(cpython_payload),
        "expected_raw_sha256": _sha256(cpython_payload),
        "acquisition_url": "materialized-v6://cpython-data228-accepted-chunks",
        "origin_key": f"github:python/cpython:{cpython['upstream_commit']}:{cpython['upstream_path']}:accepted-only",
    }

    inventory = copy.deepcopy(merged)
    inventory["sources"] = [*copy.deepcopy(merged["sources"]), mdn_row, cpython_row]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T19:57:00Z"
    inventory["terminal_refresh_rule"] = "V6 re-materializes V5 and reconciles only exact terminal NumPy and Gutenberg authorities identified by NEXT100-063 V3."
    payloads = dict(payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads[cpython["source_id"]] = cpython_payload

    v5_dedup = v3.audit_payloads(inventory, payloads)
    v3.verify_report(v5_dedup)
    _require(v5_dedup["source_count"] == 23, "reconstructed V5 source-count drift")
    _require(v5._family_counts(v5_dedup) == {"uk": 4, "en": 4, "code": 4}, "reconstructed V5 family-vector drift")
    expected_v5_total = 320632 + cpython_capacity
    _require(v5_dedup["terminal_candidates"]["declared_capacity_bytes_before"] == expected_v5_total, "reconstructed V5 capacity drift")

    evidence = [*inherited_evidence, mdn_evidence, cpython_evidence]
    return inventory, payloads, evidence, cpython_capacity


def audit_live(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    v6_config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_config(v6_config)
    inventory, payloads, evidence, cpython_capacity = _materialize_v5_graph(
        base_inventory,
        v4_extension,
        v5_config,
        expected_cpython_capacity=int(v6_config["parent_v5"]["expected_cpython_accepted_capacity_bytes"]),
    )

    numpy_rows, numpy_payloads, numpy_evidence = _materialize_numpy(v6_config["numpy"])
    gutenberg_rows, gutenberg_payloads, gutenberg_evidence = _materialize_gutenberg(v6_config["gutenberg"])

    final_inventory = copy.deepcopy(inventory)
    final_inventory["sources"] = [*copy.deepcopy(inventory["sources"]), *numpy_rows, *gutenberg_rows]
    payloads = {**payloads, **numpy_payloads, **gutenberg_payloads}

    dedup = v3.audit_payloads(final_inventory, payloads)
    v3.verify_report(dedup)
    expected = v6_config["expected_vector"]
    _require(dedup["source_count"] == expected["source_object_count"], "V6 source-count drift")
    counts = v5._family_counts(dedup)
    _require(counts == expected["source_family_counts"], f"V6 family-vector drift: {counts}")

    scope = dedup["terminal_candidates"]
    _require(scope["declared_capacity_bytes_before"] == expected["source_capacity_bytes_before_global_dedup"], "V6 pre-dedup total capacity drift")
    for modality, capacity in expected["source_capacity_by_modality_before_global_dedup"].items():
        _require(scope["by_modality"][modality]["declared_capacity_bytes_before"] == capacity, f"V6 {modality} pre-dedup capacity drift")

    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "parent_v5_head_sha": EXPECTED_PARENT_V5_HEAD,
        "terminal_authorities": {
            "cpython_accepted_only_capacity_bytes": cpython_capacity,
            "numpy_authority_identity_sha256": v6_config["numpy"]["authority_identity_sha256"],
            "numpy_capacity_bytes": EXPECTED_NUMPY_CAPACITY,
            "gutenberg_authority_identity_sha256": v6_config["gutenberg"]["authority_identity_sha256"],
            "gutenberg_capacity_bytes": EXPECTED_GUTENBERG_CAPACITY,
        },
        "source_vector": {
            "source_object_count": dedup["source_count"],
            "source_family_counts": counts,
            "source_capacity_bytes_before_global_dedup": scope["declared_capacity_bytes_before"],
            "source_capacity_by_modality_before_global_dedup": {modality: scope["by_modality"][modality]["declared_capacity_bytes_before"] for modality in ("uk", "en", "code")},
            "conservative_unique_capacity_bytes_after_global_dedup": scope["conservative_unique_capacity_bytes_after"],
        },
        "materialization_evidence": sorted([*evidence, *numpy_evidence, *gutenberg_evidence], key=lambda item: item["source_id"]),
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(v6_config["claim_boundary"]),
        "remaining_blockers": [
            "SOURCE_CAPACITY_STILL_BELOW_20M_RESEARCH_TARGET",
            "EXACT_RECORD_LEVEL_PRE_DECONTAMINATION_IDENTITY",
            "EVALUATION_SELECTION_RESERVATIONS_AND_DECONTAMINATION",
            "POST_COMPOSITION_QUALITY_PRIVACY_AND_BALANCE",
            "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACKING",
            "TWO_CLEAN_BYTE_IDENTICAL_BUILDS",
            "POSTPACK_UNIQUE_LOSS_LEDGER",
            "TOKENIZER_CANDIDATE_REQUALIFICATION",
            "CHECKPOINT_D05_TERMINAL_INTEGRITY",
            "MATERIAL_COMPUTE_AUTHORIZATION"
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
    _require(vector.get("source_object_count") == 31, "V6 report source-count drift")
    _require(vector.get("source_family_counts") == {"code": 5, "en": 5, "uk": 4}, "V6 report family-vector drift")
    _require(vector.get("source_capacity_bytes_before_global_dedup") == EXPECTED_TOTAL_CAPACITY, "V6 report pre-dedup total drift")
    _require(vector.get("source_capacity_by_modality_before_global_dedup") == {"uk": 100856, "en": 1838293, "code": 106031}, "V6 report modality capacities drift")
    authorities = report["terminal_authorities"]
    _require(authorities.get("cpython_accepted_only_capacity_bytes") == EXPECTED_CPYTHON_CAPACITY, "V6 report CPython capacity drift")
    _require(authorities.get("numpy_capacity_bytes") == EXPECTED_NUMPY_CAPACITY, "V6 report NumPy capacity drift")
    _require(authorities.get("gutenberg_capacity_bytes") == EXPECTED_GUTENBERG_CAPACITY, "V6 report Gutenberg capacity drift")

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
    v6_config = _load_json(v6_config_path)
    _validate_config(v6_config)
    v5_raw = Path(v5_config_path).read_bytes()
    _require(_git_blob_sha1(v5_raw) == v6_config["parent_v5"]["config_blob_sha1"], "parent V5 config file no longer matches bound Git blob")
    return (
        _load_json(base_inventory_path),
        _load_json(v4_extension_path),
        json.loads(v5_raw),
        v6_config,
    )
