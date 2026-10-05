#!/usr/bin/env python3
"""Execute the exact qualified Lesia 1892 Wikisource candidate through incumbent global dedup.

Execution glue only. The source/materializer and matcher semantics are reused unchanged.
Durable outputs are text-free and grant zero corpus, tokenizer, training, evaluation,
or compute authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import ssl
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
for location in (str(ROOT / "tools"), str(ROOT / "src")):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_franko1901_global_dedup_execution_v1 as incumbent
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data import expanded_global_dedup_v9 as v9_semantics
from twelve_six.data import incumbent_dedup_indexed_execution as indexed
from twelve_six.data import wikisource_pd_contract as contract

SCHEMA = "12-6.d03-wikisource-lesia1892-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-wikisource-lesia1892-global-dedup-survivors.v1"
PARENT_PRODUCT_HEAD = "774dcfdad408cf7f9d924a69edf516547fc98560"
PHYSICAL_PRODUCT_HEAD = "9c1e2c6e4c79b84562ffeee4ffe6ac41964edbe1"
SOURCE_EXECUTION_HEAD = "6ac08f5dd883c6e7c27ee78fcf8f97f2bc39224f"
SOURCE_EXECUTION_RUN_ID = 37269387682
SOURCE_EXECUTION_JOB_A = 111632941271
SOURCE_EXECUTION_JOB_B = 111640481968
SOURCE_EXECUTION_COMPARE_JOB = 111649599632
SOURCE_TEST_QUALIFICATION_HEAD = "51600663b4f922ef7e5384663cf28eca7fdae791"
SOURCE_TEST_QUALIFICATION_RUN_ID = 37364151903
CARRIER_PATH = "tools/run_d03_wikisource_lesia1892_global_dedup_v1.py"

EXPECTED_CANDIDATE_SHA256 = (
    "13025e767ff3f92c0809de807893a5002b691c296b345c12978249c11bc44d12"
)
EXPECTED_REPORT_IDENTITY_SHA256 = (
    "8f3ec6c8ecba70f74c6b024873afe9c1d4574be5415d7dcd54bfd1cdd0aa81ce"
)
EXPECTED_CANDIDATE_RECORDS = 107
EXPECTED_CANDIDATE_BYTES = 169_399
EXPECTED_REJECTED_RECORDS = 1
EXPECTED_BASE_OBJECTS = incumbent.EXPECTED_BASE_OBJECTS
EXPECTED_BASE_BYTES = incumbent.EXPECTED_BASE_BYTES
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_CANDIDATE_RECORDS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_CANDIDATE_BYTES

AUTHORITY_PATHS = (
    "configs/data/d03_wikisource_lesia1892_current_main_v1.json",
    "src/twelve_six/data/wikisource_pd_api.py",
    "src/twelve_six/data/wikisource_pd_contract.py",
    "src/twelve_six/data/wikisource_pd_edition.py",
    "tools/materialize_d03_wikisource_lesia1892.py",
    "src/twelve_six/data/expanded_global_dedup_v9.py",
    "src/twelve_six/data/_expanded_global_dedup_v9_impl.py",
    "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
    "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
    "tools/run_d03_franko1901_global_dedup_execution_v1.py",
    "tools/run_next100_065f_global_dedup_v8.py",
)

_CANDIDATE_KEYS = frozenset(
    {
        "source_id",
        "source_family_id",
        "language",
        "modality",
        "page_number",
        "page_title",
        "page_revision_id",
        "normalized_sha256",
        "normalized_utf8_bytes",
        "training_eligible",
        "evaluation_eligible",
        "text",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WikisourceGlobalDedupError(RuntimeError):
    """Fail-closed physical execution or authority mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WikisourceGlobalDedupError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_line(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WikisourceGlobalDedupError("duplicate JSON member")
        result[key] = value
    return result


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                WikisourceGlobalDedupError(f"{label} contains non-finite JSON number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikisourceGlobalDedupError(f"{label} is not strict UTF-8 JSON") from exc
    _require(type(value) is dict, f"{label} root must be exact object")
    return value


def _read_regular_file(path: Path, *, label: str) -> bytes:
    _require(not path.is_symlink() and path.is_file(), f"{label} must be regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WikisourceGlobalDedupError(f"{label} is unreadable") from exc


def _reconstruct_v8_with_bounded_tls_retry(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Replay exact V7 with the already-qualified bounded TLS-EOF transport retry."""

    original_capture = v8._capture_terminal_v7

    def capture_with_retry(
        historical_root: Path,
        historical_config: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, bytes]]:
        historical_v7 = v8._load_v7(historical_root)
        fetch_module = historical_v7.v6.v5.v1
        original_fetch = fetch_module.fetch_exact_source

        def retry_fetch(url: str) -> bytes:
            for attempt in range(1, 4):
                try:
                    return original_fetch(url)
                except OSError as exc:
                    retryable = (
                        isinstance(exc, URLError)
                        and isinstance(exc.reason, ssl.SSLEOFError)
                    )
                    if retryable and attempt < 3:
                        time.sleep(0.25 * attempt)
                        continue
                    try:
                        host = urlsplit(url).hostname or "unknown"
                    except ValueError:
                        host = "invalid-url"
                    url_sha256 = hashlib.sha256(url.encode("utf-8")).hexdigest()
                    exc.add_note(
                        "historical V7 source fetch failed: "
                        f"host={host}; acquisition_url_sha256={url_sha256}; "
                        f"attempts={attempt}"
                    )
                    raise
            raise AssertionError("unreachable historical fetch attempt state")

        fetch_module.fetch_exact_source = retry_fetch
        try:
            return original_capture(historical_root, historical_config)
        finally:
            fetch_module.fetch_exact_source = original_fetch

    v8._capture_terminal_v7 = capture_with_retry
    try:
        return incumbent._reconstruct_v8_with_historical_namespace(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    finally:
        v8._capture_terminal_v7 = original_capture


def _verify_authority_paths() -> dict[str, str]:
    ancestor = incumbent._git(
        "merge-base", "--is-ancestor", PARENT_PRODUCT_HEAD, "HEAD", check=False
    )
    _require(ancestor.returncode == 0, "Wikisource Product parent is not an ancestor")
    blobs: dict[str, str] = {}
    for path in AUTHORITY_PATHS:
        parent_blob = incumbent._git(
            "rev-parse", f"{PARENT_PRODUCT_HEAD}:{path}"
        ).stdout.strip()
        head_blob = incumbent._git("rev-parse", f"HEAD:{path}").stdout.strip()
        worktree_blob = incumbent._git("hash-object", str(ROOT / path)).stdout.strip()
        _require(
            bool(parent_blob) and head_blob == parent_blob and worktree_blob == head_blob,
            f"authority path drift: {path}",
        )
        blobs[path] = head_blob
    incumbent.verify_repository_authority()
    return blobs


def _verify_carrier() -> str:
    expected = incumbent._git("rev-parse", f"HEAD:{CARRIER_PATH}").stdout.strip()
    observed = incumbent._git("hash-object", str(ROOT / CARRIER_PATH)).stdout.strip()
    _require(len(expected) == 40 and expected == observed, "execution carrier drift")
    return observed


def _verify_materialization_report(
    report: Mapping[str, Any],
    *,
    candidate_raw: bytes,
) -> list[dict[str, Any]]:
    _require(
        report.get("schema_version")
        == "12-6.d03-wikisource-pd-edition-materialization.v1",
        "materialization report schema drift",
    )
    claimed_identity = report.get("report_sha256")
    _require(
        claimed_identity == EXPECTED_REPORT_IDENTITY_SHA256,
        "materialization report identity drift",
    )
    report_core = dict(report)
    report_core.pop("report_sha256", None)
    _require(
        _sha256(_canonical_line(report_core)) == claimed_identity,
        "materialization report self-hash mismatch",
    )

    authority = report.get("source_authority")
    _require(type(authority) is dict, "source authority missing")
    _require(
        authority
        == {
            "incumbent_next100022_authority_sha256": contract.INCUMBENT_AUTHORITY_SHA256,
            "index_revision_id": contract.INDEX_REVISION_ID,
            "source_family_id": contract.SOURCE_FAMILY_ID,
            "family_credit_added": False,
        },
        "source authority drift",
    )

    candidate = report.get("candidate")
    _require(type(candidate) is dict, "materialization candidate summary missing")
    _require(candidate.get("page_count") == EXPECTED_CANDIDATE_RECORDS, "page-count drift")
    _require(
        candidate.get("normalized_utf8_bytes") == EXPECTED_CANDIDATE_BYTES,
        "candidate byte-count drift",
    )
    _require(
        candidate.get("candidate_jsonl_sha256") == EXPECTED_CANDIDATE_SHA256,
        "candidate identity drift",
    )
    _require(_sha256(candidate_raw) == EXPECTED_CANDIDATE_SHA256, "candidate SHA drift")
    inventory = candidate.get("inventory")
    _require(
        type(inventory) is list and len(inventory) == EXPECTED_CANDIDATE_RECORDS,
        "candidate inventory cardinality drift",
    )

    disposition = report.get("disposition")
    _require(type(disposition) is dict, "materialization disposition missing")
    _require(
        disposition.get("accepted_page_count") == EXPECTED_CANDIDATE_RECORDS,
        "accepted-page count drift",
    )
    _require(
        disposition.get("rejected_page_count") == EXPECTED_REJECTED_RECORDS,
        "rejected-page count drift",
    )
    _require(
        disposition.get("observed_page_count")
        == EXPECTED_CANDIDATE_RECORDS + EXPECTED_REJECTED_RECORDS,
        "observed-page count drift",
    )

    truth = report.get("truth_boundary")
    _require(type(truth) is dict, "materialization truth boundary missing")
    expected_zero = {
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates": 0,
        "model_training_executed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    for key, wanted in expected_zero.items():
        _require(truth.get(key) == wanted, f"materialization truth drift: {key}")
    return copy.deepcopy(inventory)


def _project_candidate(
    candidate_raw: bytes,
    report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    inventory = _verify_materialization_report(report, candidate_raw=candidate_raw)
    inventory_by_page: dict[int, dict[str, Any]] = {}
    for item in inventory:
        _require(type(item) is dict, "candidate inventory row invalid")
        page_number = item.get("page_number")
        _require(type(page_number) is int and page_number > 0, "inventory page number invalid")
        _require(page_number not in inventory_by_page, "duplicate inventory page number")
        inventory_by_page[page_number] = item

    lines = candidate_raw.splitlines(keepends=True)
    _require(len(lines) == EXPECTED_CANDIDATE_RECORDS, "candidate row-count drift")
    sources: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    seen_pages: set[int] = set()
    seen_revisions: set[int] = set()
    seen_hashes: set[str] = set()
    projected_identity_rows: list[dict[str, Any]] = []
    total_bytes = 0

    for line_number, raw_line in enumerate(lines, 1):
        _require(raw_line.endswith(b"\n"), f"candidate newline drift at row {line_number}")
        row = _strict_json_object(raw_line, label=f"candidate row {line_number}")
        _require(set(row) == _CANDIDATE_KEYS, f"candidate row {line_number} schema drift")
        source_id = row["source_id"]
        family = row["source_family_id"]
        language = row["language"]
        modality = row["modality"]
        page_number = row["page_number"]
        page_title = row["page_title"]
        revision_id = row["page_revision_id"]
        digest = row["normalized_sha256"]
        byte_count = row["normalized_utf8_bytes"]
        text = row["text"]

        _require(type(source_id) is str and bool(source_id), "candidate source_id invalid")
        _require(family == contract.SOURCE_FAMILY_ID, "candidate source-family drift")
        _require(language == "uk" and modality == "text", "candidate language/modality drift")
        _require(type(page_number) is int and page_number > 0, "candidate page invalid")
        _require(type(revision_id) is int and revision_id > 0, "candidate revision invalid")
        _require(type(page_title) is str and bool(page_title), "candidate title invalid")
        _require(
            contract.validate_page_title(page_title) == page_number,
            "candidate title/page mismatch",
        )
        _require(
            type(digest) is str and _SHA256_RE.fullmatch(digest) is not None,
            "candidate payload digest invalid",
        )
        _require(type(byte_count) is int and byte_count >= 64, "candidate byte count invalid")
        _require(type(text) is str and bool(text), "candidate text invalid")
        _require(row["training_eligible"] is False, "candidate training flag drift")
        _require(row["evaluation_eligible"] is False, "candidate evaluation flag drift")
        _require(page_number not in seen_pages, "duplicate candidate page")
        _require(revision_id not in seen_revisions, "duplicate candidate revision")
        _require(digest not in seen_hashes, "duplicate candidate payload hash")

        normalized = contract.normalize_rendered_text(text)
        _require(normalized == text, "candidate text is not canonical")
        contract.validate_ua_page_text(text)
        payload = text.encode("utf-8")
        _require(len(payload) == byte_count, "candidate payload byte-count mismatch")
        _require(_sha256(payload) == digest, "candidate payload hash mismatch")
        expected_inventory = inventory_by_page.get(page_number)
        _require(type(expected_inventory) is dict, "candidate page missing from report inventory")
        _require(
            expected_inventory
            == {
                "page_number": page_number,
                "page_revision_id": revision_id,
                "normalized_sha256": digest,
                "normalized_utf8_bytes": byte_count,
            },
            "candidate/report inventory mismatch",
        )

        matcher_source_id = f"wikisource-lesia1892:{source_id}:rev:{revision_id}"
        _require(matcher_source_id not in payloads, "matcher source-id collision")
        matcher_row = {
            "source_id": matcher_source_id,
            "source_family": contract.SOURCE_FAMILY_ID,
            "stable_origin_id": f"uk.wikisource.org:{source_id}:oldid:{revision_id}",
            "stable_object_id": f"sha256:{digest}",
            "modality": "uk",
            "evidence_status": "DEDICATED_TERMINAL",
            "authority_ref": (
                f"PR2793:{PARENT_PRODUCT_HEAD}:physical-parent:{PHYSICAL_PRODUCT_HEAD}:"
                f"source-run:{SOURCE_EXECUTION_RUN_ID}:compare-job:{SOURCE_EXECUTION_COMPARE_JOB}"
            ),
            "declared_capacity_bytes": byte_count,
            "expected_raw_bytes": byte_count,
            "expected_raw_sha256": digest,
            "acquisition_url": f"https://uk.wikisource.org/w/index.php?oldid={revision_id}",
            "origin_key": f"wikisource-lesia1892:page:{page_number}:revision:{revision_id}",
        }
        sources.append(matcher_row)
        payloads[matcher_source_id] = payload
        projected_identity_rows.append(copy.deepcopy(matcher_row))
        total_bytes += byte_count
        seen_pages.add(page_number)
        seen_revisions.add(revision_id)
        seen_hashes.add(digest)

    _require(set(inventory_by_page) == seen_pages, "candidate/report page coverage mismatch")
    _require(total_bytes == EXPECTED_CANDIDATE_BYTES, "candidate payload byte total drift")
    receipt_core = {
        "schema_version": "12-6.d03-wikisource-lesia1892-dedup-intake-receipt.v1",
        "source_family": contract.SOURCE_FAMILY_ID,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "materialization_report_identity_sha256": EXPECTED_REPORT_IDENTITY_SHA256,
        "record_count": EXPECTED_CANDIDATE_RECORDS,
        "payload_utf8_bytes": total_bytes,
        "distinct_payload_sha256_count": len(seen_hashes),
        "matcher_projection_identity_sha256": _sha256(_canonical(projected_identity_rows)),
        "source_execution": {
            "product_head_sha": PHYSICAL_PRODUCT_HEAD,
            "execution_head_sha": SOURCE_EXECUTION_HEAD,
            "workflow_run_id": SOURCE_EXECUTION_RUN_ID,
            "job_a_id": SOURCE_EXECUTION_JOB_A,
            "job_b_id": SOURCE_EXECUTION_JOB_B,
            "compare_job_id": SOURCE_EXECUTION_COMPARE_JOB,
            "two_fresh_executions_converged": True,
        },
        "source_test_qualification": {
            "head_sha": SOURCE_TEST_QUALIFICATION_HEAD,
            "workflow_run_id": SOURCE_TEST_QUALIFICATION_RUN_ID,
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": _sha256(_canonical(receipt_core)),
    }
    return sources, payloads, receipt


def _compose_graph(
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    extension_sources: list[dict[str, Any]],
    extension_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rows = base_inventory.get("sources")
    _require(type(rows) is list and bool(rows), "base source rows missing")
    base_ids = {row.get("source_id") for row in rows if type(row) is dict}
    extension_ids = {row.get("source_id") for row in extension_sources}
    _require(
        len(base_ids) == len(rows) and None not in base_ids,
        "base source identities invalid",
    )
    _require(base_ids == set(base_payloads), "base inventory/payload coverage mismatch")
    _require(
        len(extension_ids) == len(extension_sources) and None not in extension_ids,
        "Wikisource source identities invalid",
    )
    _require(
        extension_ids == set(extension_payloads),
        "Wikisource inventory/payload coverage mismatch",
    )
    _require(
        not (base_ids & extension_ids),
        "Wikisource source-id collision with incumbent graph",
    )
    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(extension_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed Nomis-free V8 graph plus exact qualified Lesia 1892 "
        "Wikisource page revisions; pair decisions delegate unchanged to terminal "
        "PR #824/V3 semantics through the independently qualified indexed executor."
    )
    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    return inventory, payloads


def _terminal(
    report: Mapping[str, Any], expected_objects: int, expected_bytes: int
) -> Mapping[str, Any]:
    _require(report.get("source_count") == expected_objects, "dedup source count drift")
    terminal = report.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "terminal dedup summary missing")
    before = terminal.get("declared_capacity_bytes_before")
    after = terminal.get("conservative_unique_capacity_bytes_after")
    discount = terminal.get("duplicate_discount_bytes")
    _require(type(before) is int and before == expected_bytes, "pre-dedup byte drift")
    _require(type(after) is int and 0 < after <= before, "post-dedup byte value invalid")
    _require(
        type(discount) is int and discount >= 0 and before - after == discount,
        "dedup byte arithmetic drift",
    )
    return terminal


def _validate_projection(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
    expected_objects: int,
    expected_bytes: int,
) -> None:
    terminal = _terminal(report, expected_objects, expected_bytes)
    _require(
        projection.get("schema_version") == v9_semantics.SURVIVOR_SCHEMA,
        "survivor projection schema drift",
    )
    core = dict(projection)
    claimed = core.pop("survivor_authority_sha256", None)
    _require(
        type(claimed) is str and claimed == _sha256(_canonical(core)),
        "projection hash drift",
    )
    _require(
        projection.get("matcher_report_sha256") == report.get("report_sha256"),
        "projection/report identity drift",
    )
    ids = projection.get("survivor_source_ids")
    _require(
        type(ids) is list
        and all(type(item) is str and item for item in ids)
        and len(ids) == len(set(ids)),
        "survivor ids invalid",
    )
    _require(
        projection.get("pre_dedup_source_object_count") == expected_objects,
        "projection pre-dedup object count drift",
    )
    _require(
        projection.get("pre_dedup_declared_capacity_bytes") == expected_bytes,
        "projection pre-dedup byte count drift",
    )
    _require(
        projection.get("post_dedup_declared_capacity_bytes")
        == terminal.get("conservative_unique_capacity_bytes_after"),
        "projection post-dedup byte count drift",
    )
    _require(
        projection.get("post_dedup_survivor_source_object_count") == len(ids),
        "projection survivor count drift",
    )


def _wikisource_survivor_authority(
    extension_sources: list[dict[str, Any]],
    combined_report: Mapping[str, Any],
    projection: Mapping[str, Any],
    *,
    marginal_unique_bytes: int,
) -> dict[str, Any]:
    extension_by_id = {row["source_id"]: row for row in extension_sources}
    _require(
        len(extension_by_id) == EXPECTED_CANDIDATE_RECORDS,
        "Wikisource extension identity cardinality drift",
    )
    survivor_ids = projection.get("survivor_source_ids")
    _require(type(survivor_ids) is list, "combined survivor ids missing")
    selected_ids = [source_id for source_id in survivor_ids if source_id in extension_by_id]
    selected_rows: list[dict[str, Any]] = []
    for source_id in selected_ids:
        row = extension_by_id[source_id]
        selected_rows.append(
            {
                "source_id": source_id,
                "source_family": row["source_family"],
                "stable_origin_id": row["stable_origin_id"],
                "stable_object_id": row["stable_object_id"],
                "declared_capacity_bytes": row["declared_capacity_bytes"],
                "expected_raw_sha256": row["expected_raw_sha256"],
                "origin_key": row["origin_key"],
            }
        )
    survivor_bytes = sum(row["declared_capacity_bytes"] for row in selected_rows)
    _require(
        0 <= marginal_unique_bytes <= EXPECTED_CANDIDATE_BYTES,
        "Wikisource marginal capacity invalid",
    )
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "matcher_report_sha256": combined_report.get("report_sha256"),
        "selection_projection_sha256": projection.get("survivor_authority_sha256"),
        "source_family": contract.SOURCE_FAMILY_ID,
        "candidate_source_object_count": EXPECTED_CANDIDATE_RECORDS,
        "candidate_payload_utf8_bytes": EXPECTED_CANDIDATE_BYTES,
        "post_dedup_selected_extension_source_object_count": len(selected_rows),
        "post_dedup_selected_extension_payload_bytes": survivor_bytes,
        "marginal_global_unique_capacity_bytes": marginal_unique_bytes,
        "survivors": selected_rows,
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    candidate_jsonl: Path,
    materialization_report: Path,
    output_base_report: Path,
    output_combined_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    expected_execution_head: str,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    execution_head = incumbent._bind_execution_head(expected_execution_head)
    authority_blobs = _verify_authority_paths()
    carrier_blob = _verify_carrier()
    bulk_workspace = incumbent._prepare_empty_bulk_workspace(bulk_workspace)
    verified_v7_head = incumbent._verify_v7_worktree(v7_root)
    config = v8.load_config(ROOT / "configs/data/next100_065f_global_dedup_v8.json")
    matcher, base_inventory, base_payloads, removal = (
        _reconstruct_v8_with_bounded_tls_retry(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    )
    _require(
        incumbent._verify_v7_worktree(v7_root) == verified_v7_head,
        "V7 worktree drifted during reconstruction",
    )
    _require(len(base_payloads) == EXPECTED_BASE_OBJECTS, "base object-count drift")
    _require(
        incumbent._declared_capacity_bytes(
            base_inventory, base_payloads, label="Wikisource execution base"
        )
        == EXPECTED_BASE_BYTES,
        "base declared-capacity drift",
    )

    candidate_raw = _read_regular_file(candidate_jsonl, label="candidate JSONL")
    report_raw = _read_regular_file(materialization_report, label="materialization report")
    report = _strict_json_object(report_raw, label="materialization report")
    extension_sources, extension_payloads, intake_receipt = _project_candidate(
        candidate_raw, report
    )
    _require(
        len(extension_sources) == EXPECTED_CANDIDATE_RECORDS,
        "Wikisource projected object-count drift",
    )
    _require(
        incumbent._declared_capacity_bytes(
            {"sources": extension_sources},
            extension_payloads,
            label="Wikisource projection",
        )
        == EXPECTED_CANDIDATE_BYTES,
        "Wikisource projected byte-count drift",
    )
    inventory, payloads = _compose_graph(
        base_inventory, base_payloads, extension_sources, extension_payloads
    )
    _require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined object-count drift")
    _require(
        incumbent._declared_capacity_bytes(inventory, payloads, label="combined graph")
        == EXPECTED_COMBINED_BYTES,
        "combined declared-capacity drift",
    )

    indexed.attest_incumbent_runtime(matcher)
    base_report = indexed.audit_payloads_indexed(
        matcher,
        base_inventory,
        base_payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    matcher.verify_report(base_report)
    base_terminal = _terminal(base_report, EXPECTED_BASE_OBJECTS, EXPECTED_BASE_BYTES)
    base_report_sha256 = base_report.get("report_sha256")
    _require(
        type(base_report_sha256) is str and len(base_report_sha256) == 64,
        "base report identity invalid",
    )
    base_report_bytes = incumbent._canonical(dict(base_report))
    del base_report

    indexed.attest_incumbent_runtime(matcher)
    combined_report = indexed.audit_payloads_indexed(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    matcher.verify_report(combined_report)
    combined_terminal = _terminal(
        combined_report, EXPECTED_COMBINED_OBJECTS, EXPECTED_COMBINED_BYTES
    )
    projection = v9_semantics._derive_survivors(combined_report)
    _validate_projection(
        combined_report,
        projection,
        EXPECTED_COMBINED_OBJECTS,
        EXPECTED_COMBINED_BYTES,
    )
    marginal = (
        combined_terminal["conservative_unique_capacity_bytes_after"]
        - base_terminal["conservative_unique_capacity_bytes_after"]
    )
    _require(
        0 <= marginal <= EXPECTED_CANDIDATE_BYTES,
        "Wikisource marginal unique capacity invalid",
    )
    survivors = _wikisource_survivor_authority(
        extension_sources,
        combined_report,
        projection,
        marginal_unique_bytes=marginal,
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "parent_product_head_sha": PARENT_PRODUCT_HEAD,
        "execution_carrier_git_blob_sha1": carrier_blob,
        "authority_path_blobs": authority_blobs,
        "v7_head_sha": verified_v7_head,
        "base": {
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "declared_capacity_bytes": EXPECTED_BASE_BYTES,
            "report_sha256": base_report_sha256,
            "post_dedup_unique_bytes": base_terminal.get(
                "conservative_unique_capacity_bytes_after"
            ),
            "nomis1864_deauthorization": removal,
        },
        "wikisource_lesia1892": {
            "source_family": contract.SOURCE_FAMILY_ID,
            "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
            "materialization_report_identity_sha256": EXPECTED_REPORT_IDENTITY_SHA256,
            "candidate_source_object_count": EXPECTED_CANDIDATE_RECORDS,
            "candidate_normalized_utf8_bytes": EXPECTED_CANDIDATE_BYTES,
            "rejected_source_object_count": EXPECTED_REJECTED_RECORDS,
            "matcher_projection_identity_sha256": intake_receipt[
                "matcher_projection_identity_sha256"
            ],
            "intake_receipt_identity_sha256": intake_receipt["receipt_identity_sha256"],
            "post_dedup_selected_extension_source_object_count": survivors[
                "post_dedup_selected_extension_source_object_count"
            ],
            "post_dedup_selected_extension_payload_bytes": survivors[
                "post_dedup_selected_extension_payload_bytes"
            ],
            "marginal_global_unique_capacity_bytes": marginal,
            "global_dedup_loss_vs_raw_candidate_bytes": EXPECTED_CANDIDATE_BYTES - marginal,
            "source_execution": copy.deepcopy(intake_receipt["source_execution"]),
            "source_test_qualification": copy.deepcopy(
                intake_receipt["source_test_qualification"]
            ),
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "declared_capacity_bytes": EXPECTED_COMBINED_BYTES,
            "report_sha256": combined_report.get("report_sha256"),
            "post_dedup_unique_bytes": combined_terminal.get(
                "conservative_unique_capacity_bytes_after"
            ),
        },
        "survivor_authority_sha256": survivors["survivor_authority_sha256"],
        "content_boundary": {
            "raw_candidate_uploaded": False,
            "raw_source_text_emitted_in_evidence": False,
            "dedup_reports_text_free": True,
            "survivor_authority_text_free": True,
        },
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": _sha256(_canonical(evidence_core)),
    }

    try:
        published_base_report = json.loads(base_report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikisourceGlobalDedupError(
            "frozen base report is not strict UTF-8 JSON"
        ) from exc
    _require(type(published_base_report) is dict, "frozen base report root invalid")
    _require(
        incumbent._canonical(dict(published_base_report)) == base_report_bytes,
        "frozen base report canonical bytes drift",
    )
    _require(
        published_base_report.get("report_sha256") == base_report_sha256,
        "frozen base report identity drift",
    )
    incumbent._publish_json_outputs(
        (
            (output_base_report, published_base_report),
            (output_combined_report, combined_report),
            (output_survivors, survivors),
            (output_evidence, evidence),
        )
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--output-base-report", type=Path, required=True)
    parser.add_argument("--output-combined-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()
    evidence = execute(
        v7_root=args.v7_root,
        bulk_workspace=args.bulk_workspace,
        candidate_jsonl=args.candidate_jsonl,
        materialization_report=args.materialization_report,
        output_base_report=args.output_base_report,
        output_combined_report=args.output_combined_report,
        output_survivors=args.output_survivors,
        output_evidence=args.output_evidence,
        expected_execution_head=args.expected_execution_head,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    print("D03_WIKISOURCE_LESIA1892_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print(
        "MARGINAL_GLOBAL_UNIQUE_CAPACITY_BYTES="
        + str(evidence["wikisource_lesia1892"]["marginal_global_unique_capacity_bytes"])
    )
    print("SURVIVOR_AUTHORITY_SHA256=" + evidence["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
