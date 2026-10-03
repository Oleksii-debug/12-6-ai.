#!/usr/bin/env python3
"""Execute exact source-admitted Caselaw through the incumbent global-dedup authority.

This runner is execution glue only.  It composes the exact merged Caselaw intake
with the exact reconstructed V8 graph, runs the incumbent V3 all-pairs authority
and the merged performance-equivalent indexed executor, requires byte-identical
reports, and emits only text-free zero-credit evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
try:
    import resource
except ImportError:  # pragma: no cover - Windows/local fallback
    resource = None
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_expanded_global_dedup_v9 as v9_runner
import run_d03_nomis_free_clean_successor_v1 as clean_successor
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data import caselaw_source_admitted_dedup_intake as caselaw
from twelve_six.data import external_llm_provenance_quarantine_v1 as quarantine
from twelve_six.data import expanded_global_dedup_v9 as v9_semantics
from twelve_six.data import incumbent_dedup_indexed_execution as indexed

SCHEMA = "12-6.d03-caselaw-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-caselaw-global-dedup-survivors.v1"
EXPECTED_MAIN = "bd2d445dfd8fbd7ec6759c1398bb913f4e0c0093"
EXECUTION_CLAIM = 2257
EXECUTION_PR = 2259
CASELAW_FINAL_HEAD = "deaf0730fe04a12e9abb8f3cecb14d6ad2cc7a4d"
EXPECTED_BASE_OBJECTS = 263
EXPECTED_BASE_BYTES = 6_093_965
EXPECTED_CASELAW_OBJECTS = 5_658
EXPECTED_CASELAW_BYTES = 5_962_147
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_CASELAW_OBJECTS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_CASELAW_BYTES
PAYLOAD_BYTES_SEMANTICS = "DECLARED_CAPACITY_BYTES"

AUTHORITY_PATHS = (
    "src/twelve_six/data/caselaw_source_admitted_dedup_intake.py",
    "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
    "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
    "src/twelve_six/data/expanded_global_dedup_v9.py",
    "src/twelve_six/data/_expanded_global_dedup_v9_impl.py",
    "tools/run_d03_expanded_global_dedup_v9.py",
    "tools/run_d03_nomis_free_clean_successor_v1.py",
    "src/twelve_six/data/external_llm_provenance_quarantine_v1.py",
    "configs/data/d03_external_llm_provenance_quarantine_v1.json",
    "tools/run_next100_065f_global_dedup_v8.py",
    "tools/materialize_data_bulk_code1_permissive_python_bundle.py",
    "configs/data/next100_065f_global_dedup_v8.json",
    "configs/data/data_bulk_code1_permissive_python_bundle_v1.json",
    "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json",
)

NOMIS_REMOVAL_PROOF = {
    "blocked_family": clean_successor.BLOCKED_FAMILY,
    "blocked_payload_bytes": clean_successor.BLOCKED_BYTES,
    "blocked_payload_sha256": clean_successor.BLOCKED_SHA256,
    "blocked_source_id": clean_successor.BLOCKED_SOURCE_ID,
    "non_blocked_source_ids_byte_for_byte_preserved": True,
    "post_source_capacity_bytes": 2_213_956,
    "post_source_object_count": 34,
    "pre_source_capacity_bytes": 2_215_615,
    "pre_source_object_count": 35,
    "quarantine_identity_sha256": clean_successor.EXPECTED_QUARANTINE_IDENTITY,
    "removed_before_new_global_dedup": True,
}


class CaselawGlobalDedupError(RuntimeError):
    """Fail-closed physical execution or authority mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CaselawGlobalDedupError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verify_removal_proof(proof: Any) -> None:
    _require(
        type(proof) is dict and _canonical(proof) == _canonical(NOMIS_REMOVAL_PROOF),
        "Nomis deauthorization proof drift",
    )


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CaselawGlobalDedupError(f"cannot execute git: {exc}") from exc
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise CaselawGlobalDedupError(f"git {' '.join(args)} failed: {detail}")
    return proc


def verify_repository_authority() -> dict[str, str]:
    """Bind authority-bearing runtime files to the exact main that merged #1459."""
    ancestor = _git("merge-base", "--is-ancestor", EXPECTED_MAIN, "HEAD", check=False)
    _require(
        ancestor.returncode == 0,
        f"pinned main {EXPECTED_MAIN} is not an ancestor of execution HEAD",
    )
    blobs: dict[str, str] = {}
    for path in AUTHORITY_PATHS:
        expected = _git("rev-parse", f"{EXPECTED_MAIN}:{path}").stdout.strip()
        observed = _git("rev-parse", f"HEAD:{path}").stdout.strip()
        _require(bool(expected) and observed == expected, f"authority path drift: {path}")
        blobs[path] = observed
    return blobs


_HISTORICAL_MATCHER_MODULES = (
    # V5 imports pipeline during exact V7 graph reconstruction. Current main no
    # longer carries this path, so require the module to originate from V7 too.
    "twelve_six.data.pipeline",
    "twelve_six.data._data232_decontamination_matching",
    "twelve_six.data.cross_source_capacity_audit",
    "twelve_six.data.cross_source_capacity_audit_v3",
    "twelve_six.data.cross_source_capacity_audit_v4",
    "twelve_six.data.cross_source_capacity_audit_v5",
    "twelve_six.data.cross_source_capacity_audit_v6",
    "twelve_six.data.cross_source_capacity_audit_v7",
)


def _verify_clean_payload_graph(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    quarantine_authority: Mapping[str, Any],
) -> None:
    rows = inventory.get("sources")
    _require(type(rows) is list and bool(rows), "clean base source rows missing")
    ids = [row.get("source_id") for row in rows if type(row) is dict]
    _require(
        len(ids) == len(rows)
        and all(type(source_id) is str and bool(source_id) for source_id in ids)
        and len(set(ids)) == len(ids)
        and set(ids) == set(payloads)
        and all(type(raw) is bytes for raw in payloads.values()),
        "clean base inventory/payload coverage mismatch",
    )
    quarantine.reject_quarantined_inventory_rows(
        [
            {
                "source_id": row["source_id"],
                "family": row.get("source_family"),
                "payload_sha256": _sha256(payloads[row["source_id"]]),
                "payload_bytes": len(payloads[row["source_id"]]),
            }
            for row in rows
        ],
        quarantine_authority,
    )


def _declared_capacity_bytes(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    label: str,
) -> int:
    rows = inventory.get("sources")
    _require(type(rows) is list and bool(rows), f"{label} source rows missing")
    seen: set[str] = set()
    total = 0
    for row in rows:
        _require(type(row) is dict, f"{label} source row invalid")
        source_id = row.get("source_id")
        _require(
            type(source_id) is str and bool(source_id) and source_id not in seen,
            f"{label} source identity invalid or duplicate",
        )
        declared = row.get("declared_capacity_bytes")
        _require(
            type(declared) is int and declared >= 0,
            f"{label} declared capacity must be exact nonnegative int",
        )
        seen.add(source_id)
        total += declared
    _require(seen == set(payloads), f"{label} inventory/payload coverage mismatch")
    return total


def _reconstruct_clean_source_inputs(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
    v9_runner.validate_v7_checkout(v7_root)
    clean_successor.validate_runtime_bindings(ROOT)
    quarantine_authority = json.loads(
        (ROOT / clean_successor.QUARANTINE_CONFIG_PATH).read_text(encoding="utf-8")
    )
    v7, _, historical_inventory, historical_payloads = v8._capture_terminal_v7(
        v7_root, config
    )
    clean_inventory, clean_payloads, removal = clean_successor.deauthorize_exact_nomis(
        historical_inventory,
        historical_payloads,
        quarantine_authority,
        quarantine,
    )
    _verify_removal_proof(removal)
    _, bulk_rows, bulk_payloads = v8._materialize_bulk(ROOT, bulk_workspace, config)
    existing_ids = {row["source_id"] for row in clean_inventory["sources"]}
    _require(not (existing_ids & set(bulk_payloads)), "clean base/bulk source-id collision")
    inventory = copy.deepcopy(clean_inventory)
    inventory["sources"] = [*inventory["sources"], *bulk_rows]
    payloads = dict(clean_payloads)
    payloads.update(bulk_payloads)
    _verify_clean_payload_graph(inventory, payloads, quarantine_authority)
    _require(len(payloads) == EXPECTED_BASE_OBJECTS, "clean base object-count drift")
    _require(
        _declared_capacity_bytes(inventory, payloads, label="clean base")
        == EXPECTED_BASE_BYTES,
        "clean base declared-capacity drift",
    )
    return v7.v6.v3, inventory, payloads, removal


def _reconstruct_v8_with_historical_namespace(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
    for module_name in _HISTORICAL_MATCHER_MODULES:
        _require(module_name not in sys.modules, f"historical matcher preloaded: {module_name}")

    twelve_six_pkg = importlib.import_module("twelve_six")
    data_pkg = importlib.import_module("twelve_six.data")
    current_package_path = list(twelve_six_pkg.__path__)
    current_data_path = list(data_pkg.__path__)
    historical_package = str((v7_root / "src" / "twelve_six").resolve(strict=True))
    historical_data = str(
        (v7_root / "src" / "twelve_six" / "data").resolve(strict=True)
    )
    _require(
        historical_package not in current_package_path
        and historical_data not in current_data_path,
        "historical V7 package path already injected",
    )

    twelve_six_pkg.__path__ = [historical_package, *current_package_path]
    data_pkg.__path__ = [historical_data, *current_data_path]
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()
    try:
        matcher, inventory, payloads, removal = _reconstruct_clean_source_inputs(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    finally:
        twelve_six_pkg.__path__ = current_package_path
        data_pkg.__path__ = current_data_path
        sys.dont_write_bytecode = previous_dont_write_bytecode
        importlib.invalidate_caches()

    historical_root = Path(historical_data)
    for module_name in _HISTORICAL_MATCHER_MODULES:
        module = sys.modules.get(module_name)
        _require(module is not None, f"historical matcher failed to load: {module_name}")
        module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        _require(
            module_path.is_relative_to(historical_root),
            f"historical matcher escaped exact V7 worktree: {module_name}",
        )
    return matcher, inventory, payloads, removal

def _compose_graph(
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    extension_sources: list[dict[str, Any]],
    extension_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    _require(type(base_inventory) is dict, "base inventory must be exact dict")
    _require(
        all(type(key) is str and type(value) is bytes for key, value in base_payloads.items()),
        "base payload map must be exact str->bytes",
    )
    _require(
        all(type(key) is str and type(value) is bytes for key, value in extension_payloads.items()),
        "extension payload map must be exact str->bytes",
    )
    rows = base_inventory.get("sources")
    _require(isinstance(rows, list) and bool(rows), "base source rows missing")
    base_ids = {row.get("source_id") for row in rows if isinstance(row, Mapping)}
    _require(len(base_ids) == len(rows) and None not in base_ids, "base source ids invalid")
    _require(base_ids == set(base_payloads), "base inventory/payload coverage mismatch")
    extension_ids = {
        row.get("source_id") for row in extension_sources if isinstance(row, Mapping)
    }
    _require(
        len(extension_ids) == len(extension_sources) and None not in extension_ids,
        "extension source ids invalid",
    )
    _require(extension_ids == set(extension_payloads), "extension inventory/payload mismatch")
    _require(not (base_ids & extension_ids), "Caselaw source-id collision with incumbent graph")

    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(extension_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed Nomis-free V8 source graph plus exact PR #1347 "
        "source-admitted Caselaw rows; pair decisions delegate to terminal PR #824 "
        "V3 semantics and indexed execution must be byte-equivalent to the all-pairs "
        "reference."
    )
    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    quarantine_authority = json.loads(
        (ROOT / clean_successor.QUARANTINE_CONFIG_PATH).read_text(encoding="utf-8")
    )
    _verify_clean_payload_graph(inventory, payloads, quarantine_authority)
    return inventory, payloads


def _preflight_attested_lineage_warmup(matcher: Any) -> None:
    """Expose execution-induced V3 closure drift before expensive all-pairs work.

    Synthetic rows have no raw source payload and cannot contribute capacity.  Both
    attestations use the unchanged incumbent verifier; this is a tripwire, not a
    bypass or a replacement of the terminal reference/indexed comparison.
    """
    indexed.attest_incumbent_runtime(matcher)
    lineage = getattr(matcher, "_lineage_matches", None)
    _require(callable(lineage), "terminal V3 lineage function missing")
    fingerprints = [
        {
            "row": {
                "source_id": f"local-preflight-{index}",
                "source_family": "local-preflight-only",
                "stable_object_id": f"local-preflight-{index // 2}",
            }
        }
        for index in range(16)
    ]
    for _ in range(10):
        matches = lineage(fingerprints, ())
        _require(type(matches) is list and len(matches) == 8,
                 "terminal V3 lineage warmup semantics drift")
        _require(all(match.get("match_type") == "lineage_same_origin_alias"
                     and match.get("capacity_collapsing") is True
                     for match in matches),
                 "terminal V3 lineage warmup authority drift")
    indexed.attest_incumbent_runtime(matcher)


def _outer_survivor_authority(
    dedup_report: Mapping[str, Any],
    selection_projection: Mapping[str, Any],
) -> dict[str, Any]:
    sources = dedup_report.get("sources")
    _require(type(sources) is list, "dedup source vector missing")
    _require(
        all(type(row) is dict and type(row.get("source_id")) is str for row in sources),
        "dedup source row invalid",
    )
    source_ids = [row["source_id"] for row in sources]
    _require(len(set(source_ids)) == len(source_ids), "duplicate dedup source id")
    by_id = {row["source_id"]: row for row in sources}
    survivor_ids = selection_projection.get("survivor_source_ids")
    _require(type(survivor_ids) is list, "selection projection survivor ids missing")
    _require(
        all(type(source_id) is str for source_id in survivor_ids),
        "survivor source id invalid",
    )
    _require(len(set(survivor_ids)) == len(survivor_ids), "duplicate survivor source id")
    _require(all(source_id in by_id for source_id in survivor_ids), "unknown survivor id")
    caselaw_survivors = [
        source_id
        for source_id in survivor_ids
        if by_id[source_id].get("source_family") == caselaw.SOURCE_FAMILY
    ]
    caselaw_survivor_bytes = 0
    for source_id in caselaw_survivors:
        declared_capacity = by_id[source_id].get("declared_capacity_bytes")
        _require(
            type(declared_capacity) is int and declared_capacity >= 0,
            "Caselaw survivor declared capacity must be exact nonnegative int",
        )
        caselaw_survivor_bytes += declared_capacity
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "selection_projection_schema": selection_projection.get("schema_version"),
        "selection_projection_sha256": selection_projection.get(
            "survivor_authority_sha256"
        ),
        "matcher_report_sha256": dedup_report.get("report_sha256"),
        "pre_dedup_source_object_count": selection_projection.get(
            "pre_dedup_source_object_count"
        ),
        "post_dedup_survivor_source_object_count": selection_projection.get(
            "post_dedup_survivor_source_object_count"
        ),
        "pre_dedup_declared_capacity_bytes": selection_projection.get(
            "pre_dedup_declared_capacity_bytes"
        ),
        "post_dedup_declared_capacity_bytes": selection_projection.get(
            "post_dedup_declared_capacity_bytes"
        ),
        "duplicate_discount_bytes": selection_projection.get("duplicate_discount_bytes"),
        "duplicate_cluster_count": selection_projection.get("duplicate_cluster_count"),
        "duplicate_clusters": copy.deepcopy(selection_projection.get("duplicate_clusters")),
        "survivor_source_ids": copy.deepcopy(survivor_ids),
        "caselaw": {
            "source_family": caselaw.SOURCE_FAMILY,
            "pre_dedup_source_object_count": EXPECTED_CASELAW_OBJECTS,
            "pre_dedup_payload_bytes": EXPECTED_CASELAW_BYTES,
            "post_dedup_survivor_source_object_count": len(caselaw_survivors),
            "post_dedup_survivor_declared_capacity_bytes": caselaw_survivor_bytes,
        },
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "two_clean_builds_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "learned_weights_created": False,
            "final_test_payload_accessed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def _source_admission_provenance_scope(receipt: Mapping[str, Any]) -> dict[str, bool]:
    truth = receipt.get("truth_boundary")
    _require(type(truth) is dict, "source-admission truth boundary missing")
    _require(
        truth.get("upstream_source_evidence_external_llm_or_api_used") is False,
        "source-admission upstream external-LLM scope drift",
    )
    _require(
        truth.get("current_corpus_external_llm_free_claimed_by_this_adapter") is False,
        "source-admission corpus-cleanliness non-claim drift",
    )
    return {
        "upstream_source_evidence_external_llm_or_api_used": False,
        "current_corpus_external_llm_free_claimed_by_this_adapter": False,
        "historical_source_admission_report_promoted_as_corpus_global_truth": False,
    }


def _max_rss_kib() -> int | None:
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value // 1024 if value > 10_000_000 else value


PUBLICATION_SCHEMA = "12-6.d03-caselaw-output-publication.v1"
PUBLICATION_MANIFEST_MAX_BYTES = 64 * 1024
_PUBLICATION_MANIFEST_KEYS = {
    "schema_version",
    "state",
    "publication_pathset_sha256",
    "targets",
    "manifest_identity_sha256",
}
_PUBLICATION_TARGET_KEYS = {"path", "stage_path", "sha256"}
_HEX_DIGITS = frozenset("0123456789abcdef")


def _reject_manifest_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate publication manifest key: {key}")
        result[key] = value
    return result


def _reject_manifest_constant(value: str) -> Any:
    raise ValueError(f"non-finite publication manifest constant: {value}")


def _reject_manifest_float(value: str) -> Any:
    raise ValueError(f"publication manifest float is forbidden: {value}")


def _is_sha256_hex(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in _HEX_DIGITS for char in value)
    )


def _path_entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Every marker, manifest and staged payload is flushed with os.fsync().
        # Python has no portable Windows directory-handle fsync; process-crash
        # atomicity is therefore supplied by the marker/manifest/hard-link
        # protocol rather than by assuming a provider or filesystem.
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CaselawGlobalDedupError(
            f"cannot open output directory for fsync: {path}"
        ) from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise CaselawGlobalDedupError(
            f"cannot fsync output directory: {path}"
        ) from exc
    finally:
        os.close(fd)


class PublicationWriteCleanupError(CaselawGlobalDedupError):
    """A durable-create failure left residue that must remain recovery-visible."""


def _write_create_only_durable(path: Path, payload: bytes) -> None:
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CaselawGlobalDedupError(f"refusing to overwrite: {path}") from exc
    except OSError as exc:
        cleanup_error: Exception | None = None
        if created:
            try:
                path.unlink(missing_ok=True)
                _fsync_directory(path.parent)
            except (OSError, CaselawGlobalDedupError) as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None:
            raise PublicationWriteCleanupError(
                f"cannot write output and cleanup failed: {path}: {cleanup_error}"
            ) from exc
        raise CaselawGlobalDedupError(f"cannot write output: {path}") from exc


def _publication_control_paths(
    prepared: tuple[tuple[Path, bytes], ...],
) -> tuple[Path, Path, tuple[Path, ...], str]:
    _require(bool(prepared), "publication output set must not be empty")
    resolved_paths = [str(path.resolve(strict=False)) for path, _ in prepared]
    pathset_id = _sha256(_canonical({"paths": resolved_paths}))
    marker = prepared[-1][0].with_name(prepared[-1][0].name + ".incomplete")
    manifest = marker.with_name(marker.name + ".manifest")
    stages = tuple(
        path.with_name(f".{path.name}.stage-{pathset_id[:20]}")
        for path, _ in prepared
    )
    return marker, manifest, stages, pathset_id


def _publication_marker_path(
    prepared: tuple[tuple[Path, bytes], ...],
) -> Path:
    return _publication_control_paths(prepared)[0]


def _publication_manifest(
    prepared: tuple[tuple[Path, bytes], ...],
    stages: tuple[Path, ...],
    pathset_id: str,
) -> tuple[dict[str, Any], bytes]:
    targets = [
        {
            "path": str(path.resolve(strict=False)),
            "stage_path": str(stage.resolve(strict=False)),
            "sha256": _sha256(payload),
        }
        for (path, payload), stage in zip(prepared, stages, strict=True)
    ]
    core = {
        "schema_version": PUBLICATION_SCHEMA,
        "state": "INCOMPLETE_NOT_TERMINAL",
        "publication_pathset_sha256": pathset_id,
        "targets": targets,
    }
    manifest = {
        **core,
        "manifest_identity_sha256": _sha256(_canonical(core)),
    }
    return manifest, _canonical(manifest) + b"\n"


def _load_publication_manifest(manifest_path: Path) -> dict[str, Any]:
    _require(
        not manifest_path.is_symlink() and manifest_path.is_file(),
        "incomplete publication manifest is not a regular file",
    )
    try:
        with manifest_path.open("rb") as handle:
            raw = handle.read(PUBLICATION_MANIFEST_MAX_BYTES + 1)
        if len(raw) > PUBLICATION_MANIFEST_MAX_BYTES:
            raise ValueError("publication manifest exceeds bounded size")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_manifest_pairs,
            parse_constant=_reject_manifest_constant,
            parse_float=_reject_manifest_float,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise CaselawGlobalDedupError(
            "incomplete publication manifest is unreadable"
        ) from exc
    _require(type(value) is dict, "incomplete publication manifest root invalid")
    _require(
        set(value) == _PUBLICATION_MANIFEST_KEYS,
        "incomplete publication manifest keys invalid",
    )
    _require(
        raw == _canonical(value) + b"\n",
        "incomplete publication manifest is not canonical",
    )
    identity = value.get("manifest_identity_sha256")
    core = {
        key: val for key, val in value.items() if key != "manifest_identity_sha256"
    }
    _require(
        _is_sha256_hex(identity)
        and identity == _sha256(_canonical(core)),
        "incomplete publication manifest identity mismatch",
    )
    _require(
        core.get("schema_version") == PUBLICATION_SCHEMA
        and core.get("state") == "INCOMPLETE_NOT_TERMINAL",
        "incomplete publication manifest semantics invalid",
    )
    pathset = core.get("publication_pathset_sha256")
    _require(
        _is_sha256_hex(pathset),
        "incomplete publication path-set identity invalid",
    )
    targets = core.get("targets")
    _require(
        type(targets) is list and bool(targets),
        "publication manifest targets missing",
    )
    return value


def _remove_control_without_payload(
    marker_path: Path,
    manifest_path: Path,
) -> None:
    if _path_entry_exists(manifest_path):
        manifest_path.unlink()
        _fsync_directory(manifest_path.parent)
    marker_path.unlink()
    _fsync_directory(marker_path.parent)


def _recover_incomplete_publication(
    marker_path: Path,
    manifest_path: Path,
    prepared: tuple[tuple[Path, bytes], ...],
    stages: tuple[Path, ...],
    pathset_id: str,
) -> None:
    _require(
        not marker_path.is_symlink() and marker_path.is_file(),
        "incomplete publication marker is not a regular file",
    )
    expected_final_paths = [str(path.resolve(strict=False)) for path, _ in prepared]
    expected_stage_paths = [str(stage.resolve(strict=False)) for stage in stages]

    if not _path_entry_exists(manifest_path):
        _require(
            not any(_path_entry_exists(path) for path, _ in prepared)
            and not any(_path_entry_exists(stage) for stage in stages),
            "incomplete publication marker lacks manifest but payload paths exist",
        )
        _remove_control_without_payload(marker_path, manifest_path)
        return

    try:
        manifest = _load_publication_manifest(manifest_path)
    except CaselawGlobalDedupError:
        _require(
            not any(_path_entry_exists(path) for path, _ in prepared)
            and not any(_path_entry_exists(stage) for stage in stages),
            "invalid incomplete manifest coexists with payload paths",
        )
        _remove_control_without_payload(marker_path, manifest_path)
        return

    targets = manifest["targets"]
    _require(
        manifest.get("publication_pathset_sha256") == pathset_id,
        "incomplete publication path-set identity mismatch",
    )
    marker_final_paths: list[str] = []
    marker_stage_paths: list[str] = []
    for row in targets:
        _require(type(row) is dict, "publication manifest target invalid")
        _require(
            set(row) == _PUBLICATION_TARGET_KEYS,
            "publication manifest target keys invalid",
        )
        final_value = row.get("path")
        stage_value = row.get("stage_path")
        expected_sha = row.get("sha256")
        _require(
            type(final_value) is str
            and final_value
            and type(stage_value) is str
            and stage_value
            and _is_sha256_hex(expected_sha),
            "publication manifest target semantics invalid",
        )
        marker_final_paths.append(final_value)
        marker_stage_paths.append(stage_value)
    _require(
        marker_final_paths == expected_final_paths
        and marker_stage_paths == expected_stage_paths,
        "incomplete publication manifest targets do not match requested outputs",
    )

    touched_dirs: set[Path] = set()
    for row in targets:
        final_path = Path(row["path"])
        stage_path = Path(row["stage_path"])
        expected_sha = row["sha256"]
        final_exists = _path_entry_exists(final_path)
        stage_exists = _path_entry_exists(stage_path)

        if final_exists:
            _require(
                stage_exists,
                f"incomplete final output has no owning stage: {final_path}",
            )
            _require(
                not final_path.is_symlink()
                and final_path.is_file()
                and not stage_path.is_symlink()
                and stage_path.is_file(),
                "incomplete publication payload path is not regular",
            )
            try:
                same_file = os.path.samefile(final_path, stage_path)
            except OSError as exc:
                raise CaselawGlobalDedupError(
                    f"cannot verify incomplete publication ownership: {final_path}"
                ) from exc
            _require(
                same_file,
                f"incomplete publication final is not linked to its stage: {final_path}",
            )
            try:
                observed = stage_path.read_bytes()
            except OSError as exc:
                raise CaselawGlobalDedupError(
                    f"cannot inspect incomplete publication stage: {stage_path}"
                ) from exc
            _require(
                _sha256(observed) == expected_sha,
                f"incomplete publication output digest mismatch: {final_path}",
            )
            final_path.unlink()
            touched_dirs.add(final_path.parent)

        if stage_exists:
            _require(
                not stage_path.is_symlink() and stage_path.is_file(),
                f"incomplete publication stage is not regular: {stage_path}",
            )
            stage_path.unlink()
            touched_dirs.add(stage_path.parent)

    for directory in sorted(touched_dirs, key=str):
        _fsync_directory(directory)
    manifest_path.unlink()
    _fsync_directory(manifest_path.parent)
    marker_path.unlink()
    _fsync_directory(marker_path.parent)


def _rollback_current_publication(
    *,
    marker_path: Path,
    manifest_path: Path,
    marker_created: bool,
    manifest_created: bool,
    linked_finals: list[Path],
    created_stages: list[Path],
) -> list[str]:
    errors: list[str] = []
    touched_dirs: set[Path] = set()
    for path in reversed(linked_finals):
        try:
            path.unlink(missing_ok=True)
            touched_dirs.add(path.parent)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    for path in reversed(created_stages):
        try:
            path.unlink(missing_ok=True)
            touched_dirs.add(path.parent)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if not errors:
        for directory in sorted(touched_dirs, key=str):
            try:
                _fsync_directory(directory)
            except CaselawGlobalDedupError as exc:
                errors.append(str(exc))
    if not errors and manifest_created:
        try:
            manifest_path.unlink(missing_ok=True)
            _fsync_directory(manifest_path.parent)
        except (OSError, CaselawGlobalDedupError) as exc:
            errors.append(f"{manifest_path}: {exc}")
    if not errors and marker_created:
        try:
            marker_path.unlink(missing_ok=True)
            _fsync_directory(marker_path.parent)
        except (OSError, CaselawGlobalDedupError) as exc:
            errors.append(f"{marker_path}: {exc}")
    return errors


def _link_staged_output(stage_path: Path, final_path: Path) -> None:
    try:
        os.link(stage_path, final_path)
    except FileExistsError as exc:
        raise CaselawGlobalDedupError(
            f"refusing to overwrite: {final_path}"
        ) from exc
    except OSError as exc:
        raise CaselawGlobalDedupError(
            f"cannot atomically publish output: {final_path}"
        ) from exc


def _cleanup_committed_publication_residue(
    manifest_path: Path,
    stages: tuple[Path, ...],
) -> None:
    touched_dirs: set[Path] = set()
    for stage_path in stages:
        try:
            stage_path.unlink(missing_ok=True)
            touched_dirs.add(stage_path.parent)
        except OSError:
            return
    try:
        manifest_path.unlink(missing_ok=True)
        touched_dirs.add(manifest_path.parent)
    except OSError:
        return
    for directory in sorted(touched_dirs, key=str):
        try:
            _fsync_directory(directory)
        except CaselawGlobalDedupError:
            return


def _publish_json_outputs(
    outputs: tuple[tuple[Path, Mapping[str, Any]], ...],
) -> None:
    prepared_list: list[tuple[Path, bytes]] = []
    seen: set[Path] = set()
    resolved_seen: set[Path] = set()
    for path, value in outputs:
        resolved = path.resolve(strict=False)
        _require(
            path not in seen and resolved not in resolved_seen,
            f"duplicate output path: {path}",
        )
        seen.add(path)
        resolved_seen.add(resolved)
        prepared_list.append((path, _canonical(dict(value)) + b"\n"))
    prepared = tuple(prepared_list)
    _require(bool(prepared), "publication output set must not be empty")

    for path, _ in prepared:
        path.parent.mkdir(parents=True, exist_ok=True)

    marker_path, manifest_path, stages, pathset_id = _publication_control_paths(prepared)
    control_paths = (marker_path, manifest_path, *stages)
    resolved_controls = tuple(path.resolve(strict=False) for path in control_paths)
    _require(
        len(set(resolved_controls)) == len(resolved_controls)
        and not (set(resolved_controls) & resolved_seen),
        "publication control path collision",
    )

    if _path_entry_exists(marker_path):
        _recover_incomplete_publication(
            marker_path,
            manifest_path,
            prepared,
            stages,
            pathset_id,
        )

    for path, _ in prepared:
        _require(not _path_entry_exists(path), f"refusing to overwrite: {path}")
    _require(
        not _path_entry_exists(manifest_path),
        f"stale publication manifest exists without marker: {manifest_path}",
    )
    for stage in stages:
        _require(
            not _path_entry_exists(stage),
            f"stale publication stage exists without marker: {stage}",
        )

    manifest, manifest_payload = _publication_manifest(prepared, stages, pathset_id)
    _require(
        [row["stage_path"] for row in manifest["targets"]]
        == [str(stage.resolve(strict=False)) for stage in stages],
        "publication stage derivation drift",
    )

    marker_created = False
    manifest_created = False
    linked_finals: list[Path] = []
    created_stages: list[Path] = []
    try:
        _write_create_only_durable(marker_path, b"")
        marker_created = True
        _fsync_directory(marker_path.parent)

        _write_create_only_durable(manifest_path, manifest_payload)
        manifest_created = True
        _fsync_directory(manifest_path.parent)

        for (path, payload), stage_path in zip(prepared, stages, strict=True):
            _write_create_only_durable(stage_path, payload)
            created_stages.append(stage_path)
            _fsync_directory(stage_path.parent)

        for (final_path, _), stage_path in zip(prepared, stages, strict=True):
            _link_staged_output(stage_path, final_path)
            linked_finals.append(final_path)
            _fsync_directory(final_path.parent)

        # Marker removal is the sole terminal commit point.  Stages and manifest
        # deliberately remain present until after this durable state transition
        # so an interrupted pre-commit run can prove final-path ownership.
        marker_path.unlink()
        _fsync_directory(marker_path.parent)
        marker_created = False
    except Exception as exc:
        if isinstance(exc, PublicationWriteCleanupError):
            # Preserve the durable marker/control state.  Recovery must decide
            # ownership on the next invocation; do not erase evidence of an
            # incomplete create whose local cleanup already failed.
            raise
        rollback_errors = _rollback_current_publication(
            marker_path=marker_path,
            manifest_path=manifest_path,
            marker_created=marker_created,
            manifest_created=manifest_created,
            linked_finals=linked_finals,
            created_stages=created_stages,
        )
        if rollback_errors:
            raise CaselawGlobalDedupError(
                "output publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise

    _cleanup_committed_publication_residue(manifest_path, stages)




def execute(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    candidate_jsonl: Path,
    execution_evidence_json: Path,
    output_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    authority_blobs = verify_repository_authority()
    execution_head = _git("rev-parse", "HEAD").stdout.strip()
    _require(len(execution_head) == 40, "execution HEAD identity missing")
    config = v8.load_config(ROOT / "configs/data/next100_065f_global_dedup_v8.json")
    matcher, base_inventory, base_payloads, removal = _reconstruct_v8_with_historical_namespace(
        v7_root=v7_root,
        bulk_workspace=bulk_workspace,
        config=config,
    )
    _require(len(base_payloads) == EXPECTED_BASE_OBJECTS, "V8 base object count drift")
    _require(
        _declared_capacity_bytes(base_inventory, base_payloads, label="V8 base")
        == EXPECTED_BASE_BYTES,
        "V8 base declared-capacity drift",
    )
    base_comparison_payload_bytes = sum(len(raw) for raw in base_payloads.values())
    _require(base_comparison_payload_bytes > 0, "V8 base comparison payload is empty")

    projection = caselaw.validate_and_project_caselaw(
        candidate_jsonl,
        execution_evidence_json,
        upstream_head=CASELAW_FINAL_HEAD,
        retain_payloads=True,
    )
    _require(projection.sources is not None and projection.payloads is not None, "payload projection missing")
    source_admission_provenance_scope = _source_admission_provenance_scope(
        projection.receipt
    )
    extension_sources = [dict(row) for row in projection.sources]
    extension_payloads = dict(projection.payloads)
    _require(len(extension_sources) == EXPECTED_CASELAW_OBJECTS, "Caselaw projection count drift")
    caselaw_comparison_payload_bytes = sum(len(raw) for raw in extension_payloads.values())
    _require(
        caselaw_comparison_payload_bytes == EXPECTED_CASELAW_BYTES,
        "Caselaw projection comparison-payload bytes drift",
    )
    _require(
        _declared_capacity_bytes(
            {"sources": extension_sources},
            extension_payloads,
            label="Caselaw projection",
        )
        == EXPECTED_CASELAW_BYTES,
        "Caselaw projection declared-capacity drift",
    )
    inventory, payloads = _compose_graph(
        base_inventory,
        base_payloads,
        extension_sources,
        extension_payloads,
    )
    _require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined source count drift")
    _require(
        _declared_capacity_bytes(inventory, payloads, label="combined graph")
        == EXPECTED_COMBINED_BYTES,
        "combined declared-capacity drift",
    )
    combined_comparison_payload_bytes = sum(len(raw) for raw in payloads.values())
    _require(
        combined_comparison_payload_bytes
        == base_comparison_payload_bytes + caselaw_comparison_payload_bytes,
        "combined comparison-payload byte arithmetic drift",
    )

    # The merged indexed executor's runtime attestation must protect the original
    # all-pairs reference too. Differential equality is not authority if both paths
    # can observe the same mutated stdlib/runtime state before attestation.
    _preflight_attested_lineage_warmup(matcher)

    reference_started = time.perf_counter()
    reference = matcher.audit_payloads(inventory, payloads)
    reference_seconds = time.perf_counter() - reference_started
    matcher.verify_report(reference)

    indexed_started = time.perf_counter()
    indexed_report = indexed.audit_payloads_indexed(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    indexed_seconds = time.perf_counter() - indexed_started
    matcher.verify_report(indexed_report)

    reference_bytes = matcher.v1._canonical_bytes(reference)
    indexed_bytes = matcher.v1._canonical_bytes(indexed_report)
    _require(reference_bytes == indexed_bytes, "indexed report differs from incumbent all-pairs report")

    rows, _ = matcher._validate_inventory(inventory)
    validated = matcher.v1._validate_inventory(matcher._as_v1_inventory(rows))
    fingerprints = [
        matcher._fingerprint(row, payloads[row["source_id"]])
        for row in validated
    ]
    pairs, work = indexed.candidate_pair_indices_with_stats(
        matcher.v1,
        fingerprints,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    work_stats = indexed.execution_stats(
        len(fingerprints),
        len(pairs),
        index_postings=work["index_postings"],
        pair_expansion_attempts=work["pair_expansion_attempts"],
        unique_bucket_signatures=work["unique_bucket_signatures"],
    )

    terminal = indexed_report.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "terminal dedup summary missing")
    _require(
        terminal.get("declared_capacity_bytes_before") == EXPECTED_COMBINED_BYTES,
        "combined declared capacity drift",
    )
    selection_projection = v9_semantics._derive_survivors(indexed_report)
    survivors = _outer_survivor_authority(indexed_report, selection_projection)

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "GITHUB_HOSTED_FREE_LOCAL_FREE",
        "execution_claim_issue": EXECUTION_CLAIM,
        "execution_pr": EXECUTION_PR,
        "execution_head_sha": execution_head,
        "pinned_main_sha": EXPECTED_MAIN,
        "authority_path_blobs": authority_blobs,
        "baseline_v8": {
            "v7_head_sha": v8.EXPECTED_V7_HEAD,
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "payload_bytes": EXPECTED_BASE_BYTES,
            "payload_bytes_semantics": PAYLOAD_BYTES_SEMANTICS,
            "declared_capacity_bytes": EXPECTED_BASE_BYTES,
            "comparison_payload_bytes": base_comparison_payload_bytes,
            "nomis1864_deauthorization": removal,
        },
        "caselaw": {
            "source_admission_product_pr": caselaw.UPSTREAM_PRODUCT_PR,
            "source_admission_final_head": CASELAW_FINAL_HEAD,
            "candidate_sha256": caselaw.CANDIDATE_SHA256,
            "source_object_count": EXPECTED_CASELAW_OBJECTS,
            "payload_bytes": EXPECTED_CASELAW_BYTES,
            "payload_bytes_semantics": PAYLOAD_BYTES_SEMANTICS,
            "declared_capacity_bytes": EXPECTED_CASELAW_BYTES,
            "comparison_payload_bytes": caselaw_comparison_payload_bytes,
            "intake_receipt_identity_sha256": projection.receipt[
                "receipt_identity_sha256"
            ],
            "source_admission_provenance_scope": source_admission_provenance_scope,
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "payload_bytes": EXPECTED_COMBINED_BYTES,
            "payload_bytes_semantics": PAYLOAD_BYTES_SEMANTICS,
            "declared_capacity_bytes": EXPECTED_COMBINED_BYTES,
            "comparison_payload_bytes": combined_comparison_payload_bytes,
            "reference_report_sha256": reference["report_sha256"],
            "indexed_report_sha256": indexed_report["report_sha256"],
            "reports_byte_identical": True,
            "post_dedup_conservative_unique_bytes": terminal.get(
                "conservative_unique_capacity_bytes_after"
            ),
            "duplicate_discount_bytes": terminal.get("duplicate_discount_bytes"),
            "duplicate_cluster_count": terminal.get("duplicate_cluster_count"),
        },
        "indexed_execution": {
            **work_stats,
            "max_candidate_pairs": max_candidate_pairs,
            "max_index_postings": max_index_postings,
            "max_pair_expansions": max_pair_expansions,
            "reference_wall_clock_seconds": round(reference_seconds, 6),
            "indexed_wall_clock_seconds": round(indexed_seconds, 6),
            "process_max_rss_kib": _max_rss_kib(),
        },
        "survivor_authority_sha256": survivors["survivor_authority_sha256"],
        "content_boundary": {
            "raw_text_emitted": False,
            "raw_candidate_written_to_durable_evidence": False,
            "payload_artifact_required": False,
            "dedup_report_text_free": True,
            "survivor_authority_text_free": True,
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": _sha256(_canonical(evidence_core)),
    }
    _publish_json_outputs(
        (
            (output_report, indexed_report),
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
    parser.add_argument("--execution-evidence-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
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
        execution_evidence_json=args.execution_evidence_json,
        output_report=args.output_report,
        output_survivors=args.output_survivors,
        output_evidence=args.output_evidence,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    print("D03_CASELAW_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print(
        "MATCHER_REPORT_SHA256="
        + evidence["combined"]["indexed_report_sha256"]
    )
    print("SURVIVOR_AUTHORITY_SHA256=" + evidence["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
