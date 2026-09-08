#!/usr/bin/env python3
"""Fail-closed authority verifier for retained NEXT100-065F V8 evidence.

The outer V8 report is not sufficient authority by itself. This verifier also invokes
exact terminal-V7/V3 report verification on the nested dedup report and cross-binds
that nested authority to the V8 summary. It additionally binds the current-main V6
source-registry reconciliation that admitted DATA-BULK-CODE-1 only as pending input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

EXPECTED_CURRENT_MAIN = "09cfeb6da190e41e32d4476aa17d05d203e6e340"
EXPECTED_V6_REGISTRY_BLOB = "13789effe506a815e92e4f0e22ada773d366f316"
EXPECTED_V6_REGISTRY_IDENTITY = "c7b988081a270cd53c37f22721499568ec44f67daa998c87573b43c463642eae"
EXPECTED_V5_BLOB = "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf"
EXPECTED_BULK_EVIDENCE_BLOB = "b2b9b361dbf0e6c883dca4d7f006812c69cdd4e0"
EXPECTED_HISTORICAL_CAPACITY = 2_215_615
EXPECTED_PENDING_BYTES = 3_880_009
EXPECTED_COMPOSED_BYTES = 6_095_624
EXPECTED_FAMILIES = {"uk": 4, "en": 5, "code": 12}


class V8AuthorityError(RuntimeError):
    """Raised when retained V8 evidence cannot prove its exact authority chain."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V8AuthorityError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V8AuthorityError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(prefix + raw).hexdigest()


def _load_module_from_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_v7_namespace(v7_root: Path) -> None:
    package_root = (v7_root / "src" / "twelve_six").resolve()
    _require(package_root.is_dir(), f"missing terminal V7 package root: {package_root}")

    existing = sys.modules.get("twelve_six")
    if existing is not None:
        paths = [str(Path(value).resolve()) for value in getattr(existing, "__path__", [])]
        _require(str(package_root) in paths, "twelve_six already loaded from a different authority")
        return

    package = types.ModuleType("twelve_six")
    package.__package__ = "twelve_six"
    package.__path__ = [str(package_root)]
    package.__spec__ = importlib.machinery.ModuleSpec("twelve_six", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = [str(package_root)]
    sys.modules["twelve_six"] = package


def validate_current_registry(current_root: Path) -> None:
    path = current_root / "configs/data/next100_063_terminal_source_registry_v6.json"
    raw = path.read_bytes()
    _require(_git_blob_sha1(raw) == EXPECTED_V6_REGISTRY_BLOB, "current V6 registry Git blob drift")
    registry = json.loads(raw.decode("utf-8"))
    _require(
        registry.get("schema_version") == "12-6.next100-063-terminal-source-registry.v6",
        "current V6 registry schema drift",
    )
    _require(registry.get("registry_identity_sha256") == EXPECTED_V6_REGISTRY_IDENTITY, "current V6 identity drift")

    prior = registry.get("prior_registry_authority", {})
    _require(prior.get("config_git_blob_sha1") == EXPECTED_V5_BLOB, "current V6 prior-V5 binding drift")
    _require(
        prior.get("candidate_numeric_training_capacity_bytes") == EXPECTED_HISTORICAL_CAPACITY,
        "current V6 historical capacity drift",
    )

    addition = registry.get("terminal_intake_addition", {})
    _require(addition.get("source_pr") == 818, "current V6 bulk source PR drift")
    _require(addition.get("evidence_git_blob_sha1") == EXPECTED_BULK_EVIDENCE_BLOB, "current V6 bulk evidence drift")
    _require(addition.get("eligible_utf8_bytes") == EXPECTED_PENDING_BYTES, "current V6 pending intake drift")
    _require(addition.get("canonical_capacity_credit_bytes") == 0, "current V6 illegally credits pending intake")

    live = registry.get("live_pre_global_dedup_state", {})
    _require(
        live.get("credited_candidate_numeric_training_capacity_bytes") == EXPECTED_HISTORICAL_CAPACITY,
        "current V6 credited capacity drift",
    )
    _require(
        live.get("terminal_intake_eligible_bytes_pending_global_dedup") == EXPECTED_PENDING_BYTES,
        "current V6 pending-byte vector drift",
    )
    _require(
        live.get("potential_numeric_capacity_if_all_pending_intake_survives_dedup") == EXPECTED_COMPOSED_BYTES,
        "current V6 composed planning envelope drift",
    )

    boundary = registry.get("truth_boundary", {})
    _require(boundary.get("post_global_dedup_capacity_bytes") is None, "current V6 fabricates post-dedup capacity")
    _require(
        boundary.get("authorized_balanced_no_replay_loss_positions") == 0,
        "current V6 fabricates training exposure",
    )
    _require(boundary.get("model_training_executed") is False, "current V6 fabricates model training")
    _require(boundary.get("optimizer_updates") == 0, "current V6 fabricates optimizer updates")


def _nested_family_counts(nested: Mapping[str, Any]) -> dict[str, int]:
    rows = nested.get("sources", [])
    _require(isinstance(rows, list), "nested V3 sources must be a list")
    return {
        modality: len(
            {
                str(row.get("source_family"))
                for row in rows
                if isinstance(row, Mapping) and row.get("modality") == modality
            }
        )
        for modality in ("uk", "en", "code")
    }


def cross_bind_nested(report: Mapping[str, Any]) -> None:
    nested = report.get("dedup_v3")
    _require(isinstance(nested, Mapping), "V8 report missing nested V3 authority")
    vector = report.get("source_vector")
    _require(isinstance(vector, Mapping), "V8 report missing source vector")
    terminal = nested.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "nested V3 report missing terminal candidates")

    _require(nested.get("source_count") == vector.get("source_object_count"), "nested V3 source count diverges from V8")
    _require(terminal.get("source_count") == vector.get("source_object_count"), "nested terminal source count diverges from V8")
    _require(_nested_family_counts(nested) == vector.get("source_family_counts") == EXPECTED_FAMILIES, "nested family vector diverges from V8")
    _require(
        terminal.get("declared_capacity_bytes_before") == vector.get("source_capacity_bytes_before_global_dedup"),
        "nested pre-dedup capacity diverges from V8",
    )
    _require(
        terminal.get("conservative_unique_capacity_bytes_after")
        == vector.get("conservative_unique_capacity_bytes_after_global_dedup"),
        "nested post-dedup capacity diverges from V8",
    )
    _require(terminal.get("duplicate_discount_bytes") == vector.get("duplicate_discount_bytes"), "nested duplicate discount diverges from V8")
    _require(terminal.get("duplicate_cluster_count") == vector.get("duplicate_cluster_count"), "nested duplicate cluster count diverges from V8")
    _require(
        terminal.get("effective_independent_origin_count") == vector.get("effective_independent_origin_count"),
        "nested effective-origin count diverges from V8",
    )

    nested_modalities = terminal.get("by_modality")
    outer_modalities = vector.get("by_modality")
    _require(isinstance(nested_modalities, Mapping), "nested V3 modality vector missing")
    _require(isinstance(outer_modalities, Mapping), "V8 modality vector missing")
    for modality in ("uk", "en", "code"):
        inner = nested_modalities.get(modality)
        outer = outer_modalities.get(modality)
        _require(isinstance(inner, Mapping) and isinstance(outer, Mapping), f"missing {modality} modality authority")
        pairs = (
            ("source_count", "source_count"),
            ("declared_source_family_count", "source_family_count"),
            ("declared_capacity_bytes_before", "capacity_bytes_before_global_dedup"),
            ("conservative_unique_capacity_bytes_after", "conservative_unique_capacity_bytes_after_global_dedup"),
            ("duplicate_discount_bytes", "duplicate_discount_bytes"),
        )
        for inner_key, outer_key in pairs:
            _require(
                inner.get(inner_key) == outer.get(outer_key),
                f"nested {modality} {inner_key} diverges from V8 {outer_key}",
            )


def verify_nested_authority(report: Mapping[str, Any], verifier: Callable[[Mapping[str, Any]], None]) -> None:
    nested = report.get("dedup_v3")
    _require(isinstance(nested, Mapping), "V8 report missing nested V3 authority")
    verifier(nested)
    cross_bind_nested(report)


def verify_authority(current_root: Path, v7_root: Path, report: Mapping[str, Any]) -> None:
    v8 = _load_module_from_file(
        "_next100_065f_v8_structural_verifier",
        current_root / "tools/run_next100_065f_global_dedup_v8.py",
    )
    config = v8.load_config(current_root / "configs/data/next100_065f_global_dedup_v8.json")
    v8.verify_report(config, report)
    validate_current_registry(current_root)

    _install_v7_namespace(v7_root)
    v7 = importlib.import_module("twelve_six.data.cross_source_capacity_audit_v7")
    verify_nested_authority(report, v7.v6.v3.verify_report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, default=Path("."))
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = _read_json(args.report)
    verify_authority(args.current_root, args.v7_root, report)
    print(f"CURRENT_MAIN_V6_AUTHORITY={EXPECTED_CURRENT_MAIN}")
    print("PASS_V8_NESTED_AUTHORITY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
