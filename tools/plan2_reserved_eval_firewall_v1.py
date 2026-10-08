"""Plan 2 S8: independent reserved-evaluation firewall over verified S7 survivors.

Reuse DATA-232 as the only exact/near/fragment contamination matcher. The
repository-pinned LOCAL_FREE fixture proves component behavior, not real final
test custody, dataset release, tokenizer fit, or any training authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import plan2_exact_dedup_v1 as exact
from tools import plan2_near_dedup_v1 as near
from tools.plan2_physical_materialization_v1 import (
    _atomic_write,
    _json,
    _read_destination,
    _read_source,
)
from twelve_six.data.decontamination_authority_v2 import (
    DecontaminationError,
    build_report,
    verify_report,
)

SCHEMA = "12-6.plan2-reserved-eval-firewall.v1"
RESERVE_PATH = "configs/data/plan2_reserved_eval_fixture_v1.json"
RESERVE_GIT_BLOB = "eae928ce3d7c828f570501bb8d232151b99d8103"
RESERVE_SCHEMA = "12-6.plan2-reserved-eval-fixture.v1"
ROLES = ("selection_validation", "final_test")


class Plan2EvalFirewallError(ValueError):
    """Reserved data authority or decontamination proof was not established."""


def _need(value: bool, reason: str) -> None:
    if not value:
        raise Plan2EvalFirewallError(reason)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return near._canonical(value)


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + bytes([0]) + raw).hexdigest()


def _reserve(raw: bytes) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    _need(type(raw) is bytes and _git_blob(raw) == RESERVE_GIT_BLOB,
          "independent reserved fixture Git identity drift")
    try:
        value = _json(raw)
    except (ValueError, UnicodeError) as exc:
        raise Plan2EvalFirewallError("invalid reserved fixture encoding/JSON") from exc
    _need(set(value) == {
        "schema_version", "purpose", "training_allowed", "tokenizer_fit_allowed",
        "teacher_auto_reentry_allowed", "real_final_test_material_accessed",
        "reserved_sources",
    }, "reserved fixture key or schema drift")
    _need(value["schema_version"] == RESERVE_SCHEMA and
          value["purpose"] == "LOCAL_FREE_SYNTHETIC_RESERVED_FIREWALL" and
          value["training_allowed"] is False and
          value["tokenizer_fit_allowed"] is False and
          value["teacher_auto_reentry_allowed"] is False and
          value["real_final_test_material_accessed"] is False,
          "reserved boundary weakened")
    sources = value["reserved_sources"]
    _need(type(sources) is list and len(sources) == 2,
          "selection/final holdouts must both be independent")
    eval_rows: list[dict[str, str]] = []
    authorities: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    roles: set[str] = set()
    texts: list[str] = []
    for source in sources:
        _need(type(source) is dict and set(source) == {
            "role", "source_id", "source_family", "modality", "records",
        }, "reserved source manifest keys drift")
        role = source["role"]
        _need(type(role) is str and role in ROLES and role not in roles,
              "duplicate/unknown reserved role")
        roles.add(role)
        _need(all(type(source[k]) is str and bool(source[k])
                  for k in ("source_id", "source_family")) and
              source["modality"] == "uk", "reserved source identity drift")
        records = source["records"]
        _need(type(records) is list and bool(records),
              "empty reserved holdout")
        for record in records:
            _need(type(record) is dict and set(record) == {"record_id", "text"}
                  and all(type(record[k]) is str and bool(record[k])
                          for k in ("record_id", "text")), "reserved record malformed")
            _need(record["record_id"] not in seen_ids,
                  "reused reserved record identity")
            seen_ids.add(record["record_id"])
            texts.append(record["text"])
            eval_rows.append({
                "record_id": record["record_id"],
                "source_id": source["source_id"],
                "source_family": source["source_family"],
                "modality": source["modality"],
                "text": record["text"],
            })
        authorities.append({
            "authority_id": source["source_id"],
            "role": role,
            "identity_sha256": _sha(_canonical(source)),
            "source_sha": RESERVE_GIT_BLOB,
        })
    _need(roles == set(ROLES), "both reserved roles required")
    _need(all(near.match_kind(a, b) is None
              for i, a in enumerate(texts) for b in texts[i + 1:]),
          "reserved evaluation roles overlap")
    return value, sorted(eval_rows, key=lambda r: r["record_id"]), {
        "schema": "12-6.data232-reserved-authorities.v1",
        "authorities": sorted(authorities, key=lambda r: r["authority_id"]),
    }


def _retained_rows(
    cohorts: Sequence[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        near_receipt = near.inspect_near(cohorts)
        exact_receipt = exact.inspect_exact(cohorts)
    except (KeyError, TypeError, ValueError) as exc:
        raise Plan2EvalFirewallError("S7/S6 authority invalid") from exc
    expected = set(near_receipt["retained_record_ids"])
    provenance = {row["record_id"]: row for row in exact_receipt["record_inventory"]}
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for normalized, payload, _privacy in cohorts:
        sid = normalized["source_id"]
        for part in normalized["records"]:
            rid = f"{sid}:r{part['index']:08d}"
            if rid not in expected:
                continue
            _need(rid not in seen and rid in provenance,
                  "S7 survivor identity duplicated or missing")
            seen.add(rid)
            raw = payload[part["start_byte"]:part["end_byte"]]
            _need(_sha(raw) == provenance[rid]["normalized_record_sha256"],
                  "S7 survivor record bytes drift")
            try:
                decoded = raw.decode("utf-8", "strict")
            except UnicodeError as exc:
                raise Plan2EvalFirewallError("S7 survivor encoding invalid") from exc
            rows.append({
                "record_id": rid, "source_id": sid,
                "source_family": sid, "modality": "uk", "text": decoded,
            })
    _need(seen == expected and bool(rows), "S7 retained cohort incomplete")
    return near_receipt, sorted(rows, key=lambda r: r["record_id"])


def inspect_firewall(
    cohorts: Sequence[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]],
    reserved_fixture_bytes: bytes,
    *,
    generated_candidates: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Fail-closed S7->DATA232->S8 boundary, with generated-data isolation."""
    _fixture, eval_rows, authorities = _reserve(reserved_fixture_bytes)
    near_receipt, training = _retained_rows(cohorts)
    _need(
        {r["source_id"] for r in training}.isdisjoint(
            {r["source_id"] for r in eval_rows})
        and {r["source_family"] for r in training}.isdisjoint(
            {r["source_family"] for r in eval_rows}),
        "training source identity aliases independently reserved evaluation",
    )
    identity = _sha(_canonical(eval_rows))
    selection = _sha(_canonical([r for r in eval_rows
                                 if r["record_id"].startswith("reserved.selection.")]))
    final = _sha(_canonical([r for r in eval_rows
                             if r["record_id"].startswith("reserved.final.")]))
    _need(selection != final and len(training) > 0,
          "reserved identities are not separate")
    try:
        report = build_report(
            training, eval_rows,
            training_corpus_identity=near_receipt["manifest_sha256"],
            selection_validation_identity=selection,
            final_test_identity=final,
            authorities=authorities,
            quarantine_cross_source_families=True,
        )
        verify_report(report)
    except (DecontaminationError, TypeError, ValueError) as exc:
        raise Plan2EvalFirewallError("DATA232 decontamination failed closed") from exc
    excluded_hashes = {r["record_id_sha256"] for r in report["excluded_records"]}
    rejected = sorted(r["record_id"] for r in training
                      if _sha(r["record_id"].encode()) in excluded_hashes)
    kept = sorted(r["record_id"] for r in training
                  if _sha(r["record_id"].encode()) not in excluded_hashes)
    _need(len(rejected) == report["counts"]["excluded_training_records"]
          and len(kept) + len(rejected) == len(training),
          "DATA232 exclusion accounting mismatch")
    _need(isinstance(generated_candidates, (tuple, list)),
          "generated candidates must be explicitly typed")
    for row in generated_candidates:
        _need(isinstance(row, Mapping)
              and row.get("origin_type") in {"teacher", "self_generated"}
              and all(type(row.get(key)) is str and bool(row[key])
                      for key in ("record_id", "source_id", "source_family", "text")),
              "untrusted generated candidate origin")
        candidate = {key: row[key] for key in (
            "record_id", "source_id", "source_family", "text"
        )}
        candidate["modality"] = "uk"
        _need(candidate["record_id"] not in {r["record_id"] for r in training},
              "generated record duplicates authorized S7 record ID")
        try:
            overlap = build_report(
                [candidate], eval_rows,
                training_corpus_identity=near_receipt["manifest_sha256"],
                selection_validation_identity=selection,
                final_test_identity=final,
                authorities=authorities,
            )
            verify_report(overlap)
        except (DecontaminationError, TypeError, ValueError) as exc:
            raise Plan2EvalFirewallError("generated candidate verification failed") from exc
        _need(overlap["counts"]["excluded_training_records"] == 0,
              "teacher/self-generated reserved evaluation answer leakage denied")
    core = {
        "schema_version": SCHEMA,
        "upstream_s7_manifest_sha256": near_receipt["manifest_sha256"],
        "reserved_fixture_sha256": _sha(reserved_fixture_bytes),
        "reserved_roles": sorted(ROLES),
        "reserved_record_count": len(eval_rows),
        "reserved_combined_identity_sha256": identity,
        "input_training_candidate_count": len(training),
        "decontaminated_record_count": len(kept),
        "decontaminated_record_ids": kept,
        "excluded_record_ids": rejected,
        "data232_report": report,
        "generated_candidates_audited": len(generated_candidates),
        "generated_auto_reentry_authorized": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "real_final_test_material_accessed": False,
        "raw_text_emitted": False,
    }
    return {**core, "manifest_sha256": _sha(_canonical(core))}


def stage_firewall(root: Path, destination: Path) -> dict[str, Any]:
    """Immutable publication; exact same-destination restart and fresh rebuild."""
    _need(not any(p.is_symlink() for p in (destination, *destination.parents)),
          "symlink destination")
    try:
        reserved = _read_source(root, RESERVE_PATH)
        near_receipt = near.stage_near(root, destination / "near")
        prefix = destination / "near" / "exact" / "privacy"
        normalized = _json(_read_destination(
            prefix / "normalization" / "normalization-manifest.json"))
        payload = _read_destination(prefix / "normalization" / "cohort" / "normalized.utf8")
        privacy_receipt = _json(_read_destination(prefix / "privacy-manifest.json"))
        result = inspect_firewall([(normalized, payload, privacy_receipt)], reserved)
        _need(result["decontaminated_record_count"] +
              len(result["excluded_record_ids"]) == near_receipt["retained_record_count"],
              "staged S7 retained record count drift")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Plan2EvalFirewallError("physical S8 firewall staging denied") from exc
    core = {k: v for k, v in result.items() if k != "manifest_sha256"}
    core["physical_s7_manifest_sha256"] = near_receipt["manifest_sha256"]
    manifest = {**core, "manifest_sha256": _sha(_canonical(core))}
    target = destination / "reserved-eval-firewall-manifest.json"
    blob = _canonical(manifest)
    if target.exists() or target.is_symlink():
        _need(_read_destination(target) == blob, "immutable S8 firewall manifest drift")
    else:
        _atomic_write(destination, target, blob)
    _need(_read_destination(target) == blob, "S8 physical readback drift")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S8 LOCAL_FREE eval firewall")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = stage_firewall(args.root, args.out_dir)
    print(json.dumps({
        "status": "PASS_DECONTAMINATED_CANDIDATE_ONLY",
        "manifest_sha256": result["manifest_sha256"],
        "decontaminated_record_count": result["decontaminated_record_count"],
        "training_corpus_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
