"""Plan-2 S8: S7-bound, reserved DATA-232 decontamination (LOCAL_FREE only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import plan2_near_dedup_v1 as near
from tools.plan2_physical_materialization_v1 import _atomic_write, _json, _read_destination
from twelve_six.data.decontamination_authority_v2 import build_report, verify_report

SCHEMA = "12-6.plan2-eval-firewall-candidate.v1"
RESERVED = "configs/data/plan2_reserved_eval_fixture_v1.json"
RESERVED_BLOB = "4a4d403352e1a98c35c1be9447572f8e4e341f02"


class EvalFirewallError(ValueError):
    """Untrusted reserved set, training lineage or persistent evidence."""


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise EvalFirewallError(msg)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reservation(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pin reviewed synthetic holdout; neither final-test nor training authority."""
    path = root / RESERVED
    require(path.is_file() and not path.is_symlink(), "reserved fixture missing")
    raw = path.read_bytes()
    blob = hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw
    ).hexdigest()
    require(blob == RESERVED_BLOB, "reserved fixture changed")
    try:
        obj = _json(raw)
    except (ValueError, UnicodeError) as exc:
        raise EvalFirewallError("reserved JSON invalid") from exc
    require(type(obj) is dict and set(obj) == {
        "schema_version", "purpose", "release_scope", "reserved_before_packing",
        "training_prohibited", "outcome_fields_included", "sets",
    }, "reserved schema drift")
    require(obj["schema_version"] == "12-6.plan2-reserved-eval-source.v1"
            and obj["purpose"].startswith("PROJECT_AUTHORED_SYNTHETIC_ONLY")
            and obj["release_scope"] == "FIXTURE_ONLY_NOT_FINAL_TEST"
            and obj["reserved_before_packing"] is True
            and obj["training_prohibited"] is True
            and obj["outcome_fields_included"] is False,
            "reserved rights/outcome drift")
    require(type(obj["sets"]) is list and len(obj["sets"]) == 2,
            "reserved role count drift")
    roles = set()
    ids = set()
    rows = []
    meta = []
    for item in obj["sets"]:
        require(type(item) is dict and set(item) == {
            "role", "authority_id", "source_family", "members",
        }, "reserved set fields drift")
        role = item["role"]
        require(role in {"selection_validation", "final_test"}
                and role not in roles, "duplicate or unknown reserved role")
        roles.add(role)
        members = item["members"]
        require(type(members) is list and bool(members), "empty reserved role")
        for member in members:
            require(type(member) is dict and set(member) == {
                "record_id", "source_id", "source_family", "modality", "text",
            }, "outcome-bearing reserved member")
            require(all(type(x) is str and bool(x) for x in member.values())
                    and member["record_id"] not in ids
                    and member["source_family"] == item["source_family"]
                    and member["modality"] in {"uk", "en", "code"},
                    "reserved member identity drift")
            ids.add(member["record_id"])
            rows.append(member)
        meta.append({
            "authority_id": item["authority_id"],
            "identity_sha256": sha(canonical(members)),
            "role": role,
            "source_sha": RESERVED_BLOB,
        })
    require(roles == {"selection_validation", "final_test"},
            "reserved roles incomplete")
    return rows, {"fixture_sha256": sha(raw), "authorities": meta}


def inspect(
    cohorts: Sequence[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]],
    root: Path,
    *,
    origins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reverify incumbent S7/S6, remove eval matches, quarantine generated text."""
    evaluation, reserved = reservation(root)
    try:
        prior = near.inspect_near(cohorts)
    except (ValueError, TypeError, KeyError) as exc:
        raise EvalFirewallError("S7 authority invalid") from exc
    require(prior.get("schema_version") == near.SCHEMA
            and prior.get("training_corpus_authorized") is False,
            "S7 authority escalation")
    chosen = set(prior["retained_record_ids"])
    require(bool(chosen), "no retained candidate")
    declared = dict(origins or {})
    require(set(declared).issubset(chosen)
            and all(v in {"admitted_source", "teacher", "self_generated"}
                    for v in declared.values()),
            "unbound generation provenance")
    training = []
    quarantine = []
    for manifest, raw, _receipt in cohorts:
        source = manifest["source_id"]
        for offset in manifest["records"]:
            rid = f"{source}:r{offset['index']:08d}"
            if rid not in chosen:
                continue
            chunk = raw[offset["start_byte"]:offset["end_byte"]]
            require(sha(chunk) == offset["sha256"], "S7 record bytes drift")
            if declared.get(rid, "admitted_source") in {"teacher", "self_generated"}:
                quarantine.append(rid)
                continue
            try:
                value = chunk.decode("utf-8", "strict")
            except UnicodeError as exc:
                raise EvalFirewallError("record encoding invalid") from exc
            training.append({
                "record_id": rid, "source_id": source,
                "source_family": source, "modality": "uk", "text": value,
            })
    require(bool(training) and len(training) + len(quarantine) == len(chosen),
            "training candidate identity incomplete")
    meta = {"authorities": reserved["authorities"]}
    selection = next(x["identity_sha256"] for x in meta["authorities"]
                     if x["role"] == "selection_validation")
    final = next(x["identity_sha256"] for x in meta["authorities"]
                 if x["role"] == "final_test")
    try:
        report = build_report(
            training, evaluation, training_corpus_identity=prior["manifest_sha256"],
            selection_validation_identity=selection, final_test_identity=final,
            authorities=meta, quarantine_cross_source_families=True,
        )
        verify_report(report)
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise EvalFirewallError("DATA-232 rejected decontamination") from exc
    require(report["hash_only_evidence"] is True
            and report["final_test_outcomes_read"] is False
            and report["training_executed"] is False,
            "DATA-232 safety invariant drift")
    hash_ids = {sha(x["record_id"].encode()): x["record_id"] for x in training}
    require(len(hash_ids) == len(training), "training hash collision")
    removed = set(quarantine)
    for entry in report["excluded_records"]:
        key = entry["record_id_sha256"]
        require(key in hash_ids, "DATA-232 exclusion unknown")
        removed.add(hash_ids[key])
    kept = sorted(chosen - removed)
    core = {
        "schema_version": SCHEMA,
        "s7_manifest_sha256": prior["manifest_sha256"],
        "reserved_fixture_git_blob": RESERVED_BLOB,
        "reserved_fixture_sha256": reserved["fixture_sha256"],
        "selection_validation_identity_sha256": selection,
        "final_test_identity_sha256": final,
        "incumbent_decontamination_report_sha256": report["report_sha256"],
        "decontamination_authority": "DATA-232 v2",
        "reserved_before_packing": True,
        "generated_policy": "QUARANTINE_ALL_TEACHER_AND_SELF_GENERATED",
        "input_record_count": len(chosen),
        "retained_record_ids": kept,
        "excluded_record_ids": sorted(removed),
        "retained_count": len(kept),
        "excluded_count": len(removed),
        "reserved_eval_record_count": len(evaluation),
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "raw_text_emitted": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": sha(canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    """Physical S4→S8 replay, immutable manifest, stable restart readback."""
    require(not any(p.is_symlink() for p in (destination, *destination.parents)),
            "symlink destination")
    predecessor = near.stage_near(root, destination / "near")
    base = destination / "near" / "exact" / "privacy"
    try:
        normalized = json.loads(_read_destination(
            base / "normalization" / "normalization-manifest.json"))
        raw = _read_destination(base / "normalization" / "cohort" / "normalized.utf8")
        privacy = json.loads(_read_destination(base / "privacy-manifest.json"))
    except (ValueError, OSError) as exc:
        raise EvalFirewallError("S7 physical input missing") from exc
    output = inspect(((normalized, raw, privacy),), root)
    require(output["s7_manifest_sha256"] == predecessor["manifest_sha256"],
            "upstream S7 manifest changed")
    file = destination / "eval-firewall-manifest.json"
    expected = canonical(output)
    if file.exists() or file.is_symlink():
        require(_read_destination(file) == expected, "immutable firewall drift")
    else:
        _atomic_write(destination, file, expected)
    require(_read_destination(file) == expected, "firewall readback drift")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 2 S8 eval firewall")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = stage(args.root, args.out_dir)
    print(json.dumps({
        "status": "PASS_SOURCE_CANDIDATE_ONLY",
        "manifest_sha256": receipt["manifest_sha256"],
        "retained_count": receipt["retained_count"],
        "excluded_count": receipt["excluded_count"],
        "training_corpus_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
