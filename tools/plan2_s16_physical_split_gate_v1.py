"""Recheck actual Section 16 UA/EN split coverage; refusal is not release.

Uses the *physical* five-source S15 lineage, its S16 rights/privacy audit
and exact S10 candidate split.  Produces a text-free diagnostic; never
turns source-level permissions or a synthetic evaluation fixture into
production corpus, evaluation or training authorization.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_three_family_physical_v1 as physical
from tools import plan2_s15_three_family_split_probe_v1 as split
from tools import plan2_s16_source_rights_privacy_v1 as rights
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s16-physical-language-split-gaps.v1"
OUTPUT = "physical-language-split-gaps.json"
ROLES = ("train", "validation", "test")
LANGUAGES = ("uk", "en")


class PhysicalSplitDenied(ValueError):
    """The physical source/split lineage is incomplete, altered or promoted."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise PhysicalSplitDenied(reason)


def analyze_members(
    members: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    """Derive exact split and family gaps, never manufacture missing records."""
    need(isinstance(members, (tuple, list)) and bool(members),
         "no physical source records")
    need(isinstance(assignments, Mapping), "invalid split assignments")
    counts: dict[str, Counter[str]] = {role: Counter() for role in ROLES}
    byte_counts: dict[str, Counter[str]] = {role: Counter() for role in ROLES}
    family_roles: dict[str, set[str]] = defaultdict(set)
    role_families: dict[str, set[str]] = {role: set() for role in ROLES}
    source_identity: dict[str, tuple[str, str]] = {}
    seen_records: set[str] = set()
    total_bytes = 0
    for row in members:
        need(isinstance(row, Mapping), "malformed physical member")
        rid, sid, fam, lang = (
            row.get("record_id"), row.get("source_id"),
            row.get("source_family"), row.get("modality"),
        )
        size = row.get("normalized_bytes")
        need(all(type(v) is str and bool(v) for v in (rid, sid, fam)),
             "missing real source identity")
        need(type(lang) is str and lang in LANGUAGES, "unknown real language")
        need(type(size) is int and size > 0, "invalid physical byte count")
        need(rid not in seen_records, "duplicated physical record")
        seen_records.add(rid)
        prior = source_identity.setdefault(sid, (fam, lang))
        need(prior == (fam, lang), "source family or language drift")
        role = assignments.get(sid)
        need(role in ROLES, "missing or invalid physical source split")
        counts[role][lang] += 1
        byte_counts[role][lang] += size
        total_bytes += size
        family_roles[fam].add(role)
        role_families[role].add(fam)
    need(set(assignments) == set(source_identity),
         "unknown, missing or forged source role")
    need(set(assignments.values()) == set(ROLES),
         "incomplete physical source role coverage")
    missing = [
        f"MISSING_{lang.upper()}_{role.upper()}"
        for role in ROLES for lang in LANGUAGES if not counts[role][lang]
    ]
    cross_role_families = {
        fam: sorted(roles) for fam, roles in sorted(family_roles.items())
        if len(roles) > 1
    }
    if cross_role_families:
        missing.append("SOURCE_FAMILY_CROSSES_TRAIN_VALIDATION_TEST")
    return {
        "physical_records": len(seen_records),
        "physical_sources": len(source_identity),
        "physical_families": len(family_roles),
        "physical_bytes": total_bytes,
        "record_counts_by_role_language": {
            role: {lang: counts[role][lang] for lang in LANGUAGES}
            for role in ROLES
        },
        "byte_counts_by_role_language": {
            role: {lang: byte_counts[role][lang] for lang in LANGUAGES}
            for role in ROLES
        },
        "family_count_by_role": {
            role: len(role_families[role]) for role in ROLES
        },
        "cross_role_family_names": sorted(cross_role_families),
        "missing_physical_role_gates": missing,
        "every_language_has_all_roles": not any(
            not counts[role][lang] for role in ROLES for lang in LANGUAGES
        ),
        "strict_source_family_isolation": not cross_role_families,
    }


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    try:
        permissions = rights.inspect(root)
        baseline = physical.inspect(root)
        proposed = split.inspect(root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise PhysicalSplitDenied("physical rights or source split cannot be verified") from exc
    for label, proof in (("permissions", permissions),
                         ("source", baseline), ("split", proposed)):
        raw = dict(proof)
        digest = raw.pop("manifest_sha256", None)
        need(digest == books.sha(books.canonical(raw)),
             f"{label} lineage checksum drift")
    need(permissions["physical_manifest_sha256"] == baseline["manifest_sha256"]
         and proposed["upstream_three_family_sha256"] == baseline["manifest_sha256"],
         "unbound source, rights or split identity")
    need(permissions["source_level_rights_verified"] is True
         and permissions["physical_privacy_executed"] is True
         and permissions["independent_ua_en_holdouts_verified"] is False
         and permissions["s3_s9_corpus_admitted"] is False
         and permissions["training_corpus_authorized"] is False
         and proposed["production_train_mixture_admitted"] is False
         and proposed["physical_s9_admitted"] is False
         and proposed["training_corpus_authorized"] is False
         and proposed["terminal_done"] is False,
         "candidate upstream wrongly promoted to corpus grant")
    audit = analyze_members(
        baseline["source_members"],
        proposed["physical_source_cluster_assignments"],
    )
    need(audit["physical_records"] == baseline["document_identity_count"] == 15
         and audit["physical_sources"] == len(permissions["source_ids"]) == 5
         and audit["physical_families"] == baseline["source_family_count"] == 3
         and audit["physical_bytes"] == baseline["total_physical_normalized_bytes"]
         and sum(audit["record_counts_by_role_language"]["train"].values())
         == proposed["train_record_count"]
         and sum(audit["record_counts_by_role_language"]["validation"].values())
         == proposed["validation_record_count"]
         and sum(audit["record_counts_by_role_language"]["test"].values())
         == proposed["test_record_count"],
         "physical split counts/source/rights drift")
    need(sorted(permissions["source_ids"]) ==
         sorted(proposed["physical_source_cluster_assignments"]),
         "source split includes unknown or unlicensed identity")
    for role in ROLES:
        role_families = sorted({
            row["source_family"] for row in baseline["source_members"]
            if proposed["physical_source_cluster_assignments"][row["source_id"]] == role
        })
        need(role_families == proposed["source_family_roles"][role],
             "source family role claims do not match physical members")
    # Both source-level candidate admission and the S15 5-source split are
    # non-production.  A green structural report CANNOT upgrade these receipts.
    gates = sorted(set(audit["missing_physical_role_gates"]) | {
        "NO_INDEPENDENT_PRODUCTION_S3_S9_ADMISSION",
        "NO_INDEPENDENT_REAL_EVALUATION_CUSTODY",
        "NO_APPROVED_CROSS_LANGUAGE_DECONTAMINATION",
        "NO_APPROVED_PRODUCTION_LANGUAGE_BALANCE",
    })
    need(bool(gates), "unverifiable empty blocking-gate inventory")
    core = {
        "schema_version": SCHEMA,
        "decision": "DENY_PRODUCTION_SPLITS_EXACT_PHYSICAL_GAPS",
        "source_permissions_sha256": permissions["manifest_sha256"],
        "source_physical_sha256": baseline["manifest_sha256"],
        "source_split_sha256": proposed["manifest_sha256"],
        **audit,
        "blocking_gates": gates,
        "rights_and_privacy_source_candidate_verified": True,
        "all_release_gates_passed": False,
        "s3_s9_corpus_admitted": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "plan9_binding_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "unsafe source split audit destination")
    result = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / OUTPUT
    blob = books.canonical(result)
    if path.exists() or path.is_symlink():
        need(path.is_file() and not path.is_symlink()
             and _read_destination(path) == blob,
             "immutable physical split audit changed")
    else:
        _atomic_write(destination, path, blob)
    need(_read_destination(path) == blob, "physical split readback mismatch")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = stage(args.root, args.out_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "decision": receipt["decision"],
        "manifest_sha256": receipt["manifest_sha256"],
        "blocking_gates": receipt["blocking_gates"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
