"""Independently recheck physical S15 source rights and G06 privacy for S16.

This is a source-level gate. It rereads pinned bytes through the incumbent
S15/S1/S2/G06/DATA232 authorities and never promotes a source to S3--S9.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_five_source_rights_v1 as rights
from tools import plan2_s15_three_family_physical_v1 as physical
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s16-source-rights-privacy-candidate.v1"
OUTPUT = "source-rights-privacy-candidate.json"


class SourceAdmissionDenied(ValueError):
    """Physical source-level evidence is missing, altered or prematurely promoted."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise SourceAdmissionDenied(reason)


def inspect(root: Path) -> dict[str, Any]:
    """Bind exact physical source rights to fresh privacy/decontamination runs."""
    root = root.resolve(strict=True)
    try:
        rights_proof = rights.inspect(root)
        physical_proof = physical.inspect(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SourceAdmissionDenied("physical source rights/privacy evidence denied") from exc
    for name, proof in (("rights", rights_proof), ("physical", physical_proof)):
        core = dict(proof)
        digest = core.pop("manifest_sha256", None)
        _need(digest == books.sha(books.canonical(core)),
              f"{name} proof self-hash drift")
    _need(rights_proof["three_family_source_sha256"] ==
          physical_proof["manifest_sha256"], "source/rights/privacy lineage mismatch")
    _need(rights_proof["source_count"] == 5 and
          rights_proof["canonical_source_family_count"] == 3 and
          rights_proof["source_level_license_training_permission_evidenced"] is True and
          rights_proof["source_level_release_conditions_require_attribution"] is True and
          rights_proof["source_inventory_status"] == "candidate" and
          rights_proof["s3_s9_physical_admission"] is False and
          rights_proof["training_corpus_authorized"] is False and
          rights_proof["production_release_authorized"] is False,
          "source rights or corpus authority drift")
    _need(physical_proof["document_identity_count"] == 15 and
          physical_proof["g06_rejected_record_ids"] == [] and
          physical_proof["data232_excluded_record_count"] == 0 and
          physical_proof["data232_quarantined_source_family_count"] == 0 and
          physical_proof["physical_candidate_fixture_eval_clean"] is True and
          physical_proof["real_evaluation_custody_established"] is False and
          physical_proof["cross_language_semantic_dedup_established"] is False and
          physical_proof["physical_s3_s9_multi_family_admitted"] is False and
          physical_proof["training_corpus_authorized"] is False,
          "physical privacy, decontamination or holdout boundary drift")
    members = physical_proof["source_members"]
    _need(len(members) == 15 and len({m["record_id"] for m in members}) == 15,
          "physical member inventory incomplete")
    sources = sorted({m["source_id"] for m in members})
    _need(len(sources) == 5, "source identity count drift")
    core = {
        "schema_version": SCHEMA,
        "decision": "PHYSICAL_SOURCE_RIGHTS_PRIVACY_VERIFIED_CANDIDATE_ONLY",
        "rights_manifest_sha256": rights_proof["manifest_sha256"],
        "physical_manifest_sha256": physical_proof["manifest_sha256"],
        "source_inventory_sha256": rights_proof["source_inventory_sha256"],
        "source_rights_catalog_sha256": rights_proof["source_rights_catalog_sha256"],
        "g06_input_sha256": physical_proof["g06_input_sha256"],
        "g06_execution_identity_sha256": physical_proof["g06_execution_identity_sha256"],
        "data232_report_sha256": physical_proof["data232_report_sha256"],
        "source_ids": sources,
        "physical_record_count": len(members),
        "source_level_rights_verified": True,
        "physical_privacy_executed": True,
        "fixture_decontamination_executed": True,
        "independent_ua_en_holdouts_verified": False,
        "cross_language_semantic_dedup_verified": False,
        "s3_s9_corpus_admitted": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "plan9_binding_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    _need(not any(path.is_symlink() for path in (destination, *destination.parents)),
          "symlink source-admission destination")
    report = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    expected = books.canonical(report)
    if target.exists() or target.is_symlink():
        _need(target.is_file() and not target.is_symlink() and
              _read_destination(target) == expected,
              "immutable source-admission receipt changed")
    else:
        _atomic_write(destination, target, expected)
    _need(_read_destination(target) == expected, "source-admission readback changed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = stage(args.root, args.out_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"decision": report["decision"],
                      "manifest_sha256": report["manifest_sha256"],
                      "terminal_done": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
