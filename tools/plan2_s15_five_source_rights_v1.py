"""S15: reuse Plan2 S1/S2 for three lawful physical source families.

Five source identities: three separate Gutenberg books, one PHP/UK source and
one Rust Book/UK source, still ONLY three canonical source-family credits.
License grants are source-level permissions, not S3-S9 corpus admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_d03_physical_sources_v1 as d03
from tools import plan2_s15_three_family_physical_v1 as combined
from tools import plan2_source_admissibility_v1 as s2
from tools import plan2_source_inventory_v1 as s1
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-five-physical-source-rights.v1"
OUTPUT = "five-physical-source-rights.json"
BOOK_RIGHTS = "data/external/rights-evidence/plan2-books/public-domain-original-works-v1.txt"


class RightsBoundaryDenied(ValueError):
    """Physical rights evidence, registered source identity or release drift."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise RightsBoundaryDenied(reason)


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    english = books.inspect(root)
    ua = d03.inspect(root)
    physical = combined.inspect(root)
    need(english["physical_source_families"] == 1
         and english["physical_document_families"] == 3
         and ua["physical_family_count"] == 2
         and ua["physical_record_count"] == 12
         and physical["source_family_count"] == 3
         and physical["document_identity_count"] == 15
         and physical["training_corpus_authorized"] is False
         and physical["physical_s3_s9_multi_family_admitted"] is False,
         "invalid physical source cohort or pre-authorized release")
    record_groups: dict[str, list[dict[str, Any]]] = {}
    for row in ua["records"]:
        record_groups.setdefault(row["source_id"], []).append(row)
    need(len(record_groups) == 2
         and sorted(len(v) for v in record_groups.values()) == [2, 10],
         "D03 physical source/member authority disconnected")
    d03_config = json.loads(books.read_checked(root, d03.CONFIG))
    rights_by_family = {
        "php.manual.documentation": {
            "license_id": "CC-BY-3.0-or-later",
            "terms_ref": "https://creativecommons.org/licenses/by/3.0/",
            "evidence_path": d03.PREFIX + "licenses/php/license.xml",
        },
        "rust-book.documentation.uk-translation": {
            "license_id": "MIT",
            "terms_ref": "https://opensource.org/licenses/MIT",
            "evidence_path": d03.PREFIX + "licenses/rust/LICENSE-MIT",
        },
    }
    revision_by_family = {row["id"]: row["revision"]
                          for row in d03_config["source_families"]}
    rights_bytes: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    grants: list[dict[str, Any]] = []

    def bind_rights(sid: str, license_id: str, terms_ref: str,
                    evidence_path: str, basis: str,
                    permission_ref: str) -> None:
        raw = books.read_checked(root, evidence_path)
        need(len(raw) > 100 and raw.strip(), "source rights evidence missing")
        rights_bytes[evidence_path] = raw
        grants.append({
            "source_id": sid, "rights_class": "public_licensed",
            "license_id": license_id, "terms_ref": terms_ref,
            "permission_ref": permission_ref,
            "legal_basis": basis,
            "allowed_uses": ["source_candidate", "training", "release"],
            "evidence_path": evidence_path,
            "evidence_sha256": books.sha(raw), "revoked": False,
        })

    for item in english["books"]:
        sid = item["source_id"]
        need(books.sha(books.read_checked(root, item["snapshot_path"])) ==
             item["snapshot_sha256"], "Gutenberg original source drift")
        entries.append({
            "source_id": sid, "source_family": books.GUTENBERG_FAMILY,
            "location": "repo://" + item["snapshot_path"],
            "acquisition_method": "pinned_repository", "language": "en",
            "modality": "text", "update_cadence": "pinned", "status": "candidate",
            "provenance": {
                "authority_path": "configs/data/plan2_public_domain_books_v1.json",
                "upstream_revision": item["source_revision"],
                "content_sha256": item["snapshot_sha256"],
            },
        })
        bind_rights(
            sid, "Public-Domain", "https://www.gutenberg.org/policy/license",
            BOOK_RIGHTS, "public_domain",
            "repo://configs/data/plan2_public_domain_books_v1.json",
        )
    for sid, group in sorted(record_groups.items()):
        family = group[0]["source_family"]
        need(all(x["source_family"] == family for x in group)
             and family in rights_by_family, "D03 family source ID reused")
        family_identity = books.sha(books.canonical(sorted([
            {"member": x["source_path"], "sha256": x["normalized_sha256"]}
            for x in group
        ], key=lambda x: x["member"])))
        entries.append({
            "source_id": sid, "source_family": family,
            "location": "repo://" + d03.CONFIG,
            "acquisition_method": "pinned_repository", "language": "uk",
            "modality": "text", "update_cadence": "pinned", "status": "candidate",
            "provenance": {
                "authority_path": d03.CONFIG,
                "upstream_revision": revision_by_family[family],
                "content_sha256": family_identity,
            },
        })
        rights = rights_by_family[family]
        bind_rights(
            sid, rights["license_id"], rights["terms_ref"],
            rights["evidence_path"], "license_grant",
            "repo://" + d03.CONFIG,
        )
    inventory = s1.build_inventory("plan2-s15-five-real-sources-v1", entries)
    s1.verify_inventory(inventory)
    rights = s2.build_catalog(inventory, grants, rights_bytes)
    s2.verify_catalog(inventory, rights, rights_bytes)
    need(len(inventory["sources"]) == len(rights["grants"]) == 5
         and inventory["status_counts"]["candidate"] == 5
         and inventory["status_counts"]["accepted"] == 0
         and inventory["training_authorized"] is False
         and rights["training_corpus_authorized"] is False
         and {x["source_family"] for x in inventory["sources"]} ==
             set(physical["families"])
         and all({"training", "release"} <= set(x["allowed_uses"])
                 and x["revoked"] is False for x in rights["grants"]),
         "source-level license receipts changed or admitted before S3-S9")
    core = {
        "schema_version": SCHEMA,
        "decision": "SOURCE_RIGHTS_BOUND_FIVE_PHYSICAL_SOURCES_NO_CORPUS_ADMISSION",
        "three_family_source_sha256": physical["manifest_sha256"],
        "source_inventory_sha256": inventory["inventory_sha256"],
        "source_rights_catalog_sha256": rights["catalog_sha256"],
        "source_count": 5,
        "canonical_source_family_count": 3,
        "source_level_license_training_permission_evidenced": True,
        "source_level_release_conditions_require_attribution": True,
        "source_inventory_status": "candidate",
        "s3_s9_physical_admission": False,
        "real_final_test_custody_automatically_admitted": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink rights output")
    report = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable multi-source rights receipt changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "rights readback mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": report["decision"],
        "manifest_sha256": report["manifest_sha256"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
