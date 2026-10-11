"""S15 real EVAL-233 final-test decontamination of 3 physical source families.

Reuse frozen EVAL-233 resolver, EVAL-303 selection metadata and incumbent
DATA-232 matcher; never expose final-test outcomes or real final payload.
This is a candidate-only audit, not source S1-S9 admission or production fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_three_family_physical_v1 as physical
from tools import plan2_s15_three_family_split_probe_v1 as split_probe
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six.data import decontamination_authority_v2 as data232
from twelve_six.data import eval233_final_test_resolver_v1 as eval233

SCHEMA = "12-6.plan2-s15-eval233-real-final-decontamination-v1"
OUTPUT = "eval233-three-family-final-decontamination.json"
SELECTION = "configs/evaluation/eval303_selection_validation_composite_v1.json"
SELECTION_BLOB = "c2e5b27bac541e55be2c807ea65b0e2ed77b7019"
SELECTION_SHA = "7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd"


class RealFinalDenied(ValueError):
    """Missing independent eval custody, source lineage or decontamination."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise RealFinalDenied(reason)


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    source = physical.inspect(root)
    need(source["source_family_count"] == 3
         and source["document_identity_count"] == 15
         and source["real_evaluation_custody_established"] is False
         and source["production_release_authorized"] is False,
         "three-family source admission was falsely promoted")
    selection_raw = books.read_checked(root, SELECTION)
    need(books.git_blob(selection_raw) == SELECTION_BLOB,
         "EVAL-303 reserved selection authority changed")
    selection = json.loads(selection_raw.decode("utf-8", "strict"))
    contract = selection["usage_contract"]
    firewall = selection["final_test_firewall"]
    need(selection["selection_identity_sha256"] == SELECTION_SHA
         and selection["purpose"] == "selection-validation"
         and contract["selection_only"] is True
         and contract["may_fit_tokenizer"] is False
         and contract["may_train"] is False
         and contract["may_report_final_test"] is False
         and selection["rights_and_reservation"][
             "all_included_records_training_prohibited"] is True
         and firewall["outcomes_read_by_eval303"] is False
         and firewall["final_test_payload_read_by_eval303"] is False
         and firewall["data232_final_test_identity_sha256"] ==
             eval233.DATA232_FINAL_TEST_ID,
         "EVAL-303 independent selection/final rights drift")
    final_rows, authority, custody = eval233.resolve_eval233_final_test(root)
    need(len(final_rows) == 16
         and authority["identity_sha256"] == eval233.DATA232_FINAL_TEST_ID
         and custody["documents"] == 16
         and custody["final_test_outcomes_read"] is False
         and custody["selection_or_hyperparameter_use"] is False
         and custody["raw_text_persisted_in_evidence"] is False,
         "frozen real EVAL-233 final test not independently established")
    by_id = {record["record_id"]: record for record in source["source_members"]}
    physical_rows = split_probe.physical_rows(root, source)
    rejected = set(source["g06_rejected_record_ids"])
    train = [{
        "record_id": row["record_id"],
        "source_id": row["source_id"],
        "source_family": by_id[row["record_id"]]["source_family"],
        "lineage_family": by_id[row["record_id"]]["document_family"],
        "modality": by_id[row["record_id"]]["modality"],
        "text": row["text"],
    } for row in physical_rows if row["record_id"] not in rejected]
    need(bool(train), "all training source candidates are G06 quarantined")
    need({r["source_id"] for r in train}.isdisjoint(
         {r["source_id"] for r in final_rows})
         and {r["source_family"] for r in train}.isdisjoint(
         {r["source_family"] for r in final_rows}),
         "real EVAL-233 final-test source or family accidentally trained on")
    authorities = {
        "schema": "12-6.data232-reserved-authorities.v1",
        "authorities": [
            {"authority_id": "eval303-selection",
             "identity_sha256": SELECTION_SHA,
             "role": "selection_validation",
             "source_sha": SELECTION_BLOB},
            {"authority_id": "eval233-final-test",
             "identity_sha256": eval233.DATA232_FINAL_TEST_ID,
             "role": "final_test",
             "source_sha": eval233.EVAL233_HEAD},
        ],
    }
    data232.validate_authority_metadata(authorities)
    report = data232.build_report(
        train, final_rows,
        training_corpus_identity=source["source_members_sha256"],
        selection_validation_identity=SELECTION_SHA,
        final_test_identity=eval233.DATA232_FINAL_TEST_ID,
        authorities=authorities, quarantine_cross_source_families=True,
    )
    data232.verify_report(report)
    pass_without_exclusions = (
        not rejected
        and report["counts"]["excluded_training_records"] == 0
        and report["counts"]["quarantined_source_families"] == 0
    )
    core = {
        "schema_version": SCHEMA,
        "decision": "REAL_EVAL233_FINAL_CUSTODY_DECONTAMINATION_CANDIDATE_ONLY",
        "physical_three_family_sha256": source["manifest_sha256"],
        "selection_authority_git_blob": SELECTION_BLOB,
        "selection_identity_sha256": SELECTION_SHA,
        "selection_payload_scanned": False,
        "real_eval233_final_membership_identity_sha256":
            custody["source_final_set_identity_sha256"],
        "real_eval233_resolver_identity_sha256": custody["resolver_identity_sha256"],
        "data232_final_test_identity_sha256": eval233.DATA232_FINAL_TEST_ID,
        "physical_training_candidate_count": len(train),
        "g06_rejected_candidate_count": len(rejected),
        "real_final_test_record_count": len(final_rows),
        "real_final_test_custody_verified": True,
        "real_final_test_payload_read_for_decontamination_only": True,
        "real_final_test_outcomes_read": False,
        "data232_report_sha256": report["report_sha256"],
        "data232_status": report["status"],
        "excluded_physical_training_candidate_count":
            report["counts"]["excluded_training_records"],
        "quarantined_source_family_count":
            report["counts"]["quarantined_source_families"],
        "real_final_decontamination_clean": pass_without_exclusions,
        "physical_s3_s9_admitted": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(x.is_symlink() for x in (destination, *destination.parents)),
         "symlink final-custody destination")
    report = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable real final-custody receipt changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "real final-custody readback drift")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": result["decision"],
        "manifest_sha256": result["manifest_sha256"],
        "real_final_test_outcomes_read": False,
        "production_release_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
