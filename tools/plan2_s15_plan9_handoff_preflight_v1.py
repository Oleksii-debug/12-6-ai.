"""S15 -> Plan9 data handoff preflight, evidence-bound and non-authorizing.

Consumes already verified physical source/rights/S10/S12/S13/S14/EVAL233/Plan3-4
receipts WITHOUT refitting, repacking, opening evaluation outcomes, or invoking
a model optimizer. Refuses to fabricate the final S3-S9 production corpus grant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-plan9-data-handoff-preflight.v1"
OUTPUT = "plan9-data-handoff-preflight.json"


class Plan9HandoffDenied(ValueError):
    """Real data lineage or checkpoint/training permissions were overstated."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise Plan9HandoffDenied(reason)


def assemble(
    rights: dict[str, Any],
    training: dict[str, Any],
    tokenizer: dict[str, Any],
    data: dict[str, Any],
    evaluation: dict[str, Any],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    """Use immutable parent identities, not copied raw text or arbitrary flags."""
    need(rights["source_count"] == 5
         and rights["canonical_source_family_count"] == 3
         and rights["source_level_license_training_permission_evidenced"] is True
         and rights["s3_s9_physical_admission"] is False
         and rights["production_release_authorized"] is False,
         "Plan9 preflight missing 5-source legal rights boundary")
    need(training["source_families_total"] == 3
         and training["train_document_count"] > 0
         and training["heldout_document_count"] > 0
         and training["heldout_plaintext_materialized"] is False
         and training["training_corpus_authorized"] is False
         and training["physical_s9_admitted"] is False,
         "Plan9 preflight missing S10 exact isolated physical train source")
    need(rights["three_family_source_sha256"] ==
             training["upstream_three_family_sha256"]
         == data["source_cohort_manifest_sha256"]
         == evaluation["physical_three_family_sha256"],
         "Plan9 source family lineage differs across rights/S10/S13/EVAL233")
    need(tokenizer["source_train_partition_sha256"] ==
             training["manifest_sha256"]
         and tokenizer["source_s10_split_sha256"] ==
             training["s10_split_manifest_sha256"]
         and tokenizer["actual_vocab_size"] == 32768
         and tokenizer["fitted_merge_count"] == 32508
         and tokenizer["heldout_payloads_fitted"] is False
         and tokenizer["production_train_source_admitted"] is False
         and tokenizer["production_release_authorized"] is False,
         "Plan9 candidate tokenizer lacks exact 32768 source-bound S12 evidence")
    need(data["source_train_partition_sha256"] == training["manifest_sha256"]
         and data["source_s10_split_sha256"] ==
             training["s10_split_manifest_sha256"]
         and data["frozen_s12_candidate_sha256"] == tokenizer["manifest_sha256"]
         and data["train_document_count"] == training["train_document_count"]
         and data["heldout_document_count"] == training["heldout_document_count"]
         and data["target_count"] > 0
         and data["block_count"] > 0
         and data["heldout_payloads_exposed"] is False
         and data["plan9_optimizer_permission"] is False
         and data["production_release_authorized"] is False,
         "Plan9 S13/S14 replay or original three-family training source drifted")
    need(compatibility["source_fit_manifest_sha256"] ==
             tokenizer["manifest_sha256"]
         and compatibility["candidate_vocab_size"] == 32768
         and compatibility["production_32k_model_spec_compatible"] is True
         and compatibility["checkpoint_weight_reuse_approved"] is False
         and compatibility["plan9_optimizer_handoff_granted"] is False
         and compatibility["production_backend_binding_granted"] is False,
         "Plan3/4 ModelSpec cannot approve checkpoint reuse or optimizer start")
    need(evaluation["real_final_test_custody_verified"] is True
         and evaluation["real_final_decontamination_clean"] is True
         and evaluation["real_final_test_outcomes_read"] is False
         and evaluation["selection_payload_scanned"] is False
         and evaluation["training_corpus_authorized"] is False
         and evaluation["production_release_authorized"] is False,
         "real final test evidence or custody violated Plan9 isolation")
    parents = {
        "five_source_rights_sha256": rights["manifest_sha256"],
        "physical_s10_train_partition_sha256": training["manifest_sha256"],
        "exact_32768_bpe_candidate_sha256": tokenizer["manifest_sha256"],
        "independent_s13_s14_replay_sha256":
            data["independent_s13_s14_readback_sha256"],
        "real_s13_s14_candidate_sha256": data["manifest_sha256"],
        "real_final_test_custody_sha256": evaluation["manifest_sha256"],
        "plan34_32k_compatibility_sha256": compatibility["manifest_sha256"],
    }
    need(all(type(x) is str and len(x) == 64
             and set(x) <= set("0123456789abcdef")
             for x in parents.values()),
         "noncanonical parent SHA in Plan9 handoff")
    need(len(set(parents.values())) == len(parents)
         and type(data["ordered_exposure_chain_sha256"]) is str
         and len(data["ordered_exposure_chain_sha256"]) == 64
         and set(data["ordered_exposure_chain_sha256"]) <=
             set("0123456789abcdef"),
         "Plan9 handoff parent alias or malformed exposure chain")
    core = {
        "schema_version": SCHEMA,
        "decision": "PLAN9_DATA_HANDOFF_PREFLIGHT_ONLY_NOT_PRODUCTION",
        "migration_contract": "migration-contract-baseline-v1",
        "parent_manifests": parents,
        "source_family_count": 3,
        "physical_training_document_count": training["train_document_count"],
        "final_test_document_count": evaluation["real_final_test_record_count"],
        "frozen_candidate_vocab_size": 32768,
        "physical_train_targets": data["target_count"],
        "ordered_exposure_chain_sha256": data["ordered_exposure_chain_sha256"],
        "validation_final_payloads_in_handoff": False,
        "external_model_weights_or_llm_used": False,
        "checkpoint_weight_reuse_allowed": False,
        "optimizer_effect_authorized": False,
        "production_plan9_bindable": False,
        "physical_s3_s9_release_admitted": False,
        "terminal_done": False,
        "blocking_requirements": [
            "S3_S9_CANONICAL_ACCEPTED_MULTI_FAMILY_CORPUS_RELEASE",
            "INDEPENDENT_EXACT_HEAD_TWO_CLEAN_BUILD_PASS",
            "PLAN9_EXPLICIT_PRODUCTION_DATA_ACCEPTANCE",
        ],
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage_from_receipts(
    rights: dict[str, Any], training: dict[str, Any],
    tokenizer: dict[str, Any], data: dict[str, Any],
    evaluation: dict[str, Any], compatibility: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink Plan9 handoff output")
    manifest = assemble(rights, training, tokenizer, data, evaluation,
                        compatibility)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(manifest)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable Plan9 handoff preflight changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "Plan9 preflight hash changed")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt-directory", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    files = {
        "rights": ("physical-five-source-rights", "five-physical-source-rights.json"),
        "training": ("physical-three-family-train", "three-family-train-partition.json"),
        "tokenizer": ("physical-real-32768-byte-bpe", "real-32k-byte-bpe-preproduction.json"),
        "data": ("physical-three-family-s13-s14", "three-family-physical-s13-s14-candidate.json"),
        "evaluation": ("physical-real-final-custody", "eval233-three-family-final-decontamination.json"),
        "compatibility": ("plan34-real-32768-compatibility", "plan34-tokenizer-compatibility-candidate.json"),
    }
    payloads = {}
    for key, (directory, name) in files.items():
        target = args.source_receipt_directory / directory / name
        need(target.is_file() and not target.is_symlink(),
             "required immutable source receipt missing")
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8", "strict"))
        need(raw == books.canonical(value), "noncanonical Plan9 parent receipt")
        payloads[key] = value
    proof = stage_from_receipts(
        *(payloads[k] for k in files),
        destination=args.out_dir,
    )
    print(json.dumps({
        "decision": proof["decision"],
        "manifest_sha256": proof["manifest_sha256"],
        "production_plan9_bindable": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
