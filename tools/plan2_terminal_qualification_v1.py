"""Plan 2 Section 15: reproducible LOCAL_FREE audit, never a production release.

Reuses the accepted S3/S9/S10/S12/S13/S14 authorities. Both clean builds
verify actual candidate bytes and synthetic downstream fixtures independently.
In particular this tool cannot convert source_candidate rights into training
permission, or an isolated synthetic tokenizer into production authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from tools import plan2_cluster_split_v1 as split
from tools import plan2_corpus_mixture_v1 as mixture
from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_exposure_ledger_v1 as exposure
from tools import plan2_physical_materialization_v1 as physical
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_book_split_probe_v1 as book_split_probe
from tools import plan2_s15_physical_heldout_decontam_v1 as physical_heldout
from tools import plan2_s15_physical_tokenizer_candidate_v1 as real_tokenizer
from tools import plan2_s15_physical_packing_candidate_v1 as real_packing
from tools import plan2_s15_physical_exposure_candidate_v1 as real_exposure
from tools import plan2_s15_physical_readback_v1 as real_readback
from tools import plan2_s15_d03_physical_sources_v1 as d03_sources
from tools import plan2_s15_d03_normalized_materialization_v1 as d03_normalized
from tools import plan2_s15_three_family_physical_v1 as combined_sources
from tools import plan2_s15_three_family_split_probe_v1 as combined_split
from tools import plan2_s15_three_family_mixture_probe_v1 as three_family_mixture
from tools import plan2_s15_five_source_rights_v1 as five_source_rights
from tools import plan2_s15_three_family_train_materialization_v1 as train_partition
from tools import plan2_s15_three_family_train_bpe_v1 as three_family_bpe
from tools import plan2_s15_real_eval233_decontamination_v1 as real_final
from tools import plan2_tokenizer_fit_freeze_v1 as fit

SCHEMA = "12-6.plan2-final-audit-local-free.v1"
OUTPUT = "plan2-section15-audit.json"


class QualificationDenied(ValueError):
    """Unqualified, non-reproducible or unsafe release assertion."""


def _require(value: bool, reason: str) -> None:
    if not value:
        raise QualificationDenied(reason)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _inventory(base: Path) -> dict[str, str]:
    """Exact byte inventory; reject symlinks even within aborted stage trees."""
    members: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        _require(not path.is_symlink(), "symlink in qualification artifact tree")
        _require(path.is_file() or path.is_dir(), "unexpected artifact member")
        if path.is_file():
            relative = path.relative_to(base).as_posix()
            _require(relative not in members, "duplicate artifact member")
            members[relative] = _digest(path.read_bytes())
    _require(bool(members), "empty qualification artifact tree")
    return members


def _read_gate(path: Path) -> dict[str, Any]:
    """Verify a staged authority, not just an untrusted 64-character hash."""
    _require(path.is_file() and not path.is_symlink(), "physical gate member missing")
    raw = physical._read_destination(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise QualificationDenied("invalid physical gate JSON") from exc
    _require(type(value) is dict and raw == _canonical(value),
             "noncanonical physical gate JSON")
    digest = value.get("manifest_sha256")
    unsigned = {k: v for k, v in value.items() if k != "manifest_sha256"}
    _require(type(digest) is str and digest == _digest(_canonical(unsigned)),
             "physical gate manifest hash mismatch")
    _require(value.get("training_corpus_authorized") is False,
             "physical gate incorrectly authorizes training")
    for capability in ("tokenizer_fit_authorized", "evaluation_authorized",
                       "generated_auto_reentry_authorized", "raw_text_emitted",
                       "real_final_test_material_accessed",
                       "production_release_authorized",
                       "physical_tokenizer_fit_authorized"):
        if capability in value:
            _require(value[capability] is False,
                     "physical gate incorrectly authorizes " + capability)
    return value


def _audit_physical_gates(candidate: Path, mixture_hash: str) -> dict[str, str]:
    """Bind S5-S8 readbacks into the S9 physical candidate, fail closed."""
    firewall_dir = candidate / "firewall"
    near_dir = firewall_dir / "near"
    exact_dir = near_dir / "exact"
    privacy_dir = exact_dir / "privacy"
    stages = {
        "privacy": _read_gate(privacy_dir / "privacy-manifest.json"),
        "exact_dedup": _read_gate(exact_dir / "exact-dedup-manifest.json"),
        "near_dedup": _read_gate(near_dir / "near-dedup-manifest.json"),
        "reserved_eval": _read_gate(
            firewall_dir / "reserved-eval-firewall-manifest.json"
        ),
    }
    rights = stages["privacy"]["manifest_sha256"]
    # A single matching S5 receipt cannot certify other S6 sources.
    # Until individually staged S5 proofs exist for every source, refuse
    # multi-source evidence rather than silently accepting partial coverage.
    exact_sources = stages["exact_dedup"].get("sources")
    # The physical S5 authority is a single source: a copied self-hash
    # cannot attest a second S6 source or a different normalized member.
    # Real multi-source release needs independent S5 proofs for every source.
    _require(type(exact_sources) is list and len(exact_sources) == 1
             and type(exact_sources[0]) is dict
             and exact_sources[0].get("source_id") ==
             stages["privacy"].get("source_id")
             and exact_sources[0].get("normalization_manifest_sha256") ==
             stages["privacy"].get("normalization_manifest_sha256")
             and exact_sources[0].get("privacy_manifest_sha256") == rights,
             "S5-S6 privacy provenance disconnected")
    _require(stages["near_dedup"]["upstream_exact_manifest_sha256"] ==
             stages["exact_dedup"]["manifest_sha256"],
             "S6-S7 exact dedup lineage disconnected")
    _require(stages["reserved_eval"]["physical_s7_manifest_sha256"] ==
             stages["near_dedup"]["manifest_sha256"],
             "S7-S8 decontamination lineage disconnected")
    mixture_raw = physical._read_destination(
        candidate / "corpus-mixture-manifest.json"
    )
    try:
        mixture_doc = json.loads(mixture_raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise QualificationDenied("invalid physical mixture JSON") from exc
    _require(type(mixture_doc) is dict
             and mixture_raw == _canonical(mixture_doc),
             "noncanonical physical mixture JSON")
    mixture_core = {
        k: v for k, v in mixture_doc.items() if k != "dataset_candidate_sha256"
    }
    _require(
        mixture_doc.get("dataset_candidate_sha256") == _digest(_canonical(mixture_core))
        and mixture_doc["dataset_candidate_sha256"] == mixture_hash
        and mixture_doc["training_corpus_authorized"] is False
        and mixture_doc["physical_s8_manifest_sha256"] ==
        stages["reserved_eval"]["manifest_sha256"],
        "S8-S9 mixture lineage disconnected",
    )
    # A re-signed S9 candidate receipt cannot claim downstream release,
    # tokenizer-fit, evaluation or raw-payload access while the cohort is
    # explicitly component-only. Hash validity is not an authority grant.
    for capability in ("production_release_authorized",
                       "physical_tokenizer_fit_authorized",
                       "tokenizer_fit_authorized",
                       "evaluation_authorized",
                       "generated_auto_reentry_authorized",
                       "raw_text_emitted",
                       "real_final_test_material_accessed"):
        if capability in mixture_doc:
            _require(mixture_doc[capability] is False,
                     "S9 mixture incorrectly authorizes " + capability)
    return {key: value["manifest_sha256"] for key, value in stages.items()}


def _rebuild(root: Path, destination: Path) -> dict[str, Any]:
    """Reuse all extant component authorities, without creating a second pipeline."""
    d03_ua = d03_sources.stage(root, destination / "physical-d03-ua")
    _require(d03_ua["physical_family_count"] == 2
             and d03_ua["physical_record_count"] == 12
             and d03_ua["training_corpus_authorized"] is False
             and d03_ua["tokenizer_fit_authorized"] is False
             and d03_ua["production_release_authorized"] is False
             and d03_ua["terminal_done"] is False,
             "D03 historical source snapshot cannot authorize Plan2 release")
    normalized_ua = d03_normalized.stage(
        root, destination / "physical-d03-normalized")
    _require(normalized_ua["d03_source_manifest_sha256"] == d03_ua["manifest_sha256"]
             and normalized_ua["normalized_member_count"] == 12
             and normalized_ua["normalized_total_bytes"] == 48675
             and normalized_ua["physical_s3_s9_admitted"] is False
             and normalized_ua["production_release_authorized"] is False,
             "physical normalized D03 sources cannot promote Plan2 corpus")
    books_cohort = books.stage(root, destination / "public-domain-books")
    _require(books_cohort["training_corpus_authorized"] is False
             and books_cohort["production_release_authorized"] is False,
             "book source was promoted without full S5-S14 clearance")
    combined = combined_sources.stage(root, destination / "physical-three-families")
    _require(combined["source_family_count"] == 3
             and combined["document_identity_count"] == 15
             and combined["total_physical_normalized_bytes"] == 1_314_156
             and combined["gutenberg_source_sha256"] == books_cohort["manifest_sha256"]
             and combined["d03_source_sha256"] == d03_ua["manifest_sha256"]
             and combined["physical_s3_s9_multi_family_admitted"] is False
             and combined["training_corpus_authorized"] is False
             and combined["production_release_authorized"] is False
             and combined["real_evaluation_custody_established"] is False,
             "combined real physical family evidence cannot grant training")
    five_rights = five_source_rights.stage(
        root, destination / "physical-five-source-rights")
    _require(five_rights["three_family_source_sha256"] == combined["manifest_sha256"]
             and five_rights["source_count"] == 5
             and five_rights["canonical_source_family_count"] == 3
             and five_rights["source_level_license_training_permission_evidenced"] is True
             and five_rights["s3_s9_physical_admission"] is False
             and five_rights["tokenizer_fit_authorized"] is False
             and five_rights["production_release_authorized"] is False,
             "source license catalog must not grant Plan2 training admission")
    family_mixture = three_family_mixture.stage(
        root, destination / "physical-three-family-mixture")
    _require(family_mixture["source_cohort_manifest_sha256"] ==
             combined["manifest_sha256"]
             and family_mixture["source_family_count"] == 3
             and family_mixture["physical_source_count"] == 5
             and family_mixture["selected_physical_document_count"] == 15
             and family_mixture["physical_s3_s9_admitted"] is False
             and family_mixture["real_s8_authority_issued"] is False
             and family_mixture["tokenizer_fit_authorized"] is False
             and family_mixture["training_corpus_authorized"] is False,
             "S9 real three-family mixture incorrectly grants data authority")
    combined_partitions = combined_split.stage(
        root, destination / "physical-three-family-split")
    _require(combined_partitions["upstream_three_family_sha256"] ==
             combined["manifest_sha256"]
             and combined_partitions["canonical_source_family_count"] == 3
             and combined_partitions["physical_document_count"] == 15
             and combined_partitions["whole_source_cluster_count"] == 5
             and combined_partitions["cluster_leakage_count"] == 0
             and combined_partitions["physical_s9_admitted"] is False
             and combined_partitions["training_corpus_authorized"] is False
             and combined_partitions["production_release_authorized"] is False,
             "source-family split mechanics incorrectly promoted candidate")
    train_only = train_partition.stage(
        root, destination / "physical-three-family-train")
    _require(train_only["upstream_three_family_sha256"] == combined["manifest_sha256"]
             and train_only["s10_split_probe_sha256"] == combined_partitions["manifest_sha256"]
             and train_only["s10_split_manifest_sha256"] ==
                 combined_partitions["s10_split_manifest_sha256"]
             and train_only["train_document_count"] ==
                 combined_partitions["train_record_count"]
             and train_only["heldout_document_count"] ==
                 combined_partitions["validation_record_count"] +
                 combined_partitions["test_record_count"]
             and train_only["heldout_plaintext_materialized"] is False
             and train_only["training_corpus_authorized"] is False
             and train_only["tokenizer_fit_authorized"] is False
             and train_only["production_release_authorized"] is False,
             "physical S10 train bytes or hash-only holdout custody drifted")
    bpe_candidate = three_family_bpe.stage(
        root, destination / "physical-three-family-bpe")
    _require(bpe_candidate["source_train_partition_sha256"] ==
             train_only["manifest_sha256"]
             and bpe_candidate["source_s10_split_manifest_sha256"] ==
             train_only["s10_split_manifest_sha256"]
             and bpe_candidate["source_train_document_count"] ==
             train_only["train_document_count"]
             and bpe_candidate["heldout_payloads_fitted"] is False
             and bpe_candidate["production_target_vocab_frozen"] is False
             and bpe_candidate["tokenizer_fit_authorized"] is False
             and bpe_candidate["production_release_authorized"] is False,
             "real train-only tokenizer candidate or holdout boundary changed")
    real_eval = real_final.stage(root, destination / "physical-real-final-custody")
    _require(real_eval["physical_three_family_sha256"] ==
             combined["manifest_sha256"]
             and real_eval["real_final_test_record_count"] == 16
             and real_eval["real_final_test_custody_verified"] is True
             and real_eval["real_final_test_outcomes_read"] is False
             and real_eval["selection_payload_scanned"] is False
             and real_eval["training_corpus_authorized"] is False
             and real_eval["production_release_authorized"] is False,
             "real EVAL233 final-test decontamination cannot authorize training")
    probe = book_split_probe.stage(root, destination / "physical-book-split-probe")
    _require(probe["production_release_authorized"] is False
             and probe["physical_s9_admitted"] is False
             and probe["canonical_source_families"] == 1
             and probe["physical_document_clusters"] == 3,
             "physical-book mechanics probe incorrectly promoted corpus")
    heldout = physical_heldout.stage(root, destination / "physical-heldout")
    _require(heldout["terminal_done"] is False
             and heldout["physical_s9_admitted"] is False
             and heldout["production_release_authorized"] is False,
             "physical heldout audit cannot promote unreleased data")
    real_fit = real_tokenizer.stage(root, destination / "physical-tokenizer-candidate")
    _require(real_fit["production_release_authorized"] is False
             and real_fit["tokenizer_fit_authorized"] is False
             and real_fit["terminal_done"] is False,
             "physical tokenizer candidate cannot grant production training")
    real_shards = real_packing.stage(root, destination / "physical-shards-candidate")
    _require(real_shards["terminal_done"] is False
             and real_shards["production_release_authorized"] is False
             and real_shards["physical_s9_admitted"] is False
             and real_shards["tokenizer_manifest_sha256"] == real_fit["manifest_sha256"]
             and real_shards["cluster_split_sha256"] ==
             real_fit["physical_split_manifest_sha256"],
             "physical shard/tokenizer/S10 split identity mismatch")
    real_targets = real_exposure.stage(root, destination / "physical-exposures-candidate")
    _require(real_targets["terminal_done"] is False
             and real_targets["production_release_authorized"] is False
             and real_targets["training_corpus_authorized"] is False
             and real_targets["packing_manifest_sha256"] == real_shards["manifest_sha256"]
             and real_targets["cluster_split_sha256"] == real_shards["cluster_split_sha256"]
             and real_targets["tokenizer_manifest_sha256"] == real_fit["manifest_sha256"]
             and real_targets["target_count"] == real_shards["target_count"],
             "real physical packing-to-exposure identity mismatch")
    witness = real_readback.verify(
        destination / "physical-shards-candidate",
        destination / "physical-exposures-candidate",
        expected_packed_sha256=real_shards["manifest_sha256"],
        expected_exposure_sha256=real_targets["manifest_sha256"],
    )
    _require(witness["terminal_done"] is False
             and witness["production_release_authorized"] is False
             and witness["readback_target_count"] == real_targets["target_count"]
             and witness["last_chain_sha256"] == real_targets["chain_head_sha256"],
             "independent physical S13/S14 publication readback mismatch")
    cohort = physical.stage_candidate_cohort(root, destination / "source")
    current = mixture.stage_mixture(root, destination / "physical-candidate")
    _require(cohort["source_level_candidate_only"] is True
             and cohort["training_corpus_authorized"] is False,
             "source candidate unexpectedly promoted")
    _require(current["training_corpus_authorized"] is False,
             "physical mixture unexpectedly grants training")

    # A changed source-family count must refuse before reading stale S9 hashes.
    # Never mislabel an unrelated integrity error as a known corpus limitation.
    families = current["contributions"]["family"]
    _require(type(families) is dict and len(families) == 1,
             "physical source-family count changed; release requires requalification")

    gates = _audit_physical_gates(
        destination / "physical-candidate", current["dataset_candidate_sha256"]
    )
    try:
        split.stage_candidate(root, destination / "physical-split")
    except split.Plan2SplitError as exc:
        causes: list[str] = []
        reason: BaseException | None = exc
        while reason is not None:
            causes.append(str(reason))
            reason = reason.__cause__
        _require(
            "insufficient independent document families" in causes
            or "three-way cluster-safe split has empty partition" in causes,
            "unexpected physical split failure; not a family-count refusal",
        )
        physical_split = "DENIED_SINGLE_SOURCE_FAMILY"
    else:
        raise QualificationDenied(
            "single-family physical candidate unexpectedly passed safe split"
        )

    tokenizer = fit.stage(root, destination / "synthetic-tokenizer")
    packed = packing.stage(root, destination / "synthetic-shards")
    ledger = exposure.stage(root, destination / "synthetic-shards",
                            destination / "synthetic-ledger")
    _require(tokenizer["production_release_authorized"] is False
             and tokenizer["physical_tokenizer_fit_authorized"] is False
             and packed["production_release_authorized"] is False
             and ledger["production_release_authorized"] is False
             and ledger["training_corpus_authorized"] is False,
             "synthetic downstream authority escalation")
    _require(packed["manifest_sha256"] == ledger["packing_manifest_sha256"],
             "packing-to-exposure identity mismatch")
    _require(tokenizer["manifest_sha256"] == packed["tokenizer_manifest_sha256"],
             "tokenizer-to-packing identity mismatch")
    _require(packed["target_count"] == ledger["target_count"],
             "target exposure mismatch")

    return {
        "real_three_family_train_only_bpe_manifest_sha256":
            bpe_candidate["manifest_sha256"],
        "real_three_family_bpe_candidate_vocab_size":
            bpe_candidate["candidate_vocab_size"],
        "real_three_family_bpe_target_vocab_size":
            bpe_candidate["target_vocab_size"],
        "real_three_family_train_partition_manifest_sha256":
            train_only["manifest_sha256"],
        "real_three_family_train_only_document_count":
            train_only["train_document_count"],
        "real_three_family_train_only_physical_bytes":
            train_only["physical_train_bytes"],
        "real_three_family_heldout_document_count":
            train_only["heldout_document_count"],
        "real_five_source_s1_s2_rights_manifest_sha256":
            five_rights["manifest_sha256"],
        "real_five_source_s1_inventory_sha256":
            five_rights["source_inventory_sha256"],
        "real_five_source_s2_catalog_sha256":
            five_rights["source_rights_catalog_sha256"],
        "real_three_family_s9_mixture_mechanics_sha256":
            family_mixture["manifest_sha256"],
        "real_three_family_s9_candidate_record_count":
            family_mixture["selected_physical_document_count"],
        "real_three_family_source_split_probe_sha256":
            combined_partitions["manifest_sha256"],
        "real_three_family_source_cluster_count":
            combined_partitions["whole_source_cluster_count"],
        "real_three_family_source_split_leakage_count":
            combined_partitions["cluster_leakage_count"],
        "real_eval233_final_custody_audit_sha256":
            real_eval["manifest_sha256"],
        "real_eval233_final_record_count": real_eval["real_final_test_record_count"],
        "real_eval233_final_outcomes_read": real_eval["real_final_test_outcomes_read"],
        "real_eval233_decontamination_clean":
            real_eval["real_final_decontamination_clean"],
        "real_three_family_candidate_sha256": combined["manifest_sha256"],
        "real_three_family_document_count": combined["document_identity_count"],
        "real_three_family_total_bytes": combined["total_physical_normalized_bytes"],
        "real_three_family_fixture_eval_clean":
            combined["physical_candidate_fixture_eval_clean"],
        "d03_real_ua_normalized_physical_sha256":
            normalized_ua["manifest_sha256"],
        "d03_real_ua_source_candidate_sha256": d03_ua["manifest_sha256"],
        "d03_real_ua_source_candidate_families": d03_ua["physical_family_count"],
        "d03_real_ua_source_candidate_members": d03_ua["physical_record_count"],
        "d03_real_ua_source_candidate_bytes": d03_ua["normalized_source_bytes"],
        "real_multifamily_source_candidates_not_s3_s9_admitted":
            d03_ua["physical_family_count"] + books_cohort["physical_source_families"],
        "real_books_manifest_sha256": books_cohort["manifest_sha256"],
        "real_books_distinct_document_families":
            books_cohort["physical_document_families"],
        "real_books_canonical_source_families":
            books_cohort["physical_source_families"],
        "real_books_snapshot_bytes": books_cohort["physical_source_bytes"],
        "real_books_s10_mechanics_probe_sha256": probe["manifest_sha256"],
        "real_books_s10_probe_decision": probe["decision"],
        "real_books_s10_probe_document_clusters":
            probe["physical_document_clusters"],
        "real_books_physical_heldout_manifest_sha256": heldout["manifest_sha256"],
        "real_books_physical_heldout_clean":
            heldout["physical_heldout_decontamination_clean"],
        "physical_real_text_tokenizer_candidate_sha256": real_fit["manifest_sha256"],
        "physical_real_text_tokenizer_training_bytes":
            real_fit["train_document_bytes"],
        "physical_real_text_shard_manifest_sha256": real_shards["manifest_sha256"],
        "physical_real_text_packed_target_count": real_shards["target_count"],
        "physical_real_text_exposure_manifest_sha256":
            real_targets["manifest_sha256"],
        "physical_real_text_exposure_chain_head_sha256":
            real_targets["chain_head_sha256"],
        "physical_s13_s14_independent_readback_sha256":
            witness["readback_sha256"],
        "physical_source_manifest_sha256": cohort["manifest_sha256"],
        "physical_mixture_manifest_sha256": current["dataset_candidate_sha256"],
        "physical_gate_manifest_sha256": gates,
        "physical_split": physical_split,
        "physical_source_family_count": len(families),
        "synthetic_tokenizer_manifest_sha256": tokenizer["manifest_sha256"],
        "synthetic_packing_manifest_sha256": packed["manifest_sha256"],
        "synthetic_exposure_manifest_sha256": ledger["manifest_sha256"],
        "synthetic_target_count": ledger["target_count"],
        "member_sha256": _inventory(destination),
    }


def audit(root: Path) -> dict[str, Any]:
    """Compare two independently materialized complete LOCAL_FREE artifact trees."""
    root = root.resolve(strict=True)
    _require(root.is_dir(), "missing repository root")
    with (
        tempfile.TemporaryDirectory(prefix="plan2-s15-first-") as a,
        tempfile.TemporaryDirectory(prefix="plan2-s15-clean-") as b,
    ):
        first = _rebuild(root, Path(a))
        second = _rebuild(root, Path(b))
        _require(first == second, "clean rebuild hash or inventory mismatch")
    core = {
        "schema_version": SCHEMA,
        "decision": "COMPONENT_AUDIT_ONLY_NOT_TERMINAL",
        "plan": 2,
        "section": 15,
        "reproducible_clean_builds": 2,
        "local_free": True,
        "paid_compute_used": False,
        "physical_corpus_training_authorized": False,
        "real_production_tokenizer_present": False,
        "real_production_shards_present": False,
        "production_release_authorized": False,
        "terminal_done": False,
        "blocking_gates": [
            "training_rights_require_independent_authorization",
            "book_snapshots_require_s1_s9_multifamily_rights_privacy_eval_admission",
            "production_tokenizer_fit_and_compatibility_not_yet_established",
            "production_packing_shards_exposure_not_yet_materialized",
            "plan9_production_handoff_missing",
        ],
        "evidence": first,
    }
    return {**core, "audit_sha256": _digest(_canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    _require(not any(p.is_symlink() for p in (destination, *destination.parents)),
             "symlink output path")
    report = audit(root)
    destination.mkdir(parents=True, exist_ok=True)
    _require(destination.is_dir() and not destination.is_symlink(),
             "invalid output directory")
    target = destination / OUTPUT
    raw = _canonical(report)
    if target.exists() or target.is_symlink():
        _require(not target.is_symlink() and target.is_file()
                 and target.read_bytes() == raw,
                 "immutable audit report changed")
    else:
        physical._atomic_write(destination, target, raw)
    _require(target.read_bytes() == raw, "audit readback mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S15 non-release audit")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.root, args.out_dir)
    print(json.dumps({"decision": report["decision"],
                      "audit_sha256": report["audit_sha256"],
                      "production_release_authorized": False,
                      "terminal_done": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
