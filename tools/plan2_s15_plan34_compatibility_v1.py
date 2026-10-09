"""S15 check actual S12 candidate against Plan3/4 tokenizer/ModelSpec baseline.

Reuses the canonical migration_compatibility contract, never projects 96 BPE
merges into a fake 32,768-vocabulary production artifact. It is not a
checkpoint/optimizer or Plan9 training-authorization endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_three_family_train_bpe_v1 as bpe
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six.tokenization import migration_compatibility as migration

SCHEMA = "12-6.plan2-s15-plan34-tokenizer-compatibility.v1"
OUTPUT = "plan34-tokenizer-compatibility-candidate.json"


class Plan34CompatibilityDenied(ValueError):
    """A real frozen candidate was overstated as production-compatible."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise Plan34CompatibilityDenied(reason)


def assess(fitted: dict[str, Any]) -> dict[str, Any]:
    need(fitted["decision"] ==
             "THREE_FAMILY_TRAIN_ONLY_BPE_CANDIDATE_NOT_PRODUCTION"
         and fitted["source_train_document_count"] > 0
         and fitted["heldout_payloads_fitted"] is False
         and fitted["production_target_vocab_frozen"] is False
         and fitted["training_corpus_authorized"] is False
         and fitted["tokenizer_fit_authorized"] is False
         and fitted["production_release_authorized"] is False
         and fitted["target_vocab_size"] == 32768
         and type(fitted["candidate_vocab_size"]) is int
         and fitted["candidate_vocab_size"] < fitted["target_vocab_size"],
         "fitted Plan2 tokenizer not a valid nonrelease S12 candidate")
    identity = fitted["tokenizer_identity"]
    need(identity["vocab_size"] == fitted["candidate_vocab_size"]
         and identity["normalization"] == "none"
         and identity["encoding"] == "utf-8",
         "physical tokenizer identity does not bind its candidate vocabulary")
    fresh_spec = {
        "vocab_size": fitted["candidate_vocab_size"],
        "tie_word_embeddings": True,
        "lm_head_bias": False,
    }
    decision = migration.assess_tokenizer_checkpoint_migration(
        target_tokenizer_identity=identity,
        target_model_spec=fresh_spec,
        reuse_checkpoint_weights=False,
    )
    need(decision["status"] == "FRESH_INITIALIZATION_TARGET_TOKENIZER_BOUND"
         and decision["checkpoint_weight_reuse_allowed_by_tokenizer_contract"]
             is False
         and decision["full_checkpoint_compatibility_proven"] is False,
         "candidate tokenizer silently reused model/checkpoint weights")
    production_spec = {
        **fresh_spec, "vocab_size": fitted["target_vocab_size"],
    }
    try:
        migration.assess_tokenizer_checkpoint_migration(
            target_tokenizer_identity=identity,
            target_model_spec=production_spec,
            reuse_checkpoint_weights=False,
        )
    except migration.TokenizerMigrationError as exc:
        need("vocab_size must equal" in str(exc),
             "production ModelSpec refusal was for an unrelated error")
    else:
        raise Plan34CompatibilityDenied(
            "96-merge candidate spuriously accepted as 32K production ModelSpec"
        )
    core = {
        "schema_version": SCHEMA,
        "decision": "PLAN34_FRESH_CANDIDATE_COMPATIBLE_PRODUCTION_32K_REFUSED",
        "source_fit_manifest_sha256": fitted["manifest_sha256"],
        "candidate_model_spec": fresh_spec,
        "target_production_model_spec": production_spec,
        "candidate_migration_decision_sha256":
            books.sha(books.canonical(decision)),
        "candidate_vocab_size": fitted["candidate_vocab_size"],
        "required_production_vocab_size": fitted["target_vocab_size"],
        "candidate_fresh_random_initialization_migration_compatible": True,
        "checkpoint_weight_reuse_approved": False,
        "production_32k_model_spec_compatible": False,
        "production_backend_binding_granted": False,
        "plan9_optimizer_handoff_granted": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def inspect(root: Path) -> dict[str, Any]:
    return assess(bpe.inspect(root.resolve(strict=True)))


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(x.is_symlink() for x in (destination, *destination.parents)),
         "symlink Plan3/4 compatibility output")
    return stage_from_fit(bpe.inspect(root.resolve(strict=True)), destination)


def stage_from_fit(fitted: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Publish a previously recomputed exact S12 fit without fitting twice."""
    need(not any(x.is_symlink() for x in (destination, *destination.parents)),
         "symlink Plan3/4 compatibility output")
    report = assess(fitted)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable Plan3/4 compatibility result changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw,
         "Plan3/4 migration contract readback drifted")
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
        "production_backend_binding_granted": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
