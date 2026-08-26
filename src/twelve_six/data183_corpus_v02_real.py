"""DATA-183 Corpus V0.2 real-data release candidate.

This module deliberately composes DATA-110 rather than replacing its corpus path.
It adds the release evidence that DATA-183 requires: canonical origin classes,
normalized train/validation overlap proof, exact one-pass optimization-target token
supply by source/origin/stratum, no-duplication assertions, and an actual Trainer
streaming smoke through the retained shards.

The current DATA-110 external intake admits real UK/EN only.  Consequently a
successful DATA-183 build is an honest UK/EN-real + project-code candidate and
MUST remain blocked from a fully representative V0.2 claim until independently
admitted external code exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from twelve_six import data110_release_candidate as d110
from twelve_six.checkpoint import hash_json, sha256_file
from twelve_six.model import TwelveSixDecoder
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer

SCHEMA = "12-6.corpus-v0.2-real-candidate.v1"
AUTHORITY = "LOCAL_FREE_CANDIDATE_NOT_CORPUS_FREEZE_OR_REPRESENTATIVENESS_PROMOTION"
DATA110_BASE_SHA = "fd60b362c7089e20b3c0e1fb37dc839ae5a17c5c"
ORIGIN_EXTERNAL = "EXTERNAL_REAL"
ORIGIN_PROJECT = "PROJECT_AUTHORED"
ALLOWED_ORIGINS = {ORIGIN_EXTERNAL, ORIGIN_PROJECT}


class Data183Error(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Data183Error(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _canonical_origin(value: object) -> str:
    raw = str(value)
    mapping = {
        "external_real": ORIGIN_EXTERNAL,
        ORIGIN_EXTERNAL: ORIGIN_EXTERNAL,
        "project_authored": ORIGIN_PROJECT,
        ORIGIN_PROJECT: ORIGIN_PROJECT,
    }
    if raw not in mapping:
        raise Data183Error(f"unsupported origin class: {raw}")
    return mapping[raw]


def _audit_normalize(text: str) -> str:
    """Independent conservative normalization for cross-split overlap auditing."""
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(line.split()) for line in normalized.split("\n")).strip()


def _norm_sha(text: str) -> str:
    return hashlib.sha256(_audit_normalize(text).encode("utf-8")).hexdigest()


def _iter_rows(build_root: Path, manifest: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    seen_paths: set[str] = set()
    for shard in manifest["physical"]["shards"]:
        rel = str(shard["path"])
        if rel in seen_paths:
            raise Data183Error(f"duplicate shard path: {rel}")
        seen_paths.add(rel)
        path = build_root / rel
        if sha256_file(path) != str(shard["sha256"]):
            raise Data183Error(f"shard hash mismatch: {path}")
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise Data183Error(f"non-object row in {path}")
            yield row


def _target_tokens_for_row(row: Mapping[str, Any], tok: ByteTokenizer) -> int:
    record = TextRecord(str(row["record_id"]), str(row["text"]), str(row["split"]))
    total = 0
    for example in iter_packed_examples(
        [record],
        tok,
        expected_split=str(row["split"]),
        sequence_length=d110.SEQ,
        cross_document=False,
    ):
        total += sum(1 for label in example.labels if int(label) != -100)
    return total


def audit_release(build_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(_iter_rows(build_root, manifest))
    if not rows:
        raise Data183Error("release contains no rows")

    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise Data183Error("duplicate record_id survived DATA-110")

    normalized_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    normalized_hash_counts: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    external_strata: set[str] = set()
    project_code = 0
    tok = ByteTokenizer()

    tokens_origin: Counter[str] = Counter()
    tokens_source: Counter[str] = Counter()
    tokens_stratum: Counter[str] = Counter()
    tokens_source_origin: defaultdict[str, Counter[str]] = defaultdict(Counter)
    train_docs_origin: Counter[str] = Counter()

    for row in rows:
        split = str(row["split"])
        if split not in normalized_hashes:
            raise Data183Error(f"unexpected split: {split}")
        digest = _norm_sha(str(row["text"]))
        normalized_hashes[split].add(digest)
        normalized_hash_counts[digest] += 1

        origin = _canonical_origin(row.get("origin"))
        origins[origin] += 1
        stratum = str(row["stratum"])
        if origin == ORIGIN_EXTERNAL:
            external_strata.add(stratum)
        if origin == ORIGIN_PROJECT and stratum == "code":
            project_code += 1

        if split == "train":
            target_tokens = _target_tokens_for_row(row, tok)
            source_family = str(row["source_id"])
            tokens_origin[origin] += target_tokens
            tokens_source[source_family] += target_tokens
            tokens_stratum[stratum] += target_tokens
            tokens_source_origin[source_family][origin] += target_tokens
            train_docs_origin[origin] += 1

    overlap = sorted(normalized_hashes["train"] & normalized_hashes["validation"])
    duplicate_normalized_hashes = sum(1 for count in normalized_hash_counts.values() if count > 1)
    if overlap:
        raise Data183Error(f"normalized train-validation overlap detected: {len(overlap)}")
    if duplicate_normalized_hashes:
        raise Data183Error(
            f"normalized duplicate documents survived global dedup: {duplicate_normalized_hashes}"
        )
    if not {"uk", "en"}.issubset(external_strata):
        raise Data183Error(f"external real UK/EN missing after gates: {sorted(external_strata)}")
    if project_code <= 0:
        raise Data183Error("project-authored code is missing")
    if set(origins) != ALLOWED_ORIGINS:
        raise Data183Error(f"origin separation incomplete: {sorted(origins)}")

    total_target_tokens = sum(tokens_origin.values())
    if total_target_tokens <= 0:
        raise Data183Error("optimized-token supply is empty")

    source_origin = {
        source: dict(sorted(counter.items()))
        for source, counter in sorted(tokens_source_origin.items())
    }
    return {
        "schema": "12-6.data183-corpus-audit.v1",
        "origin_classes": sorted(ALLOWED_ORIGINS),
        "documents_total": len(rows),
        "documents_by_origin": dict(sorted(origins.items())),
        "train_documents_by_origin": dict(sorted(train_docs_origin.items())),
        "duplicate_record_ids": len(record_ids) - len(set(record_ids)),
        "duplicate_normalized_document_hashes": duplicate_normalized_hashes,
        "normalization_audit": {
            "form": "NFKC + newline canonicalization + per-line whitespace collapse",
            "train_unique_normalized_hashes": len(normalized_hashes["train"]),
            "validation_unique_normalized_hashes": len(normalized_hashes["validation"]),
            "train_validation_overlap": 0,
            "overlap_hashes": [],
        },
        "external_real_strata": sorted(external_strata),
        "external_real_code_present": "code" in external_strata,
        "project_authored_code_documents": project_code,
        "optimized_token_supply": {
            "definition": (
                "one finite TRAIN pass of non-ignored autoregressive target labels emitted by "
                "Product iter_packed_examples with cross_document=false; this is not a repeated-run token budget"
            ),
            "tokenizer_version": tok.identity.version,
            "sequence_length": d110.SEQ,
            "total_target_tokens": total_target_tokens,
            "by_origin": dict(sorted(tokens_origin.items())),
            "by_source_family": dict(sorted(tokens_source.items())),
            "by_stratum": dict(sorted(tokens_stratum.items())),
            "by_source_family_and_origin": source_origin,
            "source_family_semantics": "exact source_id; no inferred source grouping",
        },
    }


def trainer_streaming_proof(repo: Path, build_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run actual retained shard -> packer -> Trainer updates, one per stratum."""
    torch.manual_seed(d110.SEED)
    spec, init, provenance = d110._model(repo)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, d110._trainer_config(), device="cpu")
    tok = ByteTokenizer()
    steps: list[dict[str, Any]] = []
    for stratum in ("uk", "en", "code"):
        iterator = d110._cycling_packed(build_root, manifest, tok, stratum)
        batch = next(d110._batches(iterator))
        before_tokens = int(trainer.tokens_seen)
        metrics = trainer.train_step(batch)
        after_tokens = int(trainer.tokens_seen)
        if after_tokens <= before_tokens:
            raise Data183Error(f"Trainer consumed no optimized tokens for {stratum}")
        steps.append(
            {
                "stratum": stratum,
                "optimizer_step": int(metrics.optimizer_step),
                "optimized_tokens_delta": after_tokens - before_tokens,
                "loss": float(metrics.update_loss if metrics.update_loss is not None else metrics.loss),
            }
        )
    if trainer.optimizer_step != 3:
        raise Data183Error(f"expected three Trainer steps, got {trainer.optimizer_step}")
    return {
        "schema": "12-6.data183-trainer-streaming-proof.v1",
        "path": "retained shard -> _release_rows -> iter_packed_examples -> Trainer.train_step",
        "device": "cpu",
        "local_free": True,
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "model_provenance": provenance,
        "tokenizer_version": tok.identity.version,
        "steps": steps,
        "final_optimizer_step": trainer.optimizer_step,
        "optimized_tokens": int(trainer.tokens_seen),
        "passed": True,
    }


def build_candidate(repo: Path, source_sha: str, external_intake: Path, out: Path) -> dict[str, Any]:
    repo = repo.resolve()
    out = out.resolve()
    external_intake = external_intake.resolve()
    release = d110.build_release(repo, source_sha, external_intake, out / "data110")
    candidate_manifest = release["candidate_manifest"]

    if release.get("two_build_deterministic_identity") is not True:
        raise Data183Error("DATA-110 two-build identity proof missing")
    if release.get("two_build_shards_exact") is not True:
        raise Data183Error("DATA-110 two-build shard proof missing")
    if release["build_a_identity_sha256"] != release["build_b_identity_sha256"]:
        raise Data183Error("DATA-110 clean-build identities differ")

    audit = audit_release(out / "data110" / "build-a", candidate_manifest)
    streaming = trainer_streaming_proof(repo, out / "data110" / "build-a", candidate_manifest)

    external_code = bool(audit["external_real_code_present"])
    status = (
        "CANDIDATE_EXTERNAL_UA_EN_CODE"
        if external_code
        else "CANDIDATE_UA_EN_REAL_PROJECT_CODE"
    )
    blockers = [] if external_code else ["EXTERNAL_REAL_CODE_UNAVAILABLE"]
    blockers.extend(
        [
            "FULL_V0_2_REPRESENTATIVENESS_NOT_ESTABLISHED",
            "DATA110_CLASSIFICATION_RETEST_REQUIRED",
        ]
    )

    core = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "data110_base_sha": DATA110_BASE_SHA,
        "status": status,
        "corpus_identity_sha256": release["corpus_identity_sha256"],
        "two_clean_builds": {
            "build_a_identity_sha256": release["build_a_identity_sha256"],
            "build_b_identity_sha256": release["build_b_identity_sha256"],
            "identical_corpus_identity": True,
            "identical_shard_hashes": True,
        },
        "rights_and_policy_gates": {
            "external_training_rights_explicit": bool(
                candidate_manifest["external_intake"][
                    "all_admitted_external_sources_explicit_training_eligible"
                ]
            ),
            "normalization_materialization_reused": "DATA-25 deterministic normalization/materialization incumbent"
            in candidate_manifest["pipeline"],
            "quality_gate_reexecuted": "DATA-32 document quality incumbent re-executed"
            in candidate_manifest["pipeline"],
            "privacy_gate_reexecuted": "DATA-33 privacy/secrets incumbent re-executed"
            in candidate_manifest["pipeline"],
            "exact_dedup_reexecuted": "SQLiteExactDedupIndex exact dedup"
            in candidate_manifest["pipeline"],
            "near_dedup_decontamination_reexecuted": bool(
                candidate_manifest["dedup_decontamination"]["publication_manifest_sha256"]
            ),
            "cluster_safe_split_reexecuted": candidate_manifest["split"][
                "cluster_straddles_across_variants"
            ]
            == 0,
            "deterministic_sharding_reexecuted": True,
        },
        "origin_contract": {
            "field": "origin",
            "canonical_report_classes": sorted(ALLOWED_ORIGINS),
            "legacy_shard_values_preserved": ["external_real", "project_authored"],
            "silent_origin_coalescing": False,
        },
        "audit": audit,
        "trainer_streaming": streaming,
        "external_code": {
            "present": external_code,
            "claim": external_code,
            "when_absent_fallback": "UA/EN-real + project-code candidate",
        },
        "representativeness": {
            "full_v0_2_claim": False,
            "production_ready_claim": False,
            "semantic_universal_cleanliness_claim": False,
            "blocked": True,
            "blockers": blockers,
        },
        "truth_boundary": {
            "external_real_ua_present": "uk" in audit["external_real_strata"],
            "external_real_en_present": "en" in audit["external_real_strata"],
            "external_real_code_present": external_code,
            "project_authored_code_present": audit["project_authored_code_documents"] > 0,
            "documents_duplicated_to_reach_token_target": False,
            "local_free_only": True,
            "paid_compute": False,
        },
        "upstream_data110_classification": release["classification"],
    }
    core["report_sha256"] = hash_json(core)
    _write_json(out / "corpus-v0.2-real-candidate.json", core)
    return core


def validate_candidate(path: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = _read_json(path)
    expected_hash = str(report.get("report_sha256", ""))
    core = dict(report)
    core.pop("report_sha256", None)
    if hash_json(core) != expected_hash:
        raise Data183Error("candidate report self-hash mismatch")
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise Data183Error("candidate schema/authority mismatch")
    if expected_source_sha and report.get("source_sha") != expected_source_sha:
        raise Data183Error("source SHA mismatch")
    if report["two_clean_builds"]["identical_corpus_identity"] is not True:
        raise Data183Error("two-build corpus identity evidence failed")
    if report["two_clean_builds"]["identical_shard_hashes"] is not True:
        raise Data183Error("two-build shard evidence failed")
    if report["audit"]["normalization_audit"]["train_validation_overlap"] != 0:
        raise Data183Error("normalized train-validation overlap is not zero")
    if report["audit"]["duplicate_normalized_document_hashes"] != 0:
        raise Data183Error("normalized duplicate documents survived")
    if report["trainer_streaming"]["passed"] is not True:
        raise Data183Error("Trainer streaming proof failed")
    if report["truth_boundary"]["local_free_only"] is not True:
        raise Data183Error("LOCAL_FREE boundary weakened")
    if report["truth_boundary"]["documents_duplicated_to_reach_token_target"] is not False:
        raise Data183Error("document duplication truth weakened")
    if report["representativeness"]["full_v0_2_claim"] is not False:
        raise Data183Error("unsupported full V0.2 representativeness claim enabled")
    if not report["external_code"]["present"]:
        if report["status"] != "CANDIDATE_UA_EN_REAL_PROJECT_CODE":
            raise Data183Error("external-code fallback classification mismatch")
        if "EXTERNAL_REAL_CODE_UNAVAILABLE" not in report["representativeness"]["blockers"]:
            raise Data183Error("external code blocker missing")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("--repo-root", type=Path, default=Path("."))
    build.add_argument("--source-sha", required=True)
    build.add_argument("--external-intake", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)

    if args.cmd == "build":
        report = build_candidate(
            args.repo_root,
            args.source_sha,
            args.external_intake,
            args.output_dir,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "corpus_identity_sha256": report["corpus_identity_sha256"],
                    "external_real_code_present": report["external_code"]["present"],
                    "report_sha256": report["report_sha256"],
                },
                indent=2,
            )
        )
    else:
        report = validate_candidate(args.report, args.expected_source_sha)
        print(json.dumps({"validation": "PASS", "report_sha256": report["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
