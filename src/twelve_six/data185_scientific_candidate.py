"""DATA-185 provisional scientific corpus qualification.

Convergence layer over DATA-110. It reuses the accepted rights, quality,
privacy, deduplication, decontamination, split, sharding, packing, Trainer,
checkpoint and evaluation paths rather than creating parallel subsystems.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from twelve_six.checkpoint import hash_json, load_trainer_checkpoint, sha256_file
from twelve_six.data.corpus_v01 import verify_rebuild
from twelve_six.data110_release_candidate import (
    BATCH,
    DATA25_CONFIG,
    DATA25_EXPECTED_ID,
    DATA25_RETAINED,
    MAX_STEPS,
    MIXTURE,
    SEED,
    SEQ,
    _batches,
    _evaluate,
    _model,
    _read_jsonl,
    _trainer_config,
)
from twelve_six.model import TwelveSixDecoder
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer


SCHEMA = "12-6.data185-corpus-v1-scientific-candidate.v1"
AUTHORITY = "CONTROLLED_RESEARCH_QUALIFICATION_NOT_PRODUCTION_OR_CAPABILITY_STATUS"
REPOSITORY = "Oleksii-debug/12-6-ai."
DATA110_SCHEMA = "12-6.data110-corpus-v1-rc-learning.v1"
DATA110_RELEASE_SCHEMA = "12-6.data110-release-manifest.v1"
RESEARCH140_HEAD = "c2fa6ba71691c3d8cc86aa0a1c3c83eb10bce98"
DATA23_HEAD = "5f223f9ef77762a042e966372fdf9f064b3cc9fe"
SOURCE_FAMILY_MIN_BY_STRATUM = {"uk": 2, "en": 2, "code": 1}
MIN_REAL_TRAIN_SHARE = 0.01
QUALITY_MATERIALITY_BPB = 0.02


class Data185Error(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Data185Error(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _require_head(repo: Path, source_sha: str) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if actual != source_sha:
        raise Data185Error(f"exact-head mismatch: {actual} != {source_sha}")


def _candidate_rows(root: Path, manifest: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for shard in manifest["physical"]["shards"]:
        path = root / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise Data185Error(f"candidate shard hash mismatch: {path}")
        yield from _read_jsonl(path)


def _data25_rows(
    root: Path, manifest: Mapping[str, Any], split: str, stratum: str
) -> Iterator[dict[str, Any]]:
    for shard in manifest["shards"]:
        path = root / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise Data185Error(f"DATA-25 shard hash mismatch: {path}")
        for row in _read_jsonl(path):
            if row.get("split") == split and row.get("stratum") == stratum:
                yield row


def _finite_data25(
    root: Path,
    manifest: Mapping[str, Any],
    tok: ByteTokenizer,
    split: str,
    stratum: str,
):
    records = (
        TextRecord(str(row["record_id"]), str(row["text"]), str(row["split"]))
        for row in _data25_rows(root, manifest, split, stratum)
    )
    yield from iter_packed_examples(
        records,
        tok,
        expected_split=split,
        sequence_length=SEQ,
        cross_document=False,
    )


def _cycling_data25(
    root: Path, manifest: Mapping[str, Any], tok: ByteTokenizer, stratum: str
):
    while True:
        yielded = False
        for example in _finite_data25(root, manifest, tok, "train", stratum):
            yielded = True
            yield example
        if not yielded:
            raise Data185Error(f"DATA-25 has no train examples for {stratum}")


def _family_name(row: Mapping[str, Any]) -> str:
    # DATA-110 binds source_id provenance, but no broader semantic family taxonomy.
    # Keep the family definition exact rather than inventing domain groupings.
    return str(row["source_id"])


def _eval_examples(model: TwelveSixDecoder, examples: Sequence[Any]) -> tuple[float, int]:
    ids = torch.tensor([x.input_ids for x in examples], dtype=torch.long)
    labels = torch.tensor([x.labels for x in examples], dtype=torch.long)
    logits = model(ids).logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    tokens = int(targets.ne(-100).sum().item())
    nll = F.cross_entropy(
        logits.reshape(-1, model.spec.vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(nll.item()), tokens


@torch.no_grad()
def _evaluate_families(
    model: TwelveSixDecoder,
    root: Path,
    manifest: Mapping[str, Any],
    tok: ByteTokenizer,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _candidate_rows(root, manifest):
        if row.get("split") == "validation":
            groups[_family_name(row)].append(row)

    output: dict[str, Any] = {}
    training = model.training
    model.eval()
    try:
        for family in sorted(groups):
            rows = groups[family]
            records = (
                TextRecord(str(row["record_id"]), str(row["text"]), "validation")
                for row in rows
            )
            pending = []
            nll_sum = 0.0
            tokens = 0
            for example in iter_packed_examples(
                records,
                tok,
                expected_split="validation",
                sequence_length=SEQ,
                cross_document=False,
            ):
                pending.append(example)
                if len(pending) == 32:
                    nll, count = _eval_examples(model, pending)
                    nll_sum += nll
                    tokens += count
                    pending = []
            if pending:
                nll, count = _eval_examples(model, pending)
                nll_sum += nll
                tokens += count
            output[family] = {
                "strata": sorted({str(row["stratum"]) for row in rows}),
                "validation_documents": len(rows),
                "predicted_byte_tokens": tokens,
                "bits_per_byte": (
                    nll_sum / math.log(2.0) / tokens if tokens > 0 else None
                ),
                "status": "EVALUATED" if tokens > 0 else "NO_PACKED_TARGETS",
            }
    finally:
        model.train(training)
    return output


def _load_candidate_model(
    repo: Path,
    source_sha: str,
    evidence: Path,
    report: Mapping[str, Any],
) -> tuple[TwelveSixDecoder, Trainer, ByteTokenizer]:
    tok = ByteTokenizer()
    spec, init, _ = _model(repo)
    cfg = _trainer_config()
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    manifest = report["release"]["candidate_manifest"]
    load_trainer_checkpoint(
        evidence / str(report["checkpoints"]["retained_exact_checkpoint"]),
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=False,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],
    )
    if trainer.optimizer_step != MAX_STEPS:
        raise Data185Error("candidate retained checkpoint is not the fixed final step")
    return model, trainer, tok


def _build_stats(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(_candidate_rows(root, manifest))
    by_origin = Counter()
    by_stratum_origin = Counter()
    by_family = Counter()
    train_bytes = 0
    train_docs = 0
    for row in rows:
        if row.get("split") != "train":
            continue
        size = len(str(row["text"]).encode("utf-8"))
        origin = str(row.get("origin", "unknown"))
        stratum = str(row["stratum"])
        family = _family_name(row)
        train_bytes += size
        train_docs += 1
        by_origin[origin] += size
        by_stratum_origin[f"{stratum}:{origin}"] += size
        by_family[family] += size

    real_families = {
        stratum: sorted(
            {
                _family_name(row)
                for row in rows
                if row.get("split") == "train"
                and row.get("stratum") == stratum
                and row.get("origin") == "external_real"
            }
        )
        for stratum in ("uk", "en", "code")
    }
    all_families = {
        stratum: sorted(
            {
                _family_name(row)
                for row in rows
                if row.get("split") == "train" and row.get("stratum") == stratum
            }
        )
        for stratum in ("uk", "en", "code")
    }
    real = by_origin["external_real"]
    return {
        "train_documents": train_docs,
        "train_text_bytes": train_bytes,
        "train_text_bytes_by_origin": dict(sorted(by_origin.items())),
        "train_text_bytes_by_stratum_origin": dict(sorted(by_stratum_origin.items())),
        "train_text_bytes_by_source_family": dict(sorted(by_family.items())),
        "source_families_by_stratum": all_families,
        "real_source_families_by_stratum": real_families,
        "real_train_text_bytes": real,
        "real_train_share": real / train_bytes if train_bytes else 0.0,
    }


def _run_previous_arm(
    repo: Path,
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
    out: Path,
) -> dict[str, Any]:
    tok = ByteTokenizer()
    spec, init, _ = _model(repo)
    cfg = _trainer_config()
    rebuilt_root = out / "previous-data25"
    build_a = rebuilt_root / "a"
    build_b = rebuilt_root / "b"
    data25 = verify_rebuild(repo / DATA25_CONFIG, build_a, build_b)
    retained = _read_json(repo / DATA25_RETAINED)
    if data25 != retained or data25["corpus_identity_sha256"] != DATA25_EXPECTED_ID:
        raise Data185Error("previous DATA-25 identity did not rebuild exactly")

    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    initial = _evaluate(model, candidate_root, candidate_manifest, tok)
    iterators = {
        stratum: _cycling_data25(build_a, data25, tok, stratum)
        for stratum in ("uk", "en", "code")
    }
    batches = {stratum: _batches(iterator) for stratum, iterator in iterators.items()}
    losses: list[float] = []
    started = time.perf_counter()
    for index in range(MAX_STEPS):
        stratum = MIXTURE[index % len(MIXTURE)]
        metrics = trainer.train_microbatch(next(batches[stratum]))
        losses.append(
            float(metrics.update_loss if metrics.update_loss is not None else metrics.loss)
        )
    elapsed = time.perf_counter() - started
    final = _evaluate(model, candidate_root, candidate_manifest, tok)
    families = _evaluate_families(model, candidate_root, candidate_manifest, tok)
    return {
        "dataset_identity_sha256": DATA25_EXPECTED_ID,
        "dataset_version": data25["corpus_version"],
        "two_build_reproduced": True,
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "init_spec_sha256": init.identity_sha256(),
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
        },
        "trainer_config": asdict(cfg),
        "batch_size": BATCH,
        "sequence_length": SEQ,
        "steps": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "initial_common_eval": initial,
        "final_common_eval": final,
        "source_family_bpb": families,
        "first64_mean_loss": sum(losses[:64]) / 64,
        "last64_mean_loss": sum(losses[-64:]) / 64,
        "wall_seconds_training_only": elapsed,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "random_initialization": True,
        "foreign_pretrained_weights": False,
        "paid_compute": False,
    }


def _research140_single_pair(baseline_bpb: float, candidate_bpb: float) -> dict[str, Any]:
    # RESEARCH-140 requires at least three paired repeats for winner selection.
    # DATA-185's mandated single fixed trajectory is therefore descriptive only.
    return {
        "methodology_upstream_head": RESEARCH140_HEAD,
        "metric": "common_selection_validation_bits_per_byte",
        "direction": "lower_is_better",
        "materiality_threshold_bpb": QUALITY_MATERIALITY_BPB,
        "paired_repeats": 1,
        "oriented_delta_baseline_minus_candidate_bpb": baseline_bpb - candidate_bpb,
        "decision": "INSUFFICIENT_REPEATS",
        "winner": None,
        "reason_codes": ["fewer_than_three_paired_repeats"],
        "inferential_claim": (
            "single paired trajectory is descriptive only; no p-value or "
            "asymptotic significance claim"
        ),
    }


def qualify(repo: Path, source_sha: str, evidence: Path, out: Path) -> dict[str, Any]:
    _require_head(repo, source_sha)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    report = _read_json(evidence / "report.json")
    if report.get("schema") != DATA110_SCHEMA:
        raise Data185Error("DATA-110 learning report schema mismatch")
    if report["source"]["git_sha"] != source_sha:
        raise Data185Error("DATA-110 evidence is not bound to DATA-185 exact head")
    release = report["release"]
    if release.get("schema_version") != DATA110_RELEASE_SCHEMA:
        raise Data185Error("DATA-110 release schema mismatch")
    manifest = release["candidate_manifest"]
    candidate_root = evidence / "build-a"

    stats = _build_stats(candidate_root, manifest)
    candidate_model, candidate_trainer, tok = _load_candidate_model(
        repo, source_sha, evidence, report
    )
    candidate_common = _evaluate(candidate_model, candidate_root, manifest, tok)
    expected_bpb = float(report["evaluation"]["final_bits_per_byte"])
    if abs(float(candidate_common["bits_per_byte"]) - expected_bpb) > 1e-10:
        raise Data185Error("candidate checkpoint common-eval reproduction mismatch")
    candidate_families = _evaluate_families(
        candidate_model, candidate_root, manifest, tok
    )

    previous = _run_previous_arm(repo, candidate_root, manifest, out)
    previous_bpb = float(previous["final_common_eval"]["bits_per_byte"])
    candidate_bpb = float(candidate_common["bits_per_byte"])
    research = _research140_single_pair(previous_bpb, candidate_bpb)
    real_families = stats["real_source_families_by_stratum"]

    gates = {
        "nontrivial_real_source_share": {
            "threshold": MIN_REAL_TRAIN_SHARE,
            "observed": stats["real_train_share"],
            "passed": stats["real_train_share"] >= MIN_REAL_TRAIN_SHARE,
        },
        "source_family_diversity": {
            "minimum_real_families_by_stratum": SOURCE_FAMILY_MIN_BY_STRATUM,
            "observed_real_families_by_stratum": real_families,
            "passed": all(
                len(real_families[stratum]) >= SOURCE_FAMILY_MIN_BY_STRATUM[stratum]
                for stratum in SOURCE_FAMILY_MIN_BY_STRATUM
            ),
        },
        "real_code": {
            "present": bool(real_families["code"]),
            "passed": bool(real_families["code"]),
            "explicit_blocker": {
                "code": "RIGHTS_APPROVED_REAL_CODE_ABSENT",
                "data23_upstream_head": DATA23_HEAD,
                "data23_pilot_truth": (
                    "3 real Python files / 4,998 bytes were mechanically accepted "
                    "but 0 files / 0 bytes were training eligible under the live "
                    "rights registry."
                ),
            },
        },
        "two_build_determinism": {
            "passed": release["two_build_deterministic_identity"] is True
            and release["two_build_shards_exact"] is True,
            "build_a_identity_sha256": release["build_a_identity_sha256"],
            "build_b_identity_sha256": release["build_b_identity_sha256"],
        },
        "zero_train_validation_cluster_leakage": {
            "observed_cluster_straddles": manifest["split"][
                "cluster_straddles_across_variants"
            ],
            "passed": manifest["split"]["cluster_straddles_across_variants"] == 0,
        },
        "reserved_eval_exclusion": {
            "training_eligibility_envelope_sha256": manifest[
                "dedup_decontamination"
            ]["training_eligibility_envelope_sha256"],
            "d06_exact_removed": manifest["dedup_decontamination"][
                "exact_d06_matches_removed"
            ],
            "d06_near_removed": manifest["dedup_decontamination"][
                "near_d06_matches_removed"
            ],
            "passed": bool(
                manifest["dedup_decontamination"][
                    "training_eligibility_envelope_sha256"
                ]
            ),
        },
        "streaming_trainer_proof": {
            "optimized_tokens": report["training"]["optimized_tokens"],
            "fresh_process_resume_passed": report["runtime"][
                "fresh_process_resume"
            ]["passed"],
            "packing": report["packing"],
            "passed": report["training"]["optimized_tokens"] >= 400_000
            and report["runtime"]["fresh_process_resume"]["passed"] is True,
        },
        "common_eval_non_mutation": {
            "passed": bool(candidate_common["non_mutation_passed"])
            and bool(previous["final_common_eval"]["non_mutation_passed"]),
        },
        "single_pair_quality_non_regression": {
            "candidate_bpb": candidate_bpb,
            "previous_data25_bpb": previous_bpb,
            "allowed_regression_bpb": QUALITY_MATERIALITY_BPB,
            "passed": candidate_bpb <= previous_bpb + QUALITY_MATERIALITY_BPB,
            "selection_claim": False,
        },
    }

    reasons: list[dict[str, Any]] = []
    for gate_name, gate in gates.items():
        if gate.get("passed") is not True:
            reasons.append(
                {
                    "code": f"GATE_FAILED_{gate_name.upper()}",
                    "gate": gate_name,
                    "blocking_freeze": True,
                    "detail": gate,
                }
            )
    reasons.append(
        {
            "code": "RESEARCH140_SINGLE_PAIR_DESCRIPTIVE_ONLY",
            "blocking_freeze": False,
            "detail": research,
        }
    )

    blocking = [reason for reason in reasons if reason["blocking_freeze"]]
    status = "FREEZE_FOR_RESEARCH_V1" if not blocking else "RETEST_REQUIRED"
    result = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": REPOSITORY,
            "git_sha": source_sha,
            "branch": "data185/corpus-v1-scientific-candidate-20260826",
        },
        "status": status,
        "machine_reasons": reasons,
        "candidate": {
            "corpus_identity_sha256": release["corpus_identity_sha256"],
            "data110_report_sha256": report["report_sha256"],
            "data110_release_manifest_sha256": release["release_manifest_sha256"],
            "stats": stats,
        },
        "requirements": gates,
        "fixed_approx_1m_comparison": {
            "model_parameter_count": report["model"]["parameter_count"],
            "candidate_dataset_identity_sha256": release["corpus_identity_sha256"],
            "previous_dataset_identity_sha256": DATA25_EXPECTED_ID,
            "same_model_geometry": True,
            "same_init_seed": SEED,
            "same_optimizer_steps": MAX_STEPS,
            "same_mixture_pattern": list(MIXTURE),
            "common_evaluation_identity": {
                "candidate_validation_split_family_identity_sha256": manifest[
                    "split"
                ]["split_family_identity_sha256"],
                "candidate_validation_corpus_identity_sha256": manifest[
                    "corpus_identity_sha256"
                ],
                "packing_sequence_length": SEQ,
                "tokenizer_config_sha256": tok.identity.config_sha256,
            },
            "candidate": {
                "optimized_tokens": candidate_trainer.tokens_seen,
                "aggregate_and_stratum_bpb": candidate_common,
                "source_family_bpb": candidate_families,
            },
            "previous_data25": previous,
            "research140_decision": research,
        },
        "truth_boundary": {
            "freeze_scope_if_passed": (
                "controlled comparable research identity only; not production "
                "or capability status"
            ),
            "production_ready_claim": False,
            "intelligence_claim": False,
            "alignment_claim": False,
            "instruction_following_claim": False,
            "representative_population_claim": False,
            "foreign_pretrained_weights": False,
            "sft_rlhf_dpo": False,
            "paid_compute": False,
            "local_free": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(),
        },
    }
    result["report_sha256"] = hash_json(result)
    _write_json(out / "data185-scientific-candidate.json", result)
    return result


def validate(path: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = _read_json(path)
    supplied = report.pop("report_sha256", None)
    if supplied != hash_json(report):
        raise Data185Error("report self-hash mismatch")
    report["report_sha256"] = supplied
    if report["schema"] != SCHEMA or report["authority"] != AUTHORITY:
        raise Data185Error("schema/authority mismatch")
    if report["status"] not in {
        "FREEZE_FOR_RESEARCH_V1",
        "RETEST_REQUIRED",
        "BLOCKED",
    }:
        raise Data185Error("invalid qualification status")
    if expected_source_sha and report["source"]["git_sha"] != expected_source_sha:
        raise Data185Error("source SHA mismatch")
    if (
        report["fixed_approx_1m_comparison"]["previous_dataset_identity_sha256"]
        != DATA25_EXPECTED_ID
    ):
        raise Data185Error("previous dataset identity drift")
    if report["truth_boundary"]["local_free"] is not True:
        raise Data185Error("LOCAL_FREE truth boundary weakened")
    if report["truth_boundary"]["foreign_pretrained_weights"] is not False:
        raise Data185Error("foreign-weight truth boundary weakened")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qualify")
    q.add_argument("--repo-root", type=Path, default=Path("."))
    q.add_argument("--source-sha", required=True)
    q.add_argument("--data110-evidence", type=Path, required=True)
    q.add_argument("--output-dir", type=Path, required=True)
    v = sub.add_parser("validate")
    v.add_argument("report", type=Path)
    v.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)
    if args.cmd == "qualify":
        result = qualify(
            args.repo_root.resolve(),
            args.source_sha,
            args.data110_evidence.resolve(),
            args.output_dir.resolve(),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "machine_reasons": result["machine_reasons"],
                    "report_sha256": result["report_sha256"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        result = validate(args.report, args.expected_source_sha)
        print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
