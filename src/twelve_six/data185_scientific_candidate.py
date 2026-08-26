"""DATA-185 provisional scientific corpus qualification.

Convergence layer over DATA-110. It reuses accepted rights, quality, privacy,
deduplication, decontamination, split, sharding, packing, Trainer, checkpoint
and evaluation paths. Unsupported scientific claims remain absent or null.
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


def _candidate_rows(
    root: Path, manifest: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
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
    source_id = row.get("source_id")
    if source_id is None or not str(source_id).strip():
        return "UNBOUND_SOURCE_ID"
    return str(source_id)


def _row_hash(row: Mapping[str, Any]) -> str:
    supplied = row.get("content_sha256")
    if isinstance(supplied, str) and len(supplied) == 64:
        return supplied
    import hashlib

    return hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()


def _eval_examples(
    model: TwelveSixDecoder, examples: Sequence[Any]
) -> tuple[float, int]:
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


def _state_snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_unchanged(
    before: Mapping[str, torch.Tensor], model: TwelveSixDecoder
) -> bool:
    after = model.state_dict()
    return set(before) == set(after) and all(
        torch.equal(before[name], after[name].detach().cpu()) for name in before
    )


@torch.no_grad()
def _evaluate_data25_validation(
    model: TwelveSixDecoder,
    root: Path,
    manifest: Mapping[str, Any],
    tok: ByteTokenizer,
) -> dict[str, Any]:
    """Evaluate on the original DATA-25 validation split.

    DATA-110 selects project-authored material only from DATA-25 *train*, so the
    original DATA-25 validation split is the leakage-safe common comparison set
    for candidate-vs-previous trajectories. A separate explicit content-hash
    exclusion check is still required before this metric can be authoritative.
    """

    training = model.training
    before = _state_snapshot(model)
    model.eval()
    by_stratum: dict[str, Any] = {}
    total_nll = 0.0
    total_tokens = 0
    try:
        for stratum in ("uk", "en", "code"):
            pending: list[Any] = []
            nll_sum = 0.0
            tokens = 0
            examples = 0
            for example in _finite_data25(
                root, manifest, tok, "validation", stratum
            ):
                pending.append(example)
                examples += 1
                if len(pending) == 32:
                    nll, count = _eval_examples(model, pending)
                    nll_sum += nll
                    tokens += count
                    pending = []
            if pending:
                nll, count = _eval_examples(model, pending)
                nll_sum += nll
                tokens += count
            by_stratum[stratum] = {
                "packed_examples": examples,
                "predicted_byte_tokens": tokens,
                "bits_per_byte": (
                    nll_sum / math.log(2.0) / tokens if tokens > 0 else None
                ),
                "status": "EVALUATED" if tokens > 0 else "NO_PACKED_TARGETS",
            }
            total_nll += nll_sum
            total_tokens += tokens
    finally:
        model.train(training)
    return {
        "bits_per_byte": (
            total_nll / math.log(2.0) / total_tokens if total_tokens > 0 else None
        ),
        "predicted_byte_tokens": total_tokens,
        "by_stratum": by_stratum,
        "non_mutation_passed": _state_unchanged(before, model),
    }


def _family_eval(
    model: TwelveSixDecoder,
    rows: Sequence[Mapping[str, Any]],
    tok: ByteTokenizer,
    *,
    expected_split: str,
    record_id_key: str,
    include_families: Sequence[str] | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    strata_by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family = _family_name(row)
        strata_by_family[family].add(str(row["stratum"]))
        if row.get("split") == expected_split:
            groups[family].append(row)
    families = set(include_families or ()) | set(strata_by_family)
    output: dict[str, Any] = {}
    training = model.training
    before = _state_snapshot(model)
    model.eval()
    try:
        for family in sorted(families):
            family_rows = groups.get(family, [])
            if not family_rows:
                output[family] = {
                    "strata": sorted(strata_by_family.get(family, set())),
                    "validation_documents": 0,
                    "predicted_byte_tokens": 0,
                    "bits_per_byte": None,
                    "status": "NO_VALIDATION_EXAMPLES",
                }
                continue
            records = (
                TextRecord(
                    str(row[record_id_key]), str(row["text"]), expected_split
                )
                for row in family_rows
            )
            pending: list[Any] = []
            nll_sum = 0.0
            tokens = 0
            for example in iter_packed_examples(
                records,
                tok,
                expected_split=expected_split,
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
                "strata": sorted(strata_by_family.get(family, set())),
                "validation_documents": len(family_rows),
                "predicted_byte_tokens": tokens,
                "bits_per_byte": (
                    nll_sum / math.log(2.0) / tokens if tokens > 0 else None
                ),
                "status": "EVALUATED" if tokens > 0 else "NO_PACKED_TARGETS",
            }
    finally:
        model.train(training)
    if not _state_unchanged(before, model):
        raise Data185Error("source-family evaluation mutated model parameters")
    return output


@torch.no_grad()
def _evaluate_candidate_families(
    model: TwelveSixDecoder,
    root: Path,
    manifest: Mapping[str, Any],
    tok: ByteTokenizer,
) -> dict[str, Any]:
    rows = list(_candidate_rows(root, manifest))
    families = sorted({_family_name(row) for row in rows})
    return _family_eval(
        model,
        rows,
        tok,
        expected_split="validation",
        record_id_key="record_id",
        include_families=families,
    )


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
    by_origin: Counter[str] = Counter()
    by_stratum_origin: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
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


def _data25_validation_rows(
    root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum in ("uk", "en", "code"):
        rows.extend(_data25_rows(root, manifest, "validation", stratum))
    return rows


def _comparison_exclusion(
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
    data25_root: Path,
    data25_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_train = {
        _row_hash(row)
        for row in _candidate_rows(candidate_root, candidate_manifest)
        if row.get("split") == "train"
    }
    data25_validation_rows = _data25_validation_rows(data25_root, data25_manifest)
    validation = {_row_hash(row) for row in data25_validation_rows}
    overlap = sorted(candidate_train & validation)
    return {
        "common_eval_dataset_identity_sha256": DATA25_EXPECTED_ID,
        "common_eval_split": "validation",
        "candidate_train_unique_content_hashes": len(candidate_train),
        "data25_validation_unique_content_hashes": len(validation),
        "exact_content_overlap_count": len(overlap),
        "overlap_sha256": hash_json(overlap),
        "passed": len(overlap) == 0,
        "near_duplicate_exclusion_claim": False,
        "note": (
            "DATA-110 project rows originate from DATA-25 train only; this extra "
            "exact-content check binds the original DATA-25 validation split as "
            "the common comparison identity without asserting semantic/near-dup "
            "cleanliness beyond the incumbent corpus guarantees."
        ),
    }


def _run_previous_arm(
    repo: Path,
    candidate_model: TwelveSixDecoder,
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

    exclusion = _comparison_exclusion(
        candidate_root, candidate_manifest, build_a, data25
    )
    candidate_common = _evaluate_data25_validation(
        candidate_model, build_a, data25, tok
    )
    data25_validation_rows = _data25_validation_rows(build_a, data25)
    candidate_common_families = _family_eval(
        candidate_model,
        data25_validation_rows,
        tok,
        expected_split="validation",
        record_id_key="record_id",
    )

    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    initial = _evaluate_data25_validation(model, build_a, data25, tok)
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
    final = _evaluate_data25_validation(model, build_a, data25, tok)
    previous_common_families = _family_eval(
        model,
        data25_validation_rows,
        tok,
        expected_split="validation",
        record_id_key="record_id",
    )
    diagnostic_candidate_validation = _evaluate(
        model, candidate_root, candidate_manifest, tok
    )
    diagnostic_candidate_families = _evaluate_candidate_families(
        model, candidate_root, candidate_manifest, tok
    )
    common_identity_core = {
        "dataset_identity_sha256": DATA25_EXPECTED_ID,
        "split": "validation",
        "packing_sequence_length": SEQ,
        "cross_document": False,
        "tokenizer_config_sha256": tok.identity.config_sha256,
        "tokenizer_vocab_sha256": tok.identity.vocab_sha256,
    }
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
        "common_evaluation_identity": {
            **common_identity_core,
            "identity_sha256": hash_json(common_identity_core),
        },
        "common_evaluation_training_exclusion": exclusion,
        "candidate_model_common_eval": candidate_common,
        "candidate_model_common_source_family_bpb": candidate_common_families,
        "initial_common_eval": initial,
        "final_common_eval": final,
        "previous_model_common_source_family_bpb": previous_common_families,
        "candidate_validation_diagnostic": {
            "aggregate_and_stratum_bpb": diagnostic_candidate_validation,
            "source_family_bpb": diagnostic_candidate_families,
            "selection_metric": False,
            "caveat": (
                "not used for candidate-vs-previous selection because DATA-110 "
                "re-splits material originating from DATA-25 train"
            ),
        },
        "first64_mean_loss": sum(losses[:64]) / 64,
        "last64_mean_loss": sum(losses[-64:]) / 64,
        "wall_seconds_training_only": elapsed,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "random_initialization": True,
        "foreign_pretrained_weights": False,
        "paid_compute": False,
    }


def _research140_single_pair(
    baseline_bpb: float, candidate_bpb: float
) -> dict[str, Any]:
    return {
        "methodology_upstream_head": RESEARCH140_HEAD,
        "metric": "common_data25_validation_bits_per_byte",
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


def _truth_boundary() -> dict[str, Any]:
    return {
        "freeze_scope_if_passed": (
            "controlled comparable research identity only; not production or "
            "capability status"
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
    candidate_own_eval = _evaluate(candidate_model, candidate_root, manifest, tok)
    expected_bpb = float(report["evaluation"]["final_bits_per_byte"])
    if abs(float(candidate_own_eval["bits_per_byte"]) - expected_bpb) > 1e-10:
        raise Data185Error("candidate checkpoint own-heldout reproduction mismatch")
    candidate_own_families = _evaluate_candidate_families(
        candidate_model, candidate_root, manifest, tok
    )

    previous = _run_previous_arm(
        repo, candidate_model, candidate_root, manifest, out
    )
    previous_bpb = float(previous["final_common_eval"]["bits_per_byte"])
    candidate_bpb = float(previous["candidate_model_common_eval"]["bits_per_byte"])
    research = _research140_single_pair(previous_bpb, candidate_bpb)
    real_families = stats["real_source_families_by_stratum"]
    exclusion = previous["common_evaluation_training_exclusion"]

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
        "common_eval_training_exclusion": exclusion,
        "common_eval_non_mutation": {
            "candidate_passed": previous["candidate_model_common_eval"][
                "non_mutation_passed"
            ],
            "previous_passed": previous["final_common_eval"][
                "non_mutation_passed"
            ],
            "passed": bool(
                previous["candidate_model_common_eval"]["non_mutation_passed"]
            )
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
            "own_validation": {
                "aggregate_and_stratum_bpb": candidate_own_eval,
                "source_family_bpb": candidate_own_families,
                "selection_metric": False,
            },
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
            "common_evaluation_identity": previous["common_evaluation_identity"],
            "candidate": {
                "optimized_tokens": candidate_trainer.tokens_seen,
                "aggregate_and_stratum_bpb": previous[
                    "candidate_model_common_eval"
                ],
                "source_family_bpb": previous[
                    "candidate_model_common_source_family_bpb"
                ],
            },
            "previous_data25": previous,
            "research140_decision": research,
        },
        "truth_boundary": _truth_boundary(),
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


def _blocked_report(source_sha: str, exc: Exception) -> dict[str, Any]:
    result = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": REPOSITORY,
            "git_sha": source_sha,
            "branch": "data185/corpus-v1-scientific-candidate-20260826",
        },
        "status": "BLOCKED",
        "machine_reasons": [
            {
                "code": "QUALIFICATION_EXECUTION_BLOCKED",
                "blocking_freeze": True,
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
        ],
        "truth_boundary": _truth_boundary(),
        "unsupported_sections_absent": [
            "candidate",
            "requirements",
            "fixed_approx_1m_comparison",
        ],
    }
    result["report_sha256"] = hash_json(result)
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
    if report["truth_boundary"]["local_free"] is not True:
        raise Data185Error("LOCAL_FREE truth boundary weakened")
    if report["truth_boundary"]["foreign_pretrained_weights"] is not False:
        raise Data185Error("foreign-weight truth boundary weakened")
    if report["status"] == "BLOCKED":
        if "fixed_approx_1m_comparison" in report:
            raise Data185Error("BLOCKED report must not fabricate comparison evidence")
        return report
    if (
        report["fixed_approx_1m_comparison"]["previous_dataset_identity_sha256"]
        != DATA25_EXPECTED_ID
    ):
        raise Data185Error("previous dataset identity drift")
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
        try:
            result = qualify(
                args.repo_root.resolve(),
                args.source_sha,
                args.data110_evidence.resolve(),
                args.output_dir.resolve(),
            )
        except Exception as exc:  # fail closed into the required machine status
            result = _blocked_report(args.source_sha, exc)
            _write_json(
                args.output_dir.resolve() / "data185-scientific-candidate.json",
                result,
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
