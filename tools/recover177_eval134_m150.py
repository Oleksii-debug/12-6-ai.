"""Recover EVAL-134 against exact MILESTONE-150 retained learned Base evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from twelve_six.code_diagnostic import (
    canonical_json_sha256,
    load_suite,
    score_suite,
    serializable_scores,
    suite_file_sha256,
    summarize,
)
from twelve_six.data.corpus_v01 import verify_rebuild
from twelve_six.data.pipeline import normalize_text
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.milestone100_first_learned import _state_hash
from twelve_six.milestone150_learned_base_ladder import (
    EXPECTED_CORPUS_ID,
    SCALE_ORDER,
    SEED,
    init_spec,
    model_spec,
    validate_ladder,
)
from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

SCHEMA = "12-6.eval134-m150-code-recovery.v1"
AUTHORITY = "RESERVED_MECHANISTIC_CODE_DIAGNOSTIC_LOCAL_FREE_ONLY"
SUITE_PATH = Path("eval/reserved/code_diag_v1/probes.jsonl")
MANIFEST_PATH = Path("eval/reserved/code_diag_v1/manifest.json")
PROVENANCE_PATH = Path("eval/reserved/code_diag_v1/reservation_provenance.json")
CORPUS_CONFIG = Path("configs/data/corpus_v01.json")
EXPECTED_EVAL134_ORIGIN = "74fee51945c83ebdf39e171a894741964ba51b6d"
EXPECTED_SUITE_ID = "eval134-code-diagnostic-v1"
EXPECTED_SUITE_SHA = "df18192f6190cc5d8be9492103a15097daaaf31afdd1cd45b2f4c21af5721105"


class RecoveryError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_root(values: set[str]) -> str:
    return _sha_text("\n".join(sorted(values)) + "\n")


def _candidate_texts(probes) -> tuple[str, ...]:
    return tuple(probe.prefix + choice for probe in probes for choice in probe.choices)


def _artifact_root(path: Path) -> Path:
    candidates = (path, path / "milestone150-evidence")
    for candidate in candidates:
        if (candidate / "ladder-report.json").is_file():
            return candidate
    raise RecoveryError(f"M150 ladder-report.json not found under {path}")


def _verify_suite_identity(repo: Path) -> tuple[tuple[Any, ...], dict[str, Any]]:
    probes = load_suite(repo / SUITE_PATH)
    manifest = _read_json(repo / MANIFEST_PATH)
    provenance = _read_json(repo / PROVENANCE_PATH)
    if manifest.get("schema") != "12-6.code-diagnostic-suite.v1":
        raise RecoveryError("EVAL-134 suite schema drift")
    if manifest.get("suite_id") != EXPECTED_SUITE_ID:
        raise RecoveryError("EVAL-134 suite id drift")
    if manifest.get("data_sha256") != suite_file_sha256(repo / SUITE_PATH):
        raise RecoveryError("EVAL-134 suite data hash drift")
    unsigned = dict(manifest)
    identity = unsigned.pop("suite_identity_sha256", None)
    if identity != canonical_json_sha256(unsigned) or identity != EXPECTED_SUITE_SHA:
        raise RecoveryError("EVAL-134 suite identity drift")
    if manifest.get("status") != "RESERVED_EVALUATION_ONLY":
        raise RecoveryError("EVAL-134 suite is not reserved evaluation material")
    if provenance.get("origin_head_sha") != EXPECTED_EVAL134_ORIGIN:
        raise RecoveryError("EVAL-134 reservation origin drift")
    if provenance.get("suite_identity_sha256") != identity:
        raise RecoveryError("reservation provenance suite identity drift")
    if provenance.get("data_sha256") != manifest["data_sha256"]:
        raise RecoveryError("reservation provenance data hash drift")

    candidates = _candidate_texts(probes)
    exact = {_sha_text(value) for value in candidates}
    normalized = {_sha_text(normalize_text(value)) for value in candidates}
    if provenance.get("candidate_exact_sha256_root") != _hash_root(exact):
        raise RecoveryError("reserved exact-candidate root drift")
    if provenance.get("candidate_normalized_sha256_root") != _hash_root(normalized):
        raise RecoveryError("reserved normalized-candidate root drift")
    required = {
        "balanced_delimiters",
        "indentation_sensitive_continuation",
        "operator_type_syntax",
        "simple_function_call_structure",
        "variable_reuse",
        "string_comment_termination",
        "json_like_structure",
        "language_specific_syntax",
    }
    if {probe.category for probe in probes} != required:
        raise RecoveryError("required syntax-structure strata drift")
    return probes, {
        "suite_id": manifest["suite_id"],
        "suite_identity_sha256": identity,
        "data_sha256": manifest["data_sha256"],
        "items": len(probes),
        "candidate_continuations": len(candidates),
        "exact_registry_hashes_verified": len(exact),
        "normalized_registry_hashes_verified": len(normalized),
        "reservation_origin_pr": provenance["origin_pr"],
        "reservation_origin_head_sha": provenance["origin_head_sha"],
        "reservation_registry_blob_sha": provenance["origin_registry_blob_sha"],
        "candidate_exact_sha256_root": provenance["candidate_exact_sha256_root"],
        "candidate_normalized_sha256_root": provenance[
            "candidate_normalized_sha256_root"
        ],
    }


def verify_reservation_against_data25(repo: Path) -> dict[str, Any]:
    probes, suite = _verify_suite_identity(repo)
    candidates = _candidate_texts(probes)
    normalized_candidates = tuple(normalize_text(value) for value in candidates)
    with tempfile.TemporaryDirectory(prefix="recover177-data25-") as temp:
        base = Path(temp)
        built = verify_rebuild(repo / CORPUS_CONFIG, base / "corpus-a", base / "corpus-b")
        if built.get("corpus_identity_sha256") != EXPECTED_CORPUS_ID:
            raise RecoveryError("DATA-25 corpus identity drift")
        docs: list[str] = []
        train_by_stratum: dict[str, int] = {"uk": 0, "en": 0, "code": 0}
        for shard in built["shards"]:
            path = base / "corpus-a" / shard["path"]
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("split") != "train":
                    continue
                text = row.get("text")
                if not isinstance(text, str):
                    raise RecoveryError(f"DATA-25 train row missing text: {shard['path']}")
                docs.append(text)
                stratum = str(row.get("stratum"))
                if stratum not in train_by_stratum:
                    raise RecoveryError(f"unexpected DATA-25 stratum: {stratum}")
                train_by_stratum[stratum] += 1
        raw_training = "\n".join(docs)
        normalized_training = "\n".join(normalize_text(text) for text in docs)
        exact_overlap_hashes = sorted(
            _sha_text(candidate) for candidate in candidates if candidate in raw_training
        )
        normalized_overlap_hashes = sorted(
            _sha_text(candidate)
            for candidate in normalized_candidates
            if candidate in normalized_training
        )
        if exact_overlap_hashes or normalized_overlap_hashes:
            raise RecoveryError("reserved EVAL-134 candidates overlap DATA-25 training")
        if "qzv_" in raw_training or "qzv_" in normalized_training:
            raise RecoveryError("synthetic qzv_ namespace exists in DATA-25 training")
        suite.update(
            {
                "corpus_identity_sha256": built["corpus_identity_sha256"],
                "training_documents_scanned": len(docs),
                "training_documents_by_stratum": train_by_stratum,
                "training_raw_utf8_bytes_scanned": len(raw_training.encode("utf-8")),
                "training_overlap_count": 0,
                "normalized_training_overlap_count": 0,
                "synthetic_identifier_namespace_absent_from_training": True,
                "decontamination_boundary": (
                    "reserved prefix+choice candidates scanned against exact reconstructed "
                    "DATA-25 train rows in raw and D03-normalized text; validation rows excluded"
                ),
            }
        )
    return suite


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _short_continuation_summary(scores) -> dict[str, Any]:
    correct = [score.choices[0] for score in scores]
    return {
        "correct_continuations": len(correct),
        "exact_next_token_correct_continuations": sum(
            item.exact_next_token for item in correct
        ),
        "short_multitoken_correct_continuations": sum(
            not item.exact_next_token for item in correct
        ),
        "mean_correct_log_likelihood_nats": sum(
            item.log_likelihood_nats for item in correct
        )
        / len(correct),
        "mean_correct_nll_per_source_byte": sum(
            item.nll_per_source_byte for item in correct
        )
        / len(correct),
        "forced_boundary_joint_matches": sum(
            item.forced_boundary_matches_joint_encoding for item in correct
        ),
    }


def _byte_fragmentation(probes, tokenizer: ByteTokenizer) -> dict[str, Any]:
    rows = []
    for probe in probes:
        for role, text in (("correct", probe.completion), *(
            (f"distractor_{i}", value)
            for i, value in enumerate(probe.distractors, 1)
        )):
            source_bytes = len(text.encode("utf-8"))
            token_count = len(tokenizer.encode(text))
            rows.append(
                {
                    "probe_id": probe.id,
                    "role": role,
                    "source_bytes": source_bytes,
                    "tokens": token_count,
                    "tokens_per_source_byte": token_count / source_bytes,
                    "forced_boundary_matches_joint": (
                        tokenizer.encode(probe.prefix) + tokenizer.encode(text)
                        == tokenizer.encode(probe.prefix + text)
                    ),
                }
            )
    total_bytes = sum(row["source_bytes"] for row in rows)
    total_tokens = sum(row["tokens"] for row in rows)
    return {
        "tokenizer": "s0-byte-v1",
        "rows": rows,
        "aggregate": {
            "source_bytes": total_bytes,
            "tokens": total_tokens,
            "tokens_per_source_byte": total_tokens / total_bytes,
            "boundary_retokenization_changes": sum(
                not row["forced_boundary_matches_joint"] for row in rows
            ),
        },
    }


def _score_model(model, tokenizer, probes, *, first_party_backend=None) -> dict[str, Any]:
    state_before = _model_state_hash(model)
    mode_before = model.training
    logits_check: dict[str, Any]
    if first_party_backend is not None:
        prefix_ids = tokenizer.encode(probes[0].prefix)
        logits = list(first_party_backend.next_token_logits(prefix_ids))
        if len(logits) != tokenizer.vocab_size or not all(math.isfinite(x) for x in logits):
            raise RecoveryError("first-party checkpoint logits are invalid")
        logits_check = {
            "path": "FirstPartyInferenceBackend.next_token_logits",
            "vocab_size": len(logits),
            "finite": True,
            "argmax_token_id": max(range(len(logits)), key=logits.__getitem__),
        }
    else:
        logits_check = {
            "path": "TwelveSixDecoder.forward_random_init",
            "checkpoint_applicable": False,
        }
    scores = score_suite(model, tokenizer, probes)
    state_after = _model_state_hash(model)
    if state_before != state_after:
        raise RecoveryError("EVAL-134 scoring mutated model parameters or buffers")
    if model.training != mode_before:
        raise RecoveryError("EVAL-134 scoring failed to restore model mode")
    return {
        "summary": summarize(scores),
        "syntax_structure_strata": summarize(scores)["by_category"],
        "short_continuation_likelihood": _short_continuation_summary(scores),
        "scores": serializable_scores(scores),
        "non_mutation": {
            "passed": True,
            "state_sha256_before": state_before,
            "state_sha256_after": state_after,
            "model_mode_before": mode_before,
            "model_mode_after": model.training,
        },
        "first_party_logits_check": logits_check,
    }


def _random_cell(scale: str, producer_root: Path, ladder: dict[str, Any], probes) -> dict[str, Any]:
    phase1 = _read_json(producer_root / scale / "phase1.json")
    torch.manual_seed(SEED)
    spec = model_spec(scale)
    init = init_spec()
    model = TwelveSixDecoder(spec, init)
    state = _state_hash(model)
    expected = phase1["model"]["random_init_state_sha256"]
    if state != expected:
        raise RecoveryError(f"{scale}: random-init reconstruction state hash mismatch")
    scale_report = ladder["scales"][scale]
    if spec.identity_sha256() != scale_report["model"]["spec_sha256"]:
        raise RecoveryError(f"{scale}: random ModelSpec identity mismatch")
    tokenizer = ByteTokenizer()
    result = _score_model(model, tokenizer, probes)
    result["identity"] = {
        "kind": "deterministic_random_init_reconstruction",
        "scale": scale,
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "init_spec_sha256": init.identity_sha256(),
        "seed": SEED,
        "state_sha256": state,
        "producer_phase1_identity_sha256": phase1["identity_sha256"],
    }
    return result


def _best_cell(scale: str, producer_root: Path, ladder: dict[str, Any], probes) -> dict[str, Any]:
    scale_report = ladder["scales"][scale]
    if scale_report["fresh_verification"]["status"] != "PASS":
        raise RecoveryError(f"{scale}: producer fresh checkpoint verification is not PASS")
    checkpoint = producer_root / "retained" / scale / "best"
    backend = load_first_party_backend(checkpoint)
    diagnostics = backend.diagnostics()
    expected_checkpoint = scale_report["checkpoints"]["best_checkpoint_id"]
    if diagnostics["checkpoint_id"] != expected_checkpoint:
        raise RecoveryError(f"{scale}: retained best checkpoint identity mismatch")
    if diagnostics["git_sha"] != ladder["source"]["git_sha"]:
        raise RecoveryError(f"{scale}: retained checkpoint producer SHA mismatch")
    if diagnostics["model_spec_sha256"] != scale_report["model"]["spec_sha256"]:
        raise RecoveryError(f"{scale}: retained checkpoint ModelSpec mismatch")
    if diagnostics["parameter_count"] != scale_report["model"]["parameter_count"]:
        raise RecoveryError(f"{scale}: retained checkpoint parameter count mismatch")
    tokenizer = backend.tokenizer
    result = _score_model(
        backend.model, tokenizer, probes, first_party_backend=backend
    )
    result["identity"] = {
        "kind": "m150_retained_best_checkpoint",
        "scale": scale,
        "checkpoint_id": diagnostics["checkpoint_id"],
        "checkpoint_step": diagnostics["step"],
        "tokens_seen": diagnostics["tokens_seen"],
        "producer_git_sha": diagnostics["git_sha"],
        "model_spec_sha256": diagnostics["model_spec_sha256"],
        "parameter_count": diagnostics["parameter_count"],
        "tokenizer_version": diagnostics["tokenizer_version"],
        "tokenizer_config_sha256": diagnostics["tokenizer_config_sha256"],
        "dataset_manifest_sha256": diagnostics["dataset_manifest_sha256"],
        "run_manifest_sha256": diagnostics["run_manifest_sha256"],
        "producer_fresh_verification_status": scale_report["fresh_verification"]["status"],
    }
    return result


def _comparisons(cells: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scale in SCALE_ORDER:
        random_summary = cells[f"{scale}_random_init"]["summary"]["overall"]
        learned_summary = cells[f"{scale}_best"]["summary"]["overall"]
        result[scale] = {
            "learned_minus_random_raw_accuracy": (
                learned_summary["raw_accuracy"] - random_summary["raw_accuracy"]
            ),
            "learned_minus_random_byte_normalized_accuracy": (
                learned_summary["byte_normalized_accuracy"]
                - random_summary["byte_normalized_accuracy"]
            ),
            "learned_minus_random_mean_correct_bits_per_source_byte": (
                learned_summary["mean_correct_bits_per_source_byte"]
                - random_summary["mean_correct_bits_per_source_byte"]
            ),
            "negative_bits_per_source_byte_delta_means_lower_correct_completion_nll": True,
        }
    return result


def run(
    repo: Path,
    producer_dir: Path,
    producer_sha: str,
    producer_run_id: int,
    producer_artifact_id: int | None,
    producer_artifact_digest: str | None,
    output: Path,
) -> dict[str, Any]:
    evaluator_sha = _git_head(repo)
    producer_root = _artifact_root(producer_dir)
    ladder_path = producer_root / "ladder-report.json"
    ladder = validate_ladder(ladder_path, producer_sha)
    if ladder["truth_model"]["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise RecoveryError("producer ladder corpus is not DATA-25 identity")
    reservation = verify_reservation_against_data25(repo)
    if reservation["corpus_identity_sha256"] != ladder["truth_model"]["corpus_identity_sha256"]:
        raise RecoveryError("reservation scan corpus identity differs from M150 producer")
    probes = load_suite(repo / SUITE_PATH)
    tokenizer = ByteTokenizer()
    cells: dict[str, Any] = {}
    for scale in SCALE_ORDER:
        cells[f"{scale}_random_init"] = _random_cell(scale, producer_root, ladder, probes)
        cells[f"{scale}_best"] = _best_cell(scale, producer_root, ladder, probes)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "producer": {
            "repository": ladder["source"]["repository"],
            "git_sha": producer_sha,
            "workflow_run_id": producer_run_id,
            "artifact_id": producer_artifact_id,
            "artifact_digest": producer_artifact_digest,
            "ladder_report_sha256": ladder["report_sha256"],
            "corpus_identity_sha256": ladder["truth_model"]["corpus_identity_sha256"],
            "evaluation_identity_sha256": ladder["truth_model"]["evaluation_identity"][
                "identity_sha256"
            ],
            "m150_minimum_comparable_ladder_complete": ladder[
                "minimum_comparable_ladder_complete"
            ],
            "m150_quality_rank": ladder["rankings"]["quality"],
            "m150_efficiency_rank": ladder["rankings"]["efficiency"],
            "m150_scaling_improvement": ladder["rankings"]["scaling_improvement"],
        },
        "evaluator": {
            "git_sha": evaluator_sha,
            "worker_id": "RECOVER-177-EVAL134-CODE",
            "execution_class": "LOCAL_FREE_GITHUB_HOSTED_CPU",
        },
        "suite_and_decontamination": reservation,
        "tokenizer_comparison": {
            "canonical_byte": _byte_fragmentation(probes, tokenizer),
            "selected_learned_tokenizer_model_likelihood": {
                "status": "NOT_APPLICABLE_INCOMPATIBLE_WITH_M150_CHECKPOINTS",
                "reason": (
                    "M150 retained checkpoints bind s0-byte-v1 with vocab_size=256. "
                    "The EVAL-134 origin selected learned ByteLevel BPE control is a distinct "
                    "vocab/model geometry, so its token IDs cannot be applied to these weights."
                ),
                "foreign_weight_or_embedding_remap_attempted": False,
                "unsupported_numeric_comparison_absent": True,
            },
        },
        "evaluations": cells,
        "learned_vs_random": _comparisons(cells),
        "ten_million": {
            "status": "NOT_EVALUATED_NO_COMPARABLE_LEARNED_10M_CHECKPOINT",
            "producer_status": ladder["ten_million"]["status"],
            "numeric_results_absent": True,
        },
        "claims": {
            "code_instruction_following": False,
            "code_generation_capability": False,
            "intelligence": False,
            "production_readiness": False,
            "alignment": False,
            "foreign_pretrained_weights": False,
            "sft": False,
            "rlhf": False,
            "dpo": False,
            "paid_compute": False,
            "representative_code_corpus": False,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    _write_json(output, report)
    return report


def validate_report(path: Path, expected_evaluator_sha: str | None = None) -> dict[str, Any]:
    report = _read_json(path)
    expected_hash = report.pop("report_sha256", None)
    actual_hash = canonical_json_sha256(report)
    report["report_sha256"] = expected_hash
    if expected_hash != actual_hash:
        raise RecoveryError("recovery report self-hash mismatch")
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise RecoveryError("recovery report schema/authority mismatch")
    if expected_evaluator_sha and report["evaluator"]["git_sha"] != expected_evaluator_sha:
        raise RecoveryError("evaluator source SHA mismatch")
    if report["suite_and_decontamination"]["training_overlap_count"] != 0:
        raise RecoveryError("raw DATA-25 contamination overlap present")
    if report["suite_and_decontamination"]["normalized_training_overlap_count"] != 0:
        raise RecoveryError("normalized DATA-25 contamination overlap present")
    for scale in SCALE_ORDER:
        for role in ("random_init", "best"):
            cell = report["evaluations"][f"{scale}_{role}"]
            if cell["non_mutation"]["passed"] is not True:
                raise RecoveryError(f"{scale}/{role}: non-mutation proof missing")
        if report["evaluations"][f"{scale}_best"]["first_party_logits_check"]["finite"] is not True:
            raise RecoveryError(f"{scale}: first-party logits proof missing")
    if report["ten_million"]["numeric_results_absent"] is not True:
        raise RecoveryError("10M numeric results cannot exist without comparable checkpoint")
    if report["tokenizer_comparison"]["selected_learned_tokenizer_model_likelihood"][
        "status"
    ] != "NOT_APPLICABLE_INCOMPATIBLE_WITH_M150_CHECKPOINTS":
        raise RecoveryError("learned tokenizer compatibility boundary weakened")
    for key, value in report["claims"].items():
        if value is not False:
            raise RecoveryError(f"unsupported claim enabled: {key}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify-reservation")
    verify.add_argument("--repo-root", type=Path, default=Path("."))

    execute = sub.add_parser("run")
    execute.add_argument("--repo-root", type=Path, default=Path("."))
    execute.add_argument("--producer-dir", type=Path, required=True)
    execute.add_argument("--producer-sha", required=True)
    execute.add_argument("--producer-run-id", type=int, required=True)
    execute.add_argument("--producer-artifact-id", type=int)
    execute.add_argument("--producer-artifact-digest")
    execute.add_argument("--output", type=Path, required=True)

    check = sub.add_parser("validate")
    check.add_argument("report", type=Path)
    check.add_argument("--expected-evaluator-sha")

    args = parser.parse_args(argv)
    if args.command == "verify-reservation":
        result = verify_reservation_against_data25(args.repo_root.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "run":
        result = run(
            args.repo_root.resolve(),
            args.producer_dir.resolve(),
            args.producer_sha,
            args.producer_run_id,
            args.producer_artifact_id,
            args.producer_artifact_digest,
            args.output.resolve(),
        )
        print(
            json.dumps(
                {
                    "report_sha256": result["report_sha256"],
                    "producer_ladder_report_sha256": result["producer"][
                        "ladder_report_sha256"
                    ],
                    "training_overlap_count": result["suite_and_decontamination"][
                        "training_overlap_count"
                    ],
                    "ten_million": result["ten_million"]["status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        result = validate_report(args.report, args.expected_evaluator_sha)
        print(
            json.dumps(
                {"validation": "PASS", "report_sha256": result["report_sha256"]},
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
