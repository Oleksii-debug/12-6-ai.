"""Independent scientific verifier for the retained SCALE-141 learned ~10M Base.

The verifier consumes one exact immutable GitHub Actions artifact, rebuilds DATA-25
from repository bytes, independently re-evaluates the retained best/final checkpoints,
and emits the fail-closed authority contract consumed by RUNTIME-225.

It performs no training, optimizer update, checkpoint write, external-LLM call, or
paid-compute action.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import hash_json, verify_checkpoint
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate
from twelve_six.tokenization import ByteTokenizer

WORKER_ID = "VERIFY-218-LEARNED-10M-INDEPENDENT"
SCHEMA = "12-6.verify218-learned-10m-authority.v1"
STATUS = "VERIFIED_LEARNED_10M"

SOURCE_ARTIFACT_ID = 9_602_907_196
SOURCE_ARTIFACT_NAME = "scale141-10m-learned-fallback"
SOURCE_ARTIFACT_DIGEST = (
    "sha256:d2abd029f64207567a1d6b4ce9943ff15bfd211acdd05e9ff84156ce66607218"
)
SOURCE_WORKFLOW_RUN_ID = 32_952_786_715
SOURCE_SHA = "c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef"

EXPECTED_PARAMETER_COUNT = 10_000_640
EXPECTED_MODEL_SPEC_SHA256 = (
    "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
)
EXPECTED_INIT_SPEC_SHA256 = (
    "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
)
EXPECTED_TOKENIZER_VERSION = "s0-byte-v1"
EXPECTED_TOKENIZER_CONFIG_SHA256 = (
    "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
)
EXPECTED_TOKENIZER_VOCAB_SHA256 = (
    "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
)
EXPECTED_CORPUS_SHA256 = (
    "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
)
EXPECTED_COMMON_EVAL_SHA256 = (
    "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
)
EXPECTED_RUN_AUTHORITY = "LOCAL_FREE_LEARNED_10M_EXPERIMENT_NOT_STAGE_PROMOTION"
BEST_TARGET = 1_000_000
FINAL_TARGET = 2_000_000
EVAL_TOLERANCE = 1e-6

PROMPTS = {
    "uk": "Українська мова ",
    "en": "The training corpus ",
    "code": "def stable_",
}


class Verify218Error(RuntimeError):
    """Raised whenever the candidate cannot be independently admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Verify218Error(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Verify218Error(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise Verify218Error(f"expected JSON object: {path}")
    return value


def _verify_self_hash(value: dict[str, Any], field: str, label: str) -> str:
    supplied = value.get(field)
    _require(isinstance(supplied, str) and len(supplied) == 64, f"{label} hash missing")
    payload = dict(value)
    payload.pop(field, None)
    _require(hash_json(payload) == supplied, f"{label} self-hash mismatch")
    return supplied


def _tree_digest(root: Path) -> str:
    _require(root.is_dir(), f"checkpoint directory missing: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    _require(bool(files), f"checkpoint directory is empty: {root}")
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_source_metadata(
    *,
    artifact_id: int,
    artifact_name: str,
    artifact_digest: str,
    workflow_run_id: int,
    source_sha: str,
) -> None:
    expected = {
        "artifact_id": SOURCE_ARTIFACT_ID,
        "artifact_name": SOURCE_ARTIFACT_NAME,
        "artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "workflow_run_id": SOURCE_WORKFLOW_RUN_ID,
        "source_sha": SOURCE_SHA,
    }
    actual = {
        "artifact_id": artifact_id,
        "artifact_name": artifact_name,
        "artifact_digest": artifact_digest,
        "workflow_run_id": workflow_run_id,
        "source_sha": source_sha,
    }
    _require(actual == expected, "source artifact transport identity mismatch")


def _validate_checkpoint_identity(
    checked: dict[str, Any],
    *,
    role: str,
    expected_target: int,
) -> dict[str, Any]:
    identity = checked["identity"]
    _require(identity["git_sha"] == SOURCE_SHA, f"{role} source SHA mismatch")
    _require(
        identity["model_spec_hash"] == EXPECTED_MODEL_SPEC_SHA256,
        f"{role} ModelSpec mismatch",
    )
    _require(
        int(identity["parameter_count"]) == EXPECTED_PARAMETER_COUNT,
        f"{role} parameter count mismatch",
    )
    _require(
        identity["tokenizer_hash"] == EXPECTED_TOKENIZER_CONFIG_SHA256,
        f"{role} tokenizer config mismatch",
    )
    _require(
        identity["tokenizer_vocab_hash"] == EXPECTED_TOKENIZER_VOCAB_SHA256,
        f"{role} tokenizer vocab mismatch",
    )
    _require(
        identity["dataset_manifest_hash"] == EXPECTED_CORPUS_SHA256,
        f"{role} corpus identity mismatch",
    )
    _require(identity["precision"] == "fp32", f"{role} precision mismatch")
    _require(int(identity["step"]) > 1, f"{role} is a one-step smoke checkpoint")
    _require(
        int(identity["tokens_seen"]) >= expected_target,
        f"{role} learned exposure is below retained target",
    )
    return identity


def _compare_evaluation(
    fresh: dict[str, Any],
    recorded: dict[str, Any],
    *,
    role: str,
) -> None:
    for field in ("loss", "bits_per_byte"):
        delta = abs(float(fresh[field]) - float(recorded[field]))
        _require(delta <= EVAL_TOLERANCE, f"{role} held-out {field} mismatch: {delta}")
    _require(
        int(fresh["predicted_byte_tokens"]) == int(recorded["predicted_byte_tokens"]),
        f"{role} held-out token count mismatch",
    )
    _require(fresh["non_mutation_passed"] is True, f"{role} evaluation mutated model")
    for stratum in ("uk", "en", "code"):
        fresh_row = fresh["by_stratum"][stratum]
        recorded_row = recorded["by_stratum"][stratum]
        for field in ("loss", "bits_per_byte"):
            delta = abs(float(fresh_row[field]) - float(recorded_row[field]))
            _require(
                delta <= EVAL_TOLERANCE,
                f"{role}/{stratum} held-out {field} mismatch: {delta}",
            )
        _require(
            int(fresh_row["predicted_byte_tokens"])
            == int(recorded_row["predicted_byte_tokens"]),
            f"{role}/{stratum} token count mismatch",
        )


def _logits_snapshot(backend: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, prompt in PROMPTS.items():
        input_ids = backend.encode(prompt)
        logits = [float(value) for value in backend.next_token_logits(input_ids)]
        _require(len(logits) == 256, f"{name} logits vocab dimension mismatch")
        _require(all(math.isfinite(value) for value in logits), f"{name} logits are non-finite")
        encoded = json.dumps(logits, separators=(",", ":"), allow_nan=False).encode("utf-8")
        result[name] = {
            "input_ids": input_ids,
            "argmax_token_id": max(range(len(logits)), key=logits.__getitem__),
            "logits_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return result


def _greedy_snapshot(backend: Any) -> dict[str, Any]:
    cfg = GenerationConfig(max_new_tokens=64, sample=False)
    result: dict[str, Any] = {}
    for name, prompt in PROMPTS.items():
        generated = generate(backend, prompt, cfg)
        result[name] = {
            "generated_token_ids": list(generated.generated_token_ids),
            "text": generated.text,
            "stop_reason": generated.stop_reason,
        }
    return result


def _evaluate_checkpoint(
    checkpoint: Path,
    corpus: Path,
    manifest: dict[str, Any],
    tokenizer: ByteTokenizer,
    recorded: dict[str, Any],
    *,
    role: str,
    exercise_generation: bool,
) -> dict[str, Any]:
    backend = load_first_party_backend(
        checkpoint,
        expected_model_spec_sha256=EXPECTED_MODEL_SPEC_SHA256,
    )
    diagnostics = backend.diagnostics()
    _require(
        diagnostics["parameter_count"] == EXPECTED_PARAMETER_COUNT,
        f"{role} loaded parameter count mismatch",
    )
    _require(
        diagnostics["model_spec_sha256"] == EXPECTED_MODEL_SPEC_SHA256,
        f"{role} loaded ModelSpec mismatch",
    )
    _require(
        diagnostics["tokenizer_version"] == EXPECTED_TOKENIZER_VERSION,
        f"{role} loaded tokenizer mismatch",
    )

    model_hash_before = m100._state_hash(backend.model)
    fresh_eval = m100._evaluate(backend.model, corpus, manifest, tokenizer)
    _compare_evaluation(fresh_eval, recorded, role=role)

    output: dict[str, Any] = {
        "heldout": fresh_eval,
        "model_state_sha256_before": model_hash_before,
    }
    if exercise_generation:
        logits_a = _logits_snapshot(backend)
        logits_b = _logits_snapshot(backend)
        _require(logits_a == logits_b, "first-party logits are not deterministic")
        generation_a = _greedy_snapshot(backend)
        generation_b = _greedy_snapshot(backend)
        _require(generation_a == generation_b, "greedy generation is not deterministic")
        output["first_party_logits"] = logits_a
        output["greedy_generation"] = generation_a

    model_hash_after = m100._state_hash(backend.model)
    _require(model_hash_before == model_hash_after, f"{role} verification mutated model weights")
    output["model_state_sha256_after"] = model_hash_after
    del backend
    gc.collect()
    return output


def verify(
    *,
    repo: Path,
    artifact_root: Path,
    work_dir: Path,
    verifier_source_sha: str,
    source_artifact_id: int,
    source_artifact_name: str,
    source_artifact_digest: str,
    source_workflow_run_id: int,
    source_sha: str,
) -> dict[str, Any]:
    _validate_source_metadata(
        artifact_id=source_artifact_id,
        artifact_name=source_artifact_name,
        artifact_digest=source_artifact_digest,
        workflow_run_id=source_workflow_run_id,
        source_sha=source_sha,
    )
    _require(len(verifier_source_sha) == 40, "verifier source SHA must be full 40-hex")
    _require(all(char in "0123456789abcdef" for char in verifier_source_sha), "invalid verifier SHA")

    report = _read_json(artifact_root / "report.json")
    fresh = _read_json(artifact_root / "fresh-verification.json")
    phase1 = _read_json(artifact_root / "phase1.json")
    run_manifest = _read_json(artifact_root / "run-manifest.json")
    source_corpus = _read_json(artifact_root / "corpus-manifest.json")
    retained = _read_json(artifact_root / "retained/index.json")

    report_hash = _verify_self_hash(report, "report_sha256", "SCALE-141 report")
    fresh_hash = _verify_self_hash(fresh, "identity_sha256", "fresh verification")
    phase1_hash = _verify_self_hash(phase1, "identity_sha256", "phase1")
    run_hash = _verify_self_hash(run_manifest, "identity_sha256", "run manifest")

    for label, value in (
        ("report", report),
        ("fresh verification", fresh),
        ("phase1", phase1),
        ("run manifest", run_manifest),
        ("retained index", retained),
    ):
        _require(value["source_sha"] == SOURCE_SHA, f"{label} source SHA mismatch")

    _require(report["authority"] == EXPECTED_RUN_AUTHORITY, "producer authority mismatch")
    _require(run_manifest["authority"] == EXPECTED_RUN_AUTHORITY, "run authority mismatch")
    _require(run_manifest["foreign_pretrained_weights"] is False, "foreign weights detected")
    _require(run_manifest["instruction_tuning"] is False, "instruction tuning contaminated Base")
    _require(run_manifest["paid_compute"] is False, "producer claims paid compute")
    _require(report["success"]["paid_compute"] is False, "report claims paid compute")
    _require(report["success"]["fresh_process_resume"] is True, "fresh resume did not pass")
    _require(
        report["success"]["heldout_metric_rechecked_before_continuation"] is True,
        "held-out reload metric was not rechecked",
    )
    _require(report["corpus"]["corpus_replay"] is False, "corpus replay detected")
    _require(
        int(report["corpus"]["optimized_tokens"]) >= FINAL_TARGET,
        "producer exposure is below learned-10M campaign target",
    )
    _require(
        report["model"]["model_spec_sha256"] == EXPECTED_MODEL_SPEC_SHA256,
        "report ModelSpec mismatch",
    )
    _require(
        int(report["model"]["parameter_count"]) == EXPECTED_PARAMETER_COUNT,
        "report parameter count mismatch",
    )
    _require(
        report["model"]["init_spec_sha256"] == EXPECTED_INIT_SPEC_SHA256,
        "report InitSpec mismatch",
    )
    _require(
        report["tokenizer"] == {
            "version": EXPECTED_TOKENIZER_VERSION,
            "config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
            "vocab_sha256": EXPECTED_TOKENIZER_VOCAB_SHA256,
            "vocab_size": 256,
        },
        "report tokenizer mismatch",
    )
    _require(
        report["corpus"]["identity_sha256"] == EXPECTED_CORPUS_SHA256,
        "report corpus identity mismatch",
    )

    _require(fresh["fresh_verification"]["status"] == "PASS", "producer fresh verify not PASS")
    common = fresh["ladder_common_evaluation"]
    _require(
        common["identity"]["identity_sha256"] == EXPECTED_COMMON_EVAL_SHA256,
        "common evaluation identity mismatch",
    )
    _require(
        int(fresh["best_target_optimized_tokens"]) == BEST_TARGET,
        "retained best target mismatch",
    )
    _require(
        int(fresh["final_target_optimized_tokens"]) == FINAL_TARGET,
        "retained final target mismatch",
    )

    tokenizer = ByteTokenizer()
    _require(tokenizer.identity.version == EXPECTED_TOKENIZER_VERSION, "tokenizer version drift")
    _require(
        tokenizer.identity.config_sha256 == EXPECTED_TOKENIZER_CONFIG_SHA256,
        "tokenizer config drift",
    )
    _require(
        tokenizer.identity.vocab_sha256 == EXPECTED_TOKENIZER_VOCAB_SHA256,
        "tokenizer vocab drift",
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    rebuilt_corpus = m100._build_corpus(repo, work_dir)
    _require(rebuilt_corpus == source_corpus, "independent DATA-25 rebuild differs from artifact")
    _require(
        rebuilt_corpus["corpus_identity_sha256"] == EXPECTED_CORPUS_SHA256,
        "independent DATA-25 identity mismatch",
    )

    best_path = artifact_root / "retained/best"
    final_path = artifact_root / "retained/final"
    best_tree_before = _tree_digest(best_path)
    final_tree_before = _tree_digest(final_path)
    best_checked = verify_checkpoint(best_path)
    final_checked = verify_checkpoint(final_path)
    best_identity = _validate_checkpoint_identity(
        best_checked,
        role="best",
        expected_target=BEST_TARGET,
    )
    final_identity = _validate_checkpoint_identity(
        final_checked,
        role="final",
        expected_target=FINAL_TARGET,
    )
    _require(
        int(final_identity["step"]) > int(best_identity["step"]),
        "final checkpoint does not advance optimizer step",
    )
    _require(
        int(final_identity["tokens_seen"]) > int(best_identity["tokens_seen"]),
        "final checkpoint does not advance optimized exposure",
    )

    best_role = retained["roles"]["best"]
    final_role = retained["roles"]["final"]
    _require(best_role["checkpoint_id"] == best_checked["checkpoint_id"], "best role mismatch")
    _require(final_role["checkpoint_id"] == final_checked["checkpoint_id"], "final role mismatch")
    _require(int(best_role["target_optimized_tokens"]) == BEST_TARGET, "best target mismatch")
    _require(int(final_role["target_optimized_tokens"]) == FINAL_TARGET, "final target mismatch")
    _require(best_role["fresh_verification"] == "PASS", "best role not freshly verified")
    _require(final_role["fresh_verification"] == "PASS", "final role not freshly verified")

    resume = report["fresh_process_resume"]
    phase_recovery = phase1["recovery_resume"]
    _require(int(phase1["process"]["pid"]) == int(resume["phase1_pid"]), "phase1 PID mismatch")
    _require(int(resume["resume_pid"]) != int(resume["phase1_pid"]), "resume was not fresh process")
    _require(int(resume["loaded_step"]) == int(best_identity["step"]), "resume step mismatch")
    _require(
        int(resume["loaded_optimized_tokens"]) == int(best_identity["tokens_seen"]),
        "resume tokens mismatch",
    )
    _require(
        int(resume["first_resumed_step"]) == int(resume["loaded_step"]) + 1,
        "resume did not continue at the next optimizer step",
    )
    _require(resume["metric_recheck_passed"] is True, "resume metric recheck failed")
    _require(
        abs(float(resume["heldout_bpb_after_reload"]) - float(resume["heldout_bpb_before_stop"]))
        <= EVAL_TOLERANCE,
        "resume held-out metric changed across reload",
    )
    _require(
        phase_recovery["checkpoint_id"] == best_checked["checkpoint_id"],
        "phase1 recovery checkpoint is not retained best",
    )
    _require(int(phase_recovery["optimizer_step"]) == int(best_identity["step"]), "recovery step mismatch")
    _require(
        int(phase_recovery["tokens_seen"]) == int(best_identity["tokens_seen"]),
        "recovery tokens mismatch",
    )

    corpus_path = work_dir / "corpus-a"
    best_recorded = common["all_scheduled"][str(BEST_TARGET)]["evaluation"]
    final_recorded = common["all_scheduled"][str(FINAL_TARGET)]["evaluation"]
    best_science = _evaluate_checkpoint(
        best_path,
        corpus_path,
        rebuilt_corpus,
        tokenizer,
        best_recorded,
        role="best",
        exercise_generation=True,
    )
    final_science = _evaluate_checkpoint(
        final_path,
        corpus_path,
        rebuilt_corpus,
        tokenizer,
        final_recorded,
        role="final",
        exercise_generation=False,
    )

    best_bpb = float(best_science["heldout"]["bits_per_byte"])
    final_bpb = float(final_science["heldout"]["bits_per_byte"])
    initial_bpb = float(common["initial_bits_per_byte"])
    _require(best_bpb < initial_bpb, "retained best does not improve over random-init baseline")
    _require(best_bpb <= final_bpb + EVAL_TOLERANCE, "retained best/final role resolution failed")

    _require(_tree_digest(best_path) == best_tree_before, "best checkpoint bytes mutated")
    _require(_tree_digest(final_path) == final_tree_before, "final checkpoint bytes mutated")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "worker_id": WORKER_ID,
        "status": STATUS,
        "verified_learned_10m": True,
        "verifier_source_sha": verifier_source_sha,
        "foreign_pretrained_weights": False,
        "mechanics_only_checkpoint": False,
        "one_step_smoke": False,
        "paid_compute": False,
        "training_performed_by_verifier": False,
        "gates": {
            "checkpoint_integrity": True,
            "fresh_process_resume": True,
            "finite_first_party_logits": True,
            "heldout_bpb": True,
            "evaluation_non_mutation": True,
            "greedy_generation": True,
            "best_final_role_resolution": True,
        },
        "model": {
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "init_spec_sha256": EXPECTED_INIT_SPEC_SHA256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "tokenizer": {
            "version": EXPECTED_TOKENIZER_VERSION,
            "config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
            "vocab_sha256": EXPECTED_TOKENIZER_VOCAB_SHA256,
        },
        "corpus_identity_sha256": EXPECTED_CORPUS_SHA256,
        "source": {
            "artifact_id": SOURCE_ARTIFACT_ID,
            "artifact_name": SOURCE_ARTIFACT_NAME,
            "artifact_digest": SOURCE_ARTIFACT_DIGEST,
            "workflow_run_id": SOURCE_WORKFLOW_RUN_ID,
            "source_sha": SOURCE_SHA,
        },
        "checkpoint": {
            "role": "best",
            "checkpoint_id": best_checked["checkpoint_id"],
            "step": int(best_identity["step"]),
            "tokens_seen": int(best_identity["tokens_seen"]),
        },
        "final_checkpoint": {
            "checkpoint_id": final_checked["checkpoint_id"],
            "step": int(final_identity["step"]),
            "tokens_seen": int(final_identity["tokens_seen"]),
        },
        "independent_scientific_verification": {
            "common_evaluation_identity_sha256": EXPECTED_COMMON_EVAL_SHA256,
            "best_heldout_bits_per_byte": best_bpb,
            "final_heldout_bits_per_byte": final_bpb,
            "random_init_reference_bits_per_byte": initial_bpb,
            "best": best_science,
            "final": final_science,
            "fresh_process_resume": {
                "phase1_pid": int(resume["phase1_pid"]),
                "resume_pid": int(resume["resume_pid"]),
                "loaded_step": int(resume["loaded_step"]),
                "first_resumed_step": int(resume["first_resumed_step"]),
                "loaded_optimized_tokens": int(resume["loaded_optimized_tokens"]),
                "metric_recheck_passed": True,
            },
            "checkpoint_tree_sha256": {
                "best": best_tree_before,
                "final": final_tree_before,
            },
        },
        "source_evidence_bindings": {
            "report_sha256": report_hash,
            "fresh_verification_sha256": fresh_hash,
            "phase1_sha256": phase1_hash,
            "run_manifest_sha256": run_hash,
        },
        "truth_boundary": {
            "independent_worker": True,
            "source_substitution_allowed": False,
            "external_llm_used": False,
            "foreign_pretrained_weights": False,
            "instruction_tuning": False,
            "paid_compute": False,
            "stage_promotion_authority": False,
            "claim": "scientific admission of the exact retained learned ~10M Base only",
        },
    }
    result["identity_sha256"] = hash_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier-source-sha", required=True)
    parser.add_argument("--source-artifact-id", type=int, required=True)
    parser.add_argument("--source-artifact-name", required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    parser.add_argument("--source-workflow-run-id", type=int, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    result = verify(
        repo=args.repo_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        work_dir=args.work_dir.resolve(),
        verifier_source_sha=args.verifier_source_sha,
        source_artifact_id=args.source_artifact_id,
        source_artifact_name=args.source_artifact_name,
        source_artifact_digest=args.source_artifact_digest,
        source_workflow_run_id=args.source_workflow_run_id,
        source_sha=args.source_sha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checkpoint_id": result["checkpoint"]["checkpoint_id"],
                "best_heldout_bits_per_byte": result["independent_scientific_verification"][
                    "best_heldout_bits_per_byte"
                ],
                "final_heldout_bits_per_byte": result["independent_scientific_verification"][
                    "final_heldout_bits_per_byte"
                ],
                "identity_sha256": result["identity_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
