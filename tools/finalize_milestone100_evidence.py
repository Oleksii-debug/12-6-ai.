#!/usr/bin/env python3
"""Fail-closed convergence verdict for the first learned 12-6 Base milestone."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _hash(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _avg(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "evidence/milestone100"
    corpus = _load(evidence / "corpus_report.json")
    baseline = _load(evidence / "research41_real_corpus_baseline.json")
    learned = _load(evidence / "learned_468k_real_corpus.json")
    resume = _load(evidence / "fresh_process_resume.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    if learned["source"]["git_sha"] != head or baseline["source"]["git_sha"] != head:
        raise RuntimeError("training evidence is not bound to the current exact Git head")
    if int(learned["parameters"]) != 467_808:
        raise RuntimeError("exact 467,808-parameter milestone geometry drifted")
    if resume.get("status") != "PASS" or resume.get("evaluation_non_mutating") is not True:
        raise RuntimeError("fresh-process resume/evaluation proof did not pass")

    manifest = corpus["manifest"]
    if manifest["synthetic_training_records"] != 0 or manifest["contains_foreign_pretrained_weights"] is not False:
        raise RuntimeError("milestone corpus/initialization truth boundary was weakened")
    if manifest["representativeness"]["intended_modalities_present"] != ["uk", "en", "code"]:
        raise RuntimeError("bounded UK/EN/code modality coverage is incomplete")

    seed_proofs = []
    initial_hashes = set()
    for run in learned["seed_runs"]:
        initial_hashes.add(run["initial_model_state_sha256"])
        train_losses = [float(point["train_loss"]) for point in run["train_curve"]]
        window = min(10, max(1, len(train_losses) // 4))
        initial_train_mean = _avg(train_losses[:window])
        final_train_mean = _avg(train_losses[-window:])
        heldout_initial = run["held_out_curve"][0]
        heldout_final = run["held_out_curve"][-1]
        checkpoints = run["checkpoints"]
        if len(checkpoints) < 3:
            raise RuntimeError("fewer than three retained checkpoints")
        if final_train_mean >= initial_train_mean:
            raise RuntimeError(f"seed {run['seed']}: train loss did not decrease")
        if float(heldout_final["validation_bpb"]) >= float(heldout_initial["validation_bpb"]):
            raise RuntimeError(f"seed {run['seed']}: held-out BPB did not decrease")
        if not all(
            proof.get("model_state_equal")
            and proof.get("trainer_state_equal")
            and proof.get("validation_loss_equal")
            and proof.get("generation_equal")
            and proof.get("trainer_counters_equal")
            for proof in run["checkpoint_reload_equality"]
        ):
            raise RuntimeError(f"seed {run['seed']}: checkpoint reload equality failed")
        seed_proofs.append(
            {
                "seed": run["seed"],
                "initial_model_state_sha256": run["initial_model_state_sha256"],
                "final_model_state_sha256": run["final_model_state_sha256"],
                "initial_train_loss_window_mean": initial_train_mean,
                "final_train_loss_window_mean": final_train_mean,
                "train_loss_decreased": True,
                "initial_validation_loss": heldout_initial["validation_loss"],
                "final_validation_loss": heldout_final["validation_loss"],
                "initial_validation_bpb": heldout_initial["validation_bpb"],
                "final_validation_bpb": heldout_final["validation_bpb"],
                "held_out_bpb_decreased": True,
                "optimized_tokens": run["optimized_tokens"],
                "optimizer_steps": run["optimizer_steps"],
                "checkpoint_count": len(checkpoints),
                "checkpoint_reload_equality": True,
                "generation_before_training": run["generation_snapshots"][0],
                "generation_after_training": run["generation_snapshots"][-1],
                "peak_rss_bytes": run["peak_rss_bytes"],
                "end_to_end_tokens_per_second": run["end_to_end_tokens_per_second"],
            }
        )
    if len(initial_hashes) != len(learned["seed_runs"]):
        raise RuntimeError("independent random-initialization seeds produced identical initial states")

    seed1337 = next(run for run in learned["seed_runs"] if int(run["seed"]) == 1337)
    retained = next(
        item for item in seed1337["checkpoints"] if int(item["requested_token_budget"]) == 65_536
    )
    machine = {
        "schema_version": "12-6.machine-manifest.v1",
        "git_sha": head,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "runner_environment": os.environ.get("RUNNER_ENVIRONMENT", "unknown"),
        "runner_os": os.environ.get("RUNNER_OS", "unknown"),
        "paid_compute": False,
        "local_free_classification": "GITHUB_HOSTED_FREE_CURRENT_PROJECT_ALLOWANCE_OR_EQUIVALENT_NO_PURCHASE",
    }
    (evidence / "machine_manifest.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_core = {
        "schema_version": "12-6.milestone100-first-learned-base.v1",
        "status": "PASS_GENUINELY_LEARNED_BOUNDED_BASE",
        "exact_git_head": head,
        "model": {
            "incumbent": "RESEARCH41 controlled 12-6 decoder family",
            "parameters": learned["parameters"],
            "model_spec": learned["model_spec"],
            "model_identity_sha256": learned["model_identity_sha256"],
            "init_spec": learned["init_spec"],
            "init_identity_sha256": learned["init_identity_sha256"],
            "canonical_base": "random_init",
            "foreign_pretrained_weights": False,
        },
        "tokenizer": {
            "actual_training_tokenizer_id": learned["controls"]["tokenizer_id"],
            "config_sha256": learned["controls"]["tokenizer_config_sha256"],
            "vocab_sha256": learned["controls"]["tokenizer_vocab_sha256"],
            "experimental_incumbent": "D04 tokenizers 0.23.1 BPE/Unigram experiments; not substituted into this model-compatible milestone run",
        },
        "corpus": {
            "dataset_id": manifest["dataset_id"],
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
            "manifest_sha256": corpus["manifest_sha256"],
            "train_jsonl_sha256": corpus["train_jsonl_sha256"],
            "validation_jsonl_sha256": corpus["validation_jsonl_sha256"],
            "source_families": manifest["source_families"],
            "bytes_by_family": manifest["bytes_by_family"],
            "bytes_by_language_or_mode": manifest["bytes_by_language_or_mode"],
            "script_distribution": manifest["script_distribution"],
            "average_document_utf8_bytes": manifest["average_document_utf8_bytes"],
            "exact_duplicate_rate_after_filters": manifest["exact_duplicate_rate_after_filters"],
            "near_duplicate_status": manifest["near_duplicate_status"],
            "synthetic_training_records": 0,
            "bounded_small_vertical_representative": True,
            "broad_external_corpus_representative": False,
        },
        "systems_reused": {
            "streaming_packing": learned["controls"]["packing_id"],
            "trainer_optimizer": learned["controls"]["optimizer_by_seed"],
            "observability": "LEARN03 train_curve + grad_norm/update telemetry",
            "checkpoint_resume": "D05 save_trainer_checkpoint/load_trainer_checkpoint",
            "held_out_evaluation": "RESEARCH41/LEARN03 held-out byte-level validation",
            "first_party_inference": "TwelveSixDecoder generation snapshots in LEARN03",
        },
        "learning_proof": seed_proofs,
        "fresh_process_resume": resume,
        "retained_exact_checkpoint": retained,
        "machine_manifest": machine,
        "reproduction": {
            "command": "python tools/materialize_milestone100_real_corpus.py && python -m twelve_six.scaling_experiment run --repo-root . --source-sha $(git rev-parse HEAD) --output evidence/milestone100/research41_real_corpus_baseline.json --torch-threads 2 && python -m twelve_six.scaling_500k_evidence run --repo-root . --source-sha $(git rev-parse HEAD) --baseline evidence/milestone100/research41_real_corpus_baseline.json --output evidence/milestone100/learned_468k_real_corpus.json --checkpoint-root artifacts/milestone100/checkpoints --seeds 1337 1338 --token-budgets 4096 16384 65536 --torch-threads 2 && python tools/prove_milestone100_fresh_resume.py --repo-root . --source-sha $(git rev-parse HEAD) --report evidence/milestone100/learned_468k_real_corpus.json --output evidence/milestone100/fresh_process_resume.json && python tools/finalize_milestone100_evidence.py",
            "locked_environment_profile": "linux-x86_64",
        },
        "data102": corpus["data102"],
        "rejected_or_non_authoritative_evidence": [
            "DATA-24 exact head general x86 CI failed repository lint; only its exact resolver blob is reused and re-executed here.",
            "TRAIN-53 exact training workflow failed before training; its claimed trajectory is not execution authority.",
            "DATA-25 v0.1 is project-authored and is not used to claim external real-world corpus representativeness.",
            "RESEARCH41/LEARN03 inherited report fields that hard-code S0 project-authored fixture wording are rejected as provenance authority; the content-addressed M100 corpus manifest and file hashes are authoritative for this run."
        ],
        "truth_boundary": {
            "base_pretraining_only": True,
            "instruction_tuning": False,
            "broad_intelligence_claim": False,
            "broad_corpus_claim": False,
            "paid_compute": False,
            "data102_full_ua_breadth_goal_achieved": corpus["data102"]["newly_eligible_external_uk_bytes"] > 0,
            "data102_blocked_candidates_retained": True,
        },
    }
    summary = {**summary_core, "summary_sha256": _hash(summary_core)}
    (evidence / "milestone100_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
