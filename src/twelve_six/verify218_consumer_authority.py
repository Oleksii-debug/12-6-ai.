"""Converge VERIFY-218 scientific evidence with the maintained RUNTIME-225 consumer.

This module is verification-only. It reuses the independent VERIFY-218 artifact
consumer, adds a real fresh D02/D05 resume-load proof from the retained recovery
checkpoints, and emits the exact fail-closed authority schema consumed downstream.
It performs no optimizer update, backward pass, checkpoint save, or training run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six import scale141_10m_continuation as core
from twelve_six import scale141_10m_runtime_v2 as v2
from twelve_six import scale141_10m_runtime_v3 as v3
from twelve_six.checkpoint import hash_json, load_trainer_checkpoint, verify_checkpoint
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer
from twelve_six.verify218_learned_10m import (
    EXPECTED_CORPUS_ID,
    EXPECTED_MODEL_SPEC_SHA256,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKENIZER_VERSION,
    PRODUCER_ARTIFACT_ID,
    PRODUCER_ARTIFACT_ZIP_SHA256,
    PRODUCER_SHA,
    STATE,
    WORKER,
    Verify218Error,
    verify as verify_rich,
)

SCHEMA = "12-6.verify218-learned-10m-independent.v2"
PRODUCER_ARTIFACT_NAME = "learn217-terminal-10m-learned-base"
PRODUCER_WORKFLOW_RUN_ID = 32952787070


class Verify218ConsumerAuthorityError(RuntimeError):
    """Fail-closed convergence error for the independent 10M authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Verify218ConsumerAuthorityError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Verify218ConsumerAuthorityError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Verify218ConsumerAuthorityError(f"{path} must contain a JSON object")
    return value


def _json_normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _install_exact_runtime_contract() -> None:
    # LEARN-217 first installs v3's stable wrappers and then v2's actual-token/
    # seq-256 runtime contract. Reconstruct the same config without executing
    # either phase of training.
    v3._install()
    v2._install_runtime_contract()


def _fresh_resume_load(
    *,
    checkpoint: Path,
    run_manifest: Mapping[str, Any],
    tokenizer: ByteTokenizer,
    label: str,
) -> dict[str, Any]:
    """Load one retained recovery checkpoint into fresh model/trainer targets."""

    checked = verify_checkpoint(checkpoint)
    identity = checked.get("identity")
    _require(isinstance(identity, Mapping), f"{label} checkpoint identity missing")

    spec_raw = run_manifest.get("model_spec")
    init_raw = run_manifest.get("init_spec")
    expected_cfg = run_manifest.get("trainer_config")
    _require(isinstance(spec_raw, dict), "run manifest model_spec missing")
    _require(isinstance(init_raw, dict), "run manifest init_spec missing")
    _require(isinstance(expected_cfg, Mapping), "run manifest trainer_config missing")

    spec = ModelSpec.from_dict(spec_raw)
    init = InitSpec.from_dict(init_raw)
    _require(
        spec.identity_sha256() == EXPECTED_MODEL_SPEC_SHA256,
        f"{label} reconstructed ModelSpec mismatch",
    )
    _require(
        spec.parameter_count() == EXPECTED_PARAMETER_COUNT,
        f"{label} reconstructed parameter count mismatch",
    )

    _install_exact_runtime_contract()
    cfg = core._trainer_config()
    _require(
        _json_normalize(asdict(cfg)) == _json_normalize(dict(expected_cfg)),
        f"{label} reconstructed TrainerConfig differs from run manifest",
    )

    seed = expected_cfg.get("seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        f"{label} training seed missing",
    )
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    _require(trainer.optimizer_step == 0, f"{label} fresh trainer step is not zero")
    _require(trainer.tokens_seen == 0, f"{label} fresh trainer tokens_seen is not zero")

    result = load_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=PRODUCER_SHA,
        expected_model_spec_hash=EXPECTED_MODEL_SPEC_SHA256,
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=EXPECTED_CORPUS_ID,
        expected_run_manifest_hash=str(run_manifest["identity_sha256"]),
        expected_seed=seed,
    )
    restored_identity = result.manifest.get("identity")
    _require(
        isinstance(restored_identity, Mapping),
        f"{label} restored checkpoint identity missing",
    )
    expected_step = int(restored_identity.get("step", -1))
    expected_tokens = int(restored_identity.get("tokens_seen", -1))
    _require(expected_step > 0, f"{label} recovery checkpoint is not learned")
    _require(expected_tokens > 0, f"{label} recovery tokens_seen is not positive")
    _require(
        trainer.optimizer_step == expected_step,
        f"{label} optimizer_step was not restored exactly",
    )
    _require(
        trainer.tokens_seen == expected_tokens,
        f"{label} tokens_seen was not restored exactly",
    )
    trainer.assert_checkpoint_safe()
    state = trainer.state_dict()
    _require(
        state.optimizer_step == expected_step and state.tokens_seen == expected_tokens,
        f"{label} trainer state_dict disagrees after fresh load",
    )
    _require(bool(state.optimizer.get("state")), f"{label} optimizer state was not restored")

    return {
        "status": "PASS",
        "checkpoint_id": checked["checkpoint_id"],
        "optimizer_step": expected_step,
        "tokens_seen": expected_tokens,
        "trainer_config_match": True,
        "optimizer_state_restored": True,
        "rng_restore_requested": True,
        "checkpoint_safe_after_load": True,
        "optimizer_updates_executed": 0,
    }


def converge(
    *,
    repo: Path,
    artifact_root: Path,
    verifier_head_sha: str,
    rich_output: Path,
    output: Path,
) -> dict[str, Any]:
    """Run independent evidence checks and emit the downstream consumer authority."""

    try:
        rich = verify_rich(
            repo=repo,
            artifact_root=artifact_root,
            verifier_head_sha=verifier_head_sha,
            output_path=rich_output,
        )
    except Verify218Error as exc:
        raise Verify218ConsumerAuthorityError(str(exc)) from exc

    _require(rich.get("worker") == WORKER, "rich VERIFY-218 worker mismatch")
    _require(rich.get("state") == STATE, "rich VERIFY-218 state mismatch")

    evidence_root = artifact_root.resolve() / "scale141-evidence"
    run_manifest = _read_json(evidence_root / "run-manifest.json")
    retained_index = _read_json(evidence_root / "retained" / "index.json")
    tokenizer = ByteTokenizer()
    _require(
        tokenizer.identity.version == EXPECTED_TOKENIZER_VERSION,
        "canonical tokenizer version drift",
    )

    recovery_loads: dict[str, Any] = {}
    for label, directory, index_key in (
        ("phase1", "recovery-phase1", "phase1"),
        ("current", "recovery-current", "current"),
    ):
        proof = _fresh_resume_load(
            checkpoint=evidence_root / "retained" / directory,
            run_manifest=run_manifest,
            tokenizer=tokenizer,
            label=label,
        )
        indexed = retained_index.get("recovery", {}).get(index_key)
        _require(isinstance(indexed, Mapping), f"{label} recovery index missing")
        _require(
            indexed.get("checkpoint_id") == proof["checkpoint_id"],
            f"{label} recovery checkpoint ID differs from retained index",
        )
        _require(
            int(indexed.get("optimizer_step", -1)) == proof["optimizer_step"],
            f"{label} recovery optimizer step differs from retained index",
        )
        _require(
            int(indexed.get("tokens_seen", -1)) == proof["tokens_seen"],
            f"{label} recovery tokens_seen differs from retained index",
        )
        recovery_loads[label] = proof

    checkpoints = rich.get("checkpoints")
    _require(isinstance(checkpoints, Mapping), "rich checkpoint evidence missing")
    best = checkpoints.get("best")
    final = checkpoints.get("final")
    _require(isinstance(best, Mapping), "rich best checkpoint missing")
    _require(isinstance(final, Mapping), "rich final checkpoint missing")
    best_identity = best.get("identity")
    final_identity = final.get("identity")
    _require(isinstance(best_identity, Mapping), "rich best identity missing")
    _require(isinstance(final_identity, Mapping), "rich final identity missing")
    best_step = int(best_identity.get("step", -1))
    best_tokens = int(best_identity.get("tokens_seen", -1))
    final_step = int(final_identity.get("step", -1))
    final_tokens = int(final_identity.get("tokens_seen", -1))
    _require(best_step > 0 and best_tokens > 0, "best checkpoint is not learned")
    _require(final_step >= best_step, "final checkpoint predates best checkpoint")
    _require(final_tokens >= best_tokens, "final checkpoint tokens predate best checkpoint")

    data_and_eval = rich.get("data_and_eval")
    _require(isinstance(data_and_eval, Mapping), "rich evaluation evidence missing")
    _require(
        data_and_eval.get("best_improved_over_reconstructed_random_init") is True,
        "best checkpoint did not improve over reconstructed random init",
    )
    _require(
        data_and_eval.get("final_improved_over_reconstructed_random_init") is True,
        "final checkpoint did not improve over reconstructed random init",
    )

    source_digest = f"sha256:{PRODUCER_ARTIFACT_ZIP_SHA256}"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "worker_id": WORKER,
        "status": STATE,
        "verified_learned_10m": True,
        "foreign_pretrained_weights": False,
        "mechanics_only_checkpoint": False,
        "one_step_smoke": False,
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
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "tokenizer": {
            "version": tokenizer.identity.version,
            "config_sha256": tokenizer.identity.config_sha256,
            "vocab_sha256": tokenizer.identity.vocab_sha256,
        },
        "corpus_identity_sha256": EXPECTED_CORPUS_ID,
        "source": {
            "artifact_id": PRODUCER_ARTIFACT_ID,
            "artifact_name": PRODUCER_ARTIFACT_NAME,
            "artifact_digest": source_digest,
            "workflow_run_id": PRODUCER_WORKFLOW_RUN_ID,
            "source_sha": PRODUCER_SHA,
        },
        "checkpoint": {
            "role": "best",
            "checkpoint_id": best["checkpoint_id"],
            "step": best_step,
            "tokens_seen": best_tokens,
        },
        "verifier": {
            "head_sha": verifier_head_sha,
            "rich_authority_identity_sha256": rich["identity_sha256"],
        },
        "fresh_resume_loads": recovery_loads,
        "scientific_evidence": {
            "data_and_eval": data_and_eval,
            "best_checkpoint": best,
            "final_checkpoint": final,
        },
        "truth_boundary": {
            "training_executed": False,
            "optimizer_updates": 0,
            "paid_compute": False,
            "external_llm_used": False,
            "instruction_or_behavioral_alignment": False,
            "data25_external_real_or_representative_claim": False,
            "broad_capability_or_production_readiness_claim": False,
        },
    }
    result["identity_sha256"] = hash_json(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--verifier-head-sha", required=True)
    parser.add_argument("--rich-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = converge(
        repo=args.repo_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        verifier_head_sha=args.verifier_head_sha,
        rich_output=args.rich_output.resolve(),
        output=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "worker_id": result["worker_id"],
                "status": result["status"],
                "checkpoint_id": result["checkpoint"]["checkpoint_id"],
                "identity_sha256": result["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
