#!/usr/bin/env python3
"""Fail-closed independent verifier for LEARN-191 and LEARN-217 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


class VerificationError(RuntimeError):
    """Raised when independent evidence cannot be verified exactly."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        raw = archive.read(name)
    except KeyError as exc:
        raise VerificationError(f"missing artifact member: {name}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON member: {name}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"JSON member is not an object: {name}")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise VerificationError(f"{label}: expected {expected!r}, got {actual!r}")


def _require_true(value: Any, label: str) -> None:
    if value is not True:
        raise VerificationError(f"{label}: expected true, got {value!r}")


def _require_false(value: Any, label: str) -> None:
    if value is not False:
        raise VerificationError(f"{label}: expected false, got {value!r}")


def _verify_checkpoint_directory(
    archive: zipfile.ZipFile,
    prefix: str,
    *,
    expected_source_sha: str,
    expected_model_spec_sha256: str,
    expected_parameter_count: int,
    expected_corpus_sha256: str,
    expected_tokenizer_config_sha256: str,
    expected_tokenizer_vocab_sha256: str,
    expected_checkpoint_id: str | None = None,
    expected_tokens_seen: int | None = None,
) -> dict[str, Any]:
    manifest = _json_member(archive, f"{prefix}manifest.json")
    identity = manifest.get("identity")
    files = manifest.get("files")
    if not isinstance(identity, dict) or not isinstance(files, dict):
        raise VerificationError(f"malformed checkpoint manifest: {prefix}")

    _require_equal(identity.get("git_sha"), expected_source_sha, f"{prefix} git_sha")
    _require_equal(
        identity.get("model_spec_hash"),
        expected_model_spec_sha256,
        f"{prefix} model_spec_hash",
    )
    _require_equal(
        identity.get("parameter_count"),
        expected_parameter_count,
        f"{prefix} parameter_count",
    )
    _require_equal(
        identity.get("dataset_manifest_hash"),
        expected_corpus_sha256,
        f"{prefix} dataset_manifest_hash",
    )
    _require_equal(
        identity.get("tokenizer_hash"),
        expected_tokenizer_config_sha256,
        f"{prefix} tokenizer_hash",
    )
    _require_equal(
        identity.get("tokenizer_vocab_hash"),
        expected_tokenizer_vocab_sha256,
        f"{prefix} tokenizer_vocab_hash",
    )
    if expected_checkpoint_id is not None:
        _require_equal(
            manifest.get("checkpoint_id"),
            expected_checkpoint_id,
            f"{prefix} checkpoint_id",
        )
    if expected_tokens_seen is not None:
        _require_equal(
            identity.get("tokens_seen"),
            expected_tokens_seen,
            f"{prefix} tokens_seen",
        )

    for filename, metadata in sorted(files.items()):
        if not isinstance(metadata, dict):
            raise VerificationError(f"invalid file metadata: {prefix}{filename}")
        try:
            payload = archive.read(f"{prefix}{filename}")
        except KeyError as exc:
            raise VerificationError(f"missing checkpoint payload: {prefix}{filename}") from exc
        _require_equal(len(payload), metadata.get("bytes"), f"{prefix}{filename} bytes")
        _require_equal(
            _sha256_bytes(payload),
            metadata.get("sha256"),
            f"{prefix}{filename} sha256",
        )
    return manifest


def verify_learn191(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["producer"]
    _require_equal(_sha256_file(path), expected["artifact_sha256"], "LEARN-191 artifact sha256")

    with zipfile.ZipFile(path) as archive:
        truth = _json_member(archive, "learn191-evidence/truth.json")
        run = _json_member(archive, "learn191-evidence/run-manifest.json")
        report = _json_member(archive, "learn191-evidence/learn191-real-3m-report.json")
        phase1 = _json_member(archive, "learn191-evidence/3m/phase1.json")
        resume = _json_member(archive, "learn191-evidence/3m/resume.json")
        final_proof = _json_member(
            archive,
            "learn191-evidence/3m/final-fresh-load-proof.json",
        )

        for label, document in (
            ("truth", truth),
            ("run manifest", run),
            ("report", report),
            ("phase1", phase1),
            ("resume", resume),
            ("fresh-load proof", final_proof),
        ):
            _require_equal(document.get("source_sha"), expected["head_sha"], f"LEARN-191 {label} source_sha")

        _require_equal(truth.get("parameter_count"), expected["parameter_count"], "LEARN-191 parameter_count")
        _require_equal(truth.get("model_spec_sha256"), expected["model_spec_sha256"], "LEARN-191 ModelSpec")
        _require_equal(truth.get("corpus_identity_sha256"), expected["corpus_sha256"], "LEARN-191 corpus")
        _require_true(truth.get("random_initialization"), "LEARN-191 random initialization")
        _require_false(truth.get("foreign_pretrained_weights"), "LEARN-191 foreign pretrained weights")
        _require_false(truth.get("paid_compute"), "LEARN-191 paid compute")
        _require_false(truth.get("sft"), "LEARN-191 SFT")

        tokenizer = run.get("tokenizer")
        if not isinstance(tokenizer, dict):
            raise VerificationError("LEARN-191 tokenizer contract missing")
        _require_equal(tokenizer.get("version"), "s0-byte-v1", "LEARN-191 tokenizer version")
        _require_equal(
            tokenizer.get("config_sha256"),
            expected["tokenizer_config_sha256"],
            "LEARN-191 tokenizer config",
        )
        _require_equal(
            tokenizer.get("vocab_sha256"),
            expected["tokenizer_vocab_sha256"],
            "LEARN-191 tokenizer vocab",
        )

        evaluations = report.get("evaluations")
        if not isinstance(evaluations, list):
            raise VerificationError("LEARN-191 evaluations missing")
        targets = [item.get("target_optimized_tokens") for item in evaluations]
        actual = [item.get("actual_optimized_tokens") for item in evaluations]
        _require_equal(targets, expected["evaluation_targets"], "LEARN-191 evaluation targets")
        _require_equal(actual, expected["evaluation_actual_tokens"], "LEARN-191 actual optimized tokens")
        for index, item in enumerate(evaluations):
            validation = item.get("selection_validation")
            train_probe = item.get("train_probe")
            if not isinstance(validation, dict) or not isinstance(train_probe, dict):
                raise VerificationError(f"LEARN-191 evaluation {index} missing probes")
            _require_true(validation.get("non_mutation_passed"), f"LEARN-191 evaluation {index} validation non-mutation")
            _require_true(train_probe.get("non_mutation_passed"), f"LEARN-191 evaluation {index} train-probe non-mutation")
            bpb = validation.get("bits_per_byte")
            if not isinstance(bpb, (int, float)) or not math.isfinite(bpb) or bpb < 0:
                raise VerificationError(f"LEARN-191 evaluation {index} invalid BPB")

        fresh_resume = report.get("fresh_process_resume")
        if not isinstance(fresh_resume, dict):
            raise VerificationError("LEARN-191 fresh-process resume evidence missing")
        _require_true(fresh_resume.get("passed"), "LEARN-191 fresh-process resume")
        if fresh_resume.get("phase1_pid") == fresh_resume.get("resume_pid"):
            raise VerificationError("LEARN-191 phase1 and resume reused a PID")
        if final_proof.get("process_pid") in {
            fresh_resume.get("phase1_pid"),
            fresh_resume.get("resume_pid"),
        }:
            raise VerificationError("LEARN-191 final fresh load reused a training PID")

        checkpoints = report.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 3:
            raise VerificationError("LEARN-191 checkpoint records must contain three scheduled rungs")
        for record, target, tokens in zip(
            checkpoints,
            expected["checkpoint_targets"],
            expected["checkpoint_actual_tokens"],
            strict=True,
        ):
            _require_equal(record.get("target_optimized_tokens"), target, "LEARN-191 checkpoint target")
            _require_equal(record.get("actual_optimized_tokens"), tokens, "LEARN-191 checkpoint actual tokens")
            prefix = f"learn191-evidence/3m/checkpoint-t{target:06d}/"
            _verify_checkpoint_directory(
                archive,
                prefix,
                expected_source_sha=expected["head_sha"],
                expected_model_spec_sha256=expected["model_spec_sha256"],
                expected_parameter_count=expected["parameter_count"],
                expected_corpus_sha256=expected["corpus_sha256"],
                expected_tokenizer_config_sha256=expected["tokenizer_config_sha256"],
                expected_tokenizer_vocab_sha256=expected["tokenizer_vocab_sha256"],
                expected_checkpoint_id=record.get("checkpoint_id"),
                expected_tokens_seen=tokens,
            )

        _require_equal(
            final_proof.get("checkpoint_id"),
            expected["final_checkpoint_id"],
            "LEARN-191 fresh-load checkpoint",
        )
        _require_equal(
            final_proof.get("optimized_tokens"),
            expected["final_actual_tokens"],
            "LEARN-191 fresh-load optimized tokens",
        )

    return {
        "verify_id": contract["verify_id"],
        "result": "PASS_SCIENTIFIC_EVIDENCE",
        "source_sha": expected["head_sha"],
        "artifact_sha256": expected["artifact_sha256"],
        "final_actual_optimized_tokens": expected["final_actual_tokens"],
        "release_ci": contract["release_ci"],
    }


def verify_learn217(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["producer"]
    _require_equal(_sha256_file(path), expected["artifact_sha256"], "LEARN-217 artifact sha256")

    with zipfile.ZipFile(path) as archive:
        preflight = _json_member(archive, "scale141-evidence/learn217-identity-preflight.json")
        run = _json_member(archive, "scale141-evidence/run-manifest.json")
        report = _json_member(archive, "scale141-evidence/report.json")
        verification = _json_member(archive, "scale141-evidence/fresh-verification.json")
        summary = _json_member(archive, "scale141-evidence/learn217-terminal-summary.json")
        retained = _json_member(archive, "scale141-evidence/retained/index.json")
        recovery = _json_member(archive, "learn217-recovery-preflight/recovery-proof.json")

        for label, document in (
            ("identity preflight", preflight),
            ("run manifest", run),
            ("report", report),
            ("fresh verification", verification),
            ("terminal summary", summary),
            ("retained index", retained),
            ("recovery proof", recovery),
        ):
            _require_equal(document.get("source_sha"), expected["head_sha"], f"LEARN-217 {label} source_sha")

        _require_equal(preflight.get("parameter_count"), expected["parameter_count"], "LEARN-217 parameter_count")
        _require_equal(preflight.get("model_spec_sha256"), expected["model_spec_sha256"], "LEARN-217 ModelSpec")
        _require_equal(preflight.get("corpus_identity_sha256"), expected["corpus_sha256"], "LEARN-217 corpus")
        _require_equal(run.get("target_optimized_tokens"), 2_000_000, "LEARN-217 target optimized tokens")
        _require_false(run.get("foreign_pretrained_weights"), "LEARN-217 foreign pretrained weights")
        _require_false(run.get("paid_compute"), "LEARN-217 paid compute")
        _require_false(run.get("instruction_tuning"), "LEARN-217 instruction tuning")

        _require_equal(summary.get("status"), "TERMINAL_LEARNED_10M_PASS", "LEARN-217 terminal status")
        _require_equal(summary.get("optimized_tokens"), expected["final_actual_tokens"], "LEARN-217 actual optimized tokens")
        _require_false(summary.get("foreign_weights"), "LEARN-217 foreign weights")
        _require_false(summary.get("paid_compute"), "LEARN-217 paid compute summary")
        _require_false(summary.get("sft_rlhf_dpo"), "LEARN-217 post-training")
        _require_equal(summary.get("fresh_verification"), "PASS", "LEARN-217 fresh verification summary")
        _require_equal(
            summary.get("ladder_common_evaluation_identity"),
            expected["common_evaluation_sha256"],
            "LEARN-217 common evaluation identity",
        )

        fresh = verification.get("fresh_verification")
        if not isinstance(fresh, dict):
            raise VerificationError("LEARN-217 fresh verification block missing")
        _require_equal(fresh.get("status"), "PASS", "LEARN-217 fresh verification status")
        for key in (
            "checkpoint_identity",
            "checkpoint_load",
            "evaluation_non_mutation",
            "first_party_logits",
            "generation",
            "m150_common_evaluation_identity",
            "reproducibility_manifest_validation",
            "best_and_final_retained",
        ):
            _require_true(fresh.get(key), f"LEARN-217 fresh verification {key}")

        resume = summary.get("fresh_process_resume")
        if not isinstance(resume, dict):
            raise VerificationError("LEARN-217 fresh-process resume evidence missing")
        if resume.get("phase1_pid") == resume.get("resume_pid"):
            raise VerificationError("LEARN-217 phase1 and resume reused a PID")
        _require_true(resume.get("metric_recheck_passed"), "LEARN-217 resume metric recheck")
        _require_equal(
            resume.get("loaded_optimized_tokens"),
            expected["phase1_actual_tokens"],
            "LEARN-217 loaded phase1 optimized tokens",
        )

        historical = recovery.get("historical_failure")
        generation = recovery.get("generation_proof")
        fresh_load = recovery.get("fresh_load")
        if not all(isinstance(item, dict) for item in (historical, generation, fresh_load)):
            raise VerificationError("LEARN-217 recovery proof incomplete")
        _require_true(historical.get("reproduced"), "LEARN-217 historical immutable overwrite failure reproduced")
        _require_true(generation.get("older_generation_bytes_unchanged"), "LEARN-217 older recovery bytes unchanged")
        _require_true(fresh_load.get("optimizer_state_restored"), "LEARN-217 optimizer restore")
        _require_true(fresh_load.get("rng_restored"), "LEARN-217 RNG restore")
        _require_true(recovery.get("corrupt_current_generation_fail_closed"), "LEARN-217 corrupt recovery rejection")
        _require_false(recovery.get("full_10m_retraining_performed"), "LEARN-217 recovery preflight full retraining")

        roles = retained.get("roles")
        recovery_roles = retained.get("recovery")
        if not isinstance(roles, dict) or not isinstance(recovery_roles, dict):
            raise VerificationError("LEARN-217 retained index incomplete")
        _require_equal(roles["best"].get("checkpoint_id"), expected["best_checkpoint_id"], "LEARN-217 best checkpoint id")
        _require_equal(roles["final"].get("checkpoint_id"), expected["final_checkpoint_id"], "LEARN-217 final checkpoint id")
        _require_equal(roles["best"].get("target_optimized_tokens"), 1_000_000, "LEARN-217 best target")
        _require_equal(roles["final"].get("target_optimized_tokens"), 2_000_000, "LEARN-217 final target")

        for role, checkpoint_id, tokens in (
            ("best", expected["best_checkpoint_id"], expected["phase1_actual_tokens"]),
            ("final", expected["final_checkpoint_id"], expected["final_actual_tokens"]),
            (
                "recovery-phase1",
                recovery_roles["phase1"]["checkpoint_id"],
                recovery_roles["phase1"]["tokens_seen"],
            ),
            (
                "recovery-current",
                recovery_roles["current"]["checkpoint_id"],
                recovery_roles["current"]["tokens_seen"],
            ),
        ):
            _verify_checkpoint_directory(
                archive,
                f"scale141-evidence/retained/{role}/",
                expected_source_sha=expected["head_sha"],
                expected_model_spec_sha256=expected["model_spec_sha256"],
                expected_parameter_count=expected["parameter_count"],
                expected_corpus_sha256=expected["corpus_sha256"],
                expected_tokenizer_config_sha256=expected["tokenizer_config_sha256"],
                expected_tokenizer_vocab_sha256=expected["tokenizer_vocab_sha256"],
                expected_checkpoint_id=checkpoint_id,
                expected_tokens_seen=tokens,
            )

        aggregate_bpb = summary.get("aggregate_bpb")
        if not isinstance(aggregate_bpb, (int, float)) or not math.isfinite(aggregate_bpb) or aggregate_bpb < 0:
            raise VerificationError("LEARN-217 aggregate BPB invalid")

    return {
        "verify_id": contract["verify_id"],
        "result": "PASS_SCIENTIFIC_EVIDENCE",
        "source_sha": expected["head_sha"],
        "artifact_sha256": expected["artifact_sha256"],
        "final_actual_optimized_tokens": expected["final_actual_tokens"],
        "release_ci": contract["release_ci"],
    }


def validate_contract(contract: dict[str, Any]) -> None:
    _require_equal(contract.get("schema"), "12-6.learned-ladder-independent-verify.v1", "contract schema")
    verifications = contract.get("verifications")
    if not isinstance(verifications, list) or len(verifications) != 2:
        raise VerificationError("contract must contain exactly VERIFY-219 and VERIFY-218")
    ids = [item.get("verify_id") for item in verifications]
    _require_equal(ids, ["VERIFY-219", "VERIFY-218-LEARNED-10M-INDEPENDENT"], "verification IDs")
    comparison = contract.get("comparison_boundary")
    if not isinstance(comparison, dict):
        raise VerificationError("comparison boundary missing")
    _require_false(comparison.get("matched_optimized_budget"), "matched optimized budget")
    _require_false(comparison.get("direct_scale_ranking_authorized"), "direct scale ranking authorization")
    _require_equal(comparison.get("three_m_actual_optimized_tokens"), 131_938, "3M comparison budget")
    _require_equal(comparison.get("ten_m_actual_optimized_tokens"), 2_000_060, "10M comparison budget")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--learn191-artifact", type=Path)
    parser.add_argument("--learn217-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(contract)
    by_id = {item["verify_id"]: item for item in contract["verifications"]}
    results: list[dict[str, Any]] = []
    if args.learn191_artifact is not None:
        results.append(verify_learn191(args.learn191_artifact, by_id["VERIFY-219"]))
    if args.learn217_artifact is not None:
        results.append(
            verify_learn217(
                args.learn217_artifact,
                by_id["VERIFY-218-LEARNED-10M-INDEPENDENT"],
            )
        )
    if not results:
        results = [{"result": "PASS_CONTRACT_ONLY", "verification_ids": list(by_id)}]

    output = {
        "schema": "12-6.learned-ladder-independent-verify-result.v1",
        "status": "PASS",
        "results": results,
        "comparison_boundary": contract["comparison_boundary"],
    }
    text = json.dumps(output, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
