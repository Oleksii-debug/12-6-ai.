"""Fresh-process verification for retained SCALE-141 learned 10M checkpoints.

This is verification/retention only. It does not define a second training runtime.
The authoritative campaign remains scale141_10m_runtime_v3. Retained ``best`` is
selected on the exact MILESTONE-150 common DATA-25 / s0-byte-v1 / seq-128 held-out
evaluation identity so a genuine learned 10M artifact can later join the ladder
without changing its training provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

from twelve_six import milestone100_first_learned as m100
from twelve_six import scale141_10m_continuation as core
from twelve_six import scale141_10m_runtime_v2 as v2
from twelve_six import scale141_10m_runtime_v3 as v3
from twelve_six.checkpoint import hash_json, verify_checkpoint
from twelve_six.inference.first_party import load_first_party_backend

TRAINED_TARGETS = (500_000, 1_000_000, 1_500_000, 2_000_000)
COMMON_EVAL_TARGETS = (0,) + TRAINED_TARGETS
VERIFY_TOL = 1e-7
LADDER_EVAL_SCHEMA = "12-6.learned-base-ladder-evaluation-identity.v1"
EXPECTED_LADDER_EVAL_ID = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"


class Scale141VerificationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Scale141VerificationError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _campaign_best_target(report: dict[str, Any]) -> int:
    scheduled = report["scheduled"]
    missing = [target for target in TRAINED_TARGETS if str(target) not in scheduled]
    if missing:
        raise Scale141VerificationError(f"missing trained scheduled targets: {missing}")
    return min(
        TRAINED_TARGETS,
        key=lambda target: (
            float(scheduled[str(target)]["heldout"]["bits_per_byte"]),
            target,
        ),
    )


def _ladder_evaluation_identity(tok, manifest: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema": LADDER_EVAL_SCHEMA,
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "split": "validation",
        "strata_order": ["uk", "en", "code"],
        "metric": "autoregressive_cross_entropy_nats_and_bits_per_raw_utf8_byte",
        "target_mask": "labels[:,1:] != -100",
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "vocab_size": tok.identity.vocab_size,
            "normalization": tok.identity.normalization,
            "encoding": tok.identity.encoding,
            "special_tokens": dict(tok.identity.special_tokens),
        },
        "packing": {
            "version": m100.PACKING_VERSION,
            "sequence_length": m100.SEQ,
            "cross_document": False,
        },
    }
    value["identity_sha256"] = hash_json(value)
    if value["identity_sha256"] != EXPECTED_LADDER_EVAL_ID:
        raise Scale141VerificationError(
            "M150 common evaluation identity drifted; refusing 10M ladder retention"
        )
    return value


def _logits_snapshot(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    outputs: dict[str, Any] = {}
    for name, prompt in core.PROMPTS.items():
        ids = backend.encode(prompt)
        logits = list(backend.next_token_logits(ids))
        if len(logits) != 256 or not all(math.isfinite(float(x)) for x in logits):
            raise Scale141VerificationError("first-party logits are invalid")
        packed = b"".join(struct.pack("<f", float(x)) for x in logits)
        ranked = sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))[:8]
        outputs[name] = {
            "prompt": prompt,
            "input_ids": ids,
            "logits_float32_sha256": hashlib.sha256(packed).hexdigest(),
            "argmax_token_id": ranked[0],
            "top8_token_ids": ranked,
        }
    return {"backend_diagnostics": backend.diagnostics(), "outputs": outputs}


def _compare_heldout(recorded: dict[str, Any], fresh: dict[str, Any]) -> None:
    for key in ("loss", "bits_per_byte"):
        if abs(float(recorded[key]) - float(fresh[key])) > VERIFY_TOL:
            raise Scale141VerificationError(f"fresh held-out mismatch for {key}")
    if int(recorded["predicted_byte_tokens"]) != int(fresh["predicted_byte_tokens"]):
        raise Scale141VerificationError("fresh held-out token count mismatch")
    for modality in ("uk", "en", "code"):
        rb = recorded["by_modality"][modality]
        fb = fresh["by_modality"][modality]
        for key in ("loss", "bits_per_byte"):
            if abs(float(rb[key]) - float(fb[key])) > VERIFY_TOL:
                raise Scale141VerificationError(
                    f"fresh {modality} held-out mismatch for {key}"
                )
        if int(rb["predicted_byte_tokens"]) != int(fb["predicted_byte_tokens"]):
            raise Scale141VerificationError(
                f"fresh {modality} held-out token count mismatch"
            )


def _assert_checkpoint_identity(
    *,
    checked: dict[str, Any],
    source_sha: str,
    spec,
    tok,
    manifest: dict[str, Any],
    run: dict[str, Any],
    expected_point: dict[str, Any],
) -> None:
    identity = checked["identity"]
    if identity["git_sha"] != source_sha:
        raise Scale141VerificationError("checkpoint source SHA mismatch")
    if identity["model_spec_hash"] != spec.identity_sha256():
        raise Scale141VerificationError("checkpoint ModelSpec identity mismatch")
    if int(identity["parameter_count"]) != spec.parameter_count():
        raise Scale141VerificationError("checkpoint parameter count mismatch")
    if identity["tokenizer_hash"] != tok.identity.config_sha256:
        raise Scale141VerificationError("checkpoint tokenizer identity mismatch")
    if identity["tokenizer_vocab_hash"] != tok.identity.vocab_sha256:
        raise Scale141VerificationError("checkpoint tokenizer vocab identity mismatch")
    if identity["dataset_manifest_hash"] != manifest["corpus_identity_sha256"]:
        raise Scale141VerificationError("checkpoint corpus identity mismatch")
    if identity["run_manifest_hash"] != run["identity_sha256"]:
        raise Scale141VerificationError("checkpoint run-manifest identity mismatch")
    if int(identity["step"]) != int(expected_point["optimizer_step"]):
        raise Scale141VerificationError("checkpoint optimizer-step identity mismatch")
    if int(identity["tokens_seen"]) != int(expected_point["optimized_tokens"]):
        raise Scale141VerificationError("checkpoint optimized-token identity mismatch")


def verify(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    # Order matters: v3 replaces v2's builder, then v2 binds that builder into
    # the shared core runtime while applying seq-256/actual-token semantics.
    # This performs no training.
    v3._install()
    v2._install_runtime_contract()
    report = v3.validate(out / "report.json", source_sha)
    manifest, tok, spec, _init, _cfg, _locks, run = core._common(
        repo, source_sha, out, False
    )
    corpus = out / "corpus-a"

    campaign_best_target = _campaign_best_target(report)
    final_target = v2.TARGET_OPTIMIZED_TOKENS
    if final_target != 2_000_000:
        raise Scale141VerificationError("unexpected final optimized-token target")

    common_eval_identity = _ladder_evaluation_identity(tok, manifest)
    common_evaluations: dict[str, Any] = {}
    checked_by_target: dict[int, dict[str, Any]] = {}

    # All scheduled checkpoints still exist inside this job. Evaluate them on the
    # exact M150 common held-out identity before retaining only best/final.
    for target in COMMON_EVAL_TARGETS:
        checkpoint = out / f"checkpoint-token-{target:07d}"
        checked = verify_checkpoint(checkpoint)
        expected_point = report["scheduled"][str(target)]
        _assert_checkpoint_identity(
            checked=checked,
            source_sha=source_sha,
            spec=spec,
            tok=tok,
            manifest=manifest,
            run=run,
            expected_point=expected_point,
        )
        backend = load_first_party_backend(checkpoint)
        before_mode = bool(backend.model.training)
        evaluation = m100._evaluate(backend.model, corpus, manifest, tok)
        if not evaluation["non_mutation_passed"]:
            raise Scale141VerificationError("M150 common evaluation mutated model state")
        if bool(backend.model.training) != before_mode:
            raise Scale141VerificationError("M150 common evaluation changed model mode")
        checked_by_target[target] = checked
        common_evaluations[str(target)] = {
            "checkpoint_id": checked["checkpoint_id"],
            "evaluation": evaluation,
        }

    best_target = min(
        TRAINED_TARGETS,
        key=lambda target: (
            float(common_evaluations[str(target)]["evaluation"]["bits_per_byte"]),
            target,
        ),
    )

    common_initial_bpb = float(common_evaluations["0"]["evaluation"]["bits_per_byte"])
    common_best_bpb = float(
        common_evaluations[str(best_target)]["evaluation"]["bits_per_byte"]
    )
    common_final_bpb = float(
        common_evaluations[str(final_target)]["evaluation"]["bits_per_byte"]
    )
    campaign_initial_bpb = float(report["scheduled"]["0"]["heldout"]["bits_per_byte"])
    campaign_best_bpb = float(
        report["scheduled"][str(campaign_best_target)]["heldout"]["bits_per_byte"]
    )
    campaign_final_bpb = float(
        report["scheduled"][str(final_target)]["heldout"]["bits_per_byte"]
    )

    fresh: dict[str, Any] = {}
    for role, target in (("best", best_target), ("final", final_target)):
        checkpoint = out / f"checkpoint-token-{target:07d}"
        checked = checked_by_target[target]
        identity = checked["identity"]
        expected_point = report["scheduled"][str(target)]

        logits1 = _logits_snapshot(checkpoint)
        logits2 = _logits_snapshot(checkpoint)
        if logits1 != logits2:
            raise Scale141VerificationError("first-party logits are not reproducible")

        backend = load_first_party_backend(checkpoint)
        before_mode = bool(backend.model.training)
        fresh_eval = core._fixed_eval(
            backend.model,
            corpus,
            manifest,
            tok,
            split="validation",
            windows=core.HELDOUT_WINDOWS_PER_MODALITY,
        )
        if not fresh_eval["non_mutation_passed"]:
            raise Scale141VerificationError("fresh campaign evaluation mutated model state")
        if bool(backend.model.training) != before_mode:
            raise Scale141VerificationError("fresh campaign evaluation changed model mode")
        _compare_heldout(expected_point["heldout"], fresh_eval)

        generation = core._generation(checkpoint)
        if generation != expected_point["raw_base_generation"]:
            raise Scale141VerificationError("fresh raw Base generation mismatch")

        fresh[role] = {
            "target_optimized_tokens": target,
            "checkpoint_directory": checkpoint.name,
            "checkpoint_id": checked["checkpoint_id"],
            "checkpoint_identity": identity,
            "first_party_logits": logits1,
            "campaign_evaluation": fresh_eval,
            "ladder_common_evaluation": common_evaluations[str(target)]["evaluation"],
            "generation": generation,
        }

    result = {
        "schema": "12-6.scale141-fresh-verification.v2",
        "source_sha": source_sha,
        "process_pid": os.getpid(),
        "run_manifest_identity_sha256": run["identity_sha256"],
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "tokenizer": report["tokenizer"],
        "model": report["model"],
        "campaign_evaluation": {
            "packing_sequence_length": v2.SEQ,
            "scheduled_windows_per_modality": core.HELDOUT_WINDOWS_PER_MODALITY,
            "initial_bits_per_byte": campaign_initial_bpb,
            "best_target_optimized_tokens": campaign_best_target,
            "best_bits_per_byte": campaign_best_bpb,
            "final_bits_per_byte": campaign_final_bpb,
        },
        "ladder_common_evaluation": {
            "identity": common_eval_identity,
            "all_scheduled": common_evaluations,
            "initial_bits_per_byte": common_initial_bpb,
            "best_target_optimized_tokens": best_target,
            "best_bits_per_byte": common_best_bpb,
            "final_bits_per_byte": common_final_bpb,
            "best_improved_vs_random_init": common_best_bpb < common_initial_bpb,
        },
        "best_target_optimized_tokens": best_target,
        "final_target_optimized_tokens": final_target,
        "initial_heldout_bits_per_byte": common_initial_bpb,
        "best_heldout_bits_per_byte": common_best_bpb,
        "final_heldout_bits_per_byte": common_final_bpb,
        "heldout_improved_vs_random_init": common_best_bpb < common_initial_bpb,
        "fresh_verification": {
            "status": "PASS",
            "checkpoint_load": True,
            "first_party_logits": True,
            "evaluation_non_mutation": True,
            "checkpoint_identity": True,
            "generation": True,
            "reproducibility_manifest_validation": True,
            "m150_common_evaluation_identity": True,
            "best_and_final_retained": True,
        },
        "evidence": fresh,
    }
    result["identity_sha256"] = hash_json(result)
    _write_json(out / "fresh-verification.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()
    )
    print(
        json.dumps(
            {
                "status": result["fresh_verification"]["status"],
                "best_target_optimized_tokens": result["best_target_optimized_tokens"],
                "best_heldout_bits_per_byte": result["best_heldout_bits_per_byte"],
                "final_heldout_bits_per_byte": result["final_heldout_bits_per_byte"],
                "heldout_improved_vs_random_init": result[
                    "heldout_improved_vs_random_init"
                ],
                "campaign_best_target_optimized_tokens": result[
                    "campaign_evaluation"
                ]["best_target_optimized_tokens"],
                "ladder_evaluation_identity": result["ladder_common_evaluation"][
                    "identity"
                ]["identity_sha256"],
                "identity_sha256": result["identity_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
