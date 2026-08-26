"""Fresh-process verification for retained SCALE-141 learned 10M checkpoints.

This is verification/retention only. It does not define a second training runtime.
The authoritative campaign remains scale141_10m_runtime_v3.
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

from twelve_six import scale141_10m_continuation as core
from twelve_six import scale141_10m_runtime_v2 as v2
from twelve_six import scale141_10m_runtime_v3 as v3
from twelve_six.checkpoint import hash_json, verify_checkpoint
from twelve_six.inference.first_party import load_first_party_backend

TRAINED_TARGETS = (500_000, 1_000_000, 1_500_000, 2_000_000)
VERIFY_TOL = 1e-7


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


def _best_target(report: dict[str, Any]) -> int:
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


def verify(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    # Install the exact authoritative runtime contract, including JSON-stable
    # manifest construction and streamed evaluation, but perform no training.
    # Order matters: v3 replaces v2's builder, then v2 binds that builder into
    # the shared core runtime while also applying seq-256/actual-token semantics.
    v3._install()
    v2._install_runtime_contract()
    report = v3.validate(out / "report.json", source_sha)
    manifest, tok, spec, _init, _cfg, _locks, run = core._common(
        repo, source_sha, out, False
    )

    best_target = _best_target(report)
    final_target = v2.TARGET_OPTIMIZED_TOKENS
    if final_target != 2_000_000:
        raise Scale141VerificationError("unexpected final optimized-token target")

    initial_bpb = float(report["scheduled"]["0"]["heldout"]["bits_per_byte"])
    best_bpb = float(
        report["scheduled"][str(best_target)]["heldout"]["bits_per_byte"]
    )
    final_bpb = float(
        report["scheduled"][str(final_target)]["heldout"]["bits_per_byte"]
    )

    fresh: dict[str, Any] = {}
    for role, target in (("best", best_target), ("final", final_target)):
        checkpoint = out / f"checkpoint-token-{target:07d}"
        checked = verify_checkpoint(checkpoint)
        identity = checked["identity"]
        expected_point = report["scheduled"][str(target)]

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

        logits1 = _logits_snapshot(checkpoint)
        logits2 = _logits_snapshot(checkpoint)
        if logits1 != logits2:
            raise Scale141VerificationError("first-party logits are not reproducible")

        backend = load_first_party_backend(checkpoint)
        before_mode = bool(backend.model.training)
        fresh_eval = core._fixed_eval(
            backend.model,
            out / "corpus-a",
            manifest,
            tok,
            split="validation",
            windows=core.HELDOUT_WINDOWS_PER_MODALITY,
        )
        if not fresh_eval["non_mutation_passed"]:
            raise Scale141VerificationError("fresh evaluation mutated model state")
        if bool(backend.model.training) != before_mode:
            raise Scale141VerificationError("fresh evaluation changed model mode")
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
            "evaluation": fresh_eval,
            "generation": generation,
        }

    result = {
        "schema": "12-6.scale141-fresh-verification.v1",
        "source_sha": source_sha,
        "process_pid": os.getpid(),
        "run_manifest_identity_sha256": run["identity_sha256"],
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "tokenizer": report["tokenizer"],
        "model": report["model"],
        "best_target_optimized_tokens": best_target,
        "final_target_optimized_tokens": final_target,
        "initial_heldout_bits_per_byte": initial_bpb,
        "best_heldout_bits_per_byte": best_bpb,
        "final_heldout_bits_per_byte": final_bpb,
        "heldout_improved_vs_random_init": best_bpb < initial_bpb,
        "fresh_verification": {
            "status": "PASS",
            "checkpoint_load": True,
            "first_party_logits": True,
            "evaluation_non_mutation": True,
            "checkpoint_identity": True,
            "generation": True,
            "reproducibility_manifest_validation": True,
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
